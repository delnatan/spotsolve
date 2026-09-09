//! Calibrated local inference kernels. No Python callbacks or runtime dependencies.
//!
//! Calibration prefiltering stays outside the hot loop. Each component prepares
//! cubic weights once per row/column and depth, then reuses them across pixels.
//! The caller owns one reusable workspace per concurrent component fit.

use crate::affine::{self, AffineData};

const BACKGROUND: usize = 16;
const NUISANCE: usize = 20;
const FLUX: f64 = 1000.0;

#[derive(Clone, Copy)]
struct AxisPoint {
    base: usize,
    t: f64,
    w: [f64; 4],
    d: [f64; 4],
}

#[derive(Clone, Copy)]
struct Axis {
    start: f64,
    end: f64,
    step: f64,
    len: usize,
}

impl Axis {
    fn new(values: &[f64]) -> Result<Self, String> {
        if values.len() < 4 || values.iter().any(|v| !v.is_finite()) {
            return Err("calibration axes require at least four finite samples".into());
        }
        let step = values[1] - values[0];
        if !step.is_finite()
            || step <= 0.0
            || values.windows(2).any(|v| {
                let delta = v[1] - v[0];
                delta <= 0.0 || (delta - step).abs() > 1e-12 + 1e-8 * step.abs()
            })
        {
            return Err("calibration axes must be uniformly increasing".into());
        }
        Ok(Self {
            start: values[0],
            end: values[values.len() - 1],
            step,
            len: values.len(),
        })
    }

    fn point(self, value: f64) -> Result<AxisPoint, String> {
        if !value.is_finite() || value < self.start - 1e-10 || value > self.end + 1e-10 {
            return Err("requested PSF coordinate lies outside calibration".into());
        }
        let u = (value.clamp(self.start, self.end) - self.start) / self.step + 2.0;
        if !u.is_finite() || u.floor() < 1.0 || u.floor() > (self.len + 1) as f64 {
            return Err("calibration endpoint is inconsistent with its uniform grid".into());
        }
        let base = u.floor() as usize;
        let t = u - base as f64;
        let w = [
            (1.0 - t).powi(3) / 6.0,
            (3.0 * t.powi(3) - 6.0 * t * t + 4.0) / 6.0,
            (-3.0 * t.powi(3) + 3.0 * t * t + 3.0 * t + 1.0) / 6.0,
            t.powi(3) / 6.0,
        ];
        let d = [
            -0.5 * (1.0 - t).powi(2) / self.step,
            (1.5 * t * t - 2.0 * t) / self.step,
            (-1.5 * t * t + t + 0.5) / self.step,
            0.5 * t * t / self.step,
        ];
        Ok(AxisPoint {
            base: base - 1,
            t,
            w,
            d,
        })
    }
}

/// Owned padded log-spline coefficients, prepared once by calibration code.
/// Expected shape is (depth_len+4, offset_len+4, offset_len+4).
pub struct Calibration {
    depth: Axis,
    offset: Axis,
    coefficients: Vec<f64>,
}

impl Calibration {
    pub fn new(depth: &[f64], offset: &[f64], coefficients: Vec<f64>) -> Result<Self, String> {
        let depth = Axis::new(depth)?;
        let offset = Axis::new(offset)?;
        let size = (depth.len + 4)
            .checked_mul(offset.len + 4)
            .and_then(|v| v.checked_mul(offset.len + 4))
            .ok_or("calibration shape overflow")?;
        if coefficients.len() != size || coefficients.iter().any(|v| !v.is_finite()) {
            return Err("invalid padded spline coefficient array".into());
        }
        Ok(Self {
            depth,
            offset,
            coefficients,
        })
    }

    // Return unit response and derivatives with respect to offset y/x/depth.
    fn sample<const SECOND: bool>(&self, z: AxisPoint, y: AxisPoint, x: AxisPoint) -> [f64; 10] {
        // Only the curvature instantiation computes second spline weights.
        let second = |a: AxisPoint, step: f64| -> [f64; 4] {
            if SECOND {
                let t = a.t;
                let s = step * step;
                [(1. - t) / s, (3. * t - 2.) / s, (1. - 3. * t) / s, t / s]
            } else {
                [0.; 4]
            }
        };
        let zz = second(z, self.depth.step);
        let yy = second(y, self.offset.step);
        let xx = second(x, self.offset.step);
        let mut curvature = [0.; 6]; // yy, yx, yz, xx, xz, zz of log response
        let side = self.offset.len + 4;
        let mut log_value = 0.0;
        let mut gradient = [0.0; 3]; // z, y, x
        for iz in 0..4 {
            for iy in 0..4 {
                let start = ((z.base + iz) * side + y.base + iy) * side + x.base;
                for ix in 0..4 {
                    let c = self.coefficients[start + ix];
                    log_value += c * z.w[iz] * y.w[iy] * x.w[ix];
                    gradient[0] += c * z.d[iz] * y.w[iy] * x.w[ix];
                    gradient[1] += c * y.d[iy] * x.w[ix] * z.w[iz];
                    gradient[2] += c * x.d[ix] * z.w[iz] * y.w[iy];
                    if SECOND {
                        curvature[0] += c * z.w[iz] * yy[iy] * x.w[ix];
                        curvature[1] += c * z.w[iz] * y.d[iy] * x.d[ix];
                        curvature[2] += c * z.d[iz] * y.d[iy] * x.w[ix];
                        curvature[3] += c * z.w[iz] * y.w[iy] * xx[ix];
                        curvature[4] += c * z.d[iz] * y.w[iy] * x.d[ix];
                        curvature[5] += c * zz[iz] * y.w[iy] * x.w[ix];
                    }
                }
            }
        }
        let response = log_value.exp();
        let mut result = [0.; 10];
        result[..4].copy_from_slice(&[
            response,
            response * gradient[1],
            response * gradient[2],
            response * gradient[0],
        ]);
        if SECOND {
            let [gz, gy, gx] = gradient;
            let products = [gy * gy, gy * gx, gy * gz, gx * gx, gx * gz, gz * gz];
            for j in 0..6 {
                result[4 + j] = response * (curvature[j] + products[j]);
            }
        }
        result
    }
}

pub struct Model {
    pub shape: [usize; 2],
    pub focus_bounds: [f64; 4],
    pub defocus_bounds: [f64; 2],
    calibration: Calibration,
    pub(crate) background: Vec<[f64; BACKGROUND]>,
}

/// Buffers are resized only when image/count dimensions change.
#[derive(Default)]
pub struct Workspace {
    pub mean: Vec<f64>,
    pub jacobian: Vec<f64>, // pixel-major, parameter-minor
    pub gradient: Vec<f64>, // affine coordinates only
    pub hessian: Vec<f64>,  // affine row-major
    pub affine: AffineData,
    /// Pixel-major nonzero source-block second derivatives: 9 broad + 5 per focus.
    pub second: Vec<f64>,
    rows: Vec<AxisPoint>,
    columns: Vec<AxisPoint>,
}

fn cubic(t: f64) -> [f64; 4] {
    let mut b = [
        (1.0 - t).powi(3),
        3.0 * t * (1.0 - t).powi(2),
        3.0 * t * t * (1.0 - t),
        t.powi(3),
    ];
    let sum: f64 = b.iter().sum();
    for v in &mut b {
        *v /= sum;
    }
    b
}

impl Model {
    pub fn new(
        shape: [usize; 2],
        focus_bounds: [f64; 4],
        defocus_bounds: [f64; 2],
        calibration: Calibration,
    ) -> Result<Self, String> {
        let [h, w] = shape;
        let n = h
            .checked_mul(w)
            .and_then(|v| v.checked_mul(26))
            .ok_or("image shape overflow")?
            / 26;
        if h < 3
            || w < 3
            || focus_bounds
                .iter()
                .chain(defocus_bounds.iter())
                .any(|v| !v.is_finite())
        {
            return Err("invalid image shape or bounds".into());
        }
        let [y0, x0, y1, x1] = focus_bounds;
        if !(y0 >= -0.5
            && x0 >= -0.5
            && y0 < y1
            && x0 < x1
            && y1 <= h as f64 - 0.5
            && x1 <= w as f64 - 0.5)
        {
            return Err("focus bounds lie outside image".into());
        }
        if !(0.0 < defocus_bounds[0]
            && defocus_bounds[0] < defocus_bounds[1]
            && calibration.depth.start <= 0.0
            && calibration.depth.end >= defocus_bounds[1])
        {
            return Err("calibration must cover zero and positive defocus bounds".into());
        }
        let extent = h.max(w) as f64 - 0.5;
        if calibration.offset.start > -extent || calibration.offset.end < extent {
            return Err("PSF calibration does not cover image offsets".into());
        }
        let mut background = Vec::with_capacity(n);
        for y in 0..h {
            let by = cubic(y as f64 / (h - 1) as f64);
            for x in 0..w {
                let bx = cubic(x as f64 / (w - 1) as f64);
                let mut b = [0.0; BACKGROUND];
                for iy in 0..4 {
                    for ix in 0..4 {
                        b[4 * iy + ix] = by[iy] * bx[ix];
                    }
                }
                background.push(b);
            }
        }
        Ok(Self {
            shape,
            focus_bounds,
            defocus_bounds,
            calibration,
            background,
        })
    }

    /// Unit-flux broad response over every integer pixel-to-center offset.
    /// Cache these three small tables once per component, shared by K=1 and K=2.
    pub(crate) fn broad_kernel(&self, depth: f64) -> Result<Vec<f64>, String> {
        let [h, w] = self.shape;
        let z = self.calibration.depth.point(depth)?;
        let rows = (0..2 * h - 1)
            .map(|y| self.calibration.offset.point(y as f64 - (h - 1) as f64))
            .collect::<Result<Vec<_>, _>>()?;
        let cols = (0..2 * w - 1)
            .map(|x| self.calibration.offset.point(x as f64 - (w - 1) as f64))
            .collect::<Result<Vec<_>, _>>()?;
        let mut out = Vec::with_capacity(rows.len() * cols.len());
        for &y in &rows {
            for &x in &cols {
                out.push(self.calibration.sample::<false>(z, y, x)[0]);
            }
        }
        Ok(out)
    }

    pub fn count(theta: &[f64]) -> Result<usize, String> {
        match theta.len() {
            20 => Ok(0),
            23 => Ok(1),
            26 => Ok(2),
            _ => Err("expected 20+3*K parameters, K=0,1,2".into()),
        }
    }

    /// Mathematical mean/Jacobian evaluation; coefficients need not be feasible.
    /// This permits optimizer trial evaluation. Photon statistics enforce a
    /// positive total mean; the constrained fitter must separately enforce
    /// background rates and source-amplitude bounds.
    pub fn evaluate(&self, theta: &[f64], workspace: &mut Workspace) -> Result<(), String> {
        self.evaluate_order::<false>(theta, workspace)
    }

    pub fn evaluate_second(&self, theta: &[f64], workspace: &mut Workspace) -> Result<(), String> {
        self.evaluate_order::<true>(theta, workspace)
    }

    fn evaluate_order<const SECOND: bool>(
        &self,
        theta: &[f64],
        workspace: &mut Workspace,
    ) -> Result<(), String> {
        let count = Self::count(theta)?;
        if theta.iter().any(|v| !v.is_finite()) {
            return Err("parameters must be finite".into());
        }
        let [h, w] = self.shape;
        let p = theta.len();
        workspace.mean.resize(h * w, 0.0);
        workspace.jacobian.resize(h * w * p, 0.0);
        let stride = 9 + 5 * count;
        if SECOND {
            workspace.second.resize(h * w * stride, 0.0);
        }
        for (pixel, background) in self.background.iter().enumerate() {
            let row = &mut workspace.jacobian[pixel * p..(pixel + 1) * p];
            let mut mean = 1e-4;
            for j in 0..BACKGROUND {
                row[j] = 10.0 * background[j];
                mean += row[j] * theta[j];
            }
            workspace.mean[pixel] = mean;
        }
        for component in 0..=count {
            let (base, depth) = if component == 0 {
                (16, theta[19])
            } else {
                (NUISANCE + 3 * (component - 1), 0.0)
            };
            let z = self.calibration.depth.point(depth)?;
            workspace.rows.clear();
            workspace.columns.clear();
            for y in 0..h {
                workspace
                    .rows
                    .push(self.calibration.offset.point(y as f64 - theta[base + 1])?);
            }
            for x in 0..w {
                workspace
                    .columns
                    .push(self.calibration.offset.point(x as f64 - theta[base + 2])?);
            }
            let flux = FLUX * theta[base];
            for y in 0..h {
                for x in 0..w {
                    let unit = self.calibration.sample::<SECOND>(
                        z,
                        workspace.rows[y],
                        workspace.columns[x],
                    );
                    let pixel = y * w + x;
                    workspace.mean[pixel] += flux * unit[0];
                    let row = &mut workspace.jacobian[pixel * p..(pixel + 1) * p];
                    row[base] = FLUX * unit[0];
                    row[base + 1] = -flux * unit[1];
                    row[base + 2] = -flux * unit[2];
                    if component == 0 {
                        row[base + 3] = flux * unit[3];
                    }
                    if SECOND {
                        let start = pixel * stride
                            + if component == 0 {
                                0
                            } else {
                                9 + 5 * (component - 1)
                            };
                        let block = &mut workspace.second
                            [start..start + if component == 0 { 9 } else { 5 }];
                        // Lower triangle of each source block, excluding flux/flux.
                        // Position derivatives reverse offset signs; depth does not.
                        block[..5].copy_from_slice(&[
                            -FLUX * unit[1],
                            flux * unit[4],
                            -FLUX * unit[2],
                            flux * unit[5],
                            flux * unit[7],
                        ]);
                        if component == 0 {
                            block[5..].copy_from_slice(&[
                                FLUX * unit[3],
                                -flux * unit[6],
                                -flux * unit[8],
                                flux * unit[9],
                            ]);
                        }
                    }
                }
            }
        }
        if workspace
            .mean
            .iter()
            .chain(workspace.jacobian.iter())
            .any(|v| !v.is_finite())
        {
            return Err("nonfinite calibrated response".into());
        }
        if SECOND && workspace.second.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite calibrated second derivative".into());
        }
        Ok(())
    }

    /// Prepare geometry once. Subsequent affine_at calls do not touch PSFs.
    /// Returns initial coefficients in background, broad flux, focus flux order.
    pub fn prepare_affine(
        &self,
        theta: &[f64],
        workspace: &mut Workspace,
    ) -> Result<Vec<f64>, String> {
        workspace.affine.columns = 0;
        let count = Self::count(theta)?;
        let mut initial = vec![0.0; 17 + count];
        self.prepare_affine_into(theta, workspace, &mut initial)?;
        Ok(initial)
    }

    /// Caller-supplied coefficient storage avoids allocation in geometry loops.
    pub fn prepare_affine_into(
        &self,
        theta: &[f64],
        workspace: &mut Workspace,
        initial: &mut [f64],
    ) -> Result<(), String> {
        workspace.affine.columns = 0;
        let count = Self::count(theta)?;
        if theta.iter().any(|v| !v.is_finite()) {
            return Err("parameters must be finite".into());
        }
        let a = 17 + count;
        let p = theta.len();
        let mut ids = [0usize; 19];
        for (j, id) in ids[..17].iter_mut().enumerate() {
            *id = j;
        }
        for k in 0..count {
            ids[17 + k] = NUISANCE + 3 * k;
        }
        let mut unit = [0.0; 26];
        unit[..p].copy_from_slice(theta);
        if initial.len() != a {
            return Err("incorrect affine output size".into());
        }
        for j in 0..a {
            initial[j] = theta[ids[j]];
        }
        for j in &ids[..a] {
            unit[*j] = 1.0;
        }
        self.evaluate(&unit[..p], workspace)?;
        let n = self.shape[0] * self.shape[1];
        workspace.affine.basis.resize(n * a, 0.0);
        workspace.affine.offset.resize(n, 0.0);
        for i in 0..n {
            for j in 0..a {
                let value = workspace.jacobian[i * p + ids[j]];
                workspace.affine.basis[i * a + j] = value;
            }
            workspace.affine.offset[i] = 1e-4;
        }
        workspace.affine.columns = a;
        Ok(())
    }

    /// One-shot convenience operation, not a constrained minimizer.
    pub fn affine_statistics(
        &self,
        data: &[f64],
        theta: &[f64],
        workspace: &mut Workspace,
    ) -> Result<f64, String> {
        let initial = self.prepare_affine(theta, workspace)?;
        self.affine_at(data, &initial, workspace)
    }

    /// Poisson deviance/2, affine gradient and observed Hessian at cached geometry.
    /// No interpolation, heap allocation after sizing, or Python work here.
    /// Feasibility of background rates/fluxes remains the fitter's responsibility.
    pub fn affine_at(
        &self,
        data: &[f64],
        coefficients: &[f64],
        workspace: &mut Workspace,
    ) -> Result<f64, String> {
        let a = workspace.affine.columns;
        if a == 0 || coefficients.len() != a || coefficients.iter().any(|v| !v.is_finite()) {
            return Err("prepare geometry and supply compatible finite affine coefficients".into());
        }
        if data.len() != self.shape[0] * self.shape[1]
            || data.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("data must be finite nonnegative pixels matching the model".into());
        }
        let value = affine::value(&workspace.affine, data, coefficients, &mut workspace.mean);
        if !value.is_finite() {
            return Err("Poisson mean must be finite and positive".into());
        }
        workspace.gradient.resize(a, 0.0);
        workspace.hessian.resize(a * a, 0.0);
        affine::derivatives(
            &workspace.affine,
            data,
            &workspace.mean,
            &mut workspace.gradient,
            &mut workspace.hessian,
        );
        if !value.is_finite()
            || workspace
                .gradient
                .iter()
                .chain(workspace.hessian.iter())
                .any(|v| !v.is_finite())
        {
            return Err("nonfinite Poisson statistics".into());
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model() -> Model {
        let depth = [-0.2, 0.0, 0.2, 0.4, 0.6];
        let offset: Vec<f64> = (-4..=4).map(|v| v as f64).collect();
        // Constant log response tests partition of unity and zero derivatives.
        let calibration = Calibration::new(&depth, &offset, vec![-2.0; 9 * 13 * 13]).unwrap();
        Model::new([3, 3], [0.0, 0.0, 2.0, 2.0], [0.2, 0.5], calibration).unwrap()
    }
    #[test]
    fn constant_table_and_reused_buffers() {
        let model = model();
        let mut work = Workspace::default();
        let mut theta = vec![0.5; 23];
        theta[17] = 1.0;
        theta[18] = 1.0;
        theta[19] = 0.3;
        model.evaluate(&theta, &mut work).unwrap();
        let expected = 5.0001 + 1000.0 * (-2.0f64).exp();
        for mu in &work.mean {
            assert!((mu - expected).abs() < 1e-10);
        }
        let pointer = work.jacobian.as_ptr();
        model.evaluate(&theta, &mut work).unwrap();
        assert_eq!(pointer, work.jacobian.as_ptr());
        let value = model
            .affine_statistics(&[0.0; 9], &theta, &mut work)
            .unwrap();
        assert!((value - 9.0 * expected).abs() < 1e-9);
        assert!(work.hessian.iter().all(|v| *v == 0.0));
    }
    #[test]
    fn rejects_invalid_calibration_and_out_of_range_coordinates() {
        assert!(Calibration::new(&[0.0, 0.1, 0.3, 0.4], &[0.0, 1.0, 2.0, 3.0], vec![]).is_err());
        let mut theta = vec![0.5; 20];
        theta[19] = 0.3;
        theta[17] = 100.0;
        assert!(model().evaluate(&theta, &mut Workspace::default()).is_err());
        assert!(
            model()
                .evaluate(&[0.0; 21], &mut Workspace::default())
                .is_err()
        );
    }
}

/// Assemble C*a >= rhs in retained coefficient units: lower boxes, upper
/// boxes, then sampled background rates. Storage is reused between solves.
pub fn prepare_constraints(
    model: &Model,
    lower: &[f64],
    upper: &[f64],
    matrix: &mut Vec<f64>,
    rhs: &mut Vec<f64>,
) -> Result<(), String> {
    let n = lower.len();
    if !(17..=19).contains(&n)
        || upper.len() != n
        || lower[16..].iter().any(|v| *v < 0.0)
        || lower
            .iter()
            .zip(upper)
            .any(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
    {
        return Err("compatible finite affine bounds required".into());
    }
    let m = 2 * n + model.background.len();
    matrix.resize(m * n, 0.0);
    matrix.fill(0.0);
    rhs.resize(m, 0.0);
    rhs.fill(0.0);
    for j in 0..n {
        matrix[j * n + j] = 1.0;
        rhs[j] = lower[j];
        matrix[(n + j) * n + j] = -1.0;
        rhs[n + j] = -upper[j];
    }
    for (i, row) in model.background.iter().enumerate() {
        matrix[(2 * n + i) * n..(2 * n + i) * n + 16].copy_from_slice(row);
    }
    Ok(())
}

#[cfg(test)]
mod curvature_tests {
    use super::*;
    #[test]
    fn log_spline_chain_rule_has_mixed_and_second_depth_terms() {
        let depth = [-0.2, 0.0, 0.2, 0.4, 0.6];
        let offsets: Vec<f64> = (-4..=4).map(|i| i as f64).collect();
        let polynomial = |z: f64, y: f64, x: f64| {
            0.03 * z - 0.02 * y + 0.04 * x + 0.005 * z * y - 0.003 * x * y
                + 0.002 * z * x
                + 0.002 * z * z
        };
        let mut coefficients = Vec::new();
        for z in 0..9 {
            for y in 0..13 {
                for x in 0..13 {
                    coefficients.push(polynomial(z as f64, y as f64, x as f64));
                }
            }
        }
        let calibration = Calibration::new(&depth, &offsets, coefficients).unwrap();
        let result = calibration.sample::<true>(
            calibration.depth.point(0.3).unwrap(),
            calibration.offset.point(0.2).unwrap(),
            calibration.offset.point(-0.1).unwrap(),
        );
        let (z, y, x) = (4.5, 6.2, 5.9);
        // A cardinal cubic spline reproduces linear/mixed terms exactly;
        // coefficient z^2 interpolates to z^2 + 1/3.
        let response = (polynomial(z, y, x) + 0.002 / 3.).exp();
        let gz = (0.03 + 0.005 * y + 0.002 * x + 0.004 * z) / 0.2;
        let gy = -0.02 + 0.005 * z - 0.003 * x;
        let gx = 0.04 - 0.003 * y + 0.002 * z;
        let expected = [
            response,
            response * gy,
            response * gx,
            response * gz,
            response * gy * gy,
            response * (gy * gx - 0.003),
            response * (gy * gz + 0.005 / 0.2),
            response * gx * gx,
            response * (gx * gz + 0.002 / 0.2),
            response * (gz * gz + 0.004 / 0.04),
        ];
        for (got, want) in result.iter().zip(expected) {
            assert!((got - want).abs() < 1e-11, "{got} != {want}");
        }
    }
}
