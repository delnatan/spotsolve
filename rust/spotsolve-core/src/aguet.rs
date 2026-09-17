//! Sparse Aguet screening and spotfitlm's sampled, free-width Poisson fit.
//! See docs/AGUET_BASELINE.md for reference conventions and deliberate guards.

use crate::filters::{self, Mode};
use crate::linalg::Chol;

pub struct Settings {
    pub sigma: f64,
    pub boxsize: usize,
    pub itermax: usize,
    pub cutoff: f64,
    kernel: Vec<f64>,
    gsum: f64,
    denominator: f64,
}

impl Settings {
    pub fn new(sigma: f64, boxsize: usize, itermax: usize, cutoff: f64) -> Self {
        let radius = (4.0 * sigma).ceil() as isize;
        let kernel: Vec<f64> = (-radius..=radius)
            .map(|x| (-0.5 * (x as f64 / sigma).powi(2)).exp())
            .collect();
        let sum: f64 = kernel.iter().sum();
        let sum2: f64 = kernel.iter().map(|g| g * g).sum();
        let gsum = sum * sum;
        let n = (kernel.len() * kernel.len()) as f64;
        Self {
            sigma,
            boxsize,
            itermax,
            cutoff,
            kernel,
            gsum,
            denominator: sum2 * sum2 - gsum * gsum / n,
        }
    }
}

/// Parameter order: x, y, sigma, peak amplitude, background.
#[derive(Debug)]
pub struct Fit {
    pub theta: [f64; 5],
    pub covariance: [f64; 25],
    pub objective: f64,
    pub iterations: usize,
    /// 0 converged; -1 iteration limit; -2 covariance failure; -3 invalid fit.
    pub status: i32,
}

pub struct FitWorkspace {
    model: Vec<f64>,
    jacobian: Vec<[f64; 5]>,
    chol: Chol,
}

impl Default for FitWorkspace {
    fn default() -> Self {
        Self {
            model: Vec::new(),
            jacobian: Vec::new(),
            chol: Chol::new(5),
        }
    }
}

// The reference objective floors nonpositive observations. Use the same data
// in derivatives too, so fitting offset-subtracted pixels stays consistent.
const DATA_FLOOR: f64 = 1e-7;

fn pixel(theta: &[f64; 5], x: f64, y: f64) -> (f64, [f64; 5]) {
    let [xc, yc, sigma, amplitude, background] = *theta;
    let (dx, dy, s2) = (x - xc, y - yc, sigma * sigma);
    let r2 = dx * dx + dy * dy;
    let e = (-(dx * dx / (2.0 * s2) + dy * dy / (2.0 * s2))).exp();
    let ae = amplitude * e;
    (
        ae + background,
        [ae * dx / s2, ae * dy / s2, ae * r2 / (s2 * sigma), e, 1.0],
    )
}

fn objective(theta: &[f64; 5], data: &[f64], side: usize) -> f64 {
    if theta.iter().any(|v| !v.is_finite()) || theta[2] <= 0.0 {
        return f64::INFINITY;
    }
    let half = (side / 2) as f64;
    let mut score = 0.0;
    for (i, &d) in data.iter().enumerate() {
        let (m, _) = pixel(theta, (i % side) as f64 - half, (i / side) as f64 - half);
        if !m.is_finite() || m <= 0.0 {
            return f64::INFINITY;
        }
        let d = d.max(DATA_FLOOR);
        score += m - d - d * (m / d).ln();
    }
    score
}

/// Gradient and step curvature; optionally include model second derivatives
/// for the observed Hessian used in covariance estimation.
fn derivatives(
    ws: &mut FitWorkspace,
    theta: &[f64; 5],
    data: &[f64],
    side: usize,
    observed: bool,
) -> ([f64; 5], [f64; 25]) {
    let half = (side / 2) as f64;
    ws.model.resize(data.len(), 0.0);
    ws.jacobian.resize(data.len(), [0.0; 5]);
    for i in 0..data.len() {
        (ws.model[i], ws.jacobian[i]) =
            pixel(theta, (i % side) as f64 - half, (i / side) as f64 - half);
    }
    let mut g = [0.0; 5];
    let mut h = [0.0; 25];
    for a in 0..5 {
        for i in 0..data.len() {
            g[a] += ws.jacobian[i][a] * (1.0 - data[i].max(DATA_FLOOR) / ws.model[i]);
        }
        for b in a..5 {
            for i in 0..data.len() {
                let weight = data[i].max(DATA_FLOOR) / (ws.model[i] * ws.model[i]);
                h[a * 5 + b] += ws.jacobian[i][a] * weight * ws.jacobian[i][b];
            }
        }
    }
    if observed {
        let s = theta[2];
        let s2 = s * s;
        for i in 0..data.len() {
            let dx = (i % side) as f64 - half - theta[0];
            let dy = (i / side) as f64 - half - theta[1];
            let r2 = dx * dx + dy * dy;
            let q = [dx / s2, dy / s2, r2 / (s2 * s)];
            let log_second = [
                [-1.0 / s2, 0.0, -2.0 * dx / (s2 * s)],
                [0.0, -1.0 / s2, -2.0 * dy / (s2 * s)],
                [
                    -2.0 * dx / (s2 * s),
                    -2.0 * dy / (s2 * s),
                    -3.0 * r2 / (s2 * s2),
                ],
            ];
            let e = ws.jacobian[i][3];
            let residual = 1.0 - data[i].max(DATA_FLOOR) / ws.model[i];
            for a in 0..3 {
                for b in a..3 {
                    h[a * 5 + b] += residual * theta[3] * e * (q[a] * q[b] + log_second[a][b]);
                }
                h[a * 5 + 3] += residual * e * q[a];
            }
        }
    }
    for a in 0..5 {
        for b in 0..a {
            h[a * 5 + b] = h[b * 5 + a];
        }
    }
    (g, h)
}

/// One independent fit, initialized as in spotfitlm. No allocation per step.
/// Invalid trial means/widths are rejected rather than entering undefined math.
pub fn fit_patch(
    ws: &mut FitWorkspace,
    data: &[f64],
    side: usize,
    sigma: f64,
    itermax: usize,
) -> Fit {
    assert_eq!(data.len(), side * side);
    let low = data
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .max(DATA_FLOOR);
    let high = data
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(low);
    let mut result = Fit {
        theta: [0.0, 0.0, sigma, high - low, low],
        covariance: [f64::NAN; 25],
        objective: f64::INFINITY,
        iterations: 0,
        status: -1,
    };
    let mut mu = 1e-4_f64;
    for iteration in 0..itermax {
        result.iterations = iteration;
        let score = objective(&result.theta, data, side);
        if !score.is_finite() {
            result.status = -3;
            break;
        }
        let (g, h) = derivatives(ws, &result.theta, data, side, false);
        let scaled = (0..5).map(|i| g[i] * g[i] / h[i * 5 + i]).sum::<f64>();
        if scaled.is_finite() && scaled < 1e-5 {
            result.status = 0;
            break;
        }
        for _ in 0..30 {
            let mut system = h;
            for i in 0..5 {
                system[i * 5 + i] += mu;
            }
            if !ws.chol.factor(&system, 5) {
                mu = (mu * 2.0).clamp(1e-8, 1e8);
                continue;
            }
            let mut delta = g;
            ws.chol.solve_in_place(&mut delta);
            let mut trial = result.theta;
            for i in 0..5 {
                trial[i] -= delta[i];
            }
            let actual = score - objective(&trial, data, side);
            let mut predicted = (0..5).map(|i| delta[i] * g[i]).sum::<f64>();
            for i in 0..5 {
                for j in 0..5 {
                    predicted -= 0.5 * delta[i] * h[i * 5 + j] * delta[j];
                }
            }
            if actual >= 0.0 && predicted > 0.0 && actual / predicted >= 0.25 {
                result.theta = trial;
                mu = (mu / 3.0).clamp(1e-8, 1e8);
                break;
            }
            mu = (mu * 2.0).clamp(1e-8, 1e8);
        }
        result.iterations = iteration + 1;
    }
    result.objective = objective(&result.theta, data, side);
    let (_, hessian) = derivatives(ws, &result.theta, data, side, true);
    if ws.chol.factor(&hessian, 5) {
        for j in 0..5 {
            let mut column = [0.0; 5];
            column[j] = 1.0;
            ws.chol.solve_in_place(&mut column);
            for i in 0..5 {
                result.covariance[i * 5 + j] = column[i];
            }
        }
    } else {
        result.status = -2;
    }
    result
}

pub struct Spot {
    pub fit: Fit,
    pub seed: [usize; 2],
}
pub struct Output {
    pub spots: Vec<Spot>,
    pub background: Vec<f64>,
    pub candidates: usize,
    pub failures: Vec<[f64; 3]>, // seed y, x, status (-4 means post-fit bounds)
    pub processed_pixels: usize,
}

#[derive(Default)]
pub struct Workspace {
    fit: FitWorkspace,
    patch: Vec<f64>,
    data: Vec<f64>,
}

/// Local screening needs context, but the ROI and border cuts apply globally.
fn crop_bounds(roi: Option<&[bool]>, h: usize, w: usize, margin: usize) -> Option<[usize; 4]> {
    let Some(mask) = roi else {
        return Some([0, h, 0, w]);
    };
    let (mut y0, mut y1, mut x0, mut x1) = (h, 0, w, 0);
    for (i, &keep) in mask.iter().enumerate() {
        if keep {
            y0 = y0.min(i / w);
            y1 = y1.max(i / w + 1);
            x0 = x0.min(i % w);
            x1 = x1.max(i % w + 1);
        }
    }
    (y0 < y1).then(|| {
        [
            y0.saturating_sub(margin),
            y1.saturating_add(margin).min(h),
            x0.saturating_sub(margin),
            x1.saturating_add(margin).min(w),
        ]
    })
}

pub fn localize(
    raw: &[f64],
    h: usize,
    w: usize,
    offset: f64,
    roi: Option<&[bool]>,
    s: &Settings,
    ws: &mut Workspace,
) -> Output {
    let mut out = Output {
        spots: Vec::new(),
        background: vec![f64::NAN; h * w],
        candidates: 0,
        failures: Vec::new(),
        processed_pixels: 0,
    };
    let dom = 2 * s.sigma.ceil() as usize + 1;
    let half = s.boxsize / 2;
    let margin = (s.kernel.len() / 2)
        .max(filters::kernel_radius(s.sigma) + dom / 2)
        .max(half);
    let Some([y0, y1, x0, x1]) = crop_bounds(roi, h, w, margin) else {
        return out;
    };
    let (ch, cw) = (y1 - y0, x1 - x0);
    ws.data.clear();
    for y in y0..y1 {
        ws.data
            .extend(raw[y * w + x0..y * w + x1].iter().map(|v| v - offset));
    }
    let data = &ws.data;
    let pixels = ch * cw;
    out.processed_pixels = pixels;
    let mut tmp = vec![0.0; pixels];
    let mut fg = vec![0.0; pixels];
    filters::convolve1d(data, &mut tmp, ch, cw, &s.kernel, 0, Mode::Reflect);
    filters::convolve1d(&tmp, &mut fg, ch, cw, &s.kernel, 1, Mode::Reflect);
    let fu = filters::uniform_filter(data, ch, cw, s.kernel.len(), Mode::Reflect);
    let squares: Vec<f64> = data.iter().map(|v| v * v).collect();
    let fu2 = filters::uniform_filter(&squares, ch, cw, s.kernel.len(), Mode::Reflect);
    let log: Vec<f64> = filters::gaussian_laplace(data, ch, cw, s.sigma, Mode::Reflect)
        .into_iter()
        .map(|v| -v)
        .collect();
    let maxima = filters::maximum_filter(&log, ch, cw, dom, Mode::Reflect);
    let n = (s.kernel.len() * s.kernel.len()) as f64;
    for i in 0..pixels {
        let (y, x) = (y0 + i / cw, x0 + i % cw);
        let amplitude = (fg[i] - s.gsum * fu[i]) / s.denominator;
        let background = fu[i] - s.gsum * amplitude / n;
        out.background[y * w + x] = background;
        if log[i] != maxima[i]
            || y < dom
            || x < dom
            || y >= h.saturating_sub(dom)
            || x >= w.saturating_sub(dom)
            || y < half
            || x < half
            || y + half >= h
            || x + half >= w
            || roi.is_some_and(|m| !m[y * w + x])
        {
            continue;
        }
        let rss = (amplitude * amplitude * (s.denominator + s.gsum * s.gsum / n)
            - 2.0 * amplitude * (fg[i] - background * s.gsum)
            + n * fu2[i]
            - 2.0 * background * n * fu[i]
            + n * background * background)
            .max(0.0);
        if !rss.is_finite() || rss <= 0.0 || !(amplitude > s.cutoff * (rss / (n - 1.0)).sqrt()) {
            continue;
        }
        out.candidates += 1;
        ws.patch.clear();
        for yy in y - half..=y + half {
            ws.patch.extend(
                raw[yy * w + x - half..yy * w + x + half + 1]
                    .iter()
                    .map(|v| v - offset),
            );
        }
        let mut fit = fit_patch(&mut ws.fit, &ws.patch, s.boxsize, s.sigma, s.itermax);
        fit.theta[0] += x as f64;
        fit.theta[1] += y as f64;
        if fit.status == 0 && (fit.theta[3] <= 0.0 || fit.theta[4] <= 0.0) {
            fit.status = -3;
        }
        if fit.status == 0
            && !(fit.theta[0] > half as f64
                && fit.theta[1] > half as f64
                && fit.theta[0] < (w - half) as f64
                && fit.theta[1] < (h - half) as f64)
        {
            fit.status = -4;
        }
        if fit.status == 0 {
            out.spots.push(Spot { fit, seed: [y, x] });
        } else {
            out.failures.push([y as f64, x as f64, fit.status as f64]);
        }
    }
    out
}

pub fn localize_stack(
    raw: &[f64],
    n: usize,
    h: usize,
    w: usize,
    offset: f64,
    roi: Option<&[bool]>,
    s: &Settings,
    threads: usize,
) -> Vec<Output> {
    crate::frames::map(n, threads, Workspace::default, |t, ws| {
        localize(&raw[t * h * w..(t + 1) * h * w], h, w, offset, roi, s, ws)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_hessian_matches_gradient_differences_with_floored_data() {
        let mut ws = FitWorkspace::default();
        let theta = [0.23, -0.41, 1.4, 65.0, 12.0];
        let data: Vec<f64> = (0..81).map(|i| (i % 23) as f64 - 2.0).collect();
        let (gradient, hessian) = derivatives(&mut ws, &theta, &data, 9, true);
        for j in 0..5 {
            let step = 1e-5 * theta[j].abs().max(1.0);
            let mut plus = theta;
            let mut minus = theta;
            plus[j] += step;
            minus[j] -= step;
            let numerical_gradient =
                (objective(&plus, &data, 9) - objective(&minus, &data, 9)) / (2.0 * step);
            assert!((gradient[j] - numerical_gradient).abs() < 1e-6);
            let (gp, _) = derivatives(&mut ws, &plus, &data, 9, false);
            let (gm, _) = derivatives(&mut ws, &minus, &data, 9, false);
            for i in 0..5 {
                assert!((hessian[i * 5 + j] - (gp[i] - gm[i]) / (2.0 * step)).abs() < 1e-6);
            }
        }
    }
}
