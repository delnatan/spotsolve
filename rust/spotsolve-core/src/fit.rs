//! One window's Poisson fit: a constant background and `k` pixel-integrated
//! Gaussians,
//!
//! ```text
//! m[r, c] = c0 + sum_k A_k E(r; y_k, s_k) E(c; x_k, s_k)
//! I(d, m) = sum_used [d log(d / m) - (d - m)]
//! ```
//!
//! in window-local pixels (`[0, 0]` is the window's first pixel). Pixels not
//! `used` are left out of the likelihood, as u-track sets them to NaN.
//! `theta = [c0, A_1, y_1, x_1, (s_1), ..., A_k, y_k, x_k, (s_k)]`: each
//! component's width is its own parameter, or one fixed width for all
//! ([`Layout`]). With free widths this is [`psf::pack_var`]'s layout.
//!
//! [`Fitter::fit`] minimizes `I` by bounded Levenberg-Marquardt (Fisher
//! scoring) with Coleman-Li affine scaling: parameters stay strictly inside
//! their bounds, and a coordinate's step shrinks with its distance to the
//! bound it heads for, so a flux cannot collapse onto its floor in one step
//! while the positions have yet to adapt. `I / phi` is the log-likelihood
//! in nats for a camera of dispersion `phi`; [`Fitter::variances`] are the
//! diagonal of the inverse expected Fisher information scaled by `phi`.

use crate::linalg::Chol;
use crate::psf;

/// Fraction of each bound's range kept as a margin inside it.
pub const INTERIOR_FRAC: f64 = 1e-10;
/// A step that would cross a bound goes this fraction of the way there.
pub const STEP_IN: f64 = 0.995;
/// Relative distance to a bound at which a parameter counts as on it.
pub const BOUND_TOL: f64 = 1e-6;
/// Side of the register blocks the information matrix is summed in.
const BLOCK: usize = 4;

/// The pixels of one fit, row-major `rows x cols`.
#[derive(Clone, Debug)]
pub struct Window {
    pub rows: usize,
    pub cols: usize,
    /// ADU above the camera offset.
    pub d: Vec<f64>,
    /// Pixels in the likelihood.
    pub used: Vec<bool>,
}

impl Window {
    pub fn new(rows: usize, cols: usize, d: Vec<f64>, used: Vec<bool>) -> Self {
        assert_eq!(d.len(), rows * cols);
        assert_eq!(used.len(), rows * cols);
        Self { rows, cols, d, used }
    }
}

/// How `theta` is laid out: `k` components, each with its own free width
/// (`sigma: None`) or all at one fixed width (`Some(s)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    pub k: usize,
    pub sigma: Option<f64>,
}

impl Layout {
    /// Parameters per component.
    pub fn stride(&self) -> usize {
        3 + self.sigma.is_none() as usize
    }

    pub fn n(&self) -> usize {
        1 + self.stride() * self.k
    }

    /// Index of component `j`'s flux; its `y`, `x` (and width) follow.
    pub fn at(&self, j: usize) -> usize {
        1 + self.stride() * j
    }

    /// Width of component `j`.
    pub fn sigma(&self, theta: &[f64], j: usize) -> f64 {
        self.sigma.unwrap_or_else(|| theta[self.at(j) + 3])
    }

    /// The layout of the same components with `k` changed.
    pub fn with(&self, k: usize) -> Self {
        Self { k, ..*self }
    }
}

/// A fit linearized at one `theta` ([`Fitter::linearize`]).
pub struct Linear {
    /// The model, row-major.
    pub m: Vec<f64>,
    /// `dm / dtheta`, `n x pixels`.
    pub jac: Vec<f64>,
    /// Gradient of `I` and its expected Fisher information, `n x n`.
    pub g: Vec<f64>,
    pub f: Vec<f64>,
}

/// A finished fit.
#[derive(Clone, Debug)]
pub struct Fit {
    pub theta: Vec<f64>,
    /// `I` at `theta`; divide by `phi` for nats.
    pub divergence: f64,
    pub iterations: usize,
    pub converged: bool,
    /// No step could lower the objective before convergence.
    pub stalled: bool,
    /// Per parameter: on its lower or upper bound.
    pub at_bound: Vec<bool>,
}

pub fn at_bound(value: f64, lo: f64, hi: f64) -> bool {
    let margin = 2.0 * INTERIOR_FRAC * (hi - lo).max(1e-12);
    value - lo <= margin + BOUND_TOL * (1.0 + lo.abs()) || hi - value <= margin + BOUND_TOL * (1.0 + hi.abs())
}

/// Reusable storage for fits of any window and layout.
pub struct Fitter {
    ay: Vec<f64>,
    ax: Vec<f64>,
    /// Per component along rows (`k * rows`) and columns (`k * cols`): the
    /// profile, its derivative in the centre, and in the width.
    ey: Vec<f64>,
    dey: Vec<f64>,
    sey: Vec<f64>,
    ex: Vec<f64>,
    dex: Vec<f64>,
    sex: Vec<f64>,
    m: Vec<f64>,
    /// The used pixels (row-major indices), and the Jacobian `dm / dtheta`
    /// on them parameter-major, `jt[a * used + q]`, with its rows weighted
    /// by `1 / m` in `jw`. Rows are padded to a multiple of [`BLOCK`].
    pix: Vec<usize>,
    /// On the used pixels: `1 / m` and the score residual `1 - d / m`.
    wt: Vec<f64>,
    res: Vec<f64>,
    jt: Vec<f64>,
    jw: Vec<f64>,
    f: Vec<f64>,
    g: Vec<f64>,
    chol: Chol,
    scratch: Vec<f64>,
}

impl Default for Fitter {
    fn default() -> Self {
        Self {
            ay: Vec::new(),
            ax: Vec::new(),
            ey: Vec::new(),
            dey: Vec::new(),
            sey: Vec::new(),
            ex: Vec::new(),
            dex: Vec::new(),
            sex: Vec::new(),
            m: Vec::new(),
            pix: Vec::new(),
            wt: Vec::new(),
            res: Vec::new(),
            jt: Vec::new(),
            jw: Vec::new(),
            f: Vec::new(),
            g: Vec::new(),
            chol: Chol::new(1),
            scratch: Vec::new(),
        }
    }
}

impl Fitter {
    /// The 1-D factors of every component at `theta`, and the model.
    fn evaluate(&mut self, w: &Window, theta: &[f64], lay: Layout) {
        let (rows, cols, k) = (w.rows, w.cols, lay.k);
        assert_eq!(theta.len(), lay.n());
        if self.ay.len() != rows {
            self.ay = psf::local_axis(rows);
        }
        if self.ax.len() != cols {
            self.ax = psf::local_axis(cols);
        }
        for v in [&mut self.ey, &mut self.dey, &mut self.sey] {
            v.resize(k * rows, 0.0);
        }
        for v in [&mut self.ex, &mut self.dex, &mut self.sex] {
            v.resize(k * cols, 0.0);
        }
        for i in 0..k {
            let (q, s) = (lay.at(i), lay.sigma(theta, i));
            let (ry, rx) = (i * rows..(i + 1) * rows, i * cols..(i + 1) * cols);
            psf::factors_axis_sigma(&self.ay, &theta[q + 1..q + 2], s, &mut self.ey[ry.clone()], &mut self.dey[ry.clone()], &mut self.sey[ry]);
            psf::factors_axis_sigma(&self.ax, &theta[q + 2..q + 3], s, &mut self.ex[rx.clone()], &mut self.dex[rx.clone()], &mut self.sex[rx]);
        }
        self.m.clear();
        self.m.resize(rows * cols, theta[0]);
        for i in 0..k {
            let a = theta[lay.at(i)];
            let ex = &self.ex[i * cols..(i + 1) * cols];
            for r in 0..rows {
                let ay = a * self.ey[i * rows + r];
                for (m, e) in self.m[r * cols..(r + 1) * cols].iter_mut().zip(ex) {
                    *m += ay * e;
                }
            }
        }
    }

    /// The model at `theta`, row-major.
    pub fn model(&mut self, w: &Window, theta: &[f64], lay: Layout) -> &[f64] {
        self.evaluate(w, theta, lay);
        &self.m
    }

    /// `I(d, m)` over the used pixels, with `0 log 0 = 0`; infinite where
    /// the model is not positive.
    pub fn divergence(&mut self, w: &Window, theta: &[f64], lay: Layout) -> f64 {
        self.evaluate(w, theta, lay);
        self.current_divergence(w)
    }

    fn current_divergence(&self, w: &Window) -> f64 {
        let mut total = 0.0;
        for ((&d, &m), &u) in w.d.iter().zip(&self.m).zip(&w.used) {
            if !u {
                continue;
            }
            if !(m > 0.0) {
                return f64::INFINITY;
            }
            total += if d > 0.0 { d * (d / m).ln() } else { 0.0 } - (d - m);
        }
        total
    }

    /// The Jacobian `dm / dtheta` at the last evaluated `theta` on the used
    /// pixels, parameter-major, into `self.pix` and `self.jt`; rows past
    /// `n` up to the next multiple of [`BLOCK`] are zero.
    fn fill_jacobian(&mut self, w: &Window, theta: &[f64], lay: Layout) {
        let (rows, cols, n) = (w.rows, w.cols, lay.n());
        self.pix.clear();
        self.pix.extend((0..rows * cols).filter(|&q| w.used[q]));
        let nu = self.pix.len();
        self.jt.resize(n.div_ceil(BLOCK) * BLOCK * nu, 0.0);
        self.jt[..nu].fill(1.0);
        self.jt[n * nu..].fill(0.0);
        for i in 0..lay.k {
            let p = lay.at(i);
            let a = theta[p];
            let (ey, dey, sey) = (&self.ey[i * rows..(i + 1) * rows], &self.dey[i * rows..(i + 1) * rows], &self.sey[i * rows..(i + 1) * rows]);
            let (ex, dex, sex) = (&self.ex[i * cols..(i + 1) * cols], &self.dex[i * cols..(i + 1) * cols], &self.sex[i * cols..(i + 1) * cols]);
            let (jf, tail) = self.jt[p * nu..].split_at_mut(nu);
            let (jy, tail) = tail.split_at_mut(nu);
            let (jx, js) = tail.split_at_mut(nu);
            let mut k = 0;
            for r in 0..rows {
                for c in 0..cols {
                    if !w.used[r * cols + c] {
                        continue;
                    }
                    jf[k] = ey[r] * ex[c];
                    jy[k] = a * dey[r] * ex[c];
                    jx[k] = a * ey[r] * dex[c];
                    if lay.sigma.is_none() {
                        js[k] = a * (sey[r] * ex[c] + ey[r] * sex[c]);
                    }
                    k += 1;
                }
            }
        }
    }

    /// Gradient of `I` and expected Fisher information `J^T diag(1/m) J` at
    /// the last evaluated `theta`, into `self.g` and `self.f`. Every entry
    /// is one sum over the used pixels in raster order; the information is
    /// summed in `BLOCK x BLOCK` tiles held in registers.
    #[allow(clippy::needless_range_loop)]
    fn normal(&mut self, w: &Window, theta: &[f64], lay: Layout) {
        let n = lay.n();
        self.fill_jacobian(w, theta, lay);
        let nu = self.pix.len();
        let np = n.div_ceil(BLOCK) * BLOCK;
        self.jw.resize(np * nu, 0.0);
        self.jw[n * nu..].fill(0.0);
        self.g.clear();
        self.g.resize(n, 0.0);
        self.wt.clear();
        self.res.clear();
        for &q in &self.pix {
            let m = self.m[q];
            self.wt.push(1.0 / m);
            self.res.push(1.0 - w.d[q].max(0.0) / m);
        }
        for a in 0..n {
            let j = &self.jt[a * nu..(a + 1) * nu];
            self.g[a] = j.iter().zip(&self.res).fold(0.0, |s, (j, r)| s + j * r);
            for ((o, j), wt) in self.jw[a * nu..(a + 1) * nu].iter_mut().zip(j).zip(&self.wt) {
                *o = j * wt;
            }
        }
        self.f.clear();
        self.f.resize(n * n, 0.0);
        for a0 in (0..n).step_by(BLOCK) {
            for b0 in (a0..n).step_by(BLOCK) {
                let x: [&[f64]; BLOCK] = std::array::from_fn(|i| &self.jw[(a0 + i) * nu..(a0 + i + 1) * nu]);
                let y: [&[f64]; BLOCK] = std::array::from_fn(|j| &self.jt[(b0 + j) * nu..(b0 + j + 1) * nu]);
                let acc = tile(x, y, nu);
                // Indexed: an iterator over `acc` moves the tile out of
                // registers.
                for i in 0..BLOCK.min(n - a0) {
                    for j in 0..BLOCK.min(n - b0) {
                        if b0 + j >= a0 + i {
                            self.f[(a0 + i) * n + b0 + j] = acc[i][j];
                        }
                    }
                }
            }
        }
        for a in 0..n {
            for b in 0..a {
                self.f[a * n + b] = self.f[b * n + a];
            }
        }
    }

    /// The fit linearized at `theta`: the model, the Jacobian `dm / dtheta`
    /// parameter-major over every pixel (zero on unused ones), and the
    /// gradient and expected Fisher information of `I`.
    pub fn linearize(&mut self, w: &Window, theta: &[f64], lay: Layout) -> Linear {
        self.evaluate(w, theta, lay);
        self.normal(w, theta, lay);
        let (n, p, nu) = (lay.n(), w.rows * w.cols, self.pix.len());
        let mut jac = vec![0.0; n * p];
        for a in 0..n {
            for (k, &q) in self.pix.iter().enumerate() {
                jac[a * p + q] = self.jt[a * nu + k];
            }
        }
        Linear { m: self.m.clone(), jac, g: self.g.clone(), f: self.f.clone() }
    }

    /// Minimize `I` from `theta0` within `[lo, hi]`, until no parameter can
    /// lower it by more than `tol` (in `I` units: nats times `phi`) moving
    /// alone ([`coordinate_gain`]), or `max_iter` steps.
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        w: &Window,
        theta0: &[f64],
        lay: Layout,
        lo: &[f64],
        hi: &[f64],
        max_iter: usize,
        tol: f64,
    ) -> Fit {
        let n = lay.n();
        assert!(theta0.len() == n && lo.len() == n && hi.len() == n);
        let inside = |q: usize, v: f64| {
            let margin = INTERIOR_FRAC * (hi[q] - lo[q]);
            v.clamp(lo[q] + margin, hi[q] - margin)
        };
        let mut t: Vec<f64> = theta0.iter().enumerate().map(|(q, &v)| inside(q, v)).collect();
        self.chol.ensure(n);
        let mut cur = self.divergence(w, &t, lay);
        let (mut lam, mut nu) = (1e-2, 2.0);
        let (mut converged, mut stalled, mut iterations) = (false, false, 0);
        let mut a = vec![0.0; n * n];
        let (mut step, mut t2, mut v) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        loop {
            // self.m is the model at t here.
            self.normal(w, &t, lay);
            if coordinate_gain(&t, &self.g, &self.f, lo, hi) <= tol {
                converged = true;
                break;
            }
            if iterations == max_iter {
                break;
            }
            iterations += 1;
            // v: distance to the bound the gradient heads for.
            for q in 0..n {
                v[q] = if self.g[q] >= 0.0 { t[q] - lo[q] } else { hi[q] - t[q] }.max(1e-12);
            }
            let mut accepted = false;
            while lam < 1e12 {
                // Coleman-Li's |g| / v and Marquardt's lambda F_qq: both scale
                // with each parameter's information, so the path does not
                // depend on parameter units or the camera gain.
                a.copy_from_slice(&self.f);
                for q in 0..n {
                    a[q * n + q] += self.g[q].abs() / v[q] + lam * self.f[q * n + q];
                }
                if !self.chol.factor(&a, n) {
                    lam *= 10.0;
                    continue;
                }
                for q in 0..n {
                    step[q] = -self.g[q];
                }
                self.chol.solve_in_place(&mut step);
                for q in 0..n {
                    let bound = if step[q] > 0.0 { hi[q] } else { lo[q] };
                    let d = if (t[q] + step[q] - bound) * step[q] > 0.0 { STEP_IN * (bound - t[q]) } else { step[q] };
                    t2[q] = inside(q, t[q] + d);
                }
                let mut pred = 0.0;
                for q in 0..n {
                    let dq = t2[q] - t[q];
                    let fd: f64 = (0..n).map(|r| self.f[q * n + r] * (t2[r] - t[r])).sum();
                    pred -= dq * self.g[q] + 0.5 * dq * fd;
                }
                let i2 = self.divergence(w, &t2, lay);
                let rho = if pred > 0.0 { (cur - i2) / pred } else { -1.0 };
                if i2 < cur && rho > 1e-4 {
                    std::mem::swap(&mut t, &mut t2);
                    cur = i2;
                    lam = (lam * (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3))).max(1e-12);
                    nu = 2.0;
                    accepted = true;
                    break;
                }
                lam *= nu;
                nu *= 2.0;
            }
            if !accepted {
                // Leave self.m at t for the caller.
                self.evaluate(w, &t, lay);
                stalled = true;
                break;
            }
        }
        let at_bound = (0..n).map(|q| at_bound(t[q], lo[q], hi[q])).collect();
        Fit { theta: t, divergence: cur, iterations, converged, stalled, at_bound }
    }

    /// The variances of the estimates: the diagonal of the inverse expected
    /// Fisher information at `theta`, times `phi`. `None` when the
    /// information cannot be factored.
    pub fn variances(&mut self, w: &Window, theta: &[f64], lay: Layout, phi: f64) -> Option<Vec<f64>> {
        let n = lay.n();
        self.evaluate(w, theta, lay);
        self.normal(w, theta, lay);
        self.chol.ensure(n);
        if !self.chol.factor(&self.f, n) {
            return None;
        }
        let mut var = vec![0.0; n];
        self.chol.inv_diag(&mut var, &mut self.scratch);
        var.iter_mut().for_each(|v| *v *= phi);
        Some(var)
    }
}

/// `acc[i][j] = sum_k x[i][k] y[j][k]`, as two interleaved partial sums
/// (even and odd `k`) added at the end: a fixed order, so the result does
/// not depend on the machine, that maps onto two-lane vector registers.
#[inline(always)]
fn tile(x: [&[f64]; BLOCK], y: [&[f64]; BLOCK], nu: usize) -> [[f64; BLOCK]; BLOCK] {
    let mut acc = [[[0.0f64; 2]; BLOCK]; BLOCK];
    let pairs = nu / 2;
    for p in 0..pairs {
        let k = 2 * p;
        let xk: [[f64; 2]; BLOCK] = std::array::from_fn(|i| [x[i][k], x[i][k + 1]]);
        let yk: [[f64; 2]; BLOCK] = std::array::from_fn(|j| [y[j][k], y[j][k + 1]]);
        for i in 0..BLOCK {
            for j in 0..BLOCK {
                for l in 0..2 {
                    acc[i][j][l] += xk[i][l] * yk[j][l];
                }
            }
        }
    }
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let tail = if nu % 2 == 1 { x[i][nu - 1] * y[j][nu - 1] } else { 0.0 };
            acc[i][j][0] + acc[i][j][1] + tail
        })
    })
}

/// The most any one parameter can lower the objective's quadratic model by
/// moving alone, downhill and within its bounds: `g^2 / (2 F)` when the
/// Newton step fits, else `g r - F r^2 / 2` with `r` the room to the bound.
/// In the objective's units, so the test against `tol` does not depend on
/// parameter units or the camera gain.
fn coordinate_gain(t: &[f64], g: &[f64], f: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    let p = t.len();
    (0..p)
        .map(|q| {
            let room = if g[q] >= 0.0 { t[q] - lo[q] } else { hi[q] - t[q] }.max(0.0);
            let (gq, fq) = (g[q].abs(), f[q * p + q].max(1e-300));
            let step = gq / fq;
            if step <= room { 0.5 * gq * step } else { gq * room - 0.5 * fq * room * room }
        })
        .fold(0.0, f64::max)
}
