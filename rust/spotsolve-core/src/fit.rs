//! One window's Poisson fit: a constant background and `k` pixel-integrated
//! Gaussians of one width,
//!
//! ```text
//! m[r, c] = c0 + sum_k A_k E(r; y_k, s) E(c; x_k, s)
//! I(d, m) = sum_used [d log(d / m) - (d - m)]
//! ```
//!
//! in window-local pixels (`[0, 0]` is the window's first pixel). Pixels not
//! `used` are left out of the likelihood, as u-track sets them to NaN.
//! `theta = [c0, A_1, y_1, x_1, ..., A_k, y_k, x_k]`, with the shared width
//! `s` appended when it is free ([`Layout`]); with `k = 1` and a free width
//! this is [`psf::pack_var`]'s layout.
//!
//! [`Fitter::fit`] minimizes `I` by bounded Levenberg-Marquardt (Fisher
//! scoring) with Coleman-Li affine scaling: parameters stay strictly inside
//! their bounds, and a coordinate's step shrinks with its distance to the
//! bound it heads for, so a flux cannot collapse onto its floor in one step
//! while the positions have yet to adapt. `I / phi` is the log-likelihood
//! in nats for a camera of dispersion `phi`; [`Fitter::covariance`] is the
//! inverse expected Fisher information scaled by `phi`.

use crate::linalg::Chol;
use crate::psf;

/// Fraction of each bound's range kept as a margin inside it.
pub const INTERIOR_FRAC: f64 = 1e-10;
/// A step that would cross a bound goes this fraction of the way there.
pub const STEP_IN: f64 = 0.995;
/// Relative distance to a bound at which a parameter counts as on it.
pub const BOUND_TOL: f64 = 1e-6;

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

    pub fn n_used(&self) -> usize {
        self.used.iter().filter(|&&u| u).count()
    }
}

/// How `theta` is laid out: `k` components, and the width fixed at
/// `Some(s)` or free (`None`, the last parameter).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    pub k: usize,
    pub sigma: Option<f64>,
}

impl Layout {
    pub fn n(&self) -> usize {
        1 + 3 * self.k + self.sigma.is_none() as usize
    }

    pub fn sigma(&self, theta: &[f64]) -> f64 {
        self.sigma.unwrap_or_else(|| theta[theta.len() - 1])
    }
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
    centres: Vec<f64>,
    m: Vec<f64>,
    jac: Vec<f64>,
    f: Vec<f64>,
    g: Vec<f64>,
    chol: Chol,
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
            centres: Vec::new(),
            m: Vec::new(),
            jac: Vec::new(),
            f: Vec::new(),
            g: Vec::new(),
            chol: Chol::new(1),
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
        let s = lay.sigma(theta);
        self.centres.clear();
        self.centres.extend((0..k).map(|i| theta[2 + 3 * i]));
        psf::factors_axis_sigma(&self.ay, &self.centres, s, &mut self.ey, &mut self.dey, &mut self.sey);
        self.centres.clear();
        self.centres.extend((0..k).map(|i| theta[3 + 3 * i]));
        psf::factors_axis_sigma(&self.ax, &self.centres, s, &mut self.ex, &mut self.dex, &mut self.sex);
        self.m.clear();
        self.m.resize(rows * cols, theta[0]);
        for i in 0..k {
            let a = theta[1 + 3 * i];
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

    /// Gradient of `I` and expected Fisher information `J^T diag(1/m) J` at
    /// the last evaluated `theta`, into `self.g` and `self.f`.
    fn normal(&mut self, w: &Window, theta: &[f64], lay: Layout) {
        let (rows, cols, k, n) = (w.rows, w.cols, lay.k, lay.n());
        let free = lay.sigma.is_none();
        self.f.clear();
        self.f.resize(n * n, 0.0);
        self.g.clear();
        self.g.resize(n, 0.0);
        self.jac.resize(n, 0.0);
        for r in 0..rows {
            for c in 0..cols {
                let q = r * cols + c;
                if !w.used[q] {
                    continue;
                }
                let j = &mut self.jac;
                j[0] = 1.0;
                if free {
                    j[n - 1] = 0.0;
                }
                for i in 0..k {
                    let a = theta[1 + 3 * i];
                    let (ey, ex) = (self.ey[i * rows + r], self.ex[i * cols + c]);
                    j[1 + 3 * i] = ey * ex;
                    j[2 + 3 * i] = a * self.dey[i * rows + r] * ex;
                    j[3 + 3 * i] = a * ey * self.dex[i * cols + c];
                    if free {
                        j[n - 1] += a * (self.sey[i * rows + r] * ex + ey * self.sex[i * cols + c]);
                    }
                }
                let m = self.m[q];
                let (wt, res) = (1.0 / m, 1.0 - w.d[q].max(0.0) / m);
                for a in 0..n {
                    self.g[a] += j[a] * res;
                    let ja = j[a] * wt;
                    for b in a..n {
                        self.f[a * n + b] += ja * j[b];
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

    /// Gradient of `I` and expected Fisher information at `theta`
    /// (`n` and `n x n`, row-major), in `I` units.
    pub fn gradient_and_information(&mut self, w: &Window, theta: &[f64], lay: Layout) -> (Vec<f64>, Vec<f64>) {
        self.evaluate(w, theta, lay);
        self.normal(w, theta, lay);
        (self.g.clone(), self.f.clone())
    }

    /// Minimize `I` from `theta0` within `[lo, hi]`, until no parameter can
    /// gain more than `tol` (in `I` units: nats times `phi`) by moving one
    /// conditional standard error, or `max_iter` steps.
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
        let gate = (2.0 * tol).sqrt();
        loop {
            // self.m is the model at t here.
            self.normal(w, &t, lay);
            if projected_score(&t, &self.g, &self.f, lo, hi) <= gate {
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
                a.copy_from_slice(&self.f);
                for q in 0..n {
                    a[q * n + q] += self.g[q].abs() / v[q] + lam / v[q];
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

    /// Inverse expected Fisher information at `theta`, times `phi`: the
    /// covariance of the estimates, `n x n` row-major. `None` when the
    /// information cannot be factored.
    pub fn covariance(&mut self, w: &Window, theta: &[f64], lay: Layout, phi: f64) -> Option<Vec<f64>> {
        let n = lay.n();
        self.evaluate(w, theta, lay);
        self.normal(w, theta, lay);
        self.chol.ensure(n);
        if !self.chol.factor(&self.f, n) {
            return None;
        }
        let mut cov = vec![0.0; n * n];
        let mut col = vec![0.0; n];
        for j in 0..n {
            col.fill(0.0);
            col[j] = 1.0;
            self.chol.solve_in_place(&mut col);
            for i in 0..n {
                cov[i * n + j] = phi * col[i];
            }
        }
        Some(cov)
    }
}

/// Largest feasible, information-scaled gradient component: each parameter
/// may move one conditional standard error, or to its bound, downhill.
fn projected_score(t: &[f64], g: &[f64], f: &[f64], lo: &[f64], hi: &[f64]) -> f64 {
    let p = t.len();
    (0..p)
        .map(|q| {
            let room = if g[q] >= 0.0 { t[q] - lo[q] } else { hi[q] - t[q] };
            g[q].abs() * room.max(0.0).min(1.0 / f[q * p + q].max(1e-30).sqrt())
        })
        .fold(0.0, f64::max)
}
