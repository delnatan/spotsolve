//! Bounded Levenberg-Marquardt with Coleman-Li scaling.
//!
//! Minimize Poisson I-divergence:
//! `I(d, m) = sum(d * log(d/m) - (d - m))`, with `0 * log(0) = 0`.
//! The step uses gradient `J^T ((m-d)/m)` and expected Fisher information
//! `J^T diag(1/m) J`; this is Fisher scoring, not the observed Hessian.
//!
//! Parameters stay strictly inside their bounds because Coleman-Li scaling
//! divides by distance to a bound. `Interior` enforces this invariant.
//! Accept steps using the predicted decrease of the actual feasible step;
//! certify convergence with the information-scaled projected gradient.
//!
//! Historical benchmarks are in `docs/archive/DETECTOR_DESIGN_NOTES.md`;
//! floating-point port contracts are in `docs/archive/PORTING_NOTES.md`.

use crate::linalg::Chol;
use crate::psf;

/// Fraction of each bound's own range kept as a margin.
///
/// A relative margin accommodates different position and flux scales.
const INTERIOR_FRAC: f64 = 1e-10;

/// Box constraints, with the interiority margin precomputed.
pub struct Bounds {
    lo: Vec<f64>,
    hi: Vec<f64>,
    margin: Vec<f64>,
}

impl Bounds {
    pub fn new(lo: &[f64], hi: &[f64]) -> Self {
        assert_eq!(lo.len(), hi.len());
        let margin = lo
            .iter()
            .zip(hi)
            .map(|(&l, &h)| INTERIOR_FRAC * (h - l).max(1e-12))
            .collect();
        Self {
            lo: lo.to_vec(),
            hi: hi.to_vec(),
            margin,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.lo.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lo.is_empty()
    }
    #[inline]
    pub fn lo(&self) -> &[f64] {
        &self.lo
    }
    #[inline]
    pub fn hi(&self) -> &[f64] {
        &self.hi
    }

    #[inline]
    fn pull_inside(&self, i: usize, v: f64) -> f64 {
        v.max(self.lo[i] + self.margin[i])
            .min(self.hi[i] - self.margin[i])
    }
}

/// A parameter vector kept strictly inside its bounds. Constructors enforce
/// this invariant, and no mutable slice is exposed.
#[derive(Clone)]
pub struct Interior(Vec<f64>);

impl Interior {
    pub fn new(theta: &[f64], b: &Bounds) -> Self {
        assert_eq!(theta.len(), b.len());
        Interior(
            theta
                .iter()
                .enumerate()
                .map(|(i, &v)| b.pull_inside(i, v))
                .collect(),
        )
    }

    /// `self <- interior(base + delta)`, preserving the margin to each bound.
    pub fn set_step(&mut self, base: &Interior, delta: &[f64], b: &Bounds) {
        debug_assert_eq!(delta.len(), b.len());
        self.0.clear();
        self.0.extend(
            base.0
                .iter()
                .zip(delta)
                .enumerate()
                .map(|(i, (&t, &d))| b.pull_inside(i, t + d)),
        );
    }

    /// `self <- interior(theta)`, reusing the allocation.
    pub fn set_from(&mut self, theta: &[f64], b: &Bounds) {
        assert_eq!(theta.len(), b.len());
        self.0.clear();
        self.0
            .extend(theta.iter().enumerate().map(|(i, &v)| b.pull_inside(i, v)));
    }

    pub fn copy_from(&mut self, other: &Interior) {
        self.0.clear();
        self.0.extend_from_slice(&other.0);
    }

    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// The Poisson I-divergence. `0*log(0) := 0`.
///
/// `d <= 0` is expected in real, offset-subtracted data wherever read noise
/// dips a pixel below the fitted baseline, so the log's argument is guarded as
/// well as its coefficient.
pub fn i_divergence(d: &[f64], m: &[f64]) -> f64 {
    debug_assert_eq!(d.len(), m.len());
    let mut s = 0.0;
    for (&di, &mi) in d.iter().zip(m) {
        let term = if di > 0.0 { di * (di / mi).ln() } else { 0.0 };
        s += term - (di - mi);
    }
    s
}

/// Outcome of one bounded fit at fixed `K`.
#[derive(Clone, Copy, Debug)]
pub struct FitInfo {
    /// Poisson I-divergence at the solution, in nats.
    pub i_div: f64,
    pub n_iter: usize,
    /// The information-scaled projected gradient meets the stopping tolerance.
    pub converged: bool,
    /// Lambda saturated without an improving step. Distinct from `converged`.
    pub stalled: bool,
}

/// Optimizer settings.
#[derive(Clone, Copy, Debug)]
pub struct FitOpts {
    pub max_iter: usize,
    /// Objective tolerance in nats. Convergence requires the information-scaled
    /// projected score to be <= max(tol_grad, sqrt(2 * tol_obj)).
    pub tol_obj: f64,
    /// Backstop only -- absolute, and the natural scale here is set by fluxes
    /// running to ~2000 electrons, so this is near f64 noise.
    pub tol_grad: f64,
    pub lambda0: f64,
}

impl Default for FitOpts {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol_obj: 1e-8,
            tol_grad: 1e-6,
            lambda0: 1e-2,
        }
    }
}

/// Reusable fit buffers. `ensure` grows storage as needed; iterations do
/// not allocate after the workspace reaches the required size.
pub struct FitWorkspace {
    factors: psf::Factors,
    ay: Vec<f64>,
    ax: Vec<f64>,
    /// `d > 0` mask and the guarded numerator: loop invariants of the objective.
    d_pos: Vec<bool>,
    d_safe: Vec<f64>,
    m: Vec<f64>,
    j: Vec<f64>,
    m_trial: Vec<f64>,
    j_trial: Vec<f64>,
    /// `p x p` Fisher information, and the damped copy the inner loop solves.
    f: Vec<f64>,
    a: Vec<f64>,
    /// `W * J`, parameter-major like `j`.
    wj: Vec<f64>,
    resid: Vec<f64>,
    grad: Vec<f64>,
    delta: Vec<f64>,
    /// Coleman-Li `sqrt(v)` and the two damping diagonals built from it.
    s: Vec<f64>,
    jac_of_v: Vec<f64>,
    dinv2: Vec<f64>,
    chol: Chol,
    theta: Interior,
    theta_trial: Interior,
}

impl FitWorkspace {
    pub fn new() -> Self {
        Self {
            factors: psf::Factors::new(0, 0, 0),
            ay: Vec::new(),
            ax: Vec::new(),
            d_pos: Vec::new(),
            d_safe: Vec::new(),
            m: Vec::new(),
            j: Vec::new(),
            m_trial: Vec::new(),
            j_trial: Vec::new(),
            f: Vec::new(),
            a: Vec::new(),
            wj: Vec::new(),
            resid: Vec::new(),
            grad: Vec::new(),
            delta: Vec::new(),
            s: Vec::new(),
            jac_of_v: Vec::new(),
            dinv2: Vec::new(),
            chol: Chol::new(0),
            theta: Interior(Vec::new()),
            theta_trial: Interior(Vec::new()),
        }
    }

    fn ensure(&mut self, h: usize, w: usize, k: usize, p: usize) {
        let n = h * w;
        self.factors.ensure(h, w, k);
        if self.ay.len() != h {
            self.ay = psf::local_axis(h);
        }
        if self.ax.len() != w {
            self.ax = psf::local_axis(w);
        }
        grow(&mut self.d_safe, n);
        self.d_pos.resize(self.d_pos.len().max(n), false);
        for v in [&mut self.m, &mut self.m_trial, &mut self.resid] {
            grow(v, n);
        }
        for v in [&mut self.j, &mut self.j_trial, &mut self.wj] {
            grow(v, p * n);
        }
        for v in [&mut self.f, &mut self.a] {
            grow(v, p * p);
        }
        for v in [
            &mut self.grad,
            &mut self.delta,
            &mut self.s,
            &mut self.jac_of_v,
            &mut self.dinv2,
        ] {
            grow(v, p);
        }
        if self.chol.capacity() < p * p {
            self.chol = Chol::new(p);
        }
    }

    /// The fitted parameters. Valid after [`fit_var_sigma`].
    pub fn theta(&self) -> &[f64] {
        self.theta.as_slice()
    }

    /// The `p x p` Fisher information at the solution, row-major.
    ///
    /// This is the matrix of *this* fit, the one whose parameters are reported.
    /// Never take standard errors from a proposal fit.
    pub fn fisher(&self, p: usize) -> &[f64] {
        &self.f[..p * p]
    }

    /// The objective's gradient at the returned parameters.
    ///
    /// At active bounds, components give the KKT multipliers.
    pub fn gradient(&self, p: usize) -> &[f64] {
        &self.grad[..p]
    }
}

impl Default for FitWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

fn grow(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// Fit `[background, amplitude, y, x, sigma, ...]` on a row-major patch.
/// `halo` adds fixed neighboring light and background shape. Data and model
/// must use the same units (the detector uses ADU above the offset).
/// Results remain in [`FitWorkspace::theta`] and [`FitWorkspace::fisher`].
///
/// Do not shorten fits to accelerate model selection: proposed models often
/// start farther from their optimum, so truncation can favor fewer emitters.
pub fn fit_var_sigma(
    ws: &mut FitWorkspace,
    theta0: &[f64],
    h: usize,
    w: usize,
    d: &[f64],
    bounds: &Bounds,
    halo: Option<&[f64]>,
    opts: FitOpts,
) -> FitInfo {
    let n = h * w;
    let p = theta0.len();
    let k = psf::n_emitters_var(theta0);
    assert_eq!(d.len(), n);
    assert_eq!(bounds.len(), p);
    ws.ensure(h, w, k, p);

    ws.theta.set_from(theta0, bounds);
    ws.theta_trial.copy_from(&ws.theta);

    // Data-only terms of the objective: `d` is fixed for the whole fit, so the
    // mask and the guarded numerator are loop invariants [P5].
    for i in 0..n {
        ws.d_pos[i] = d[i] > 0.0;
        ws.d_safe[i] = if d[i] > 0.0 { d[i] } else { 1.0 };
    }

    let mut lam = opts.lambda0;
    let mut nu = 2.0f64;

    eval(ws, h, w, halo, p, false);
    let mut i_cur = idiv(ws, d, n, false);

    let mut converged = false;
    let mut stalled = false;
    let mut it = 0usize;

    for iter in 1..=opts.max_iter {
        it = iter;
        // grad = J^T (W (m - d)), F = J^T (W J), with W = 1/m held fixed within
        // the iteration -- Fisher scoring, so F is the expected information.
        for i in 0..n {
            ws.resid[i] = (ws.m[i] - d[i]) / ws.m[i];
        }
        for q in 0..p {
            let col = &ws.j[q * n..q * n + n];
            let mut g = 0.0;
            for i in 0..n {
                g += col[i] * ws.resid[i];
            }
            ws.grad[q] = g;
        }
        fisher(ws, p, n, false);

        if projected_score(ws, bounds, p) <= opts.tol_grad.max((2.0 * opts.tol_obj).sqrt()) {
            converged = true;
            break;
        }

        coleman_li_scale(&ws.theta, &ws.grad[..p], bounds, &mut ws.s[..p]);
        // Both extra diagonals are fixed for this outer iteration; only the
        // scalar `lam` in front of the second changes as the inner loop grows
        // it. Built once here rather than per lambda trial [P5].
        //
        // Preserve `|g|/(s*s)` and `(1/s)^2` evaluation order for parity [P2].
        for q in 0..p {
            let s = ws.s[q];
            ws.jac_of_v[q] = ws.grad[q].abs() / (s * s);
            let dinv = 1.0 / s;
            ws.dinv2[q] = dinv * dinv;
        }

        let mut step_accepted = false;

        for _ in 0..30 {
            // Coleman-Li system:
            // (F + diag(|grad|/s^2) + lambda*diag(1/s^2)) delta = -grad.
            // Both diagonal terms use s^2; using s for either under-damps bound steps.
            ws.a[..p * p].copy_from_slice(&ws.f[..p * p]);
            for q in 0..p {
                // Two separate adds, in this order, so the summation is
                // (F_ii + u_i) + lam*v_i -- not F_ii + (u_i + lam*v_i) [P2].
                ws.a[q * p + q] += ws.jac_of_v[q];
                ws.a[q * p + q] += lam * ws.dinv2[q];
            }
            for q in 0..p {
                ws.delta[q] = -ws.grad[q];
            }
            if !ws.chol.factor(&ws.a[..p * p], p) {
                lam *= 10.0; // not positive definite, or non-finite
                continue;
            }
            ws.chol.solve_in_place(&mut ws.delta[..p]);
            scale_into_box(&ws.theta, &mut ws.delta[..p], bounds);

            let (theta_ref, trial) = (&ws.theta, &mut ws.theta_trial);
            trial.set_step(theta_ref, &ws.delta[..p], bounds);
            // Predicted decrease in I, in nats, of the actual feasible step,
            // after boundary scaling and interior projection. Using the
            // unscaled step's promise can reject a good bounded step and drive
            // lambda to saturation.
            for q in 0..p {
                ws.delta[q] = ws.theta_trial.as_slice()[q] - ws.theta.as_slice()[q];
            }
            let pred_dec = quadratic_decrease(&ws.grad[..p], &ws.f[..p * p], &ws.delta[..p]);
            eval(ws, h, w, halo, p, true);
            let i_trial = idiv(ws, d, n, true);

            let actual_dec = i_cur - i_trial;
            // Gain ratio: actual / predicted decrease of the feasible step.
            // Using only the sign of the improvement can accept tiny steps along
            // poorly constrained directions and drive damping to its floor.
            let rho = if pred_dec > 0.0 {
                actual_dec / pred_dec
            } else {
                -1.0
            };

            if i_trial < i_cur && rho > 1e-4 {
                i_cur = i_trial;
                accept(ws);
                // Nielsen (1999): aggressive for a trustworthy step, gentle for
                // a marginal one.
                lam = (lam * (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3))).max(1e-12);
                nu = 2.0;
                step_accepted = true;
                break;
            }
            lam *= nu;
            nu *= 2.0;
            if lam > 1e12 {
                break; // give up on this outer iteration
            }
        }

        if !step_accepted {
            stalled = true; // lambda saturated; NOT the same as converged
            break;
        }
    }

    // Final Fisher information, from the accepted model. This is the matrix
    // whose parameters are reported, and the only one standard errors may come
    // from.
    fisher(ws, p, n, true);
    // A small damped step is not a stationarity certificate. Re-evaluate
    // the score at the returned parameters, including on the last allowed
    // iteration or when objective differences ran out of precision.
    // Computed even for a zero-iteration evaluation, so that
    // [`FitWorkspace::gradient`] never returns the previous fit's.
    for q in 0..p {
        ws.grad[q] = (0..n)
            .map(|i| ws.j[q * n + i] * ((ws.m[i] - d[i]) / ws.m[i]))
            .sum();
    }
    if opts.max_iter > 0 {
        converged =
            projected_score(ws, bounds, p) <= opts.tol_grad.max((2.0 * opts.tol_obj).sqrt());
        if converged {
            stalled = false;
        }
    }

    FitInfo {
        i_div: i_cur,
        n_iter: it,
        converged,
        stalled,
    }
}

/// Largest feasible, information-scaled gradient component. A coordinate can
/// move at most one conditional standard error or its distance to the bound
/// in the descent direction. Unlike a damped LM step, this does not shrink as
/// lambda grows. Away from the bounds, its square/2 is the coordinate-wise
/// quadratic improvement in nats. It tests stationarity, not identifiability.
fn projected_score(ws: &FitWorkspace, bounds: &Bounds, p: usize) -> f64 {
    (0..p)
        .map(|q| {
            let g = ws.grad[q];
            let distance = if g >= 0.0 {
                ws.theta.as_slice()[q] - bounds.lo()[q]
            } else {
                bounds.hi()[q] - ws.theta.as_slice()[q]
            };
            let scale = 1.0 / ws.f[q * p + q].max(1e-30).sqrt();
            g.abs() * distance.max(0.0).min(scale)
        })
        .fold(0.0, f64::max)
}

fn quadratic_decrease(grad: &[f64], fisher: &[f64], delta: &[f64]) -> f64 {
    let p = grad.len();
    let linear = grad.iter().zip(delta).map(|(g, d)| g * d).sum::<f64>();
    let quadratic = (0..p)
        .map(|q| delta[q] * (0..p).map(|r| fisher[q * p + r] * delta[r]).sum::<f64>())
        .sum::<f64>();
    -linear - 0.5 * quadratic
}

/// Compute `F = J^T diag(1/m) J`. Mirror the upper triangle so the result
/// is exactly symmetric despite floating-point rounding.
fn fisher(ws: &mut FitWorkspace, p: usize, n: usize, clip: bool) {
    // `W` multiplies the SECOND factor, matching `J.T @ (W[:,None] * J)`.
    //
    // Branch outside the inner loops. `eval` already floors the model at 1e-9.
    if clip {
        for q in 0..p {
            for i in 0..n {
                ws.wj[q * n + i] = ws.j[q * n + i] / ws.m[i].max(1e-9);
            }
        }
    } else {
        for q in 0..p {
            for i in 0..n {
                ws.wj[q * n + i] = ws.j[q * n + i] / ws.m[i];
            }
        }
    }
    // Accumulate four independent matrix entries per pass. Each entry keeps
    // its original pixel summation order; do not reassociate or fuse operations.
    // This hides addition latency without changing the numerical result.
    let (j, wj, f) = (&ws.j, &ws.wj, &mut ws.f);
    for q1 in 0..p {
        let c1 = &j[q1 * n..q1 * n + n];
        let mut q2 = q1;
        while q2 + 4 <= p {
            let (b0, b1) = (&wj[q2 * n..q2 * n + n], &wj[(q2 + 1) * n..(q2 + 1) * n + n]);
            let (b2, b3) = (
                &wj[(q2 + 2) * n..(q2 + 2) * n + n],
                &wj[(q2 + 3) * n..(q2 + 3) * n + n],
            );
            let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
            for i in 0..n {
                let v = c1[i];
                s0 += v * b0[i];
                s1 += v * b1[i];
                s2 += v * b2[i];
                s3 += v * b3[i];
            }
            for (t, s) in [s0, s1, s2, s3].into_iter().enumerate() {
                f[q1 * p + q2 + t] = s;
                f[(q2 + t) * p + q1] = s;
            }
            q2 += 4;
        }
        while q2 < p {
            let c2 = &wj[q2 * n..q2 * n + n];
            let mut s = 0.0;
            for i in 0..n {
                s += c1[i] * c2[i];
            }
            f[q1 * p + q2] = s;
            f[q2 * p + q1] = s;
            q2 += 1;
        }
    }
}

/// Evaluate model and Jacobian at the current or trial iterate, clipping the
/// model away from zero -- `W = 1/m` is singular at `m = 0`.
fn eval(
    ws: &mut FitWorkspace,
    h: usize,
    w: usize,
    halo: Option<&[f64]>,
    p: usize,
    trial: bool,
) {
    let n = h * w;
    let (theta, m, j) = if trial {
        (ws.theta_trial.as_slice(), &mut ws.m_trial, &mut ws.j_trial)
    } else {
        (ws.theta.as_slice(), &mut ws.m, &mut ws.j)
    };
    psf::model_and_jac_var_sigma_ax(
        theta,
        &ws.ay[..h],
        &ws.ax[..w],
        halo,
        &mut ws.factors,
        &mut m[..n],
        &mut j[..p * n],
    );
    for v in m[..n].iter_mut() {
        *v = v.max(1e-9);
    }
}

fn idiv(ws: &FitWorkspace, d: &[f64], n: usize, trial: bool) -> f64 {
    let m = if trial { &ws.m_trial } else { &ws.m };
    let mut s = 0.0;
    for i in 0..n {
        let term = if ws.d_pos[i] {
            ws.d_safe[i] * (ws.d_safe[i] / m[i]).ln()
        } else {
            0.0
        };
        s += term - (d[i] - m[i]);
    }
    s
}

/// The trial becomes the iterate. The model and Jacobian buffers are swapped,
/// not copied: both pairs are grown together by `ensure`, and the trial's are
/// overwritten by the next `eval` before anything reads them.
fn accept(ws: &mut FitWorkspace) {
    let (a, b) = (&mut ws.theta, &ws.theta_trial);
    a.copy_from(b);
    std::mem::swap(&mut ws.m, &mut ws.m_trial);
    std::mem::swap(&mut ws.j, &mut ws.j_trial);
}

/// `sqrt` of the distance to whichever bound the step is heading toward.
///
/// `v_i` is the distance to the LOWER bound where the step wants to decrease
/// `theta_i` (`grad >= 0`) and to the UPPER bound otherwise. The two extra
/// diagonals the caller adds are both divided by this value *squared*.
fn coleman_li_scale(theta: &Interior, grad: &[f64], b: &Bounds, s: &mut [f64]) {
    let t = theta.as_slice();
    for i in 0..s.len() {
        let v = if grad[i] >= 0.0 {
            t[i] - b.lo()[i]
        } else {
            b.hi()[i] - t[i]
        };
        s[i] = v.max(1e-12).sqrt();
    }
}

/// If a step crosses a bound, move that coordinate 99.5% of the way there.
/// A single coordinate near a bound must not shrink every other coordinate.
/// `Interior::set_step` applies the final strict-interiority margin.
fn scale_into_box(theta: &Interior, delta: &mut [f64], b: &Bounds) {
    let t = theta.as_slice();
    for i in 0..delta.len() {
        let trial = t[i] + delta[i];
        if trial > b.hi()[i] {
            delta[i] = 0.995 * (b.hi()[i] - t[i]);
        } else if trial < b.lo()[i] {
            delta[i] = 0.995 * (b.lo()[i] - t[i]);
        }
    }
}

#[cfg(test)]
mod width_tests {
    use super::*;

    #[test]
    fn predicted_decrease_uses_the_feasible_step() {
        // Quadratic f(x) = x²/2 at x=2. A bound allows only a -0.1 step:
        // its actual and predicted improvement are 0.195, not the 2.0
        // improvement promised by the unconstrained Newton step of -2.
        let prediction = quadratic_decrease(&[2.0], &[1.0], &[-0.1]);
        assert!((prediction - (2.0 - 0.5 * 1.9 * 1.9)).abs() < 1e-14);
    }
}
