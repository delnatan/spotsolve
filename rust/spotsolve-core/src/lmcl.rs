//! Bounded Levenberg-Marquardt with Coleman-Li affine scaling.
//!
//! Ports `lmga.py`. **The name differs deliberately**: `lmga` meant "LM with
//! geodesic acceleration", and the geodesic acceleration was implemented,
//! measured and removed long ago -- it changed the converged objective by under
//! 1e-13 on ordinary fits while costing 133 model/Jacobian evaluations per fit
//! instead of 18. What actually distinguishes this optimizer is the Coleman-Li
//! affine scaling, so it is named for that. The `02_lmga` fixture keeps its
//! name; it is the same code.
//!
//! # Objective
//!
//! The Poisson I-divergence
//!
//! ```text
//! I(d, m) = sum_i [ d_i*log(d_i/m_i) - (d_i - m_i) ]      (0*log(0) := 0)
//! ```
//!
//! minimized by Fisher scoring: `W = diag(1/m)` is held fixed within an
//! iteration, so `F = J^T W J` is the *expected* Fisher information -- exact
//! for Poisson's canonical link, where Fisher scoring coincides with IRLS.
//!
//! # Strict interiority is a precondition, not a preference
//!
//! Coleman-Li divides by `v_i`, the distance from parameter `i` to the bound
//! its step is heading toward. A parameter resting exactly *on* a bound sets
//! `v_i = 0`, and the damage is not local to it: the fraction-to-boundary rule
//! computes one scalar step scale from `min_i (bound_i - theta_i)/delta_i`, so
//! a single stuck coordinate collapses the step for **every** coordinate. The
//! step then buys ~1e-11 nats, its gain ratio reads ~2e-8 -- which measures the
//! clipping, not the quadratic model -- so it is rejected, lambda ratchets up,
//! and the fit burns its whole budget on micro-steps.
//!
//! Measured on a 39x39 frame before this was fixed in the Python: 68.6% of LM
//! iterations began with a parameter on a bound, 46.5% of inner trials were
//! scaled to the 1e-8 floor, and 73.2% of all fits exhausted `max_iter`.
//! Removing the clip made the frame solve 2.5x faster *and* made the emitter
//! count reproducible.
//!
//! The Python documented this invariant in its module docstring and violated it
//! in its body for months. Here [`Interior`] makes the broken state
//! unrepresentable: there is no way to obtain one that is not strictly inside
//! its box, and no `&mut [f64]` escapes [P1].

use crate::linalg::Chol;
use crate::psf;

/// Fraction of each bound's own range kept as a margin.
///
/// **Relative**, because the ranges are not comparable: a position is bounded
/// over ~20 px and an amplitude over ~1e4 electrons, so one absolute epsilon
/// would be a completely different constraint for each [P1].
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

/// A parameter vector guaranteed strictly inside its box.
///
/// The constructors are the only way in and every one of them pulls the value
/// inside, so the optimizer cannot express the state that broke it [P1].
/// Deliberately exposes no mutable slice.
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

    /// `self <- interior(base + delta)`. The only way to advance an iterate.
    ///
    /// NOT a clamp onto the bounds: clamping parks a parameter exactly on one,
    /// which is precisely the state this type exists to prevent.
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

    /// `self <- interior(theta)`, reusing this vector's allocation.
    ///
    /// `fit` is called on the order of 700k times per frame, so even a
    /// once-per-fit allocation is worth not making [P6].
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
    /// The gradient or the objective genuinely converged. NOT merely "stopped".
    pub converged: bool,
    /// Lambda saturated without an improving step. Distinct from `converged`.
    pub stalled: bool,
}

/// Tuning, matching `lmga.fit`'s keyword defaults.
#[derive(Clone, Copy, Debug)]
pub struct FitOpts {
    pub max_iter: usize,
    /// Objective resolution, **in nats**: a fit has converged when its
    /// information-scaled projected score is <= sqrt(2*tol_obj).
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

/// Everything one fit needs, allocated once and borrowed for the duration.
///
/// The Python's LM inner loop allocates two dense `p x p` matrices per lambda
/// trial -- whose off-diagonals are known to be zero -- purely to write
/// `F + diag(u) + lam*diag(v)`, and there are ~700k such trials per frame [P6].
/// It also re-derives the separable pixel axes and the I-divergence's data-only
/// terms on every one of the ~160 model evaluations a fit makes [P5]. All of
/// that lives here instead.
///
/// `ensure` grows the buffers on demand, so after the first few patches of a
/// run `fit` allocates nothing at all.
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

    /// The fitted parameters. Valid after [`fit`].
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
    /// Interior components are zero to the stationarity tolerance, so what
    /// this is actually read for is the components at an active bound -- the
    /// KKT multipliers.
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

/// Fit a variable-sigma theta `[b, A0, y0, x0, sigma0, ...]` on a `h x w`
/// patch by bounded Fisher-scoring LM: the Poisson maximum likelihood.
///
/// `d` is the patch data in photoelectrons, row-major. `halo` is the
/// parameter-free additive contribution (frozen neighbours plus the
/// background's shape term), or `None`. Results are left in the workspace:
/// [`FitWorkspace::theta`] and [`FitWorkspace::fisher`].
///
/// Each step is assessed on the actual feasible quadratic step, and
/// convergence is certified by an information-scaled projected gradient. The
/// trajectory intentionally need not follow the Python reference's.
///
/// # Why `max_iter` must not be cut for speed
///
/// Truncating iterations does *not* add symmetric noise. A proposal fit starts
/// further from its optimum than the incumbent it is compared against, so
/// truncation systematically leaves the proposal's objective too high and
/// biases model selection toward the smaller model -- it under-credits exactly
/// the moves that add an emitter. Measured: capping at 40 iterations still left
/// 21% of fits more than 0.1 nat above their optimum, with a p99 gap of 72
/// nats. Make each iteration cheaper instead.
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
        // NOTE the two are written differently on purpose: `|g|/(s*s)` and
        // `(1/s)^2`. Those are not the same float, and the Python computes them
        // exactly this way [P2].
        for q in 0..p {
            let s = ws.s[q];
            ws.jac_of_v[q] = ws.grad[q].abs() / (s * s);
            let dinv = 1.0 / s;
            ws.dinv2[q] = dinv * dinv;
        }

        let mut step_accepted = false;

        for _ in 0..30 {
            // Coleman-Li form:
            //   (F + diag(|grad|/s^2) + lam*diag(1/s^2)) delta = -grad
            //
            // The scaled-space system is (D F D + diag(|grad|) + lam I) shat =
            // -D grad with D = diag(s), s = sqrt(v). Mapping back to the
            // unscaled step delta = D shat sends BOTH extra diagonals through
            // D^-1 (.) D^-1, so both pick up 1/s^2 -- not 1/s for one of them.
            // The mismatched version under-damps every parameter approaching a
            // bound (at v = 0.01 it applies 10|g| where the correct term is
            // 100|g|). Over 200 randomized 1-3 emitter fits the consistent form
            // reached a lower converged I 6 times to 1 with 193 ties, better by
            // 3.5 nats on average -- large on the scale a Bayes factor is
            // decided on -- in 10.9 iterations against 20.4.
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
            // LM gain ratio: how much of the promised improvement was real.
            // lambda MUST be driven by this and not by the sign of the
            // improvement alone. Accepting any decrease and halving lambda for
            // it lets lambda collapse to its floor while the quadratic model is
            // worthless, and then nothing damps the near-null directions of F.
            // Traced on a real patch (K=7, one emitter at the amplitude floor
            // so cond(F) = 1.5e20): every iteration predicted 5.2e4 nats,
            // delivered 1.05e-3, halved lambda anyway, and took the identical
            // 2.3e-5 step again -- 3000+ iterations, finishing 364 nats above
            // the optimum.
            let rho = if pred_dec > 0.0 {
                actual_dec / pred_dec
            } else {
                -1.0
            };

            if i_trial < i_cur && rho > 1e-4 {
                i_cur = i_trial;
                accept(ws, p, n);
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

/// `F = J^T W J` with `W = diag(1/m)`, into `ws.f`.
///
/// Only the upper triangle is computed and then mirrored. `F` is symmetric by
/// construction, and computing both halves independently is not merely wasted
/// work: it produces `F_ij != F_ji` in the last ulp, because the two are
/// separate reductions over different orders. That is the whole reason [P3]
/// exists. Mirroring makes the matrix exactly symmetric, so
/// `Chol::factor`'s symmetrization becomes a no-op and the question of which
/// triangle gets read cannot arise at all.
///
/// This is the dominant arithmetic in the fit: `p^2 * n` per outer iteration
/// against `p*n` for everything else. The Python reaches it through BLAS
/// `dgemm`, so it is the one place the port does NOT start ahead.
fn fisher(ws: &mut FitWorkspace, p: usize, n: usize, clip: bool) {
    // `W` multiplies the SECOND factor, matching `J.T @ (W[:,None] * J)`.
    //
    // `clip` is loop-invariant, so it is tested once rather than `p*n` times;
    // that leaves the inner loop a bare divide. `eval` has already floored `m`
    // at the same 1e-9, so the clipped form is a no-op on a post-`eval` model
    // and is kept only because `fisher` is also called on a raw `m` [P3].
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
    // Four columns of `wj` per pass over `c1`.
    //
    // **This does not reassociate anything** [P2]. Each `s*` accumulates one
    // output element over `i` in the same increasing order the scalar loop
    // used, so every individual sum is bit-identical; what changes is only how
    // many *different* sums are in flight. A lone dot product is bound by the
    // ~3-cycle latency of the dependent `FADD`, not by throughput, so it
    // retires one add per three cycles however wide the machine is. Several
    // independent chains fill those slots with useful work.
    //
    // Measured on frame 0 of `beads_80pct-glycerol_crop.tif` (1141 emitters,
    // 14k window fits), median of 9 runs, interleaved builds, `positions /
    // amplitudes / se` SHA equal for every arm:
    //
    // ```text
    //   scalar    2.844 s  2.888 s
    //   width 2   2.531 s
    //   width 4   2.541 s  2.568 s     <- 11.2% under scalar
    //   width 8   2.637 s
    // ```
    //
    // 2 and 4 tie; 8 regresses, because `p` is 5 at the median (`K = 1`) and
    // only reaches 37, so a width-8 block almost never fills and the work
    // falls through to the scalar tail with the wider prologue already paid.
    // 4 is kept over 2 for the larger windows, where it has the longer runs to
    // amortize. This buys nothing on its own if `p` is small -- see §4: at
    // these sizes the ranking is not the FLOP count.
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

fn accept(ws: &mut FitWorkspace, p: usize, n: usize) {
    let (a, b) = (&mut ws.theta, &ws.theta_trial);
    a.copy_from(b);
    ws.m[..n].copy_from_slice(&ws.m_trial[..n]);
    ws.j[..p * n].copy_from_slice(&ws.j_trial[..p * n]);
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

/// Shrink `delta` so `theta + delta` stays inside the box, by one scalar factor
/// covering every coordinate (0.995 of the distance to the nearest bound).
fn scale_into_box(theta: &Interior, delta: &mut [f64], b: &Bounds) {
    let t = theta.as_slice();
    let mut scale = 1.0f64;
    let mut any = false;
    for i in 0..delta.len() {
        let trial = t[i] + delta[i];
        if trial > b.hi()[i] {
            any = true;
            scale = scale.min(0.995 * (b.hi()[i] - t[i]) / delta[i]);
        } else if trial < b.lo()[i] {
            any = true;
            scale = scale.min(0.995 * (b.lo()[i] - t[i]) / delta[i]);
        }
    }
    if any {
        let scale = scale.max(1e-8);
        for v in delta.iter_mut() {
            *v *= scale;
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
