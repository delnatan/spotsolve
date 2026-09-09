//! Laplace-approximated Bayes factors for changes in emitter count.
//!
//! Ports `evidence.py`.
//!
//! # Why a Bayes factor and not a likelihood-ratio test
//!
//! The LRT for "is there one more emitter" is not chi2(3). Under H0 the extra
//! emitter sits on the BOUNDARY of the parameter space (`A -> 0`), and exactly
//! there its position is UNIDENTIFIABLE -- the likelihood is flat in two of the
//! three added directions. Both regularity conditions for Wilks' theorem fail.
//!
//! The practical consequence is worse than a fixed offset. The statistic is a
//! maximum over the larger model's parameter space, so a more thorough
//! optimizer finds a larger one: measured on synthetic single-emitter fields,
//! the false-positive rate at the nominal 5% chi2(3) cutoff moved from 3% to
//! 10% purely by raising the restart count from 4 to 25. **A test whose size
//! depends on how hard you search cannot be fixed by choosing a different
//! critical value.**
//!
//! A Bayes factor integrates the added dimensions against a proper prior rather
//! than maximizing over them, so the Occam factor charges for the search volume
//! automatically.
//!
//! # The model
//!
//! Poisson point process on emitters, `p(K) = Poisson(lam*Area)`; positions
//! uniform on the patch; amplitudes `A_k ~ Exp(1/A_s)`; background
//! `b ~ Uniform[0, b_max]`.
//!
//! For a `K -> K+1` move the Area factors cancel exactly -- the prior odds
//! contribute `lam*Area/(K+1)` and the new position prior contributes
//! `1/Area` -- so what survives is area-free:
//!
//! ```text
//! log BF = (I_K - I_{K+1})                     data term, in nats
//!        + log(lam) - log(K+1)                 prior odds on the count
//!        + d log pi_A                          amplitude prior difference
//!        + (3/2) log(2 pi)                     Laplace volume, 3 new params
//!        - (1/2)(log|F_{K+1}| - log|F_K|)      Occam factor
//! ```
//!
//! `b` has the same range on both sides of every comparison, so its uniform
//! prior cancels and never appears. Both priors have zero curvature, so the
//! Hessian of the negative log posterior **is** the Fisher information exactly.
//!
//! # What is deliberately absent
//!
//! There is no `amplitudes_resolved` / `A/SE >= 3` veto here. `RESOLVED_TAU` is
//! a statement about whether the Laplace approximation can be *computed*;
//! applied as a veto on proposal fits it silently became the pipeline's
//! strictest **detection rule**. Instrumented: ADD accepted 96-99% of what FIND
//! proposed and `log BF <= 0` fired on *none* of them, while 68-79% of every
//! true emitter lost inside 2 sigma died at that guard inside `_try_split`.
//! Removing it gained 1.7 recall points at no runtime cost. It belongs on the
//! removal path, where `_prune` re-tests the same conditions on a joint fit
//! with more information [P12]. Do not add it back here.
//!
//! The two principled boundary corrections (a truncated-normal
//! `sum log Phi(A_k/SE_k)`, and a prior-overflow cap on the Laplace volume) are
//! correct, are exactly 0 nats wherever the fit is well determined, and are
//! worth nothing end to end. They cost an eigendecomposition per proposal. Not
//! ported.

use crate::linalg::{self, Chol};

pub use crate::linalg::COND_GUARD;

/// The empirical-Bayes prior parameters, re-estimated each round.
///
/// These are **not** knobs: `lam` is the emitter density and `A_s` the
/// amplitude prior scale, and both are re-estimated from the current
/// configuration every round.
#[derive(Clone, Copy, Debug)]
pub struct Prior {
    /// Emitters per px^2.
    pub lam: f64,
    /// Amplitude prior scale, total flux in photoelectrons.
    pub a_s: f64,
}

/// Reusable storage for the factorizations the Bayes factors need.
pub struct Evidence {
    chol: Chol,
    scratch: Vec<f64>,
}

impl Default for Evidence {
    fn default() -> Self {
        Self::new()
    }
}

/// A precomputed `logdet` of an incumbent's Fisher matrix.
///
/// Every proposal in one search step is scored against the SAME incumbent, so
/// without this its matrix is refactorized once per proposal -- measured at
/// 44.5% of all `logdet_cond` calls on a 39x39 frame. Passing it in is exact,
/// not an approximation: it is the same function of the same matrix [P5].
pub type LogDet = (f64, bool);

impl Evidence {
    pub fn new() -> Self {
        Self {
            chol: Chol::new(linalg::P_MAX),
            scratch: Vec::new(),
        }
    }

    /// `(log|F|, ok)` from one factorization, with no condition number.
    ///
    /// Split from [`Evidence::logdet_cond`] because the condition number is
    /// extra work and most callers throw it away: [`Evidence::log_bf_add`]
    /// reads the `after` matrix's only, and [`Evidence::log_bf_remove`] reads
    /// neither.
    pub fn logdet(&mut self, f: &[f64], n: usize) -> LogDet {
        linalg::logdet(f, n, &mut self.chol)
    }

    /// `diag(F^-1)` into `out`; `false` if `F` is not positive definite.
    ///
    /// Read by `refine`'s reported standard errors and by `prune`'s `A/SE`
    /// test. Goes through the same Cholesky as everything else rather than a
    /// separate `np.linalg.inv`, so a decision at the `PRUNE_TAU` boundary
    /// cannot turn on which route to the inverse was taken [P13].
    pub fn inv_diag(
        &mut self,
        f: &[f64],
        n: usize,
        out: &mut [f64],
        scratch: &mut Vec<f64>,
    ) -> bool {
        if !self.chol.factor(f, n) {
            return false;
        }
        self.chol.inv_diag(out, scratch);
        true
    }

    /// `(log|F|, scaled condition number, ok)`.
    pub fn logdet_cond(&mut self, f: &[f64], n: usize) -> (f64, f64, bool) {
        linalg::logdet_cond(f, n, &mut self.chol, &mut self.scratch)
    }

    /// `log BF` for `K_before -> K_before + 1`. Positive favours the larger
    /// model.
    ///
    /// Returns `(log_bf, scaled_cond_after)` so the caller can apply
    /// [`COND_GUARD`] to the resulting configuration -- an ill-conditioned
    /// Fisher matrix makes the Occam term meaningless, so it is not weighed
    /// against anything.
    ///
    /// Fails **closed**: a non-positive-definite Fisher matrix on either side
    /// returns `-inf`. An ill-posed larger model must never be accepted, and an
    /// ill-posed smaller model is not evidence in favour of the larger one.
    #[allow(clippy::too_many_arguments)]
    pub fn log_bf_add(
        &mut self,
        i_before: f64,
        i_after: f64,
        f_before: &[f64],
        f_after: &[f64],
        n_before: usize,
        n_after: usize,
        sum_a_before: f64,
        sum_a_after: f64,
        k_before: usize,
        prior: Prior,
        before: Option<LogDet>,
    ) -> (f64, f64) {
        let (ld_b, ok_b) = match before {
            Some(v) => v,
            None => self.logdet(f_before, n_before),
        };
        let (ld_a, cond_a, ok_a) = self.logdet_cond(f_after, n_after);
        if !(ok_b && ok_a) {
            return (f64::NEG_INFINITY, cond_a);
        }
        (
            log_bf_add_from_logdet(
                i_before,
                i_after,
                ld_b,
                ld_a,
                sum_a_before,
                sum_a_after,
                k_before,
                prior,
            ),
            cond_a,
        )
    }

    /// `log BF` for `K_full -> K_full - 1` (a removal). Positive favours the
    /// smaller model.
    ///
    /// This is the **exact** negation of the corresponding addition. That
    /// antisymmetry is what stops a greedy search cycling between a move and
    /// its opposite, so it is enforced structurally -- by negating the very
    /// same expression -- rather than by writing the removal formula out again
    /// and hoping. The `03_evidence` fixture asserts the residual is `0.0`
    /// exactly, not merely small.
    #[allow(clippy::too_many_arguments)]
    pub fn log_bf_remove(
        &mut self,
        i_full: f64,
        i_reduced: f64,
        f_full: &[f64],
        f_reduced: &[f64],
        n_full: usize,
        n_reduced: usize,
        sum_a_full: f64,
        sum_a_reduced: f64,
        k_full: usize,
        prior: Prior,
        full: Option<LogDet>,
    ) -> f64 {
        let (ld_full, ok_full) = match full {
            Some(v) => v,
            None => self.logdet(f_full, n_full),
        };
        let (ld_reduced, ok_reduced) = self.logdet(f_reduced, n_reduced);
        if !ok_reduced {
            return f64::NEG_INFINITY; // cannot trust the reduced model; keep what we have
        }
        if !ok_full {
            return f64::INFINITY; // the current model is degenerate; removal is right
        }
        -log_bf_add_from_logdet(
            i_reduced,
            i_full,
            ld_reduced,
            ld_full,
            sum_a_reduced,
            sum_a_full,
            k_full - 1,
            prior,
        )
    }
}

/// Difference of the full `Exp(1/A_s)` prior over ALL emitters when one is
/// added.
///
/// Deliberately not "the prior of the added emitter". For a birth the total
/// flux genuinely increases and this reduces to the familiar
/// `-log(A_s) - A_new/A_s`. For a **split** the parent's flux is merely
/// redistributed between two children, the flux difference vanishes, and the
/// correct cost is just the `-log(A_s)` of carrying one more amplitude.
/// Charging a split `A_child/A_s` as well over-penalizes -- by ~1 nat at
/// typical bead flux -- precisely the move that resolves close pairs.
fn d_log_amplitude_prior(sum_a_before: f64, sum_a_after: f64, a_s: f64) -> f64 {
    -a_s.ln() - (sum_a_after - sum_a_before) / a_s
}

/// The Bayes factor itself, once both log-determinants are in hand.
///
/// Shared by the add and remove paths so their antisymmetry is exact by
/// construction. The term order matches `evidence.py`'s summation exactly.
#[allow(clippy::too_many_arguments)]
fn log_bf_add_from_logdet(
    i_before: f64,
    i_after: f64,
    ld_before: f64,
    ld_after: f64,
    sum_a_before: f64,
    sum_a_after: f64,
    k_before: usize,
    prior: Prior,
) -> f64 {
    (i_before - i_after) + prior.lam.ln() - ((k_before + 1) as f64).ln()
        + d_log_amplitude_prior(sum_a_before, sum_a_after, prior.a_s)
        + 1.5 * (2.0 * std::f64::consts::PI).ln()
        - 0.5 * (ld_after - ld_before)
}
