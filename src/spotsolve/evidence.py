"""Laplace-approximated Bayes factors for changes in emitter count.

Why a Bayes factor and not an F-test / likelihood-ratio test
-----------------------------------------------------------
The LRT for "is there one more emitter" is not chi2(3). Under H0 the extra
emitter sits on the BOUNDARY of the parameter space (A -> 0), and exactly
there its position is UNIDENTIFIABLE -- the likelihood is flat in two of
the three added directions. Both regularity conditions for Wilks' theorem
fail.

The practical consequence is worse than a fixed offset. The statistic is a
maximum over the larger model's parameter space, so a more thorough
optimizer finds a larger one: measured on synthetic single-emitter fields,
the false-positive rate at the nominal 5% chi2(3) cutoff moved from 3% to
10% purely by raising the restart count from 4 to 25. A test whose size
depends on how hard you search cannot be calibrated by choosing a
different critical value.

A Bayes factor has no such dependence. It integrates the added dimensions
against a proper prior rather than maximizing over them, so the Occam
factor charges for the search volume automatically.

The model
---------
Poisson point process on emitters, p(K) = Poisson(lam*Area); positions
uniform on the patch; amplitudes A_k ~ Exp(1/A_s); background
b ~ Uniform[0, b_max].

For a K -> K+1 move the Area factors cancel exactly: the prior odds
contribute lam*Area/(K+1) and the new emitter's position prior contributes
1/Area. What survives is area-free:

    log BF = (I_K - I_{K+1})                     data term
           + log(lam) - log(K+1)                 prior odds on count
           + d log pi_A                          amplitude prior difference
           + (3/2) log(2 pi)                     Laplace volume, 3 new params
           - (1/2)(log|F_{K+1}| - log|F_K|)      Occam factor

`b` appears with the same range on both sides of every comparison, so its
uniform prior cancels and is never needed here.

Both exponential and uniform priors have zero curvature, so the Hessian of
the negative log posterior is the Fisher information exactly: H = F.
"""

import numpy as np
from scipy.linalg import cho_factor, LinAlgError

__all__ = ["COND_GUARD", "logdet", "logdet_cond",
           "log_bf_add", "log_bf_remove"]


COND_GUARD = 1e3
# Threshold on the DIAGONALLY SCALED condition number. The raw cond(F) is
# useless as an absolute test because F mixes parameters with different
# units -- background (counts), amplitude (total flux), position (pixels).
# Measured at sigma=1.2: a pristine isolated emitter reads 8.4e6 raw but
# 1.6 scaled; a genuinely degenerate pair at 0.5 sigma reads 1.3e11 raw and
# 1.8e4 scaled. No fixed raw threshold separates them; the scaled form does,
# and is invariant to reparameterization.


def logdet(F):
    """(log|F|, ok) from one Cholesky factorization, with no condition number.

    Split out from `logdet_cond` because the condition number costs a full SVD
    and MOST CALLERS THROW IT AWAY. `log_bf_add` reads the cond of the `after`
    matrix only; `log_bf_remove` reads neither. Measured on a 39x39 frame under
    the legacy split/birth moves, that was ~2900 SVDs of a p ~ 25 matrix per
    frame computed for nothing.

    Keeping the two entry points separate rather than making the cond lazy is
    deliberate: `ok` must stay exactly the conjunction `logdet_cond` reported,
    so a caller that skips the cond still fails closed on a non-positive-
    definite F, and still fails closed on a non-positive diagonal -- which is
    checked here for that reason and not because anything below it is needed.
    """
    try:
        c, _ = cho_factor(F)
    except (LinAlgError, np.linalg.LinAlgError):
        return np.inf, False
    ld = 2.0 * float(np.sum(np.log(np.diag(c))))
    d = np.diag(F)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return ld, False
    return ld, True


def logdet_cond(F):
    """(log|F|, scaled condition number, ok). `ok` is False if F is not
    positive definite, which callers must treat as "this model is ill-posed"
    rather than as evidence for anything.

    Prefer `logdet` unless the condition number is actually consumed.

    Note what this deliberately does NOT test. An emitter whose amplitude has
    collapsed to the lower bound has a position block scaling as A^2, which
    can underflow to numerical zero and make log|F| meaningless -- but no
    test on F ALONE can separate that from a perfectly well-posed matrix
    expressed in badly chosen units, because the two produce the same
    signature (huge raw cond, small scaled cond). A relative-diagonal floor
    added here was reverted for exactly that reason: it rejected a valid F
    rescaled by diag(exp(U(-8,8))), which is the reparameterization
    invariance this function exists to provide.

    That pathology is therefore prevented at its source, where the flux scale
    is actually known: see `core._bounds`, which floors the amplitude
    relative to the window rather than at an absolute constant.
    """
    try:
        c, _ = cho_factor(F)
    except (LinAlgError, np.linalg.LinAlgError):
        return np.inf, np.inf, False
    logdet = 2.0 * float(np.sum(np.log(np.diag(c))))
    d = np.diag(F)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        return logdet, np.inf, False
    s = 1.0 / np.sqrt(d)
    try:
        cond = float(np.linalg.cond(F * s[:, None] * s[None, :]))
    except np.linalg.LinAlgError:
        cond = np.inf
    return logdet, cond, True


def _d_log_amplitude_prior(sumA_before, sumA_after, A_s):
    """Difference of the full Exp(1/A_s) prior over all emitters when one is
    added.

    This is deliberately NOT "the prior of the added emitter". For a BIRTH
    total flux genuinely increases and the expression reduces to the
    familiar -log(A_s) - A_new/A_s. For a SPLIT the parent's flux is merely
    redistributed between two children, the flux difference vanishes, and
    the correct cost is just the -log(A_s) of carrying one more amplitude.
    Charging a split A_child/A_s as well over-penalizes -- by ~1 nat at
    typical bead flux -- precisely the move that resolves close pairs.
    """
    return -np.log(A_s) - (sumA_after - sumA_before) / A_s


def _log_bf_add_from_logdet(I_before, I_after, ld_before, ld_after,
                            sumA_before, sumA_after, K_before, lam, A_s):
    """The Bayes factor itself, once both log-determinants are in hand.

    Split out so `log_bf_remove` can reuse it. Removal is the exact negation of
    addition, so it used to obtain that negation by CALLING `log_bf_add`, which
    re-factorized both Fisher matrices it had already factorized itself -- four
    Cholesky decompositions and four condition numbers per death or merge
    proposal where two suffice.
    """
    return float(
        (I_before - I_after)
        + np.log(lam)
        - np.log(K_before + 1)
        + _d_log_amplitude_prior(sumA_before, sumA_after, A_s)
        + 1.5 * np.log(2.0 * np.pi)
        - 0.5 * (ld_after - ld_before)
    )


def log_bf_add(I_before, I_after, F_before, F_after,
               sumA_before, sumA_after, K_before, lam, A_s, before=None):
    """log BF for K_before -> K_before+1 (an ADD; legacy BIRTH or SPLIT).

    Positive favours the larger model. Returns (log_bf, scaled_cond_after)
    so the caller can apply COND_GUARD to the resulting configuration.

    `before` is an optional precomputed `logdet(F_before)`. Every proposal in
    one search step is scored against the SAME incumbent, so without it the
    incumbent's Fisher matrix is refactorized once per proposal -- measured at
    44.5% of all `logdet_cond` calls on a 39x39 frame. Passing it in is exact,
    not an approximation: it is the same function of the same matrix.

    Fails CLOSED: a non-positive-definite Fisher matrix on either side
    returns -inf. An ill-posed larger model must never be accepted, and an
    ill-posed smaller model is not evidence in favour of the larger one.
    """
    ld_b, ok_b = logdet(F_before) if before is None else before
    ld_a, cond_a, ok_a = logdet_cond(F_after)
    if not (ok_b and ok_a):
        return -np.inf, cond_a
    return _log_bf_add_from_logdet(I_before, I_after, ld_b, ld_a,
                                   sumA_before, sumA_after, K_before,
                                   lam, A_s), cond_a


def log_bf_remove(I_full, I_reduced, F_full, F_reduced,
                  sumA_full, sumA_reduced, K_full, lam, A_s, full=None):
    """log BF for K_full -> K_full-1 (DEATH or MERGE).

    Positive favours the smaller model. This is the exact negation of the
    corresponding addition, so DEATH inverts ADD and MERGE inverts SPLIT.
    That antisymmetry is what stops a greedy search cycling between a move
    and its opposite.

    Neither condition number is read here, so both determinants come from
    `logdet`. `full` is the incumbent's precomputed one, as in `log_bf_add`.
    """
    ld_full, ok_full = logdet(F_full) if full is None else full
    ld_reduced, ok_reduced = logdet(F_reduced)
    if not ok_reduced:
        return -np.inf      # cannot trust the reduced model; keep what we have
    if not ok_full:
        return np.inf       # the current model is degenerate; removal is right
    return -_log_bf_add_from_logdet(
        I_reduced, I_full, ld_reduced, ld_full,
        sumA_reduced, sumA_full, K_full - 1, lam, A_s,
    )
