"""Bounded Levenberg-Marquardt with Coleman-Li affine scaling.

Objective is the Poisson I-divergence:

    I(d, m) = sum_i [ d_i*log(d_i/m_i) - (d_i - m_i) ]      (0*log(0) := 0)

which is minimized by Fisher scoring: W = diag(1/m) held fixed within an
iteration, so the normal-equations matrix F = J^T W J is the *expected*
Fisher information (exact for Poisson's canonical link -- Fisher scoring
coincides with IRLS here; see psf.py verification notes).

Geodesic acceleration (Transtrum & Sethna) was implemented here and then
removed: measured on this problem it changed the converged objective by
under 1e-13 on ordinary fits and by 0.007 on the hardest close-pair fits at
0.5 sigma separation (below the identifiability limit anyway), while
costing 133 model/Jacobian evaluations per fit instead of 18 -- 94 ms
against 0.7 ms. The curvature term needs a nested jvp per inner lambda
trial, which dominates everything else on 81-pixel patches. The
implementation is not kept in this repository; the reference is Transtrum &
Sethna if a harder problem ever wants it.

Coleman-Li scaling keeps iterates strictly inside [lower, upper] without
active-set logic: define v_i = (upper_i - theta_i) if grad_i < 0 else
(theta_i - lower_i), scale by D_CL = diag(sqrt(|v|)), and add the
Jacobian-of-v term to the scaled normal equations before solving.

STRICTLY inside is not a stylistic preference, it is the precondition the
method runs on: v_i is a denominator, and a parameter resting exactly on its
bound sets it to zero. `_to_interior` enforces it on entry and after every
accepted step, and the note there records what happens when it is not
enforced -- which is what this optimizer did until it was added.
"""

import numpy as np
from scipy.linalg import get_lapack_funcs

from . import psf
from .structs import FitResult

__all__ = ["fit", "i_divergence"]


_POTRS, = get_lapack_funcs(("potrs",), (np.empty(0, dtype=np.float64),))

# The kernels behind cho_factor/cho_solve, called without their wrappers.
#
# This loop factorizes a ~25x25 matrix around 700k times per frame, and at that
# size the wrappers cost more than the arithmetic: `_asarray_validated`, dtype
# normalization, alignment checks, the batch-dispatch decorator and a
# `get_lapack_funcs` lookup on EVERY cho_solve. Measured on a representative
# p=25 system, 5.21 us per factor+solve through scipy against 1.28 us here.
#
# `_batched_linalg._cholesky` is what `cho_factor` itself dispatches to in this
# scipy, so the factor is bit-identical -- verified equal on 2000 randomized
# systems. Raw dpotrf is NOT: it disagrees with the batched kernel in the last
# ulp, which is enough to move a converged fit in the 11th digit. If the
# private module ever disappears, the fallback below is the public path again.
try:                                            # scipy >= 1.15
    from scipy.linalg import _batched_linalg

    def _chol_solve(A, rhs):
        """(delta, ok) for A delta = rhs, A symmetric positive definite."""
        c, err = _batched_linalg._cholesky(A, False, False, False)
        if err:
            return None, False
        delta, info = _POTRS(c, rhs, lower=False)
        return delta, info == 0
except ImportError:                             # pragma: no cover
    from scipy.linalg import cho_factor, cho_solve

    def _chol_solve(A, rhs):
        try:
            c, low = cho_factor(A, check_finite=False)
        except np.linalg.LinAlgError:
            return None, False
        return cho_solve((c, low), rhs, check_finite=False), True


def i_divergence(d, m):
    d = np.asarray(d)
    m = np.asarray(m)
    # np.where evaluates both branches eagerly, so d*log(d/m) is computed
    # (and warns) even at masked-out d<=0 entries unless the log's own
    # argument is also guarded; d<=0 is expected in real (offset-subtracted)
    # data wherever read noise dips a pixel below the fitted baseline.
    safe_ratio = np.where(d > 0, d, 1.0) / m
    term = np.where(d > 0, d * np.log(safe_ratio), 0.0)
    return float(np.sum(term - (d - m)))


_INTERIOR_FRAC = 1e-10


def _to_interior(theta, lower, upper, frac=_INTERIOR_FRAC):
    """Pull `theta` strictly inside the box, by a fraction of each bound's own
    range.

    Coleman-Li requires a STRICTLY interior iterate, and the whole method
    silently collapses without one. The distance-to-bound v_i is what scales
    every coordinate; a parameter sitting exactly ON its bound has v_i = 0, so

      * `_coleman_li_scale` floors it at sqrt(1e-12) = 1e-6 and the damping
        term |grad_i|/v_i becomes ~1e12, which is fine -- that coordinate is
        supposed to freeze; but
      * the fraction-to-boundary rule in `fit` computes room = (bound_i -
        theta_i)/delta_i = 0 for it, and that scale multiplies the WHOLE step.
        One stuck coordinate therefore shrinks every other coordinate's step to
        the 1e-8 floor as well.

    The step then buys ~1e-11 nats, its gain ratio rho reads ~2e-8 -- which is
    measuring the clip, not the quality of the quadratic model -- so the step
    is rejected, lambda is multiplied by nu, and nu doubles. Lambda ratchets
    away and the fit spends its whole iteration budget taking micro-steps.

    Measured on FOV1 before this was added, with `theta_trial` produced by
    np.clip onto the bounds: 68.6% of all LM iterations began with at least one
    parameter exactly on a bound (always a POSITION, railed at a box edge),
    46.5% of inner trials were clipped, 92% of those collapsed to the 1e-8
    floor, and 73.2% of all fits exhausted max_iter=100 with lambda at ~1e4 and
    max|grad| still ~13. Keeping the iterate interior instead: fits that reach
    a convergence test rise from 24% to 64%, the frame solves 2.5x faster, and
    N stops depending on the lattice phase (measured on the superseded
    box-sequential solver, whose count was phase-dependent because of it).

    `frac` is RELATIVE to each parameter's own range, because the ranges are
    not comparable -- a position is bounded over ~20 px and an amplitude over
    ~1e4 e-. An absolute margin would be a different constraint for each.
    """
    margin = frac * np.maximum(upper - lower, 1e-12)
    return np.minimum(np.maximum(theta, lower + margin), upper - margin)


def _pen_zero_val(theta):
    return 0.0


def _pen_zero_vec(theta):
    return 0.0


def _coleman_li_scale(theta, grad, lower, upper):
    """sqrt of the distance to whichever bound the step is heading toward.

    v_i is the distance to the LOWER bound where the step wants to decrease
    theta_i (grad >= 0) and to the UPPER bound otherwise. sqrt(v) is what the
    scaling matrix D needs; the two extra diagonals the caller adds to the
    normal equations are both |grad|/v and lam/v, i.e. both divided by this
    value SQUARED (see fit()). Each is positive semi-definite by construction.
    """
    below = grad >= 0
    v = np.where(below, theta - lower, upper - theta)
    return np.sqrt(np.maximum(v, 1e-12))


def fit(
    theta0,
    yy,
    xx,
    sigma,
    d,
    lower,
    upper,
    halo=0.0,
    max_iter=100,
    tol_obj=1e-8,
    tol_grad=1e-6,
    tol_step=1e-10,
    lambda0=1e-2,
    free_sigma=False,
    penalty=None,
):
    """Fit theta by bounded Fisher-scoring LM (MAP when `penalty` is given).

    If `free_sigma` is True, `theta` includes sigma as its last entry and
    `sigma` (the scalar argument) is ignored; the model used is
    psf.model_free_sigma. If `free_sigma == "per_emitter"`, theta carries one
    sigma per emitter using psf.pack_var_sigma.

    `penalty` and what it makes this
    --------------------------------
    With `penalty=None` this maximizes the likelihood. Given one -- any object
    exposing `value(theta)`, `grad(theta)` and `hess_diag(theta)`, all in nats
    and in theta's own coordinates -- it maximizes the POSTERIOR instead: the
    penalty's value joins the objective, its gradient joins `grad`, and its
    curvature joins the Gauss-Newton matrix.

    Two consequences that callers depend on, and they pull in opposite
    directions, so both are deliberate:

    `I` is returned WITHOUT the penalty -- the data term alone. Everything
    downstream differences `I` against another fit's and then adds the prior
    itself (`evidence`, via its width and flux priors), so returning a
    penalized `I` would charge the prior twice.

    `F` is returned WITH it. The Laplace evidence wants the Hessian of the log
    POSTERIOR, `F + Lambda`, not the likelihood Fisher -- README section 15
    recorded that omission, and this is where it is repaired: a curved prior's
    curvature belongs in the matrix whose log-determinant becomes the Occam
    factor. With a flat prior `hess_diag` is zero and `F` is what it always
    was.

    Convergence: `tol_obj` in NATS
    ------------------------------
    The primary stopping test is on the predicted decrease in I, which for
    the LM step delta solving A delta = -grad is -0.5 * grad . delta. That is
    the only criterion in units the caller actually cares about: everything
    downstream compares I-divergences on the scale of a log Bayes factor, so
    "this fit cannot improve I by more than 1e-8 nats" is a statement about
    the decision, not about the parameterization.

    `tol_grad` and `tol_step` are kept as backstops but neither can be the
    primary test. Both are ABSOLUTE, and the natural scale of this problem is
    set by the fluxes, which run to ~2000 e-; a gradient of 1e-6 and a step of
    1e-10 are near float64 noise at that scale. Measured on a bead-matched
    39x39 field before this was added, 34.5% of the 9400 patch fits in a
    single `detect()` run exhausted max_iter=100 without satisfying either
    one, and only 62.7% reported convergence.

    That is not merely slow, it is BIASED. A proposal fit (BIRTH, SPLIT)
    starts further from its optimum than the incumbent it is compared
    against, so it is the proposal whose I is left too high -- and the Bayes
    factor, which is a difference of I-divergences, systematically
    under-credits exactly the moves that add an emitter.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    theta = _to_interior(np.array(theta0, dtype=float), lower, upper)
    p = theta.shape[0]
    d_flat = np.asarray(d, dtype=float).reshape(-1)

    lam = lambda0
    nu = 2.0        # Nielsen's lambda-growth factor; see the gain-ratio note below

    # The grid never changes within a fit, so extract the separable axes once
    # instead of on each of the ~160 model evaluations a fit makes.
    ay, ax_ = psf.axes(yy, xx)

    def eval_model_jac(th):
        if free_sigma == "per_emitter":
            m = psf.model_var_sigma(th, yy, xx, halo)
            J = psf.jac_var_sigma(th, yy, xx, halo)
        elif free_sigma:
            m = psf.model_free_sigma(th, yy, xx, halo)
            J = psf.jac_free_sigma(th, yy, xx, halo)
        else:
            m, J = psf.model_and_jac_ax(th, ay, ax_, sigma, halo)
        return m.reshape(-1), J.reshape(-1, p)

    # `d` is fixed for the whole fit, so the two data-only pieces of the
    # I-divergence -- the d > 0 mask and the guarded numerator -- are loop
    # invariants. This is `i_divergence(d_flat, m)` with those hoisted out;
    # the arithmetic on the surviving entries is unchanged.
    d_pos = d_flat > 0
    d_safe = np.where(d_pos, d_flat, 1.0)

    def idiv(m):
        term = np.where(d_pos, d_safe * np.log(d_safe / m), 0.0)
        return float(np.sum(term - (d_flat - m)))

    # The penalty is optional and diagonal; `_zeros` keeps the hot loop free
    # of `if penalty is None` branches without allocating when there is none.
    if penalty is None:
        pen_val = _pen_zero_val
        pen_grad = pen_diag = _pen_zero_vec
    else:
        pen_val, pen_grad, pen_diag = (penalty.value, penalty.grad,
                                       penalty.hess_diag)

    m, J = eval_model_jac(theta)
    m = np.maximum(m, 1e-9)
    # `I_cur` is the PENALIZED objective while the loop runs -- that is what LM
    # must decrease monotonically -- and the data term alone is recovered at
    # the end for the caller.
    I_cur = idiv(m) + pen_val(theta)

    converged = False
    stalled = False
    it = 0
    for it in range(1, max_iter + 1):
        W = 1.0 / m
        grad = J.T @ (W * (m - d_flat)) + pen_grad(theta)   # d(objective)/dtheta
        F = J.T @ (W[:, None] * J)
        F.reshape(-1)[:: p + 1] += pen_diag(theta)

        if np.max(np.abs(grad)) < tol_grad:
            converged = True
            break

        s = _coleman_li_scale(theta, grad, lower, upper)
        Dinv = 1.0 / s

        # Both extra diagonals are fixed for this outer iteration -- only the
        # scalar `lam` in front of the second one changes as the inner loop
        # grows it. Building them once here rather than per lambda trial.
        jac_of_v_term = np.abs(grad) / (s * s)
        dinv2 = Dinv ** 2

        step_accepted = False
        step_norm = 0.0
        pred_dec = np.inf
        for _ in range(30):  # inner loop: grow lambda until step accepted
            # Coleman-Li form:
            #   (F + diag(|grad|/s^2) + lam*diag(1/s^2)) delta = -grad
            # The scaled-space system is (D F D + diag(|grad|) + lam I) shat
            # = -D grad with D = diag(s), s = sqrt(v). Mapping it back to the
            # unscaled step delta = D shat sends BOTH extra diagonals through
            # D^-1 (.) D^-1, i.e. both pick up 1/s^2 -- not 1/s for one of
            # them and 1/s^2 for the other, which is what this used to do.
            # The mismatched version under-damps every parameter that is
            # approaching a bound (at v = 0.01 it applies 10|g| where the
            # correct term is 100|g|). Measured over 200 randomized 1-3
            # emitter fits: the consistent form reaches a lower converged
            # I-divergence 6 times to 1 with 193 ties, is better by 3.5 nats
            # on average -- a large error on the scale a log Bayes factor is
            # decided on -- and gets there in 10.9 iterations against 20.4.
            # The term must stay positive semi-definite; a signed version
            # subtracts curvature wherever grad < 0 and can make the matrix
            # indefinite.
            # Written as an in-place add on a view of A's diagonal rather than
            # F + np.diag(u) + lam*np.diag(v), which allocates two dense p x p
            # matrices whose off-diagonals are known to be zero. The two adds
            # stay separate so the summation order -- (F_ii + u_i) + lam*v_i --
            # is the one the dense form produced.
            A = F.copy()
            diag_A = A.reshape(-1)[:: p + 1]
            diag_A += jac_of_v_term
            diag_A += lam * dinv2

            delta, ok = _chol_solve(A, -grad)
            if not ok:               # not positive definite (or non-finite)
                lam *= 10.0
                continue

            # Predicted decrease in I for this step, in nats. A delta = -grad,
            # so the quadratic model improves by -0.5 * grad . delta. Measured
            # BEFORE the bound-clipping below, so it reflects the step the
            # model actually proposed.
            pred_dec = -0.5 * float(grad @ delta)

            # scale step to stay strictly interior (0.995 of distance to bound)
            trial = theta + delta
            over_upper = trial > upper
            over_lower = trial < lower
            if np.any(over_upper) or np.any(over_lower):
                scale = 1.0
                if np.any(over_upper):
                    room = (upper[over_upper] - theta[over_upper]) / delta[over_upper]
                    scale = min(scale, 0.995 * np.min(room))
                if np.any(over_lower):
                    room = (lower[over_lower] - theta[over_lower]) / delta[over_lower]
                    scale = min(scale, 0.995 * np.min(room))
                scale = max(scale, 1e-8)
                delta = delta * scale

            # NOT np.clip onto the bounds: that parks a parameter exactly on
            # one, which destroys the strict interiority the scaling above
            # depends on. See `_to_interior`.
            theta_trial = _to_interior(theta + delta, lower, upper)
            m_trial, J_trial = eval_model_jac(theta_trial)
            m_trial = np.maximum(m_trial, 1e-9)
            I_trial = idiv(m_trial) + pen_val(theta_trial)

            actual_dec = I_cur - I_trial
            # LM gain ratio: how much of the promised improvement was real.
            # lambda MUST be driven by this and not by the sign of the
            # improvement alone. Accepting any decrease and halving lambda
            # for it -- what this used to do -- lets lambda collapse to its
            # floor while the quadratic model is worthless, and then nothing
            # damps the near-null directions of F. Traced on a real patch
            # (K=7, 15x14, one emitter parked at A_MIN so its position block
            # of F reads 5.7e-12 and cond(F) = 1.5e20): every iteration
            # predicted a decrease of 5.2e4 nats, delivered 1.05e-3, halved
            # lambda anyway, and took the identical 2.3e-5 step again. It
            # crawled for 3000+ iterations and still finished 364 nats above
            # the optimum. With the gain ratio in charge, rho = 2e-8 raises
            # lambda instead, which regularizes exactly that direction.
            rho = actual_dec / pred_dec if pred_dec > 0 else -1.0

            if I_trial < I_cur and rho > 1e-4:
                theta, m, J = theta_trial, m_trial, J_trial
                step_norm = np.linalg.norm(delta)
                I_cur = I_trial
                # Nielsen (1999): a smooth decrease that is aggressive for a
                # trustworthy step and gentle for a marginal one.
                lam = max(lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), 1e-12)
                nu = 2.0
                step_accepted = True
                pred_dec = min(pred_dec, actual_dec)
                break
            else:
                lam *= nu
                nu *= 2.0
                if lam > 1e12:
                    break  # give up on this outer iteration; try again next it

        if not step_accepted:
            stalled = True   # lambda saturated; NOT the same as converged
            break

        if pred_dec < tol_obj:
            # Nothing left to gain on the scale a Bayes factor is decided on.
            converged = True
            break

        if step_norm < tol_step:
            converged = True
            break

    W = 1.0 / np.maximum(m, 1e-9)
    F_final = J.T @ (W[:, None] * J)
    F_final.reshape(-1)[:: p + 1] += pen_diag(theta)

    return FitResult(
        theta=theta,
        I=idiv(np.maximum(m, 1e-9)),
        F=F_final,
        n_iter=it,
        converged=converged,
        stalled=stalled,
    )
