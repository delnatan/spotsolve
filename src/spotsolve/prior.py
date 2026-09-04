"""The priors the Bayes factor charges, and the fit is penalized by.

Two of them, and they are used differently. The FLUX prior is read only by
`evidence`, when it prices one more emitter. The WIDTH prior is read by
`evidence` AND by `lmga`, which maximizes the posterior under it -- see
`WidthPrior`, and README section 8b for why a fit and an evidence that
disagree about what a width costs will disagree about what exists.

The flux prior's shape is a known, open defect
----------------------------------------------
`evidence.log_bf_add` charges every new emitter a prior cost. That cost comes
from an `Exp(1/A_s)` amplitude prior with `A_s` set to the mean detected flux
-- a shape that was assumed and never checked. It is wrong in a way that
decides detections.

An exponential density has its MODE AT ZERO. For a SPLIT, where total flux is
conserved, it therefore asserts that one emitter of flux `2A` is `e^log(A_s)`
times more likely a priori than two of flux `A`: at bead brightness, a factor
of about 1800. For a near-monodisperse population -- beads, or one species of
fluorophore -- that is backwards. The single bright emitter is the thing that
should be nearly impossible.

Measured, decomposing the split decision for an isolated equal pair at 900 e-
each on a 4 e- background (sigma = 1.2):

    d/sigma   data term   Occam    prior     log BF   verdict
      0.50       +3.21    +3.11    -9.36      -3.04   rejected
      0.75       +4.80    +3.04    -9.36      -1.53   rejected
      1.00      +12.81    +2.11    -9.37      +5.55   accepted

The prior demands 11.4 nats (`-log lam` = -3.9, `-log A_s` = -7.5) and the data
supplies 3.2 -- which is exactly what the Fisher information says is available
there, so the data was never the problem. Changing ONLY the amplitude prior,
split acceptance against true separation:

    amplitude prior            0.4s   0.5s   0.6s   0.75s   1.0s
    Exp(1/A_s)   (the old one)   0%     0%     8%     25%    83%
    tight, mode away from zero  58%    83%    83%     67%    92%

with recovered positions at |err| 0.24-0.39 px, consistent with the CRLB. The
result was IDENTICAL for N(900,150) and N(900,300), so what matters is the
prior's shape, not a tuned width -- which is the argument for estimating it
rather than choosing it.

What this module does NOT contain, and why
-----------------------------------------
The fix implied by that table -- a prior with its mode away from zero,
estimated from the frame's own detections by Kiefer-Wolfowitz NPMLE -- was
built, measured, and removed on 2026-09-03 without ever having been wired in.
`detect` has always run the exponential. See README section 13; the short
version is that it was 200 lines of EM in the module the Rust port needs,
reachable only by a caller constructing it by hand, and INCORRECT if they had:
a curved flux prior makes `Lambda` non-zero on the amplitude block, and
`evidence` still omits it there (README section 15).

So the shape defect above is real, unfixed, and recorded. Re-adding the
estimator means adding the `Lambda` term with it.

`detect(prior=...)` still accepts any `FluxPrior`, so a caller who has a
population model can supply one.
"""

import numpy as np
from scipy.special import gammaln

__all__ = ["FluxPrior", "ExponentialFlux",
           "WidthPrior", "UniformWidth", "FocusMixtureWidth",
           "FOCUS_WIDTH_GAMMA"]




class FluxPrior:
    """A density over one emitter's TOTAL FLUX, in photoelectrons.

    The only thing `evidence` asks of a prior is `logpdf`. Keeping that the
    whole interface is deliberate: it is what lets the exponential remain
    available as an exact reproduction of the old behaviour, so a change in
    detections can be attributed to the prior's SHAPE and not to the rewrite.
    """

    def logpdf(self, a):
        raise NotImplementedError


class ExponentialFlux(FluxPrior):
    """`Exp(1/A_s)` -- the historical prior, kept as the fallback and baseline.

    With this prior the generalized difference in `evidence` reduces ALGEBRAICALLY
    to the expression it replaced:

        sum_after log g - sum_before log g
          = [-(K+1) log A_s - sumA_after/A_s] - [-K log A_s - sumA_before/A_s]
          = -log A_s - (sumA_after - sumA_before)/A_s

    which is `_d_log_amplitude_prior`'s old body exactly. So the refactor is
    verifiable rather than a leap: with this prior the pipeline is bit-identical.
    """

    def __init__(self, A_s):
        self.A_s = max(float(A_s), 1e-12)

    def logpdf(self, a):
        a = np.asarray(a, dtype=float)
        return np.where(a > 0.0, -np.log(self.A_s) - a / self.A_s, -np.inf)

    def __repr__(self):
        return f"ExponentialFlux(A_s={self.A_s:.1f})"


# ---------------------------------------------------------------------------
# The width prior: which objects the model is allowed to contain, and at what
# price
# ---------------------------------------------------------------------------
#
# `SIGMA_SLACK` used to do two jobs at once, and conflating them is what made
# the detector tile defocused sources. It defined
#
#   1. the MODEL SPACE  -- what widths a fit may represent, and
#   2. the REPORTING BAND -- what counts as a detection.
#
# The model space has to cover every photon on the sensor or the light it
# cannot represent is tiled (see `core.SIGMA_SLACK`). The reporting band is a
# downstream contract about which fitted objects the caller is handed. Clipping
# (1) to (2) is what manufactured the tiles.
#
# Separating them is not free, though, and the naive separation is worse than
# the conflation. Under ONE uniform prior over the enlarged space every emitter
# pays for the enlargement: widening [0.95, 2.0] to [0.95, 8.0] taxes every
# in-focus add `log(7.05/1.05) = 1.9` nats, making the detector less sensitive
# everywhere in order to accommodate defocus. And it does not even buy what it
# was meant to: under a flat prior, going from a 2.0 to a 3.2 bound charges a
# WIDE fit only `log(2.25/1.05) = 0.76` nats in total, so a wide emitter
# swallowing a genuine close neighbour is charged essentially nothing for it --
# which is exactly the monotone residual degradation `core.SIGMA_SLACK`'s sweep
# measured past 2.0 and read as an argument for the hard bound.
#
# So the bound at 2.0 was a hard cutoff standing in for a prior term that
# belongs in the formula -- structurally the same defect README section 15
# records for `COND_GUARD` standing in for the prior's curvature.
#
# The term is a MIXTURE. The field is a superposition of two Poisson processes:
# in-focus emitters at rate `lam_focus` with widths in the reporting band, and
# defocused nuisance objects at rate `lam_wide` with widths above it. The
# likelihood is identical for both -- they are both Gaussians, fitted by the
# same code. Only the width prior's support and the rate differ. An in-focus
# add then pays exactly what it paid before the model space was enlarged, and a
# wide one pays against its own, rarer rate and its own, much broader width
# prior.


class WidthPrior:
    """The count-and-width part of the configuration prior, area-free.

    Three methods, and the split between them is the whole design:

    `logpdf(sigma)` / `curvature(sigma)` are the SMOOTH per-emitter width
    density and its negative log-curvature. They are what the FIT sees: at
    fixed N and fixed class assignment they are the only part of the prior that
    depends on sigma, so they are exactly what a MAP fit should be penalized
    by. `core._fit_any` hands them to `lmga` and the fit maximizes the
    posterior rather than the likelihood.

    `log_config(sigmas)` adds the per-class Poisson counts on top and is what
    the EVIDENCE differences between two configurations. The rate terms are a
    prior over COUNTS; they are constant in a fixed-N fit and jump only when an
    emitter changes class, which is a model-selection event and not something
    the optimizer should be walking across.

    That is deliberately a whole-configuration quantity rather than "the cost of
    the added emitter", for the same reason `_d_log_amplitude_prior` is: under a
    mixture the incumbents' terms do NOT cancel, because a neighbour that widens
    across the class boundary in the joint refit moves between the two processes
    and changes both their counts.

    Positions and area cancel exactly, as they do in `evidence`'s header: the
    Poisson count prior contributes `(lam_c * Area)^K_c / K_c!` per class and
    the uniform positions contribute `Area^-K_c`, leaving `lam_c^K_c / K_c!`.
    """

    is_flat = False
    """True when the density is constant over the model space, so a MAP fit
    under it IS the ML fit. `core._fit_any` then passes no penalty at all
    rather than a constant one -- not as an optimization, but because
    `lmga` compares `I_cur - I_trial` against `tol_obj = 1e-8` and a constant
    added to both sides of that subtraction costs low-order bits of it. On a
    39x39 field the two agree; on a dense frame, thousands of fits later, they
    do not, and a baseline that moves is not a baseline."""

    def logpdf(self, sigma):
        raise NotImplementedError

    def curvature(self, sigma):
        """`-d^2/dsigma^2 log pdf`, clamped at zero. The Laplace evidence wants
        the Hessian of the log POSTERIOR, `F + Lambda`; this is Lambda's width
        block, and `lmga` adds it to the Gauss-Newton matrix so the fit's own
        `F` already carries it. Clamped because a heavy-tailed prior is not
        log-concave in its tails and an indefinite Hessian is not a curvature
        the optimizer can use."""
        raise NotImplementedError

    def log_config(self, sigmas):
        raise NotImplementedError


class UniformWidth(WidthPrior):
    """One class, `sigma ~ U[lo, hi]` -- the single-band model, kept exact.

    Flat, so `curvature` is zero and the MAP fit reduces to the ML fit: passing
    this reproduces the pre-mixture pipeline bit-for-bit, in the fit as well as
    in the evidence. It is the `ExponentialFlux` of width priors, and a change
    in detections is attributable to the mixture's SHAPE rather than to the
    machinery around it.
    """

    is_flat = True

    def __init__(self, lam, lo, hi):
        self.lam = max(float(lam), 1e-12)
        self.lo = float(lo)
        self.hi = float(hi)

    def logpdf(self, sigma):
        return np.full(np.shape(sigma), -np.log(self.hi - self.lo))

    def curvature(self, sigma):
        return np.zeros(np.shape(sigma))

    def log_config(self, sigmas):
        k = len(np.asarray(sigmas).ravel())
        if k == 0:
            return 0.0
        return float(k * (np.log(self.lam) - np.log(self.hi - self.lo))
                     - gammaln(k + 1))

    def __repr__(self):
        return (f"UniformWidth(lam={self.lam:.4g}, "
                f"[{self.lo:.2f}, {self.hi:.2f}])")


FOCUS_WIDTH_GAMMA = 0.20
# Half-width, in units of the PSF sigma, of the width prior's Cauchy core.
#
# Why a prior on the width at all, when the model space's bounds were supposed
# to be the statement about the optics. Because a FLAT prior over the in-focus
# band asserts that an in-focus emitter is as likely to sit at 2.0 sigma as at
# 1.0, and the data says otherwise by a mile: on a field with no defocus at all
# the fitted widths come back with median 1.00 and p90 1.04. Asserting that
# much ignorance is not free, and it is not paid where you would expect --
# measured on `crlb.py`, an ISOLATED emitter loses nothing (sigma is orthogonal
# to position by symmetry, 0.0527 -> 0.0557 px), while a pair at 1-2 sigma
# loses 67% of its localization (0.120 -> 0.200 px) even in the ORACLE arm,
# where N is fixed at truth and nothing is searched. That is a fit-level
# degeneracy: two narrow emitters at 1.5 sigma and one wide one plus a faint
# one describe nearly the same pixels, and with a flat prior nothing chooses.
#
# Why CAUCHY rather than a normal. The tail has to stay cheap. A genuinely
# defocused source at 1.5-2.0x must remain representable -- that is what the
# model space is FOR -- while a fit drifting to 1.3x on noise must be pulled
# back. A Gaussian with a scale tight enough to do the second forbids the
# first; a Cauchy shrinks the core hard and charges the tail almost nothing,
# which is exactly the asymmetry the problem has.
#
# The core is SYMMETRIC about the PSF width, not one-sided. Nothing images
# narrower than the PSF, so a fit below `sigma0` is as much a broken fit as one
# far above it is a defocused object, and both should be pulled back.
#
# At 0.20, conditioned on the half above `sigma0`, the prior's median is 1.17x
# and its quartile 1.35x; the confocal simulation's own pushforward -- z
# uniform over the in-focus slab, through sigma(z) -- has median 1.12x and
# quartile 1.54x. So this is a weakly-informative stand-in for that
# pushforward, in the right neighbourhood and deliberately not tuned to it,
# since the pushforward is only computable where the optics are known.
#
# SWEPT, and the useful result is that it barely matters. `bench_sim.py`,
# moderate arm (5 emitters/um^2), 6 frames, everything else at the default:
#
#   gamma   recall   med err    RMSE   rsd z   |z|>3   tiles
#    0.10    93.1%     0.076   0.228    1.28    6.9%    2.83
#    0.15    92.6%     0.074   0.223    1.26    6.5%    3.17
#    0.20    92.6%     0.076   0.220    1.25    7.0%    2.33
#    0.30    91.7%     0.076   0.221    1.32    7.5%    2.33
#    0.50    90.3%     0.079   0.204    1.32    8.1%    3.17
#
# Anything in 0.10-0.30 is within a point of recall and a few thousandths of a
# pixel; 0.50 is clearly too loose. 0.20 is kept for the best pull spread and
# the fewest tiles, but the honest statement is that the SHAPE is what matters
# and the scale does not, which is the same thing this module's header found
# for the flux prior's bandwidth. Do not tune it against a score.
#
# What it costs a genuinely wide fit, which is the number that matters for the
# tail: 0.22 nats at 1.1x, 0.69 at 1.2x, 1.61 at 1.4x, 2.30 at 1.6x and 3.26 at
# 2.0x. A defocused source has data worth far more than three nats; a fit
# drifting on noise does not.


class FocusMixtureWidth(WidthPrior):
    """Two classes split at `mid`: in-focus below it, defocused nuisance above.

    `lo <= sigma <= mid` is an in-focus emitter, reported as a detection.
    `mid < sigma <= hi` is a defocused object: modelled to the end, so that its
    flux is explained and stops refilling FIND's candidate list, and then
    handed back as a nuisance record rather than as a detection.

    The width DENSITY is one continuous Cauchy centred at the PSF width over
    the whole model space -- not one density per class. A per-class density
    would jump at the boundary, and the fit walks across that boundary
    continuously; only the COUNT prior is allowed to be discontinuous there,
    because changing class is a model-selection event. See `WidthPrior`.

    Neither rate is a knob. Both are re-estimated from the frame each round,
    exactly as `lam` and `A_s` already are -- so "wide objects are rarer than
    in-focus ones" is read off the data instead of asserted. `core` floors each
    count at one, which says a frame may a priori hold one object of either
    class; at a typical 4096 usable px and 22 in-focus detections that opens a
    wide birth at about 3.1 nats against, and the floor stops mattering the
    moment one is found.

    The bounds `lo` and `hi` are the fit's own box constraints, so a sigma
    outside them is not a configuration the optimizer can return; they are used
    here only to normalize the density over the space it actually lives on.
    """

    def __init__(self, lam_focus, lam_wide, lo, mid, hi, sigma0,
                 gamma=FOCUS_WIDTH_GAMMA):
        if not lo < mid < hi:
            raise ValueError("expected lo < mid < hi")
        self.lam_focus = max(float(lam_focus), 1e-12)
        self.lam_wide = max(float(lam_wide), 1e-12)
        self.lo, self.mid, self.hi = float(lo), float(mid), float(hi)
        self.sigma0 = float(sigma0)
        self.scale = max(float(gamma) * self.sigma0, 1e-9)
        # Normalizer of the Cauchy core over [lo, hi], so `logpdf` is a density
        # on the model space and comparable across configurations.
        self._logZ = float(np.log(self.scale) + np.log(
            np.arctan((self.hi - self.sigma0) / self.scale)
            - np.arctan((self.lo - self.sigma0) / self.scale)))

    def _u(self, sigma):
        return (np.asarray(sigma, dtype=float) - self.sigma0) / self.scale

    def in_focus(self, sigmas):
        """The class label, which is a FUNCTION of the fitted width and not a
        parameter of its own -- which is why nothing downstream has to carry
        one."""
        return np.asarray(sigmas, dtype=float) <= self.mid

    def logpdf(self, sigma):
        return -np.log1p(self._u(sigma) ** 2) - self._logZ

    def curvature(self, sigma):
        u = self._u(sigma)
        return np.maximum(2.0 * (1.0 - u ** 2) / (1.0 + u ** 2) ** 2, 0.0) \
            / self.scale ** 2

    def log_config(self, sigmas):
        sigmas = np.asarray(sigmas, dtype=float).ravel()
        if len(sigmas) == 0:
            return 0.0
        k_f = int(np.count_nonzero(self.in_focus(sigmas)))
        k_w = int(len(sigmas) - k_f)
        return float(
            k_f * np.log(self.lam_focus) - gammaln(k_f + 1)
            + k_w * np.log(self.lam_wide) - gammaln(k_w + 1)
            + np.sum(self.logpdf(sigmas)))

    def __repr__(self):
        return (f"FocusMixtureWidth(lam {self.lam_focus:.4g}/"
                f"{self.lam_wide:.4g}, [{self.lo:.2f}, {self.mid:.2f}, "
                f"{self.hi:.2f}], gamma {self.scale / self.sigma0:.2f})")

