"""Priors for the Bayes factor, estimated from the ensemble rather than assumed.

Why this module exists
----------------------
`evidence.log_bf_add` charges every new emitter a prior cost. Until now that
cost came from an `Exp(1/A_s)` amplitude prior with `A_s` set to the mean
detected flux -- a shape that was assumed and never checked. It is wrong in a
way that decides detections.

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

The estimator
-------------
Kiefer-Wolfowitz NPMLE: with observations `y_i` carrying their own standard
errors `se_i`, find the latent distribution `G` maximizing

    sum_i log( integral phi(y_i; a, se_i) dG(a) )

over all distributions on a grid. This is CONCAVE in the grid weights, so it has
no local optima -- the property the rest of this pipeline's model selection does
not have. Solved here by EM, which needs no dependency beyond numpy. See
Koenker & Mizera (2014) for the convex-programming treatment and Jiang (2020)
for the heteroscedastic case this one is.

Deconvolution is the point. The observed spread of fitted amplitudes is the
latent spread CONVOLVED with per-emitter measurement error, and that error is
large for faint emitters. Thresholding or fitting a shape to the observed
histogram inherits the error; the NPMLE removes it.
"""

import numpy as np

__all__ = ["FluxPrior", "ExponentialFlux", "MixturePrior", "npmle",
           "fit_flux_prior", "NPMLE_GRID", "NPMLE_MIN_N"]


NPMLE_GRID = 96
# Grid points for the latent distribution. The NPMLE's solution is atomic with
# far fewer support points than this, so the grid only has to be fine enough
# not to quantize them; it is not a resolution parameter.

NPMLE_MIN_N = 40
# Detections below which the frame is not asked to estimate its own prior and
# `fit_flux_prior` falls back to the exponential. An NPMLE on a handful of
# points fits its own noise, and the failure mode is a spiky prior that makes
# confident nonsense of the Bayes factor.

NPMLE_MAX_ITER = 400
NPMLE_TOL = 1e-7


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


class MixturePrior(FluxPrior):
    """The NPMLE's atoms, smoothed into a density.

    The NPMLE returns a DISCRETE `G`, which cannot be a prior density here: any
    amplitude not sitting exactly on an atom would score `-inf` and veto the
    move. Smoothing with a Gaussian of width `bandwidth` is the minimal repair.

    `bandwidth` is not a tuning knob for the ANSWER -- the measurement in this
    module's header found identical split acceptance for prior widths differing
    by 2x -- but it must not be zero, and it should not be so large that it
    smears a genuinely tight population back out. The default ties it to the
    atoms' own spread.

    The density is renormalized over `a > 0`, since a flux cannot be negative
    and the smoothing kernel does not know that.
    """

    def __init__(self, atoms, weights, bandwidth, floor=1e-300):
        self.atoms = np.asarray(atoms, dtype=float)
        w = np.asarray(weights, dtype=float)
        self.weights = w / max(w.sum(), 1e-300)
        self.bandwidth = max(float(bandwidth), 1e-9)
        self.floor = float(floor)
        # Mass the kernel puts below zero, removed once here rather than per
        # call. `erf` via the complementary form keeps it stable in the tail.
        from scipy.special import erfc
        z = self.atoms / (self.bandwidth * np.sqrt(2.0))
        keep = 1.0 - 0.5 * erfc(z)          # P(N(atom, h) > 0)
        self._norm = float(np.sum(self.weights * keep))
        self._norm = max(self._norm, 1e-12)

    def logpdf(self, a):
        a = np.atleast_1d(np.asarray(a, dtype=float))
        d = (a[:, None] - self.atoms[None, :]) / self.bandwidth
        # log-sum-exp over atoms, so a faint emitter far from every atom does
        # not underflow to a hard -inf and silently veto a move.
        logk = (-0.5 * d ** 2
                - np.log(self.bandwidth * np.sqrt(2.0 * np.pi))
                + np.log(np.maximum(self.weights, self.floor))[None, :])
        m = logk.max(axis=1)
        out = m + np.log(np.sum(np.exp(logk - m[:, None]), axis=1))
        out = out - np.log(self._norm)
        return np.where(a > 0.0, out, -np.inf)

    def mean(self):
        return float(np.sum(self.weights * self.atoms))

    def __repr__(self):
        nz = int(np.sum(self.weights > 1e-4))
        return (f"MixturePrior({nz} atoms, mean={self.mean():.0f}, "
                f"h={self.bandwidth:.0f})")


def npmle(y, se, grid, max_iter=NPMLE_MAX_ITER, tol=NPMLE_TOL):
    """Kiefer-Wolfowitz NPMLE weights on `grid`, by EM. Returns `weights`.

    `y` are the observations and `se` their per-observation standard errors --
    heteroscedastic on purpose, because the whole reason the observed histogram
    misleads is that a faint emitter's amplitude is measured far worse than a
    bright one's.

    The objective is concave in the weights, so EM cannot get stuck in a local
    optimum; it only converges slowly, which is why the stopping rule is on the
    log-likelihood's increment rather than on the weights.
    """
    y = np.asarray(y, dtype=float).ravel()
    se = np.asarray(se, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float).ravel()
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, se = y[ok], se[ok]
    if len(y) == 0:
        return np.full(len(grid), 1.0 / len(grid))

    # Likelihood matrix L[i, j] = phi(y_i; grid_j, se_i). Formed once: it does
    # not depend on the weights, and it is the whole cost of an EM step.
    d = (y[:, None] - grid[None, :]) / se[:, None]
    logL = -0.5 * d ** 2 - np.log(se[:, None] * np.sqrt(2.0 * np.pi))
    logL -= logL.max(axis=1, keepdims=True)      # per-row scale, cancels in EM
    L = np.exp(logL)

    w = np.full(len(grid), 1.0 / len(grid))
    prev = -np.inf
    for _ in range(max_iter):
        num = L * w[None, :]
        den = num.sum(axis=1, keepdims=True)
        np.maximum(den, 1e-300, out=den)
        w = (num / den).mean(axis=0)
        ll = float(np.log(den).sum())
        if ll - prev < tol * max(abs(ll), 1.0):
            break
        prev = ll
    return w


def fit_flux_prior(amplitudes, se=None, A_s=None, n_grid=NPMLE_GRID,
                   min_n=NPMLE_MIN_N, bandwidth=None):
    """Estimate the flux prior from a frame's own detections.

    Falls back to `ExponentialFlux(A_s)` when there are too few detections to
    estimate a shape, or when no standard errors are available -- both are
    "we do not know the population", and asserting a tight prior on no evidence
    is a worse error than the loose one it replaces.

    `se` is the per-emitter `SE(A)`, i.e. `DetectResult.se[:, 0]`.
    """
    amplitudes = np.asarray(amplitudes, dtype=float).ravel()
    finite = np.isfinite(amplitudes) & (amplitudes > 0)
    fallback = ExponentialFlux(
        A_s if A_s is not None
        else (float(np.mean(amplitudes[finite])) if finite.any() else 1.0))

    if se is None or finite.sum() < min_n:
        return fallback
    se = np.asarray(se, dtype=float).ravel()
    good = finite & np.isfinite(se) & (se > 0)
    if good.sum() < min_n:
        return fallback

    a, s = amplitudes[good], se[good]
    hi = float(np.percentile(a, 99.5))
    hi = max(hi * 1.15, hi + 3.0 * float(np.median(s)))
    grid = np.linspace(max(hi * 1e-4, 1e-6), hi, n_grid)
    w = npmle(a, s, grid)

    if bandwidth is None:
        # Wide enough that the atomic solution is a density, narrow enough not
        # to smear a tight population: a fraction of the atoms' own spread,
        # floored at the grid pitch so it can never be degenerate.
        mean = float(np.sum(w * grid))
        var = float(np.sum(w * (grid - mean) ** 2))
        bandwidth = max(0.25 * np.sqrt(max(var, 0.0)),
                        float(grid[1] - grid[0]))
    return MixturePrior(grid, w, bandwidth)
