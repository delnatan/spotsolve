"""Camera calibration and global image-level quantities, in photoelectrons.

Everything in the pipeline downstream of this module works in PHOTOELECTRONS,
d_e = (adu - offset) / g. The Poisson weighting W = 1/m used by the fitter is
only valid there.

What the gain actually is
-------------------------
`g` is NOT a preprocessing scale factor. It is the DISPERSION parameter of a
quasi-Poisson likelihood, and it enters the inference in exactly one place:
the decision about how many emitters there are. Three exact identities
(verified numerically to machine precision, see the module tests):

  * the I-divergence is homogeneous,      I(d/g, m/g) = I(d, m) / g
  * therefore the FIT does not depend on g at all: fitted positions are
    invariant and amplitudes scale exactly as 1/g. Measured over
    g = 1 .. 50: positions agreed to 7e-7 px, amplitudes to 7e-7 relative.
  * the Fisher determinant picks up log|F| = (1 - K) log g + log|F_ADU|

so the whole gain dependence of a birth/death decision collapses to

    log BF(g) = dI_ADU / g  +  1.5 * log g  +  const

(the 1/g on the data evidence, +0.5 log g from the Laplace volume, +log g
from -log A_s). The constant was reproducible to 8 decimals over a 200x
range of g.

Two consequences that matter operationally:

  1. Rescaling d_e, amplitudes, background and A_s when g changes -- which is
     what a gain-refinement step does -- cannot change the fitted
     configuration. It is a change of units. Its ONLY effect on the answer is
     to move the detection threshold, by the law above.

  2. `gain_ratio_from_residual` is a Pearson dispersion estimator, and
     dispersion estimators are inflated by LACK OF FIT. So when the model is
     wrong it raises g, which raises the detection threshold, which suppresses
     detections exactly where the model is already failing. That is why
     pinning a measured gain beats refining it on the real bead frames, and
     why a climbing g should be read as a symptom of unmodelled structure
     rather than as a gain measurement.

The gain is a property of the CAMERA, not of the field. Measure it once and
pass it in; `refine_gain` is off by default for the reasons above.
"""

import numpy as np
import scipy.ndimage as ndi

from . import psf

__all__ = ["LOG_SEED_Z", "LOG_SEED_THRESHOLD", "robust_background",
           "emitter_free_mask",
           "estimate_gain",
           "robust_spread", "lower_half_spread", "gain_ratio_from_residual",
           "render_model"]


LOG_SEED_Z = 6.496256820534083
# FIND's seed threshold, in STANDARD DEVIATIONS of the LoG response under the
# null -- not in the filter's raw output units, which is what the old
# `LOG_SEED_THRESHOLD = 1.5` was, and which was a latent bug.
#
# `find_candidates` normalizes its LoG by `core.log_kernel_l2(sigma)`, so the
# response has unit variance under a correct model at EVERY sigma. Before that
# normalization the response's null sd was the kernel's own L2 norm, which
# scales as sigma^-3:
#
#     sigma   ||w||_2   the old 1.5, in sd    false peaks/px
#      0.8     0.8365          1.79              3.8e-2
#      1.0     0.4010          3.72              5.8e-4
#      1.2     0.2309          6.48              0
#      2.0     0.0499         29.97              0
#      3.0     0.0148        100.51              0
#
# So the same literal constant was a 1.8-sigma cut at sigma=0.8 and a
# 100-sigma cut at sigma=3.0 -- a factor of 56 in significance, driven by a
# parameter nobody thought of as touching the threshold. Any instrument whose
# PSF was not 1.2 px got a silently different detector.
#
# The value here is exactly `1.5 / ||w||_2(sigma=1.2)`, so it reproduces the
# historical behaviour bit-for-bit on the sigma this pipeline was tuned at.
# That is the ONLY reason it is this number and not a round one: it makes the
# normalization a provable no-op. It is NOT a defensible operating point --
# it is a ~6.5 sigma cut on pixels and ~10 sigma on peak heights, which costs
# 43 points of recall at peak SNR 1.1 (recall 23.1% at this value, 65.8% at a
# 10x lower one). See `bench.py`'s AMP_ARMS. Lowering it is a separate,
# measured change.
#
# Peak heights are NOT distributed like pixels -- local maxima are a biased
# sample, with sd 0.1508 against the pixel sd 0.2316 at sigma=1.2 -- so this
# number is a sigma count on PIXELS. A threshold calibrated on peaks needs the
# peak null, which is not analytic and wants a parametric bootstrap.

LOG_SEED_THRESHOLD = LOG_SEED_Z
# Deprecated alias. The unit changed under this name, so prefer `LOG_SEED_Z`;
# anything still reading this and expecting raw filter units is wrong by
# 1/||w||_2.




def emitter_free_mask(shape, positions, sigma, radius_factor=3.0):
    """Pixels no emitter's support reaches, as an (H, W) boolean.

    Written as a per-emitter stamp over the disc's bounding box rather than a
    sweep of the whole frame, because the sweep is O(N*H*W): at 512x512 with
    12667 emitters it was 41% of the frame's runtime, and it was being paid
    TWICE per round -- once here for the scalar fallback and once in
    `core.background_map` for the surface. Share the result; do not recompute
    it. `backend.emitter_free_mask` is the same predicate in Rust.
    """
    H, W = shape
    free = np.ones((H, W), dtype=bool)
    if positions is None or len(positions) == 0:
        return free
    r = radius_factor * sigma
    r2 = r * r
    for cy, cx in np.atleast_2d(np.asarray(positions, dtype=float)):
        y0 = max(int(np.ceil(cy - r)), 0)
        y1 = min(int(np.floor(cy + r)) + 1, H)
        x0 = max(int(np.ceil(cx - r)), 0)
        x1 = min(int(np.floor(cx + r)) + 1, W)
        if y0 >= y1 or x0 >= x1:
            continue
        dy = np.arange(y0, y1, dtype=float) - cy
        dx = np.arange(x0, x1, dtype=float) - cx
        free[y0:y1, x0:x1] &= (dy[:, None] ** 2 + dx[None, :] ** 2) > r2
    return free


def robust_background(d, positions=None, sigma=None, radius_factor=3.0,
                      free=None):
    """Median of the pixels no emitter reaches; a low quantile before we have
    an emitter list to mask with.

    Neither half of this can be skipped. The image MEDIAN is not a background
    estimate for a crowded field -- on beads_60x_still.tif it sits at 99 ADU
    against a true background near 12-19. But a fixed low QUANTILE is not one
    either: on a sparse field, where most pixels genuinely are background, the
    10th percentile lands well below the true level (measured: 9.7 against a
    true 20 on an emitter-free frame). Masking out each emitter's support and
    taking the median of what remains is right in both regimes, and reduces to
    the plain median when there are no emitters.

    `free` accepts a precomputed `emitter_free_mask`, which is how the round
    loop avoids stamping the same mask twice.

    A biased background here does not stay contained: it inflates the residual
    spread, and the gain feedback in `detect()` cannot tell that inflation from
    genuine camera gain, so it silently absorbs the bias into g_eff.
    """
    d = np.asarray(d)
    if positions is None or len(positions) == 0:
        return float(np.median(d))
    if sigma is None:
        return float(np.percentile(d, 10.0))
    H, W = d.shape
    # `free` may be supplied by a caller that needs the same mask anyway. It
    # must have been built with THIS `radius_factor`; nothing here can check
    # that, and a mismatch would silently change the estimate.
    if free is None:
        free = emitter_free_mask(d.shape, positions, sigma, radius_factor)
    if free.sum() >= max(16, 0.02 * H * W):
        return float(np.median(d[free]))
    return float(np.percentile(d, 10.0))


def estimate_gain(raw, offset, frac=0.2):
    """Photon-transfer gain estimate from the dimmest pixels, in ADU per
    photoelectron. Needs no emitter model and no camera calibration.

    For emitter-free pixels the Poisson relation is Var = gain*(mean-offset),
    so the ratio of a high-pass variance to the mean intensity IS the gain.
    Two details make it work in practice:

      * Which pixels count as background is decided on a SMOOTHED copy.
        Selecting the dimmest RAW pixels selects on their own noise, biasing
        the mean down and the gain up.
      * `frac` is small (the dimmest 20%). Measured across synthetic fields
        the estimate is essentially unbiased there -- 1.00, 2.03, 4.69, 10.02
        against true 1, 2, 4.7, 10 -- while frac=0.4 and 0.6 drift high as
        density rises because PSF tails leak into the selection. On
        beads_60x_still.tif frac=0.2 gives 4.2 against 9.9 and 14.3.

    KNOWN LIMIT -- this fails at high emitter density. The leak that pushes
    frac=0.4 high at moderate density reaches frac=0.2 eventually: on a
    bead-matched field at density 0.055 emitters/px^2 this returns 12.5
    against a true 4.23, because by then even the dimmest fifth of the image
    is sitting on PSF tails. Everything downstream inherits it -- d_e is
    scaled by 1/g_eff, so the background reads 1.26 instead of 4.0 and every
    Poisson data term in the Bayes factor shrinks by the same factor, costing
    recall (0.71 at density 0.055 against 0.90 at 0.034). This is a property
    of the estimator, not of the model search: `refine_gain` does not rescue
    it either, since a residual-based correction cannot see a bias that is
    already baked into the units the residual is measured in. Pass a measured
    `gain` for crowded fields.

    SECOND KNOWN LIMIT -- it does not transfer between fields. On the two
    real bead frames it returns 4.23 and 2.94, though both were taken on the
    same camera and the gain is a camera constant. The frac sweep tells them
    apart: a field with genuine emitter-free background shows a plateau at
    small frac, a crowded one does not. Two independent checks favour 4.23 on
    BOTH frames -- `gain_ratio_from_residual` reads 0.93-0.98 on the second
    field with 4.23 pinned, and the fitted bead amplitudes agree across fields
    at that gain (median 939 vs 933 e-), which they could not if the gain were
    off by 1.4x on one of them. Prefer a measured gain over this estimator.

    This runs BEFORE any model search. Estimating gain only from the fitted
    residual (as an earlier version did) is too late: the search then runs
    its first passes with the data term inflated by the true gain, and on a
    dense field it over-splits catastrophically before the correction ever
    lands.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.shape[1] < 5:
        return 1.0
    hp = (raw[:, :-2] - 2.0 * raw[:, 1:-1] + raw[:, 2:]) / np.sqrt(6.0)
    mu = raw[:, 1:-1] - offset
    sel = ndi.uniform_filter(raw[:, 1:-1], size=3) <= np.quantile(
        ndi.uniform_filter(raw[:, 1:-1], size=3), frac
    )
    if sel.sum() < 32:
        return 1.0
    m = float(np.mean(mu[sel]))
    if m <= 1e-6:
        return 1.0
    return float(np.clip(np.var(hp[sel]) / m, 0.05, 200.0))


def robust_spread(r):
    """IQR-based standard deviation, immune to the handful of near-zero-model
    pixels that dominate np.var on an offset-corrected image.

    This is a REPORTING statistic. Do not drive the gain feedback with it --
    see `gain_ratio_from_residual` for why the symmetric spread cannot
    separate a wrong gain from an incomplete model.
    """
    lo, hi = np.percentile(np.asarray(r), [15.9, 84.1])
    return float(0.5 * (hi - lo))


def lower_half_spread(r):
    """Half-spread measured from the LOWER tail only: median - p15.9.

    An emitter the model has missed pushes the normalized residual UP and
    never down, so the upper tail carries model error while the lower tail
    carries noise alone.
    """
    r = np.asarray(r)
    return float(np.median(r) - np.percentile(r, 15.9))


def gain_ratio_from_residual(d_e, model, seed=0):
    """Multiplicative correction to g_eff, read off the residual.

    If the true gain is g_t and the loop is currently using g_e, then
    d_e = counts * g_t/g_e, so Var(d_e) = m * g_t/g_e and the normalized
    residual (d_e - m)/sqrt(m) has spread sqrt(g_t/g_e). Squaring it and
    multiplying into g_eff therefore converges in ONE step -- provided the
    spread that goes in measures noise and nothing else.

    Two things are needed to make that true, and the previous version of
    this feedback had neither:

    * The spread must come from the lower tail. Measured on a bead-matched
      field as emitters are deleted from an otherwise perfect model, the
      symmetric 15.9/84.1 half-range runs 1.00 -> 5.28 between a complete
      model and one missing 35% of its emitters, while median - p15.9 moves
      only 0.96 -> 1.31. Both still track a genuinely wrong gain (1.37 at
      g_eff = 0.5x true, 0.68 at 2x), which is the only thing this feedback
      is entitled to respond to.

    * It must be normalized by the same statistic on a PARAMETRIC BOOTSTRAP
      draw, Poisson(model). The normalized Poisson residual is positively
      skewed at these count rates, so median - p15.9 reads ~0.96 rather than
      1.0 under a perfect model and a perfect gain; feeding that back
      unnormalized would shrink g_eff by ~7% every pass forever. The
      reference draw carries the same skewness at the same count level, so
      the ratio is 1.0 by construction when the model is right.

    Left unguarded, this feedback is what wrecks the real bead frame: the
    17 beads excluded by `border_margin` leave a residual whose symmetric
    spread is 1.38, which took a gain that `estimate_gain` had measured at
    4.227 (true 4.23) to 8.06 -- and at twice the true gain every Poisson
    data term in the Bayes factor halves, so clean isolated beads stop
    clearing the evidence threshold everywhere in the image.
    """
    m = np.maximum(np.asarray(model, dtype=float), 1e-6)
    s_obs = lower_half_spread((np.asarray(d_e) - model) / np.sqrt(m))
    ref = np.random.default_rng(seed).poisson(m).astype(float)
    s_ref = lower_half_spread((ref - m) / np.sqrt(m))
    if not np.isfinite(s_obs) or not np.isfinite(s_ref) or s_ref <= 0:
        return 1.0
    return s_obs / s_ref


# ------------------------------------------------------------------ model


def render_model(positions, amplitudes, sigma, shape, background, truncate=4.0):
    """Global model image: background plus every emitter's PSF, summed.

    Contributions from overlapping emitters ADD, which is what the physics
    says and what every patch fit assumes locally. (An earlier version
    stitched per-patch models by AVERAGING them where bounding boxes
    overlapped, and filled patch-free pixels with the image median; that
    produced block artifacts, a systematically high model, and a residual
    with median -3 in normalized units.)

    Each emitter is rendered only within +/- truncate*sigma of its centre,
    so cost is linear in emitter count rather than N*H*W.
    """
    H, W = shape
    m = np.full((H, W), float(background), dtype=float)
    if len(positions) == 0:
        return m
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    amplitudes = np.asarray(amplitudes, dtype=float).ravel()
    rad = int(np.ceil(truncate * sigma))
    for (cy, cx), A in zip(positions, amplitudes):
        y0 = max(0, int(np.floor(cy)) - rad)
        y1 = min(H, int(np.ceil(cy)) + rad + 1)
        x0 = max(0, int(np.floor(cx)) - rad)
        x1 = min(W, int(np.ceil(cx)) + rad + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        th = psf.pack(0.0, [A], [cy], [cx])
        m[y0:y1, x0:x1] += psf.model(th, yy * 1.0, xx * 1.0, sigma)
    return m
