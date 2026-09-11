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

so the whole gain dependence of the box search's decision -- keep an
emitter iff the I-divergence falls by more than ADD_NATS -- is the 1/g on
the data term: dI_ADU / g > ADD_NATS. Rescaling the data when g changes
cannot change a fitted configuration; its only effect is to move that
threshold. The gain is a property of the CAMERA, not of the field: measure
it once and pass it in. `estimate_gain` is the fallback.
"""

import numpy as np
import scipy.ndimage as ndi

from .. import psf

__all__ = ["SEED_ALPHA", "seed_threshold", "robust_background",
           "emitter_free_mask", "GAIN_FRAC", "estimate_gain", "robust_spread",
           "render_model"]


SEED_ALPHA = 0.05
# The seeder's family-wise false-seed rate per frame, and the only free number
# in `seed_threshold`. It can be this loose because FIND IS A SEEDER, NOT A
# DECISION RULE: every candidate still has to lower its box's deviance by
# ADD_NATS, and every placement inside a box passes this same test, so a
# spurious seed costs runtime rather than a detection. What an over-tight seed
# costs is unrecoverable -- a box never forms around light FIND did not seed.
#
# Measured under `detect` (retired 2026-09-11), whose ADD step played the
# same role: the historical cut at alpha ~ 2e-8 missed faint emitters that a
# looser seed found (bench amp arms, peak SNR 1.1: recall 23.1% against
# 65.8% at a 10x lower threshold), while at alpha = 0.05 the decision rule,
# not the seed, did the rejecting. The derived cut is a LOWER bound on the
# seeder's conservatism, because the LoG response is standard normal only
# where the model is right; on frames with model mismatch the effective test
# count is larger than the pixel geometry says.

def seed_threshold(shape, sigma, alpha=SEED_ALPHA):
    """FIND's seed cut in sd of the LoG null, derived rather than carried.

    `find_candidates` normalizes its response by `core.log_kernel_l2(sigma)`,
    so under the null it is standard normal per pixel; and it keeps only local
    maxima in a `2*ceil(sigma)+1` window, so that window sets the number of
    INDEPENDENT tests. A family-wise rate `alpha` over `n` of them is the
    Bonferroni cut `Phi^-1(1 - alpha/n)`.

    It therefore scales with the frame and with the PSF, which a constant
    cannot: on 64^2 at sigma 0.818 it is 3.70, on 512^2 at sigma 1.45 it is
    4.43.
    """
    h, w = shape[:2]
    win = 2 * int(np.ceil(float(sigma))) + 1
    n = max(float(h) * float(w) / float(win) ** 2, 1.0)
    return float(_norm_isf(min(max(float(alpha), 1e-12), 0.999) / n))


def _norm_isf(p):
    """Standard-normal upper-tail inverse. `scipy.special.ndtri` is already a
    dependency through `scipy`, and this keeps the Rust port's contract to one
    special function it must match."""
    from scipy.special import ndtri
    return -ndtri(p)




def _widths(sigma, n):
    """`sigma` as one width per emitter, whether a scalar or an array was
    given. The pipeline's fits may leave every emitter at its own width, and
    the two places that stamp an emitter's SUPPORT -- the background mask and
    the model render -- have to use that width or they mask and render the
    wrong footprint for a defocused source."""
    a = np.asarray(sigma, dtype=float)
    return np.full(n, float(a)) if a.ndim == 0 else a.ravel()


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
    pos = np.atleast_2d(np.asarray(positions, dtype=float))
    radii = radius_factor * _widths(sigma, len(pos))
    for (cy, cx), r in zip(pos, radii):
        r2 = r * r
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
    spread, and a gain read off the residual cannot tell that inflation from
    genuine camera gain.
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


GAIN_FRAC = 0.2
# The share of dimmest pixels `estimate_gain` reads; see its docstring.


def estimate_gain(raw, offset, frac=GAIN_FRAC):
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
    Poisson data term shrinks by the same factor, costing recall (0.71 at
    density 0.055 against 0.90 at 0.034, measured under `detect`). A
    residual-based correction cannot rescue it: the bias is already baked
    into the units the residual is measured in. Pass a measured `gain` for
    crowded fields.

    SECOND KNOWN LIMIT -- it does not transfer between fields. On the two
    real bead frames it returns 4.23 and 2.94, though both were taken on the
    same camera and the gain is a camera constant. The frac sweep tells them
    apart: a field with genuine emitter-free background shows a plateau at
    small frac, a crowded one does not. The fitted bead amplitudes agree
    across the two fields at 4.23 (median 939 vs 933 e-), which they could
    not if the gain were off by 1.4x on one of them. Prefer a measured gain
    over this estimator.

    This runs BEFORE any model search: a gain estimated from the fitted
    residual arrives too late, after the search has run with the data term
    inflated by the wrong gain. `boxsearch::estimate_gain` is its Rust port.
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

    This is a REPORTING statistic: lack of fit inflates it exactly as a wrong
    gain would, so it cannot tell the two apart.
    """
    lo, hi = np.percentile(np.asarray(r), [15.9, 84.1])
    return float(0.5 * (hi - lo))


def render_model(positions, amplitudes, sigma, shape, background, truncate=4.0):
    """Global model image: background plus every emitter's PSF, summed.

    `sigma` is a scalar or one width per emitter.

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
    widths = _widths(sigma, len(positions))
    for (cy, cx), A, s_k in zip(positions, amplitudes, widths):
        rad = int(np.ceil(truncate * s_k))
        y0 = max(0, int(np.floor(cy)) - rad)
        y1 = min(H, int(np.ceil(cy)) + rad + 1)
        x0 = max(0, int(np.floor(cx)) - rad)
        x1 = min(W, int(np.ceil(cx)) + rad + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        th = psf.pack(0.0, [A], [cy], [cx])
        m[y0:y1, x0:x1] += psf.model(th, yy * 1.0, xx * 1.0, s_k)
    return m
