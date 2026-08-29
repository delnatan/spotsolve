"""Aguet's point-source detector: a calibrated seeder for round 0.

At every pixel, fit `A * g + c` -- one Gaussian of KNOWN sigma at the pixel
centre, plus a local constant background -- by least squares, and test
`A` against the local noise level. Both parameters are linear, so the fit has
a closed form in terms of three convolutions and costs `O(1)` per pixel
regardless of how many emitters there are.

    Aguet, Antonescu, Mettlen, Schmid & Danuser, "Advances in analysis of low
    signal-to-noise images link dynamin and AP2". Ported from
    `aguet_matlab/pointSourceDetection.m` (DanuserLab u-track3D), cross-checked
    against sfwloc's Rust port of the same routine.

Why this and not the LoG seeder
-------------------------------
`core.find_candidates` tests a LoG response against a threshold in units of
the response's own null standard deviation, which requires knowing that null --
i.e. it assumes `Var = m`, the Poisson law, with `m` the current model. In
round 0 there IS no fitted model: `detect` seeds `bmap` with a global 10th
percentile, so the assumed null is wrong wherever the background is not flat,
and the noise scale is inherited from a gain estimate rather than measured.

This detector assumes neither. It estimates the background LOCALLY (`c_est`,
one nuisance parameter per site) and the noise scale LOCALLY (`sigma_res`,
from the window's own residuals), so its threshold is an `alpha`, a
probability, rather than a number whose meaning depends on a variance model
being right. That is what makes it the right instrument before a model exists,
and it is why it degrades gracefully when the gain is mis-estimated -- the
failure mode `beads_60x_still*.tif` actually exhibits.

What it does NOT fix
--------------------
Nothing here resolves two emitters inside ~1.5 sigma: `A * g` is a ONE-emitter
model, and its fitted amplitude over an unresolved pair has a single maximum
between them. That is SPLIT's job and stays SPLIT's job. Nor does it help
against background structure at PSF scale -- `c` is a local CONSTANT, so a
bump the size of a PSF is not absorbed by it and is detected as a source.
Measured on a background correlation-length sweep, PSF-scale structure costs
false positives (0 -> 13), not recall.

The test
--------
With `n` pixels in the window, `RSS` the residual sum of squares of the
two-parameter fit, and `kLevel = Phi^-1(1 - alpha/2)`:

    sigma_A     = sqrt(RSS/(n-3) * C00)      SE of the amplitude
    sigma_res   = sqrt(RSS/(n-1))            SE of one residual
    SE_sigma_c  = sigma_res/sqrt(2(n-1)) * kLevel
    T           = (A_est - sigma_res*kLevel) / sqrt((sigma_A^2+SE_sigma_c^2)/n)
    p           = t_sf(T, df2)               Welch-Satterthwaite df2

Note the null being tested. It is **not** `A = 0`: the numerator subtracts
`sigma_res * kLevel`, so the hypothesis is "`A` exceeds the noise floor by at
least `kLevel` sigma". That is a minimum-detectable-amplitude formulation, and
it is why `alpha` appears twice -- once setting the floor, once as the p-value
cut.
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.stats import norm, t as student_t

from . import psf

__all__ = ["detect_spots", "significance_map", "local_regression",
           "log_local_maxima", "ALPHA"]

ALPHA = 0.05
# Aguet's own default. Unlike a filter-response cut this is a probability, so
# it means the same thing at every sigma, every gain and every background --
# which is the entire point of using this detector at round 0. It is a
# PER-PIXEL level, not a per-frame error rate: at alpha=0.05 over 512^2
# pixels the significance mask alone would admit a great many pixels, and it
# is the AND with a LoG local maximum that makes the pair selective.


def _kernels(sigma, truncate=4.0):
    """`(g, w, n, gsum, g2sum, c00)` for the significance window.

    `g` is UNNORMALIZED `exp(-x^2/2 sigma^2)`, matching the MATLAB. The 2-D
    kernel is `g' * g`, so its sums factor: `sum(g2d) = sum(g)^2` and
    `sum(g2d^2) = sum(g^2)^2`.

    Note `w = ceil(truncate*sigma)` here, which is NOT the rounding rule
    scipy's Gaussian kernels use (`int(truncate*sigma + 0.5)`). They agree at
    sigma=1.2 and diverge whenever the fractional part is below 0.5 -- at
    sigma=1.3, 6 against 5. Keep them distinct; this one sizes the regression
    window and the other sizes the LoG.
    """
    w = int(np.ceil(truncate * sigma))
    x = np.arange(-w, w + 1, dtype=float)
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    n = float((2 * w + 1) ** 2)
    gsum = float(g.sum()) ** 2
    g2sum = float((g ** 2).sum()) ** 2
    # C00 of inv(J'J) for J = [g.ravel(), ones]; closed form for a 2x2.
    c00 = 1.0 / (g2sum - gsum ** 2 / n)
    return g, w, n, gsum, g2sum, c00


def _moments(img, sigma, truncate=4.0):
    """The three convolutions the closed-form fit needs: `fg`, `fu`, `fu2`."""
    g, w, n, gsum, g2sum, c00 = _kernels(sigma, truncate)
    ones = np.ones(2 * w + 1)
    # 'reflect' is scipy's name for MATLAB's 'symmetric' padding: the edge
    # pixel IS repeated. g and ones are symmetric, so correlate == convolve.
    def sep(a, k):
        return ndi.correlate1d(ndi.correlate1d(a, k, axis=0, mode="reflect"),
                               k, axis=1, mode="reflect")
    fg = sep(img, g)
    fu = sep(img, ones)
    fu2 = sep(img * img, ones)
    return fg, fu, fu2, n, gsum, g2sum, c00


def local_regression(img, sigma, truncate=4.0):
    """Per-pixel `(A_est, c_est)`: amplitude and local background.

    `A_est` is the PEAK amplitude of the unnormalized Gaussian, not total
    flux. Divide by `psf.peak_factor(sigma)` for the flux `spotsolve` uses.
    """
    img = np.asarray(img, float)
    fg, fu, _, n, gsum, g2sum, _ = _moments(img, sigma, truncate)
    a_est = (fg - gsum * fu / n) / (g2sum - gsum ** 2 / n)
    c_est = (fu - a_est * gsum) / n
    return a_est, c_est


def significance_map(img, sigma, alpha=ALPHA, truncate=4.0):
    """Pixels where the fitted amplitude significantly exceeds the noise."""
    img = np.asarray(img, float)
    fg, fu, fu2, n, gsum, g2sum, c00 = _moments(img, sigma, truncate)
    a_est = (fg - gsum * fu / n) / (g2sum - gsum ** 2 / n)
    c_est = (fu - a_est * gsum) / n

    rss = (a_est ** 2 * g2sum - 2.0 * a_est * (fg - c_est * gsum)
           + (fu2 - 2.0 * c_est * fu + n * c_est ** 2))
    # Cancellation in the line above can push a genuinely-zero RSS negative.
    np.maximum(rss, 0.0, out=rss)

    k_level = norm.ppf(1.0 - alpha / 2.0)
    sigma_a2 = rss / (n - 3.0) * c00
    sigma_res = np.sqrt(rss / (n - 1.0))
    se_sigma_c = sigma_res / np.sqrt(2.0 * (n - 1.0)) * k_level

    sa2, sc2 = sigma_a2, se_sigma_c ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        df2 = (n - 1.0) * (sa2 + sc2) ** 2 / (sa2 ** 2 + sc2 ** 2)
        scomb = np.sqrt((sa2 + sc2) / n)
        T = (a_est - sigma_res * k_level) / scomb
        pval = student_t.sf(T, df2)
    # A degenerate window (RSS == 0, so scomb == 0) is not evidence of a
    # source; NaN must not pass the test.
    return np.nan_to_num(pval, nan=1.0) < alpha


def log_local_maxima(img, sigma, truncate=4.0):
    """Local maxima of the LoG response, on a `2*ceil(sigma)+1` footprint.

    The border strip of that width is cleared: a reflect-padded curvature
    estimate at a true edge is an artifact of the padding, and the original
    detector discards it rather than trusting it.
    """
    img = np.asarray(img, float)
    log_f = -ndi.gaussian_laplace(img, sigma, mode="reflect")
    dom = 2 * int(np.ceil(sigma)) + 1
    lm = log_f == ndi.maximum_filter(log_f, size=dom, mode="reflect")
    lm[:dom, :] = lm[-dom:, :] = False
    lm[:, :dom] = lm[:, -dom:] = False
    return lm


def detect_spots(img, sigma, alpha=ALPHA, truncate=4.0):
    """`(cand, amp, strength)`, the same contract as `core.find_candidates`.

    `cand` is `(M, 2)` float `(y, x)` pixel coordinates, `amp` is TOTAL FLUX
    in the same units as `img`, and both are sorted by descending amplitude so
    the caller's brightest-first loop is unchanged.
    """
    img = np.asarray(img, float)
    sig = significance_map(img, sigma, alpha, truncate)
    lm = log_local_maxima(img, sigma, truncate)
    ys, xs = np.nonzero(sig & lm)
    if len(ys) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    a_est, _ = local_regression(img, sigma, truncate)
    peak = a_est[ys, xs]
    cand = np.stack([ys, xs], axis=1).astype(float)
    flux = np.maximum(peak, 1e-2) / psf.peak_factor(sigma)
    order = np.argsort(-peak)
    return cand[order], flux[order], peak[order]
