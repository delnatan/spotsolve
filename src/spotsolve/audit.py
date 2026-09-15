"""Residual audit: is anything left in the image that the model does not explain?

Every other check in this pipeline asks whether a decision was made correctly
given the data in front of it. This one asks the outside question -- after the
search has finished, does the residual still contain point-like structure? A
model that has missed an emitter leaves a POSITIVE PSF-shaped residual; one
that has piled two PSFs onto a single real emitter, or inflated a patch
background, leaves a NEGATIVE one. Both are invisible in a summary statistic
like the robust spread, which averages them away against the noise.

The statistic
-------------
At each candidate position the question "how much flux is unexplained here?"
has an exact answer. With residual r = d - m and Poisson variance Var(r_i) =
m_i (in units where variance equals the mean: ADU above the offset divided
by `Localizations.dispersion`), the maximum-likelihood amplitude of a
unit-flux PSF g
added at that position, and its variance, are

    A_hat = sum_i (r_i g_i / m_i) / sum_i (g_i^2 / m_i)
    Var(A_hat) = 1 / sum_i (g_i^2 / m_i)

so the score

    z = sum_i (r_i g_i / m_i) / sqrt( sum_i (g_i^2 / m_i) )

is A_hat in units of its own standard error.

Calibration
-----------
Under a correct model z is asymptotically standard normal at every position.
Measured on bead-matched fields (peaks 94-198 e-, background 4 e-) with the
TRUE model supplied, it comes out mean -0.003, sd 1.04 -- but with tails
heavier than Gaussian, because Poisson counts at these rates are skewed:
|z| > 5 occurs at 7.6e-5 per pixel against the normal 5.7e-7. So the
threshold is calibrated by measurement, not by reading off a normal table.
False findings per image with a correct model, after local-maximum picking:

    threshold   39x39, N=52          60x60, N=120
              pos      neg          pos      neg
      4.0     0.53     0.37         1.13     0.37
      4.5     0.27     0.17         0.52     0.15
      5.0     0.18     0.05         0.20     0.03
      5.5     0.10     0.00         0.10     0.02

The default of 5.0 costs about one spurious finding per five images, which
is the right trade for a diagnostic whose job is to be believed when it says
something is wrong.

This is the score (Rao) test for adding one emitter, which is why it is the
right screen to pair with the Bayes factor rather than a substitute for it:
it costs one correlation per image instead of a fit per candidate, and it is
evaluated at the CURRENT model, so it answers "did the search stop too early
or too eagerly" without re-running the search.

`z` is computed by correlation, so the whole map costs a few FFT-free
`scipy.ndimage` passes regardless of how many emitters there are.
"""

import numpy as np
import scipy.ndimage as ndi

from . import psf

__all__ = ["score_map", "residual_peaks", "audit_result", "format_report"]


def score_map(d_e, model, sigma, truncate=4.0):
    """Per-pixel score z for adding one PSF, as described in the module docstring.

    `d_e` and `model` must be in units where Var = mean: ADU above the offset
    divided by the result's `dispersion` (photoelectrons, near enough).
    """
    d_e = np.asarray(d_e, dtype=float)
    m = np.maximum(np.asarray(model, dtype=float), 1e-6)
    r = d_e - m

    # Unit-flux PSF kernel on a pixel grid, using the same pixel-integrated
    # Gaussian the model itself is built from -- not a sampled Gaussian.
    rad = int(np.ceil(truncate * sigma))
    ax = np.arange(-rad, rad + 1, dtype=float)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    g = psf.model(psf.pack(0.0, [1.0], [0.0], [0.0]), yy, xx, sigma)

    # sum_i r_i g_i / m_i  and  sum_i g_i^2 / m_i, both as correlations.
    num = ndi.correlate(r / m, g, mode="nearest")
    den = ndi.correlate(1.0 / m, g * g, mode="nearest")
    den = np.maximum(den, 1e-30)
    return num / np.sqrt(den)


def residual_peaks(d_e, model, sigma, z_thresh=5.0, min_sep=None):
    """Local extrema of the score map that clear +/- `z_thresh`.

    Returns (positive, negative), each an (M, 3) array of (y, x, z), sorted by
    |z| descending. Positive entries are flux the model has not accounted for
    -- a missed emitter. Negative entries are flux the model has invented --
    two PSFs stacked on one emitter, or a background that has been fitted too
    high.
    """
    z = score_map(d_e, model, sigma)
    if min_sep is None:
        min_sep = 2 * int(np.ceil(sigma)) + 1

    def extrema(arr, sign):
        mx = ndi.maximum_filter(arr, size=min_sep)
        hit = (arr == mx) & (arr > z_thresh)
        ys, xs = np.nonzero(hit)
        if ys.size == 0:
            return np.empty((0, 3))
        v = arr[ys, xs] * sign
        out = np.stack([ys.astype(float), xs.astype(float), v], axis=1)
        return out[np.argsort(-np.abs(out[:, 2]))]

    return extrema(z, +1.0), extrema(-z, -1.0)


def audit_result(d_e, model, sigma, z_thresh=5.0, border=0):
    """Summarize the residual audit of one result as a dict.

    `border` drops findings within that many pixels of the frame, for callers
    that deliberately do not model the rim; pass 0 to audit the whole image
    (the default, because "the rim is unmodelled" is itself a finding).
    """
    pos, neg = residual_peaks(d_e, model, sigma, z_thresh=z_thresh)
    H, W = np.asarray(d_e).shape
    if border > 0:
        def keep(a):
            if len(a) == 0:
                return a
            k = ((a[:, 0] >= border) & (a[:, 0] <= H - 1 - border)
                 & (a[:, 1] >= border) & (a[:, 1] <= W - 1 - border))
            return a[k]
        pos, neg = keep(pos), keep(neg)
    z = score_map(d_e, model, sigma)
    return {
        "positive": pos,
        "negative": neg,
        "n_missed": len(pos),
        "n_piled": len(neg),
        "z_max": float(z.max()) if z.size else 0.0,
        "z_min": float(z.min()) if z.size else 0.0,
        "z_median": float(np.median(z)),
        "z_robust_std": float(0.5 * (np.percentile(z, 84.1) - np.percentile(z, 15.9))),
        "clean": len(pos) == 0 and len(neg) == 0,
    }


def format_report(a, label="", max_list=6):
    """One-block human-readable form of `audit_result`."""
    lines = []
    head = f"residual audit{' [' + label + ']' if label else ''}"
    lines.append(f"{head}: z median {a['z_median']:+.2f}, robust sd {a['z_robust_std']:.2f}, "
                 f"range {a['z_min']:+.1f} .. {a['z_max']:+.1f}")
    if a["clean"]:
        lines.append("  CLEAN: no PSF-shaped excess or deficit above threshold")
        return "\n".join(lines)
    if a["n_missed"]:
        lines.append(f"  {a['n_missed']} UNEXPLAINED POSITIVE peak(s) -- missed emitters:")
        for y, x, z in a["positive"][:max_list]:
            lines.append(f"      y={y:6.1f} x={x:6.1f}   z=+{z:6.1f}")
        if a["n_missed"] > max_list:
            lines.append(f"      ... and {a['n_missed'] - max_list} more")
    if a["n_piled"]:
        lines.append(f"  {a['n_piled']} NEGATIVE peak(s) -- over-modelled / piled-up PSFs:")
        for y, x, z in a["negative"][:max_list]:
            lines.append(f"      y={y:6.1f} x={x:6.1f}   z={z:7.1f}")
        if a["n_piled"] > max_list:
            lines.append(f"      ... and {a['n_piled'] - max_list} more")
    return "\n".join(lines)
