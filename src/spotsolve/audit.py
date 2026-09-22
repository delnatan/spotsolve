"""Spatial residual diagnostics for missed or excess modeled signal.

For residual r = d - m and unit-flux PSF g, the score for adding flux is

    z = sum(r*g/m) / sqrt(sum(g*g/m)).

This is the Poisson score at the supplied model, or the standardized
weighted-least-squares amplitude with weights fixed at 1/m. It is not the
exact Poisson maximum-likelihood amplitude. Positive and negative peaks
identify patches to inspect; neither proves an emitter is missing or spurious.

Inputs use units where variance approximately equals the mean (for ADU data,
divide data and model by dispersion). The default peak threshold is a
heuristic diagnostic level, not a calibrated false-positive probability.
"""

import numpy as np
import scipy.ndimage as ndi

from . import psf

__all__ = ["score_map", "residual_peaks", "audit_result", "format_report"]


def score_map(d_e, model, sigma, truncate=4.0):
    """Per-pixel score z for adding one PSF, as described in the module docstring.

    `d_e` and `model` must be in units where Var = mean: ADU above the offset
    divided by the result's `dispersion`. These are not calibrated photon counts.
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
    |z| descending. Signs indicate unmodeled or excess modeled signal;
    noise, background and PSF mismatch can also produce extrema.
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
        lines.append(f"  {a['n_missed']} UNEXPLAINED POSITIVE peak(s) -- inspect for missing signal:")
        for y, x, z in a["positive"][:max_list]:
            lines.append(f"      y={y:6.1f} x={x:6.1f}   z=+{z:6.1f}")
        if a["n_missed"] > max_list:
            lines.append(f"      ... and {a['n_missed'] - max_list} more")
    if a["n_piled"]:
        lines.append(f"  {a['n_piled']} NEGATIVE peak(s) -- inspect for excess modeled signal:")
        for y, x, z in a["negative"][:max_list]:
            lines.append(f"      y={y:6.1f} x={x:6.1f}   z={z:7.1f}")
        if a["n_piled"] > max_list:
            lines.append(f"      ... and {a['n_piled'] - max_list} more")
    return "\n".join(lines)
