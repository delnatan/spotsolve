"""Sparse Aguet screening and spotfitlm-compatible single-source fitting."""

from functools import lru_cache
import operator
import os

import numpy as np
from scipy import stats

from .native import _roi, _rs
from .results import Localizations, REJECT_DTYPE


@lru_cache(maxsize=32)
def _cutoff(sigma, significance):
    """The reference's degrees of freedom are constant for each kernel."""
    radius = int(np.ceil(4 * sigma))
    g = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    n = float(len(g) ** 2)
    gsum = g.sum() ** 2
    c00 = 1 / ((g * g).sum() ** 2 - gsum * gsum / n)
    k = stats.norm.isf(significance / 2)
    a = (n - 1) / (n - 3) * c00
    b = k * k / (2 * (n - 1))
    df = (n - 1) * (a + b) ** 2 / (a * a + b * b)
    return float(k + stats.t.isf(significance, df) * np.sqrt((a + b) / n))


def _positive_int(value, name):
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _settings(shape, sigma, roi, offset, significance, boxsize, itermax, n_threads):
    sigma, offset, significance = map(float, (sigma, offset, significance))
    if min(shape) == 0:
        raise ValueError("frame dimensions must be nonzero")
    if not np.isfinite(sigma) or not 0 < sigma <= max(shape):
        raise ValueError("sigma must be positive, finite and no larger than the frame")
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    if not np.isfinite(significance) or not 0 < significance < 1:
        raise ValueError("significance must be between 0 and 1")
    boxsize = _positive_int(boxsize, "boxsize")
    if boxsize < 3 or boxsize % 2 == 0:
        raise ValueError("boxsize must be odd and at least 3")
    workers = (os.cpu_count() or 1) if n_threads is None else n_threads
    return dict(sigma=sigma, offset=offset, roi=_roi(roi, shape),
                cutoff=_cutoff(sigma, significance), boxsize=boxsize,
                itermax=_positive_int(itermax, "itermax"),
                n_threads=_positive_int(workers, "n_threads"))


def _result(output, raw, settings, significance, images):
    params, covariance, seeds, background, info = output
    width, peak = params[:, 2], params[:, 3]
    d_peak = 2 * np.pi * width ** 2
    d_width = 4 * np.pi * peak * width
    flux_var = (d_peak ** 2 * covariance[:, 3, 3]
                + d_width ** 2 * covariance[:, 2, 2]
                + d_peak * d_width * (covariance[:, 3, 2] + covariance[:, 2, 3]))
    se = np.sqrt(np.column_stack((np.maximum(flux_var, 0), covariance[:, 1, 1],
                                 covariance[:, 0, 0])))
    info.update(method="aguet", significance=float(significance),
                boxsize=settings["boxsize"], itermax=settings["itermax"],
                seed_positions=seeds.astype(int).tolist(), peak_amplitude=peak.tolist(),
                fitted_background=params[:, 4].tolist(),
                background_kind="local screening estimate; NaN outside ROI crop",
                amplitude_convention="continuous sampled-Gaussian flux")
    model = _rs.aguet_render(params, background) if images else None
    return Localizations(
        positions=np.ascontiguousarray(params[:, [1, 0]]), amplitudes=peak * d_peak,
        se=se, fit_sigma=width, sigma_se=np.sqrt(covariance[:, 2, 2]),
        rejects=np.empty(0, dtype=REJECT_DTYPE), background=background,
        sigma=settings["sigma"], dispersion=float("nan"), info=info,
        model_image=model, residual=raw-settings["offset"]-model if images else None,
    )


def localize_aguet(frame, sigma, *, roi=None, offset=0.0, significance=0.05,
                   boxsize=9, itermax=50, images=False):
    """Localize sparse spots using the spotfitlm baseline. Returns Localizations.

    `sigma` sets the screening width and initializes the free-width fit.
    `significance` is the per-pixel screening level, not a frame-wide false
    discovery rate. Each candidate receives one sampled-Gaussian Poisson fit;
    there is no multi-emitter search, width filter or trajectory filtering.

    `roi` selects integer seed centers; surrounding image pixels remain valid
    fit/filter context. Work is cropped to the ROI plus its required context.
    Fitted centers may move outside the ROI. The reference frame-border cuts
    still apply. `boxsize` must be odd and >=3; oversized boxes yield no fits.

    Amplitudes are continuous Gaussian fluxes, 2*pi*peak*sigma_fit**2.
    Flux SE includes width covariance. Failed fits are listed in
    `info['failures']` as (seed_y, seed_x, status); `rejects` is empty because
    this baseline has no width-reporting band. Dispersion is unavailable.

    The fit uses data above `offset`; observations are floored at
    1e-7 consistently in the likelihood and its derivatives. Offset is not a
    gain/read-noise correction. Optional model/residual images use the sampled
    PSF and a diagnostic screening-background map, NaN outside the ROI crop.
    """
    raw = np.ascontiguousarray(frame, dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {raw.shape}")
    settings = _settings(raw.shape, sigma, roi, offset, significance, boxsize, itermax, 1)
    output = _rs.aguet_localize_stack(raw[None], **settings)[0]
    return _result(output, raw, settings, significance, images)


def localize_aguet_stack(stack, sigma, *, roi=None, offset=0.0, significance=0.05,
                         boxsize=9, itermax=50, n_threads=None, images=False):
    """Run localize_aguet on independent frames with native frame workers.

    Input shape is (T, H, W). The shared 2-D ROI and all detection settings
    have the same meaning as in localize_aguet. Worker count defaults to the
    machine's cores; output is deterministic and stays in frame order.
    """
    raw = np.ascontiguousarray(stack, dtype=float)
    if raw.ndim != 3:
        raise ValueError(f"expected a (T, H, W) stack, got shape {raw.shape}")
    settings = _settings(raw.shape[1:], sigma, roi, offset, significance, boxsize, itermax, n_threads)
    return [_result(output, raw[t], settings, significance, images)
            for t, output in enumerate(_rs.aguet_localize_stack(raw, **settings))]
