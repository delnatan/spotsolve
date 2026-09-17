"""Multi-emitter localization of one frame or a parallel frame stack.

The Rust box search measures local noise/background, proposes LoG peaks,
selects emitter counts, refines joint fits and classifies fitted widths.
`selection="fixed"` uses a 10-nat cost plus `count_penalty`; `"bic"` compares
count models. Frames are independent, each with its own workspace.

Input stays in ADU above `offset`; local dispersion scales fit comparisons
and uncertainties. An ROI restricts search and crops work while retaining
context. Fits may lie outside it. Background reference levels use masked
pixels, so masked results can differ from filtering an unmasked result.
Use one mask rather than separate tile calls, which can duplicate sources.

This module converts native output to `Localizations`. Defaults come from
the Rust extension. Current methods and measurements are in docs/DETECTION.md
and docs/COUNT_SELECTION.md; historical notes are in
docs/archive/DETECTOR_DESIGN_NOTES.md.
"""

import os

import numpy as np

from .results import REJECT_DTYPE, Localizations

try:
    import spotsolve_rs as _rs
except ImportError as error:          # pragma: no cover - build problem
    raise ImportError(
        "spotsolve needs its Rust extension; build it with `maturin develop "
        "--release -m rust/spotsolve-py/Cargo.toml`") from error

__all__ = ["localize", "localize_stack", "SLACK", "BAND", "BAND_Z", "K_MAX",
           "PEAK_Z"]

SLACK = tuple(_rs.BOX_SLACK)
"""Widths a fit may take, as multiples of `sigma`: the model space."""
BAND = tuple(_rs.BOX_BAND)
"""Widths reported as detections, as multiples of `sigma`."""
BAND_Z = float(_rs.BOX_BAND_Z)
"""A width outside `BAND` by no more than this many SEs is still reported."""
K_MAX = int(_rs.BOX_K_MAX)
"""Most emitters one box fits jointly."""
PEAK_Z = float(_rs.BOX_PEAK_Z)
"""The default LoG cut, in sds of the local noise, for candidates and
placements alike."""

_REASONS = ("too_narrow", "too_wide", "edge")    # classes 1, 2, 3


def _roi(roi, shape):
    if roi is None:
        return None
    roi = np.ascontiguousarray(roi, dtype=bool)
    if roi.shape != tuple(shape):
        raise ValueError(f"roi has shape {roi.shape}, frame {tuple(shape)}")
    return roi


def _kw(sigma, offset, roi, shape, k_max, threshold, slack, band,
        selection, count_penalty):
    return dict(sigma=float(sigma), offset=float(offset), roi=_roi(roi, shape),
                k_max=int(k_max), threshold=threshold,
                selection=selection, count_penalty=float(count_penalty),
                slack=tuple(map(float, slack)),
                band=None if band is None else tuple(map(float, band)))


def _result(out, raw, kw, images):
    pos, amp, sig, se, sig_se, cls, bmap, info = out
    dispersion = info.pop("dispersion")
    sigma = kw["sigma"]
    info.update(selection=kw["selection"], count_penalty=kw["count_penalty"])
    focus = cls == 0
    out_band = ~focus
    rejects = np.empty(int(out_band.sum()), dtype=REJECT_DTYPE)
    rejects["y"], rejects["x"] = pos[out_band, 0], pos[out_band, 1]
    rejects["flux"], rejects["sigma"] = amp[out_band], sig[out_band]
    rejects["sigma_ratio"] = sig[out_band] / sigma
    rejects["reason"] = np.asarray(_REASONS)[cls[out_band].astype(int) - 1]
    model = residual = None
    if images:
        model = _rs.box_render(pos, amp, sig, bmap)
        residual = raw - kw["offset"] - model
    return Localizations(
        positions=pos[focus], amplitudes=amp[focus], se=se[focus],
        fit_sigma=sig[focus], sigma_se=sig_se[focus], rejects=rejects,
        background=bmap, sigma=sigma, dispersion=dispersion, info=info,
        model_image=model, residual=residual)


def localize(frame, sigma, *, offset=0.0, roi=None, k_max=K_MAX,
             threshold=None, slack=SLACK, band=BAND, images=True,
             selection="fixed", count_penalty=0.0):
    """Localize one frame. Returns `Localizations`.

    `frame` is in camera units (ADU) and `offset` is the camera's offset,
    ADU. Nothing else about the camera is needed: the noise is measured from
    the frame, and fluxes and the background come back in ADU above `offset`.
    `sigma` is the in-focus PSF width (px). `slack` and `band` are the widths
    a fit may take and the widths reported as detections, as multiples of
    `sigma`; a width outside `band` by no more than `BAND_Z` of its own SE is
    still a detection, and `band=None` reports every fit.

    `threshold` (default `PEAK_Z`) is the one cut on the LoG statistic, in
    sds of the local noise: a peak must clear it to get a box, and a residual
    peak inside a box must clear it before one more emitter is tried there.
    The selected count rule then decides which fits to keep. Lower the
    threshold for dim data, raise it for fewer false positives and speed;
    the measured tradeoff is documented in docs/DETECTION.md. `images=False` skips `model_image` and
    `residual`.

    `selection="fixed"` keeps the existing greedy 10-nat rule.
    `selection="bic"` compares background-only and multiple emitter counts
    using I/phi + K*(2*log(n_pixels) + count_penalty), with all four emitter
    parameters free. It follows forward and backward fit paths, then makes
    one removal comparison pass after refinement. This is an experimental
    BIC-inspired score, not calibrated evidence or a false-positive rate.
    `count_penalty` is a finite non-negative extra cost per emitter; it also
    adds to the 10-nat cost in fixed mode. Higher values favor fewer emitters.
    """
    raw = np.ascontiguousarray(frame, dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {raw.shape}")
    kw = _kw(sigma, offset, roi, raw.shape, k_max, threshold, slack, band,
             selection, count_penalty)
    return _result(_rs.box_localize(raw, **kw), raw, kw, images)


def localize_stack(stack, sigma, *, offset=0.0, roi=None, k_max=K_MAX,
                   threshold=None, slack=SLACK, band=BAND, n_threads=None,
                   images=False, selection="fixed", count_penalty=0.0):
    """Localize every frame of a `(T, H, W)` stack, in parallel.

    Returns one `Localizations` per frame, in frame order, each what
    `localize` returns for that frame; each frame's noise is its own.
    `n_threads` defaults to the machine's cores. `images` defaults to False
    here: for a long timecourse the model and residual are two more copies of
    the movie. `selection` and `count_penalty` have the same meaning as
    in `localize`.
    """
    raw = np.ascontiguousarray(stack, dtype=float)
    if raw.ndim != 3:
        raise ValueError(f"expected a (T, H, W) stack, got shape {raw.shape}")
    kw = _kw(sigma, offset, roi, raw.shape[1:], k_max, threshold, slack,
             band, selection, count_penalty)
    outs = _rs.box_localize_stack(
        raw, **kw, n_threads=int(n_threads or os.cpu_count() or 1))
    return [_result(o, raw[t], kw, images) for t, o in enumerate(outs)]
