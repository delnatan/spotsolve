"""The detector: one frame, or a whole timecourse in parallel.

`localize` and `localize_stack` run the box search in Rust
(`spotsolve_core::boxsearch`) with the GIL released: the noise map, FIND, the
background surface, the search in each box, the polish and the
classification of every fit.

There is no camera calibration to pass. The frame stays in ADU above
`offset`, and the noise every decision is scaled by -- gain, read noise and
haze alike -- is measured from the frame itself (`boxsearch::noise_map`). This module only arranges the
arrays into `Localizations`. Its defaults are the Rust constants, read from
the extension, so there is no second copy of them here.

In each box an emitter exists iff it lowers the box's Poisson deviance by
`spotsolve_rs.BOX_ADD_NATS` nats, in units of the box's own dispersion. The reasoning and measurement behind every
constant sit beside it in `rust/spotsolve-core/src/boxsearch.rs`.

Frames of a timecourse are independent, so `localize_stack` hands them to a
pool of native threads, each with its own workspace.

An `roi` confines the SEARCH, not the answer: a source just outside it
whose light crosses into it is fitted where it actually is. So pass one ROI
covering everything wanted, in one call. Tiling a frame into separate ROI
calls double-reports sources at the seams (measured: about 1.6 per frame at
one seam through a 64x64 frame) and loses recall there, because neither
tile's boxes see the other's emitters. One call never double-reports.

An `roi` also confines the WORK. FIND, the background surface and the level
they are measured against run on the ROI's bounding box plus a 41 px margin,
not on the frame, which is what a masked call used to waste: a 32x32 ROI on
a 512x512 frame went from 25.4 ms to 3.4 ms. The margin is wide enough that
nothing inside the ROI can tell, so this costs no accuracy; an ROI whose
bounding box is the whole frame (scattered cells, a diagonal band) simply
buys nothing.

One thing it does change: the flat level FIND works against, and the
background's fallback, are measured over the ROI's pixels rather than the
frame's. Under a cell mask the frame's own 10th percentile is the dark field
OUTSIDE the cell -- 11.1 e- against 18-20 inside it on `hyp7gem_wt_crop` --
so this is the level the mask asked for. It moves N by up to 5% against
earlier versions on masked calls. Unmasked calls are unchanged, bit for bit.
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


def _kw(sigma, offset, roi, shape, k_max, threshold, slack, band):
    return dict(sigma=float(sigma), offset=float(offset), roi=_roi(roi, shape),
                k_max=int(k_max), threshold=threshold,
                slack=tuple(map(float, slack)),
                band=None if band is None else tuple(map(float, band)))


def _result(out, raw, kw, images):
    pos, amp, sig, se, sig_se, cls, bmap, info = out
    dispersion = info.pop("dispersion")
    sigma = kw["sigma"]
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
             threshold=None, slack=SLACK, band=BAND, images=True):
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
    Every emitter tried still has to pay `ADD_NATS`. Lower it for dim data,
    raise it for fewer false positives and speed; the measured trade sits
    beside the Rust constant. `images=False` skips `model_image` and
    `residual`.
    """
    raw = np.ascontiguousarray(frame, dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {raw.shape}")
    kw = _kw(sigma, offset, roi, raw.shape, k_max, threshold, slack, band)
    return _result(_rs.box_localize(raw, **kw), raw, kw, images)


def localize_stack(stack, sigma, *, offset=0.0, roi=None, k_max=K_MAX,
                   threshold=None, slack=SLACK, band=BAND, n_threads=None,
                   images=False):
    """Localize every frame of a `(T, H, W)` stack, in parallel.

    Returns one `Localizations` per frame, in frame order, each what
    `localize` returns for that frame; each frame's noise is its own.
    `n_threads` defaults to the machine's cores. `images` defaults to False
    here: for a long timecourse the model and residual are two more copies of
    the movie.
    """
    raw = np.ascontiguousarray(stack, dtype=float)
    if raw.ndim != 3:
        raise ValueError(f"expected a (T, H, W) stack, got shape {raw.shape}")
    kw = _kw(sigma, offset, roi, raw.shape[1:], k_max, threshold, slack,
             band)
    outs = _rs.box_localize_stack(
        raw, **kw, n_threads=int(n_threads or os.cpu_count() or 1))
    return [_result(o, raw[t], kw, images) for t, o in enumerate(outs)]
