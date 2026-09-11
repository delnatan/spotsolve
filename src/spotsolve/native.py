"""The box search: one frame, or a whole timecourse in parallel.

`localize` and `localize_stack` are the detector. The whole algorithm runs
in Rust (`spotsolve_core::boxsearch`) with the GIL released: the photon
conversion and gain estimate, FIND, the background surface, the box search,
the polish and the classification of every fit. This module only arranges
the arrays into a `DetectResult`.

`box.localize_boxes` is the Python reference, where every constant's
measurement lives; the port is held to statistical parity with it
(`tests/test_localize.py`).

Frames of a timecourse are independent, so `localize_stack` hands them to a
pool of native threads, each with its own workspace.

An `roi` confines the SEARCH, not the answer: a source just outside it
whose light crosses into it is fitted where it actually is. So pass one ROI
covering everything wanted, in one call. Tiling a frame into separate ROI
calls double-reports sources at the seams (measured: about 1.6 per frame at
one seam through a 64x64 frame) and loses recall there, because neither
tile's boxes see the other's emitters. One call never double-reports.
"""

import os

import numpy as np

from . import box, calibrate, core
from .structs import DetectResult, width_reject_records

__all__ = ["localize", "localize_stack"]

_REASONS = ((1, "too_narrow"), (2, "too_wide"), (3, "edge"))


def _rs():
    import spotsolve_rs
    return spotsolve_rs


def _check_constants(rs):
    """The Rust copies of the reference's constants, compared, so a drift is
    a loud error rather than a quiet algorithmic difference."""
    pairs = {
        "BOX_ADD_NATS": box.ADD_NATS, "BOX_OWN_RADIUS": box.OWN_RADIUS,
        "BOX_SWEEPS": box.SWEEPS, "BOX_FIT_TOL_OBJ": box.FIT_TOL_OBJ,
        "BOX_FIT_MAX_ITER": box.FIT_MAX_ITER,
        "BOX_EDGE_MARGIN": box.EDGE_MARGIN,
        "BOX_POLISH_SWEEPS": core.REFINE_SWEEPS,
        "BOX_POLISH_MAX_ITER": core.REFINE_MAX_ITER,
        "BOX_POLISH_TOL_OBJ": core.REFINE_TOL_OBJ,
        "BOX_POLISH_MOVE_TOL": core.REFINE_TOL,
        "BOX_BG_KERNEL": core.BG_KERNEL, "BOX_BG_FLOOR": core.BG_FLOOR,
        "BOX_BG_MASK_RADIUS": core.BG_MASK_RADIUS,
        "BOX_BG_MIN_PIXELS": core.BG_MIN_PIXELS,
        "BOX_SEED_ALPHA": calibrate.SEED_ALPHA,
        "BOX_GAIN_FRAC": calibrate.GAIN_FRAC,
        "BOX_A_MIN": core.A_MIN, "BOX_A_MIN_REL": core.A_MIN_REL,
    }
    for name, py in pairs.items():
        if getattr(rs, name) != py:
            raise RuntimeError(f"{name} is {getattr(rs, name)} in spotsolve_rs "
                               f"and {py} in Python; rebuild the extension")


def _roi(roi, shape):
    if roi is None:
        return None
    roi = np.ascontiguousarray(roi, dtype=bool)
    if roi.shape != tuple(shape):
        raise ValueError(f"roi has shape {roi.shape}, frame {tuple(shape)}")
    return roi


def _kw(sigma, offset, gain, read_noise, roi, shape, k_max, threshold, slack,
        band, sweeps, polish):
    return dict(sigma=float(sigma), offset=float(offset),
                gain=None if gain is None else float(gain),
                read_noise=float(read_noise), roi=_roi(roi, shape),
                k_max=int(k_max), threshold=threshold,
                slack=tuple(map(float, slack)),
                band=None if band is None else tuple(map(float, band)),
                sweeps=int(sweeps), polish=bool(polish))


def _result(out, raw, kw, images, rs):
    """One frame's native arrays as a `DetectResult`, fields as
    `box._result` fills them."""
    pos, amp, sig, se, cls, bmap, info = out
    g = info.pop("gain")
    sigma, band, slack = kw["sigma"], kw["band"], kw["slack"]
    shift = kw["read_noise"] ** 2
    H, W = bmap.shape
    focus = cls == 0
    idx = np.arange(len(amp))
    rejects = np.concatenate([
        width_reject_records(idx[cls == c], pos[cls == c], amp[cls == c],
                             sig[cls == c], sigma, reason)
        for c, reason in _REASONS])
    model = rs.box_render(pos, amp, sig, bmap) if images else None
    return DetectResult(
        positions=pos[focus], amplitudes=amp[focus], sigma=sigma,
        lam=float(focus.sum()) / max(H * W, 1),
        A_s=float(np.mean(amp[focus])) if focus.any() else 0.0, gain=g,
        background=bmap - shift, n_outer_passes=1,
        model_image=None if model is None else model - shift,
        residual=None if model is None else (raw - kw["offset"]) / g + shift - model,
        se=se[focus], history=[dict(info, N=int(len(amp)))],
        width_rejects=rejects if len(rejects) else None,
        width_filter=dict(band=band, slack=slack),
        fit_sigma=sig[focus], sigma_ratio=sig[focus] / sigma)


def localize(data_img, sigma=1.2, offset=0.0, gain=None, read_noise=0.0,
             roi=None, k_max=12, threshold=None, slack=core.SIGMA_SLACK,
             band=core.FOCUS_BAND, sweeps=box.SWEEPS, polish=True):
    """Localize one frame. Returns a `DetectResult`.

    `sigma` is the in-focus PSF width (px); `gain` (ADU per photoelectron),
    `offset` (ADU) and `read_noise` (e- rms) come from the camera's
    calibration, and `gain=None` estimates it from the frame. `slack` and
    `band` are the widths a fit may take and the widths reported as
    detections, as multiples of `sigma`; every fit outside `band` is returned
    in `width_rejects`. `history[0]` holds the gain used and the work done.
    """
    rs = _rs()
    _check_constants(rs)
    raw = np.ascontiguousarray(data_img, dtype=float)
    kw = _kw(sigma, offset, gain, read_noise, roi, raw.shape, k_max,
             threshold, slack, band, sweeps, polish)
    return _result(rs.box_localize(raw, **kw), raw, kw, True, rs)


def localize_stack(stack, sigma=1.2, offset=0.0, gain=None, read_noise=0.0,
                   roi=None, k_max=12, threshold=None, slack=core.SIGMA_SLACK,
                   band=core.FOCUS_BAND, sweeps=box.SWEEPS, polish=True,
                   n_threads=None, images=False):
    """Localize every frame of a `(T, H, W)` stack, in parallel.

    Returns one `DetectResult` per frame, in frame order, each what `localize`
    returns for that frame. `gain=None` estimates it per frame. `n_threads`
    defaults to the machine's cores. `images=False` leaves `model_image` and
    `residual` as None: for a long timecourse they are two more copies of the
    movie.
    """
    rs = _rs()
    _check_constants(rs)
    raw = np.ascontiguousarray(stack, dtype=float)
    if raw.ndim != 3:
        raise ValueError(f"expected a (T, H, W) stack, got shape {raw.shape}")
    kw = _kw(sigma, offset, gain, read_noise, roi, raw.shape[1:], k_max,
             threshold, slack, band, sweeps, polish)
    outs = rs.box_localize_stack(raw, **kw,
                                 n_threads=int(n_threads or os.cpu_count() or 1))
    return [_result(o, raw[t], kw, images, rs) for t, o in enumerate(outs)]
