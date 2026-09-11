"""The box search, native: one frame, or a whole timecourse in parallel.

`localize` and `localize_stack` run `box.localize_boxes`'s algorithm in Rust
(`spotsolve_core::boxsearch`), with the GIL released. `box.localize_boxes`
stays the reference: every constant's measurement lives there, and the port
is held to statistical parity with it (`tests/test_localize.py`).

Frames of a timecourse are independent, so `localize_stack` hands them to a
pool of native threads, each with its own workspace.
"""

import os

import numpy as np

from . import box, calibrate, core

__all__ = ["localize", "localize_stack"]


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
        "BOX_POLISH_SWEEPS": core.REFINE_SWEEPS,
        "BOX_POLISH_MAX_ITER": core.REFINE_MAX_ITER,
        "BOX_POLISH_TOL_OBJ": core.REFINE_TOL_OBJ,
        "BOX_POLISH_MOVE_TOL": core.REFINE_TOL,
        "BOX_BG_KERNEL": core.BG_KERNEL, "BOX_BG_FLOOR": core.BG_FLOOR,
        "BOX_BG_MASK_RADIUS": core.BG_MASK_RADIUS,
        "BOX_BG_MIN_PIXELS": core.BG_MIN_PIXELS,
        "BOX_SEED_ALPHA": calibrate.SEED_ALPHA,
        "BOX_A_MIN": core.moves.A_MIN, "BOX_A_MIN_REL": core.A_MIN_REL,
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


def _finish(out, d_e, shift, g_eff, sigma, band, slack, rs):
    pos, amp, sig, se, bmap, counts = out
    model = (None if d_e is None else
             rs.box_render(pos, amp, sig, bmap))
    history = [dict(counts, N=int(len(amp)))]
    return box._result(pos, amp, sig, se, bmap, model, d_e, shift, g_eff,
                       sigma, band, slack, history)


def localize(data_img, sigma=1.2, offset=0.0, gain=None, read_noise=0.0,
             roi=None, k_max=12, threshold=None, slack=core.SIGMA_SLACK,
             band=core.FOCUS_BAND, sweeps=box.SWEEPS, polish=True):
    """Localize one frame. Same arguments and `DetectResult` as
    `box.localize_boxes`, which documents them; `history[0]` also counts the
    polish's fits."""
    rs = _rs()
    _check_constants(rs)
    raw = np.asarray(data_img, dtype=float)
    g_eff = calibrate.estimate_gain(raw, offset) if gain is None else float(gain)
    shift = float(read_noise) ** 2
    d_e = np.ascontiguousarray((raw - offset) / g_eff + shift)
    out = rs.box_localize(d_e, float(sigma), roi=_roi(roi, raw.shape),
                          k_max=int(k_max), threshold=threshold,
                          slack=tuple(map(float, slack)), sweeps=int(sweeps),
                          polish=bool(polish))
    return _finish(out, d_e, shift, g_eff, sigma, band, slack, rs)


def localize_stack(stack, sigma=1.2, offset=0.0, gain=None, read_noise=0.0,
                   roi=None, k_max=12, threshold=None, slack=core.SIGMA_SLACK,
                   band=core.FOCUS_BAND, sweeps=box.SWEEPS, polish=True,
                   n_threads=None, images=False):
    """Localize every frame of a `(T, H, W)` stack, in parallel.

    Returns one `DetectResult` per frame, in frame order, each the same as
    `localize` would return for that frame. `gain=None` estimates it per
    frame. `n_threads` defaults to the machine's cores. `images=False`
    leaves `model_image` and `residual` as None -- for a long timecourse they
    are two more copies of the movie.
    """
    rs = _rs()
    _check_constants(rs)
    raw = np.ascontiguousarray(stack, dtype=float)
    if raw.ndim != 3:
        raise ValueError(f"expected a (T, H, W) stack, got shape {raw.shape}")
    T = raw.shape[0]
    gains = (np.array([calibrate.estimate_gain(f, offset) for f in raw])
             if gain is None else np.full(T, float(gain)))
    shift = float(read_noise) ** 2
    outs = rs.box_localize_stack(
        raw, float(sigma), float(offset), gains, shift=shift,
        roi=_roi(roi, raw.shape[1:]), k_max=int(k_max), threshold=threshold,
        slack=tuple(map(float, slack)), sweeps=int(sweeps),
        polish=bool(polish), n_threads=int(n_threads or os.cpu_count() or 1))
    return [_finish(o, (raw[t] - offset) / gains[t] + shift if images else None,
                    shift, gains[t], sigma, band, slack, rs)
            for t, o in enumerate(outs)]
