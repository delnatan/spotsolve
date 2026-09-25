"""Multi-emitter localization using the native score-gated search.

Frames are independent. An ROI restricts seeds, placements and
preprocessing; fitted centers may move outside it. Background and
dispersion use the available ROI context, so masked results can differ
from full-frame results.
"""

import os
import operator

import numpy as np

from .results import Localizations

try:
    import spotsolve_rs as _rs
    if getattr(_rs, "BOX_OUTPUT_VERSION", 0) != 4:
        raise ImportError("incompatible localization output; rebuild spotsolve_rs")
except ImportError as error:          # pragma: no cover - build problem
    raise ImportError(
        "spotsolve needs its bundled Rust extension; reinstall a compatible wheel "
        "or run `maturin develop --release` from the repository root") from error

__all__ = ["localize", "localize_stack", "SLACK", "K_MAX", "FP_PER_MPX"]

SLACK = tuple(_rs.BOX_SLACK)
"""Widths a fit may take, as multiples of `sigma`: the model space."""
K_MAX = int(_rs.BOX_K_MAX)
"""Most emitters one seed's window fits; a safety cap."""
FP_PER_MPX = float(_rs.BOX_FP_PER_MPX)
"""Default expected false emitters per 10^6 pixels of pure noise."""


def _positive_int(value, name):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _roi(roi, shape):
    if roi is None:
        return None
    roi = np.ascontiguousarray(roi, dtype=bool)
    if roi.shape != tuple(shape):
        raise ValueError(f"roi has shape {roi.shape}, frame {tuple(shape)}")
    return roi


def _kw(sigma, offset, roi, shape, fp_per_mpx, slack):
    sigma = float(sigma)
    if not np.isfinite(sigma) or not 0 < sigma <= max(shape):
        raise ValueError("sigma must be positive, finite and no larger than the frame")
    return dict(sigma=sigma, offset=float(offset), roi=_roi(roi, shape),
                fp_per_mpx=float(fp_per_mpx), slack=tuple(map(float, slack)))


def _result(out, raw, kw, images):
    pos, amp, sig, se, sig_se, flags, bmap, info = out
    dispersion = info.pop("dispersion")
    sigma = kw["sigma"]
    info.update(fp_per_mpx=kw["fp_per_mpx"])
    model = residual = None
    if images:
        model = _rs.box_render(pos, amp, sig, bmap)
        residual = raw - kw["offset"] - model
    return Localizations(
        positions=pos, amplitudes=amp, se=se,
        fit_sigma=sig, sigma_se=sig_se, flags=flags,
        background=bmap, sigma=sigma, dispersion=dispersion, info=info,
        model_image=model, residual=residual)


def localize(frame, sigma, *, offset=0.0, roi=None, fp_per_mpx=FP_PER_MPX,
             slack=SLACK, images=True):
    """Localize one frame. Returns `Localizations`.

    `frame`, `offset`, flux and background use camera units (ADU). `sigma`
    is the in-focus PSF width in pixels; `slack` bounds fitted widths as
    multiples of `sigma`. Boundary solutions are flagged.

    One statistic proposes every emitter: the efficient score z for one more
    emitter of width `sigma`, given everything already fitted nearby. Seeds
    are local maxima of z over the frame above a threshold u; each starts as
    one emitter of one joint model of the frame: every emitter plus a
    bilinear background on 16-px nodes (the returned background map), fitted
    to convergence in small coupled groups. Then counts change: an emitter
    is removed if dropping it costs less than u^2 / 2 dispersion-scaled
    nats, and one is added where the residual's z exceeds u * kappa and the
    refit gains (u * kappa)^2 / 2, until nothing changes. `info["kappa"]`
    >= 1 is the residual score's spread far from any emitter (an empirical
    null that absorbs PSF and background misfit; 1 in the first add round).
    `fp_per_mpx` sets u: the expected number of false emitters per 10^6
    pixels of pure noise (calibrated on simulated noise). Lower it for fewer
    false positives, raise it for dim data. `info` reports `u`, `adds`,
    `removed` and `outer` (rounds).

    Every emitter is returned with `FitFlag` diagnostics; there are no
    brightness or width cuts after fitting. Wide fits are kept as fitted,
    so filter by width downstream. `images=False` skips `model_image` and
    `residual`.

    `info['fisher_fraction']` is an (N, 4) array for `(flux, y, x, sigma)`:
    conditional/marginal Fisher variance. Small values mean strong coupling
    to other fitted parameters, not necessarily poor absolute precision.
    NaN means covariance unavailable. These are diagnostics only.
    """
    raw = np.ascontiguousarray(frame, dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {raw.shape}")
    kw = _kw(sigma, offset, roi, raw.shape, fp_per_mpx, slack)
    return _result(_rs.box_localize(raw, **kw), raw, kw, images)


def localize_stack(stack, sigma, *, offset=0.0, roi=None, fp_per_mpx=FP_PER_MPX,
                   slack=SLACK, n_threads=None, images=False):
    """Localize every frame of a `(T, H, W)` stack, in parallel.

    Returns one `Localizations` per frame, in frame order, each what
    `localize` returns for that frame; each frame's background and
    dispersion are its own. `n_threads` defaults to the machine's cores.
    `images` defaults to False here: for a long timecourse the model and
    residual are two more copies of the movie.
    """
    raw = np.ascontiguousarray(stack, dtype=float)
    if raw.ndim != 3:
        raise ValueError(f"expected a (T, H, W) stack, got shape {raw.shape}")
    kw = _kw(sigma, offset, roi, raw.shape[1:], fp_per_mpx, slack)
    outs = _rs.box_localize_stack(
        raw, **kw, n_threads=_positive_int((os.cpu_count() or 1) if n_threads is None else n_threads,
                                "n_threads"))
    return [_result(o, raw[t], kw, images) for t, o in enumerate(outs)]
