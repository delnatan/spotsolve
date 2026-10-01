"""Localization with the native detector, after u-track's pointSourceDetection.

Frames are independent. An ROI restricts seeds; fitted centers may move
outside it. The dispersion uses the available ROI context, so masked
results can differ from full-frame results.
"""

import os
import operator

import numpy as np

from .results import Localizations

try:
    import spotsolve_rs as _rs
    if getattr(_rs, "DETECT_OUTPUT_VERSION", 0) != 6:
        raise ImportError("incompatible localization output; rebuild spotsolve_rs")
except ImportError as error:          # pragma: no cover - build problem
    raise ImportError(
        "spotsolve needs its bundled Rust extension; reinstall a compatible wheel "
        "or run `maturin develop --release` from the repository root") from error

__all__ = ["localize", "localize_stack", "FP_PER_MPX", "MAX_MIXTURES", "WIDTH"]

FP_PER_MPX = float(_rs.DETECT_FP_PER_MPX)
MAX_MIXTURES = int(_rs.DETECT_MAX_MIXTURES)
WIDTH = tuple(float(v) for v in _rs.DETECT_WIDTH)
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


def _kw(sigma, offset, roi, shape, fp_per_mpx, width, fit_mixtures, max_mixtures):
    sigma = float(sigma)
    if not np.isfinite(sigma) or not 0 < sigma <= max(shape):
        raise ValueError("sigma must be positive, finite and no larger than the frame")
    return dict(sigma=sigma, offset=float(offset), roi=_roi(roi, shape),
                fp_per_mpx=float(fp_per_mpx), width=tuple(map(float, width)),
                fit_mixtures=bool(fit_mixtures),
                max_mixtures=_positive_int(max_mixtures, "max_mixtures"))


def _result(out, raw, kw, images):
    pos, amp, sig, se, sig_se, flags, bmap, info = out
    dispersion = info.pop("dispersion")
    sigma = kw["sigma"]
    info.update(method="detect", fp_per_mpx=kw["fp_per_mpx"], width=kw["width"],
                fit_mixtures=kw["fit_mixtures"])
    model = residual = None
    if images:
        model = _rs.detect_render(pos, amp, sig, bmap)
        residual = raw - kw["offset"] - model
    return Localizations(
        positions=pos, amplitudes=amp, se=se,
        fit_sigma=sig, sigma_se=sig_se, flags=flags,
        background=bmap, sigma=sigma, dispersion=dispersion, info=info,
        model_image=model, residual=residual)


def localize(frame, sigma, *, offset=0.0, roi=None, fp_per_mpx=FP_PER_MPX,
             width=WIDTH, fit_mixtures=False, max_mixtures=MAX_MIXTURES,
             images=True):
    """Localize one frame. Returns `Localizations`.

    `frame`, `offset`, flux and background use camera units (ADU). `sigma`
    is the in-focus PSF width in pixels.

    The frame is screened by the Poisson score of one emitter at each pixel;
    local maxima of the Laplacian of Gaussian where that score is high
    enough become seeds. Each seed is fitted alone on a window of
    `ceil(4 sigma)` px around it: a constant level plus one emitter, held
    within `2 sigma` of the seed, with the pixels of other significant spots
    left out. Each emitter's width is fitted; emitters are reported within
    `width` (multiples of `sigma`, by default `sigma` to `1.5 sigma`; the
    upper bound is capped at half the window's half-side, about `2 sigma`).
    A component may widen beyond that to the window's half-side:
    out-of-focus light, fitted so narrower components need not split it,
    counted in `info["out_of_focus"]` and not reported. Equal bounds fix the width. An emitter is kept if its
    likelihood ratio against the level alone, in nats scaled by the frame's
    measured dispersion, reaches `u^2 / 2`. `fp_per_mpx` sets `u`: the
    expected number of false emitters per 10^6 pixels of noise, for a search
    over position and width.

    With `fit_mixtures`, after u-track's FitMixtures, a window takes more
    components: each next one where the residual's score peaks anywhere in
    the window, kept while each refit gains `u^2 / 2` nats, up to
    `max_mixtures`; then the weakest is removed while removing it, the rest
    refitted, costs less. Neighbours inside a window are fitted with it, and
    each component is reported by the fit of the seed nearest to it; copies
    of one emitter from two windows are merged.
    `info["mixture"]` numbers the windows that held several components (0:
    fitted alone). Use mixtures wherever spots come closer than about
    `4 sigma`; alone, a fit is biased by a neighbour's light.

    `info["z"]` is each emitter's `sqrt(2 * likelihood ratio)`, at least
    `info["u"]`. Every emitter is returned with `FitFlag` diagnostics.
    `images=False` skips `model_image` and `residual`.
    """
    raw = np.ascontiguousarray(frame, dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {raw.shape}")
    kw = _kw(sigma, offset, roi, raw.shape, fp_per_mpx, width, fit_mixtures, max_mixtures)
    return _result(_rs.detect_localize(raw, **kw), raw, kw, images)


def localize_stack(stack, sigma, *, offset=0.0, roi=None, fp_per_mpx=FP_PER_MPX,
                   width=WIDTH, fit_mixtures=False, max_mixtures=MAX_MIXTURES,
                   n_threads=None, images=False):
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
    kw = _kw(sigma, offset, roi, raw.shape[1:], fp_per_mpx, width, fit_mixtures, max_mixtures)
    outs = _rs.detect_localize_stack(
        raw, **kw, n_threads=_positive_int((os.cpu_count() or 1) if n_threads is None else n_threads,
                                "n_threads"))
    return [_result(o, raw[t], kw, images) for t, o in enumerate(outs)]
