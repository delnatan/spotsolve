"""Localization with the native detector, after u-track's pointSourceDetection.

The camera's gain is an input (`estimate_gain` measures it from a movie);
the read variance follows from the background. The bar on each emitter's
likelihood ratio is set by the movie: the probability that an emitter of
that ratio is real, not noise, is fitted to the candidates of every frame. An ROI restricts seeds; fitted centers may
move outside it. The noise measurement uses the available ROI context, so masked
results can differ from full-frame results.
"""

import os
import operator

import numpy as np

from .results import Localizations

try:
    import spotsolve_rs as _rs
    if getattr(_rs, "DETECT_OUTPUT_VERSION", 0) != 9:
        raise ImportError("incompatible localization output; rebuild spotsolve_rs")
except ImportError as error:          # pragma: no cover - build problem
    raise ImportError(
        "spotsolve needs its bundled Rust extension; reinstall a compatible wheel "
        "or run `maturin develop --release` from the repository root") from error

__all__ = ["localize", "localize_stack", "estimate_gain", "Progress", "Cancelled",
           "P_REAL", "MAX_MIXTURES", "WIDTH"]

P_REAL = float(_rs.DETECT_P_REAL)
"""Default least probability that a reported emitter is real, not noise."""
MAX_MIXTURES = int(_rs.DETECT_MAX_MIXTURES)
WIDTH = tuple(float(v) for v in _rs.DETECT_WIDTH)

Progress = _rs.Progress
Progress.__doc__ = """A `localize_stack` run as another thread sees it: `done` frame fits of
`total` (each frame once alone and, with mixtures, once more), and
`cancel()` to stop it. Watching does not change the result."""


class Cancelled(Exception):
    """`localize_stack` stopped because its `progress` was cancelled."""


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


def _sigma(sigma, shape):
    sigma = float(sigma)
    if not np.isfinite(sigma) or not 0 < sigma <= max(shape):
        raise ValueError("sigma must be positive, finite and no larger than the frame")
    return sigma


def _kw(sigma, gain, offset, roi, shape, p_real, width, fit_mixtures, max_mixtures, motion_blur, model_error=0.0):
    gain = float(gain)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be positive and finite")
    return dict(sigma=_sigma(sigma, shape), gain=gain, offset=float(offset), roi=_roi(roi, shape),
                p_real=float(p_real), width=tuple(map(float, width)),
                fit_mixtures=bool(fit_mixtures),
                max_mixtures=_positive_int(max_mixtures, "max_mixtures"),
                motion_blur=bool(motion_blur), model_error=_model_error(model_error))


def _model_error(e):
    e = float(e)
    if not np.isfinite(e) or e < 0:
        raise ValueError("model_error must be finite and non-negative")
    return e


def _result(out, raw, kw, images):
    pos, amp, sig, se, sig_se, flags, bmap, info = out
    dispersion = info.pop("dispersion")
    sigma = kw["sigma"]
    info.update(method="detect", p_real_min=kw["p_real"], width=kw["width"],
                fit_mixtures=kw["fit_mixtures"], motion_blur=kw["motion_blur"])
    model = residual = None
    if images:
        model = _rs.detect_render(pos, amp, sig, bmap)
        residual = raw - kw["offset"] - model
    return Localizations(
        positions=pos, amplitudes=amp, se=se,
        fit_sigma=sig, sigma_se=sig_se, flags=flags,
        background=bmap, sigma=sigma, dispersion=dispersion, info=info,
        model_image=model, residual=residual)


def localize(frame, sigma, *, gain, offset=0.0, roi=None, p_real=P_REAL,
             width=WIDTH, fit_mixtures=False, max_mixtures=MAX_MIXTURES,
             motion_blur=False, model_error=0.0, images=True):
    """Localize one frame. Returns `Localizations`.

    `frame`, `offset`, flux and background use camera units (ADU). `sigma`
    is the in-focus PSF width in pixels. `gain` is the camera's pixel
    variance per ADU of light above the offset: its conversion gain (ADU per
    photoelectron) times any excess-noise factor (2 for an EMCCD), from the
    camera's calibration or measured from a movie by `estimate_gain`.

    Every pixel's variance is the camera's noise curve at the light the
    model puts on it, `info["read_variance"] + gain * light`, the read
    variance measured from the background given the gain. Every window is
    fitted at maximum quasi-likelihood under that curve (the Poisson
    likelihood for a Poisson camera), and every decision compares its
    deviance.

    The frame is screened by the score of one emitter at each pixel;
    local maxima of the Laplacian of Gaussian where that score is high
    enough become seeds. Each seed is fitted alone on a window of
    `ceil(4 sigma)` px around it: a constant level plus one emitter, held
    within `2 sigma` of the seed, with the pixels of other significant spots
    left out. Each emitter's width is fitted; emitters are reported within
    `width` (multiples of `sigma`, by default `sigma` to `1.5 sigma`; the
    upper bound is capped at half the window's half-side, about `2 sigma`).
    A component may widen beyond that to the window's half-side:
    out-of-focus light, fitted so narrower components need not split it,
    counted in `info["out_of_focus"]` and not reported, once its width is
    2 standard errors beyond the upper bound; less wide, it is an emitter at
    the bound, flagged `AT_BOUND`. Equal bounds fix the width.

    An emitter is kept if it is real, not noise, with probability at least
    `p_real`. The probability comes from its likelihood ratio against the
    level alone, `detect_z = sqrt(2 * ratio)` in nats: candidates fitted alone above a low bar have a density of `detect_z` that
    noise alone, by theory, would give only in part,
    and `P(real | detect_z)` is the part noise does not explain. The emitter is
    kept where `detect_z >= u`, the bar at which that reaches `p_real`. A single
    frame is a movie of one; `localize_stack` pools every frame.

    With `fit_mixtures`, after u-track's FitMixtures, a window takes more
    components: each next one where the residual's score peaks anywhere in
    the window, kept while each refit gains `u^2 / 2` nats, up to
    `max_mixtures`; then the weakest is removed while removing it, the rest
    refitted, costs less. Neighbours inside a window are fitted with it, and
    each component is reported by the fit of the seed nearest to it; copies
    of one emitter from two windows are merged. `info["mixture"]` numbers
    the windows that held several components (0: fitted alone). Use
    mixtures wherever spots come closer than about `6 sigma`; alone, a fit
    is biased by a neighbour's light.

    With `motion_blur` as well, a close pair is two emitters only if it
    gains `u^2 / 2` nats over one elongated spot, as an emitter moving
    within the exposure is; otherwise it is one emitter at the ellipse's
    centre, flagged `ELONGATED`, its width the geometric mean of the
    ellipse's axes. Pairs closer than about `2 sigma` are so reported, so
    leave it off where the exposure is short enough that emitters stay
    round.

    `info["detect_z"]` is each emitter's `sqrt(2 * likelihood ratio)`, at least
    `info["u"]`; `info["p_real"]` its probability of being real, at least
    `p_real`. Every emitter is returned with `FitFlag` diagnostics.
    `model_error` is the PSF model's relative error: every pixel's variance
    gains `(model_error * (light - level))^2`, `level` its window's
    background, so that light the Gaussians do not describe exactly (PSF
    wings and rings, a bead's size, defocus) is not read as another emitter,
    and a bright emitter's standard errors include its misfit.
    `estimate_model_error` measures it from a movie; 0 is the camera alone.

    `images=False` skips `model_image` and `residual`.
    """
    raw = np.ascontiguousarray(frame, dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {raw.shape}")
    kw = _kw(sigma, gain, offset, roi, raw.shape, p_real, width, fit_mixtures, max_mixtures, motion_blur, model_error)
    return _result(_rs.detect_localize(raw, **kw), raw, kw, images)


def localize_stack(stack, sigma, *, gain, offset=0.0, roi=None, p_real=P_REAL,
                   width=WIDTH, fit_mixtures=False, max_mixtures=MAX_MIXTURES,
                   motion_blur=False, model_error=0.0, n_threads=None, images=False, progress=None):
    """Localize every frame of a `(T, H, W)` stack, in parallel.

    Returns one `Localizations` per frame, in frame order. Each frame's
    background is its own; the read variance, the bar `u` and
    `P(real | detect_z)` are the movie's, fitted to every frame, so a frame's
    result depends on the stack it is in. `n_threads` defaults to the machine's cores.
    `images` defaults to False here: for a long timecourse the model and
    residual are two more copies of the movie.

    To follow a long run, pass a `Progress` and read it from another thread;
    its `cancel()` stops the run, which then raises `Cancelled`. Split into
    several calls instead, each would be a movie of its own, with its own
    read variance, bar `u` and `P(real | detect_z)`.
    """
    raw = np.ascontiguousarray(stack, dtype=float)
    if raw.ndim != 3:
        raise ValueError(f"expected a (T, H, W) stack, got shape {raw.shape}")
    kw = _kw(sigma, gain, offset, roi, raw.shape[1:], p_real, width, fit_mixtures, max_mixtures, motion_blur, model_error)
    outs = _rs.detect_localize_stack(raw, **kw, n_threads=_threads(n_threads), progress=progress)
    if outs is None:
        raise Cancelled("localize_stack was cancelled")
    return [_result(o, raw[t], kw, images) for t, o in enumerate(outs)]


def estimate_gain(stack, sigma, *, offset=0.0, roi=None, width=WIDTH, n_threads=None):
    """The camera's gain, pixel variance per ADU of light, from a movie.

    For `gain` in `localize` and `localize_stack`, where the camera's
    calibration is not at hand. `stack` is one frame or a `(T, H, W)` movie
    in ADU, `offset` the camera's. Bright emitters span a wide range of
    light: fitted with every light in their windows, their residuals,
    corrected for each pixel's leverage, have the variance the camera's
    curve gives them, and the gain is the value at which their median
    matches. Frames are probed, spread over the movie, until their pixels
    fix the gain. Without bright emitters it is the background's variance
    per unit level: exact without read noise, otherwise an upper bound.
    """
    raw = np.ascontiguousarray(stack, dtype=float)
    if raw.ndim == 2:
        raw = raw[None]
    if raw.ndim != 3 or raw.shape[0] == 0:
        raise ValueError(f"expected a frame or a (T, H, W) stack, got shape {raw.shape}")
    return float(_rs.detect_estimate_gain(
        raw, _sigma(sigma, raw.shape[1:]), float(offset), roi=_roi(roi, raw.shape[1:]),
        width=tuple(map(float, width)), n_threads=_threads(n_threads)))


def estimate_model_error(stack, sigma, *, gain, offset=0.0, roi=None, width=WIDTH, n_threads=None):
    """The PSF model's relative error, for `model_error`, from a movie.

    Measured as `estimate_gain` measures the gain, on the same bright
    emitters fitted with every light in their windows, given the camera's
    `gain`: the error `e` at which their leverage-corrected squared
    residuals, over the camera's curve plus `(e * (light - level))^2`, have
    the median of chi-squared(1). 0 when the camera's curve alone already
    explains them; NaN without enough bright emitters.
    """
    raw = np.ascontiguousarray(stack, dtype=float)
    if raw.ndim == 2:
        raw = raw[None]
    if raw.ndim != 3 or raw.shape[0] == 0:
        raise ValueError(f"expected a frame or a (T, H, W) stack, got shape {raw.shape}")
    return float(_rs.detect_estimate_model_error(
        raw, _sigma(sigma, raw.shape[1:]), float(offset), gain=float(gain), roi=_roi(roi, raw.shape[1:]),
        width=tuple(map(float, width)), n_threads=_threads(n_threads)))


def _threads(n_threads):
    return _positive_int((os.cpu_count() or 1) if n_threads is None else n_threads, "n_threads")
