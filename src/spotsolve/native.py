"""Multi-emitter localization using the native box search.

Frames are independent. An ROI restricts candidate seeds and preprocessing;
fitted centers may move outside it. Noise and background use the available
ROI context, so masked results can differ from filtering full-frame results.
"""

import os
import operator

import numpy as np

from .results import Localizations

try:
    import spotsolve_rs as _rs
    if getattr(_rs, "BOX_OUTPUT_VERSION", 0) != 2:
        raise ImportError("incompatible localization output; rebuild spotsolve_rs")
except ImportError as error:          # pragma: no cover - build problem
    raise ImportError(
        "spotsolve needs its bundled Rust extension; reinstall a compatible wheel "
        "or run `maturin develop --release` from the repository root") from error

__all__ = ["localize", "localize_stack", "SLACK", "K_MAX",
           "PEAK_Z"]

SLACK = tuple(_rs.BOX_SLACK)
"""Widths a fit may take, as multiples of `sigma`: the model space."""
K_MAX = int(_rs.BOX_K_MAX)
"""Most emitters one box fits jointly."""
PEAK_Z = float(_rs.BOX_PEAK_Z)
"""The default LoG cut, in sds of the local noise, for candidates and
placements alike."""


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


def _kw(sigma, offset, roi, shape, k_max, threshold, slack,
        selection, count_penalty):
    sigma = float(sigma)
    if not np.isfinite(sigma) or not 0 < sigma <= max(shape):
        raise ValueError("sigma must be positive, finite and no larger than the frame")
    return dict(sigma=sigma, offset=float(offset), roi=_roi(roi, shape),
                k_max=_positive_int(k_max, "k_max"), threshold=threshold,
                selection=selection, count_penalty=float(count_penalty),
                slack=tuple(map(float, slack)))


def _result(out, raw, kw, images):
    pos, amp, sig, se, sig_se, flags, bmap, info = out
    dispersion = info.pop("dispersion")
    sigma = kw["sigma"]
    info.update(selection=kw["selection"], count_penalty=kw["count_penalty"])
    model = residual = None
    if images:
        model = _rs.box_render(pos, amp, sig, bmap)
        residual = raw - kw["offset"] - model
    return Localizations(
        positions=pos, amplitudes=amp, se=se,
        fit_sigma=sig, sigma_se=sig_se, flags=flags,
        background=bmap, sigma=sigma, dispersion=dispersion, info=info,
        model_image=model, residual=residual)


def localize(frame, sigma, *, offset=0.0, roi=None, k_max=K_MAX,
             threshold=None, slack=SLACK, images=True,
             selection="fixed", count_penalty=0.0):
    """Localize one frame. Returns `Localizations`.

    `frame`, `offset`, flux and background use camera units (ADU). Local
    dispersion scales uncertainties and fit comparisons. `sigma` is the
    reference PSF width in pixels; `slack` sets optimization bounds on fitted
    widths as multiples of `sigma`. Boundary solutions are flagged.

    Every selected emitter is returned with `FitFlag` diagnostics. There are
    no brightness or width cuts after fitting; apply those downstream if needed.

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

    `info['fisher_fraction']` is an (N, 4) array for `(flux, y, x, sigma)`.
    Each entry is conditional/marginal Fisher variance: small values mean
    strong coupling to other fitted parameters, not necessarily poor absolute
    precision. NaN means covariance unavailable. These are diagnostics only.
    """
    raw = np.ascontiguousarray(frame, dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {raw.shape}")
    kw = _kw(sigma, offset, roi, raw.shape, k_max, threshold, slack,
             selection, count_penalty)
    return _result(_rs.box_localize(raw, **kw), raw, kw, images)


def localize_stack(stack, sigma, *, offset=0.0, roi=None, k_max=K_MAX,
                   threshold=None, slack=SLACK, n_threads=None,
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
             selection, count_penalty)
    outs = _rs.box_localize_stack(
        raw, **kw, n_threads=_positive_int((os.cpu_count() or 1) if n_threads is None else n_threads,
                                "n_threads"))
    return [_result(o, raw[t], kw, images) for t, o in enumerate(outs)]
