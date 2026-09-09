"""Thin Python interface to the native single-pass sparse localizer."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseResult:
    """Independent one-emitter fits from one full-frame candidate pass.

    Standard-error columns are ``(flux, y, x)``. Returned rows have valid
    conditional Fisher uncertainty and pass the Aguet local significance test;
    ``candidate_count`` records how many significant maxima were fitted.
    """

    positions: np.ndarray
    amplitudes: np.ndarray
    se: np.ndarray
    fit_sigma: np.ndarray
    test_statistic: np.ndarray
    p_value: np.ndarray
    status: tuple[str, ...]
    iterations: np.ndarray
    candidate_count: int
    sigma: float
    gain: float
    alpha: float
    background: float
    model_image: np.ndarray
    residual: np.ndarray

    @property
    def sigma_ratio(self):
        return self.fit_sigma / self.sigma


def localize_sparse(
    data_img,
    sigma=1.2,
    *,
    offset=0.0,
    gain=1.0,
    alpha=0.05,
    fit_sigma=False,
    sigma_bounds=(0.7, 2.2),
    fit_radius_sigma=4.0,
    max_iter=100,
):
    """Localize a sparse frame with one candidate pass and independent fits.

    The input is converted to photoelectrons as ``(data_img-offset)/gain``.
    At every pixel, the Aguet detector fits a fixed-width Gaussian plus a local
    constant by linear least squares. It estimates noise from the local
    residuals and tests whether the peak amplitude clears the resulting noise
    floor. Significant pixels are intersected with LoG local maxima, then each
    candidate is fitted once with a pixel-integrated Gaussian PSF.

    ``alpha`` is the local test size and ``p_value`` reports the resulting
    one-sided p-value. It is not a frame-level false-discovery guarantee.
    ``fit_sigma=False`` fixes the fitted width at ``sigma``; ``True`` fits it
    within ``sigma_bounds * sigma`` after candidate selection.

    This method assumes fitting windows do not materially overlap. It has no
    split, add, prune, or neighbor refit pass, making it a useful sparse-field
    reference and a poor choice for interacting emitters.
    """
    raw = np.ascontiguousarray(data_img, dtype=float)
    if raw.ndim != 2 or min(raw.shape, default=0) < 3 or np.any(~np.isfinite(raw)):
        raise ValueError("data_img must be a finite two-dimensional image at least 3x3")
    sigma = float(sigma)
    gain = float(gain)
    offset = float(offset)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(offset):
        raise ValueError("offset must be finite and gain finite and positive")
    photons = np.ascontiguousarray((raw - offset) / gain, dtype=float)
    alpha = float(alpha)
    bounds = tuple(float(value) for value in sigma_bounds)
    if len(bounds) != 2:
        raise ValueError("sigma_bounds must contain lower and upper ratios")

    try:
        import spotsolve_rs
    except ImportError as error:
        raise RuntimeError(
            "localize_sparse requires the Rust extension; build it with "
            "`maturin develop --release -m rust/spotsolve-py/Cargo.toml`"
        ) from error
    if not hasattr(spotsolve_rs, "localize_sparse"):
        raise RuntimeError("rebuild spotsolve_rs: this extension lacks sparse localization")
    result = spotsolve_rs.localize_sparse(
        photons,
        sigma,
        alpha,
        fit_sigma=bool(fit_sigma),
        sigma_bounds=bounds,
        fit_radius_sigma=float(fit_radius_sigma),
        max_iter=int(max_iter),
    )
    return SparseResult(
        positions=result["positions"],
        amplitudes=result["amplitudes"],
        se=result["se"],
        fit_sigma=result["fit_sigma"],
        test_statistic=result["test_statistic"],
        p_value=result["p_value"],
        status=tuple(result["status"]),
        iterations=np.asarray(result["iterations"], dtype=np.uint32),
        candidate_count=int(result["candidate_count"]),
        sigma=sigma,
        gain=gain,
        alpha=alpha,
        background=float(result["background"]),
        model_image=result["model_image"],
        residual=result["residual"],
    )
