"""What the detector returns: `Localizations`, one per frame."""

from dataclasses import dataclass, field
from enum import IntFlag

import numpy as np

from . import psf

__all__ = ["FitFlag", "Localizations"]


class FitFlag(IntFlag):
    """Combinable diagnostics, not brightness or width acceptance criteria.

    Zero means no reported issue; it does not certify the model is correct.
    EDGE means three fitted sigmas cross the image boundary, not that a
    Gaussian (which has infinite tails) is fully contained otherwise.
    """

    OK = 0
    EDGE = 1
    NOT_CONVERGED = 2
    STALLED = 4
    COVARIANCE_UNAVAILABLE = 8
    AT_BOUND = 16


@dataclass(frozen=True)
class Localizations:
    """One frame's detections.

    Every array over detections is aligned: row `k` of `positions`, `amplitudes`,
    `se`, `fit_sigma`, `sigma_se` and `flags` is one emitter. Fluxes and the background
    are in ADU above the camera offset; positions are pixels, `(y, x)`, with
    pixel centres at integers. Divide fluxes by the camera gain for
    photoelectrons.
    """

    positions: np.ndarray
    """(N, 2) float `(y, x)`."""
    amplitudes: np.ndarray
    """(N,) total flux of the pixel-integrated Gaussian, ADU above the
    offset and the fitted local level.
    """
    se: np.ndarray
    """(N, 3) standard errors of `(flux, y, x)`, from the expected Fisher
    information of the emitter's final window fit (every component and the
    level free), scaled by the frame's dispersion. NaN without a covariance.
    """
    fit_sigma: np.ndarray
    """(N,) each emitter's own fitted width, px; `sigma` when widths are
    fixed."""
    sigma_se: np.ndarray
    """(N,) standard error of `fit_sigma`, px; NaN when widths are fixed."""
    flags: np.ndarray
    """(N,) uint8 bitmask of `FitFlag` diagnostics; all rows are retained."""
    background: np.ndarray
    """(H, W) screening level, ADU per pixel above the offset: the constant of
    the window regression at each pixel, NaN outside the processed crop. Each
    emitter's own fitted level is `info["fitted_background"]`.
    """
    sigma: float
    """Reference width used for candidate searching and fit initialization, px."""
    dispersion: float
    """The frame's measured pixel variance per unit of signal, ADU: about the
    camera gain plus `gain^2 * read_noise^2 / background`.
    """
    info: dict = field(default_factory=dict)
    """Method-specific work counts, settings and fit diagnostics."""
    model_image: np.ndarray = None
    """(H, W) diagnostic model when requested: the reported emitters on the
    screening level.
    """
    residual: np.ndarray = None
    """(H, W) data minus offset minus `model_image`, ADU, when requested."""

    @property
    def sigma_ratio(self):
        """Fitted width relative to the reference PSF width: `fit_sigma / sigma`."""
        return self.fit_sigma / self.sigma

    @property
    def peak(self):
        """(N,) model peak signal above background, in ADU per pixel.

        `flux * peak_factor(fit_sigma)` for the integrated PSF. It assumes an
        emitter centered on a pixel; subpixel placement lowers
        the brightest observed pixel. Peak depends on fitted width as well as
        flux, so it is useful for image inspection but not a substitute for flux
        or its uncertainty.
        """
        amp = np.asarray(self.amplitudes, float)
        sig = np.asarray(self.fit_sigma, float)
        return amp * psf.peak_factor(sig)

    def __len__(self):
        return len(self.amplitudes)
