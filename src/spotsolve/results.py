"""What the detector returns: `Localizations`, one per frame."""

from dataclasses import dataclass, field

import numpy as np

from . import psf

__all__ = ["REJECT_DTYPE", "Localizations"]


REJECT_DTYPE = np.dtype([
    ("y", np.float64),
    ("x", np.float64),
    ("flux", np.float64),
    ("sigma", np.float64),
    ("sigma_ratio", np.float64),
    ("reason", "U10"),
])
"""Width-rejected fits, retained in the model. Reasons: `too_narrow`,
`too_wide`, or `edge` for an out-of-band fit near the frame border.
"""


@dataclass(frozen=True)
class Localizations:
    """One frame's detections.

    Every array over detections is aligned: row `k` of `positions`, `amplitudes`,
    `se`, `fit_sigma` and `sigma_se` is one emitter. Fluxes and the background
    are in ADU above the camera offset; positions are pixels, `(y, x)`, with
    pixel centres at integers. Divide fluxes by the camera gain for
    photoelectrons.
    """

    positions: np.ndarray
    """(N, 2) float `(y, x)`."""
    amplitudes: np.ndarray
    """(N,) total flux, ADU above the offset. Aguet converts sampled-Gaussian
    peak amplitude to continuous flux, 2*pi*peak*fit_sigma**2.
    """
    se: np.ndarray
    """(N, 3) standard errors of `(flux, y, x)`. Multi-emitter fits use Fisher
    information scaled by local dispersion; Aguet uses the observed Hessian
    and includes amplitude-width covariance in flux uncertainty.
    """
    fit_sigma: np.ndarray
    """(N,) each emitter's own fitted width, px."""
    sigma_se: np.ndarray
    """(N,) standard error of `fit_sigma`, px."""
    rejects: np.ndarray
    """Fits outside the reporting band, as `REJECT_DTYPE` rows. Empty for Aguet;
    its failed fits are recorded in `info['failures']`.
    """
    background: np.ndarray
    """(H, W) background, ADU per pixel above the offset. Aguet returns the
    diagnostic screening estimate, with NaNs outside the processed crop.
    """
    sigma: float
    """The in-focus PSF width the search was run at, px."""
    dispersion: float
    """The frame's measured pixel variance per unit of signal, ADU: about the
    camera gain plus `gain^2 * read_noise^2 / background`. Unavailable (NaN)
    for the Aguet baseline.
    """
    info: dict = field(default_factory=dict)
    """Method-specific work counts, settings and fit diagnostics."""
    model_image: np.ndarray = None
    """(H, W) diagnostic model when requested. Multi-emitter rendering includes
    width-rejected fits; Aguet renders accepted fits on its screening background.
    """
    residual: np.ndarray = None
    """(H, W) data minus offset minus `model_image`, ADU, when requested."""

    @property
    def sigma_ratio(self):
        """`fit_sigma / sigma`: 1.0 on in-focus spots when `sigma` is right."""
        return self.fit_sigma / self.sigma

    @property
    def peak(self):
        """(N,) model peak signal above background, in ADU per pixel.

        Uses `flux * peak_factor(fit_sigma)` for the integrated PSF and
        `flux / (2*pi*fit_sigma**2)` for Aguet's sampled Gaussian. The integrated
        value assumes an emitter centered on a pixel; subpixel placement lowers
        the brightest observed pixel. Peak depends on fitted width as well as
        flux, so it is useful for image inspection but not a substitute for flux
        or its uncertainty.
        """
        amp = np.asarray(self.amplitudes, float)
        sig = np.asarray(self.fit_sigma, float)
        if self.info.get("method") == "aguet":
            return amp / (2.0 * np.pi * sig ** 2)
        return amp * psf.peak_factor(sig)

    def __len__(self):
        return len(self.amplitudes)
