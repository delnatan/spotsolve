"""What the detector returns: `Localizations`, one per frame."""

from dataclasses import dataclass, field

import numpy as np

__all__ = ["REJECT_DTYPE", "Localizations"]


REJECT_DTYPE = np.dtype([
    ("y", np.float64),
    ("x", np.float64),
    ("flux", np.float64),
    ("sigma", np.float64),
    ("sigma_ratio", np.float64),
    ("reason", "U10"),
])
"""One fit the reporting band does not accept. `reason` is `"too_narrow"` or
`"too_wide"` for an interior fit outside the band, or `"edge"` for an
out-of-band fit the frame border cuts -- not an interior width measurement at
all. These objects are modelled to the end; they are only not reported as
detections."""


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
    peak amplitude to continuous flux, 2*pi*peak*fit_sigma**2."""
    se: np.ndarray
    """(N, 3) standard errors of `(flux, y, x)`. Multi-emitter fits use Fisher
    information scaled by local dispersion; Aguet uses the observed Hessian
    and includes amplitude-width covariance in flux uncertainty."""
    fit_sigma: np.ndarray
    """(N,) each emitter's own fitted width, px."""
    sigma_se: np.ndarray
    """(N,) standard error of `fit_sigma`, px."""
    rejects: np.ndarray
    """Fits outside the reporting band, as `REJECT_DTYPE` rows. Empty for Aguet;
    its failed fits are recorded in `info['failures']`."""
    background: np.ndarray
    """(H, W) background, ADU per pixel above the offset. Aguet returns the
    diagnostic screening estimate, with NaNs outside the processed crop."""
    sigma: float
    """The in-focus PSF width the search was run at, px."""
    dispersion: float
    """The frame's measured pixel variance per unit of signal, ADU: about the
    camera gain plus `gain^2 * read_noise^2 / background`. Unavailable (NaN)
    for the Aguet baseline."""
    info: dict = field(default_factory=dict)
    """Method-specific work counts, settings and fit diagnostics."""
    model_image: np.ndarray = None
    """(H, W) diagnostic model when requested. Multi-emitter rendering includes
    width-rejected fits; Aguet renders accepted fits on its screening background."""
    residual: np.ndarray = None
    """(H, W) data minus offset minus `model_image`, ADU, when requested."""

    @property
    def sigma_ratio(self):
        """`fit_sigma / sigma`: 1.0 on in-focus spots when `sigma` is right."""
        return self.fit_sigma / self.sigma

    def __len__(self):
        return len(self.amplitudes)
