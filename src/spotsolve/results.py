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
    `se` and `fit_sigma` is one emitter. Fluxes and the background are in
    photoelectrons; positions are pixels, `(y, x)`, with pixel centres at
    integers.
    """

    positions: np.ndarray
    """(N, 2) float `(y, x)`."""
    amplitudes: np.ndarray
    """(N,) total flux, photoelectrons."""
    se: np.ndarray
    """(N, 3) standard errors of `(flux, y, x)`, from the Fisher information of
    the fit that produced them; NaN where it was singular."""
    fit_sigma: np.ndarray
    """(N,) each emitter's own fitted width, px."""
    rejects: np.ndarray
    """Fits outside the reporting band, as `REJECT_DTYPE` rows."""
    background: np.ndarray
    """(H, W) background surface, photoelectrons per pixel."""
    sigma: float
    """The in-focus PSF width the search was run at, px."""
    gain: float
    """ADU per photoelectron: the caller's, or the frame's estimate."""
    read_noise: float
    """Read noise the likelihood assumed, electrons rms."""
    info: dict = field(default_factory=dict)
    """Work done: candidates, boxes, search and polish fits."""
    model_image: np.ndarray = None
    """(H, W) background plus every fitted emitter, when requested."""
    residual: np.ndarray = None
    """(H, W) data minus `model_image`, photoelectrons, when requested."""

    @property
    def sigma_ratio(self):
        """`fit_sigma / sigma`: 1.0 on in-focus spots when `sigma` is right."""
        return self.fit_sigma / self.sigma

    def __len__(self):
        return len(self.amplitudes)
