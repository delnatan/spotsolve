"""Emitter localization and Brownian-motion LAP tracking for microscopy.

`localize` / `localize_stack` follow u-track's pointSourceDetection: a
significance screen on the camera's noise, one quasi-likelihood window fit per seed, and a decision whose
one knob, `p_real`, is the least probability that a reported emitter is real
rather than noise, fitted per movie. The camera's `gain` is an input;
`estimate_gain` measures it from a movie. `fit_mixtures=True` fits overlapping spots jointly.
Each returns `Localizations`; stack functions process frames in native workers.

    gain = estimate_gain(movie, sigma=1.45, offset=100)
    locs = localize(frame, sigma=1.45, gain=gain, offset=100)
    dense = localize(frame, sigma=1.45, gain=gain, offset=100, fit_mixtures=True)

Positions are (y, x) pixels, amplitudes are flux above the offset, and `se`
columns are (flux, y, x). See docs/DETECTION.md.

`loctable` converts results to tables; `link` joins their rows into tracks
frame to frame, by least squared displacement within a search radius.
`FitFlag` describes numerical and geometric fit issues.
"""

from .native import (  # noqa: F401
    P_REAL,
    MAX_MIXTURES,
    WIDTH,
    Cancelled,
    Progress,
    estimate_gain,
    localize,
    localize_stack,
)
from .results import FitFlag, Localizations  # noqa: F401
from .tracking import link  # noqa: F401

__version__ = "0.5.3"

__all__ = [
    "localize",
    "localize_stack",
    "estimate_gain",
    "Progress",
    "Cancelled",
    "Localizations",
    "FitFlag",
    "link",
    "P_REAL",
    "MAX_MIXTURES",
    "WIDTH",
    "__version__",
]
