"""Emitter localization and Brownian-motion LAP tracking for microscopy.

`localize` / `localize_stack` follow u-track's pointSourceDetection: a
Poisson significance screen, one window fit per seed, and a likelihood-ratio
decision whose one knob, `fp_per_mpx`, is the expected false positives per
10^6 noise pixels. `fit_mixtures=True` fits overlapping spots jointly.
Each returns `Localizations`; stack functions process frames in native workers.

    locs = localize(frame, sigma=1.45, offset=100)
    dense = localize(frame, sigma=1.45, offset=100, fit_mixtures=True)

Positions are (y, x) pixels, amplitudes are flux above the offset, and `se`
columns are (flux, y, x). See docs/DETECTION.md.

`loctable` converts results to tables; `link` joins their rows into tracks
frame to frame, by least squared displacement within a search radius.
`FitFlag` describes numerical and geometric fit issues.
"""

from .native import (  # noqa: F401
    FP_PER_MPX,
    MAX_MIXTURES,
    WIDTH,
    localize,
    localize_stack,
)
from .results import FitFlag, Localizations  # noqa: F401
from .tracking import link  # noqa: F401

__version__ = "0.5.0"

__all__ = [
    "localize",
    "localize_stack",
    "Localizations",
    "FitFlag",
    "link",
    "FP_PER_MPX",
    "MAX_MIXTURES",
    "WIDTH",
    "__version__",
]
