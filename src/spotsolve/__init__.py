"""Emitter localization and Brownian-motion LAP tracking for microscopy.

`localize` / `localize_stack` fit each frame as one Poisson model whose
one knob, `fp_per_mpx`, is the expected false positives per 10^6 noise
pixels. `localize_aguet` /
`localize_aguet_stack` provide the independent-fit spotfitlm sparse baseline.
Both return `Localizations`; stack functions process frames in native workers.

    locs = localize(frame, sigma=1.45, offset=100)
    sparse = localize_aguet(frame, sigma=1.45, offset=100)

Positions are (y, x) pixels, amplitudes are flux above the offset, and `se`
columns are (flux, y, x). Noise models and diagnostics differ by detector;
see docs/DETECTION.md and docs/AGUET_BASELINE.md.

`loctable` converts results to tables; `link` joins their rows into tracks
frame to frame, by least squared displacement within a search radius.
`FitFlag` describes numerical and geometric fit issues.
"""

from .native import (  # noqa: F401
    FP_PER_MPX,
    SLACK,
    localize,
    localize_stack,
)
from .results import FitFlag, Localizations  # noqa: F401
from .aguet import localize_aguet, localize_aguet_stack  # noqa: F401
from .tracking import link  # noqa: F401

__version__ = "0.2.0"

__all__ = [
    "localize",
    "localize_stack",
    "localize_aguet",
    "localize_aguet_stack",
    "Localizations",
    "FitFlag",
    "link",
    "SLACK",
    "FP_PER_MPX",
    "__version__",
]
