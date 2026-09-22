"""Emitter localization and Brownian-motion LAP tracking for microscopy.

`localize` / `localize_stack` jointly fit emitters with fixed-cost selection
(default) or experimental `selection="bic"`. `localize_aguet` /
`localize_aguet_stack` provide the independent-fit spotfitlm sparse baseline.
Both return `Localizations`; stack functions process frames in native workers.

    locs = localize(frame, sigma=1.45, offset=100, selection="bic", count_penalty=2)
    sparse = localize_aguet(frame, sigma=1.45, offset=100)

Positions are (y, x) pixels, amplitudes are flux above the offset, and `se`
columns are (flux, y, x). Noise models and diagnostics differ by detector;
see docs/DETECTION.md, docs/COUNT_SELECTION.md and docs/AGUET_BASELINE.md.

`loctable` converts results to tables; `link` and `fit_link_params` operate
on those tables using frame-to-frame linear assignment and Brownian motion.
`FitFlag` describes numerical and geometric fit issues.
"""

from .native import (  # noqa: F401
    K_MAX,
    PEAK_Z,
    SLACK,
    localize,
    localize_stack,
)
from .results import FitFlag, Localizations  # noqa: F401
from .aguet import localize_aguet, localize_aguet_stack  # noqa: F401
from .tracking import LinkParams, fit_link_params, link  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "localize",
    "localize_stack",
    "localize_aguet",
    "localize_aguet_stack",
    "Localizations",
    "FitFlag",
    "link",
    "fit_link_params",
    "LinkParams",
    "SLACK",
    "K_MAX",
    "PEAK_Z",
    "__version__",
]
