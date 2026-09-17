"""Spot detection, localization and tracking for fluorescence microscopy.

`localize` / `localize_stack` jointly fit emitters with fixed-cost selection
(default) or experimental `selection="bic"`. `localize_aguet` /
`localize_aguet_stack` provide the independent-fit spotfitlm sparse baseline.
Both return `Localizations`; stack functions process frames in native workers.

    locs = localize(frame, sigma=1.45, offset=100, selection="bic", count_penalty=2)
    sparse = localize_aguet(frame, sigma=1.45, offset=100)

Positions are (y, x) pixels, amplitudes are flux above the offset, and `se`
columns are (flux, y, x). Noise models and diagnostics differ by detector;
see docs/DETECTION.md, docs/COUNT_SELECTION.md and docs/AGUET_BASELINE.md.

`calibrate_sigma` estimates the PSF width with the multi-emitter detector.
`loctable` converts results to tables; `link` and `fit_link_params` operate
on those tables. `flag_aggregates` supplies optional brightness-based flags.
"""

from .native import (  # noqa: F401
    BAND,
    BAND_Z,
    K_MAX,
    PEAK_Z,
    SLACK,
    localize,
    localize_stack,
)
from .results import REJECT_DTYPE, Localizations  # noqa: F401
from .aguet import localize_aguet, localize_aguet_stack  # noqa: F401
from .calibration import SigmaCalibration, calibrate_sigma  # noqa: F401
from .tracking import LinkParams, fit_link_params, link  # noqa: F401
from .aggregates import (  # noqa: F401
    AGG_AMP_RATIO,
    AGG_LINK,
    aggregate_report,
    flag_aggregates,
)

__version__ = "0.1.0"

__all__ = [
    "localize",
    "localize_stack",
    "localize_aguet",
    "localize_aguet_stack",
    "Localizations",
    "REJECT_DTYPE",
    "calibrate_sigma",
    "SigmaCalibration",
    "link",
    "fit_link_params",
    "LinkParams",
    "SLACK",
    "BAND",
    "BAND_Z",
    "K_MAX",
    "PEAK_Z",
    "flag_aggregates",
    "aggregate_report",
    "AGG_AMP_RATIO",
    "AGG_LINK",
    "__version__",
]
