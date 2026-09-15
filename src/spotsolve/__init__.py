"""spotsolve -- multi-emitter spot detection and localization.

The question this package answers is not "where are the spots?" but "how many
are there, and where?" -- the two are one estimation problem, and a detector
that answers the first with a fixed threshold cannot answer the second when
emitters overlap. In each small box of the frame an emitter exists iff it
lowers the box's Poisson deviance by a fixed number of nats, in units of the
noise measured from the frame itself, and the frame is fitted jointly with
every emitter at its own width. It runs in Rust.

    import spotsolve

    locs = spotsolve.localize(frame, sigma=1.27, offset=100.0)
    locs.positions      # (N, 2) float (y, x), pixels
    locs.amplitudes     # (N,) total flux, ADU above the offset
    locs.se             # (N, 3) SE of (flux, y, x)
    locs.fit_sigma      # (N,) each emitter's own fitted width
    locs.rejects        # fits outside the reporting band, with a reason

    movie = spotsolve.localize_stack(stack, sigma=1.27, offset=100.0)

`sigma` and `offset` are the only calibration; no gain or read noise is
needed. `calibrate_sigma` measures `sigma` from the data.

Linking is the movie-level half, and it reads that table rather than the
detector's objects:

    tracks = spotsolve.link(locs)     # the table plus a `track_id` column

    native        localize, localize_stack        the detector
    results       Localizations                   what it returns
    calibration   calibrate_sigma                 the in-focus PSF width
    loctable      results as `polars` tables
    tracking      link, fit_link_params           trajectories from the table
    aggregates    over-bright spots, flagged after the fact
    audit metrics simulate psf                    is the answer any good?
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
