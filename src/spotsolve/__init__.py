"""spotsolve -- multi-emitter spot detection and localization.

The question this package answers is not "where are the spots?" but "how many
are there, and where?" -- the two are one estimation problem, and a detector
that answers the first with a fixed threshold cannot answer the second when
emitters overlap. In each small box of the frame, an emitter exists iff it
lowers the box's Poisson deviance by `box.ADD_NATS` nats, and the whole frame
is fitted jointly with every emitter at its own width.

    import spotsolve

    result = spotsolve.localize(frame, sigma=1.27, gain=2.0, offset=100.0,
                                read_noise=1.6)
    result.positions      # (N, 2) float (y, x), pixels
    result.amplitudes     # (N,) total flux, photoelectrons
    result.se             # (N, 3) CRLB: SE of (flux, y, x)
    result.fit_sigma      # (N,) each emitter's own fitted width
    result.width_rejects  # fits outside the reporting band, with a reason

    movie = spotsolve.localize_stack(stack, sigma=1.27, gain=2.0,
                                     offset=100.0, read_noise=1.6)

The detector runs entirely in the `spotsolve_rs` Rust extension, and
`localize_stack` spreads a timecourse's frames over native threads.
`box.localize_boxes` is its Python reference, where the measurement behind
every constant lives. `sigma`, `gain`, `offset` and `read_noise` are the
caller's calibration; `gain=None` estimates it from the frame.

`localize_sparse` is the independent-source alternative: one Rust candidate
pass and one bounded fit per peak, with fixed or fitted width, and no model
selection between overlapping sources.

The other modules are the layers these are built from, imported directly
when needed:

    psf lmga patches calibrate core box    the model, the fit, the reference
    backend                                Python / Rust fitter dispatch
    audit metrics simulate                 is the answer any good?
    loctable                               results as `polars` tables
"""

from .native import localize, localize_stack  # noqa: F401
from .sparse import SparseResult, localize_sparse  # noqa: F401
from .core import (  # noqa: F401
    refine,
    background_map,
    find_candidates,
    flag_aggregates,
    aggregate_report,
    log_kernel_l2,
)
from .structs import (  # noqa: F401
    WIDTH_REJECT_DTYPE,
    DetectResult,
    FitResult,
    Patch,
)

# The constants the detector shares with its reference; `native` asserts on
# every call that the Rust copies still agree.
from .core import (  # noqa: F401
    LINK_FACTOR,
    HALO_FACTOR,
    BBOX_PAD,
    SIGMA_SLACK,
    FOCUS_BAND,
    BG_KERNEL,
    BG_FLOOR,
    BG_MASK_RADIUS,
    BG_MIN_PIXELS,
    REFINE_MAX_ITER,
    REFINE_SWEEPS,
    REFINE_TOL,
    REFINE_TOL_OBJ,
    AGG_AMP_RATIO,
    AGG_LINK,
)
from .calibrate import SEED_ALPHA  # noqa: F401


__version__ = "0.1.0"

__all__ = [
    "localize",
    "localize_stack",
    "localize_sparse",
    "SparseResult",
    "refine",
    "background_map",
    "find_candidates",
    "flag_aggregates",
    "aggregate_report",
    "log_kernel_l2",
    "DetectResult",
    "FitResult",
    "Patch",
    "WIDTH_REJECT_DTYPE",
    "LINK_FACTOR",
    "HALO_FACTOR",
    "BBOX_PAD",
    "SEED_ALPHA",
    "SIGMA_SLACK",
    "FOCUS_BAND",
    "BG_KERNEL",
    "BG_FLOOR",
    "BG_MASK_RADIUS",
    "BG_MIN_PIXELS",
    "REFINE_MAX_ITER",
    "REFINE_SWEEPS",
    "REFINE_TOL",
    "REFINE_TOL_OBJ",
    "AGG_AMP_RATIO",
    "AGG_LINK",
    "__version__",
]
