"""spotsolve -- multi-emitter spot detection and localization.

The question this package answers is not "where are the spots?" but "how many
are there, and where?" -- the two are one estimation problem, and a detector
that answers the first with a fixed threshold cannot answer the second when
emitters overlap. `detect` decides N by a Laplace Bayes factor evaluated on a
bounded Poisson MLE, so every emitter in the returned list has paid for itself
in evidence.

    import spotsolve

    result = spotsolve.detect(image, sigma=1.45, gain=2.401, offset=100.0)
    result.positions    # (N, 2) float (y, x), pixels
    result.amplitudes   # (N,) total flux, photoelectrons
    result.se           # (N, 3) CRLB: SE of (flux, y, x)
    result.fit_sigma    # (N,) each emitter's own fitted width
    result.aggregates   # objects too wide to be a point source -- see §8b

The pipeline itself lives in `spotsolve.core`; the names re-exported here are
its public surface. The other modules are the layers it is built from, and are
imported directly when you need them:

    psf lmga prior evidence patches moves calibrate   the model and the fit
    backend                                          Python / Rust dispatch
    audit metrics simulate                           is the answer any good?
    loctable                                         results as `polars` tables

`localize_sparse` is the independent-source counterpart: one Rust candidate
pass and one bounded fit per peak, with fixed or fitted width. It intentionally
does not run the dense add/split/prune loop.

Every emitter carries its own width, bounded to `SIGMA_SLACK` and fitted by
MAP under a prior centred on the PSF; one that lands outside `FOCUS_BAND` is
modelled to the end but returned in `aggregates` rather than as a detection.
That is what keeps a defocused source from being tiled into several spurious
in-focus ones -- README section 8b.

Pass `impl="rs"` to `detect` to run the four inner passes in the `spotsolve_rs`
Rust extension instead of the Python reference; `backend.available()` says
whether it is installed here. The Rust core implements the FIXED-width layout,
so `impl` is ignored unless `slack=None`.
"""

from .core import (  # noqa: F401
    detect,
    refine,
    render,
    background_map,
    find_candidates,
    flag_aggregates,
    aggregate_report,
    log_kernel_l2,
)
from .prior import (  # noqa: F401
    FluxPrior,
    ExponentialFlux,
    WidthPrior,
    UniformWidth,
    FocusMixtureWidth,
    FOCUS_WIDTH_GAMMA,
)
from .structs import (  # noqa: F401
    WIDTH_REJECT_DTYPE,
    DetectResult,
    FitResult,
    Patch,
)
from .sparse import SparseResult, localize_sparse  # noqa: F401

# Tuning constants. These are the pipeline's dials and are part of the public
# surface: `PRUNE_TAU` is its only precision/recall knob, and the Rust backend
# asserts on load that its copies still agree with these.
from .core import (  # noqa: F401
    LINK_FACTOR,
    HALO_FACTOR,
    BBOX_PAD,
    CAND_THRESHOLD,
    SEED_ALPHA,
    PRUNE_TAU,
    SPLIT_DISPS,
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
    EVIDENCE_TOL_OBJ,
    AGG_MASK_RADIUS,
    AGG_AMP_RATIO,
    AGG_LINK,
)


__version__ = "0.1.0"

__all__ = [
    "detect",
    "localize_sparse",
    "SparseResult",
    "refine",
    "render",
    "background_map",
    "find_candidates",
    "flag_aggregates",
    "aggregate_report",
    "log_kernel_l2",
    "DetectResult",
    "FitResult",
    "Patch",
    "LINK_FACTOR",
    "HALO_FACTOR",
    "BBOX_PAD",
    "CAND_THRESHOLD",
    "SEED_ALPHA",
    "PRUNE_TAU",
    "SPLIT_DISPS",
    "SIGMA_SLACK",
    "FOCUS_BAND",
    "WIDTH_REJECT_DTYPE",
    "FluxPrior",
    "ExponentialFlux",
    "WidthPrior",
    "UniformWidth",
    "FocusMixtureWidth",
    "FOCUS_WIDTH_GAMMA",
    "BG_KERNEL",
    "BG_FLOOR",
    "BG_MASK_RADIUS",
    "BG_MIN_PIXELS",
    "REFINE_MAX_ITER",
    "REFINE_SWEEPS",
    "REFINE_TOL",
    "REFINE_TOL_OBJ",
    "EVIDENCE_TOL_OBJ",
    "AGG_MASK_RADIUS",
    "AGG_AMP_RATIO",
    "AGG_LINK",
    "__version__",
]
