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
    result.se           # (N, 2) reported standard errors, pixels

The pipeline itself lives in `spotsolve.core`; the names re-exported here are
its public surface. The other modules are the layers it is built from, and are
imported directly when you need them:

    psf lmga evidence patches moves calibrate   the model and the fit
    backend                                     Python / Rust pass dispatch
    audit metrics simulate                      is the answer any good?
    loctable                                    results as `polars` tables
    aguet                                       the reference detector

Pass `impl="rs"` to `detect` to run the four inner passes in the `spotsolve_rs`
Rust extension instead of the Python reference; `backend.available()` says
whether it is installed here. Results are identical either way.
"""

from .core import (  # noqa: F401
    detect,
    refine,
    render,
    background_map,
    find_candidates,
    find_aggregates,
    flag_aggregates,
    aggregate_report,
    render_aggregates,
    log_kernel_l2,
)
from .structs import DetectResult, FitResult, Patch  # noqa: F401

# Tuning constants. These are the pipeline's dials and are part of the public
# surface: `PRUNE_TAU` is its only precision/recall knob, and the Rust backend
# asserts on load that its copies still agree with these.
from .core import (  # noqa: F401
    LINK_FACTOR,
    HALO_FACTOR,
    BBOX_PAD,
    CAND_THRESHOLD,
    PRUNE_TAU,
    SPLIT_DISPS,
    BG_KERNEL,
    BG_FLOOR,
    BG_MASK_RADIUS,
    BG_MIN_PIXELS,
    REFINE_MAX_ITER,
    REFINE_SWEEPS,
    REFINE_TOL,
    REFINE_TOL_OBJ,
    EVIDENCE_TOL_OBJ,
    AGG_FLUX_RATIO,
    AGG_SIGMA_LO,
    AGG_SIGMA_HI,
    AGG_MASK_RADIUS,
    AGG_AMP_RATIO,
    AGG_LINK,
)


__version__ = "0.1.0"

__all__ = [
    "detect",
    "refine",
    "render",
    "background_map",
    "find_candidates",
    "find_aggregates",
    "flag_aggregates",
    "aggregate_report",
    "render_aggregates",
    "log_kernel_l2",
    "DetectResult",
    "FitResult",
    "Patch",
    "LINK_FACTOR",
    "HALO_FACTOR",
    "BBOX_PAD",
    "CAND_THRESHOLD",
    "PRUNE_TAU",
    "SPLIT_DISPS",
    "BG_KERNEL",
    "BG_FLOOR",
    "BG_MASK_RADIUS",
    "BG_MIN_PIXELS",
    "REFINE_MAX_ITER",
    "REFINE_SWEEPS",
    "REFINE_TOL",
    "REFINE_TOL_OBJ",
    "EVIDENCE_TOL_OBJ",
    "AGG_FLUX_RATIO",
    "AGG_SIGMA_LO",
    "AGG_SIGMA_HI",
    "AGG_MASK_RADIUS",
    "AGG_AMP_RATIO",
    "AGG_LINK",
    "__version__",
]
