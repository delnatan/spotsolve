"""Thin array/result adapters for the native calibrated inference algorithm."""
from time import perf_counter
import numpy as np

from .model import FocusedModel
from .types import FitOptions, CountFit, ComponentFits, PositionUncertainty


def prepare_rust(model):
    """Copy calibration once into a reusable native model and workspace.

    The native fit_component method owns the complete K=0/1/2 search and
    releases the GIL for its duration. Separate prepared instances may run
    concurrently. Python only preprocesses calibration and adapts arrays.
    """
    if not isinstance(model, FocusedModel):
        raise TypeError("prepare_rust requires a FocusedModel")
    import spotsolve_rs
    if not hasattr(getattr(spotsolve_rs, "CalibratedModel", None), "fit_component"):
        raise RuntimeError("rebuild spotsolve_rs: this extension lacks calibrated component inference")
    return spotsolve_rs.CalibratedModel(
        np.ascontiguousarray(model.psf.depth_um), np.ascontiguousarray(model.psf.offsets_px),
        np.ascontiguousarray(model.psf.coefficients), model.shape,
        model.focus_bounds, model.defocus_bounds_um,
    )


def fit_component(data, model, *, candidate_centres=(), options=None,
                  proposal_method="moments", proposal_alpha=0.05):
    """Fit K=0/1/2 in Rust; return hypotheses, not existence probabilities.

    Experimental proposal_method="aguet" replaces moment/peak focused starts
    with native Aguet maxima. Nested starts, splits and broad restarts remain.
    proposal_alpha controls proposals only, never dense source acceptance.
    """
    started = perf_counter()
    native = prepare_rust(model)
    options = FitOptions() if options is None else options
    centres = np.ascontiguousarray(candidate_centres, dtype=float)
    if centres.shape == (0,):
        centres = centres.reshape(0, 2)
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError("candidate_centres must be finite (y,x) pairs")
    records = native.fit_component(np.ascontiguousarray(data, dtype=float), centres,
        seed_sigma=model.seed_sigma, max_iter=options.max_iter,
        screen_iter=options.screen_iter, keep_screened=options.keep_screened, gtol=options.gtol,
        proposal_method=proposal_method, proposal_alpha=proposal_alpha)
    fits = tuple(CountFit(count, record["theta"], record["objective"],
        record["status"] == "converged", record["kkt"], tuple(record["boundary"]),
        record["evaluations"], record["starts"], 20, 0.,
        record["inner_iterations"], record["inner_failures"])
        for count, record in enumerate(records))
    return ComponentFits(model, fits, perf_counter()-started)


def position_uncertainty(data, model, fit):
    """Native conditional observed-Hessian covariance with explicit validity status."""
    native = prepare_rust(model)
    data = np.ascontiguousarray(data, dtype=float)
    lo, hi = native.parameter_bounds(data, fit.n_focus)
    result = native.position_uncertainty(data, np.ascontiguousarray(fit.theta, dtype=float), lo, hi)
    return PositionUncertainty(result["covariance"], result["status"],
        result["nuisance_fixed_covariance"], result["broad_conditioned_absent"])
