"""Frozen Python reference count search, independent of the native algorithm."""
from time import perf_counter
import numpy as np
from .._numerics import _weighted_geometry
from .types import FitOptions
from ..types import ComponentFits, FLUX_UNIT
from .profiled import fit_profiled_count
from ..model import FocusedModel


def fit_component(data, model, *, candidate_centres=(), options=None):
    """Frozen Python K=0/1/2 reference; production fitting lives in Rust."""
    if not isinstance(model, FocusedModel):
        raise TypeError("reference fitting requires a FocusedModel")
    return _fit_counts(data, model, fit_profiled_count,
                       candidate_centres=candidate_centres, options=options)


def _fit_counts(data, model, count_fitter, *, candidate_centres=(), options=None,
                dense_starts=False, focus_rate=0.):
    """Shared start machinery; extra arguments serve historical comparisons."""
    data = np.asarray(data, dtype=float)
    if data.shape != model.shape or np.any(~np.isfinite(data)) or np.any(data < 0):
        raise ValueError("data must match model.shape and be finite/nonnegative")
    if not np.isfinite(focus_rate) or focus_rate < 0:
        raise ValueError("focus_rate must be finite and nonnegative")
    options = (FitOptions(max_iter=400, screen_iter=48, keep_screened=4)
               if options is None else options)
    if options.max_iter < 1 or options.screen_iter < 1 or options.keep_screened < 1:
        raise ValueError("optimization budgets must be positive")
    seeds = np.asarray(candidate_centres, dtype=float)
    if seeds.size and (seeds.ndim != 2 or seeds.shape[1] != 2
                       or np.any(~np.isfinite(seeds))):
        raise ValueError("candidate_centres must be finite (y,x) pairs")
    started = perf_counter()
    y0, x0, y1, x1 = model.focus_bounds
    clipped = lambda point: np.clip(point, [y0, x0], [y1, x1])
    allowed = ((model.grid.yy >= y0) & (model.grid.yy <= y1)
               & (model.grid.xx >= x0) & (model.grid.xx <= x1))
    if not np.any(allowed):
        raise ValueError("focus_bounds must contain at least one pixel center")
    peak = np.array(np.unravel_index(
        np.argmax(np.where(allowed, data, -np.inf)), data.shape), dtype=float)
    b0 = max(float(np.percentile(data, 20)), 1e-3)
    moment, axis, excess = _weighted_geometry(data, b0)
    midpoint = np.array([(data.shape[0] - 1) / 2, (data.shape[1] - 1) / 2])
    nuisance_centres = np.unique(np.stack([midpoint, moment, peak]), axis=0)
    nuisance_starts = model.nuisance_starts(b0, excess, nuisance_centres, midpoint)
    zero = count_fitter(data, model, 0, nuisance_starts, options, focus_rate)

    points = np.unique(np.stack([clipped(peak), clipped(moment),
                                *[clipped(point) for point in seeds]]), axis=0)
    focus_flux = max(float(data.max()) - b0, 1) * 2 * np.pi * model.seed_sigma**2
    single_starts = [np.r_[zero.theta, 0, clipped(peak)]]
    for point in points:
        single_starts.append(np.r_[zero.theta, focus_flux / FLUX_UNIT, point])
        single_starts.append(np.r_[nuisance_starts[-1], focus_flux / FLUX_UNIT, point])
    one = count_fitter(data, model, 1, single_starts, options, focus_rate)

    centre = one.positions[0]
    offset = model.nuisance_size
    total = one.theta[offset]
    pair_starts = [np.r_[one.theta, 0, centre]]
    separations = ((0.5, 1.0, 1.75) if not dense_starts
                   else (0.25, 0.5, 0.75, 1, 1.5, 1.75, 2.0))
    angles = axis + np.arange(4 if not dense_starts else 8) * np.pi / (4 if not dense_starts else 8)
    for q in (0.5, 0.8):
        for angle in angles:
            for polarity in ((1,) if q == 0.5 else (1, -1)):
                direction = polarity * np.array([np.sin(angle), np.cos(angle)])
                for separation in separations:
                    vector = separation * model.seed_sigma * direction
                    first = clipped(centre + (1 - q) * vector)
                    second = clipped(centre - q * vector)
                    pair_starts.append(np.r_[one.theta[:offset], total * q, first,
                                             total * (1 - q), second])
    for left, point in enumerate(points):
        for other in points[left + 1:]:
            pair_starts.append(np.r_[one.theta[:offset], total / 2, point,
                                     total / 2, other])
    two = count_fitter(data, model, 2, pair_starts, options, focus_rate)
    return ComponentFits(model, (zero, one, two), perf_counter() - started, focus_rate)
