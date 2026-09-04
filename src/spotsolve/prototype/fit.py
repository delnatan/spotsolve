"""Likelihood fits and multistart orchestration for prototype hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .models import Hypothesis, ROIGrid, decode, evaluate


@dataclass(frozen=True)
class FitOptions:
    max_iter: int = 250
    screen_iter: int = 24
    keep_screened: int = 4
    ftol: float = 1e-11
    gtol: float = 1e-6
    wide_ratio_bounds: tuple[float, float] = (1.25, 4.0)
    log_slope_bound: float = 1.5
    log_curvature_bound: float = 1.5
    pair_max_separation_ratio: float = 2.5


@dataclass(frozen=True)
class StartGrid:
    """Dimensionless H2 start grid.

    Angles are offsets from the data's dominant second-moment axis.  The
    default is deliberately modest: every start is first screened cheaply and
    only the best few receive a full optimization.
    """

    separation_ratios: tuple[float, ...] = (0.5, 1.0, 1.75)
    flux_ratios: tuple[float, ...] = (1.0, 4.0)
    angle_offsets: tuple[float, ...] = (
        0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)


@dataclass
class LocalFit:
    hypothesis: Hypothesis
    theta: np.ndarray
    objective: float
    converged: bool
    message: str
    n_iter: int
    n_eval: int
    attempts: int
    elapsed_s: float
    at_boundary: bool
    collapsed: bool
    physical: dict = field(default_factory=dict)


@dataclass
class FitCollection:
    fits: dict[Hypothesis, LocalFit]
    elapsed_s: float
    estimated_axis: float
    shape: tuple[int, int]
    sigma: float
    options: FitOptions
    start_grid: StartGrid

    def __getitem__(self, hypothesis):
        return self.fits[Hypothesis(hypothesis)]

    @property
    def simple_objective(self):
        return min(self[Hypothesis.H0].objective,
                   self[Hypothesis.HSMOOTH].objective,
                   self[Hypothesis.H1].objective,
                   self[Hypothesis.HWIDE].objective)

    @property
    def pair_gain(self):
        """Uncalibrated log-likelihood gain of H2 over the best simple model."""
        return self.simple_objective - self[Hypothesis.H2].objective

    @property
    def total_attempts(self):
        return sum(f.attempts for f in self.fits.values())

    @property
    def total_evaluations(self):
        return sum(f.n_eval for f in self.fits.values())


def _poisson_objective(data, mean):
    positive = data > 0
    safe_data = np.where(positive, data, 1.0)
    term = np.where(positive, safe_data * np.log(safe_data / mean), 0.0)
    return float(np.sum(term - (data - mean)))


def _bounds(hypothesis, data, sigma, options):
    hypothesis = Hypothesis(hypothesis)
    h, w = data.shape
    positive = data[data > 0]
    typical = float(np.median(positive)) if positive.size else 1.0
    b_hi = max(float(np.max(data)) * 10.0, typical * 20.0, 100.0)
    flux_hi = max(float(np.sum(np.maximum(data, 0.0))) * 5.0, 100.0)
    base_lo = [np.log(1e-4), -options.log_slope_bound, -options.log_slope_bound]
    base_hi = [np.log(b_hi), options.log_slope_bound, options.log_slope_bound]
    pos_lo, pos_hi = [-0.5, -0.5], [h - 0.5, w - 0.5]
    if hypothesis is Hypothesis.H0:
        return np.array(base_lo), np.array(base_hi)
    if hypothesis is Hypothesis.HSMOOTH:
        curvature = float(options.log_curvature_bound)
        return (np.array(base_lo + [-curvature, -curvature, -curvature]),
                np.array(base_hi + [curvature, curvature, curvature]))
    if hypothesis is Hypothesis.H1:
        return (np.array(base_lo + [np.log(1e-4)] + pos_lo),
                np.array(base_hi + [np.log(flux_hi)] + pos_hi))
    if hypothesis is Hypothesis.HWIDE:
        wr_lo, wr_hi = options.wide_ratio_bounds
        return (np.array(base_lo + [np.log(1e-4)] + pos_lo + [np.log(wr_lo)]),
                np.array(base_hi + [np.log(flux_hi)] + pos_hi + [np.log(wr_hi)]))
    separation_bound = options.pair_max_separation_ratio * float(sigma)
    return (
        np.array(base_lo + [np.log(1e-4)] + pos_lo
                 + [0.0, -np.pi, 0.5]),
        np.array(base_hi + [np.log(flux_hi)] + pos_hi
                 + [separation_bound, np.pi, 0.98]),
    )


def _fit_once(hypothesis, data, grid, sigma, start, bounds, options, max_iter):
    lower, upper = bounds
    start = np.clip(np.asarray(start, dtype=float), lower + 1e-9, upper - 1e-9)
    evaluations = 0

    def objective(theta):
        nonlocal evaluations
        evaluations += 1
        mean, jac = evaluate(hypothesis, theta, grid, sigma)
        mean = np.maximum(mean, 1e-12)
        value = _poisson_objective(data, mean)
        gradient = jac.reshape(-1, theta.size).T @ (
            1.0 - data.reshape(-1) / mean.reshape(-1)
        )
        return value, gradient

    started = perf_counter()
    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        jac=True,
        bounds=list(zip(lower, upper)),
        options={"maxiter": int(max_iter), "ftol": options.ftol,
                 "gtol": options.gtol, "maxls": 30},
    )
    elapsed = perf_counter() - started
    return result, evaluations, elapsed


def _fit_best(hypothesis, data, grid, sigma, starts, options):
    bounds = _bounds(hypothesis, data, sigma, options)
    starts = [np.asarray(s, dtype=float) for s in starts]
    if not starts:
        raise ValueError(f"no starts supplied for {Hypothesis(hypothesis).value}")

    attempts = evaluations = iterations = 0
    elapsed = 0.0
    screened = []
    screen_iter = min(options.screen_iter, options.max_iter)
    for start in starts:
        result, neval, seconds = _fit_once(
            hypothesis, data, grid, sigma, start, bounds, options, screen_iter)
        attempts += 1
        evaluations += neval
        iterations += int(result.nit)
        elapsed += seconds
        if np.isfinite(result.fun):
            screened.append(result)

    if not screened:
        raise RuntimeError(f"all {Hypothesis(hypothesis).value} starts failed")
    screened.sort(key=lambda r: float(r.fun))

    finalists = screened[:max(1, min(options.keep_screened, len(screened)))]
    refined = []
    for candidate in finalists:
        # A screen that already met L-BFGS-B's convergence criterion is a full
        # optimum; running the identical fit again only repeats work.  Starts
        # stopped by the short iteration budget still receive the full solve.
        if candidate.success:
            refined.append(candidate)
            continue
        result, neval, seconds = _fit_once(
            hypothesis, data, grid, sigma, candidate.x, bounds, options,
            options.max_iter)
        attempts += 1
        evaluations += neval
        iterations += int(result.nit)
        elapsed += seconds
        if np.isfinite(result.fun):
            refined.append(result)
    best = min(refined or finalists, key=lambda r: float(r.fun))

    lower, upper = bounds
    span = np.maximum(upper - lower, 1e-12)
    distance = np.minimum(best.x - lower, upper - best.x) / span
    if Hypothesis(hypothesis) is Hypothesis.H2:
        # q=0.5 is the equal-flux symmetry convention, not a physical rail.
        distance[-1] = (upper[-1] - best.x[-1]) / span[-1]
        # Angle is periodic, so +/- pi is a coordinate seam rather than a
        # physical boundary.
        distance[-2] = 1.0
    at_boundary = bool(np.any(distance < 1e-5))
    physical = decode(hypothesis, best.x, sigma)
    collapsed = False
    if Hypothesis(hypothesis) is Hypothesis.H2:
        collapsed = bool(physical["separation"] < 0.1 * sigma
                         or physical["flux_fraction"] > 0.97)
    return LocalFit(
        hypothesis=Hypothesis(hypothesis),
        theta=np.asarray(best.x, dtype=float),
        objective=float(best.fun),
        converged=bool(best.success),
        message=str(best.message),
        n_iter=iterations,
        n_eval=evaluations,
        attempts=attempts,
        elapsed_s=elapsed,
        at_boundary=at_boundary,
        collapsed=collapsed,
        physical=physical,
    )


def _initial_background(data):
    positive = data[data > 0]
    value = float(np.percentile(positive, 20.0)) if positive.size else 1.0
    return max(value, 1e-3)


def _weighted_geometry(data, background):
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    weight = np.maximum(np.asarray(data, dtype=float) - background, 0.0)
    total = float(weight.sum())
    if total <= 1e-12:
        y, x = np.unravel_index(int(np.argmax(data)), data.shape)
        return np.array([float(y), float(x)]), 0.0, 1.0
    centre = np.array([(weight * yy).sum() / total,
                       (weight * xx).sum() / total])
    dy, dx = yy - centre[0], xx - centre[1]
    covariance = np.array([
        [(weight * dy * dy).sum(), (weight * dy * dx).sum()],
        [(weight * dy * dx).sum(), (weight * dx * dx).sum()],
    ]) / total
    values, vectors = np.linalg.eigh(covariance)
    vector = vectors[:, int(np.argmax(values))]
    angle = float(np.mod(np.arctan2(vector[0], vector[1]), np.pi))
    return centre, angle, total


def _unique_angles(angles, period=2.0 * np.pi, tolerance=1e-7):
    unique = []
    for angle in angles:
        angle = float(np.mod(angle + period / 2.0, period) - period / 2.0)
        if all(abs(np.angle(np.exp(2j * np.pi * (angle - old) / period)))
               > tolerance
               for old in unique):
            unique.append(angle)
    return unique


def fit_hypotheses(data, sigma, *, centre=None, start_grid=None, options=None):
    """Fit H0, H1, Hwide and H2 to the same oracle ROI.

    ``centre`` identifies the ROI's candidate center, not a fixed source
    location.  All position parameters remain free.  If omitted, a positive
    residual centroid is used.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or min(data.shape) < 3:
        raise ValueError("data must be a two-dimensional ROI at least 3x3")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    options = FitOptions() if options is None else options
    start_grid = StartGrid() if start_grid is None else start_grid
    grid = ROIGrid.from_shape(data.shape)
    started = perf_counter()

    b0 = _initial_background(data)
    background_start = np.array([np.log(b0), 0.0, 0.0])
    h0 = _fit_best(Hypothesis.H0, data, grid, sigma,
                   [background_start], options)
    hsmooth = _fit_best(
        Hypothesis.HSMOOTH, data, grid, sigma,
        [background_start.tolist() + [0.0, 0.0, 0.0]], options)
    fitted_b = h0.physical["background_center"]
    moment_centre, axis, excess = _weighted_geometry(data, fitted_b)
    if centre is None:
        centre = moment_centre
    centre = np.asarray(centre, dtype=float)
    if centre.shape != (2,):
        raise ValueError("centre must be a (y, x) pair")
    peak = np.array(np.unravel_index(int(np.argmax(data)), data.shape), dtype=float)
    flux0 = max(excess, 1.0)

    common = h0.theta[:3].tolist()
    h1_starts = [
        common + [np.log(flux0), centre[0], centre[1]],
        common + [np.log(flux0), peak[0], peak[1]],
    ]
    h1 = _fit_best(Hypothesis.H1, data, grid, sigma, h1_starts, options)
    focus_centre = h1.physical["positions"][0]
    focus_flux = max(float(h1.physical["fluxes"][0]), 1.0)

    wr_lo, wr_hi = options.wide_ratio_bounds
    ratios = [wr_lo, 1.5, 2.0, 3.0, wr_hi]
    ratios = sorted(set(float(np.clip(r, wr_lo, wr_hi)) for r in ratios))
    wide_starts = [
        common + [np.log(focus_flux), focus_centre[0], focus_centre[1], np.log(r)]
        for r in ratios
    ]
    hwide = _fit_best(Hypothesis.HWIDE, data, grid, sigma,
                      wide_starts, options)

    base_angles = axis + np.asarray(start_grid.angle_offsets)
    h2_starts = []
    for flux_ratio in start_grid.flux_ratios:
        q = float(flux_ratio / (1.0 + flux_ratio))
        # Equal-flux pairs are invariant to a pi rotation. Unequal pairs are
        # not: the polarity says which end is brighter.
        angles = _unique_angles(
            base_angles if np.isclose(q, 0.5)
            else np.concatenate([base_angles, base_angles + np.pi]),
            period=np.pi if np.isclose(q, 0.5) else 2.0 * np.pi,
        )
        for angle in angles:
            for separation_ratio in start_grid.separation_ratios:
                separation = min(float(separation_ratio),
                                 options.pair_max_separation_ratio) * sigma
                h2_starts.append(
                    common + [np.log(focus_flux), focus_centre[0], focus_centre[1],
                              separation, angle, q]
                )
    h2 = _fit_best(Hypothesis.H2, data, grid, sigma, h2_starts, options)

    return FitCollection(
        fits={Hypothesis.H0: h0, Hypothesis.HSMOOTH: hsmooth,
              Hypothesis.H1: h1,
              Hypothesis.H2: h2, Hypothesis.HWIDE: hwide},
        elapsed_s=perf_counter() - started,
        estimated_axis=axis,
        shape=tuple(data.shape),
        sigma=float(sigma),
        options=options,
        start_grid=start_grid,
    )
