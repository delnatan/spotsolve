"""Separable Poisson component fitting and conditional position uncertainty.

The inner bounded convex solve profiles affine parameters. The outer search
uses the envelope gradient and the existing count/start machinery. No new
source-count rule, amplitude penalty, or detector default is introduced.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize, nnls
from scipy.signal import correlate2d

from .._numerics import _poisson_objective, _projected_gradient, ROIGrid
from ..model import parameter_bounds as _bounds, evaluate_component
from ..types import CountFit, PositionUncertainty
from .types import FitOptions


@dataclass
class LinearFit:
    coefficients: np.ndarray
    objective: float
    kkt: float
    iterations: int
    converged: bool


def constrained_kkt(coefficients, gradient, lower, upper, constraints):
    """Stationarity residual after nonnegative active-constraint multipliers.

    Rows of constraints mean C @ coefficients >= 0. Include feasibility;
    this is not the box-only projected gradient at a pixel boundary.
    """
    a, g = np.asarray(coefficients), np.asarray(gradient)
    eye = np.eye(len(a))
    normals = [eye[a <= lower+1e-8], -eye[a >= upper-1e-8]]
    slack = constraints @ a
    normals.append(constraints[slack <= 1e-8])
    normals = np.concatenate(normals)
    if len(normals):
        norms = np.linalg.norm(normals, axis=1)
        normals = normals[norms > 0] / norms[norms > 0, None]
        weights = nnls(normals.T, g, maxiter=max(100, 10*len(normals)))[0]
        g = g - normals.T @ weights
    return float(max(np.max(np.abs(g)), np.max(lower-a), np.max(a-upper),
                     np.max(-slack, initial=0)))


def background_constraints(model, count):
    """Nonnegative background pixel rates, separately from emitter light.

    The fixed 1e-4 numerical floor remains in the mean. These constraints
    apply to the polynomial's sampled pixel responses, not between pixels
    or outside this fixed ROI. There are still only grid_size**2 coefficients.
    """
    return model.rate_constraints(count)


def fit_kkt(theta, gradient, lower, upper, model, count):
    constraints = background_constraints(model, count)
    if constraints is None:
        return _projected_gradient(theta, gradient, lower, upper)
    return constrained_kkt(theta, gradient, lower, upper, constraints)


def solve_poisson_linear(data, basis, offset, initial, lower, upper,
                         *, tolerance=1e-7, max_iter=160, constraints=None):
    """Bounded damped Newton solve for mu=offset+basis@coefficients.

    Uses the observed Hessian, feasible Armijo steps and a diagonal-scaled
    projected gradient fallback. Rank deficiency does not imply a unique
    coefficient vector; the objective and KKT residual are the contract.
    """
    y = np.asarray(data, float).ravel()
    b = np.asarray(basis, float)
    c = np.asarray(offset, float).ravel()
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    a = np.clip(np.asarray(initial, float), lo, hi)
    if (b.shape != (len(y), len(a)) or c.shape != y.shape
            or lo.shape != a.shape or hi.shape != a.shape
            or np.any(lo > hi) or np.any(y < 0)
            or not all(np.all(np.isfinite(v)) for v in (y,b,c,a,lo,hi))):
        raise ValueError("incompatible/nonfinite linear Poisson problem")
    if tolerance <= 0 or max_iter < 1:
        raise ValueError("positive tolerance and iteration budget required")
    if constraints is not None:
        constraints = np.asarray(constraints, float)
        if (constraints.ndim != 2 or constraints.shape[1] != len(a)
                or not np.all(np.isfinite(constraints))
                or np.min(constraints @ a, initial=0) < -1e-8):
            raise ValueError("linear constraints require a compatible feasible initial point")
        # Most backgrounds have positive rates everywhere. Solve that interior
        # problem with the existing Newton primitive, then check feasibility.
        free_fit = solve_poisson_linear(y, b, c, a, lo, hi,
                                       tolerance=tolerance, max_iter=max_iter)
        if free_fit.converged and np.min(constraints @ free_fit.coefficients, initial=0) >= 0:
            return free_fit

        def objective(value):
            mean = c + b @ value
            if np.any(mean <= 0):
                # SLSQP can evaluate an infeasible trial before line search.
                return np.inf, np.zeros_like(value)
            return _poisson_objective(y, mean), b.T @ (1-y/mean)

        result = minimize(objective, a, jac=True, method="SLSQP",
                          bounds=list(zip(lo, hi)), constraints=[dict(
                              type="ineq", fun=lambda value: constraints @ value,
                              jac=lambda value: constraints)],
                          options=dict(maxiter=max_iter, ftol=1e-14))
        a = result.x.copy()
        # SLSQP's objective-change stopping rule can leave an accurate mean
        # but a less accurate envelope gradient. Polish the active face with
        # observed Newton curvature; no ridge enters the statistical model.
        for polish in range(12):
            value, gradient = objective(a)
            kkt = constrained_kkt(a, gradient, lo, hi, constraints)
            if kkt <= tolerance or not np.isfinite(value):
                break
            slack = constraints @ a
            face = np.concatenate([np.eye(len(a))[a <= lo+1e-8],
                                   np.eye(len(a))[a >= hi-1e-8],
                                   constraints[slack <= 1e-8]])
            tangent = null_space(face) if len(face) else np.eye(len(a))
            reduced = b @ tangent
            hessian = reduced.T @ ((y/(c+b@a)**2)[:, None]*reduced)
            if not tangent.shape[1] or np.any(np.diag(hessian) <= 0):
                break
            step = -tangent @ np.linalg.lstsq(hessian, tangent.T @ gradient, rcond=1e-12)[0]
            if gradient @ step >= 0:
                break
            margins = np.r_[a-lo, hi-a, slack]
            changes = np.r_[step, -step, constraints @ step]
            falling = changes < -1e-12
            alpha = min(1., np.min(np.maximum(margins[falling], 0)/-changes[falling], initial=1.))
            accepted = False
            for _ in range(30):
                trial = a+alpha*step
                trial_value = objective(trial)[0]
                if (np.isfinite(trial_value) and alpha > 0
                        and trial_value <= value+1e-4*alpha*(gradient @ step)+1e-12):
                    a = trial
                    accepted = True
                    break
                alpha *= .5
            if not accepted:
                break
        value, gradient = objective(a)
        kkt = constrained_kkt(a, gradient, lo, hi, constraints)
        return LinearFit(a, value, kkt, free_fit.iterations+result.nit+polish,
                         np.isfinite(value) and kkt <= tolerance)
    mu = c + b @ a
    if np.any(mu <= 0):
        raise ValueError("initial mean must be positive")
    value = _poisson_objective(y, mu)
    for iteration in range(max_iter + 1):
        g = b.T @ (1-y/mu)
        kkt = _projected_gradient(a, g, lo, hi)
        if kkt <= tolerance or iteration == max_iter:
            return LinearFit(a, value, kkt, iteration, kkt <= tolerance)
        free = ~(((a <= lo+1e-10) & (g >= 0)) |
                 ((a >= hi-1e-10) & (g <= 0)))
        ids = np.flatnonzero(free)
        h = b[:, ids].T @ ((y/mu**2)[:, None]*b[:, ids])
        scale = np.sqrt(np.maximum(np.diag(h), 1e-12))
        normalized = h/scale[:, None]/scale[None, :]
        # The ridge is solely a step safeguard; it is not added to likelihood
        # or used for reported covariance.
        step = np.zeros_like(a)
        step[ids] = -np.linalg.solve(normalized+1e-10*np.eye(len(ids)),
                                     g[ids]/scale)/scale
        step[(a <= lo+1e-10) & (step < 0)] = 0
        step[(a >= hi-1e-10) & (step > 0)] = 0
        if g @ step >= 0:
            step[ids] = -g[ids]/np.maximum(np.diag(h), 1e-12)
        accepted = False
        for direction in (step, -g/np.maximum(np.sum(b*b*(y/mu**2)[:, None], axis=0), 1e-12)):
            alpha = 1.
            for _ in range(45):
                trial = np.clip(a+alpha*direction, lo, hi)
                delta = trial-a
                trial_mu = c+b@trial
                if g@delta < 0 and np.all(trial_mu > 0):
                    trial_value = _poisson_objective(y, trial_mu)
                    if trial_value <= value + 1e-4*(g@delta) + 1e-12:
                        a, mu, value = trial, trial_mu, trial_value
                        accepted = True
                        break
                alpha *= .5
            if accepted:
                break
        if not accepted:
            return LinearFit(a, value, kkt, iteration, False)


def linear_indices(model, count):
    return model.affine_indices(count)


def profile_at(data, model, count, theta, *, tolerance=1e-7):
    """Fit affine coefficients at fixed geometry and return envelope gradient."""
    theta = np.asarray(theta, float).copy()
    lo, hi = _bounds(data, model, count)
    ids = linear_indices(model, count)
    reference = theta.copy()
    # Unit affine coefficients retain unit-amplitude geometry derivatives.
    # After fitting amplitudes, rescale those columns instead of evaluating
    # every calibrated PSF a second time. This also works at zero amplitude.
    reference[ids] = 1
    unit_mean, jac = evaluate_component(count, reference, model)
    basis = jac.reshape(-1, len(theta))[:, ids]
    offset = unit_mean.ravel()-basis.sum(axis=1)
    constraints = background_constraints(model, count)
    linear = solve_poisson_linear(data, basis, offset, theta[ids], lo[ids], hi[ids],
                                  tolerance=tolerance,
                                  constraints=None if constraints is None else constraints[:, ids])
    theta[ids] = linear.coefficients
    mean = offset+basis @ linear.coefficients
    for amplitude, geometry in model.amplitude_geometry(count):
        jac[..., list(geometry)] *= theta[amplitude]
    gradient = jac.reshape(-1,len(theta)).T @ (1-np.asarray(data).ravel()/mean)
    return theta, linear, gradient


def _count_bounds(data, model, count, fixed_positions):
    lo, hi = _bounds(data, model, count)
    fixed_positions = {} if fixed_positions is None else dict(fixed_positions)
    position_ids = {model.nuisance_size+3*k+j for k in range(count) for j in (1, 2)}
    for index, value in fixed_positions.items():
        if index not in position_ids or not np.isfinite(value) or not lo[index] <= value <= hi[index]:
            raise ValueError("fixed_positions must specify allowed focused coordinates")
        lo[index] = hi[index] = value
    return lo, hi, fixed_positions


def fit_profiled_count(data, model, count, starts, options, focus_rate=0., *, fixed_positions=None):
    if focus_rate != 0:
        raise ValueError("profiled solver fits likelihood only; focus_rate must be zero")
    lo, hi, fixed_positions = _count_bounds(data, model, count, fixed_positions)
    linear = linear_indices(model, count)
    nonlinear = np.setdiff1d(np.arange(len(lo)), np.r_[linear, list(fixed_positions)])
    evaluations = inner_iterations = inner_failures = 0
    def optimize(start, budget):
        current = np.clip(np.array(start, float), lo, hi)

        def objective(x):
            nonlocal current, evaluations, inner_iterations, inner_failures
            current[nonlinear] = x
            current, fit, gradient = profile_at(data, model, count, current)
            evaluations += 1
            inner_iterations += fit.iterations
            if not fit.converged:
                inner_failures += 1
                # Retry from a fresh interior amplitude guess to separate a
                # poor warm start from failure of the inner solve.
                restart = current.copy()
                restart[linear] = np.maximum(restart[linear], .01)
                current, fit, gradient = profile_at(data, model, count, restart, tolerance=1e-6)
                inner_iterations += fit.iterations
                if not fit.converged:
                    raise RuntimeError(f"inner Poisson solve failed: KKT={fit.kkt:g}")
            return fit.objective, gradient[nonlinear]

        if len(nonlinear):
            result = minimize(objective, current[nonlinear].copy(), jac=True,
                              method="L-BFGS-B", bounds=list(zip(lo[nonlinear], hi[nonlinear])),
                              options=dict(maxiter=budget, ftol=options.ftol,
                                           gtol=options.gtol, maxls=30))
            objective(result.x)
            success = bool(result.success)
        else:
            objective(np.empty(0))
            success = True
        mean, jac = evaluate_component(count, current, model)
        value = _poisson_objective(data, mean)
        gradient = jac.reshape(-1,len(current)).T @ (1-np.asarray(data).ravel()/mean.ravel())
        kkt = fit_kkt(current, gradient, lo, hi, model, count)
        return (value, current.copy(), success, kkt)

    starts = [np.clip(np.asarray(start, float), lo, hi) for start in starts]
    candidates = [optimize(start, min(options.screen_iter, options.max_iter)) for start in starts]
    candidates.sort(key=lambda item:item[0])
    refined = [optimize(item[1], options.max_iter) for item in candidates[:options.keep_screened]]
    restart_seeds = []
    if model.broad_index is not None and count:
        best = min(candidates+refined,key=lambda item:item[0])[1]
        absent = best.copy(); base = model.broad_index
        absent[base] = 0
        null_mean = evaluate_component(count, absent, model)[0]
        residual_score = np.asarray(data)/null_mean-1
        h,w = model.shape
        kernel_grid = ROIGrid.from_shape((2*h-1,2*w-1))
        for parameter in model.broad_starts:
            kernel = model.broad_unit(h-1,w-1,parameter,grid=kernel_grid)[0]
            numerator = correlate2d(residual_score,kernel,mode='same')
            information = correlate2d(1/null_mean,kernel*kernel,mode='same')
            best_position = np.unravel_index(np.argmax(numerator/np.sqrt(information)), data.shape)
            seed = best.copy()
            seed[base+1:base+3] = best_position
            seed[base+3] = parameter
            restart_seeds.append(seed)
        refined.extend(optimize(seed,options.max_iter) for seed in restart_seeds)
    candidates.extend((_poisson_objective(data, evaluate_component(count, start, model)[0]),start,False,None)
                      for start in starts)
    value,theta,success,kkt = min(candidates+refined,key=lambda item:item[0])
    if kkt is None:
        mean, jac = evaluate_component(count, theta, model)
        gradient = jac.reshape(-1,len(theta)).T @ (1-np.asarray(data).ravel()/mean.ravel())
        kkt = fit_kkt(theta, gradient, lo, hi, model, count)
    boundary = np.minimum(theta-lo,hi-theta)/np.maximum(hi-lo,1) < 1e-6
    return CountFit(count,theta.copy(),value,success,kkt,
                    tuple(np.flatnonzero(boundary).tolist()),evaluations,len(starts)+len(restart_seeds),
                    model.nuisance_size,0.,inner_iterations,inner_failures)


@dataclass
class PositionProfile:
    coordinates: np.ndarray
    delta_nll: np.ndarray
    fits: tuple
    status: str


def profile_position(data, model, fit, source, axis, coordinates, *, options=None):
    """Refit every other parameter at fixed y or x of one source.

    This is a local, fixed-count profile likelihood, not a marginalized
    posterior, probability of existence, or automatically calibrated interval.
    Negative delta_nll reports a better optimum than the supplied reference.
    With multiple sources, permutations/alternate modes need separate checks.
    """
    data = np.asarray(data, float)
    coordinates = np.asarray(coordinates, float)
    if (source not in range(fit.n_focus) or axis not in (0, 1)
            or coordinates.ndim != 1 or not len(coordinates)
            or not np.all(np.isfinite(coordinates))
            or data.shape != model.shape or np.any(~np.isfinite(data)) or np.any(data < 0)):
        raise ValueError("valid data, source, axis (0=y,1=x), and coordinate vector required")
    options = FitOptions(max_iter=400, screen_iter=48, keep_screened=2) if options is None else options
    index = model.nuisance_size+3*source+1+axis
    lo, hi = _bounds(data, model, fit.n_focus)
    if np.any(coordinates < lo[index]) or np.any(coordinates > hi[index]):
        raise ValueError("profile coordinates must lie in focus_bounds")
    # Visit the reference neighborhood first and reuse nearby fits, always
    # also retaining the reference start to avoid one-way path dependence.
    output = [None]*len(coordinates)
    previous = fit.theta
    for j in np.argsort(np.abs(coordinates-fit.theta[index])):
        current = fit_profiled_count(data, model, fit.n_focus,
                    [fit.theta, previous], options,
                    fixed_positions={index: coordinates[j]})
        output[j] = current
        previous = current.theta
    delta = np.array([f.objective-fit.objective for f in output])
    status = "conditional_profile_likelihood"
    if np.min(delta) < -1e-5:
        status = "reference_not_minimum"
    elif any(f.projected_gradient_max > 1e-3 for f in output):
        status = "nonstationary_profile"
    return PositionProfile(coordinates.copy(), delta, tuple(output), status)


def refine_from_profiles(data, model, fit, profiles, *, options=None):
    """Release fixed coordinates and refine better nuisance modes found by profiles.

    Returns one fixed-count fit. Callers comparing counts must retain nested
    starts when updating higher-count fits. This supplies a likelihood search
    step, not a new source-acceptance rule or a global-optimum certificate.
    """
    starts = [fit.theta]
    for profile in profiles:
        for candidate in profile.fits:
            if candidate.n_focus != fit.n_focus or candidate.nuisance_size != model.nuisance_size:
                raise ValueError("profiles must have the same model dimensions and source count")
            if candidate.objective < fit.objective-1e-5:
                starts.append(candidate.theta)
    if len(starts) == 1:
        return fit
    options = FitOptions(max_iter=400, screen_iter=48, keep_screened=4) if options is None else options
    return fit_profiled_count(data, model, fit.n_focus, starts, options)



def position_uncertainty(data, model, fit):
    """Observed-Hessian Schur complement; conditional on regular support.

    No eigenvalue clipping, Fisher substitution, or evidence interpretation.
    An exactly absent broad source is excluded, conditioning on its absence.
    """
    if not fit.n_focus:
        return PositionUncertainty(None,"no_focused_sources")
    theta = fit.theta
    lo, hi = _bounds(np.asarray(data),model,fit.n_focus)
    active = np.ones(len(theta),bool)
    if model.broad_index is not None and theta[model.broad_index] == 0:
        active[model.broad_index:model.broad_index+4] = False
    ids = np.flatnonzero(active)
    constraints = background_constraints(model, fit.n_focus)
    if constraints is not None and np.min(constraints @ theta) <= 1e-7:
        return PositionUncertainty(None,"background_pixel_boundary")
    if np.any(np.minimum(theta[ids]-lo[ids], hi[ids]-theta[ids]) <= 1e-7):
        return PositionUncertainty(None,"parameter_boundary")
    if fit.projected_gradient_max > 1e-3:
        return PositionUncertainty(None,"nonstationary_fit")

    def gradient(t):
        mean,jac = evaluate_component(fit.n_focus,t,model)
        return jac.reshape(-1,len(t)).T@(1-np.asarray(data).ravel()/mean.ravel())

    h = np.empty((len(ids),len(ids)))
    for col,j in enumerate(ids):
        delta = np.zeros_like(theta)
        delta[j] = min(1e-4*max(1,abs(theta[j])), .25*(theta[j]-lo[j]), .25*(hi[j]-theta[j]))
        if constraints is not None:
            affected = np.abs(constraints[:, j]) > 0
            if np.any(affected):
                delta[j] = min(delta[j], .25*np.min((constraints @ theta)[affected]
                                                       /np.abs(constraints[affected, j])))
        h[:,col] = (gradient(theta+delta)[ids]-gradient(theta-delta)[ids])/(2*delta[j])
    h = (h+h.T)/2
    diag = np.diag(h)
    if not np.all(np.isfinite(h)) or np.any(diag <= 0):
        return PositionUncertainty(None,"nonpositive_curvature")
    scale = np.sqrt(diag)
    h = h/scale[:,None]/scale[None,:]
    if np.linalg.eigvalsh(h)[0] <= 1e-8:
        return PositionUncertainty(None,"singular_or_nonpositive_curvature")
    position_ids = [model.nuisance_size+3*k+j for k in range(fit.n_focus) for j in (1,2)]
    r = np.array([np.flatnonzero(ids==j)[0] for j in position_ids])
    n = np.setdiff1d(np.arange(len(ids)),r)
    rr, rn, nn = h[np.ix_(r,r)], h[np.ix_(r,n)], h[np.ix_(n,n)]
    schur = rr-rn@np.linalg.solve(nn,rn.T)
    cov = np.linalg.solve(schur,np.eye(len(r)))/scale[r,None]/scale[None,r]
    fixed = np.linalg.solve(rr,np.eye(len(r)))/scale[r,None]/scale[None,r]
    return PositionUncertainty(cov,"conditional_observed_hessian",fixed)
