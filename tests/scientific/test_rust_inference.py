"""Cross-language checks for the initial calibrated Rust kernels."""
from pathlib import Path

import numpy as np
import pytest

from scripts.check_inference import load_contract
from spotsolve.inference import prepare_rust
from spotsolve.inference._numerics import _poisson_objective


@pytest.fixture(scope="module")
def contract():
    extension = pytest.importorskip("spotsolve_rs")
    if not hasattr(getattr(extension, "CalibratedModel", None), "fit_component"):
        pytest.skip("installed extension predates the calibrated affine solver")
    return load_contract(Path(__file__).resolve().parents[1]/"fixtures/08_inference.npz")


@pytest.mark.parametrize("count", [0, 1, 2])
def test_rust_mean_and_jacobian_match_retained_model(contract, count):
    model, values = contract
    native = prepare_rust(model)
    theta = values["theta"][:20+3*count].copy()
    mean, jac = native.evaluate(theta)
    expected_mean, expected_jac = model.evaluate(count, theta)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jac, expected_jac, rtol=1e-8, atol=1e-8)
    if count == 2:
        np.testing.assert_allclose(mean, values["expected_mean"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(jac, values["expected_jacobian"], rtol=1e-8, atol=1e-8)
    # Outputs own their storage and remain unchanged when workspace is reused.
    snapshot = mean.copy()
    theta[16] = 0
    for k in range(count):
        theta[20+3*k] = 0
    new_mean, new_jac = native.evaluate(theta)
    np.testing.assert_array_equal(mean, snapshot)
    expected_mean, expected_jac = model.evaluate(count, theta)
    np.testing.assert_allclose(new_mean, expected_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(new_jac, expected_jac, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("zero_data", [False, True])
def test_rust_affine_statistics_match_poisson_calculus(contract, zero_data):
    model, values = contract
    native = prepare_rust(model)
    theta = values["trial"]
    data = np.zeros(model.shape) if zero_data else values["data"].copy()
    mean, jac = model.evaluate(2, theta)
    basis = jac.reshape(-1, len(theta))[:, model.affine_indices(2)]
    gradient = basis.T @ (1-data.ravel()/mean.ravel())
    hessian = basis.T @ ((data.ravel()/mean.ravel()**2)[:, None]*basis)
    value, actual_gradient, actual_hessian = native.affine_statistics(data, theta)
    np.testing.assert_allclose(value, _poisson_objective(data, mean), atol=1e-8)
    np.testing.assert_allclose(actual_gradient, gradient, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(actual_hessian, hessian, rtol=1e-8, atol=1e-8)


def test_rust_derivatives_at_calibration_endpoint_and_finite_window(contract):
    model, values = contract
    native = prepare_rust(model)
    theta = values["theta"].copy()
    theta[17:20] = [-.5, 20.5, model.psf.depth_um[-1]]
    mean, jac = native.evaluate(theta)
    expected_mean, expected_jac = model.evaluate(2, theta)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jac, expected_jac, rtol=1e-8, atol=1e-8)
    delta = np.zeros_like(theta); delta[21] = 1e-5
    numeric = (native.evaluate(theta+delta)[0]-native.evaluate(theta-delta)[0])/2e-5
    np.testing.assert_allclose(numeric, jac[..., 21], rtol=2e-6, atol=1e-7)


def test_rust_input_validation_and_recovery(contract):
    model, values = contract
    native = prepare_rust(model)
    theta = values["theta"].copy()
    for bad in (np.ones(21), np.full(26, np.nan)):
        with pytest.raises(ValueError):
            native.evaluate(bad)
    bad = theta.copy(); bad[17] = 100
    with pytest.raises(ValueError, match="outside calibration"):
        native.evaluate(bad)
    with pytest.raises(ValueError, match="nonnegative"):
        native.affine_statistics(-np.ones(model.shape), theta)
    with pytest.raises(ValueError, match="contiguous"):
        native.affine_statistics(values["data"][:, ::-1], theta)
    with pytest.raises(ValueError, match="shape"):
        native.affine_statistics(np.ones((3, 3)), theta)
    np.testing.assert_allclose(native.evaluate(theta)[0], values["expected_mean"], atol=1e-10)


def test_cached_affine_geometry_matches_new_coefficients_without_repreparing(contract):
    model, values = contract
    native = prepare_rust(model)
    theta = values["trial"].copy()
    ids = model.affine_indices(2)
    with pytest.raises(ValueError, match="prepare geometry"):
        native.affine_at(values["data"], theta[ids])
    coefficients = native.prepare_affine(theta)
    np.testing.assert_array_equal(coefficients, theta[ids])
    coefficients[16] = 0.0
    coefficients[-1] *= .7
    updated = theta.copy(); updated[ids] = coefficients
    mean, jac = model.evaluate(2, updated)
    basis = jac.reshape(-1, len(theta))[:, ids]
    expected_gradient = basis.T @ (1-values["data"].ravel()/mean.ravel())
    expected_hessian = basis.T @ ((values["data"].ravel()/mean.ravel()**2)[:, None]*basis)
    # Ordinary render calls do not change the explicitly prepared geometry.
    unrelated = theta.copy(); unrelated[21] += .3
    native.evaluate(unrelated)
    for _ in range(2):
        value, gradient, hessian = native.affine_at(values["data"], coefficients)
        np.testing.assert_allclose(value, _poisson_objective(values["data"], mean), atol=1e-8)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-8, atol=1e-8)
    bad = theta.copy(); bad[19] = 100
    with pytest.raises(ValueError):
        native.prepare_affine(bad)
    with pytest.raises(ValueError, match="prepare geometry"):
        native.affine_at(values["data"], coefficients)


def assert_affine_certificate(native, data, model, ids, lower, upper, fit, tolerance=1e-7):
    """Independently verify the returned primal/dual certificate in NumPy."""
    a = fit['coefficients']
    c = np.vstack([np.eye(len(a)), -np.eye(len(a)), model.rate_constraints(len(ids)-17)[:, ids]])
    rhs = np.r_[lower, -upper, np.zeros(np.prod(model.shape))]
    slack = c @ a-rhs
    value, gradient, _ = native.affine_at(data, a)
    lam = fit['multipliers']
    residual = max(np.max(np.abs(gradient-c.T@lam)), np.max(-slack),
                   np.max(np.abs(lam*slack)))
    assert np.min(lam) >= 0
    assert np.min(slack) >= -1e-8
    np.testing.assert_allclose(value, fit['objective'], atol=1e-9)
    np.testing.assert_allclose(residual, fit['kkt'], atol=1e-8)
    assert fit['converged'], (fit['status'], fit['kkt'], fit['iterations'])
    assert residual <= tolerance


@pytest.mark.parametrize('count', [0, 1, 2])
def test_rust_affine_solve_matches_python_and_frozen_profile(contract, count):
    from spotsolve.inference.reference.profiled import profile_at
    model, values = contract
    native = prepare_rust(model)
    theta = values['trial'][:20+3*count].copy()
    ids = model.affine_indices(count)
    lo, hi = model.parameter_bounds(values['data'], count)
    a = native.prepare_affine(theta)
    result = native.solve_affine(values['data'], a, lo[ids], hi[ids])
    reference = profile_at(values['data'], model, count, theta)[1]
    assert_affine_certificate(native, values['data'], model, ids, lo[ids], hi[ids], result)
    np.testing.assert_allclose(result['objective'], reference.objective, atol=1e-8)
    np.testing.assert_allclose(result['coefficients'], reference.coefficients, atol=1e-6)
    if count == 2:
        np.testing.assert_allclose(result['coefficients'], values['expected_profile_theta'][ids], atol=1e-6)
    # Warm starts remain explicit data, and returned arrays own their storage.
    saved = result['coefficients'].copy()
    again = native.solve_affine(values['data'], saved, lo[ids], hi[ids])
    assert again['iterations'] == 0
    np.testing.assert_array_equal(result['coefficients'], saved)


@pytest.mark.parametrize('scene', ['zeros', 'signed', 'boundary', 'noisy'])
def test_rust_affine_solve_background_and_flux_boundaries(contract, scene):
    from spotsolve.inference.reference.profiled import profile_at
    model, values = contract
    native = prepare_rust(model)
    theta = values['theta'].copy()
    ids = model.affine_indices(2)
    theta[[16, 20, 23]] = 0
    if scene == 'signed':
        theta[:16] = np.repeat([.6, -.1, -.1, .6], 4)
    else:
        theta[:16] = np.repeat([0, .2, .6, .8], 4)
    data = model.evaluate(2, theta)[0]
    if scene == 'zeros': data = np.zeros(model.shape)
    if scene == 'noisy': data = np.random.default_rng(71).poisson(data).astype(float)
    theta[:16] = .5
    theta[[16, 20, 23]] = .1
    lo, hi = model.parameter_bounds(data, 2)
    a = native.prepare_affine(theta)
    result = native.solve_affine(data, a, lo[ids], hi[ids])
    assert_affine_certificate(native, data, model, ids, lo[ids], hi[ids], result)
    if scene == 'zeros':
        # Without reuse there is one Hessian per iteration plus the initial one.
        assert result['hessian_evaluations'] < result['iterations']+1
        np.testing.assert_allclose(result['objective'], np.prod(model.shape)*1e-4, atol=1e-8)
    else:
        reference = profile_at(data, model, 2, theta)[1]
        np.testing.assert_allclose(result['objective'], reference.objective, atol=1e-7)
    if scene == 'signed': assert np.min(result['coefficients'][:16]) < -.09


def test_rust_affine_solver_rejects_bad_inputs_and_reports_budget(contract):
    model, values = contract
    native = prepare_rust(model)
    ids = model.affine_indices(2)
    lo, hi = model.parameter_bounds(values['data'], 2)
    a = values['trial'][ids].copy()
    with pytest.raises(ValueError, match='prepare geometry'):
        native.solve_affine(values['data'], a, lo[ids], hi[ids])
    native.prepare_affine(values['trial'])
    for bad in (np.full(len(a), np.nan), -np.ones(len(a))):
        with pytest.raises(ValueError):
            native.solve_affine(values['data'], bad, lo[ids], hi[ids])
    for kwargs in ({'tolerance': np.nan}, {'tolerance': 0}, {'max_iter': 0}):
        with pytest.raises(ValueError):
            native.solve_affine(values['data'], a, lo[ids], hi[ids], **kwargs)
    result = native.solve_affine(values['data'], a, lo[ids], hi[ids], max_iter=1)
    assert not result['converged']
    assert result['status'] == 'iteration_limit'
    assert result['kkt'] > 1e-7


@pytest.mark.parametrize('scene', ['poisson_pair', 'coincident'])
def test_rust_affine_solve_noisy_pair_and_unidentifiable_flux_split(contract, scene):
    from spotsolve.inference.reference.profiled import profile_at
    model, values = contract
    native = prepare_rust(model)
    theta = values['trial'].copy()
    ids = model.affine_indices(2)
    if scene == 'coincident':
        theta[24:26] = theta[21:23]
        data = model.evaluate(2, theta)[0]
        theta[:16] = .5
        theta[[16, 20, 23]] = .1
    else:
        data = np.random.default_rng(19).poisson(values['expected_mean']).astype(float)
    lo, hi = model.parameter_bounds(data, 2)
    initial = native.prepare_affine(theta)
    result = native.solve_affine(data, initial, lo[ids], hi[ids])
    assert_affine_certificate(native, data, model, ids, lo[ids], hi[ids], result)
    expected = profile_at(data, model, 2, theta)[1]
    np.testing.assert_allclose(result['objective'], expected.objective, atol=1e-7)
    fitted = theta.copy(); fitted[ids] = result['coefficients']
    reference = theta.copy(); reference[ids] = expected.coefficients
    # Coincident sources have a unique sum/mean, not individually identifiable fluxes.
    np.testing.assert_allclose(model.evaluate(2, fitted)[0], model.evaluate(2, reference)[0], atol=1e-6)
