"""Profiled gradients and bounded local geometry search against independent references."""
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from scripts.check_inference import load_contract
from spotsolve.inference import prepare_rust
from spotsolve.inference.reference.profiled import profile_at, solve_poisson_linear


@pytest.fixture(scope='module')
def contract():
    extension = pytest.importorskip('spotsolve_rs')
    if not hasattr(getattr(extension, 'CalibratedModel', None), 'fit_component'):
        pytest.skip('installed extension predates geometry fitting')
    return load_contract(Path(__file__).resolve().parents[1]/'fixtures/08_inference.npz')


def python_local(data, model, count, theta, lower, upper):
    """One L-BFGS-B start, without the Python fitter's extra proposal searches."""
    current = theta.copy()
    ids = np.setdiff1d(np.arange(len(theta)), model.affine_indices(count))
    def objective(x):
        nonlocal current
        current[ids] = x
        affine_ids = model.affine_indices(count)
        unit = current.copy(); unit[affine_ids] = 1
        _, jac = model.evaluate(count, unit)
        basis = jac.reshape(-1,len(current))[:, affine_ids]
        inner = solve_poisson_linear(data, basis, np.full(data.size,1e-4),
            current[affine_ids], lower[affine_ids], upper[affine_ids],
            constraints=model.rate_constraints(count)[:, affine_ids])
        assert inner.converged
        current[affine_ids] = inner.coefficients
        mean, jac = model.evaluate(count, current)
        gradient = jac.reshape(-1,len(current)).T@(1-data.ravel()/mean.ravel())
        return inner.objective, gradient[ids]
    result = minimize(objective, theta[ids], jac=True, method='L-BFGS-B',
                      bounds=list(zip(lower[ids], upper[ids])),
                      options=dict(maxiter=400, gtol=1e-6, ftol=1e-13, maxls=30))
    value, _ = objective(result.x)
    return value, current


def check_report(native, model, data, lo, hi, result):
    assert result['converged'], (result['status'], result['geometry_kkt'], result['inner_failures'])
    assert result['geometry_kkt'] <= 1e-6
    assert result['inner_kkt'] <= 1e-7
    theta = result['theta']; count = (len(theta)-20)//3
    assert np.min(theta-lo) >= -1e-8
    assert np.min(hi-theta) >= -1e-8
    assert np.min(model.rate_constraints(count)@theta) >= -1e-8
    # Recompute gradients at the returned point, not at the last line-search trial.
    mean, jac = model.evaluate(count, theta)
    gradient = jac.reshape(-1,len(theta)).T @ (1-data.ravel()/mean.ravel())
    np.testing.assert_allclose(gradient, result['gradient'], atol=1e-8)
    profile = native.profile_at(data, theta, lo, hi)
    assert profile['converged']
    np.testing.assert_allclose(profile['objective'], result['objective'], atol=1e-8)


@pytest.mark.parametrize('count', [0,1,2])
def test_profiled_gradient_matches_python_and_finite_differences(contract, count):
    model, values = contract; native = prepare_rust(model)
    theta = values['trial'][:20+3*count].copy()
    data = values['data']; lo,hi = model.parameter_bounds(data,count)
    result = native.profile_at(data, theta, lo,hi)
    expected_theta, linear, gradient = profile_at(data,model,count,theta)
    assert result['converged']
    np.testing.assert_allclose(result['theta'],expected_theta,atol=1e-6)
    np.testing.assert_allclose(result['objective'],linear.objective,atol=1e-8)
    np.testing.assert_allclose(result['gradient'],gradient,atol=1e-7)
    if count == 2:
        np.testing.assert_allclose(result['gradient'],values['expected_profile_gradient'],atol=1e-6)
    ids = np.setdiff1d(np.arange(len(theta)),model.affine_indices(count))
    for j in ids:
        delta = np.zeros_like(theta);delta[j] = 1e-5
        above = native.profile_at(data,theta+delta,lo,hi,tolerance=1e-9)
        below = native.profile_at(data,theta-delta,lo,hi,tolerance=1e-9)
        assert above['converged'] and below['converged']
        numeric = (above['objective']-below['objective'])/2e-5
        np.testing.assert_allclose(result['gradient'][j],numeric,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('count',[0,1,2])
def test_noiseless_geometry_from_displaced_start(contract,count):
    model,values = contract;native = prepare_rust(model)
    true = values['theta'][:20+3*count].copy()
    data = model.evaluate(count,true)[0];lo,hi=model.parameter_bounds(data,count)
    theta=true.copy();theta[17:20]+=[.2,-.2,.02]
    for k in range(count):theta[21+3*k:23+3*k]+=[.15,-.1]
    result=native.fit_geometry(data,theta,lo,hi)
    check_report(native,model,data,lo,hi,result)
    assert result['objective'] < 1e-10
    ids=np.setdiff1d(np.arange(len(theta)),model.affine_indices(count))
    np.testing.assert_allclose(result['theta'][ids],true[ids],atol=1e-5)
    expected,_=python_local(data,model,count,theta,lo,hi)
    np.testing.assert_allclose(result['objective'],expected,atol=1e-8)


@pytest.mark.parametrize('scene',['noisy_pair','boundary','absent_broad','haze_boundary'])
def test_geometry_noise_and_boundaries(contract,scene):
    model,values=contract;native=prepare_rust(model)
    theta=values['theta'].copy()
    if scene=='absent_broad':theta[16]=0
    if scene=='haze_boundary':theta[:16]=np.repeat([0,.2,.6,.8],4)
    data=model.evaluate(2,theta)[0]
    if scene=='noisy_pair':data=np.random.default_rng(19).poisson(data).astype(float)
    if scene=='haze_boundary':data=np.random.default_rng(42).poisson(data).astype(float)
    lo,hi=model.parameter_bounds(data,2)
    if scene=='boundary':
        lo[21]=theta[21]+.15
        theta[21]=lo[21]+.1
        # Fixed defocus checks the projected gradient on equal bounds too.
        lo[19]=hi[19]=theta[19]
    elif scene=='absent_broad':
        lo[16]=hi[16]=0
        theta[17:20]+=[.2,-.2,.02]
        theta[21]+=.1
    else:
        theta[21]+=.1
    result=native.fit_geometry(data,theta,lo,hi)
    check_report(native,model,data,lo,hi,result)
    expected,reference=python_local(data,model,2,theta,lo,hi)
    np.testing.assert_allclose(result['objective'],expected,atol=1e-6)
    if scene=='haze_boundary':
        assert np.min(model.rate_constraints(2)@result['theta']) < 1e-8
    if scene=='boundary':
        np.testing.assert_allclose(result['theta'][21],lo[21],atol=1e-8)
        assert result['theta'][19]==lo[19]
    if scene=='absent_broad':
        assert result['theta'][16]==0
        np.testing.assert_array_equal(result['theta'][17:20],theta[17:20])
        np.testing.assert_array_equal(result['gradient'][17:20],np.zeros(3))


def test_geometry_failures_are_explicit_and_preserve_accepted_state(contract):
    model,values=contract;native=prepare_rust(model)
    theta=values['trial'].copy();data=values['data'];lo,hi=model.parameter_bounds(data,2)
    initial=native.profile_at(data,theta,lo,hi)
    limited=native.fit_geometry(data,theta,lo,hi,max_iter=1)
    assert limited['status']=='iteration_limit' and not limited['converged']
    assert limited['objective'] <= initial['objective']
    check=native.profile_at(data,limited['theta'],lo,hi)
    np.testing.assert_allclose(check['gradient'],limited['gradient'],atol=1e-6)
    failed=native.fit_geometry(data,theta,lo,hi,inner_max_iter=1)
    assert failed['status']=='inner_failure' and failed['gradient'] is None
    profile=native.profile_at(data,theta,lo,hi,max_iter=1)
    assert not profile['converged'] and profile['gradient'] is None
    for settings in ({'gtol':np.nan},{'gtol':0},{'max_iter':0},{'inner_max_iter':0}):
        with pytest.raises(ValueError):native.fit_geometry(data,theta,lo,hi,**settings)
    bad=hi.copy();bad[19]=1
    with pytest.raises(ValueError,match='geometry bounds'):
        native.fit_geometry(data,theta,lo,bad)
    with pytest.raises(ValueError):native.fit_geometry(-data,theta,lo,hi)
    with pytest.raises(ValueError):native.fit_geometry(data[:,::-1],theta,lo,hi)
    # A failed call cannot mutate arrays returned by earlier calls.
    np.testing.assert_allclose(check['theta'],limited['theta'],atol=1e-6)
