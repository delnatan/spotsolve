"""Observed (not Fisher/BFGS) curvature and nuisance-aware uncertainty."""
from pathlib import Path

import numpy as np
import pytest

from scripts.check_inference import load_contract
from spotsolve.inference import prepare_rust, CountFit
from spotsolve.inference.reference import position_uncertainty


@pytest.fixture(scope='module')
def contract():
    extension = pytest.importorskip('spotsolve_rs')
    if not hasattr(getattr(extension, 'CalibratedModel', None), 'fit_component'):
        pytest.skip('installed extension predates observed uncertainty')
    return load_contract(Path(__file__).resolve().parents[1]/'fixtures/08_inference.npz')


def python_gradient(model, data, theta):
    mean, jac = model.evaluate((len(theta)-20)//3, theta)
    return jac.reshape(-1,len(theta)).T@(1-data.ravel()/mean.ravel())


def reference_uncertainty(model, data, theta):
    count = (len(theta)-20)//3
    fit = CountFit(count, theta, 0., True, 0., (), 0, 1, model.nuisance_size)
    return position_uncertainty(data, model, fit)


@pytest.mark.parametrize('count', [0, 1, 2])
def test_observed_hessian_matches_independent_gradient_differences(contract, count):
    model, values = contract; native = prepare_rust(model)
    theta = values['trial'][:20+3*count].copy()
    data = np.random.default_rng(71).poisson(values['expected_mean']).astype(float)
    result = native.observed_curvature(data, theta)
    gradient = python_gradient(model, data, theta)
    np.testing.assert_allclose(result['gradient'], gradient, atol=1e-8)
    h = np.empty((len(theta),len(theta)))
    for j in range(len(theta)):
        delta = np.zeros_like(theta); delta[j] = 2e-6
        h[:,j] = (python_gradient(model,data,theta+delta)-python_gradient(model,data,theta-delta))/(4e-6)
    np.testing.assert_allclose(result['hessian'], h, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(result['hessian'], result['hessian'].T)
    mean, jac = model.evaluate(count,theta)
    j = jac.reshape(-1,len(theta))
    fisher = j.T@((1/mean.ravel())[:,None]*j)
    observed_outer = j.T@((data.ravel()/mean.ravel()**2)[:,None]*j)
    assert np.max(np.abs(result['hessian']-fisher)) > 1
    # A gradient/Jacobian-only port would miss this residual-curvature term.
    assert np.max(np.abs(result['hessian']-observed_outer)) > 1


def test_mixed_flux_derivatives_survive_zero_flux_and_depth_endpoint(contract):
    model, values = contract; native = prepare_rust(model)
    theta = values['theta'].copy();theta[[16,20,23]] = 0
    theta[19] = model.psf.depth_um[-1]
    data = values['data']
    actual = native.observed_curvature(data,theta)['hessian']
    assert abs(actual[16,19]) > 1
    assert abs(actual[20,21]) > 1
    for j in [16,17,18,19,20,21,22,23,24,25]:
        delta = np.zeros_like(theta); delta[j]=2e-6
        if j == 19:
            numeric = (3*python_gradient(model,data,theta)-4*python_gradient(model,data,theta-delta)
                       +python_gradient(model,data,theta-2*delta))/(4e-6)
        else:
            numeric = (python_gradient(model,data,theta+delta)-python_gradient(model,data,theta-delta))/(4e-6)
        np.testing.assert_allclose(actual[:,j],numeric,rtol=3e-5,atol=3e-5)


def test_frozen_joint_position_covariance_and_nuisance_coupling(contract):
    model, values = contract; native = prepare_rust(model)
    theta, data = values['theta'], values['data']
    lo,hi=model.parameter_bounds(data,2)
    result=native.position_uncertainty(data,theta,lo,hi)
    expected=reference_uncertainty(model,data,theta)
    assert result['status']==expected.status=='conditional_observed_hessian'
    np.testing.assert_allclose(result['covariance'], values['expected_covariance'], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(result['nuisance_fixed_covariance'],expected.nuisance_fixed_covariance,rtol=1e-5,atol=1e-8)
    assert np.all(np.diag(result['covariance'])>np.diag(result['nuisance_fixed_covariance']))
    assert np.max(np.abs(result['covariance'][:2,2:]))>1e-5
    assert np.linalg.eigvalsh(result['covariance'])[0]>0
    np.testing.assert_allclose(result['covariance'],result['covariance'].T,atol=1e-12)


@pytest.mark.parametrize('scene', ['noisy', 'absent_broad'])
def test_covariance_matches_python_at_fitted_and_conditionally_absent_models(contract,scene):
    model,values=contract;native=prepare_rust(model)
    theta=values['theta'].copy()
    if scene=='absent_broad':theta[16]=0
    data=model.evaluate(2,theta)[0]
    if scene=='noisy':data=np.random.default_rng(19).poisson(data).astype(float)
    lo,hi=model.parameter_bounds(data,2)
    if scene=='noisy':
        fit=native.fit_geometry(data,theta,lo,hi)
        assert fit['converged'];theta=fit['theta']
    result=native.position_uncertainty(data,theta,lo,hi)
    expected=reference_uncertainty(model,data,theta)
    assert result['status']==expected.status=='conditional_observed_hessian'
    np.testing.assert_allclose(result['covariance'],expected.covariance,rtol=2e-5,atol=1e-8)
    if scene=='absent_broad':
        moved=theta.copy();moved[17:20]=[9.,9.,.5]
        alternative=native.position_uncertainty(data,moved,lo,hi)
        np.testing.assert_allclose(alternative['covariance'],result['covariance'],atol=1e-12)


@pytest.mark.parametrize('scene,status', [
    ('background','background_pixel_boundary'),
    ('zero_focus','parameter_boundary'),
    ('coordinate','parameter_boundary'),
    ('nonstationary','nonstationary_fit'),
    ('coincident','singular_or_nonpositive_curvature'),
    ('no_focus','no_focused_sources')])
def test_unreliable_uncertainty_is_withheld(contract,scene,status):
    model,values=contract;native=prepare_rust(model)
    theta=values['theta'].copy(); data=values['data']
    if scene=='background':theta[:16]=np.repeat([0,.2,.6,.8],4)
    if scene=='zero_focus':theta[20]=0
    if scene=='coordinate':theta[21]=model.focus_bounds[0]
    if scene=='nonstationary':theta[21]+=.2
    if scene=='coincident':
        theta[24:26]=theta[21:23]
        data=model.evaluate(2,theta)[0]
    if scene=='no_focus':theta=theta[:20]
    lo,hi=model.parameter_bounds(data,(len(theta)-20)//3)
    result=native.position_uncertainty(data,theta,lo,hi)
    assert result['status']==status
    assert result['covariance'] is None and result['nuisance_fixed_covariance'] is None
    if scene=='nonstationary':assert result['gradient_max']>1e-3


def test_uncertainty_input_validation_and_output_ownership(contract):
    model,values=contract;native=prepare_rust(model)
    theta,data=values['theta'].copy(),values['data']
    lo,hi=model.parameter_bounds(data,2)
    saved=native.position_uncertainty(data,theta,lo,hi)['covariance']
    snapshot=saved.copy()
    hessian=native.observed_curvature(data,theta)['hessian'];h_snapshot=hessian.copy()
    bad=theta.copy();bad[21]=model.focus_bounds[0]
    assert native.position_uncertainty(data,bad,lo,hi)['covariance'] is None
    for image,t in [(-data,theta),(data,np.full(26,np.nan)),(data,np.ones(21)),(data[:,::-1],theta)]:
        with pytest.raises(ValueError):native.observed_curvature(image,t)
    with pytest.raises(ValueError):native.position_uncertainty(data,theta,lo[:-1],hi)
    with pytest.raises(ValueError):native.position_uncertainty(data,theta,hi,lo)
    np.testing.assert_array_equal(saved,snapshot)
    np.testing.assert_array_equal(hessian,h_snapshot)


@pytest.mark.parametrize('flux', [0., 1e-15, 7e-13])
def test_numerically_absent_broad_has_zero_face_covariance(contract, flux):
    model, values = contract
    native = prepare_rust(model)
    theta = values['theta'].copy()
    theta[16] = 0.
    data = model.evaluate(2, theta)[0]
    lo, hi = native.parameter_bounds(data, 2)
    expected = native.position_uncertainty(data, theta, lo, hi)
    theta[16] = flux
    # Geometry at its bounds is irrelevant once broad light is conditioned absent.
    theta[17:20] = [0., 20., model.defocus_bounds_um[0]]
    snapshot = theta.copy()
    result = native.position_uncertainty(data, theta, lo, hi)
    assert result['broad_conditioned_absent']
    assert result['status'] == expected['status'] == 'conditional_observed_hessian'
    np.testing.assert_allclose(result['covariance'], expected['covariance'], atol=1e-12)
    np.testing.assert_array_equal(theta, snapshot)


@pytest.mark.parametrize('case', ['weak', 'positive_lower_bound', 'pixel_guard', 'large_response', 'nonstationary'])
def test_absent_broad_conditioning_does_not_hide_weak_light_or_bad_fits(contract, case):
    from dataclasses import replace
    from spotsolve.inference import PixelPSFBank
    model, values = contract
    theta = values['theta'].copy()
    theta[16] = 0.
    data = model.evaluate(2, theta)[0]
    if case == 'large_response':
        bank = PixelPSFBank(values['depth_um'], values['offsets_px'], values['responses'] * 1e8)
        model = replace(model, psf=bank)
        theta[[20, 23]] /= 1e8
    native = prepare_rust(model)
    lo, hi = native.parameter_bounds(data, 2)
    theta[16] = 7e-13
    if case == 'weak':
        theta[16] = 1e-8  # Small, but outside numerical-zero tolerance.
    if case == 'positive_lower_bound':
        lo[16] = theta[16]
    if case == 'pixel_guard':
        theta[16] = 1e-10
        theta[17:20] = [0., 20., model.defocus_bounds_um[0]]
    if case == 'nonstationary':
        theta[21] += .2
    result = native.position_uncertainty(data, theta, lo, hi)
    assert result['broad_conditioned_absent'] == (case == 'nonstationary')
    assert result['status'] == ('nonstationary_fit' if case == 'nonstationary' else 'parameter_boundary')
    assert result['covariance'] is None


def test_pair_proposal_policies_agree_on_absent_broad_uncertainty(contract):
    from spotsolve.inference import fit_component, position_uncertainty as native_uncertainty, FitOptions
    model, values = contract
    theta = values['theta'].copy()
    theta[:16] = .5
    theta[16:20] = [0., 10., 10., .4]
    theta[20:] = [1.5, 10., 9., 1.5, 10., 11.]
    data = model.evaluate(2, theta)[0]
    covariances = []
    for policy in ('moments', 'aguet'):
        fit = fit_component(data, model, proposal_method=policy,
                            options=FitOptions(max_iter=200, screen_iter=24, keep_screened=2))[2]
        result = native_uncertainty(data, model, fit)
        assert result.broad_conditioned_absent
        assert result.status == 'conditional_observed_hessian'
        # Put emitter labels in common x order before comparing joint blocks.
        order = np.argsort(fit.positions[:, 1])
        ids = np.array([[2*k, 2*k+1] for k in order]).ravel()
        covariances.append(result.covariance[np.ix_(ids, ids)])
    np.testing.assert_allclose(*covariances, rtol=1e-6, atol=1e-10)
