"""Complete native inference with a separate, explicit Python reference."""
from pathlib import Path

import numpy as np
import pytest

from scripts.check_inference import load_contract
from spotsolve.inference import fit_component, prepare_rust, FitOptions
from spotsolve.inference._numerics import _poisson_objective
from spotsolve.inference.reference import fit_component as reference_fit, FitOptions as ReferenceOptions


@pytest.fixture(scope='module')
def contract():
    extension = pytest.importorskip('spotsolve_rs')
    if not hasattr(getattr(extension,'CalibratedModel',None),'fit_component'):
        pytest.skip('installed extension predates native component inference')
    return load_contract(Path(__file__).resolve().parents[1]/'fixtures/08_inference.npz')


@pytest.mark.parametrize('truth_count',[0,1,2])
def test_complete_native_counts_match_python_controls_and_preserve_nesting(contract,truth_count):
    model,values=contract
    theta=values['theta'][:20+3*truth_count].copy()
    data=model.evaluate(truth_count,theta)[0]
    centres=theta[20:].reshape(-1,3)[:,1:]
    options=FitOptions(max_iter=200,screen_iter=24,keep_screened=2)
    native=fit_component(data,model,candidate_centres=centres,options=options)
    python=reference_fit(data,model,candidate_centres=centres,options=ReferenceOptions(**vars(options)))
    for count in range(3):
        fit=native[count];reference=python[count]
        np.testing.assert_allclose(fit.objective,reference.objective,atol=2e-5)
        assert fit.starts==reference.starts
        assert fit.n_focus==count and fit.nuisance_size==20
        assert fit.projected_gradient_max<1e-3
        lo,hi=model.parameter_bounds(data,count)
        assert np.min(fit.theta-lo)>-1e-8 and np.min(hi-fit.theta)>-1e-8
        assert np.min(model.rate_constraints(count)@fit.theta)>-1e-8
    assert np.min(native.likelihood_gains)>-1e-8
    assert native[truth_count].objective<1e-8
    if truth_count:
        # Fit labels can permute; compare the reported emitter set.
        expected=theta[20:].reshape(-1,3)[:,1:]
        np.testing.assert_allclose(native[truth_count].positions[np.argsort(native[truth_count].positions[:,1])],
                                   expected[np.argsort(expected[:,1])],atol=2e-4)


def test_native_counts_zero_image_and_invalid_inputs(contract):
    model,_=contract
    result=fit_component(np.zeros(model.shape),model,options=FitOptions(max_iter=80,screen_iter=8,keep_screened=1))
    for fit in result.fits:
        np.testing.assert_allclose(fit.objective,np.prod(model.shape)*1e-4,atol=1e-8)
    assert np.min(result.likelihood_gains)>-1e-8
    for data in (np.full(model.shape,np.nan),np.full(model.shape,-1.),np.zeros((3,3))):
        with pytest.raises(ValueError):fit_component(data,model)
    for centres in ([[1.,np.nan]],[1.,2.],np.zeros((1,3))):
        with pytest.raises(ValueError):fit_component(np.zeros(model.shape),model,candidate_centres=centres)
    with pytest.raises(ValueError):fit_component(np.zeros(model.shape),model,options=FitOptions(max_iter=0))
    with pytest.raises(ValueError):fit_component(np.zeros(model.shape),model,options=FitOptions(gtol=np.nan))


def test_default_path_never_calls_python_algorithm_and_reuses_native_model(contract,monkeypatch):
    model,values=contract
    data=values['data'];native=prepare_rust(model)
    def forbidden(*args,**kwargs):
        raise AssertionError("native inference called the Python reference")
    for name in ('evaluate','broad_unit','nuisance_starts','parameter_bounds'):
        monkeypatch.setattr(type(model),name,forbidden)
    options=FitOptions(max_iter=200,screen_iter=24,keep_screened=2)
    public=fit_component(data,model,options=options)
    original=public[2].theta.copy()
    # The same prepared model accepts multiple images without Python orchestration.
    repeated=native.fit_component(data,np.empty((0,2)),seed_sigma=model.seed_sigma,**vars(options))
    for fit,record in zip(public.fits,repeated):
        np.testing.assert_array_equal(fit.theta,record['theta'])
        assert fit.objective==record['objective']
    np.testing.assert_array_equal(public[2].theta,original)
    from spotsolve.inference import position_uncertainty
    covariance=position_uncertainty(data,model,public[2])
    assert covariance.status=='conditional_observed_hessian'
    assert np.min(np.linalg.eigvalsh(covariance.covariance))>0


def test_noisy_haze_native_search_is_feasible_stationary_and_nested(contract):
    model,values=contract
    theta=values['theta'].copy();theta[:16]=np.repeat([0.,.2,.6,.8],4)
    data=np.random.default_rng(42).poisson(model.evaluate(2,theta)[0]).astype(float)
    fits=fit_component(data,model,options=FitOptions(max_iter=200,screen_iter=24,keep_screened=2))
    assert min(fits.likelihood_gains)>=-1e-8
    for count,fit in enumerate(fits.fits):
        assert fit.projected_gradient_max<1e-3
        assert np.min(model.rate_constraints(count)@fit.theta)>-1e-8
        np.testing.assert_allclose(fit.objective,_poisson_objective(data,model.evaluate(count,fit.theta)[0]),atol=1e-8)
    # This fixed noisy draw exercises absent-broad restarts; it is a regression
    # control, not a claim of posterior calibration or ground-truth recovery.
    assert fits[2].objective<218.762


def test_outdated_extension_fails_explicitly_without_python_fallback(contract,monkeypatch):
    import spotsolve_rs
    model,_=contract
    monkeypatch.setattr(spotsolve_rs,'CalibratedModel',None)
    with pytest.raises(RuntimeError,match='rebuild spotsolve_rs'):
        fit_component(np.zeros(model.shape),model)


def test_aguet_proposals_preserve_empty_search_nesting_and_user_centres(contract):
    model, _ = contract
    native = prepare_rust(model)
    data = np.zeros(model.shape)
    options = dict(max_iter=80, screen_iter=8, keep_screened=1)
    fits = native.fit_component(data, np.empty((0, 2)), seed_sigma=model.seed_sigma,
                                proposal_method='aguet', **options)
    assert fits[0]['proposal_centres'] == []
    # An empty significance map still fits all nested hypotheses and K=2 splits.
    assert fits[1]['starts'] == 4
    assert fits[2]['starts'] == 40
    np.testing.assert_allclose([f['objective'] for f in fits], data.size * 1e-4, atol=1e-8)
    supplied = native.fit_component(data, np.array([[7., 7.], [7., 7.]]),
                                    proposal_method='aguet', **options)
    assert supplied[0]['proposal_centres'] == [[7., 7.]]
    assert supplied[1]['starts'] == 6
    # The public adapter must use the same native policy.
    public = fit_component(data, model, proposal_method='aguet', options=FitOptions(**options))
    assert [f.starts for f in public.fits] == [f['starts'] for f in fits]


@pytest.mark.parametrize("centre", [7., 10.])
def test_aguet_proposes_isolated_peak_and_keeps_null_hypothesis_identical(contract, centre):
    model, values = contract
    native = prepare_rust(model)
    theta = values['theta'][:23].copy()
    theta[:16] = .5
    theta[16] = 0.
    theta[20:23] = [2., centre, centre]
    data = model.evaluate(1, theta)[0]
    options = dict(max_iter=200, screen_iter=24, keep_screened=2)
    aguet = native.fit_component(data, np.empty((0, 2)), seed_sigma=model.seed_sigma,
                                 proposal_method='aguet', **options)
    moments = native.fit_component(data, np.empty((0, 2)), seed_sigma=model.seed_sigma, **options)
    assert aguet[0]['proposal_centres'] == [[centre, centre]]
    np.testing.assert_array_equal(aguet[0]['theta'], moments[0]['theta'])
    assert aguet[0]['objective'] == moments[0]['objective']
    np.testing.assert_allclose(aguet[1]['theta'][21:23], [centre, centre], atol=2e-4)
    assert aguet[1]['objective'] < 1e-8
    assert min(-np.diff([f['objective'] for f in aguet])) >= -1e-8


@pytest.mark.parametrize('settings', [
    {'proposal_method': 'unknown'},
    {'proposal_method': 'aguet', 'proposal_alpha': 0.},
    {'proposal_method': 'aguet', 'proposal_alpha': 1.},
    {'proposal_method': 'aguet', 'proposal_alpha': np.nan},
    {'proposal_method': 'aguet', 'seed_sigma': 1e100},
])
def test_native_proposal_settings_are_validated(contract, settings):
    model, _ = contract
    with pytest.raises(ValueError):
        prepare_rust(model).fit_component(np.zeros(model.shape), np.empty((0, 2)), **settings)
