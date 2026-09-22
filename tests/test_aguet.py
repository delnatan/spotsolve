"""Reference decisions, fit outputs, ROI semantics and thread determinism."""
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from spotsolve import localize_aguet, localize_aguet_stack
from spotsolve.aguet import _cutoff, _settings
from spotsolve.native import _rs
from spotsolve import loctable


def sampled_frame(seed=26, shape=(128, 128)):
    y, x = np.indices(shape)
    mean = np.full(shape, 20.)
    for yy, xx in [(30.2, 30.3), (60.4, 70.1), (90.2, 45.7)]:
        mean += 180 * np.exp(-((y-yy)**2+(x-xx)**2)/(2*1.45**2))
    return np.random.default_rng(seed).poisson(mean).astype(float)


def assert_same(a, b):
    for field in fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, strict=True)
        elif isinstance(x, float) and np.isnan(x):
            assert np.isnan(y)
        else:
            assert x == y


def test_candidate_seeds_match_frozen_spotfitlm_reference():
    fixture = json.loads((Path(__file__).parent / 'fixtures/09_aguet.json').read_text())
    for case in fixture['candidate_frames']:
        data = np.array(case['data'])
        result = localize_aguet(data, case['sigma'], boxsize=9, itermax=100)
        # One reference seed is too close to the border for its C patch read.
        expected = sorted(p for p in case['seeds'] if 4 <= p[0] < data.shape[0]-4
                          and 4 <= p[1] < data.shape[1]-4)
        actual = sorted(result.info['seed_positions'] + [p[:2] for p in result.info['failures']])
        assert actual == expected


@pytest.mark.parametrize('sigma', [.8, 1.45, 2.4])
@pytest.mark.parametrize('alpha', [.001, .05, .8])
def test_constant_cutoff_matches_original_pixelwise_t_test(sigma, alpha):
    radius = int(np.ceil(4*sigma))
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    g = np.exp(-(x*x+y*y)/(2*sigma*sigma)).ravel()
    design = np.column_stack((g, np.ones_like(g)))
    c00 = np.linalg.inv(design.T @ design)[0, 0]
    n = len(g)
    cutoff = _cutoff(sigma, alpha)
    sd = np.geomspace(.001, 100, 31)
    amplitude = sd * (cutoff + np.linspace(-1e-7, 1e-7, len(sd)))
    rss = sd**2 * (n-1)
    var_a = rss/(n-3) * c00
    k = stats.norm.ppf(1-alpha/2)
    var_b = (sd/np.sqrt(2*(n-1))*k)**2
    df = (n-1)*(var_a+var_b)**2/(var_a**2+var_b**2)
    t = (amplitude - sd*k)/np.sqrt((var_a+var_b)/n)
    # Ignore the exact floating-point boundary; pin either side closely.
    keep = np.arange(len(sd)) != len(sd)//2
    np.testing.assert_array_equal((stats.t.sf(t,df)<alpha)[keep], (amplitude>cutoff*sd)[keep])


def test_roi_crops_work_but_keeps_context_global_coordinates_and_fit_results():
    image = sampled_frame()
    full = localize_aguet(image, 1.45, images=True)
    roi = np.zeros(image.shape, bool)
    roi[27:34, 27:34] = True
    cropped = localize_aguet(image, 1.45, roi=roi, images=True)
    seed = np.array(full.info['seed_positions'])
    selected = roi[seed[:,0], seed[:,1]]
    assert selected.sum() == 1 and len(cropped) == 1
    for name in ('positions', 'amplitudes', 'fit_sigma', 'se', 'sigma_se'):
        np.testing.assert_array_equal(getattr(cropped, name), getattr(full, name)[selected])
    assert cropped.info['processed_pixels'] < image.size/10
    np.testing.assert_allclose(cropped.background[roi], full.background[roi], atol=1e-11)
    assert np.isnan(cropped.background[0,0])
    np.testing.assert_allclose(cropped.residual, image-cropped.model_image, equal_nan=True)
    for mask in (np.ones(image.shape,bool), np.eye(image.shape[0],dtype=bool),
                 np.indices(image.shape)[1] == 30):
        result = localize_aguet(image,1.45,roi=mask)
        selected = mask[seed[:,0],seed[:,1]]
        np.testing.assert_array_equal(result.positions,full.positions[selected])


def test_stack_threading_and_empty_inputs():
    stack = np.stack([sampled_frame(seed=i) for i in (14,26,32)])
    serial = localize_aguet_stack(stack,1.45,n_threads=1,images=True)
    parallel = localize_aguet_stack(stack,1.45,n_threads=3,images=True)
    for raw, a, b in zip(stack,serial,parallel):
        assert_same(a,b)
        assert_same(a,localize_aguet(raw,1.45,images=True))
    assert localize_aguet_stack(stack[:0],1.45) == []
    empty = localize_aguet(stack[0],1.45,roi=np.zeros(stack.shape[1:],bool))
    assert len(empty)==0 and empty.info['processed_pixels']==0
    assert empty.se.shape==(0,3) and empty.positions.shape==(0,2)
    for raw in (np.zeros((12,12)), np.full((12,12),20.), np.ones((3,3))):
        assert len(localize_aguet(raw,1.2)) == 0
    assert len(localize_aguet(stack[0],1.45,boxsize=301))==0


def test_flux_uncertainty_uses_amplitude_width_covariance_and_tables_work():
    image = sampled_frame()
    result = localize_aguet(image,1.45)
    settings = _settings(image.shape,1.45,None,0,.05,9,50,1)
    theta,cov,_,_,_ = _rs.aguet_localize_stack(image[None],**settings)[0]
    assert len(theta)>0
    for i,t in enumerate(theta):
        gradient=np.array([0.,0.,4*np.pi*t[3]*t[2],2*np.pi*t[2]**2,0.])
        np.testing.assert_allclose(result.se[i,0]**2,gradient@cov[i]@gradient,rtol=1e-12)
    tables = loctable.frame_tables(result,frame=0)
    assert tables[0].height==len(result)
    np.testing.assert_array_equal(tables[0]["bg"], theta[:, 4])
    np.testing.assert_array_equal(tables[0]["sigma_se"], result.sigma_se)
    assert np.isfinite(result.se).all() and np.all(result.se>0)
    shifted=localize_aguet(image+100,1.45,offset=100)
    assert_same(result,shifted)


@pytest.mark.parametrize('kw', [dict(sigma=0),dict(sigma=np.nan),dict(sigma=1000),
    dict(significance=0),dict(significance=1),dict(significance=np.nan),dict(boxsize=2),
    dict(boxsize=4),dict(boxsize=3.5),dict(itermax=0),dict(offset=np.inf)])
def test_invalid_parameters(kw):
    options=dict(sigma=1.2);options.update(kw)
    with pytest.raises(ValueError):localize_aguet(np.ones((16,16)),**options)
    with pytest.raises(ValueError):localize_aguet_stack(np.ones((2,16,16)),**options)


def test_invalid_shapes_masks_pixels_and_worker_counts():
    for image in (np.ones((0,12)),np.ones(12),np.full((12,12),np.nan)):
        with pytest.raises(ValueError):localize_aguet(image,1.2)
    with pytest.raises(ValueError):localize_aguet(np.ones((12,12)),1.2,roi=np.ones((2,2)))
    for threads in (0,-1,1.5):
        with pytest.raises(ValueError):localize_aguet_stack(np.ones((2,12,12)),1.2,n_threads=threads)


def test_failed_fits_are_not_detections_and_render_uses_sampled_psf():
    result = localize_aguet(sampled_frame(), 1.45, itermax=1)
    assert result.info['candidates'] > 0 and len(result) == 0
    assert len(result.info['failures']) == result.info['candidates']
    assert all(failure[2] < 0 for failure in result.info['failures'])
    params = np.array([[10.3, 11.2, 1.4, 60., 20.]])
    background = np.full((24, 25), 20.)
    model = _rs.aguet_render(params, background)
    y, x = np.indices(background.shape)
    expected = 20 + 60*np.exp(-((x-10.3)**2+(y-11.2)**2)/(2*1.4**2))
    np.testing.assert_allclose(model[7:15,7:15], expected[7:15,7:15], rtol=1e-14)
    with pytest.raises(ValueError):
        _rs.aguet_render(params, np.empty((0, 25)))
