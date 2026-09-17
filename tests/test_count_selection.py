"""Behavior of the experimental group-count selection and its API."""

import numpy as np
import pytest

from spotsolve import localize, localize_stack
from spotsolve.metrics import match
from spotsolve.simulate import simulate


@pytest.mark.parametrize("kw", [dict(n_emitters=1),
                                dict(n_emitters=2, min_separation=5)])
def test_bic_keeps_real_sources_and_can_choose_background(kw):
    sim = simulate(shape=(32, 32), sigma=1.2, seed=100,
                   amplitude_range=(1500, 1500), **kw)
    res = localize(sim.image, sigma=1.2, selection="bic", count_penalty=2)
    m = match(sim.positions, res.positions, radius=0.3)
    assert m.n_matched == len(sim.positions) == len(res)
    assert np.all(np.isfinite(res.se)) and np.all(res.se > 0)
    assert res.info["selection_fits"] > 0
    np.testing.assert_allclose(res.residual, sim.image - res.model_image)
    # K=0 is a real competitor, including when candidates have strong signal.
    empty = localize(sim.image, sigma=1.2, selection="bic", count_penalty=1e6)
    assert len(empty) == 0 and len(empty.rejects) == 0
    assert empty.positions.shape == (0, 2)
    assert np.isfinite(empty.model_image).all()


def test_bic_stack_and_roi_preserve_the_frame_result():
    stack = np.stack([simulate(shape=(32, 32), n_emitters=6, seed=i,
                              amplitude_range=(300, 900)).image for i in (40, 41)])
    kw = dict(sigma=1.2, selection="bic", count_penalty=2, images=True)
    serial = localize_stack(stack, n_threads=1, **kw)
    parallel = localize_stack(stack, n_threads=2, **kw)
    for frame, a, b in zip(stack, serial, parallel):
        single = localize(frame, roi=np.ones(frame.shape, dtype=bool), **kw)
        for res in (a, b):
            np.testing.assert_array_equal(res.positions, single.positions)
            np.testing.assert_array_equal(res.amplitudes, single.amplitudes)
            np.testing.assert_array_equal(res.se, single.se)
            np.testing.assert_array_equal(res.model_image, single.model_image)
            for key in res.info:
                np.testing.assert_equal(res.info[key], single.info[key])


def test_bic_score_keeps_camera_scale_invariance():
    frame = simulate(shape=(32, 32), n_emitters=1, seed=100,
                     amplitude_range=(1500, 1500)).image
    a = localize(frame, sigma=1.2, selection="bic", count_penalty=2)
    b = localize(4 * frame + 100, sigma=1.2, offset=100,
                 selection="bic", count_penalty=2)
    np.testing.assert_allclose(a.positions, b.positions, atol=1e-3)
    np.testing.assert_allclose(4 * a.amplitudes, b.amplitudes, rtol=1e-3)


@pytest.mark.parametrize("selection,penalty", [("unknown", 0), ("bic", -1),
                                              ("bic", np.nan), ("fixed", np.inf)])
def test_count_options_are_validated_for_frames_and_stacks(selection, penalty):
    kw = dict(sigma=1.2, selection=selection, count_penalty=penalty)
    with pytest.raises(ValueError):
        localize(np.ones((16, 16)), **kw)
    with pytest.raises(ValueError):
        localize_stack(np.ones((1, 16, 16)), **kw)


def test_bic_empty_roi_and_empty_stack():
    res = localize(np.ones((16, 16)), sigma=1.2, selection="bic",
                   roi=np.zeros((16, 16), dtype=bool))
    assert len(res) == 0 and res.info["selection_fits"] == 0
    assert localize_stack(np.empty((0, 16, 16)), sigma=1.2, selection="bic") == []
