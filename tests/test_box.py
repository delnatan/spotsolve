"""`box.localize_boxes`: the read-noise model and the ROI."""

import numpy as np
import pytest

from spotsolve.box import localize_boxes
from spotsolve.simulate import simulate

pytest.importorskip("spotsolve_rs")

SIGMA = 1.2


def _frame(density, background, read_noise, seed):
    # `simulate` rounds a density up to one emitter; 0 means an empty frame.
    count = dict(n_emitters=0) if density == 0 else dict(density=density)
    sim = simulate(shape=(64, 64), **count,
                   amplitude_range=(150.0, 500.0), background=background,
                   sigma=SIGMA, seed=seed)
    noise = np.random.default_rng(1000 + seed).standard_normal(sim.image.shape)
    return sim, sim.image + read_noise * noise


def test_read_noise_model_stops_single_pixel_inventions():
    # bg 1 e-, sigma_r 2.5 e-, no emitters: the Poisson model reads the
    # read-noise spikes as sources (13.3 per frame over six seeds).
    _, img = _frame(0.0, 1.0, 2.5, 17)
    poisson = localize_boxes(img, sigma=SIGMA, gain=1.0)
    shifted = localize_boxes(img, sigma=SIGMA, gain=1.0, read_noise=2.5)
    assert len(poisson.amplitudes) >= 5
    assert len(shifted.amplitudes) == 0
    # The shift is internal: what is reported is in the frame's own units.
    np.testing.assert_allclose(shifted.residual, img - shifted.model_image,
                               atol=1e-9)
    assert np.median(shifted.background) == pytest.approx(1.0, abs=0.3)


def test_full_roi_is_no_roi():
    _, img = _frame(0.02, 5.0, 0.0, 18)
    a = localize_boxes(img, sigma=SIGMA, gain=1.0)
    b = localize_boxes(img, sigma=SIGMA, gain=1.0,
                       roi=np.ones(img.shape, dtype=bool))
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.amplitudes, b.amplitudes)
    assert a.history == b.history


def test_roi_confines_the_search():
    _, img = _frame(0.03, 5.0, 0.0, 19)
    roi = np.zeros(img.shape, dtype=bool)
    roi[:, :24] = True
    full = localize_boxes(img, sigma=SIGMA, gain=1.0)
    part = localize_boxes(img, sigma=SIGMA, gain=1.0, roi=roi)
    # Placements are on ROI pixels, but a fit follows the light: a source
    # just outside whose wing crosses the ROI settles where it really is
    # (measured: 2.3 px out, on a true source). Its reach is the placement's.
    assert np.all(part.positions[:, 1] < 24 + 3 * SIGMA)
    if part.width_rejects is not None and len(part.width_rejects):
        assert np.all(part.width_rejects["x"] < 24 + 3 * SIGMA)
    # No box forms outside, so the work shrinks with the area searched.
    assert part.history[0]["search_fits"] < 0.6 * full.history[0]["search_fits"]
    # And what the ROI's interior holds is found either way.
    inner = full.positions[full.positions[:, 1] < 24 - 3 * SIGMA]
    d = np.linalg.norm(inner[:, None] - part.positions[None], axis=-1)
    assert np.all(d.min(axis=1) < 0.2)
