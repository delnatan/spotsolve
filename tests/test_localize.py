"""The native box search against its reference, `box.localize_boxes`.

The gate is statistical parity, not a trajectory: the same count to 1% and
97% of emitters within 0.1 px. (Measured when written: exact counts and
every emitter matched on the referee frames, both real 256x256 frames, and
beads_60x_still bar one emitter.)
"""

import numpy as np
import pytest

from spotsolve import box, native as L
from spotsolve.metrics import match
from spotsolve.simulate import simulate

rs = pytest.importorskip("spotsolve_rs")

SIGMA = 1.2


def _sim(seed, density=0.034, spread=0.2, **kw):
    return simulate(shape=(64, 64), density=density,
                    amplitude_range=(900.0, 1900.0), sigma=SIGMA,
                    sigma_spread=spread, seed=seed, **kw)


def test_constants_agree_with_the_reference():
    L._check_constants(rs)


@pytest.mark.parametrize("seed,density,spread",
                         [(17, 0.015, 0.4), (18, 0.034, 0.2), (19, 0.055, 0.4)])
def test_native_matches_the_reference(seed, density, spread):
    img = _sim(seed, density, spread).image
    ref = box.localize_boxes(img, sigma=SIGMA, gain=1.0)
    nat = L.localize(img, sigma=SIGMA, gain=1.0)
    n = len(ref.amplitudes)
    assert abs(len(nat.amplitudes) - n) <= max(1, 0.01 * n)
    m = match(ref.positions, nat.positions, radius=0.1)
    assert m.n_matched >= 0.97 * n
    # The same classification rule reports both, so the rejects agree too.
    count = (lambda r: 0 if r.width_rejects is None else len(r.width_rejects))
    assert abs(count(nat) - count(ref)) <= max(1, 0.05 * count(ref))
    # And the SEs come from the same polish.
    j, k = m.matched_true_idx, m.matched_est_idx
    np.testing.assert_allclose(nat.se[k], ref.se[j], rtol=0.05)


def test_read_noise_and_roi_carry_over():
    sim = simulate(shape=(64, 64), n_emitters=0, background=1.0, sigma=SIGMA,
                   seed=17)
    img = sim.image + 2.5 * np.random.default_rng(1017).standard_normal((64, 64))
    assert len(L.localize(img, sigma=SIGMA, gain=1.0).amplitudes) >= 5
    shifted = L.localize(img, sigma=SIGMA, gain=1.0, read_noise=2.5)
    assert len(shifted.amplitudes) == 0
    np.testing.assert_allclose(shifted.residual, img - shifted.model_image,
                               atol=1e-9)

    img = _sim(19).image
    roi = np.zeros(img.shape, dtype=bool)
    roi[:, :24] = True
    part = L.localize(img, sigma=SIGMA, gain=1.0, roi=roi)
    full = L.localize(img, sigma=SIGMA, gain=1.0)
    # Placements are on ROI pixels, but a fit follows the light: a source
    # just outside whose wing crosses the ROI settles where it really is
    # (measured: 2.3 px out, on a true source). Its reach is the placement's.
    assert np.all(part.positions[:, 1] < 24 + 3 * SIGMA)
    assert part.history[0]["search_fits"] < 0.6 * full.history[0]["search_fits"]


def test_stack_is_frame_by_frame_and_thread_count_free():
    stack = np.stack([_sim(s).image for s in (17, 18, 19, 20, 21)])
    one = L.localize_stack(stack, sigma=SIGMA, gain=1.0, n_threads=1)
    many = L.localize_stack(stack, sigma=SIGMA, gain=1.0, n_threads=3)
    for t, (a, b) in enumerate(zip(one, many)):
        single = L.localize(stack[t], sigma=SIGMA, gain=1.0)
        for r in (a, b):
            np.testing.assert_array_equal(r.positions, single.positions)
            np.testing.assert_array_equal(r.amplitudes, single.amplitudes)
            np.testing.assert_array_equal(r.se, single.se)
        assert a.model_image is None and a.residual is None
    with_images = L.localize_stack(stack[:1], sigma=SIGMA, gain=1.0,
                                   images=True)[0]
    np.testing.assert_allclose(with_images.model_image,
                               L.localize(stack[0], sigma=SIGMA,
                                          gain=1.0).model_image)


def test_gain_estimate_is_the_reference_estimator():
    from spotsolve import calibrate
    for seed in (17, 18):
        raw = 2.0 * _sim(seed, background=20.0).image + 100.0
        got = L.localize(raw, sigma=SIGMA, offset=100.0).gain
        assert got == pytest.approx(calibrate.estimate_gain(raw, 100.0), rel=1e-9)
    # A frame too narrow to high-pass falls back to unit gain, as in Python.
    assert L.localize(np.full((8, 4), 120.0), sigma=SIGMA, offset=100.0).gain == 1.0
