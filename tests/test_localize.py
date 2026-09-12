"""The native detector against simulation truth.

The Python reference it was ported from was retired on 2026-09-11 (last
present in commit `ea6b17f`). Until then this file held the two to
statistical parity; at retirement they agreed exactly on every cell below --
counts, matches and the gain estimate.
"""

import numpy as np
import pytest

from spotsolve import native as L
from spotsolve.metrics import match
from spotsolve.simulate import simulate

rs = pytest.importorskip("spotsolve_rs")

SIGMA = 1.2


def _sim(seed, density=0.034, spread=0.2, **kw):
    return simulate(shape=(64, 64), density=density,
                    amplitude_range=(900.0, 1900.0), sigma=SIGMA,
                    sigma_spread=spread, seed=seed, **kw)


# Recall / precision within 1 px of truth. Recall is low by design:
# `sigma_spread` puts many true emitters outside the reporting band, and at
# 0.055 px^-2 neighbours closer than the PSF are not separable.
#
# Re-measured 2026-09-12 when `birth` moved from the frame-derived cut (~3.9
# here) to BIRTH_Z = 3.0. All three cells are BRIGHT and dense, which is the
# losing side of that trade -- the gate is load-bearing exactly where a
# mis-modelled bright emitter can pay ADD_NATS for a satellite. They are kept
# at these values rather than retuned because they are the arms that hold the
# gate honest; the faint arms it was moved for gain 4-6 F1 points and are
# covered by the sweep recorded at `BIRTH_Z`.
#
#   seed  density spread   true  found  matched   recall  prec   (was, birth 3.9)
#    17    0.015   0.4      47     35      32      .681   .914    .681 / 1.000
#    18    0.034   0.2     107     80      76      .710   .950    .710 /  .962
#    19    0.055   0.4     172     84      71      .413   .845    .436 /  .987
@pytest.mark.parametrize("seed,density,spread,recall,precision",
                         [(17, 0.015, 0.4, 0.681, 0.914),
                          (18, 0.034, 0.2, 0.710, 0.950),
                          (19, 0.055, 0.4, 0.413, 0.845)])
def test_referee_cells_hold_their_recall_and_precision(seed, density, spread,
                                                       recall, precision):
    sim = _sim(seed, density, spread)
    res = L.localize(sim.image, sigma=SIGMA, gain=1.0)
    m = match(sim.positions, res.positions, radius=1.0)
    assert m.recall >= recall - 0.05
    assert m.precision >= precision - 0.05
    # Every SE comes from the polish's Fisher matrix.
    assert np.all(np.isfinite(res.se)) and np.all(res.se > 0)


def test_read_noise_model_stops_single_pixel_inventions():
    # bg 1 e-, sigma_r 2.5 e-, no emitters: the plain Poisson model reads the
    # read-noise spikes as sources (13.3 per frame over six seeds).
    sim = simulate(shape=(64, 64), n_emitters=0, background=1.0, sigma=SIGMA,
                   seed=17)
    img = sim.image + 2.5 * np.random.default_rng(1017).standard_normal((64, 64))
    assert len(L.localize(img, sigma=SIGMA, gain=1.0).amplitudes) >= 5
    shifted = L.localize(img, sigma=SIGMA, gain=1.0, read_noise=2.5)
    assert len(shifted.amplitudes) == 0
    # The shift is internal: what is reported is in the frame's own units.
    np.testing.assert_allclose(shifted.residual, img - shifted.model_image,
                               atol=1e-9)
    assert np.median(shifted.background) == pytest.approx(1.0, abs=0.3)


def test_full_roi_is_no_roi():
    img = _sim(18).image
    a = L.localize(img, sigma=SIGMA, gain=1.0)
    b = L.localize(img, sigma=SIGMA, gain=1.0,
                   roi=np.ones(img.shape, dtype=bool))
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.amplitudes, b.amplitudes)
    assert a.info == b.info


def test_roi_confines_the_search():
    img = _sim(19).image
    roi = np.zeros(img.shape, dtype=bool)
    roi[:, :24] = True
    part = L.localize(img, sigma=SIGMA, gain=1.0, roi=roi)
    full = L.localize(img, sigma=SIGMA, gain=1.0)
    # Placements are on ROI pixels, but a fit follows the light: a source
    # just outside whose wing crosses the ROI settles where it really is
    # (measured: 2.3 px out, on a true source). Its reach is the placement's.
    assert np.all(part.positions[:, 1] < 24 + 3 * SIGMA)
    # No box forms outside, so the work shrinks with the area searched.
    assert part.info["search_fits"] < 0.6 * full.info["search_fits"]


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
    # The retired Python estimator's values on these frames (true gain 2.0;
    # at density 0.034 PSF tails already push it high). The port matched them
    # to 1e-15 once `filters::uniform_filter` used scipy's running sum.
    for seed, want in ((17, 2.301759069778574), (18, 2.163707060619947)):
        raw = 2.0 * _sim(seed, background=20.0).image + 100.0
        got = L.localize(raw, sigma=SIGMA, offset=100.0).gain
        assert got == pytest.approx(want, rel=1e-9)
    # A frame too narrow to high-pass falls back to unit gain.
    assert L.localize(np.full((8, 4), 120.0), sigma=SIGMA, offset=100.0).gain == 1.0


def test_the_roi_crop_is_invisible_to_the_roi():
    """`localize` runs FIND and the background on the ROI's bounding box plus
    `crop_margin` (41 px, set by the background's 25 px kernel), not on the
    frame. The margin's promise is that the answer inside the ROI does not
    depend on how much frame surrounds it.

    Checked here by handing the same ROI more context than the crop needs:
    the whole 192^2 frame against a sub-array that still contains the crop.
    Measured agreement on both: identical N, and positions to 1.3e-12 px --
    filter summation order over a differently-shaped array, eleven orders
    below the 5.7e-2 margin the candidate list carries (`filters`' module
    note).
    """
    sim = simulate(shape=(192, 192), density=0.01,
                   amplitude_range=(900.0, 1900.0), sigma=SIGMA,
                   sigma_spread=0.2, seed=23)
    img = sim.image
    # Both cuts are pinned: the caller otherwise derives them from the
    # array's shape, which is the thing varying here.
    for y0, x0, side in ((80, 80, 32), (0, 0, 40), (144, 100, 48)):
        roi = np.zeros(img.shape, dtype=bool)
        roi[y0:y0 + side, x0:x0 + side] = True
        full = L.localize(img, sigma=SIGMA, gain=1.0, roi=roi,
                          seed_threshold=4.0, birth_threshold=4.0)
        p = 41 + 20                       # the margin, with slack
        sy, sx = slice(max(0, y0 - p), y0 + side + p), slice(max(0, x0 - p), x0 + side + p)
        sub = L.localize(np.ascontiguousarray(img[sy, sx]), sigma=SIGMA, gain=1.0,
                         roi=np.ascontiguousarray(roi[sy, sx]),
                         seed_threshold=4.0, birth_threshold=4.0)
        assert len(full) == len(sub), f"N differs at ({y0}, {x0})"
        offset = np.array([sy.start, sx.start])
        np.testing.assert_allclose(full.positions, sub.positions + offset, atol=1e-9)
        np.testing.assert_allclose(full.amplitudes, sub.amplitudes, rtol=1e-9)


def test_an_empty_roi_asks_for_nothing():
    img = _sim(18).image
    res = L.localize(img, sigma=SIGMA, gain=1.0,
                     roi=np.zeros(img.shape, dtype=bool))
    assert len(res) == 0
    assert res.background.shape == img.shape
    assert res.info["candidates"] == 0 and res.info["boxes"] == 0
