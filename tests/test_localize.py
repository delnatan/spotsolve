"""The native detector against simulation truth.

The Python reference it was ported from was retired on 2026-09-11 (last
present in commit `ea6b17f`). The gain-free search was prototyped on a copy
of it and ported on 2026-09-14; the port matched the prototype's referee
numbers within spike-in noise.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

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
# Re-measured 2026-09-14 when the detector stopped taking a gain and read
# noise (seed 3.0, birth 2.5, significance-based width band). All three cells
# are BRIGHT and dense, the side of the operating point that pays for the dim
# recall the new cuts were chosen for; kept, not retuned, because they are
# the arms that hold precision honest.
#
#   seed  density spread   true  found   recall  prec   (was, 2026-09-12)
#    17    0.015   0.4      47     38     .745   .921    .681 / .914
#    18    0.034   0.2     107     94     .804   .915    .710 / .950
#    19    0.055   0.4     172    106     .488   .792    .413 / .845
@pytest.mark.parametrize("seed,density,spread,recall,precision",
                         [(17, 0.015, 0.4, 0.745, 0.921),
                          (18, 0.034, 0.2, 0.804, 0.915),
                          (19, 0.055, 0.4, 0.488, 0.792)])
def test_referee_cells_hold_their_recall_and_precision(seed, density, spread,
                                                       recall, precision):
    sim = _sim(seed, density, spread)
    res = L.localize(sim.image, sigma=SIGMA)
    m = match(sim.positions, res.positions, radius=1.0)
    assert m.recall >= recall - 0.05
    assert m.precision >= precision - 0.05
    # Every SE comes from the polish's Fisher matrix.
    assert np.all(np.isfinite(res.se)) and np.all(res.se > 0)
    assert np.all(np.isfinite(res.sigma_se)) and np.all(res.sigma_se > 0)


def test_read_noise_needs_no_model():
    # bg 1 e-, sigma_r 2.5 e-, no emitters. Plain Poisson read these spikes as
    # 13.3 sources per frame and needed the read noise passed in; the measured
    # noise already contains it.
    sim = simulate(shape=(64, 64), n_emitters=0, background=1.0, sigma=SIGMA,
                   seed=17)
    img = sim.image + 2.5 * np.random.default_rng(1017).standard_normal((64, 64))
    res = L.localize(img, sigma=SIGMA, band=None)
    assert len(res) == 0
    np.testing.assert_allclose(res.residual, img - res.model_image, atol=1e-9)
    assert np.median(res.background) == pytest.approx(1.0, abs=0.3)


def test_the_answer_does_not_depend_on_the_camera_gain():
    # The same photons at several gains (true 1 here): the measured dispersion
    # reads the gain exactly, and the detections are the same ones. Not bit
    # for bit -- rescaling perturbs rounding, and inside a crowded cluster the
    # search path is sensitive to that: measured, N stays within 2 at every
    # gain from 0.25 to 32 on all three referee cells, but on the dense ones a
    # cluster decomposes differently. On this sparse cell 43 of 43 match at
    # every gain but one, where 37 do.
    sim = _sim(17, 0.015, 0.4, background=20.0)
    ref = L.localize(sim.image + 100.0, sigma=SIGMA, offset=100.0)
    for g in (0.5, 4.0, 32.0):
        res = L.localize(g * sim.image + 100.0, sigma=SIGMA, offset=100.0)
        assert res.dispersion == pytest.approx(g * ref.dispersion, rel=1e-9)
        assert abs(len(res) - len(ref)) <= 2
        dist, j = cKDTree(res.positions).query(ref.positions)
        same = dist < 0.01
        assert same.mean() >= 0.8
        np.testing.assert_allclose(res.amplitudes[j[same]],
                                   g * ref.amplitudes[same], rtol=3e-2)
    assert 1.0 <= ref.dispersion <= 1.6


def test_full_roi_is_no_roi():
    img = _sim(18).image
    a = L.localize(img, sigma=SIGMA)
    b = L.localize(img, sigma=SIGMA,
                   roi=np.ones(img.shape, dtype=bool))
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.amplitudes, b.amplitudes)
    assert a.info == b.info


def test_roi_confines_the_search():
    img = _sim(19).image
    roi = np.zeros(img.shape, dtype=bool)
    roi[:, :24] = True
    part = L.localize(img, sigma=SIGMA, roi=roi)
    full = L.localize(img, sigma=SIGMA)
    # Placements are on ROI pixels, but a fit follows the light: a source
    # just outside whose wing crosses the ROI settles where it really is
    # (measured: 2.3 px out, on a true source). Its reach is the placement's.
    assert np.all(part.positions[:, 1] < 24 + 3 * SIGMA)
    # No box forms outside, so the work shrinks with the area searched.
    assert part.info["search_fits"] < 0.6 * full.info["search_fits"]


def test_stack_is_frame_by_frame_and_thread_count_free():
    stack = np.stack([_sim(s).image for s in (17, 18, 19, 20, 21)])
    one = L.localize_stack(stack, sigma=SIGMA, n_threads=1)
    many = L.localize_stack(stack, sigma=SIGMA, n_threads=3)
    for t, (a, b) in enumerate(zip(one, many)):
        single = L.localize(stack[t], sigma=SIGMA)
        for r in (a, b):
            np.testing.assert_array_equal(r.positions, single.positions)
            np.testing.assert_array_equal(r.amplitudes, single.amplitudes)
            np.testing.assert_array_equal(r.se, single.se)
        assert a.model_image is None and a.residual is None
    with_images = L.localize_stack(stack[:1], sigma=SIGMA,
                                   images=True)[0]
    np.testing.assert_allclose(with_images.model_image,
                               L.localize(stack[0], sigma=SIGMA).model_image)


def test_the_roi_crop_is_invisible_to_the_roi():
    """`localize` runs FIND and the background on the ROI's bounding box plus
    `crop_margin` (41 px, set by the background's 25 px kernel), not on the
    frame. The margin's promise is that the answer inside the ROI does not
    depend on how much frame surrounds it.

    Checked here by handing the same ROI more context than the crop needs:
    the whole 192^2 frame against a sub-array that still contains the crop.
    The noise map's medians are exact on a grid `NOISE_STRIDE` (12 px) apart
    in the INPUT array's coordinates, so the sub-array is cut on that grid:
    within one call the crop cannot see it, but an image shifted by a
    non-multiple of 12 samples its noise at different pixels.
    Measured agreement on both: identical N, and positions to 1.3e-12 px --
    filter summation order over a differently-shaped array, eleven orders
    below the 5.7e-2 margin the candidate list carries (`filters`' module
    note).
    """
    sim = simulate(shape=(192, 192), density=0.01,
                   amplitude_range=(900.0, 1900.0), sigma=SIGMA,
                   sigma_spread=0.2, seed=23)
    img = sim.image
    for y0, x0, side in ((80, 80, 32), (0, 0, 40), (144, 100, 48)):
        roi = np.zeros(img.shape, dtype=bool)
        roi[y0:y0 + side, x0:x0 + side] = True
        full = L.localize(img, sigma=SIGMA, roi=roi)
        p = 41 + 20                       # the margin, with slack
        a, b = max(0, (y0 - p) // 12 * 12), max(0, (x0 - p) // 12 * 12)
        sy, sx = slice(a, y0 + side + p), slice(b, x0 + side + p)
        sub = L.localize(np.ascontiguousarray(img[sy, sx]), sigma=SIGMA,
                         roi=np.ascontiguousarray(roi[sy, sx]))
        assert len(full) == len(sub), f"N differs at ({y0}, {x0})"
        offset = np.array([sy.start, sx.start])
        np.testing.assert_allclose(full.positions, sub.positions + offset, atol=1e-9)
        np.testing.assert_allclose(full.amplitudes, sub.amplitudes, rtol=1e-9)


def test_an_empty_roi_asks_for_nothing():
    img = _sim(18).image
    res = L.localize(img, sigma=SIGMA,
                     roi=np.zeros(img.shape, dtype=bool))
    assert len(res) == 0
    assert res.background.shape == img.shape
    assert res.info["candidates"] == 0 and res.info["boxes"] == 0
