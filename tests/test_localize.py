"""Native localization against simulation truth and diagnostic contracts."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from spotsolve import native as L
from spotsolve.metrics import match
from spotsolve.simulate import simulate

rs = pytest.importorskip("spotsolve_rs")

SIGMA = 1.2
# Simulated frames count photons.
GAIN = 1.0


def _sim(seed, density=0.034, spread=0.2, **kw):
    return simulate(shape=(64, 64), density=density,
                    amplitude_range=(900.0, 1900.0), sigma=SIGMA,
                    sigma_spread=spread, seed=seed, **kw)


# Widths are lognormal with sd 0.2-0.4, many wider than the default band,
# so the band is opened to its cap. Floors five points under what is achieved.
@pytest.mark.parametrize("seed,density,spread,recall,precision",
                         [(17, 0.015, 0.4, 0.894, 0.955),
                          (18, 0.034, 0.2, 0.748, 0.988),
                          (19, 0.055, 0.4, 0.593, 0.990)])
def test_crowded_fields_hold_their_recall_and_precision(seed, density, spread,
                                                        recall, precision):
    sim = _sim(seed, density, spread)
    res = L.localize(sim.image, sigma=SIGMA, gain=GAIN, fit_mixtures=True,
                     width=(1.0, float("inf")))
    m = match(sim.positions, res.positions, radius=1.0)
    assert m.recall >= recall - 0.05
    assert m.precision >= precision - 0.05
    assert np.all(res.info["detect_z"] >= res.info["u"])
    assert np.all(np.isfinite(res.se)) and np.all(res.se > 0)
    assert np.all(np.isfinite(res.sigma_se)) and np.all(res.sigma_se > 0)


@pytest.mark.parametrize("width", [(1.0, 1.0), (0.8, float("inf"))])
def test_separated_emitters_are_recovered(width):
    sim = simulate(shape=(96, 96), n_emitters=30, amplitude_range=(300.0, 3000.0),
                   sigma=SIGMA, sigma_spread=0.1, min_separation=10.0, seed=5)
    res = L.localize(sim.image, sigma=SIGMA, gain=GAIN, width=width)
    m = match(sim.positions, res.positions, radius=1.0)
    assert m.recall == 1.0 and m.precision == 1.0
    assert np.all(np.isfinite(res.se)) and np.all(res.se > 0)
    assert np.all(res.info["detect_z"] >= res.info["u"])
    if width[0] < width[1]:
        assert np.all(np.isfinite(res.sigma_se)) and np.all(res.sigma_se > 0)
    else:
        assert np.all(res.fit_sigma == SIGMA) and np.all(np.isnan(res.sigma_se))


# Mixtures resolve a close pair. With motion blur, a pair is two emitters
# only if it gains the bar over one elongated spot: at 3 sigma it does;
# closer, it is one elongated emitter at its centroid.
@pytest.mark.parametrize("motion_blur", [False, True])
@pytest.mark.parametrize("separation", [1.5, 2.0, 3.0])
def test_mixtures_resolve_a_pair_or_report_one_elongated_spot(separation, motion_blur):
    from spotsolve import psf, FitFlag
    yy, xx = np.mgrid[:48, :48]
    d = separation * SIGMA
    truth = np.array([[24.0, 24.0 - d / 2], [24.3, 24.0 + d / 2]])
    flux = np.array([1500., 1200.])
    mean = psf.model(psf.pack(20., flux, truth[:, 0], truth[:, 1]), yy, xx, SIGMA)
    image = np.random.default_rng(9).poisson(mean).astype(float)
    mixed = L.localize(image, SIGMA, gain=GAIN, fit_mixtures=True, motion_blur=motion_blur)
    assert np.all(mixed.info["mixture"] > 0)
    if separation >= 3.0 or not motion_blur:
        m = match(truth, mixed.positions, radius=min(1.0, d / 2))
        assert m.recall == 1.0 and len(mixed) == 2
        assert not np.any(mixed.flags & FitFlag.ELONGATED)
    else:
        centroid = flux @ truth / flux.sum()
        assert len(mixed) == 1 and mixed.flags[0] & FitFlag.ELONGATED
        assert np.hypot(*(mixed.positions[0] - centroid)) < 0.5
        assert mixed.amplitudes[0] == pytest.approx(flux.sum(), rel=0.05)


def test_read_noise_comes_from_the_background():
    # bg 1 e-, sigma_r 2.5 e-, no emitters: given the gain, the background's
    # variance gives the read variance.
    sim = simulate(shape=(64, 64), n_emitters=0, background=1.0, sigma=SIGMA,
                   seed=17)
    img = sim.image + 2.5 * np.random.default_rng(1017).standard_normal((64, 64))
    res = L.localize(img, sigma=SIGMA, gain=GAIN)
    assert len(res) == 0
    np.testing.assert_allclose(res.residual, img - res.model_image, atol=1e-9)
    assert np.median(res.background) == pytest.approx(1.0, abs=0.3)
    assert res.info["read_variance"] == pytest.approx(6.25, rel=0.25)


def test_the_answer_does_not_depend_on_the_camera_gain():
    # The same photons at several gains, each given: the detections agree to
    # within the fit's tolerance, and the measured gain scales with it.
    sim = _sim(17, 0.015, 0.4, background=20.0)
    ref = L.localize(sim.image + 100.0, sigma=SIGMA, gain=GAIN, offset=100.0)
    est = L.estimate_gain(sim.image + 100.0, SIGMA, offset=100.0)
    for g in (0.5, 4.0, 32.0):
        res = L.localize(g * sim.image + 100.0, sigma=SIGMA, gain=g * GAIN, offset=100.0)
        assert res.dispersion == g * GAIN
        assert L.estimate_gain(g * sim.image + 100.0, SIGMA, offset=100.0) == pytest.approx(g * est, rel=1e-9)
        assert len(res) == len(ref)
        assert np.all(np.abs(res.positions - ref.positions) < 0.01 * ref.se[:, 1:])
        assert np.all(np.abs(res.amplitudes - g * ref.amplitudes) < 0.01 * g * ref.se[:, 0])
        np.testing.assert_allclose(res.info["detect_z"], ref.info["detect_z"], rtol=1e-4)
    # Widths spread 0.4 scatter one small frame's gain; where the PSF is the
    # model's, it is the camera's.
    assert 0.9 <= L.estimate_gain(_sim(17, 0.015, 0.0, background=20.0).image, SIGMA) <= 1.2


@pytest.mark.parametrize("gain", [0.0, -1.0, float("nan")])
def test_the_gain_must_be_a_camera_gain(gain):
    with pytest.raises(ValueError, match="gain"):
        L.localize(np.ones((16, 16)), 1.2, gain=gain)


def test_full_roi_is_no_roi():
    img = _sim(18).image
    a = L.localize(img, sigma=SIGMA, gain=GAIN)
    b = L.localize(img, sigma=SIGMA, gain=GAIN,
                   roi=np.ones(img.shape, dtype=bool))
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.amplitudes, b.amplitudes)
    for key in a.info:
        np.testing.assert_equal(a.info[key], b.info[key])


def test_roi_confines_the_search():
    img = _sim(19).image
    roi = np.zeros(img.shape, dtype=bool)
    roi[:, :24] = True
    part = L.localize(img, sigma=SIGMA, gain=GAIN, roi=roi)
    full = L.localize(img, sigma=SIGMA, gain=GAIN)
    # Seeds are ROI pixels and a centre stays within 2 sigma of its seed.
    assert np.all(part.positions[:, 1] < 24 + 2 * SIGMA)
    assert part.info["seeds"] < 0.6 * full.info["seeds"]


def test_stack_shares_one_bar_and_is_thread_count_free():
    stack = np.stack([_sim(s).image for s in (17, 18, 19, 20, 21)])
    one = L.localize_stack(stack, sigma=SIGMA, gain=GAIN, n_threads=1)
    many = L.localize_stack(stack, sigma=SIGMA, gain=GAIN, n_threads=3)
    for a, b in zip(one, many):
        for key in ("positions", "amplitudes", "se", "flags"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
        np.testing.assert_array_equal(a.info["detect_z"], b.info["detect_z"])
        assert a.info["u"] == one[0].info["u"]
        assert np.all(a.info["detect_z"] >= a.info["u"])
        assert np.all(a.info["p_real"] >= L.P_REAL)
        assert a.model_image is None and a.residual is None
    # A frame alone is a movie of one.
    alone = L.localize(stack[0], sigma=SIGMA, gain=GAIN)
    first = L.localize_stack(stack[:1], sigma=SIGMA, gain=GAIN, images=True)[0]
    np.testing.assert_array_equal(first.positions, alone.positions)
    np.testing.assert_allclose(first.model_image, alone.model_image)


def test_the_roi_crop_is_invisible_to_the_roi():
    """An ROI is processed on its bounding box plus a margin, so more
    surrounding frame does not change the answer."""
    sim = simulate(shape=(192, 192), density=0.01,
                   amplitude_range=(900.0, 1900.0), sigma=SIGMA,
                   sigma_spread=0.2, seed=23)
    img = sim.image
    for y0, x0, side in ((80, 80, 32), (0, 0, 40), (144, 100, 48)):
        roi = np.zeros(img.shape, dtype=bool)
        roi[y0:y0 + side, x0:x0 + side] = True
        full = L.localize(img, sigma=SIGMA, gain=GAIN, roi=roi)
        p = 30                            # the margin, with room to spare
        a, b = max(0, y0 - p), max(0, x0 - p)
        sy, sx = slice(a, y0 + side + p), slice(b, x0 + side + p)
        sub = L.localize(np.ascontiguousarray(img[sy, sx]), sigma=SIGMA, gain=GAIN,
                         roi=np.ascontiguousarray(roi[sy, sx]))
        assert len(full) == len(sub), f"N differs at ({y0}, {x0})"
        offset = np.array([sy.start, sx.start])
        np.testing.assert_allclose(full.positions, sub.positions + offset, atol=1e-9)
        np.testing.assert_allclose(full.amplitudes, sub.amplitudes, rtol=1e-9)


def test_an_empty_roi_asks_for_nothing():
    img = _sim(18).image
    res = L.localize(img, sigma=SIGMA, gain=GAIN,
                     roi=np.zeros(img.shape, dtype=bool))
    assert len(res) == 0
    assert res.background.shape == img.shape
    assert res.info["seeds"] == 0
    assert res.info["detect_z"].shape == (0,)
    assert res.flags.shape == (0,)


def test_diagnostics_follow_all_returned_rows():
    image = _sim(17, density=0.015, spread=0.4).image
    raw = rs.detect_localize(image, sigma=SIGMA, gain=GAIN)
    result = L.localize(image, sigma=SIGMA, gain=GAIN, images=False)
    assert raw[-1]["detect_z"].shape == (len(raw[0]),)
    np.testing.assert_array_equal(result.info["detect_z"], raw[-1]["detect_z"])
    np.testing.assert_array_equal(result.positions, raw[0])
    np.testing.assert_array_equal(result.flags, raw[5])
    assert len(result) == len(raw[0])
    info = result.info
    assert info["seeds"] == (len(result) + info["weak"] + info["unconfined"]
                             + info["duplicates"] + info["out_of_focus"])
    assert np.all(info["mixture"] == 0)


def test_narrow_broad_and_bright_sources_keep_their_measurements():
    from spotsolve import psf, FitFlag
    yy, xx = np.mgrid[:96, :96]
    truth = np.array([[24., 24.], [48., 48.], [72., 72.]])
    widths = np.array([.73, 1.6, 1.0])
    flux = np.array([10000., 20000., 1000000.])
    theta = psf.pack_var_sigma(30., flux, truth[:, 0], truth[:, 1], widths)
    mean = psf.model_var_sigma(theta, yy, xx)
    image = np.random.default_rng(73).poisson(mean).astype(float)
    result = L.localize(image, 1.0, gain=GAIN, width=(0.6, float("inf")))
    distance, found = cKDTree(result.positions).query(truth)
    assert np.all(distance < .1)
    np.testing.assert_allclose(result.amplitudes[found], flux, rtol=.05)
    np.testing.assert_allclose(result.fit_sigma[found], widths, rtol=.05)
    assert np.all(result.flags[found] == int(FitFlag.OK))
    np.testing.assert_allclose(result.residual, image - result.model_image)


def test_edge_flag_does_not_depend_on_a_width_reporting_band():
    from spotsolve import psf, FitFlag
    yy, xx = np.mgrid[:48, :48]
    mean = psf.model(psf.pack(20., [15000.], [1.5], [24.]), yy, xx, 1.2)
    image = np.random.default_rng(21).poisson(mean).astype(float)
    result = L.localize(image, 1.2, gain=GAIN)
    distance, found = cKDTree(result.positions).query([[1.5, 24.]])
    assert distance[0] < .2
    assert result.flags[found[0]] & int(FitFlag.EDGE)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_native_wrapper_requires_positive_integer_counts(value):
    with pytest.raises(ValueError, match="n_threads"):
        L.localize_stack(np.ones((1, 16, 16)), 1.2, gain=GAIN, n_threads=value)


def test_a_watched_stack_gives_what_it_gives_unwatched():
    import threading
    stack = np.stack([_sim(s).image for s in (17, 18, 19)])
    plain = L.localize_stack(stack, SIGMA, gain=GAIN, fit_mixtures=True, n_threads=2)
    progress = L.Progress()
    seen = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            seen.append(progress.done)
    watcher = threading.Thread(target=watch)
    watcher.start()
    watched = L.localize_stack(stack, SIGMA, gain=GAIN, fit_mixtures=True, n_threads=2, progress=progress)
    stop.set()
    watcher.join()
    assert (progress.done, progress.total) == (6, 6)
    assert seen == sorted(seen)
    for a, b in zip(plain, watched):
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.info["detect_z"], b.info["detect_z"])
    cancelled = L.Progress()
    cancelled.cancel()
    with pytest.raises(L.Cancelled):
        L.localize_stack(stack, SIGMA, gain=GAIN, progress=cancelled)
