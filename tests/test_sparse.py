"""Public contracts for the native sparse-field reference localizer."""

import numpy as np
import pytest

import spotsolve
from spotsolve import psf


def _render(shape, sigma, positions, amplitudes, background=4.0, sigmas=None):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    if sigmas is None:
        theta = psf.pack(background, amplitudes, positions[:, 0], positions[:, 1])
        return psf.model(theta, yy, xx, sigma)
    theta = psf.pack_var_sigma(
        background, amplitudes, positions[:, 0], positions[:, 1], sigmas
    )
    return psf.model_var_sigma(theta, yy, xx)


@pytest.fixture(scope="module", autouse=True)
def native_sparse():
    extension = pytest.importorskip("spotsolve_rs")
    if not hasattr(extension, "localize_sparse"):
        pytest.skip("installed extension predates sparse localization")


def test_fixed_width_recovers_separated_emitters_and_uncertainty():
    shape = (41, 43)
    sigma = 1.2
    positions = np.array([[10.25, 11.7], [29.1, 31.35]])
    amplitudes = np.array([900.0, 650.0])
    image = _render(shape, sigma, positions, amplitudes)
    result = spotsolve.localize_sparse(image, sigma)
    order = np.argsort(result.positions[:, 0])
    np.testing.assert_allclose(result.positions[order], positions, atol=1e-4)
    np.testing.assert_allclose(result.amplitudes[order], amplitudes, atol=2e-3)
    np.testing.assert_array_equal(result.fit_sigma, sigma)
    assert result.status == ("conditional_fisher", "conditional_fisher")
    assert np.all(np.isfinite(result.se))
    assert np.all(result.p_value < result.alpha)
    np.testing.assert_allclose(result.model_image + result.residual, image)


def test_fitted_width_recovers_isolated_widths():
    shape = (45, 45)
    sigma = 1.2
    positions = np.array([[12.4, 12.1], [32.2, 31.6]])
    amplitudes = np.array([1100.0, 800.0])
    widths = np.array([1.05, 1.65])
    image = _render(shape, sigma, positions, amplitudes, sigmas=widths)
    result = spotsolve.localize_sparse(
        image,
        sigma,
        fit_sigma=True,
        sigma_bounds=(0.7, 1.8),
    )
    order = np.argsort(result.positions[:, 0])
    np.testing.assert_allclose(result.positions[order], positions, atol=2e-4)
    np.testing.assert_allclose(result.fit_sigma[order], widths, atol=2e-4)
    assert np.all(np.isfinite(result.se))


def test_blank_image_and_input_validation():
    result = spotsolve.localize_sparse(np.full((15, 15), 4.0))
    assert result.positions.shape == (0, 2)
    assert result.se.shape == (0, 3)
    np.testing.assert_array_equal(result.residual, 0.0)
    for bad in (np.ones(4), np.ones((2, 4)), np.full((5, 5), np.nan)):
        with pytest.raises(ValueError):
            spotsolve.localize_sparse(bad)
    with pytest.raises(ValueError):
        spotsolve.localize_sparse(np.ones((5, 5)), gain=0)
    for alpha in (0.0, 1.0, np.nan):
        with pytest.raises(ValueError):
            spotsolve.localize_sparse(np.ones((15, 15)), alpha=alpha)
    with pytest.raises(ValueError):
        spotsolve.localize_sparse(
            np.ones((5, 5)), fit_sigma=True, sigma_bounds=(1.1, 0.9)
        )


def test_aguet_significance_rejects_poisson_background_peaks():
    rng = np.random.default_rng(4)
    false_frames = 0
    for _ in range(100):
        result = spotsolve.localize_sparse(
            rng.poisson(4.0, size=(25, 25)).astype(float), 1.2
        )
        false_frames += bool(len(result.positions))
    assert false_frames <= 5


def test_aguet_statistic_matches_the_closed_form_test():
    from scipy.stats import norm, t as student_t

    sigma = 1.2
    shape = (25, 25)
    mean = _render(
        shape, sigma, np.array([[12.0, 12.0]]), np.array([80.0]), background=4.0
    )
    image = np.random.default_rng(19).poisson(mean).astype(float)
    result = spotsolve.localize_sparse(image, sigma)
    nearest = np.argmin(np.linalg.norm(result.positions - [12.0, 12.0], axis=1))
    y, x = np.rint(result.positions[nearest]).astype(int)

    radius = int(np.ceil(4.0 * sigma))
    axis = np.arange(-radius, radius + 1)
    gaussian = np.exp(-(axis**2) / (2.0 * sigma**2))
    kernel = gaussian[:, None] * gaussian[None, :]
    patch = image[
        y - radius : y + radius + 1, x - radius : x + radius + 1
    ]
    n = patch.size
    gsum = kernel.sum()
    g2sum = np.square(kernel).sum()
    amplitude = (
        np.sum(patch * kernel) - gsum * np.sum(patch) / n
    ) / (g2sum - gsum**2 / n)
    background = (np.sum(patch) - amplitude * gsum) / n
    rss = np.square(patch - amplitude * kernel - background).sum()
    c00 = 1.0 / (g2sum - gsum**2 / n)
    k_level = norm.ppf(1.0 - 0.05 / 2.0)
    sigma_a2 = rss / (n - 3.0) * c00
    sigma_res = np.sqrt(rss / (n - 1.0))
    se_sigma_c = sigma_res / np.sqrt(2.0 * (n - 1.0)) * k_level
    sigma_c2 = se_sigma_c**2
    degrees = (n - 1.0) * (sigma_a2 + sigma_c2) ** 2 / (
        sigma_a2**2 + sigma_c2**2
    )
    statistic = (amplitude - sigma_res * k_level) / np.sqrt(
        (sigma_a2 + sigma_c2) / n
    )
    np.testing.assert_allclose(result.test_statistic[nearest], statistic, rtol=2e-12)
    np.testing.assert_allclose(
        result.p_value[nearest], student_t.sf(statistic, degrees), rtol=2e-12
    )


def test_repeated_draw_spread_agrees_with_reported_position_uncertainty():
    shape = (25, 25)
    sigma = 1.2
    position = np.array([12.25, 11.7])
    image = _render(
        shape, sigma, position[None, :], np.array([700.0]), background=4.0
    )
    rng = np.random.default_rng(123)
    errors = []
    uncertainties = []
    for _ in range(100):
        result = spotsolve.localize_sparse(
            rng.poisson(image).astype(float), sigma
        )
        nearest = np.argmin(np.linalg.norm(result.positions - position, axis=1))
        errors.append(result.positions[nearest] - position)
        uncertainties.append(result.se[nearest, 1:])
    empirical = np.std(errors, axis=0, ddof=1)
    reported = np.median(uncertainties, axis=0)
    np.testing.assert_allclose(empirical / reported, 1.0, atol=0.2)


def test_sparse_api_has_no_python_optimizer_fallback(monkeypatch):
    image = _render(
        (21, 21), 1.2, np.array([[10.2, 9.8]]), np.array([700.0])
    )
    import spotsolve.lmga

    monkeypatch.setattr(
        spotsolve.lmga,
        "fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("called the Python optimizer")
        ),
    )
    assert len(spotsolve.localize_sparse(image).positions) == 1


def test_isolated_source_agrees_with_calibrated_dense_fit():
    from scipy.special import ndtr
    from spotsolve.inference import FocusedModel, PixelPSFBank, fit_component

    depths = np.linspace(0, 0.6, 7)
    offsets = np.arange(-22, 22.01, 0.5)
    responses = []
    for depth in depths:
        width = 1.2 + 3 * depth
        axis = ndtr((offsets + 0.5) / width) - ndtr((offsets - 0.5) / width)
        responses.append(np.maximum(axis[:, None] * axis[None, :], 1e-100))
    bank = PixelPSFBank(depths, offsets, np.asarray(responses))
    model = FocusedModel((21, 21), (5, 5, 15, 15), bank, seed_sigma=1.2)
    theta = np.r_[np.full(16, 0.4), 0.0, 10.0, 10.0, 0.25,
                  0.9, 10.25, 9.7]
    image = model.evaluate(1, theta)[0]
    dense = fit_component(image, model, candidate_centres=[[10.25, 9.7]])[1]
    sparse = spotsolve.localize_sparse(image, 1.2)
    assert len(sparse.positions) == 1
    np.testing.assert_allclose(sparse.positions, dense.positions, atol=2e-4)
    np.testing.assert_allclose(sparse.amplitudes, dense.fluxes, rtol=2e-5)
