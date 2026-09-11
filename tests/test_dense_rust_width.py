"""Variable-width Rust fitting: objective quality, derivatives and dispatch."""

import json
from pathlib import Path

import numpy as np
import pytest

from spotsolve import backend, core, lmga, metrics, prior, psf
from spotsolve.simulate import simulate

rs = pytest.importorskip("spotsolve_rs")


@pytest.mark.parametrize("mode", ["ml", "map"])
def test_variable_width_objective_and_curvature(mode):
    fixture = json.loads((Path(__file__).parent / "fixtures/02_lmga.json").read_text())
    for case in fixture["var_sigma_cases"]:
        p = case["penalty"]
        wp = prior.FocusMixtureWidth(
            p["lam_focus"], p["lam_wide"], p["lo"], p["mid"], p["hi"],
            p["sigma0"], p["scale"] / p["sigma0"],
        )
        result = backend.get("rs").fit_var_sigma(
            case["theta0"], case["h"], case["w"], case["data"], 0.0,
            case["lower"], case["upper"], 100,
            wprior=wp if mode == "map" else None,
        )
        expected = case[mode]
        def objective(theta, data_i):
            return data_i - (np.sum(wp.logpdf(np.asarray(theta)[4::4]))
                             if mode == "map" else 0.0)
        assert objective(result.theta, result.I) <= objective(expected["theta"], expected["I"]) + 1e-6
        yy, xx = np.mgrid[:case["h"], :case["w"]].astype(float)
        model = psf.model_var_sigma(result.theta, yy, xx)
        jac = psf.jac_var_sigma(result.theta, yy, xx).reshape(-1, len(result.theta))
        fisher = jac.T @ (jac / model.reshape(-1, 1))
        grad = jac.T @ (1 - np.asarray(case["data"]).ravel() / model.ravel())
        if mode == "map":
            fisher += np.diag(core._WidthPenalty(wp, case["K"]).hess_diag(result.theta))
            u = (result.theta[4::4] - wp.sigma0) / wp.scale
            grad[4::4] += 2*u / (wp.scale * (1 + u*u))
        np.testing.assert_allclose(result.F, fisher, rtol=1e-10, atol=1e-10)
        assert result.I == pytest.approx(lmga.i_divergence(case["data"], model), rel=0, abs=1e-10)
        if result.converged:
            room = np.where(grad >= 0, result.theta - case["lower"], case["upper"] - result.theta)
            score = np.max(np.abs(grad) * np.minimum(room, 1/np.sqrt(np.diag(fisher))))
            assert score <= np.sqrt(2e-8) * (1 + 1e-6)


@pytest.mark.parametrize("k", [0, 2])
def test_halo_and_map_return_contract(k):
    h, w = 9, 12
    yy, xx = np.mgrid[:h, :w].astype(float)
    theta = np.array([3.0] + [400.0, 3.1, 4.8, 1.24, 600.0, 5.2, 7.3, 1.35][:4*k])
    lo = np.array([0.0] + [1e-4, -0.5, -0.5, 0.84] * k)
    hi = np.array([100.0] + [5000.0, h-0.5, w-0.5, 2.64] * k)
    halo = 0.1 * yy + 0.05 * xx
    model = psf.model_var_sigma(theta, yy, xx, halo)
    data = np.random.default_rng(42).poisson(model).astype(float)
    wp = core._width_prior(core.SIGMA_SLACK, core.FOCUS_BAND, 1.2, 0.01, 0.001)
    fit = backend.get("rs").fit_var_sigma(
        theta, h, w, data, halo, lo, hi, 0, wprior=wp,
    )
    # At zero iterations, directly check the returned likelihood and curvature
    # without assuming agreement between two optimizer trajectories.
    assert fit.n_iter == 0
    assert not fit.converged and not fit.stalled
    assert fit.I == pytest.approx(lmga.i_divergence(data, model), rel=0, abs=1e-10)
    jac = psf.jac_var_sigma(theta, yy, xx, halo).reshape(-1, len(theta))
    expected_f = jac.T @ (jac / model.reshape(-1, 1))
    expected_f += np.diag(core._WidthPenalty(wp, k).hess_diag(theta))
    np.testing.assert_allclose(fit.F, expected_f, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("band", [core.FOCUS_BAND, None])
def test_detect_variable_width_uses_rust_for_every_fit(monkeypatch, band):
    sim = simulate(shape=(29, 31), n_emitters=18, amplitude_range=(700, 1500),
                   background=5, sigma_spread=0.2, seed=17)
    opts = dict(sigma=sim.sigma, gain=1.0, verbose=0, band=band)

    def forbidden(*args, **kwargs):
        raise AssertionError("Rust detection called the Python optimizer")

    monkeypatch.setattr(lmga, "fit", forbidden)
    got = core.detect(sim.image, impl="rs", **opts)
    truth = sim.positions
    if band is not None:
        truth = truth[(sim.sigmas >= band[0]*sim.sigma) & (sim.sigmas <= band[1]*sim.sigma)]
    match = metrics.match(truth, got.positions, radius=1.2)
    assert match.precision >= 0.8
    assert match.recall >= 0.75
    assert match.rmse < 0.5
    assert np.isfinite(got.model_image).all()
    assert np.isfinite(got.fit_sigma).all()
    assert any(r["added"] for r in got.history)
    assert any(r["split"] for r in got.history)


@pytest.mark.parametrize("bad", ["bounds", "sigma", "prior", "shape", "tolerance"])
def test_invalid_native_fit_arguments_raise_value_error(bad):
    kwargs = dict(theta0=np.array([3., 400., 4., 4., 1.2]), h=9, w=10,
                  d=np.ones((9, 10)), halo=np.zeros((9, 10)),
                  lower=np.array([0., 0., -0.5, -0.5, 0.84]),
                  upper=np.array([100., 5000., 8.5, 9.5, 2.64]))
    if bad == "bounds":
        kwargs["lower"] = np.zeros(2)
    elif bad == "sigma":
        kwargs["lower"][-1] = 0.
    elif bad == "prior":
        kwargs["width_prior"] = (1.2, 0., 1.)
    elif bad == "shape":
        kwargs["d"] = np.ones((10, 9))
    else:
        kwargs["tol_obj"] = float("nan")
    with pytest.raises(ValueError):
        rs.lmcl_fit_var_sigma(**kwargs)
