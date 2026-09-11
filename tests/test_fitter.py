"""The Rust variable-width fitter against the frozen `02_lmga` fixture, and
against `psf`'s analytic model and Jacobian."""

import json
from pathlib import Path

import numpy as np
import pytest

from spotsolve import psf

rs = pytest.importorskip("spotsolve_rs")


def _i_divergence(d, m):
    d = np.asarray(d, float)
    term = np.where(d > 0, d * np.log(np.where(d > 0, d, 1.0) / m), 0.0)
    return float(np.sum(term - (d - m)))


def test_objective_and_information_match_the_reference():
    fixture = json.loads((Path(__file__).parent / "fixtures/02_lmga.json").read_text())
    for case in fixture["var_sigma_cases"]:
        h, w = case["h"], case["w"]
        data = np.ascontiguousarray(case["data"], dtype=float)
        theta, i_div, fit_f, _, converged, _ = rs.lmcl_fit_var_sigma(
            np.asarray(case["theta0"], float), h, w, data, np.zeros((h, w)),
            np.asarray(case["lower"], float), np.asarray(case["upper"], float),
            100)
        # No worse than the retired Python reference's optimum.
        assert i_div <= case["ml"]["I"] + 1e-6
        yy, xx = np.mgrid[:h, :w].astype(float)
        model = psf.model_var_sigma(theta, yy, xx)
        jac = psf.jac_var_sigma(theta, yy, xx).reshape(-1, len(theta))
        fisher = jac.T @ (jac / model.reshape(-1, 1))
        grad = jac.T @ (1 - data.ravel() / model.ravel())
        np.testing.assert_allclose(fit_f, fisher, rtol=1e-10, atol=1e-10)
        assert i_div == pytest.approx(_i_divergence(data, model), rel=0,
                                      abs=1e-10)
        if converged:
            room = np.where(grad >= 0, theta - case["lower"],
                            case["upper"] - theta)
            score = np.max(np.abs(grad) * np.minimum(room, 1 / np.sqrt(np.diag(fisher))))
            assert score <= np.sqrt(2e-8) * (1 + 1e-6)


@pytest.mark.parametrize("bad", ["bounds", "sigma", "shape", "tolerance"])
def test_invalid_native_fit_arguments_raise_value_error(bad):
    kwargs = dict(theta0=np.array([3., 400., 4., 4., 1.2]), h=9, w=10,
                  d=np.ones((9, 10)), halo=np.zeros((9, 10)),
                  lower=np.array([0., 0., -0.5, -0.5, 0.84]),
                  upper=np.array([100., 5000., 8.5, 9.5, 2.64]))
    if bad == "bounds":
        kwargs["lower"] = np.zeros(2)
    elif bad == "sigma":
        kwargs["lower"][-1] = 0.
    elif bad == "shape":
        kwargs["d"] = np.ones((10, 9))
    else:
        kwargs["tol_obj"] = float("nan")
    with pytest.raises(ValueError):
        rs.lmcl_fit_var_sigma(**kwargs)
