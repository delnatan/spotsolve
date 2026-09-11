"""The Rust variable-width fitter against the Python reference's fixture."""

import json
from pathlib import Path

import numpy as np
import pytest

from spotsolve import psf
from spotsolve.deprecated import backend, lmga

rs = pytest.importorskip("spotsolve_rs")


def test_objective_and_information_match_the_reference():
    fixture = json.loads((Path(__file__).parent / "fixtures/02_lmga.json").read_text())
    for case in fixture["var_sigma_cases"]:
        r = backend.get("rs").fit_var_sigma(
            case["theta0"], case["h"], case["w"], case["data"], 0.0,
            case["lower"], case["upper"], 100)
        # No worse than the reference optimum.
        assert r.I <= case["ml"]["I"] + 1e-6
        yy, xx = np.mgrid[:case["h"], :case["w"]].astype(float)
        model = psf.model_var_sigma(r.theta, yy, xx)
        jac = psf.jac_var_sigma(r.theta, yy, xx).reshape(-1, len(r.theta))
        fisher = jac.T @ (jac / model.reshape(-1, 1))
        grad = jac.T @ (1 - np.asarray(case["data"]).ravel() / model.ravel())
        np.testing.assert_allclose(r.F, fisher, rtol=1e-10, atol=1e-10)
        assert r.I == pytest.approx(lmga.i_divergence(case["data"], model),
                                    rel=0, abs=1e-10)
        if r.converged:
            room = np.where(grad >= 0, r.theta - case["lower"],
                            case["upper"] - r.theta)
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
