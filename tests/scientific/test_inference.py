"""Contracts for the retained independent reference and native port fixture."""
import subprocess
import sys
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from spotsolve.inference import FocusedModel
from spotsolve.inference.reference import FitOptions, fit_component
from spotsolve.inference.reference.profiled import profile_at, solve_poisson_linear
from spotsolve.inference.model import evaluate_component
from .test_psf_bank import bank


def make_model(bank):
    return FocusedModel((21, 21), (7, 7, 13, 13), bank, seed_sigma=1.2)


@pytest.mark.parametrize("amplitudes", [(2., .9, .4), (0., .9, 0.)])
def test_one_pass_profile_matches_independent_two_pass_calculation(bank, monkeypatch, amplitudes):
    model = make_model(bank)
    theta = np.r_[np.linspace(.3, .6, 16), amplitudes[0], 11.1, 10.7, .337,
                  amplitudes[1], 10.2, 9.7, amplitudes[2], 9.3, 10.7]
    data = model.evaluate(2, theta)[0]
    trial = theta.copy(); trial[-1] += .05
    ids = model.affine_indices(2)
    reference = trial.copy(); reference[ids] = 0
    offset, jac = model.evaluate(2, reference)
    lo, hi = model.parameter_bounds(data, 2)
    linear = solve_poisson_linear(data, jac.reshape(-1, len(theta))[:, ids], offset,
        trial[ids], lo[ids], hi[ids], constraints=model.rate_constraints(2)[:, ids])
    reference = trial.copy(); reference[ids] = linear.coefficients
    expected_mean, expected_jac = model.evaluate(2, reference)
    expected_gradient = expected_jac.reshape(-1, len(theta)).T @ (1-data.ravel()/expected_mean.ravel())
    calls = []
    original = type(bank).evaluate
    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(type(bank), "evaluate", counted)
    fitted, result, gradient = profile_at(data, model, 2, trial)
    assert len(calls) == 3  # one broad + two focus, formerly six evaluations
    assert result.converged
    np.testing.assert_allclose(result.objective, linear.objective, atol=1e-10)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-7)
    np.testing.assert_allclose(model.evaluate(2, fitted)[0], expected_mean, atol=2e-8)


def test_native_import_does_not_load_reference_or_removed_prototype():
    subprocess.run([sys.executable, "-c",
        "import sys; import spotsolve.inference; "
        "assert not any(n.startswith(('spotsolve.prototype', 'spotsolve.inference.reference')) for n in sys.modules)"], check=True)


def test_retained_public_count_search_preserves_nesting(bank):
    model = FocusedModel((11, 11), (3, 3, 7, 7), bank, seed_sigma=1.2)
    theta = np.r_[np.full(16, .4), .7, 5.3, 5.7, .4, .9, 5.2, 4.7]
    data = evaluate_component(1, theta, model)[0]
    result = fit_component(data, model, options=FitOptions(max_iter=80, screen_iter=12, keep_screened=2))
    assert result[1].objective < 1e-3
    assert np.min(result.likelihood_gains) >= -1e-8
    np.testing.assert_allclose(result[1].positions, [[5.2, 4.7]], atol=.01)


def test_frozen_port_fixture():
    from scripts.check_inference import load_contract, check_contract
    path = Path(__file__).resolve().parents[1]/"fixtures/08_inference.npz"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
    check_contract(*load_contract(path))
