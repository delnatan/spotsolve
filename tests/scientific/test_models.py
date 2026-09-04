import numpy as np
import pytest

from spotsolve.prototype.models import Hypothesis, ROIGrid, evaluate
from spotsolve.prototype.parameterize import pair_components


THETA = {
    Hypothesis.H0: np.array([np.log(4.0), 0.10, -0.08]),
    Hypothesis.HSMOOTH: np.array([np.log(4.0), 0.10, -0.08,
                                  0.12, -0.06, 0.09]),
    Hypothesis.H1: np.array([np.log(4.0), 0.10, -0.08,
                             np.log(500.0), 4.2, 5.1]),
    Hypothesis.HWIDE: np.array([np.log(4.0), 0.10, -0.08,
                                np.log(500.0), 4.2, 5.1, np.log(1.8)]),
    Hypothesis.H2: np.array([np.log(4.0), 0.10, -0.08,
                             np.log(900.0), 4.2, 5.1, 0.6, -0.7, 0.7]),
}


@pytest.mark.parametrize("hypothesis", list(Hypothesis))
def test_analytic_jacobian_matches_finite_difference(hypothesis):
    grid = ROIGrid.from_shape((9, 10))
    theta = THETA[hypothesis]
    mean, jac = evaluate(hypothesis, theta, grid, sigma=1.2)
    finite = np.empty_like(jac)
    for k in range(len(theta)):
        step = 1e-6 * max(abs(theta[k]), 1.0)
        plus, minus = theta.copy(), theta.copy()
        plus[k] += step
        minus[k] -= step
        finite[:, :, k] = (
            evaluate(hypothesis, plus, grid, 1.2)[0]
            - evaluate(hypothesis, minus, grid, 1.2)[0]
        ) / (2.0 * step)
    assert mean.shape == grid.shape
    assert np.all(mean > 0)
    np.testing.assert_allclose(jac, finite, rtol=2e-6, atol=2e-6)


def test_pair_parameterization_preserves_flux_centroid():
    theta = THETA[Hypothesis.H2]
    fluxes, positions = pair_components(theta)
    centroid = np.sum(fluxes[:, None] * positions, axis=0) / fluxes.sum()
    np.testing.assert_allclose(centroid, theta[4:6], atol=1e-14)
    assert fluxes[0] >= fluxes[1]
