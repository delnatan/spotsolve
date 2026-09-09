"""Local fit records and numerical search budgets.

Count fits are hypotheses, not selected sources or posterior probabilities.
Penalty fields on records are retained for historical comparison adapters;
the active inference path fits the unpenalized Poisson likelihood only.
"""
from dataclasses import dataclass
import numpy as np

FLUX_UNIT = 1000.0

@dataclass(frozen=True)
class FitOptions:
    max_iter: int = 400
    screen_iter: int = 48
    keep_screened: int = 4
    gtol: float = 1e-6

@dataclass
class CountFit:
    n_focus: int
    theta: np.ndarray
    objective: float
    optimizer_success: bool
    projected_gradient_max: float
    boundary_parameters: tuple[int, ...]
    evaluations: int
    starts: int
    nuisance_size: int
    focus_penalty: float = 0.0
    inner_iterations: int = 0
    inner_retries: int = 0

    @property
    def penalized_objective(self):
        return self.objective + self.focus_penalty

    @property
    def positions(self):
        return self.theta[self.nuisance_size:].reshape(self.n_focus, 3)[:, 1:].copy()

    @property
    def fluxes(self):
        return FLUX_UNIT * self.theta[self.nuisance_size:].reshape(self.n_focus, 3)[:, 0]


@dataclass
class ComponentFits:
    model: object
    fits: tuple[CountFit, CountFit, CountFit]
    elapsed_s: float
    focus_rate: float = 0.0

    def __getitem__(self, count):
        if count not in (0, 1, 2):
            raise ValueError("count must be zero, one, or two")
        return self.fits[count]

    @property
    def likelihood_gains(self):
        """Raw likelihood differences at returned fits; NOT Bayes factors.

        With a nonzero focus_rate these are evaluated at penalized optima,
        not maximum likelihood gains, and need not be nonnegative.
        """
        return np.array([self[0].objective - self[1].objective,
                         self[1].objective - self[2].objective])


@dataclass
class PositionUncertainty:
    covariance: np.ndarray | None
    status: str
    nuisance_fixed_covariance: np.ndarray | None = None
    broad_conditioned_absent: bool = False
