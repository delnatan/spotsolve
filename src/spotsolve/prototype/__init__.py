"""Opt-in Python prototypes for the next detector architecture.

Nothing in this package is imported by :mod:`spotsolve` or used by
``spotsolve.detect``.  Its interfaces may change while the scientific model is
being validated.
"""

from .calibration import (
    BootstrapCalibration,
    CalibrationSpec,
    CorrelatedHazeNull,
    NullModel,
    calibrate_local,
)
from .fit import FitCollection, FitOptions, LocalFit, StartGrid, fit_hypotheses
from .frame import (
    FrameSelection,
    ProposedFit,
    ProposedFitCollection,
    SelectedComponent,
    fit_proposal_components,
    select_proposed_frame,
)
from .models import Hypothesis, ROIGrid, evaluate
from .proposals import (
    Proposal,
    ProposalComponent,
    ProposalOptions,
    ProposalResult,
    estimate_background,
    generate_proposals,
    poisson_score_map,
)
from .proposal_calibration import (
    ProposalBootstrapCalibration,
    ProposalCalibrationSpec,
    calibrate_proposal_search,
)
from .select import SelectionDecision, SelectionStatistics, select_local

__all__ = [
    "BootstrapCalibration",
    "CalibrationSpec",
    "CorrelatedHazeNull",
    "FitCollection",
    "FitOptions",
    "FrameSelection",
    "Hypothesis",
    "LocalFit",
    "NullModel",
    "Proposal",
    "ProposalBootstrapCalibration",
    "ProposalCalibrationSpec",
    "ProposalComponent",
    "ProposalOptions",
    "ProposalResult",
    "ProposedFit",
    "ProposedFitCollection",
    "SelectedComponent",
    "ROIGrid",
    "StartGrid",
    "SelectionDecision",
    "SelectionStatistics",
    "calibrate_local",
    "calibrate_proposal_search",
    "evaluate",
    "estimate_background",
    "fit_hypotheses",
    "fit_proposal_components",
    "generate_proposals",
    "poisson_score_map",
    "select_local",
    "select_proposed_frame",
]
