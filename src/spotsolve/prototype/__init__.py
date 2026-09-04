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
from .models import Hypothesis, ROIGrid, evaluate
from .select import SelectionDecision, SelectionStatistics, select_local

__all__ = [
    "BootstrapCalibration",
    "CalibrationSpec",
    "CorrelatedHazeNull",
    "FitCollection",
    "FitOptions",
    "Hypothesis",
    "LocalFit",
    "NullModel",
    "ROIGrid",
    "StartGrid",
    "SelectionDecision",
    "SelectionStatistics",
    "calibrate_local",
    "evaluate",
    "fit_hypotheses",
    "select_local",
]
