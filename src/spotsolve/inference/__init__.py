"""Calibrated local inference in Rust, exposed through thin Python adapters.

Returns fixed-count hypotheses and conditional uncertainty, not detections or
posterior existence probabilities. The older full-frame detect API is unchanged.
Python numerical comparisons live explicitly in spotsolve.inference.reference.
"""
from .model import FocusedModel, evaluate_component
from .psf_bank import PixelPSFBank
from .types import FitOptions, CountFit, ComponentFits, PositionUncertainty
from .rust import fit_component, position_uncertainty, prepare_rust

__all__ = [
    "FocusedModel", "PixelPSFBank", "FitOptions", "CountFit", "ComponentFits",
    "fit_component", "evaluate_component", "PositionUncertainty",
    "position_uncertainty", "prepare_rust",
]
