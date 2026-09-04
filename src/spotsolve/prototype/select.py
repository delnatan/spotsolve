"""Calibrated decisions built on explicit local-hypothesis fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Hypothesis


@dataclass(frozen=True)
class SelectionStatistics:
    focus_gain: float
    pair_gain: float
    pair_eligible: bool


@dataclass(frozen=True)
class SelectionDecision:
    hypothesis: Hypothesis
    n_focus: int
    focus_p: float
    pair_p: float | None
    statistics: SelectionStatistics
    nuisance_winner: Hypothesis


def selection_statistics(fits):
    """Likelihood gains for focused-light existence and pair multiplicity."""
    nuisance = min(fits[Hypothesis.H0].objective,
                   fits[Hypothesis.HSMOOTH].objective,
                   fits[Hypothesis.HWIDE].objective)
    pair = fits[Hypothesis.H2]
    pair_eligible = not pair.at_boundary and not pair.collapsed
    focused = min(
        fits[Hypothesis.H1].objective,
        pair.objective if pair_eligible else np.inf,
    )
    simple = min(nuisance, fits[Hypothesis.H1].objective)
    return SelectionStatistics(
        focus_gain=float(nuisance - focused),
        pair_gain=(float(simple - pair.objective) if pair_eligible else 0.0),
        pair_eligible=pair_eligible,
    )


def select_local(fits, calibration, *, alpha_focus=0.01, alpha_pair=0.01):
    """Select nuisance, one focus emitter, or a focused pair.

    The existence test is conservative across background, smooth-background,
    and wide-source nulls. Conditional
    on accepting focused light, the multiplicity test is conservative across
    all nuisance nulls and H1. Some nuisance classes may look redundant after
    the first test, but retaining them protects the composed decision against
    finite-sample leakage between stages. Boundary-railed or collapsed H2 fits
    contribute no pair evidence.
    """
    if not 0 < alpha_focus < 1 or not 0 < alpha_pair < 1:
        raise ValueError("alpha values must lie strictly between zero and one")
    calibration.check_compatible(fits)
    stats = selection_statistics(fits)
    nuisance_winner = min(
        (Hypothesis.H0, Hypothesis.HSMOOTH, Hypothesis.HWIDE),
        key=lambda kind: fits[kind].objective,
    )
    focus_p = calibration.conservative_p(
        "focus", stats.focus_gain,
        (Hypothesis.H0, Hypothesis.HSMOOTH, Hypothesis.HWIDE))
    if focus_p > alpha_focus:
        return SelectionDecision(
            hypothesis=nuisance_winner,
            n_focus=0,
            focus_p=focus_p,
            pair_p=None,
            statistics=stats,
            nuisance_winner=nuisance_winner,
        )

    pair_p = calibration.conservative_p(
        "pair", stats.pair_gain,
        (Hypothesis.H0, Hypothesis.HSMOOTH,
         Hypothesis.H1, Hypothesis.HWIDE))
    if pair_p <= alpha_pair:
        hypothesis, count = Hypothesis.H2, 2
    else:
        hypothesis, count = Hypothesis.H1, 1
    return SelectionDecision(
        hypothesis=hypothesis,
        n_focus=count,
        focus_p=focus_p,
        pair_p=pair_p,
        statistics=stats,
        nuisance_winner=nuisance_winner,
    )
