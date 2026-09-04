"""Bridge from full-frame proposals to fixed local hypothesis fits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import FitOptions, StartGrid, fit_hypotheses
from .models import Hypothesis
from .proposals import ProposalOptions, generate_proposals


@dataclass(frozen=True)
class ProposedFit:
    component_index: int
    proposal_index: int
    origin: tuple[int, int]
    fits: object


@dataclass(frozen=True)
class ProposedFitCollection:
    fitted: tuple[ProposedFit, ...]
    skipped_edge_components: int
    skipped_budget_components: int


@dataclass(frozen=True)
class SelectedComponent:
    proposed_fit: ProposedFit
    decision: object
    global_positions: np.ndarray


@dataclass(frozen=True)
class FrameSelection:
    proposals: object
    proposed_fits: ProposedFitCollection
    components: tuple[SelectedComponent, ...]

    @property
    def focused_positions(self):
        arrays = [item.global_positions for item in self.components
                  if len(item.global_positions)]
        return np.concatenate(arrays, axis=0) if arrays else np.empty((0, 2))


def fit_proposal_components(data, sigma, proposal_result, *, roi_size=13,
                            max_components=64, options=None, start_grid=None):
    """Fit one ranked representative from each connected proposal component.

    Stage 3 only establishes the proposal-to-local-fit bridge. Components with
    several candidates still receive one H0/Hsmooth/H1/Hwide/H2 contest; the
    reversible multi-candidate solver belongs to Stage 4. Near an edge, the
    fixed-size ROI is shifted inward instead of discarded, so the proposal
    search and its bootstrap see the same boundary behavior.
    """
    data = np.asarray(data, dtype=float)
    if roi_size < 3 or roi_size % 2 == 0:
        raise ValueError("roi_size must be an odd integer at least three")
    if max_components < 1:
        raise ValueError("max_components must be positive")
    if data.shape[0] < roi_size or data.shape[1] < roi_size:
        raise ValueError("data dimensions must be at least roi_size")
    options = FitOptions() if options is None else options
    start_grid = StartGrid() if start_grid is None else start_grid
    half = roi_size // 2
    fitted = []
    skipped_edge = 0
    components = proposal_result.components[:max_components]
    for component_index, component in enumerate(components):
        proposal_index = max(
            component.proposal_indices,
            key=lambda index: proposal_result.proposals[index].score)
        proposal = proposal_result.proposals[proposal_index]
        iy, ix = np.rint(proposal.centre).astype(int)
        y0 = int(np.clip(iy - half, 0, data.shape[0] - roi_size))
        x0 = int(np.clip(ix - half, 0, data.shape[1] - roi_size))
        y1, x1 = y0 + roi_size, x0 + roi_size
        local_centre = proposal.centre - np.array([y0, x0], dtype=float)
        fits = fit_hypotheses(
            data[y0:y1, x0:x1], sigma, centre=local_centre,
            options=options, start_grid=start_grid)
        fitted.append(ProposedFit(
            component_index=component_index,
            proposal_index=proposal_index,
            origin=(y0, x0),
            fits=fits,
        ))
    return ProposedFitCollection(
        fitted=tuple(fitted),
        skipped_edge_components=skipped_edge,
        skipped_budget_components=max(
            0, len(proposal_result.components) - max_components),
    )


def select_proposed_frame(data, sigma, calibration, *, alpha_focus=0.01,
                          alpha_pair=0.01):
    """Run the calibrated Stage 3 single-representative component path."""
    from .select import select_local

    spec = calibration.spec
    proposal_options = ProposalOptions(**spec.proposal_options)
    fit_options = FitOptions(**spec.fit_options)
    start_grid = StartGrid(**spec.start_grid)
    calibration.check_pipeline(
        np.asarray(data).shape, sigma, proposal_options,
        spec.roi_size, spec.max_components)
    proposals = generate_proposals(
        data, sigma, options=proposal_options)
    fitted = fit_proposal_components(
        data, sigma, proposals, roi_size=spec.roi_size,
        max_components=spec.max_components, options=fit_options,
        start_grid=start_grid)
    selected = []
    for proposed in fitted.fitted:
        decision = select_local(
            proposed.fits, calibration,
            alpha_focus=alpha_focus, alpha_pair=alpha_pair)
        if decision.hypothesis is Hypothesis.H1:
            local = proposed.fits[Hypothesis.H1].physical["positions"]
        elif decision.hypothesis is Hypothesis.H2:
            local = proposed.fits[Hypothesis.H2].physical["positions"]
        else:
            local = np.empty((0, 2))
        selected.append(SelectedComponent(
            proposed_fit=proposed, decision=decision,
            global_positions=local + np.asarray(proposed.origin)[None, :],
        ))
    return FrameSelection(
        proposals=proposals, proposed_fits=fitted,
        components=tuple(selected))
