"""End-to-end calibration of maxima selected by proposal generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .calibration import CalibrationSpec, _null_from_dict
from .fit import FitOptions, StartGrid
from .frame import fit_proposal_components
from .models import Hypothesis
from .proposals import ProposalOptions, generate_proposals
from .select import selection_statistics


@dataclass(frozen=True)
class ProposalCalibrationSpec:
    frame_shape: tuple[int, int]
    roi_size: int
    sigma: float
    fit_options: dict
    start_grid: dict
    proposal_options: dict
    max_components: int
    noise_model: str = "poisson"

    @classmethod
    def create(cls, frame_shape, roi_size, sigma, fit_options, start_grid,
               proposal_options, max_components):
        return cls(
            frame_shape=tuple(map(int, frame_shape)),
            roi_size=int(roi_size), sigma=float(sigma),
            fit_options=asdict(fit_options), start_grid=asdict(start_grid),
            proposal_options=asdict(proposal_options),
            max_components=int(max_components),
        )

    @property
    def local_spec(self):
        return CalibrationSpec(
            shape=(self.roi_size, self.roi_size), sigma=self.sigma,
            fit_options=self.fit_options, start_grid=self.start_grid,
            noise_model=self.noise_model)

    @property
    def fingerprint(self):
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class ProposalBootstrapCalibration:
    spec: ProposalCalibrationSpec
    focus_null: dict[Hypothesis, np.ndarray]
    pair_null: dict[Hypothesis, np.ndarray]
    null_models: tuple
    proposal_counts: dict[Hypothesis, np.ndarray]
    draws_per_model: int
    seed: int

    @property
    def fingerprint(self):
        def arrays(values):
            return {kind.value: np.asarray(draws, dtype=float).tolist()
                    for kind, draws in sorted(
                        values.items(), key=lambda item: item[0].value)}
        payload = {
            "spec": asdict(self.spec),
            "focus_null": arrays(self.focus_null),
            "pair_null": arrays(self.pair_null),
            "proposal_counts": arrays(self.proposal_counts),
            "null_models": [model.to_dict() for model in self.null_models],
            "draws_per_model": self.draws_per_model,
            "seed": self.seed,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def check_compatible(self, fits):
        observed = CalibrationSpec.create(
            fits.shape, fits.sigma, fits.options, fits.start_grid)
        if observed.fingerprint != self.spec.local_spec.fingerprint:
            raise ValueError("local fit configuration does not match proposal calibration")

    def check_pipeline(self, frame_shape, sigma, proposal_options,
                       roi_size, max_components):
        observed = ProposalCalibrationSpec.create(
            frame_shape, roi_size, sigma,
            FitOptions(**self.spec.fit_options),
            StartGrid(**self.spec.start_grid), proposal_options,
            max_components)
        if observed.fingerprint != self.spec.fingerprint:
            raise ValueError("proposal pipeline does not match calibration")

    def null_values(self, statistic, hypothesis):
        hypothesis = Hypothesis(hypothesis)
        values = self.focus_null if statistic == "focus" else self.pair_null
        if statistic not in ("focus", "pair"):
            raise ValueError("statistic must be 'focus' or 'pair'")
        if hypothesis not in values or np.asarray(values[hypothesis]).size == 0:
            raise KeyError(f"no {statistic} null draws for {hypothesis.value}")
        draws = np.asarray(values[hypothesis], dtype=float)
        return draws[None, :] if draws.ndim == 1 else draws

    def p_value(self, statistic, observed, hypothesis):
        values = self.null_values(statistic, hypothesis)
        counts = np.count_nonzero(values >= float(observed), axis=1)
        return float(np.max((1.0 + counts) / (values.shape[1] + 1.0)))

    def conservative_p(self, statistic, observed, hypotheses):
        return max(self.p_value(statistic, observed, hypothesis)
                   for hypothesis in hypotheses)

    def to_dict(self):
        def encode(values):
            return {kind.value: np.asarray(draws, dtype=float).tolist()
                    for kind, draws in values.items()}
        return {
            "schema": 1,
            "spec": asdict(self.spec),
            "spec_fingerprint": self.spec.fingerprint,
            "calibration_fingerprint": self.fingerprint,
            "focus_null": encode(self.focus_null),
            "pair_null": encode(self.pair_null),
            "proposal_counts": encode(self.proposal_counts),
            "null_models": [model.to_dict() for model in self.null_models],
            "draws_per_model": self.draws_per_model,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != 1:
            raise ValueError("unsupported proposal-calibration schema")
        raw = value["spec"]
        spec = ProposalCalibrationSpec(
            frame_shape=tuple(raw["frame_shape"]), roi_size=int(raw["roi_size"]),
            sigma=float(raw["sigma"]), fit_options=dict(raw["fit_options"]),
            start_grid=dict(raw["start_grid"]),
            proposal_options=dict(raw["proposal_options"]),
            max_components=int(raw["max_components"]),
            noise_model=str(raw["noise_model"]),
        )
        if value.get("spec_fingerprint") != spec.fingerprint:
            raise ValueError("proposal-calibration specification fingerprint mismatch")

        def decode(values):
            return {Hypothesis(kind): np.asarray(draws, dtype=float)
                    for kind, draws in values.items()}
        result = cls(
            spec=spec, focus_null=decode(value["focus_null"]),
            pair_null=decode(value["pair_null"]),
            null_models=tuple(_null_from_dict(item)
                              for item in value["null_models"]),
            proposal_counts=decode(value["proposal_counts"]),
            draws_per_model=int(value["draws_per_model"]),
            seed=int(value["seed"]),
        )
        if value.get("calibration_fingerprint") != result.fingerprint:
            raise ValueError("proposal-calibration content fingerprint mismatch")
        return result

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def calibrate_proposal_search(null_models, draws_per_model, *, roi_size=13,
                              max_components=64, seed=0, fit_options=None,
                              start_grid=None, proposal_options=None,
                              progress=None):
    """Bootstrap frame maxima after the complete proposal and fitting search."""
    null_models = tuple(null_models)
    if not null_models or draws_per_model < 1:
        raise ValueError("null models and positive draw count are required")
    fit_options = FitOptions() if fit_options is None else fit_options
    start_grid = StartGrid() if start_grid is None else start_grid
    proposal_options = (ProposalOptions() if proposal_options is None
                        else proposal_options)
    shape, sigma = null_models[0].shape, null_models[0].sigma
    allowed = {Hypothesis.H0, Hypothesis.HSMOOTH,
               Hypothesis.H1, Hypothesis.HWIDE}
    present = {model.hypothesis for model in null_models}
    if not allowed.issubset(present):
        missing = ", ".join(sorted(kind.value for kind in allowed - present))
        raise ValueError(f"missing required null classes: {missing}")
    for model in null_models:
        if model.hypothesis not in allowed:
            raise ValueError(f"unsupported null class {model.hypothesis.value}")
        if model.shape != shape or not np.isclose(model.sigma, sigma):
            raise ValueError("all frame nulls must share shape and sigma")

    focus = {Hypothesis.H0: [], Hypothesis.HSMOOTH: [], Hypothesis.HWIDE: []}
    pair = {kind: [] for kind in allowed}
    counts = {kind: [] for kind in allowed}
    rng = np.random.default_rng(seed)
    completed = 0
    total = len(null_models) * int(draws_per_model)
    for model in null_models:
        model_focus, model_pair, model_counts = [], [], []
        for _ in range(int(draws_per_model)):
            data = model.sample(rng)
            proposals = generate_proposals(
                data, sigma, options=proposal_options)
            proposed = fit_proposal_components(
                data, sigma, proposals, roi_size=roi_size,
                max_components=max_components, options=fit_options,
                start_grid=start_grid)
            statistics = [selection_statistics(item.fits)
                          for item in proposed.fitted]
            model_focus.append(max(
                [0.0] + [item.focus_gain for item in statistics]))
            model_pair.append(max(
                [0.0] + [item.pair_gain for item in statistics]))
            model_counts.append(len(proposed.fitted))
            completed += 1
            if progress is not None:
                progress(completed, total, model.hypothesis)
        if model.hypothesis in focus:
            focus[model.hypothesis].append(model_focus)
        pair[model.hypothesis].append(model_pair)
        counts[model.hypothesis].append(model_counts)

    convert = lambda values: {
        kind: np.asarray(rows, dtype=float) for kind, rows in values.items()}
    return ProposalBootstrapCalibration(
        spec=ProposalCalibrationSpec.create(
            shape, roi_size, sigma, fit_options, start_grid,
            proposal_options, max_components),
        focus_null=convert(focus), pair_null=convert(pair),
        null_models=null_models, proposal_counts=convert(counts),
        draws_per_model=int(draws_per_model), seed=int(seed),
    )
