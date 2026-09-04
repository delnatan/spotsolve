"""Parametric-bootstrap calibration for nonregular local model selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from .fit import FitOptions, StartGrid, fit_hypotheses
from .models import Hypothesis, ROIGrid, evaluate
from .select import selection_statistics


@dataclass(frozen=True)
class CalibrationSpec:
    shape: tuple[int, int]
    sigma: float
    fit_options: dict
    start_grid: dict
    noise_model: str = "poisson"

    @classmethod
    def create(cls, shape, sigma, options, start_grid):
        return cls(
            shape=tuple(map(int, shape)),
            sigma=float(sigma),
            fit_options=asdict(options),
            start_grid=asdict(start_grid),
        )

    @property
    def fingerprint(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NullModel:
    hypothesis: Hypothesis
    label: str
    theta: np.ndarray
    shape: tuple[int, int]
    sigma: float
    centre: np.ndarray

    @classmethod
    def from_fit(cls, fit, shape, sigma, label=None):
        hypothesis = Hypothesis(fit.hypothesis)
        if hypothesis is Hypothesis.H2:
            raise ValueError("H2 is an alternative, not a calibration null")
        shape = tuple(map(int, shape))
        if hypothesis in (Hypothesis.H0, Hypothesis.HSMOOTH):
            centre = np.array([(shape[0] - 1.0) / 2.0,
                               (shape[1] - 1.0) / 2.0])
        else:
            centre = np.asarray(fit.physical["positions"][0], dtype=float)
        return cls(hypothesis=hypothesis,
                   label=hypothesis.value if label is None else str(label),
                   theta=np.asarray(fit.theta, dtype=float).copy(),
                   shape=shape, sigma=float(sigma), centre=centre)

    def mean(self):
        grid = ROIGrid.from_shape(self.shape)
        return evaluate(self.hypothesis, self.theta, grid, self.sigma)[0]

    def sample(self, rng):
        return rng.poisson(self.mean()).astype(float)

    def to_dict(self):
        return {
            "kind": "fitted_model",
            "hypothesis": self.hypothesis.value,
            "label": self.label,
            "theta": self.theta.tolist(),
            "shape": list(self.shape),
            "sigma": self.sigma,
            "centre": self.centre.tolist(),
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            hypothesis=Hypothesis(value["hypothesis"]),
            label=str(value["label"]),
            theta=np.asarray(value["theta"], dtype=float),
            shape=tuple(value["shape"]),
            sigma=float(value["sigma"]),
            centre=np.asarray(value["centre"], dtype=float),
        )


@dataclass(frozen=True)
class CorrelatedHazeNull:
    """A generative diffuse-field null, not a fitted fixed mean image."""

    shape: tuple[int, int]
    sigma: float
    background: float
    peak_above_background: float
    correlation_length: float
    label: str = "correlated_haze"
    hypothesis: Hypothesis = Hypothesis.HSMOOTH

    @property
    def centre(self):
        return np.array([(self.shape[0] - 1.0) / 2.0,
                         (self.shape[1] - 1.0) / 2.0])

    def sample(self, rng):
        raw = gaussian_filter(
            rng.normal(size=self.shape), self.correlation_length,
            mode="reflect")
        raw -= raw.min()
        scale = float(raw.max())
        haze = (np.zeros(self.shape) if scale <= 0.0 else
                raw / scale * self.peak_above_background)
        return rng.poisson(self.background + haze).astype(float)

    def to_dict(self):
        return {
            "kind": "correlated_haze",
            "hypothesis": self.hypothesis.value,
            "label": self.label,
            "shape": list(self.shape),
            "sigma": self.sigma,
            "background": self.background,
            "peak_above_background": self.peak_above_background,
            "correlation_length": self.correlation_length,
        }

    @classmethod
    def from_dict(cls, value):
        if Hypothesis(value["hypothesis"]) is not Hypothesis.HSMOOTH:
            raise ValueError("correlated haze must be a smooth null")
        return cls(
            shape=tuple(value["shape"]), sigma=float(value["sigma"]),
            background=float(value["background"]),
            peak_above_background=float(value["peak_above_background"]),
            correlation_length=float(value["correlation_length"]),
            label=str(value["label"]),
        )


def _null_from_dict(value):
    kind = value.get("kind", "fitted_model")
    if kind == "fitted_model":
        return NullModel.from_dict(value)
    if kind == "correlated_haze":
        return CorrelatedHazeNull.from_dict(value)
    raise ValueError(f"unsupported null generator kind {kind!r}")


@dataclass
class BootstrapCalibration:
    spec: CalibrationSpec
    focus_null: dict[Hypothesis, np.ndarray]
    pair_null: dict[Hypothesis, np.ndarray]
    null_models: tuple[NullModel, ...]
    draws_per_model: int
    seed: int

    @property
    def fingerprint(self):
        payload = {
            "spec": asdict(self.spec),
            "null_models": [model.to_dict() for model in self.null_models],
            "draws_per_model": self.draws_per_model,
            "seed": self.seed,
            "focus_null": {
                kind.value: np.asarray(values, dtype=float).tolist()
                for kind, values in sorted(
                    self.focus_null.items(), key=lambda item: item[0].value)
            },
            "pair_null": {
                kind.value: np.asarray(values, dtype=float).tolist()
                for kind, values in sorted(
                    self.pair_null.items(), key=lambda item: item[0].value)
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def check_compatible(self, fits):
        observed = CalibrationSpec.create(
            fits.shape, fits.sigma, fits.options, fits.start_grid)
        if observed.fingerprint != self.spec.fingerprint:
            raise ValueError(
                "fit configuration does not match calibration "
                f"({observed.fingerprint[:12]} != {self.spec.fingerprint[:12]})")

    def null_values(self, statistic, hypothesis):
        hypothesis = Hypothesis(hypothesis)
        if statistic == "focus":
            values = self.focus_null
        elif statistic == "pair":
            values = self.pair_null
        else:
            raise ValueError("statistic must be 'focus' or 'pair'")
        if hypothesis not in values or np.asarray(values[hypothesis]).size == 0:
            raise KeyError(f"no {statistic} null draws for {hypothesis.value}")
        draws = np.asarray(values[hypothesis], dtype=float)
        return draws[None, :] if draws.ndim == 1 else draws

    def p_value(self, statistic, observed, hypothesis):
        """Right-tail bootstrap p-value with the finite-sample plus-one rule."""
        values = self.null_values(statistic, hypothesis)
        # Each row is one fitted nuisance cell. Taking the largest conditional
        # p-value prevents a benign width/background cell from diluting a bad
        # one through pooling.
        counts = np.count_nonzero(values >= float(observed), axis=1)
        cell_p = (1.0 + counts) / (values.shape[1] + 1.0)
        return float(np.max(cell_p))

    def conservative_p(self, statistic, observed, hypotheses):
        return max(self.p_value(statistic, observed, hypothesis)
                   for hypothesis in hypotheses)

    def to_dict(self):
        def encode(values):
            return {kind.value: np.asarray(draws, dtype=float).tolist()
                    for kind, draws in values.items()}
        return {
            "schema": 2,
            "spec": asdict(self.spec),
            "fit_fingerprint": self.spec.fingerprint,
            "calibration_fingerprint": self.fingerprint,
            "focus_null": encode(self.focus_null),
            "pair_null": encode(self.pair_null),
            "null_models": [model.to_dict() for model in self.null_models],
            "draws_per_model": self.draws_per_model,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != 2:
            raise ValueError("unsupported calibration schema")
        spec_value = value["spec"]
        spec = CalibrationSpec(
            shape=tuple(spec_value["shape"]), sigma=float(spec_value["sigma"]),
            fit_options=dict(spec_value["fit_options"]),
            start_grid=dict(spec_value["start_grid"]),
            noise_model=spec_value["noise_model"],
        )
        if value.get("fit_fingerprint") != spec.fingerprint:
            raise ValueError(
                "fit fingerprint does not match calibration specification")

        def decode(values):
            return {Hypothesis(kind): np.asarray(draws, dtype=float)
                    for kind, draws in values.items()}
        result = cls(
            spec=spec,
            focus_null=decode(value["focus_null"]),
            pair_null=decode(value["pair_null"]),
            null_models=tuple(_null_from_dict(v) for v in value["null_models"]),
            draws_per_model=int(value["draws_per_model"]),
            seed=int(value["seed"]),
        )
        if value.get("calibration_fingerprint") != result.fingerprint:
            raise ValueError("calibration content fingerprint does not match")
        return result

    def save(self, path):
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def calibrate_local(null_models, draws_per_model, *, seed=0, options=None,
                    start_grid=None, progress=None):
    """Fit bootstrap draws from fitted nuisance and H1 null models.

    Multiple models of the same class may be supplied. Their bootstrap draws
    remain in separate rows, and the largest cell-specific p-value is used.
    This prevents an easy nuisance cell from diluting a difficult one.
    Runtime proposal selection is intentionally absent from this Stage 2
    calibration and must be included in the later end-to-end calibration.
    """
    null_models = tuple(null_models)
    if not null_models:
        raise ValueError("at least one null model is required")
    if draws_per_model < 1:
        raise ValueError("draws_per_model must be positive")
    options = FitOptions() if options is None else options
    start_grid = StartGrid() if start_grid is None else start_grid
    shape, sigma = null_models[0].shape, null_models[0].sigma
    allowed = {
        Hypothesis.H0, Hypothesis.HSMOOTH,
        Hypothesis.H1, Hypothesis.HWIDE,
    }
    present = {model.hypothesis for model in null_models}
    if not allowed.issubset(present):
        missing = ", ".join(sorted(kind.value for kind in allowed - present))
        raise ValueError(f"missing required null classes: {missing}")
    for model in null_models:
        if model.hypothesis not in allowed:
            raise ValueError(f"unsupported null class {model.hypothesis.value}")
        if model.shape != shape or not np.isclose(model.sigma, sigma):
            raise ValueError("all null models must share shape and sigma")

    focus = {
        Hypothesis.H0: [],
        Hypothesis.HSMOOTH: [],
        Hypothesis.HWIDE: [],
    }
    pair = {kind: [] for kind in allowed}
    rng = np.random.default_rng(seed)
    completed = 0
    total = len(null_models) * int(draws_per_model)
    for model in null_models:
        model_focus = []
        model_pair = []
        for _ in range(int(draws_per_model)):
            data = model.sample(rng)
            fits = fit_hypotheses(data, sigma, centre=model.centre,
                                  start_grid=start_grid, options=options)
            stats = selection_statistics(fits)
            if model.hypothesis in focus:
                model_focus.append(stats.focus_gain)
            model_pair.append(stats.pair_gain)
            completed += 1
            if progress is not None:
                progress(completed, total, model.hypothesis)
        if model.hypothesis in focus:
            focus[model.hypothesis].append(model_focus)
        pair[model.hypothesis].append(model_pair)

    focus_arrays = {kind: np.asarray(values, dtype=float)
                    for kind, values in focus.items()}
    pair_arrays = {kind: np.asarray(values, dtype=float)
                   for kind, values in pair.items()}
    return BootstrapCalibration(
        spec=CalibrationSpec.create(shape, sigma, options, start_grid),
        focus_null=focus_arrays,
        pair_null=pair_arrays,
        null_models=null_models,
        draws_per_model=int(draws_per_model),
        seed=int(seed),
    )
