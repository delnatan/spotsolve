#!/usr/bin/env python3
"""Independent calibration/evaluation benchmark for local model selection.

This exercises Stage 2 at one imaging condition. Calibration draws and
evaluation draws use disjoint random streams. The result is still oracle-ROI
performance: proposal generation and its maximization bias enter in Stage 3.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spotsolve.prototype import (
    CorrelatedHazeNull,
    Hypothesis,
    NullModel,
    calibrate_local,
    fit_hypotheses,
    select_local,
)
from tests.scientific.metrics import focused_metrics
from tests.scientific.scenarios import (
    blank,
    focused_pair,
    focused_single,
    smooth_haze,
    wide_source,
)


def _revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _wilson_interval(successes, trials, z=1.959963984540054):
    if trials < 1:
        return [math.nan, math.nan]
    rate = successes / trials
    scale = 1.0 + z * z / trials
    centre = (rate + z * z / (2.0 * trials)) / scale
    half = (z / scale * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)))
    return [centre - half, centre + half]


def _oracle_centre(scenario):
    if len(scenario.focused_positions):
        return np.mean(scenario.focused_positions, axis=0)
    if len(scenario.nuisance_positions):
        return scenario.nuisance_positions[0]
    return np.array([(scenario.image.shape[0] - 1.0) / 2.0,
                     (scenario.image.shape[1] - 1.0) / 2.0])


def _fit(scenario):
    return fit_hypotheses(
        scenario.image, scenario.sigma, centre=_oracle_centre(scenario))


def _phase_offsets(args):
    offsets = [np.zeros(2)]
    rng = np.random.default_rng(args.calibration_seed + 8_171)
    offsets.extend(rng.uniform(-0.45, 0.45, size=(args.phase_cells - 1, 2)))
    return offsets


def _random_centre(shape, seed):
    centre = np.array([(shape[0] - 1.0) / 2.0,
                       (shape[1] - 1.0) / 2.0])
    return centre + np.random.default_rng(seed + 7_919).uniform(-0.45, 0.45, 2)


def _null_models(args):
    common = dict(shape=(args.size, args.size), sigma=args.sigma,
                  background=args.background, poisson=False)
    scenarios = [(Hypothesis.H0, blank(**common))]
    centre = np.array([(args.size - 1.0) / 2.0,
                       (args.size - 1.0) / 2.0])
    for phase_index, offset in enumerate(_phase_offsets(args)):
        scenarios.append((Hypothesis.H1, focused_single(
            photons=args.photons, centre=centre + offset, **common)))
        for ratio in args.wide_ratios:
            scenarios.append((Hypothesis.HWIDE, wide_source(
                photons=args.photons, width_ratio=ratio,
                centre=centre + offset, **common)))
    models = []
    for hypothesis, scenario in scenarios:
        fits = _fit(scenario)
        label = hypothesis.value
        if hypothesis is Hypothesis.HWIDE:
            label = (f"wide_{scenario.metadata['width_ratio']:g}_"
                     f"phase_{scenario.metadata['centre']}")
        elif hypothesis is Hypothesis.H1:
            label = f"focus_phase_{scenario.metadata['centre']}"
        elif hypothesis is Hypothesis.HSMOOTH:
            label = f"smooth_{len(models)}"
        models.append(NullModel.from_fit(
            fits[hypothesis], scenario.image.shape, scenario.sigma, label=label))
    models.extend(
        CorrelatedHazeNull(
            shape=(args.size, args.size), sigma=args.sigma,
            background=args.background,
            peak_above_background=args.haze_peak,
            correlation_length=correlation,
            label=f"correlated_haze_{correlation:g}")
        for correlation in args.haze_correlation_lengths
    )
    return models


def _selected_positions(fits, decision):
    if decision.hypothesis is Hypothesis.H1:
        return fits[Hypothesis.H1].physical["positions"]
    if decision.hypothesis is Hypothesis.H2:
        return fits[Hypothesis.H2].physical["positions"]
    return np.empty((0, 2))


def _evaluate_one(label, scenario, calibration, args):
    fits = _fit(scenario)
    decision = select_local(
        fits, calibration, alpha_focus=args.alpha_focus,
        alpha_pair=args.alpha_pair)
    positions = _selected_positions(fits, decision)
    metric = focused_metrics(positions, scenario)
    return {
        "label": label,
        "metadata": scenario.metadata,
        "selected": decision.hypothesis.value,
        "n_focus": decision.n_focus,
        "focus_gain": decision.statistics.focus_gain,
        "pair_gain": decision.statistics.pair_gain,
        "pair_eligible": decision.statistics.pair_eligible,
        "focus_p": decision.focus_p,
        "pair_p": decision.pair_p,
        "metric": metric,
        "elapsed_s": fits.elapsed_s,
    }


def _evaluation_factories(args):
    common = dict(shape=(args.size, args.size), sigma=args.sigma,
                  background=args.background)
    factories = {
        "blank": lambda seed: blank(seed=seed, **common),
        "single": lambda seed: focused_single(
            photons=args.photons, centre=_random_centre(
                (args.size, args.size), seed), seed=seed, **common),
        "smooth_haze": lambda seed: smooth_haze(
            peak_above_background=args.haze_peak,
            correlation_length=args.haze_correlation_length,
            seed=seed, **common),
    }
    for ratio in args.wide_ratios:
        factories[f"wide_{ratio:g}"] = lambda seed, ratio=ratio: wide_source(
            photons=args.photons, width_ratio=ratio,
            centre=_random_centre((args.size, args.size), seed),
            seed=seed, **common)
    return factories


def run(args):
    null_models = _null_models(args)
    calibration_started = perf_counter()
    calibration = calibrate_local(
        null_models, args.calibration_draws, seed=args.calibration_seed)
    calibration_elapsed = perf_counter() - calibration_started
    if args.calibration_output is not None:
        calibration.save(args.calibration_output)

    evaluation = []
    conditional = {}
    factories = _evaluation_factories(args)
    for class_index, (label, factory) in enumerate(factories.items()):
        records = []
        for trial in range(args.null_evaluation_draws):
            seed = args.evaluation_seed + class_index * 100_000 + trial
            record = _evaluate_one(label, factory(seed), calibration, args)
            records.append(record)
            evaluation.append(record)
        conditional[label] = {
            "draws": len(records),
            "false_or_true_focus_rate": float(np.mean([
                record["n_focus"] > 0 for record in records])),
            "focus_rate_ci95": _wilson_interval(sum(
                record["n_focus"] > 0 for record in records), len(records)),
            "pair_rate": float(np.mean([
                record["n_focus"] == 2 for record in records])),
            "pair_rate_ci95": _wilson_interval(sum(
                record["n_focus"] == 2 for record in records), len(records)),
            "mean_focus_count": float(np.mean(
                [record["n_focus"] for record in records])),
        }

    pair_cells = []
    for ratio_index, ratio in enumerate(args.flux_ratios):
        for sep_index, separation in enumerate(args.separations):
            records = []
            for trial in range(args.pair_evaluation_draws):
                seed = (args.evaluation_seed + 1_000_000
                        + ratio_index * 100_000 + sep_index * 10_000 + trial)
                rng = np.random.default_rng(seed + 7_919)
                centre = (np.array([(args.size - 1.0) / 2.0,
                                    (args.size - 1.0) / 2.0])
                          + rng.uniform(-0.45, 0.45, 2))
                scenario = focused_pair(
                    shape=(args.size, args.size), sigma=args.sigma,
                    background=args.background, bright_photons=args.photons,
                    flux_ratio=ratio, separation_ratio=separation,
                    angle=float(rng.uniform(0.0, 2.0 * np.pi)),
                    centre=centre, seed=seed)
                record = _evaluate_one(
                    f"pair_r{ratio:g}_d{separation:g}", scenario,
                    calibration, args)
                records.append(record)
                evaluation.append(record)
            exact = [record["metric"]["exact_count"] for record in records]
            selected_pair = [record["n_focus"] == 2 for record in records]
            pair_cells.append({
                "flux_ratio": ratio,
                "separation_ratio": separation,
                "draws": len(records),
                "exact_selected_and_localized_rate": float(np.mean(exact)),
                "exact_rate_ci95": _wilson_interval(sum(exact), len(records)),
                "pair_selection_rate": float(np.mean(selected_pair)),
                "pair_selection_rate_ci95": _wilson_interval(
                    sum(selected_pair), len(records)),
                "single_selection_rate": float(np.mean(
                    [record["n_focus"] == 1 for record in records])),
                "nuisance_selection_rate": float(np.mean(
                    [record["n_focus"] == 0 for record in records])),
            })

    return _jsonable({
        "schema": 1,
        "kind": "independent_oracle_roi_bootstrap_evaluation",
        "git_revision": _revision(),
        "configuration": vars(args),
        "calibration": {
            "fingerprint": calibration.fingerprint,
            "fit_fingerprint": calibration.spec.fingerprint,
            "elapsed_s": calibration_elapsed,
            "focus_null_sizes": {
                kind.value: {"cells": int(np.asarray(values).shape[0]),
                             "draws_per_cell": int(np.asarray(values).shape[1])}
                for kind, values in calibration.focus_null.items()},
            "pair_null_sizes": {
                kind.value: {"cells": int(np.asarray(values).shape[0]),
                             "draws_per_cell": int(np.asarray(values).shape[1])}
                for kind, values in calibration.pair_null.items()},
            "minimum_p_by_class": {
                "focus": {kind.value: 1.0 / (np.asarray(values).shape[1] + 1)
                          for kind, values in calibration.focus_null.items()},
                "pair": {kind.value: 1.0 / (np.asarray(values).shape[1] + 1)
                         for kind, values in calibration.pair_null.items()},
            },
        },
        "conditional_evaluation": conditional,
        "pair_cells": pair_cells,
        "timing": {
            "median_evaluation_roi_s": float(np.median(
                [record["elapsed_s"] for record in evaluation])),
            "p95_evaluation_roi_s": float(np.quantile(
                [record["elapsed_s"] for record in evaluation], 0.95)),
        },
        "raw_evaluation": evaluation,
    })


def _print_summary(result):
    calibration = result["calibration"]
    print("independent oracle-ROI bootstrap evaluation")
    print(f"  calibration: {calibration['elapsed_s']:.2f} s  "
          f"fingerprint {calibration['fingerprint'][:12]}")
    timing = result["timing"]
    print(f"  evaluation median / p95 ROI: "
          f"{1000*timing['median_evaluation_roi_s']:.1f} / "
          f"{1000*timing['p95_evaluation_roi_s']:.1f} ms")
    print("  class-conditional evaluation:")
    for label, values in result["conditional_evaluation"].items():
        print(f"    {label:12s} focus {100*values['false_or_true_focus_rate']:6.1f}%  "
              f"pair {100*values['pair_rate']:6.1f}%  "
              f"mean N {values['mean_focus_count']:.2f}")
    print("  focused-pair evaluation:")
    for cell in result["pair_cells"]:
        print(f"    ratio {cell['flux_ratio']:g}  "
              f"d={cell['separation_ratio']:.2f} sigma  "
              f"exact {100*cell['exact_selected_and_localized_rate']:6.1f}%  "
              f"H2/H1/nuis {100*cell['pair_selection_rate']:5.1f}/"
              f"{100*cell['single_selection_rate']:5.1f}/"
              f"{100*cell['nuisance_selection_rate']:5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-draws", type=int, default=99)
    parser.add_argument("--evaluation-draws", type=int, default=99)
    parser.add_argument("--null-evaluation-draws", type=int)
    parser.add_argument("--pair-evaluation-draws", type=int)
    parser.add_argument("--calibration-seed", type=int, default=2026090401)
    parser.add_argument("--evaluation-seed", type=int, default=2026090402)
    parser.add_argument("--size", type=int, default=13)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--background", type=float, default=4.0)
    parser.add_argument("--photons", type=float, default=900.0)
    parser.add_argument("--haze-peak", type=float, default=8.0)
    parser.add_argument("--haze-correlation-length", type=float, default=5.0)
    parser.add_argument("--haze-correlation-lengths", type=float, nargs="+",
                        default=[2.0, 3.5, 5.0])
    parser.add_argument("--phase-cells", type=int, default=2)
    parser.add_argument("--wide-ratios", type=float, nargs="+",
                        default=[1.5, 2.0, 3.0])
    parser.add_argument("--separations", type=float, nargs="+",
                        default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--flux-ratios", type=float, nargs="+",
                        default=[1.0, 4.0])
    parser.add_argument("--alpha-focus", type=float, default=0.05)
    parser.add_argument("--alpha-pair", type=float, default=0.05)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.null_evaluation_draws is None:
        args.null_evaluation_draws = args.evaluation_draws
    if args.pair_evaluation_draws is None:
        args.pair_evaluation_draws = args.evaluation_draws
    if args.phase_cells < 1 or not args.haze_correlation_lengths:
        parser.error("phase and haze-correlation cell counts must be positive")
    result = run(args)
    _print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
