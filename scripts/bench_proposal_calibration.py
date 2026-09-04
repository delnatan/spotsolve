#!/usr/bin/env python3
"""End-to-end Stage 3 calibration including the proposal maximum search."""

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
    ProposalOptions,
    calibrate_proposal_search,
    select_proposed_frame,
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
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _wilson(successes, trials, z=1.959963984540054):
    rate = successes / trials
    scale = 1.0 + z * z / trials
    centre = (rate + z * z / (2.0 * trials)) / scale
    half = z / scale * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
    return [centre - half, centre + half]


def _midpoint(shape):
    return np.array([(shape[0] - 1.0) / 2.0,
                     (shape[1] - 1.0) / 2.0])


def _phase_offsets(count, seed):
    offsets = [np.zeros(2)]
    offsets.extend(np.random.default_rng(seed).uniform(
        -0.45, 0.45, size=(count - 1, 2)))
    return offsets


def _null_models(args):
    shape = (args.size, args.size)
    centre = _midpoint(shape)
    common = dict(shape=shape, sigma=args.sigma)
    models = [NullModel(
        hypothesis=Hypothesis.H0, label="background",
        theta=np.array([np.log(args.background), 0.0, 0.0]),
        centre=centre, **common)]
    for phase_index, offset in enumerate(_phase_offsets(
            args.phase_cells, args.calibration_seed + 8_171)):
        position = centre + offset
        models.append(NullModel(
            hypothesis=Hypothesis.H1, label=f"focus_phase_{phase_index}",
            theta=np.array([
                np.log(args.background), 0.0, 0.0,
                np.log(args.photons), position[0], position[1]]),
            centre=position, **common))
        for ratio in args.wide_ratios:
            models.append(NullModel(
                hypothesis=Hypothesis.HWIDE,
                label=f"wide_{ratio:g}_phase_{phase_index}",
                theta=np.array([
                    np.log(args.background), 0.0, 0.0,
                    np.log(args.photons), position[0], position[1],
                    np.log(ratio)]),
                centre=position, **common))
    models.extend(CorrelatedHazeNull(
        shape=shape, sigma=args.sigma, background=args.background,
        peak_above_background=args.haze_peak,
        correlation_length=correlation,
        label=f"correlated_haze_{correlation:g}")
        for correlation in args.haze_correlation_lengths)
    return models


def _random_centre(shape, seed):
    return (_midpoint(shape)
            + np.random.default_rng(seed + 7_919).uniform(-0.45, 0.45, 2))


def _evaluate(scenario, calibration, args):
    started = perf_counter()
    result = select_proposed_frame(
        scenario.image, scenario.sigma, calibration,
        alpha_focus=args.alpha_focus, alpha_pair=args.alpha_pair)
    elapsed = perf_counter() - started
    metric = focused_metrics(result.focused_positions, scenario)
    return {
        "n_focus": len(result.focused_positions),
        "metric": metric,
        "proposal_count": len(result.proposals.proposals),
        "component_count": len(result.proposals.components),
        "local_fit_count": len(result.proposed_fits.fitted),
        "skipped_edge_components": result.proposed_fits.skipped_edge_components,
        "skipped_budget_components": result.proposed_fits.skipped_budget_components,
        "elapsed_s": elapsed,
    }


def run(args):
    shape = (args.size, args.size)
    proposal_options = ProposalOptions(
        score_threshold=args.score_threshold,
        max_proposals=args.max_proposals)
    started = perf_counter()
    calibration = calibrate_proposal_search(
        _null_models(args), args.calibration_draws,
        roi_size=args.roi_size, max_components=args.max_components,
        seed=args.calibration_seed, proposal_options=proposal_options)
    calibration_elapsed = perf_counter() - started
    if args.calibration_output is not None:
        calibration.save(args.calibration_output)

    common = dict(shape=shape, sigma=args.sigma, background=args.background)
    factories = {
        "blank": lambda seed: blank(seed=seed, **common),
        "single": lambda seed: focused_single(
            photons=args.photons, centre=_random_centre(shape, seed),
            seed=seed, **common),
        "smooth_haze": lambda seed: smooth_haze(
            peak_above_background=args.haze_peak,
            correlation_length=args.evaluation_haze_correlation,
            seed=seed, **common),
    }
    for ratio in args.wide_ratios:
        factories[f"wide_{ratio:g}"] = lambda seed, ratio=ratio: wide_source(
            photons=args.photons, width_ratio=ratio,
            centre=_random_centre(shape, seed), seed=seed, **common)

    raw = []
    null_cells = {}
    for class_index, (kind, factory) in enumerate(factories.items()):
        records = []
        for trial in range(args.null_evaluation_draws):
            seed = args.evaluation_seed + class_index * 100_000 + trial
            record = _evaluate(factory(seed), calibration, args)
            record.update(kind=kind, trial=trial)
            records.append(record)
            raw.append(record)
        if kind == "single":
            errors = sum(record["n_focus"] == 2 for record in records)
            error_name = "false_split"
        else:
            errors = sum(record["n_focus"] > 0 for record in records)
            error_name = "false_focus"
        null_cells[kind] = {
            "draws": len(records), "error_kind": error_name,
            "error_rate": errors / len(records),
            "error_rate_ci95": _wilson(errors, len(records)),
            "focused_false_emitters_per_frame": float(np.mean([
                record["metric"]["false_positive"] for record in records])),
            "mean_proposals": float(np.mean([
                record["proposal_count"] for record in records])),
            "mean_local_fits": float(np.mean([
                record["local_fit_count"] for record in records])),
        }

    pair_cells = []
    for ratio_index, ratio in enumerate(args.flux_ratios):
        for sep_index, separation in enumerate(args.separations):
            records = []
            for trial in range(args.pair_evaluation_draws):
                seed = (args.evaluation_seed + 1_000_000
                        + ratio_index * 100_000 + sep_index * 10_000 + trial)
                rng = np.random.default_rng(seed + 7_919)
                scenario = focused_pair(
                    bright_photons=args.photons, flux_ratio=ratio,
                    separation_ratio=separation,
                    angle=float(rng.uniform(0.0, 2.0 * np.pi)),
                    centre=_random_centre(shape, seed), seed=seed, **common)
                record = _evaluate(scenario, calibration, args)
                record.update(kind="pair", flux_ratio=ratio,
                              separation_ratio=separation, trial=trial)
                records.append(record)
                raw.append(record)
            exact = sum(record["metric"]["exact_count"] for record in records)
            pair_cells.append({
                "flux_ratio": ratio, "separation_ratio": separation,
                "draws": len(records), "exact_rate": exact / len(records),
                "exact_rate_ci95": _wilson(exact, len(records)),
                "proposal_miss_rate": float(np.mean([
                    record["local_fit_count"] == 0 for record in records])),
                "mean_local_fits": float(np.mean([
                    record["local_fit_count"] for record in records])),
            })

    return _jsonable({
        "schema": 1,
        "kind": "proposal_search_bootstrap_evaluation",
        "git_revision": _revision(),
        "configuration": vars(args),
        "calibration": {
            "fingerprint": calibration.fingerprint,
            "spec_fingerprint": calibration.spec.fingerprint,
            "elapsed_s": calibration_elapsed,
            "null_cells": {
                kind.value: int(np.asarray(values).shape[0])
                for kind, values in calibration.pair_null.items()},
            "draws_per_cell": args.calibration_draws,
        },
        "null_cells": null_cells,
        "pair_cells": pair_cells,
        "timing": {
            "median_frame_s": float(np.median([
                record["elapsed_s"] for record in raw])),
            "p95_frame_s": float(np.quantile([
                record["elapsed_s"] for record in raw], 0.95)),
        },
        "raw": raw,
    })


def _print_summary(result):
    print("Stage 3 end-to-end proposal-search calibration")
    print(f"  calibration: {result['calibration']['elapsed_s']:.1f} s; "
          f"fingerprint {result['calibration']['fingerprint'][:12]}")
    timing = result["timing"]
    print(f"  median / p95 frame: {timing['median_frame_s']:.3f} / "
          f"{timing['p95_frame_s']:.3f} s")
    print("  held-out null classes:")
    for kind, cell in result["null_cells"].items():
        print(f"    {kind:12s} {cell['error_kind']} "
              f"{100*cell['error_rate']:5.1f}%  "
              f"mean fits {cell['mean_local_fits']:.2f}")
    print("  focused pairs:")
    for cell in result["pair_cells"]:
        print(f"    ratio {cell['flux_ratio']:g} "
              f"d={cell['separation_ratio']:.2f} sigma: "
              f"exact {100*cell['exact_rate']:5.1f}%  "
              f"miss {100*cell['proposal_miss_rate']:4.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-draws", type=int, default=99)
    parser.add_argument("--null-evaluation-draws", type=int, default=100)
    parser.add_argument("--pair-evaluation-draws", type=int, default=48)
    parser.add_argument("--calibration-seed", type=int, default=2026090404)
    parser.add_argument("--evaluation-seed", type=int, default=2026090405)
    parser.add_argument("--size", type=int, default=33)
    parser.add_argument("--roi-size", type=int, default=13)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--background", type=float, default=4.0)
    parser.add_argument("--photons", type=float, default=900.0)
    parser.add_argument("--score-threshold", type=float, default=2.0)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--max-components", type=int, default=64)
    parser.add_argument("--phase-cells", type=int, default=2)
    parser.add_argument("--wide-ratios", type=float, nargs="+",
                        default=[1.5, 2.0, 3.0])
    parser.add_argument("--haze-peak", type=float, default=8.0)
    parser.add_argument("--haze-correlation-lengths", type=float, nargs="+",
                        default=[2.0, 3.5, 5.0])
    parser.add_argument("--evaluation-haze-correlation", type=float, default=5.0)
    parser.add_argument("--flux-ratios", type=float, nargs="+",
                        default=[1.0, 4.0])
    parser.add_argument("--separations", type=float, nargs="+",
                        default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--alpha-focus", type=float, default=0.01)
    parser.add_argument("--alpha-pair", type=float, default=0.01)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    _print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
