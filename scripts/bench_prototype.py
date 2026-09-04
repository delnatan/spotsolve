#!/usr/bin/env python3
"""Oracle-ROI benchmark for the explicit hypothesis prototype.

This is intentionally not a production calibration.  It exposes raw objective
gains, per-null conditional error, fit counts, and timing so the local model can
be falsified before candidate generation is involved.
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

from spotsolve import detect
from spotsolve.prototype import Hypothesis, fit_hypotheses
from tests.scientific.metrics import focused_metrics, match_positions
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
        converted = value.item()
        return (None if isinstance(converted, float) and not math.isfinite(converted)
                else converted)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _oracle_record(label, scenario):
    shape_centre = np.array([(scenario.image.shape[0] - 1.0) / 2.0,
                             (scenario.image.shape[1] - 1.0) / 2.0])
    if len(scenario.focused_positions):
        centre = np.mean(scenario.focused_positions, axis=0)
    elif len(scenario.nuisance_positions):
        centre = scenario.nuisance_positions[0]
    else:
        centre = shape_centre
    fits = fit_hypotheses(scenario.image, scenario.sigma, centre=centre)
    pair = fits[Hypothesis.H2]
    rows, cols, distances = match_positions(
        pair.physical["positions"], scenario.focused_positions,
        max_distance=scenario.sigma)
    return {
        "label": label,
        "metadata": scenario.metadata,
        "pair_gain": fits.pair_gain,
        "objectives": {kind.value: fit.objective
                       for kind, fit in fits.fits.items()},
        "winner_without_calibration": min(
            fits.fits, key=lambda kind: fits[kind].objective).value,
        "pair_separation": pair.physical["separation"],
        "pair_flux_fraction": pair.physical["flux_fraction"],
        "pair_collapsed": pair.collapsed,
        "pair_matches": len(rows),
        "pair_match_distances": distances,
        "elapsed_s": fits.elapsed_s,
        "attempts": fits.total_attempts,
        "evaluations": fits.total_evaluations,
    }


def _baseline_record(label, scenario):
    started = perf_counter()
    result = detect(scenario.image, sigma=scenario.sigma, gain=1.0,
                    offset=0.0, max_rounds=8, verbose=0)
    elapsed = perf_counter() - started
    metric = focused_metrics(result.positions, scenario)
    nuisance = result.aggregates
    return {
        "label": label,
        "metadata": scenario.metadata,
        "n_focus": len(result.positions),
        "n_nuisance": 0 if nuisance is None else len(nuisance),
        "metric": metric,
        "elapsed_s": elapsed,
    }


def _null_factories(shape, sigma, background, photons):
    return {
        "blank": lambda seed: blank(shape=shape, sigma=sigma,
                                     background=background, seed=seed),
        "single": lambda seed: focused_single(
            shape=shape, sigma=sigma, background=background, photons=photons,
            seed=seed),
        "wide_1.5": lambda seed: wide_source(
            shape=shape, sigma=sigma, background=background, photons=photons,
            width_ratio=1.5, seed=seed),
        "wide_2": lambda seed: wide_source(
            shape=shape, sigma=sigma, background=background, photons=photons,
            width_ratio=2.0, seed=seed),
        "wide_3": lambda seed: wide_source(
            shape=shape, sigma=sigma, background=background, photons=photons,
            width_ratio=3.0, seed=seed),
        "smooth_haze": lambda seed: smooth_haze(
            shape=shape, sigma=sigma, background=background,
            peak_above_background=8.0, correlation_length=5.0, seed=seed),
    }


def run(args):
    shape = (args.size, args.size)
    null_records = []
    baseline_records = []
    factories = _null_factories(shape, args.sigma, args.background, args.photons)
    for null_index, (label, factory) in enumerate(factories.items()):
        for trial in range(args.trials):
            scenario = factory(args.seed + 100_000 * null_index + trial)
            null_records.append(_oracle_record(label, scenario))
            if not args.skip_baseline:
                baseline_records.append(_baseline_record(label, scenario))

    null_gains = np.array([r["pair_gain"] for r in null_records])
    # Diagnostic only. Stage 2 will calibrate each null class and the full
    # proposal process; this pooled finite-sample quantile must not become a
    # production threshold.
    threshold = float(np.quantile(null_gains, 0.99, method="higher"))

    pair_records = []
    pair_cells = []
    for ratio_index, ratio in enumerate(args.flux_ratios):
        for sep_index, separation in enumerate(args.separations):
            records = []
            baseline_cell = []
            for trial in range(args.trials):
                seed = (args.seed + 1_000_000 + ratio_index * 100_000
                        + sep_index * 10_000 + trial)
                scenario = focused_pair(
                    shape=shape, sigma=args.sigma, background=args.background,
                    bright_photons=args.photons, flux_ratio=ratio,
                    separation_ratio=separation, angle=0.37, seed=seed)
                record = _oracle_record(f"pair_r{ratio:g}_d{separation:g}", scenario)
                records.append(record)
                pair_records.append(record)
                if not args.skip_baseline:
                    br = _baseline_record(record["label"], scenario)
                    baseline_records.append(br)
                    baseline_cell.append(br)
            accepted = [r for r in records if r["pair_gain"] > threshold]
            correctly_localized = [r for r in accepted if r["pair_matches"] == 2]
            distances = [np.asarray(r["pair_match_distances"], dtype=float)
                         for r in correctly_localized]
            joined = np.concatenate(distances) if distances else np.empty(0)
            cell = {
                "flux_ratio": ratio,
                "separation_ratio": separation,
                "trials": len(records),
                "prototype_power": len(accepted) / len(records),
                "prototype_exact_localized_power": len(correctly_localized) / len(records),
                "prototype_conditional_rmse": (float(np.sqrt(np.mean(joined ** 2)))
                                               if len(joined) else np.nan),
                "prototype_median_gain": float(np.median(
                    [r["pair_gain"] for r in records])),
            }
            if baseline_cell:
                cell["baseline_exact_count_rate"] = float(np.mean(
                    [r["metric"]["exact_count"] for r in baseline_cell]))
                cell["baseline_false_emitters_per_frame"] = float(np.mean(
                    [r["metric"]["false_positive"] for r in baseline_cell]))
            pair_cells.append(cell)

    conditional_null = {}
    for label in factories:
        group = [r for r in null_records if r["label"] == label]
        conditional_null[label] = {
            "trials": len(group),
            "pair_selection_rate": float(np.mean(
                [r["pair_gain"] > threshold for r in group])),
            "median_gain": float(np.median([r["pair_gain"] for r in group])),
            "max_gain": float(np.max([r["pair_gain"] for r in group])),
        }

    all_oracle = null_records + pair_records
    result = {
        "schema": 1,
        "kind": "oracle_roi_diagnostic_not_production_calibration",
        "git_revision": _revision(),
        "configuration": vars(args),
        "exploratory_pooled_null_q99": threshold,
        "conditional_null": conditional_null,
        "pair_cells": pair_cells,
        "timing": {
            "median_roi_s": float(np.median([r["elapsed_s"] for r in all_oracle])),
            "p95_roi_s": float(np.quantile([r["elapsed_s"] for r in all_oracle], 0.95)),
            "median_attempts": float(np.median([r["attempts"] for r in all_oracle])),
            "median_evaluations": float(np.median([r["evaluations"] for r in all_oracle])),
            "baseline_median_frame_s": (float(np.median(
                [r["elapsed_s"] for r in baseline_records]))
                if baseline_records else np.nan),
        },
        "raw": {"null": null_records, "pair": pair_records,
                "baseline": baseline_records},
    }
    return _jsonable(result)


def _print_summary(result):
    print("oracle ROI diagnostic (pooled q99 is exploratory, not calibrated)")
    print(f"  pooled null q99: {result['exploratory_pooled_null_q99']:.3f}")
    timing = result["timing"]
    print(f"  median / p95 ROI: {1000*timing['median_roi_s']:.1f} / "
          f"{1000*timing['p95_roi_s']:.1f} ms; "
          f"median evaluations {timing['median_evaluations']:.0f}")
    if timing["baseline_median_frame_s"] is not None:
        print(f"  baseline median full-frame time: "
              f"{1000*timing['baseline_median_frame_s']:.1f} ms")
    print("  conditional null pair-selection rates:")
    for label, record in result["conditional_null"].items():
        print(f"    {label:12s} {100*record['pair_selection_rate']:6.1f}%  "
              f"median/max gain {record['median_gain']:6.2f}/{record['max_gain']:6.2f}")
    print("  pair power:")
    for cell in result["pair_cells"]:
        baseline = cell.get("baseline_exact_count_rate")
        suffix = "" if baseline is None else f"  baseline {100*baseline:5.1f}%"
        print(f"    ratio {cell['flux_ratio']:g}  d={cell['separation_ratio']:.2f} sigma  "
              f"select {100*cell['prototype_power']:5.1f}%  "
              f"select+localize {100*cell['prototype_exact_localized_power']:5.1f}%"
              f"{suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--size", type=int, default=13)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--background", type=float, default=4.0)
    parser.add_argument("--photons", type=float, default=900.0)
    parser.add_argument("--separations", type=float, nargs="+",
                        default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--flux-ratios", type=float, nargs="+", default=[1.0, 4.0])
    parser.add_argument("--skip-baseline", action="store_true")
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
