#!/usr/bin/env python3
"""Stage 3 benchmark for permissive full-frame proposal generation."""

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

from spotsolve.prototype import ProposalOptions, generate_proposals
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
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _centre(shape, seed):
    midpoint = np.array([(shape[0] - 1.0) / 2.0,
                         (shape[1] - 1.0) / 2.0])
    return midpoint + np.random.default_rng(seed + 91_771).uniform(-0.5, 0.5, 2)


def _component_covers_truth(component, truth):
    y0, x0, y1, x1 = component.bounds
    return bool(np.all(
        (truth[:, 0] >= y0 - 0.5) & (truth[:, 0] <= y1 - 0.5)
        & (truth[:, 1] >= x0 - 0.5) & (truth[:, 1] <= x1 - 0.5)))


def _proposal_record(scenario, options):
    started = perf_counter()
    result = generate_proposals(
        scenario.image, scenario.sigma, options=options)
    elapsed = perf_counter() - started
    truth = scenario.focused_positions
    covered = (False if len(truth) == 0 else any(
        _component_covers_truth(component, truth)
        for component in result.components))
    nearest = math.nan
    axis_error = math.nan
    if len(truth) and result.proposals:
        truth_centre = np.average(
            truth, axis=0, weights=scenario.focused_fluxes)
        proposal = min(
            result.proposals,
            key=lambda item: np.linalg.norm(item.centre - truth_centre))
        nearest = float(np.linalg.norm(proposal.centre - truth_centre))
        if len(truth) == 2:
            truth_axis = float(scenario.metadata["angle"])
            axis_error = 0.5 * abs(np.angle(np.exp(
                2j * (proposal.pair_axis - truth_axis))))
    return {
        "proposal_count": len(result.proposals),
        "component_count": len(result.components),
        "raw_maximum_count": result.local_maxima_above_threshold,
        "budget_exhausted": result.budget_exhausted,
        "truth_component_recalled": covered,
        "nearest_truth_centroid_distance": nearest,
        "nearest_pair_axis_error": axis_error,
        "maximum_score": (result.proposals[0].score
                          if result.proposals else -math.inf),
        "elapsed_s": elapsed,
    }


def _wilson(successes, trials, z=1.959963984540054):
    rate = successes / trials
    scale = 1.0 + z * z / trials
    centre = (rate + z * z / (2.0 * trials)) / scale
    half = z / scale * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
    return [centre - half, centre + half]


def run(args):
    shape = (args.size, args.size)
    options = ProposalOptions(
        score_threshold=args.score_threshold,
        max_proposals=args.max_proposals)
    raw = []
    source_cells = []

    for background in args.backgrounds:
        for photons in args.photons:
            records = []
            for trial in range(args.trials):
                seed = (args.seed + int(background * 10_000)
                        + int(photons * 100) + trial)
                scenario = focused_single(
                    shape=shape, sigma=args.sigma, background=background,
                    photons=photons, centre=_centre(shape, seed), seed=seed)
                record = _proposal_record(scenario, options)
                record.update(kind="single", background=background,
                              photons=photons, trial=trial)
                records.append(record)
                raw.append(record)
            recalled = sum(record["truth_component_recalled"] for record in records)
            source_cells.append({
                "kind": "single", "background": background,
                "photons": photons, "trials": len(records),
                "recall": recalled / len(records),
                "recall_ci95": _wilson(recalled, len(records)),
                "mean_proposals": float(np.mean([
                    record["proposal_count"] for record in records])),
                "mean_components": float(np.mean([
                    record["component_count"] for record in records])),
            })

            for ratio in args.flux_ratios:
                for separation in args.separations:
                    records = []
                    for trial in range(args.trials):
                        seed = (args.seed + 1_000_000
                                + int(background * 10_000)
                                + int(photons * 100) + int(ratio * 1_000)
                                + int(separation * 100) + trial)
                        rng = np.random.default_rng(seed + 31_337)
                        scenario = focused_pair(
                            shape=shape, sigma=args.sigma,
                            background=background, bright_photons=photons,
                            flux_ratio=ratio, separation_ratio=separation,
                            angle=float(rng.uniform(0.0, 2.0 * np.pi)),
                            centre=_centre(shape, seed), seed=seed)
                        record = _proposal_record(scenario, options)
                        record.update(
                            kind="pair", background=background,
                            photons=photons, flux_ratio=ratio,
                            separation_ratio=separation, trial=trial)
                        records.append(record)
                        raw.append(record)
                    recalled = sum(
                        record["truth_component_recalled"] for record in records)
                    source_cells.append({
                        "kind": "pair", "background": background,
                        "photons": photons, "flux_ratio": ratio,
                        "separation_ratio": separation,
                        "trials": len(records),
                        "recall": recalled / len(records),
                        "recall_ci95": _wilson(recalled, len(records)),
                        "mean_proposals": float(np.mean([
                            record["proposal_count"] for record in records])),
                        "mean_components": float(np.mean([
                            record["component_count"] for record in records])),
                        "median_pair_axis_error": float(np.nanmedian([
                            record["nearest_pair_axis_error"]
                            for record in records])),
                    })

    null_factories = {
        "blank": lambda seed: blank(
            shape=shape, sigma=args.sigma, background=args.null_background,
            seed=seed),
        "smooth_haze": lambda seed: smooth_haze(
            shape=shape, sigma=args.sigma, background=args.null_background,
            peak_above_background=args.haze_peak,
            correlation_length=args.haze_correlation_length, seed=seed),
    }
    for ratio in args.wide_ratios:
        null_factories[f"wide_{ratio:g}"] = lambda seed, ratio=ratio: wide_source(
            shape=shape, sigma=args.sigma, background=args.null_background,
            photons=args.null_photons, width_ratio=ratio,
            centre=_centre(shape, seed), seed=seed)

    null_cells = []
    for class_index, (kind, factory) in enumerate(null_factories.items()):
        records = []
        for trial in range(args.null_trials):
            seed = args.seed + 10_000_000 + class_index * 100_000 + trial
            record = _proposal_record(factory(seed), options)
            record.update(kind=kind, trial=trial)
            records.append(record)
            raw.append(record)
        null_cells.append({
            "kind": kind,
            "trials": len(records),
            "mean_proposals": float(np.mean([
                record["proposal_count"] for record in records])),
            "p95_proposals": float(np.quantile([
                record["proposal_count"] for record in records], 0.95)),
            "maximum_proposals": int(max(
                record["proposal_count"] for record in records)),
            "mean_components": float(np.mean([
                record["component_count"] for record in records])),
            "p95_components": float(np.quantile([
                record["component_count"] for record in records], 0.95)),
            "maximum_components": int(max(
                record["component_count"] for record in records)),
            "budget_exhaustion_rate": float(np.mean([
                record["budget_exhausted"] for record in records])),
        })

    return _jsonable({
        "schema": 1,
        "kind": "proposal_recall_and_cost",
        "git_revision": _revision(),
        "configuration": vars(args),
        "source_cells": source_cells,
        "null_cells": null_cells,
        "timing": {
            "median_frame_ms": 1000.0 * float(np.median([
                record["elapsed_s"] for record in raw])),
            "p95_frame_ms": 1000.0 * float(np.quantile([
                record["elapsed_s"] for record in raw], 0.95)),
        },
        "raw": raw,
    })


def _print_summary(result):
    print("Stage 3 permissive-proposal benchmark")
    timing = result["timing"]
    print(f"  median / p95 frame: {timing['median_frame_ms']:.3f} / "
          f"{timing['p95_frame_ms']:.3f} ms")
    print("  worst source-cell recall:")
    cells = sorted(result["source_cells"], key=lambda cell: cell["recall"])
    for cell in cells[:min(12, len(cells))]:
        detail = ("" if cell["kind"] == "single" else
                  f" r={cell['flux_ratio']:g} d={cell['separation_ratio']:.2f}")
        print(f"    {cell['kind']:6s} bg={cell['background']:g} "
              f"photons={cell['photons']:g}{detail}: "
              f"{100*cell['recall']:5.1f}%")
    print("  nuisance proposal counts:")
    for cell in result["null_cells"]:
        print(f"    {cell['kind']:12s} mean/p95/max "
              f"{cell['mean_proposals']:.2f}/{cell['p95_proposals']:.1f}/"
              f"{cell['maximum_proposals']} proposals; "
              f"{cell['mean_components']:.2f}/{cell['p95_components']:.1f}/"
              f"{cell['maximum_components']} components")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--null-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026090403)
    parser.add_argument("--size", type=int, default=33)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--backgrounds", type=float, nargs="+",
                        default=[1.0, 4.0, 20.0, 100.0])
    parser.add_argument("--photons", type=float, nargs="+",
                        default=[150.0, 300.0, 900.0])
    parser.add_argument("--flux-ratios", type=float, nargs="+",
                        default=[1.0, 4.0])
    parser.add_argument("--separations", type=float, nargs="+",
                        default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--score-threshold", type=float, default=2.0)
    parser.add_argument("--max-proposals", type=int, default=64)
    parser.add_argument("--null-background", type=float, default=4.0)
    parser.add_argument("--null-photons", type=float, default=900.0)
    parser.add_argument("--haze-peak", type=float, default=8.0)
    parser.add_argument("--haze-correlation-length", type=float, default=5.0)
    parser.add_argument("--wide-ratios", type=float, nargs="+",
                        default=[1.5, 2.0, 3.0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trials < 1 or args.null_trials < 1:
        parser.error("trial counts must be positive")
    result = run(args)
    _print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
