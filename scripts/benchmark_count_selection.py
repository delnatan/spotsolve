"""Compare count selection at fixed false-detection budgets on held-out frames.

Run from the repository with its installed native extension:
    python scripts/benchmark_count_selection.py --out /tmp/count-selection.json

The calibration split chooses a penalty separately for each condition and
method. The evaluation split never selects parameters. This is a simulation
diagnostic, not a calibrated operating guarantee for experimental movies.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from spotsolve import localize
from spotsolve.metrics import match
from spotsolve.simulate import simulate


ARMS = {
    "empty": dict(n_emitters=0),
    "faint_sparse": dict(density=0.003, amplitude_range=(150, 300)),
    "faint_dense": dict(density=0.034, amplitude_range=(150, 300)),
    "bright_dense": dict(density=0.034, amplitude_range=(900, 1900)),
    "bright_variable_width": dict(density=0.034, amplitude_range=(900, 1900),
                                  sigma_spread=0.2),
    "blur_haze": dict(density=0.034, amplitude_range=(150, 600)),
}


def make_frame(arm, seed):
    sim = simulate(shape=(64, 64), sigma=1.2, background=20, seed=seed,
                   **ARMS[arm])
    if arm == "blur_haze":
        rng = np.random.default_rng(seed + 1_000_000)
        haze = gaussian_filter(rng.normal(size=sim.image.shape), 5)
        haze = 40 * (haze - haze.min()) / np.ptp(haze)
        # Elongation and structured background are absent from the fit model.
        sim.image = rng.poisson(gaussian_filter1d(sim.clean, 0.8, axis=1) + haze)
    return sim


def summarize(rows):
    nt = sum(r["true"] for r in rows)
    ne = sum(r["found"] for r in rows)
    tp = sum(r["matched"] for r in rows)
    fp = np.array([r["found"] - r["matched"] for r in rows])
    return dict(
        frames=len(rows), false_per_frame=float(fp.mean()),
        false_per_frame_se=float(fp.std(ddof=1) / np.sqrt(len(fp))),
        recall=tp / nt if nt else None, precision=tp / ne if ne else None,
        ms_per_frame=1000 * sum(r["seconds"] for r in rows) / len(rows),
        fits_per_frame=sum(r["fits"] for r in rows) / len(rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-frames", type=int, default=12)
    parser.add_argument("--evaluation-frames", type=int, default=24)
    parser.add_argument("--penalties", type=float, nargs="+", default=[0, 2, 4, 8, 16, 32])
    parser.add_argument("--budgets", type=float, nargs="+", default=[1, 3, 5])
    parser.add_argument("--threshold", type=float, default=2.75)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.calibration_frames, args.evaluation_frames) < 2:
        parser.error("each split needs at least two frames")
    curves, choices = [], []
    for arm in ARMS:
        frames = {
            "calibration": [make_frame(arm, 12000 + i) for i in range(args.calibration_frames)],
            "evaluation": [make_frame(arm, 24000 + i) for i in range(args.evaluation_frames)],
        }
        for selection in ("fixed", "bic"):
            for penalty in args.penalties:
                row = dict(arm=arm, selection=selection, count_penalty=penalty)
                for split, sims in frames.items():
                    results = []
                    for i, sim in enumerate(sims):
                        t0 = perf_counter()
                        res = localize(sim.image, sigma=1.2, selection=selection,
                                       count_penalty=penalty, threshold=args.threshold,
                                       images=False)
                        seconds = perf_counter() - t0
                        m = match(sim.positions, res.positions, radius=1.0)
                        results.append(dict(frame=i, true=m.n_true, found=m.n_est,
                                            matched=m.n_matched, seconds=seconds,
                                            fits=sum(res.info[k] for k in
                                                     ("search_fits", "polish_fits", "selection_fits"))))
                    row[split] = summarize(results)
                    row[split + "_frames"] = results
                curves.append(row)
            for budget in args.budgets:
                eligible = [r for r in curves if r["arm"] == arm
                            and r["selection"] == selection
                            and r["calibration"]["false_per_frame"] <= budget]
                best = max(eligible, key=lambda r: (r["calibration"]["recall"] or 0,
                           -r["calibration"]["false_per_frame"], -r["count_penalty"]),
                           default=None)
                choices.append(dict(arm=arm, selection=selection, budget=budget,
                                    count_penalty=best["count_penalty"] if best else None,
                                    evaluation=best["evaluation"] if best else None))
            print(json.dumps(dict(arm=arm, selection=selection, choices=choices[-len(args.budgets):])),
                  flush=True)
        # Save progress so long runs remain inspectable and reproducible.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(
            settings={**vars(args), "out": str(args.out)},
            calibration_seed=12000, evaluation_seed=24000,
            shape=[64, 64], sigma=1.2, background=20, match_radius=1.0,
            curves=curves, choices=choices), indent=2) + "\n")


if __name__ == "__main__":
    main()
