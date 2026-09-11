"""Scientific controls and timings for the variable-width detector.

Run after a release build: python scripts/check_detect_width.py --repeats 3
Rust is the default; --impl py rs additionally measures the Python reference.
--search passes groups adds the native group search (Rust only) as an arm.
Counts and trajectories need not agree between implementations.
"""

import argparse
import json
from time import perf_counter

import numpy as np

from spotsolve import audit, detect, metrics
from spotsolve.simulate import simulate


def check(size, seed, repeats, arms, density, spread):
    sim = simulate(shape=(size, size), density=density,
                   amplitude_range=(900, 1900), background=5,
                   sigma_spread=spread, seed=seed)
    widths = sim.sigmas if sim.sigmas is not None else np.full(len(sim.positions), sim.sigma)
    focus = (widths >= 0.8*sim.sigma) & (widths <= 2.0*sim.sigma)
    truth = sim.positions[focus]
    distances = np.linalg.norm(sim.positions[:, None] - sim.positions[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    crowded = np.min(distances, axis=1)[focus] < 2.5*sim.sigma
    timings = {arm: [] for arm in arms}
    results = {}
    for repeat in range(repeats):
        order = arms if repeat % 2 == 0 else arms[::-1]
        for arm in order:
            impl, search = arm
            start = perf_counter()
            results[arm] = detect(sim.image, sigma=sim.sigma, gain=1.0,
                                  impl=impl, search=search, verbose=0)
            timings[arm].append(perf_counter() - start)
    for (impl, search), result in results.items():
        match = metrics.match(truth, result.positions, radius=sim.sigma)
        found = np.zeros(len(truth), dtype=bool)
        found[match.matched_true_idx] = True
        residual = audit.audit_result(sim.image, result.model_image, sim.sigma)
        yield dict(
            impl=impl, search=search, size=size, seed=seed, density=density,
            width_spread=spread,
            truth_count=len(sim.positions), focus_truth_count=len(truth),
            detected_count=len(result.positions),
            seconds=float(np.median(timings[(impl, search)])),
            precision=match.precision, recall=match.recall, f1=match.f1,
            rmse_px=match.rmse,
            crowded_recall=float(found[crowded].mean()) if crowded.any() else None,
            residual_positive_peaks=residual["n_missed"],
            residual_negative_peaks=residual["n_piled"],
            residual_score_spread=residual["z_robust_std"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[39, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19])
    parser.add_argument("--densities", type=float, nargs="+", default=[0.034])
    parser.add_argument("--spreads", type=float, nargs="+", default=[0.2])
    parser.add_argument("--impl", choices=["py", "rs"], nargs="+", default=["rs"])
    parser.add_argument("--search", choices=["passes", "groups"], nargs="+",
                        default=["passes"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.sizes) <= 8:
        parser.error("repeats must be positive and sizes must exceed the 8-pixel border")
    if min(args.densities) <= 0 or min(args.spreads) < 0:
        parser.error("densities must be positive and width spreads nonnegative")
    # The group search is native only; it has no Python arm.
    arms = [(impl, search) for search in args.search for impl in args.impl
            if not (search == "groups" and impl == "py")]
    # Import/cache warmup outside the timings.
    for impl, search in arms:
        detect(np.full((9, 9), 5.), gain=1., impl=impl, search=search,
               verbose=0)
    for size in args.sizes:
        for density in args.densities:
            for spread in args.spreads:
                for seed in args.seeds:
                    for row in check(size, seed, args.repeats, arms, density, spread):
                        print(json.dumps(row), flush=True)
