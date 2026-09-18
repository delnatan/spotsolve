"""Measure conservative linking against known trajectories and detector output.

    python scripts/benchmark_tracking.py --out /tmp/tracking.json

All cutoffs are reported; this sweep does not select or calibrate a threshold.
Coordinate clutter includes independent false detections near real emitters.
The image arm renders Poisson movies and runs the actual multi-emitter detector.
Its truth labels use one-to-one matching within 1 px, which is imperfect for
unresolved emitters. Rejected fits/non-finite errors are excluded before linking.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

import spotsolve
from spotsolve import psf
from spotsolve.metrics import match


def make_movie(seed, n_frames, arm):
    rng = np.random.default_rng(seed)
    n = 70
    pos = rng.uniform(9., 55., (n, 2))
    diffusion = rng.choice([0., 0.15, 0.6], n, p=[0.2, 0.4, 0.4])
    flux = rng.uniform(250., 700., n)
    drift = np.array([0.08, -0.04]) if arm == "detector_drift" else np.zeros(2)
    yy, xx = np.mgrid[:64, :64].astype(float)
    tables, identities = [], []
    offset = 0
    for f in range(n_frames):
        if f:
            pos += rng.normal(size=pos.shape) * np.sqrt(2 * diffusion[:, None]) + drift
        if arm == "nearby_clutter":
            seen = rng.random(n) < 0.9
            truth = np.flatnonzero(seen)
            se = 0.3 * np.exp(rng.normal(0., 0.3, len(truth)))
            obs = pos[seen] + rng.normal(size=(len(truth), 2)) * se[:, None]
            # False spots are spatially correlated with emitters, but have
            # no continuing identity from one frame to the next.
            parents = rng.choice(n, n // 3, replace=False)
            false = pos[parents] + rng.normal(0., 0.65, (len(parents), 2))
            obs = np.concatenate([obs, false])
            se = np.concatenate([se, np.full(len(false), 0.4)])
            truth = np.concatenate([truth, np.full(len(false), -1)])
        else:
            clean = psf.model(psf.pack(20., flux, pos[:, 0], pos[:, 1]), yy, xx, 1.2)
            result = spotsolve.localize(rng.poisson(clean).astype(float), sigma=1.2,
                                        images=False)
            valid = np.all(np.isfinite(result.se[:, 1:]) & (result.se[:, 1:] > 0), axis=1)
            obs = result.positions[valid]
            se = result.se[valid, 1:]
            # Exclude invisible centers before assigning identity; retain
            # their photons in the rendering at image boundaries.
            visible = np.flatnonzero(np.all((pos >= 0) & (pos <= 63), axis=1))
            matched = match(pos[visible], obs, radius=1.)
            truth = np.full(len(obs), -1)
            truth[matched.matched_est_idx] = visible[matched.matched_true_idx]
        if se.ndim == 1:
            se = np.repeat(se[:, None], 2, axis=1)
        tables.append(pl.DataFrame(dict(
            loc_id=np.arange(offset, offset + len(obs)), frame=np.full(len(obs), f),
            y=obs[:, 0], x=obs[:, 1], se_y=se[:, 0], se_x=se[:, 1])))
        identities.append(truth)
        offset += len(obs)
    return pl.concat(tables), np.concatenate(identities), diffusion, drift


def evaluate(tracks, who, diffusion, drift):
    frame = tracks["frame"].to_numpy()
    tid = tracks["track_id"].to_numpy()
    order = np.lexsort((frame, tid))
    a, b = order[:-1], order[1:]
    linked = tid[a] == tid[b]
    assert np.all(frame[b[linked]] == frame[a[linked]] + 1)
    a, b = a[linked], b[linked]
    correct = (who[a] >= 0) & (who[a] == who[b])
    accepted = tracks["track_accepted"].to_numpy()[b]
    observed = set(zip(frame[who >= 0], who[who >= 0]))
    available = sum((f + 1, w) in observed for f, w in observed)
    # Truth-only mean D for identities contributing correct accepted links.
    # This measures selection of slow/fast populations, not a fitted D.
    retained_d = diffusion[who[a[correct & accepted]]]
    all_d = [diffusion[w] for f, w in observed if (f + 1, w) in observed]
    return dict(
        detections=len(tracks), clutter_detections=int(np.sum(who < 0)),
        available_true_links=available, links=len(a),
        wrong_links=int(np.sum(~correct)), correct_links=int(np.sum(correct)),
        accepted_links=int(accepted.sum()),
        accepted_wrong_links=int(np.sum(accepted & ~correct)),
        accepted_correct_links=int(np.sum(accepted & correct)),
        rejected_links=int(tracks["link_rejected"].sum()),
        tracks=tracks["track_id"].n_unique(),
        accepted_tracks=tracks.filter(pl.col("track_accepted"))["track_id"].n_unique(),
        available_mean_true_d=float(np.mean(all_d)) if all_d else None,
        accepted_correct_mean_true_d=float(np.mean(retained_d)) if len(retained_d) else None,
        imposed_drift=drift.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--margins", type=float, nargs="+", default=[0., 0.5, 1., 2.])
    parser.add_argument("--min-track-length", type=int, default=4)
    parser.add_argument("--arms", nargs="+", choices=["nearby_clutter", "detector", "detector_drift"],
                        default=["nearby_clutter", "detector", "detector_drift"])
    args = parser.parse_args()
    if args.frames < 2 or args.min_track_length < 1:
        parser.error("need at least two frames and a positive minimum track length")
    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            locs, who, diffusion, drift = make_movie(seed, args.frames, arm)
            start = perf_counter()
            params = spotsolve.fit_link_params(locs)
            fit_seconds = perf_counter() - start
            start = perf_counter()
            spotsolve.link(locs, params)
            baseline_seconds = perf_counter() - start
            for margin in args.margins:
                start = perf_counter()
                tracks = spotsolve.link(locs, params, min_link_margin=margin,
                                        min_track_length=args.min_track_length, diagnostics=True)
                seconds = perf_counter() - start
                row = dict(arm=arm, seed=seed, margin=margin, seconds=seconds,
                           fit_seconds=fit_seconds, baseline_seconds=baseline_seconds,
                           fitted_d_mean=params.d_mean,
                           **evaluate(tracks, who, diffusion, drift))
                rows.append(row)
                print(json.dumps(row), flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(dict(
                settings={**vars(args), "out": str(args.out)}, rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
