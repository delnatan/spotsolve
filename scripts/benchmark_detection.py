"""Score the detector on seeded simulated scenarios and write a JSON report.

- iso: isolated emitters by flux; recall, precision, rms error, error/SE;
- pairs: equal pairs by separation in sigma; both resolved, detections per pair;
- fields: random fields by density; recall, precision, rms error;
- noise: false detections per 10^6 pixels of Poisson and gain-scaled noise.

    python scripts/benchmark_detection.py mixtures --out report.json

`single` and `mixtures` run `localize` without and with `fit_mixtures`.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import spotsolve
from spotsolve import psf
from spotsolve.metrics import match

SIGMA = 1.45
BG = 20.0
SHAPE = (128, 128)


def detector(name, **kw):
    return lambda f: spotsolve.localize(f, SIGMA, images=False,
                                        fit_mixtures=name == "mixtures", **kw)


def render(positions, fluxes, shape=SHAPE, bg=BG):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]] * 1.0
    if len(positions) == 0:
        return np.full(shape, bg)
    theta = psf.pack(bg, fluxes, positions[:, 0], positions[:, 1])
    return psf.model(theta, yy, xx, SIGMA)


def grid_positions(rng, spacing, border=8.0):
    c = np.arange(border, SHAPE[0] - border + 1e-9, spacing)
    yy, xx = np.meshgrid(c, c, indexing="ij")
    p = np.stack([yy.ravel(), xx.ravel()], 1)
    return p + rng.uniform(-0.5, 0.5, p.shape)


def run(det, frame):
    t = perf_counter()
    r = det(frame)
    return r, perf_counter() - t


def scored(truth, r):
    m = match(truth, r.positions, radius=1.0)
    out = dict(n_true=m.n_true, n_est=m.n_est, n_matched=m.n_matched)
    if m.n_matched:
        d = r.positions[m.matched_est_idx] - truth[m.matched_true_idx]
        se = r.se[m.matched_est_idx, 1:]
        out["sq_err"] = float((d ** 2).sum())
        z = np.abs(d / se).ravel()
        out["z"] = z[np.isfinite(z)].tolist()
    return out


def summarize(rows, seconds, frames):
    n_true = sum(r["n_true"] for r in rows)
    n_est = sum(r["n_est"] for r in rows)
    n_matched = sum(r["n_matched"] for r in rows)
    z = np.concatenate([r.get("z", []) for r in rows]) if rows else np.array([])
    return dict(
        recall=n_matched / max(n_true, 1),
        precision=n_matched / max(n_est, 1),
        rmse=float(np.sqrt(sum(r.get("sq_err", 0.0) for r in rows) / max(n_matched, 1))),
        median_abs_z=float(np.median(z)) if z.size else None,
        n_true=n_true, n_est=n_est,
        ms_per_frame=1e3 * seconds / frames)


def iso(det, rng, frames):
    out = {}
    for flux in (100, 200, 400, 800, 1600):
        rows, secs = [], 0.0
        for _ in range(frames):
            p = grid_positions(rng, 16.0)
            r, t = run(det, rng.poisson(render(p, np.full(len(p), flux))).astype(float))
            rows.append(scored(p, r))
            secs += t
        out[str(flux)] = summarize(rows, secs, frames)
    return out


def pairs(det, rng, frames, flux=800.0):
    out = {}
    for sep in (0.5, 1.0, 1.5, 2.0, 3.0):
        s = sep * SIGMA
        both = per_pair = n = 0
        for _ in range(frames):
            mid = grid_positions(rng, 20.0, border=10.0)
            ang = rng.uniform(0, np.pi, len(mid))
            off = 0.5 * s * np.stack([np.sin(ang), np.cos(ang)], 1)
            p = np.concatenate([mid - off, mid + off])
            r, _ = run(det, rng.poisson(render(p, np.full(len(p), flux))).astype(float))
            k = len(mid)
            m = match(p, r.positions, radius=min(1.0, max(0.5 * s, 0.25)))
            hit = np.zeros(len(p), bool)
            hit[m.matched_true_idx] = True
            both += int((hit[:k] & hit[k:]).sum())
            if len(r.positions):
                d = np.linalg.norm(mid[:, None] - r.positions[None], axis=-1)
                per_pair += int((d <= 0.5 * s + 2 * SIGMA).sum())
            n += k
        out[str(sep)] = dict(both_resolved=both / n, detections_per_pair=per_pair / n)
    return out


def fields(det, rng, frames):
    out = {}
    for density in (0.005, 0.01, 0.02, 0.04):
        rows, secs = [], 0.0
        for _ in range(frames):
            n = rng.poisson(density * (SHAPE[0] - 8) * (SHAPE[1] - 8))
            p = rng.uniform(4, SHAPE[0] - 4, (n, 2))
            f = rng.uniform(150, 3000, n)
            r, t = run(det, rng.poisson(render(p, f)).astype(float))
            rows.append(scored(p, r))
            secs += t
        out[str(density)] = summarize(rows, secs, frames)
    return out


def noise(det, rng, frames):
    out = {}
    for gain in (1.0, 3.0):
        n_est, secs = 0, 0.0
        for _ in range(frames):
            frame = gain * rng.poisson(BG / gain, SHAPE).astype(float)
            r, t = run(det, frame)
            n_est += len(r.positions)
            secs += t
        out[f"gain_{gain:g}"] = dict(
            per_mpx=1e6 * n_est / (frames * SHAPE[0] * SHAPE[1]),
            ms_per_frame=1e3 * secs / frames)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("detector", choices=("single", "mixtures"))
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--kw", default="{}", help="JSON keyword arguments for the detector")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    kw = json.loads(a.kw)
    det = detector(a.detector, **kw)
    report = dict(detector=a.detector, kw=kw, version=spotsolve.__version__,
                  sigma=SIGMA, background=BG, shape=SHAPE, frames=a.frames, seed=a.seed)
    scenarios = (("iso", iso), ("pairs", pairs), ("fields", fields), ("noise", noise))
    for i, (name, fn) in enumerate(scenarios):
        report[name] = fn(det, np.random.default_rng([a.seed, i]), a.frames)
        print(name, json.dumps(report[name]))
    a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
