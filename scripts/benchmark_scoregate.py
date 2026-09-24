"""Score-gated prototype against the production box search.

Scenes (sigma 1.45, Poisson background 50 ADU, phi = 1):
  noise     pure background, false emitters per Mpx
  isolated  spots >= 16 px apart, flux log-uniform; recall by ideal SNR
  pairs     equal-flux pairs at fixed separations; emitters counted per pair
  blobs     single wide Gaussians (2, 3 sigma); emitters counted per blob

Writes output/scoregate/results.json.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from spotsolve import psf, scoregate
from spotsolve.native import localize

SIGMA, B = 1.45, 50.0
G_NORM = float(np.sqrt((scoregate.psf_kernel(SIGMA) ** 2).sum()))
OUT = Path(__file__).resolve().parents[1] / "output" / "scoregate"


def render(shape, pos, flux, sig):
    img = np.full(shape, B)
    for (y, x), f, s in zip(pos, flux, sig):
        r = int(np.ceil(6 * s))
        y0, x0 = max(int(y) - r, 0), max(int(x) - r, 0)
        y1, x1 = min(int(y) + r + 2, shape[0]), min(int(x) + r + 2, shape[1])
        th = psf.pack_var_sigma(0.0, f, y - y0, x - x0, s)
        yy, xx = np.mgrid[:y1 - y0, :x1 - x0].astype(float)
        img[y0:y1, x0:x1] += psf.model_var_sigma(th, yy, xx)
    return img


def grid(shape, step, rng, jitter=3.0):
    gy, gx = np.meshgrid(np.arange(12, shape[0] - 12, step),
                         np.arange(12, shape[1] - 12, step), indexing="ij")
    return np.c_[gy.ravel(), gx.ravel()] + rng.uniform(-jitter, jitter, (gy.size, 2))


ARMS = {
    "prod": lambda f: localize(f, SIGMA, images=False),
    "prod_nats12": lambda f: localize(f, SIGMA, images=False, count_penalty=2.0),
    "prod_wlo0.95": lambda f: localize(f, SIGMA, images=False, slack=(0.95, 2.2)),
    "sg_u4.47": lambda f: scoregate.localize(f, SIGMA, u=4.47),
    "sg_u4.47_wlo0.95": lambda f: scoregate.localize(f, SIGMA, u=4.47, slack=(0.95, 2.2)),
    "sg_u4.0": lambda f: scoregate.localize(f, SIGMA, u=4.0),
    "sg_u5.0": lambda f: scoregate.localize(f, SIGMA, u=5.0),
}


def run(arm, f):
    t = time.perf_counter()
    r = ARMS[arm](f)
    dt = time.perf_counter() - t
    if hasattr(r, "stats"):
        fits = r.stats.fits
        extra = dict(score_evals=r.stats.score_evals, lr_fail=r.stats.lr_fail,
                     removals=r.stats.removals, seeds=len(r.seeds))
    else:
        fits = r.info["search_fits"] + r.info["polish_fits"] + r.info["selection_fits"]
        extra = dict(seeds=r.info["candidates"])
    return np.asarray(r.positions).reshape(-1, 2), dict(fits=fits, sec=dt, **extra)


def near_counts(det, centres, radius):
    if len(det) == 0:
        return np.zeros(len(centres), int)
    return np.array([len(v) for v in cKDTree(det).query_ball_point(centres, radius)])


def main():
    rng = np.random.default_rng(20260923)
    res = {}
    noise = [rng.poisson(B, (1024, 1024)).astype(float) for _ in range(2)]
    shape = (512, 512)
    P = grid(shape, 16, rng)
    flux = np.exp(rng.uniform(np.log(40), np.log(2000), len(P)))
    iso = [rng.poisson(render(shape, P, flux, np.full(len(P), SIGMA))).astype(float)
           for _ in range(3)]
    snr = flux * G_NORM / np.sqrt(B)
    bands = [0, 3, 4, 5, 6, 8, np.inf]
    C = grid(shape, 20, rng, 2.0)
    pair_flux = 500.0      # ideal SNR ~ 14 each
    pairs = {}
    for sep in [0.5, 1.0, 1.5, 2.0, 3.0]:
        ang = rng.uniform(0, np.pi, len(C))
        off = 0.5 * sep * SIGMA * np.c_[np.sin(ang), np.cos(ang)]
        pos = np.r_[C + off, C - off]
        pairs[sep] = rng.poisson(render(shape, pos, np.full(len(pos), pair_flux),
                                        np.full(len(pos), SIGMA))).astype(float)
    Cb = grid(shape, 32, rng, 2.0)
    blobs = {(ws, fb): rng.poisson(render(shape, Cb, np.full(len(Cb), fb),
                                          np.full(len(Cb), ws * SIGMA))).astype(float)
             for ws in [2.0, 3.0] for fb in [1500.0, 5000.0]}

    for arm in ARMS:
        out = res[arm] = {}
        n = 0
        cost = []
        for f in noise:
            det, c = run(arm, f)
            n += len(det)
            cost.append(c)
        out["noise_fp_per_mpx"] = n / len(noise) / (noise[0].size / 1e6)
        out["noise_cost"] = cost[0]
        hit = np.zeros(len(bands) - 1)
        tot = np.zeros(len(bands) - 1)
        fp = 0
        cost = []
        for f in iso:
            det, c = run(arm, f)
            cost.append(c)
            ok = near_counts(det, P, 1.5) > 0
            fp += len(det) - (near_counts(P, det, 1.5) > 0).sum()
            for k in range(len(bands) - 1):
                m = (snr >= bands[k]) & (snr < bands[k + 1])
                hit[k] += ok[m].sum()
                tot[k] += m.sum()
        out["iso_recall"] = {f"[{bands[k]},{bands[k+1]})": hit[k] / tot[k]
                             for k in range(len(bands) - 1)}
        out["iso_fp_per_frame"] = fp / len(iso)
        out["iso_cost"] = cost[0]
        out["pairs"] = {}
        for sep, f in pairs.items():
            det, c = run(arm, f)
            k = near_counts(det, C, 0.5 * sep * SIGMA + 1.5)
            out["pairs"][sep] = {"N=0": float(np.mean(k == 0)), "N=1": float(np.mean(k == 1)),
                                 "N=2": float(np.mean(k == 2)), "N>=3": float(np.mean(k >= 3)),
                                 "fits": c["fits"]}
        out["blobs"] = {}
        for (ws, fb), f in blobs.items():
            det, _ = run(arm, f)
            k = near_counts(det, Cb, 2.0 * ws * SIGMA)
            out["blobs"][f"w{ws}_F{int(fb)}"] = {"mean_N": float(k.mean()),
                                                 "N>=2": float(np.mean(k >= 2))}
        print(arm, json.dumps(out, default=float), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
