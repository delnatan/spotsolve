"""Sweep the score-gate threshold u: synthetic scenes and GEM stability.

Synthetic scenes are those of benchmark_scoregate.py. GEM stability links
each frame's emitters to the next by mutual nearest neighbour within 2.5 px
and reports how much flux and width jump between frames.

Writes output/scoregate/u_sweep.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_scoregate as bench  # noqa: E402

from spotsolve import scoregate  # noqa: E402
from spotsolve.native import localize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
US = [4.0, 4.25, 4.47, 4.75, 5.0]
LINK_R = 2.5


def stability(frames):
    """Frame-to-frame link rate and flux/width jumps; rows (y, x, A, w)."""
    d_a, d_w, linked, tot = [], [], 0, 0
    for a, b in zip(frames[:-1], frames[1:]):
        tot += len(a)
        if len(a) == 0 or len(b) == 0:
            continue
        da, ia = cKDTree(b[:, :2]).query(a[:, :2])
        _, ib = cKDTree(a[:, :2]).query(b[:, :2])
        m = (da < LINK_R) & (ib[ia] == np.arange(len(a)))
        linked += m.sum()
        d_a += list(np.abs(np.log(a[m, 2] / b[ia[m], 2])))
        d_w += list(np.abs(a[m, 3] - b[ia[m], 3]))
    return {"per_frame": tot / (len(frames) - 1), "linked": linked / max(tot, 1),
            "dlogA_med": float(np.median(d_a)), "dlogA_p90": float(np.percentile(d_a, 90)),
            "dw_med": float(np.median(d_w)), "dw_p90": float(np.percentile(d_w, 90))}


def main():
    rng = np.random.default_rng(20260923)
    s, b = bench.SIGMA, bench.B
    res = {}

    noise = [rng.poisson(b, (1024, 1024)).astype(float) for _ in range(4)]
    refs = [localize(f, s, images=False) for f in noise]
    shape = (512, 512)
    P = bench.grid(shape, 16, rng)
    flux = np.exp(rng.uniform(np.log(40), np.log(2000), len(P)))
    iso = [rng.poisson(bench.render(shape, P, flux, np.full(len(P), s))).astype(float)
           for _ in range(3)]
    iso_refs = [localize(f, s, images=False) for f in iso]
    snr = flux * bench.G_NORM / np.sqrt(b)
    bands = [0, 3, 4, 5, 6, 8, np.inf]
    C = bench.grid(shape, 20, rng, 2.0)
    pairs = {}
    for sep in [1.5, 2.0, 3.0]:
        ang = rng.uniform(0, np.pi, len(C))
        off = 0.5 * sep * s * np.c_[np.sin(ang), np.cos(ang)]
        pos = np.r_[C + off, C - off]
        f = rng.poisson(bench.render(shape, pos, np.full(len(pos), 500.0),
                                     np.full(len(pos), s))).astype(float)
        pairs[sep] = (f, localize(f, s, images=False))

    stack = tifffile.imread(ROOT / "data" / "hyp7gem_wt_01_crop_128x128.tif").astype(float) - 100.0
    gem_refs = [localize(f, s, images=False) for f in stack]

    def sg(f, ref, u):
        return scoregate.localize(f, s, u=u, background=ref.background,
                                  dispersion=ref.dispersion)

    res["prod_gem"] = stability([np.c_[r.positions, r.amplitudes, r.fit_sigma / s]
                                 for r in gem_refs])
    for u in US:
        out = res[f"u{u}"] = {}
        rs = [sg(f, r, u) for f, r in zip(noise, refs)]
        out["noise_fp_per_mpx"] = sum(len(r.amplitudes) for r in rs) / (len(noise) * 1024 ** 2 / 1e6)
        out["noise_seeds_per_mpx"] = sum(len(r.seeds) for r in rs) / (len(noise) * 1024 ** 2 / 1e6)
        hit, tot, fp, fits = np.zeros(6), np.zeros(6), 0, 0
        for f, ref in zip(iso, iso_refs):
            r = sg(f, ref, u)
            det = r.positions
            fits += r.stats.fits
            ok = bench.near_counts(det, P, 1.5) > 0
            fp += len(det) - (bench.near_counts(P, det, 1.5) > 0).sum()
            for k in range(6):
                m = (snr >= bands[k]) & (snr < bands[k + 1])
                hit[k] += ok[m].sum()
                tot[k] += m.sum()
        out["iso_recall"] = {f"[{bands[k]},{bands[k+1]})": hit[k] / tot[k] for k in range(6)}
        out["iso_fp_per_frame"] = fp / len(iso)
        out["iso_fits_per_frame"] = fits / len(iso)
        out["pairs_N2"] = {}
        for sep, (f, ref) in pairs.items():
            k = bench.near_counts(sg(f, ref, u).positions, C, 0.5 * sep * s + 1.5)
            out["pairs_N2"][sep] = float(np.mean(k == 2))
        gem = [sg(f, r, u) for f, r in zip(stack, gem_refs)]
        out["gem"] = stability([np.c_[r.positions, r.amplitudes, r.fit_sigma / s] for r in gem])
        out["gem"]["fits_per_frame"] = float(np.mean([r.stats.fits for r in gem]))
        print(u, json.dumps(out, default=float), flush=True)
    (ROOT / "output" / "scoregate" / "u_sweep.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
