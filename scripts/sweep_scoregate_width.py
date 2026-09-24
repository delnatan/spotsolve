"""Sweep the score gate's upper width bound (slack[1]) at the default u.

Scenes (sigma 1.45, background 50 ADU):
  noise    pure background, false emitters per Mpx
  pairs    equal (500/500) and unequal (500/200) flux pairs; fraction N=2
  spread   isolated in-focus spots, true width U(0.9, 1.25) sigma; recall,
           split rate (>= 2 emitters within 2.5 px) and false positives
  blobs    wide Gaussians (1.5, 2, 3 sigma); mean emitters and N >= 2
  gem      GEM crop: emitters/frame, stability, share of fits at the bound

Writes output/scoregate/width_sweep.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_scoregate as bench  # noqa: E402
from sweep_scoregate_u import stability  # noqa: E402

from spotsolve import scoregate  # noqa: E402
from spotsolve.native import localize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HIS = [2.2, 1.5, 1.3]


def main():
    rng = np.random.default_rng(20260924)
    s, b = bench.SIGMA, bench.B
    shape = (512, 512)

    def scene(img):
        f = rng.poisson(img).astype(float)
        return f, localize(f, s, images=False)

    noise = [scene(np.full((1024, 1024), b)) for _ in range(3)]
    C = bench.grid(shape, 20, rng, 2.0)
    pairs = {}
    for fl in [(500.0, 500.0), (500.0, 200.0)]:
        for sep in [1.0, 1.5, 2.0, 2.5]:
            ang = rng.uniform(0, np.pi, len(C))
            off = 0.5 * sep * s * np.c_[np.sin(ang), np.cos(ang)]
            pos = np.r_[C + off, C - off]
            flux = np.r_[np.full(len(C), fl[0]), np.full(len(C), fl[1])]
            pairs[f"{int(fl[0])}/{int(fl[1])}@{sep}"] = (sep, scene(bench.render(shape, pos, flux, np.full(len(pos), s))))
    P = bench.grid(shape, 16, rng)
    flux = np.exp(rng.uniform(np.log(100), np.log(2000), len(P)))
    wid = rng.uniform(0.9, 1.25, len(P)) * s
    spread = [scene(bench.render(shape, P, flux, wid)) for _ in range(2)]
    Cb = bench.grid(shape, 32, rng, 2.0)
    blobs = {f"w{ws}_F{int(fb)}": (ws, scene(bench.render(shape, Cb, np.full(len(Cb), fb),
                                                          np.full(len(Cb), ws * s))))
             for ws in [1.5, 2.0, 3.0] for fb in [1500.0, 5000.0]}
    stack = tifffile.imread(ROOT / "data" / "hyp7gem_wt_01_crop_128x128.tif").astype(float) - 100.0
    gem = [(f, localize(f, s, images=False)) for f in stack]

    def sg(fr, hi):
        f, ref = fr
        return scoregate.localize(f, s, slack=(scoregate.SLACK[0], hi),
                                  background=ref.background, dispersion=ref.dispersion)

    res = {}
    for hi in HIS:
        out = res[f"hi{hi}"] = {}
        out["noise_fp_per_mpx"] = sum(len(sg(fr, hi).amplitudes) for fr in noise) / (3 * 1024 ** 2 / 1e6)
        out["pairs_N2"] = {}
        for k, (sep, fr) in pairs.items():
            n = bench.near_counts(sg(fr, hi).positions, C, 0.5 * sep * s + 1.5)
            out["pairs_N2"][k] = {"N=2": float(np.mean(n == 2)), "N=1": float(np.mean(n == 1)),
                                  "N>=3": float(np.mean(n >= 3))}
        rec = split = fp = 0
        for fr in spread:
            det = sg(fr, hi).positions
            rec += np.mean(bench.near_counts(det, P, 1.5) > 0)
            split += np.mean(bench.near_counts(det, P, 2.5) >= 2)
            fp += len(det) - (bench.near_counts(P, det, 2.5) > 0).sum()
        out["spread"] = {"recall": rec / 2, "split": split / 2, "fp_per_frame": fp / 2}
        out["blobs"] = {}
        for k, (ws, fr) in blobs.items():
            n = bench.near_counts(sg(fr, hi).positions, Cb, 2.0 * ws * s)
            out["blobs"][k] = {"mean_N": float(n.mean()), "N>=2": float(np.mean(n >= 2))}
        rs = [sg(fr, hi) for fr in gem]
        out["gem"] = stability([np.c_[r.positions, r.amplitudes, r.fit_sigma / s] for r in rs])
        w = np.concatenate([r.fit_sigma / s for r in rs])
        out["gem"]["at_upper_bound"] = float(np.mean(w > 0.98 * hi))
        out["gem"]["w_median"] = float(np.median(w))
        out["gem"]["fits_per_frame"] = float(np.mean([r.stats.fits for r in rs]))
        print(hi, json.dumps(out, default=float), flush=True)
    (ROOT / "output" / "scoregate" / "width_sweep.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
