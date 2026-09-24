"""Freeze score-gate prototype outputs as the Rust port's parity contract.

Writes tests/fixtures/10_scoregate.json. Each case embeds its input frame
(whole ADU, offset included) because most source TIFFs are not tracked,
and records u, phi, the seeds (y, x, z) in visit order, every emitter
(A, y, x, w) in output order, and the search counters. Two small cases
also record the median background map.

Run once from the prototype; the Python prototype is retired afterwards.
"""

import json
from pathlib import Path

import numpy as np
import tifffile

from spotsolve import psf, scoregate

ROOT = Path(__file__).resolve().parents[1]
B = 50.0


def render(shape, pos, flux, sig):
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    th = psf.pack_var_sigma(B, flux, pos[:, 0], pos[:, 1], sig)
    return psf.model_var_sigma(th, yy, xx)


def cases():
    rng = np.random.default_rng(20260924)
    s = 1.45
    yield "noise", rng.poisson(np.full((96, 96), B)).astype(float), s, 0.0, True
    g = np.arange(12, 96, 24, dtype=float)
    pos = np.array([(y, x) for y in g for x in g]) + rng.uniform(-2, 2, (16, 2))
    flux = np.exp(rng.uniform(np.log(150), np.log(2000), len(pos)))
    yield "isolated", rng.poisson(render((96, 96), pos, flux, np.full(len(pos), s))).astype(float), s, 0.0, True
    pos = np.array([[20.0, 20.0], [20.0, 20.0 + 2 * s], [44.0, 40.0], [44.0 + 1.5 * s, 40.0]])
    yield "pairs", rng.poisson(render((64, 64), pos, np.full(4, 500.0), np.full(4, s))).astype(float), s, 0.0, False
    pos = np.array([[30.0, 28.0], [30.0, 40.0]])
    yield "blob", rng.poisson(render((64, 64), pos, np.array([5000.0, 800.0]),
                                     np.array([2.0 * s, s]))).astype(float), s, 0.0, False
    gem = tifffile.imread(ROOT / "data" / "hyp7gem_wt_01_crop_128x128.tif")[0].astype(float)
    yield "gem", gem, 1.45, 100.0, False
    beads = tifffile.imread(ROOT / "data" / "beads_80pct-glycerol_crop.tif")[0, :96, :96].astype(float)
    yield "beads", beads, 1.15, 100.0, False


def main():
    out = {"fp_per_mpx": scoregate.FP_PER_MPX, "slack": list(scoregate.SLACK), "cases": []}
    for name, frame, sigma, offset, keep_bg in cases():
        r = scoregate.localize(frame, sigma, offset=offset)
        case = {"name": name, "sigma": sigma, "offset": offset, "shape": list(frame.shape),
                "frame": frame.astype(int).ravel().tolist(), "u": r.u, "phi": r.dispersion,
                "seeds": np.c_[r.seeds, r.seed_z].tolist(),
                "emitters": np.c_[r.amplitudes, r.positions, r.fit_sigma].tolist(),
                "fits": r.stats.fits, "adds": r.stats.adds, "lr_fail": r.stats.lr_fail}
        if keep_bg:
            case["background"] = r.background.ravel().tolist()
        out["cases"].append(case)
        print(f"{name}: u {r.u:.3f} phi {r.dispersion:.3f} seeds {len(r.seeds)} emitters {len(r.amplitudes)} fits {r.stats.fits}")
    path = ROOT / "tests" / "fixtures" / "10_scoregate.json"
    path.write_text(json.dumps(out))
    print(path, path.stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main()
