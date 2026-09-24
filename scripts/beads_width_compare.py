"""Tiling check on glycerol beads: the score gate at two upper width bounds.

Beads in 80% glycerol move ~0.85 px/frame (median, no drift), so a bead reappears
close by in adjacent frames; defocused beads test whether a tight bound tiles one wide
object into several in-focus emitters. sigma = 1.15 px is the p10-p25
fitted width of bright isolated beads; offset 100 ADU.

Writes output/scoregate/beads_compare.json (stack + per-arm emitters).
"""

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

from spotsolve import scoregate
from spotsolve.native import localize

ROOT = Path(__file__).resolve().parents[1]
SIGMA, OFFSET = 1.15, 100.0
ARMS = {"sg22": 2.2, "sg15": 1.5}


def rows(pos, amp, sig, z0):
    out = []
    for (y, x), a, w in zip(pos, amp, sig):
        iy = int(np.clip(round(y), 0, z0.shape[0] - 1))
        ix = int(np.clip(round(x), 0, z0.shape[1] - 1))
        out.append([round(float(y), 2), round(float(x), 2), round(float(a), 1),
                    round(float(w) / SIGMA, 2), round(float(z0[iy, ix]), 1)])
    return out


def main():
    st = tifffile.imread(ROOT / "data" / "beads_80pct-glycerol_crop.tif").astype(float) - OFFSET
    frames = []
    for f in st:
        p = localize(f, SIGMA, images=False)
        fr = {}
        for name, hi in ARMS.items():
            r = scoregate.localize(f, SIGMA, slack=(scoregate.SLACK[0], hi),
                                   background=p.background, dispersion=p.dispersion)
            fr[name] = rows(r.positions, r.amplitudes, r.fit_sigma, r.z0)
        fr["prod"] = rows(p.positions, p.amplitudes, p.fit_sigma, r.z0)
        frames.append(fr)
    # Tiling: emitters of the tight arm within 1.5 * w * sigma of a wide (w > 1.5)
    # emitter of the loose arm.
    for k, fr in enumerate(frames):
        a, b = np.array(fr["sg22"]), np.array(fr["sg15"])
        wide = a[a[:, 3] > 1.5]
        n = [len(v) for v in cKDTree(b[:, :2]).query_ball_point(wide[:, :2], 1.5 * wide[:, 3] * SIGMA)] if len(wide) else []
        fr["tiles"] = n
    print("per frame  sg22 %.0f  sg15 %.0f  prod %.0f" % tuple(np.mean([len(f[a]) for f in frames]) for a in ["sg22", "sg15", "prod"]))
    wide = np.concatenate([[r[3] for r in f["sg22"]] for f in frames])
    tiles = np.concatenate([f["tiles"] for f in frames])
    print(f"sg22 wide (w>1.5σ): {np.mean(wide > 1.5):.2f} of emitters, {np.sum(wide > 1.5) / len(frames):.1f}/frame; "
          f"sg15 emitters per wide object: mean {tiles.mean():.2f}, >=2 in {np.mean(tiles >= 2):.2f}")
    out = ROOT / "output" / "scoregate"
    (out / "beads_compare.json").write_text(json.dumps({
        "sigma": SIGMA, "offset": OFFSET, "shape": list(st.shape),
        "stack": np.round(st).tolist(), "frames": frames}))


if __name__ == "__main__":
    main()
