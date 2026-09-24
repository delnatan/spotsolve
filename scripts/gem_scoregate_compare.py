"""Production vs score-gated prototype on the GEM 128x128 crop, for visual
checks across adjacent frames.

Writes output/scoregate/gem_compare.json: the offset-subtracted stack
(rounded to 0.1 ADU) and, per frame, both arms' emitters with the frame-wide
K = 0 score z at each emitter's pixel.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

from spotsolve import scoregate
from spotsolve.native import localize

ROOT = Path(__file__).resolve().parents[1]
SIGMA, OFFSET = 1.45, 100.0


def main(n_frames=None):
    st = tifffile.imread(ROOT / "data" / "hyp7gem_wt_01_crop_128x128.tif").astype(float) - OFFSET
    st = st[:n_frames] if n_frames else st
    frames = []
    for t, f in enumerate(st):
        p = localize(f, SIGMA, images=False)
        s = scoregate.localize(f, SIGMA, background=p.background, dispersion=p.dispersion)

        def rows(pos, amp, sig):
            out = []
            for (y, x), a, w in zip(pos, amp, sig):
                iy, ix = int(np.clip(round(y), 0, 127)), int(np.clip(round(x), 0, 127))
                out.append([round(float(y), 2), round(float(x), 2), round(float(a), 1),
                            round(float(w) / SIGMA, 2), round(float(s.z0[iy, ix]), 1)])
            return out

        frames.append({"prod": rows(p.positions, p.amplitudes, p.fit_sigma),
                       "sg": rows(s.positions, s.amplitudes, s.fit_sigma),
                       "phi": round(float(p.dispersion), 3),
                       "sg_fits": s.stats.fits,
                       "prod_fits": p.info["search_fits"] + p.info["polish_fits"]})
        print(t, len(frames[-1]["prod"]), len(frames[-1]["sg"]), flush=True)
    out = ROOT / "output" / "scoregate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gem_compare.json").write_text(json.dumps({
        "sigma": SIGMA, "offset": OFFSET, "shape": list(st.shape),
        "stack": np.round(st, 1).tolist(), "frames": frames}))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
