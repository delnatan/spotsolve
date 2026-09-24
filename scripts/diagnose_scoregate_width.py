"""What are the dim wide emitters? Width statistics per emitter class.

For each emitter the score gate keeps (upper bound 2.2 sigma), refit its
neighbourhood with every other emitter frozen in the halo, twice: width
free and width fixed at sigma. Report
  gain  (I_fixed - I_free) / phi, nats bought by the free width
  z_w   (w - sigma) / se_w, se_w from the free fit's Fisher matrix * phi
Classes on glycerol beads: in-focus (w < 1.2), bright wide (w > 1.5,
F >= 2500), dim wide (w > 1.5, F < 2500). Controls: synthetic spots at the
beads' background and dispersion, in-focus (w = sigma) and truly wide
(w = 1.7 sigma), at dim fluxes.

Writes output/scoregate/width_diagnosis.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

import spotsolve_rs as _rs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_scoregate as bench  # noqa: E402

from spotsolve import scoregate  # noqa: E402
from spotsolve.native import localize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIGMA, OFFSET = 1.15, 100.0
N_FRAMES = 4


def refit(d, bmap, phi, ems, k, sigma):
    """(gain nats, w, se_w) for emitter k; others frozen. ems rows (A, y, x, w)."""
    H, W = d.shape
    r = int(np.ceil(4 * sigma))
    A, y, x, w = ems[k]
    y0, x0 = max(int(round(y)) - r, 0), max(int(round(x)) - r, 0)
    y1, x1 = min(int(round(y)) + r + 1, H), min(int(round(x)) + r + 1, W)
    sub = np.ascontiguousarray(d[y0:y1, x0:x1])
    bg = bmap[y0:y1, x0:x1]
    level = float(np.median(bg))
    others = np.delete(ems, k, axis=0)
    halo = np.ascontiguousarray(bg - level + scoregate._render(others, y0, x0, y1 - y0, x1 - x0))
    h, wd = sub.shape
    smax = max(float(sub.max()), 1.0)

    def fit(lo_w, hi_w, w0):
        lo = np.array([0.0, 1e-4, -0.5, -0.5, lo_w])
        hi = np.array([max(4 * smax, 10.0), 8 * smax / float(scoregate.psf.peak_factor(sigma)) * 2.2 ** 2,
                       h - 0.5, wd - 0.5, hi_w])
        th = np.clip(np.array([level, A, y - y0, x - x0, w0]), lo + 1e-9, hi - 1e-9)
        return _rs.lmcl_fit_var_sigma(th, h, wd, sub, halo, lo, hi, 100, tol_obj=1e-8)

    th_f, i_free, F, *_ = fit(scoregate.SLACK[0] * sigma, scoregate.SLACK[1] * sigma, w)
    _, i_fix, *_ = fit(0.9999 * sigma, 1.0001 * sigma, sigma)
    try:
        se_w = float(np.sqrt(np.linalg.inv(F)[4, 4] * phi))
    except np.linalg.LinAlgError:
        se_w = np.nan
    return (i_fix - i_free) / phi, float(th_f[4]), se_w


def classify(rows, sigma):
    """rows: (A, w_fit/sigma, gain, w_refit, se_w) -> per-class summary."""
    rows = np.asarray(rows)
    A, wr = rows[:, 0], rows[:, 1]
    out = {}
    for name, m in [("in_focus", wr < 1.2), ("bright_wide", (wr > 1.5) & (A >= 2500)),
                    ("dim_wide", (wr > 1.5) & (A < 2500)), ("dim_all", A < 2500)]:
        x = rows[m]
        if not len(x):
            continue
        z = (x[:, 3] - sigma) / x[:, 4]
        out[name] = {"n": int(m.sum()), "gain_med": float(np.median(x[:, 2])),
                     "gain_gt_8": float(np.mean(x[:, 2] > 8.0)),
                     "se_w_med_sigma": float(np.nanmedian(x[:, 4]) / sigma),
                     "z_w_med": float(np.nanmedian(z)), "z_w_gt_3": float(np.mean(z > 3)),
                     "frac_w_gt_1.5": float(np.mean(x[:, 1] > 1.5))}
    return out


def diagnose(frames, refs, sigma):
    rows = []
    for f, ref in zip(frames, refs):
        r = scoregate.localize(f, sigma, background=ref.background, dispersion=ref.dispersion)
        ems = np.c_[r.amplitudes, r.positions, r.fit_sigma]
        for k in range(len(ems)):
            gain, w, se = refit(f, ref.background, r.dispersion, ems, k, sigma)
            rows.append((ems[k, 0], ems[k, 3] / sigma, gain, w, se))
    return rows


def main():
    st = tifffile.imread(ROOT / "data" / "beads_80pct-glycerol_crop.tif").astype(float) - OFFSET
    frames = list(st[:N_FRAMES])
    refs = [localize(f, SIGMA, images=False) for f in frames]
    phi = float(np.median([r.dispersion for r in refs]))
    bg = float(np.median([np.median(r.background) for r in refs]))
    res = {"beads": classify(diagnose(frames, refs, SIGMA), SIGMA), "phi": phi, "background": bg}

    rng = np.random.default_rng(7)
    shape = (256, 256)
    P = bench.grid(shape, 16, rng)
    flux = rng.choice([700.0, 1200.0, 2000.0], len(P))
    for name, wmul in [("synthetic_in_focus", 1.0), ("synthetic_wide_1.7", 1.7)]:
        img = np.full(shape, bg - bench.B) + bench.render(shape, P, flux, np.full(len(P), wmul * SIGMA))
        syn = [phi * rng.poisson(np.maximum(img, 0) / phi) for _ in range(2)]
        srefs = [localize(f, SIGMA, images=False) for f in syn]
        res[name] = classify(diagnose(syn, srefs, SIGMA), SIGMA)
    (ROOT / "output" / "scoregate" / "width_diagnosis.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if isinstance(v, dict):
            for c, s in v.items():
                print(f"{k:20s} {c:12s} " + " ".join(f"{a}={b:.2f}" if isinstance(b, float) else f"{a}={b}" for a, b in s.items()))
    print("phi", phi, "background", bg)


if __name__ == "__main__":
    main()
