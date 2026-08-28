"""Head-to-head: boxsolve vs sfwloc's Rust DAOPHOT, on the same frames.

Step 1 of the architecture decision (see the plan). This produces the artifact
to look at; it deliberately does NOT draw a conclusion, because the comparison
as constructed is confounded -- read `comparison/NOTES.md`, which this script
writes alongside the images.

    python compare_methods.py                 # all frames, writes comparison/

The two methods do not model the background the same way, and on these frames
that difference is larger than anything else being compared:

  * `boxsolve` fits ONE scalar background for the whole frame
    (`calibrate.robust_background`: median of the pixels no emitter reaches).
  * `sfwloc_py.daophot_fit` fits a local background MAP internally
    (`bg_kernel_size`, default 25) and does not return it.

So DAOPHOT's amplitudes are defined against a background this script cannot
see. Rendering its output against a flat background counts whatever its map
absorbed as unexplained residual. To bracket that, every DAOPHOT row is
rendered twice: once against `robust_background` (pessimistic -- the naive
comparison) and once against the scalar background that best fits its own
output (optimistic -- the most favourable flat reading of its answer). The
truth is somewhere between, and pinning it down is step 2.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import audit
import boxsolve
import calibrate
import psf

SFWLOC = os.path.expanduser("~/Projects/github/sfwloc")
sys.path.insert(0, SFWLOC)
try:
    import sfwloc_py
except ImportError as e:                                    # pragma: no cover
    sfwloc_py = None
    print(f"WARNING: sfwloc_py not importable ({e}); DAOPHOT rows will be skipped")

OUT = "comparison"
SIGMA = 1.2
OFFSET = 100.0
GAIN = 4.23

FRAMES = [
    ("FOV1", "beads_60x_still.tif", GAIN),
    ("FOV2", "beads_60x_still_02.tif", GAIN),
    ("dense184", os.path.join(SFWLOC, "data", "beads_dense_frame184.tif"), None),
]


def best_scalar_background(d_e, pos, amps, sigma):
    """The scalar b minimizing the Poisson I-divergence at fixed positions and
    amplitudes. dI/db = sum(1 - d/(b + p)) = 0, monotone in b, so a bracketed
    root solve is exact and cheap.

    This is the most favourable flat-background reading of a result whose own
    background model was a map: it is an upper bound on how well that result
    can look under this script's rendering, not an estimate of its background.
    """
    p = calibrate.render_model(pos, amps, sigma, d_e.shape, 0.0)

    def dIdb(b):
        return float(np.sum(1.0 - d_e / np.maximum(p + b, 1e-9)))

    # dI/db is INCREASING in b (d/(b+p) shrinks as b grows), so the root is
    # bracketed by dIdb(lo) < 0 < dIdb(hi). Getting this test backwards returns
    # b = lo for every input, which silently drives the background to zero --
    # and a near-zero background then deflates the audit's own score statistic
    # (its denominator is sqrt(sum g^2/m)), so the broken version looked like an
    # improvement rather than an error.
    lo, hi = 1e-6, max(float(d_e.max()), 1.0)
    if dIdb(lo) >= 0:
        return lo
    if dIdb(hi) <= 0:
        return hi
    return float(brentq(dIdb, lo, hi, xtol=1e-9))


def summarize(tag, d_e, pos, amps, bg, sigma, dt, extra=""):
    model = calibrate.render_model(pos, amps, sigma, d_e.shape, bg)
    a = audit.audit_result(d_e, model, sigma)
    ai = audit.audit_result(d_e, model, sigma, border=3)
    excess = float(np.sum(d_e - bg))
    return dict(
        method=tag, note=extra, N=len(amps), sumA=float(np.sum(amps)),
        background=float(bg), image_flux_above_bg=excess,
        flux_ratio=float(np.sum(amps)) / excess if excess > 0 else np.nan,
        missed=a["n_missed"], piled=a["n_piled"],
        missed_interior=ai["n_missed"], piled_interior=ai["n_piled"],
        z_min=a["z_min"], z_max=a["z_max"], z_robust_sd=a["z_robust_std"],
        seconds=dt, _model=model, _audit=a, _pos=pos, _amps=amps,
    )


def run_boxsolve(raw, gain, sigma):
    t = time.time()
    r = boxsolve.detect_boxes(raw, sigma=sigma, offset=OFFSET, gain=gain,
                              n_outer=1, k_max=16, verbose=0)
    dt = time.time() - t
    d_e = (raw.astype(float) - OFFSET) / r.gain
    return d_e, r, dt


def run_daophot(d_e, sigma, **kw):
    """(positions, amplitudes, seconds) or ('FAILED', message, seconds).

    `daophot_fit` can raise rather than return a partial answer -- on the
    154x154 dense frame at alpha=0.05 it reports "group N (13 members) did not
    converge within its iteration budget". That is a result worth recording, so
    it is caught and reported rather than allowed to abort the sweep.
    """
    if sfwloc_py is None:
        return None
    bg0 = float(np.median(d_e))
    t = time.time()
    try:
        out = sfwloc_py.daophot_fit(d_e, sigma, bg0, **kw)
    except Exception as e:                                  # noqa: BLE001
        return "FAILED", str(e), time.time() - t
    dt = time.time() - t
    amps = np.asarray(out[0], float).ravel()
    pos = np.asarray(out[1], float).reshape(-1, 2)
    return pos, amps, dt


def panel(ax_row, d_e, res, title, sigma):
    """data | model | normalized residual | score map with audit rings."""
    m = res["_model"]
    nr = (d_e - m) / np.sqrt(np.maximum(m, 1e-6))
    z = audit.score_map(d_e, m, sigma)
    a = res["_audit"]

    ax_row[0].imshow(d_e, cmap="gray")
    if len(res["_pos"]):
        ax_row[0].plot(res["_pos"][:, 1], res["_pos"][:, 0], "r+", ms=6, mew=1.0)
    ax_row[0].set_ylabel(title, fontsize=8)
    ax_row[0].set_title(f"data + {res['N']} positions", fontsize=8)

    ax_row[1].imshow(m, cmap="gray")
    ax_row[1].set_title(f"model  (sum A = {res['sumA']:.0f},"
                        f" {100*res['flux_ratio']:.0f}% of flux)", fontsize=8)

    ax_row[2].imshow(nr, cmap="RdBu_r", vmin=-5, vmax=5)
    ax_row[2].set_title(f"norm. residual (med {np.median(nr):+.2f})", fontsize=8)

    ax_row[3].imshow(z, cmap="RdBu_r", vmin=-8, vmax=8)
    if len(a["positive"]):
        ax_row[3].scatter(a["positive"][:, 1], a["positive"][:, 0], s=90,
                          facecolors="none", edgecolors="yellow", lw=1.1)
    if len(a["negative"]):
        ax_row[3].scatter(a["negative"][:, 1], a["negative"][:, 0], s=90,
                          facecolors="none", edgecolors="cyan", lw=1.1)
    ax_row[3].set_title(f"score z: {a['n_missed']} missed / {a['n_piled']} piled"
                        f"  |z|max {max(abs(a['z_max']), abs(a['z_min'])):.1f}",
                        fontsize=8)
    for ax in ax_row:
        ax.set_xticks([]); ax.set_yticks([])


def write_notes(rows):
    """The reading instructions. Written next to the images because the
    headline numbers in them are not yet a fair comparison."""
    def tab(frame):
        out = ["| method | note | N | sum A | % of flux | audit frame | audit interior | z max | s |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["frame"] != frame:
                continue
            if r.get("N", -1) < 0:
                out.append(f"| {r['method']} | **{r['note']}** | — | — | — | — | — | — | "
                           f"{r.get('seconds', float('nan')):.2f} |")
                continue
            out.append(
                f"| {r['method']} | {r['note']} | {r['N']} | {r['sumA']:.0f} | "
                f"{100*r['flux_ratio']:.0f}% | {r['missed']}/{r['piled']} | "
                f"**{r['missed_interior']}/{r['piled_interior']}** | "
                f"{r['z_max']:+.1f} | {r['seconds']:.2f} |")
        return "\n".join(out)

    frames = []
    for f in dict.fromkeys(r["frame"] for r in rows):
        frames.append(f"### {f}\n\n{tab(f)}\n")

    txt = f"""# boxsolve vs sfwloc DAOPHOT — read this before the images

Generated by `compare_methods.py`. **These numbers are not yet a fair
comparison.** They exist to show what the gap looks like and to scope the work
that would make them fair (step 2 of the plan).

## The confound

The two methods do not model the background the same way:

- `boxsolve` fits **one scalar background** for the frame
  (`calibrate.robust_background`).
- `sfwloc_py.daophot_fit` fits a **local background map** internally
  (`bg_kernel_size`, default 25) and **does not return it**.

So DAOPHOT's amplitudes are defined against a background this script cannot
see. Every DAOPHOT row is therefore rendered twice, to bracket it:

- `flat robust_background (pessimistic)` — the naive comparison; whatever its
  background map absorbed is counted as unexplained residual;
- `best-fit scalar bg (optimistic)` — the single background minimizing the
  Poisson I-divergence at its own fixed positions and amplitudes, i.e. the most
  favourable flat reading of its answer.

Look at `*_matchedN.png`: DAOPHOT's residual is dominated by a **smooth,
large-scale** blue/red pattern, which is the signature of a background error,
not of missed point sources. Its *positions* look broadly comparable to
`boxsolve`'s. **Do not read the audit column as a verdict on DAOPHOT's
detection quality.**

## What is nonetheless established

- DAOPHOT's amplitudes are **total flux**, not peak height (a peak-height
  reading implies ~5x the flux present in the image), so `sum A` is comparable.
- `source_cost` swept 0.1 -> 8.13 changes nothing on FOV1 (N=40 throughout).
  **The acceptance criterion is not the binding constraint** — so whatever
  separates these two, it is not Bayes factor vs chi2(3). The FIND threshold
  `alpha` is what moves N.
- DAOPHOT is **~10-40x faster**.
- `daophot_fit` **raises** on the dense frame at some settings
  (`group N (13 members) did not converge within its iteration budget`)
  rather than returning a partial result.

## What this says about boxsolve, unflatteringly

On `dense184` (154x154, ~3x the density of the bead frames) **boxsolve is not
clean either**: 65 unexplained interior peaks at gain 4.23. Its clean audits on
FOV1/FOV2 do not generalize to this frame. Note also that `dense184`'s gain is
not measured — `calibrate.estimate_gain` returns 2.236 against the 4.23
measured on the bead frames, and that alone moves N from 428 to 596. Per the
gain law (`log BF = dI/g + 1.5 log g + const`) that is a pure threshold shift,
not a different fit, but it means neither number should be quoted as *the*
answer for that frame.

## Results

{"".join(frames)}
Columns: `audit interior` excludes a 3 px rim, which is where both methods'
known problems live. `% of flux` is `sum A` over the image flux above the
rendered background.
"""
    p = os.path.join(OUT, "NOTES.md")
    with open(p, "w") as f:
        f.write(txt)
    print(f"wrote {p}")


def main(args):
    os.makedirs(OUT, exist_ok=True)
    rows = []

    for label, path, gain in FRAMES:
        if not os.path.exists(path):
            print(f"  skip {label}: {path} not found")
            continue
        raw = tifffile.imread(path)
        print(f"\n=== {label}  {raw.shape}  {path}")

        # The gain is a dispersion parameter: it cannot change the fitted
        # configuration, only the detection threshold (log BF = dI/g +
        # 1.5 log g + const). So on a frame whose gain is not measured, run
        # both the estimate and the value measured on the bead frames -- a
        # disagreement between them is a threshold difference, not a fit
        # difference, and it is the first thing to suspect if the audit is bad.
        gains = [gain] if gain else [None, GAIN]
        results = []
        for gi, g in enumerate(gains):
            d_e, r, dt = run_boxsolve(raw, g, SIGMA)
            b = summarize("boxsolve", d_e, r.positions, r.amplitudes,
                          r.background, SIGMA, dt,
                          extra=f"gain={r.gain:.3f}"
                                + ("" if g else " (estimated)"))
            results.append(b)
            print(f"  boxsolve g={r.gain:5.3f}{'*' if not g else ' '} "
                  f"N={b['N']:4d}  sumA={b['sumA']:8.0f}"
                  f"  flux={100*b['flux_ratio']:5.1f}%  audit {b['missed']}/{b['piled']}"
                  f"  interior {b['missed_interior']}/{b['piled_interior']}"
                  f"  {b['seconds']:.2f}s")
        # DAOPHOT is compared against the last (pinned-gain) photoelectron image
        box = results[-1]

        for alpha in args.alphas:
            got = run_daophot(d_e, SIGMA, alpha=alpha)
            if got is None:
                break
            pos, amps, ddt = got
            if isinstance(pos, str):            # daophot_fit raised
                print(f"  daophot a={alpha:<5g} FAILED after {ddt:.2f}s: {amps}")
                rows.append(dict(frame=label,
                                 shape=f"{raw.shape[0]}x{raw.shape[1]}",
                                 method=f"daophot a={alpha:g}",
                                 note=f"RAISED: {amps}", N=-1, seconds=ddt))
                continue
            for bgmode in ("robust", "bestfit"):
                bg = (calibrate.robust_background(d_e, pos, SIGMA)
                      if bgmode == "robust"
                      else best_scalar_background(d_e, pos, amps, SIGMA))
                res = summarize(f"daophot a={alpha:g}", d_e, pos, amps, bg,
                                SIGMA, ddt,
                                extra=("flat robust_background (pessimistic)"
                                       if bgmode == "robust"
                                       else "best-fit scalar bg (optimistic)"))
                results.append(res)
                print(f"  daophot a={alpha:<5g} [{bgmode:7s}] N={res['N']:4d}"
                      f"  sumA={res['sumA']:8.0f}  flux={100*res['flux_ratio']:5.1f}%"
                      f"  audit {res['missed']}/{res['piled']}"
                      f"  interior {res['missed_interior']}/{res['piled_interior']}"
                      f"  {res['seconds']:.2f}s")

        for res in results:
            rows.append(dict(frame=label, shape=f"{raw.shape[0]}x{raw.shape[1]}",
                             **{k: v for k, v in res.items()
                                if not k.startswith("_")}))

        # ---- per-frame figure: one row per configuration ----
        show = [x for x in results
                if x["method"] == "boxsolve" or "optimistic" in x.get("note", "")]
        fig, ax = plt.subplots(len(show), 4,
                               figsize=(13, 3.05 * len(show)), squeeze=False)
        for i, res in enumerate(show):
            panel(ax[i], d_e, res,
                  f"{res['method']}\n{res['note'][:28]}", SIGMA)
        fig.suptitle(
            f"{label} ({raw.shape[0]}x{raw.shape[1]}, sigma={SIGMA}, "
            f"photoelectrons)   yellow = unexplained flux, cyan = over-modelled"
            f"\nDAOPHOT rows use its most favourable flat background - see NOTES.md",
            fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = os.path.join(OUT, f"{label}_methods.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  wrote {p}")

        # ---- side-by-side at matched N ----
        dao = [x for x in results
               if x["method"] != "boxsolve" and "optimistic" in x.get("note", "")]
        if dao:
            best = min(dao, key=lambda x: abs(x["N"] - box["N"]))
            fig, ax = plt.subplots(2, 4, figsize=(13, 6.2), squeeze=False)
            panel(ax[0], d_e, box, f"boxsolve\nN={box['N']}", SIGMA)
            panel(ax[1], d_e, best,
                  f"{best['method']}\nN={best['N']} (closest to boxsolve)", SIGMA)
            fig.suptitle(f"{label}: matched-N comparison  "
                         f"(boxsolve N={box['N']} vs DAOPHOT N={best['N']})",
                         fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            p = os.path.join(OUT, f"{label}_matchedN.png")
            fig.savefig(p, dpi=130)
            plt.close(fig)
            print(f"  wrote {p}")

    write_notes(rows)

    # ---- csv ----
    if rows:
        keys = [k for k in rows[0] if not k.startswith("_")]
        p = os.path.join(OUT, "results.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r_ in rows:
                w.writerow({k: r_.get(k, "") for k in keys})
        print(f"\nwrote {p}  ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0.01, 0.05, 0.1],
                    help="DAOPHOT FIND significance levels to sweep")
    main(ap.parse_args())
