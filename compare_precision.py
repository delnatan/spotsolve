"""Localization precision: boxsolve vs sfwloc's DAOPHOT, against known truth.

This is the one comparison between these two that is **not confounded by the
background convention** (see `comparison/NOTES.md`). Positions are positions:
whatever each method assigns to background, a centroid is either on the true
emitter or it is not. So this is the part of the head-to-head that can support
a conclusion today.

Three things are measured, on synthetic fields with ground truth:

1. **Localization error** of matched detections -- median, p90, and RMSE. RMSE
   alone hides the tail, which is where a mis-assigned centroid lives.
2. **Efficiency against the CRLB.** Both methods report a position CRLB, so the
   question is not only "how close" but "how close relative to the information
   the data actually contains". A method at 1.0x is extracting everything
   available; 2x means it is leaving half the precision on the table.
3. **Pile-up.** Two ways, because they are different failures:
   - `duplicates`: two or more detections matched to the SAME true emitter --
     one real source modelled as several;
   - `close_pairs`: detections closer to each other than `pile_sep` (default
     1 sigma) -- below the identifiability limit, so at least one is spurious
     regardless of what truth says.

    python compare_precision.py --seeds 8
"""

import argparse
import os
import sys

import numpy as np

import boxsolve
import calibrate
import simulate

SFWLOC = os.path.expanduser("~/Projects/github/sfwloc")
sys.path.insert(0, SFWLOC)
try:
    import sfwloc_py
except ImportError as e:                                    # pragma: no cover
    sfwloc_py = None
    print(f"WARNING: sfwloc_py not importable ({e})")

SIGMA, GAIN, OFFSET, BG_E = 1.2, 4.23, 100.0, 4.0
AMP = (900.0, 1900.0)


def field(size, density, seed):
    n = max(1, int(round(density * size * size)))
    sim = simulate.simulate(shape=(size, size), n_emitters=n, background=BG_E,
                            amplitude_range=AMP, sigma=SIGMA, border=1.0,
                            seed=seed)
    return sim, sim.image * GAIN + OFFSET


def greedy_match(pred, truth, radius):
    """Globally-nearest-first greedy matching. Returns (pairs, dists).

    Sorting all candidate pairs by distance before the greedy loop matters:
    picking whichever pair appears first instead biases the error downward for
    whichever method happens to list its detections in a luckier order.
    """
    if len(pred) == 0 or len(truth) == 0:
        return [], np.empty(0)
    d = np.linalg.norm(np.asarray(pred)[:, None, :]
                       - np.asarray(truth)[None, :, :], axis=-1)
    cand = [(d[i, j], i, j) for i in range(d.shape[0]) for j in range(d.shape[1])
            if d[i, j] <= radius]
    cand.sort()
    used_p, used_t, pairs = set(), set(), []
    for dist, i, j in cand:
        if i in used_p or j in used_t:
            continue
        used_p.add(i); used_t.add(j)
        pairs.append((i, j, dist))
    return pairs, np.array([p[2] for p in pairs])


def pileup(pred, truth, radius, pile_sep):
    """(duplicates, close_pairs) -- see the module docstring."""
    pred = np.atleast_2d(np.asarray(pred, float))
    if len(pred) == 0:
        return 0, 0
    dup = 0
    if len(truth):
        d = np.linalg.norm(pred[:, None, :] - np.asarray(truth)[None, :, :],
                           axis=-1)
        nearest = np.argmin(d, axis=1)
        within = d[np.arange(len(pred)), nearest] <= radius
        for t in np.unique(nearest[within]):
            k = int(np.sum(nearest[within] == t))
            if k > 1:
                dup += k - 1
    pp = np.linalg.norm(pred[:, None, :] - pred[None, :, :], axis=-1)
    np.fill_diagonal(pp, np.inf)
    close = int(np.sum(np.triu(pp < pile_sep, 1)))
    return dup, close


def run_boxsolve(adu):
    r = boxsolve.detect_boxes(adu, sigma=SIGMA, offset=OFFSET, gain=GAIN,
                              n_outer=1, k_max=16, verbose=0)
    # se is (N,3) = (SE_A, SE_y, SE_x); the position CRLB is the RMS of the two
    crlb = (np.sqrt(0.5 * (r.se[:, 1] ** 2 + r.se[:, 2] ** 2))
            if r.se is not None and len(r.se) else np.full(len(r.positions), np.nan))
    return r.positions, crlb


def run_daophot(d_e, **kw):
    if sfwloc_py is None:
        return None
    out = sfwloc_py.daophot_fit(d_e, SIGMA, float(np.median(d_e)), **kw)
    pos = np.asarray(out[1], float).reshape(-1, 2)
    # returns (amps, pos, group_ids, group_sizes, fallback, sigma_amplitude,
    #          sigma_position, fitted_sigma); sigma_position is already the RMS
    crlb = np.asarray(out[6], float).ravel() if len(out) > 6 else np.full(len(pos), np.nan)
    return pos, crlb


def main(args):
    rows = {}
    for dens in args.densities:
        for s in range(args.seeds):
            sim, adu = field(args.size, dens, 2000 + s)
            d_e = (adu - OFFSET) / GAIN
            truth = sim.positions

            trials = [("boxsolve", run_boxsolve(adu))]
            for a in args.alphas:
                try:
                    got = run_daophot(d_e, alpha=a)
                except Exception as e:                       # noqa: BLE001
                    print(f"  daophot alpha={a} FAILED on dens={dens} seed={s}: {e}")
                    continue
                if got is not None:
                    trials.append((f"daophot a={a:g}", got))

            for name, (pos, crlb) in trials:
                pairs, dists = greedy_match(pos, truth, args.radius)
                dup, close = pileup(pos, truth, args.radius, args.pile_sep)
                mc = np.array([crlb[i] for i, _, _ in pairs]) if len(pairs) else np.empty(0)
                rows.setdefault(name, []).append(dict(
                    dens=dens, n_true=len(truth), n_est=len(pos),
                    matched=len(pairs),
                    precision=len(pairs) / max(len(pos), 1),
                    recall=len(pairs) / max(len(truth), 1),
                    med=np.median(dists) if len(dists) else np.nan,
                    p90=np.percentile(dists, 90) if len(dists) else np.nan,
                    rmse=np.sqrt(np.mean(dists ** 2)) if len(dists) else np.nan,
                    crlb=np.nanmedian(mc) if len(mc) else np.nan,
                    dup=dup, close=close))

    print(f"\nfield {args.size}x{args.size}, sigma={SIGMA}, match radius "
          f"{args.radius} px, pile separation {args.pile_sep} px, "
          f"{args.seeds} seeds/density\n")
    hdr = (f"{'method':13s} {'dens':>6} {'Ntrue':>6} {'Nest':>6} {'prec':>6} "
           f"{'recall':>7} {'med err':>8} {'p90 err':>8} {'RMSE':>7} "
           f"{'CRLB':>7} {'err/CRLB':>9} {'dup':>5} {'close':>6}")
    print(hdr)
    print("-" * len(hdr))
    for dens in args.densities:
        for name, rs in rows.items():
            g = [r for r in rs if r["dens"] == dens]
            if not g:
                continue
            f = lambda k: np.nanmean([r[k] for r in g])
            eff = f("rmse") / f("crlb") if f("crlb") > 0 else np.nan
            print(f"{name:13s} {dens:6.3f} {f('n_true'):6.1f} {f('n_est'):6.1f} "
                  f"{f('precision'):6.3f} {f('recall'):7.3f} {f('med'):8.4f} "
                  f"{f('p90'):8.4f} {f('rmse'):7.4f} {f('crlb'):7.4f} "
                  f"{eff:9.2f} {f('dup'):5.2f} {f('close'):6.2f}")
        print()

    print("dup   = detections sharing one true emitter (one source modelled twice)")
    print("close = detection pairs closer than the identifiability limit")
    print("err/CRLB = achieved RMSE over the method's own reported precision;")
    print("           1.0 means it extracts all the information in the data.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=39)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--densities", type=float, nargs="*",
                    default=[0.015, 0.034, 0.055])
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.01, 0.05])
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--pile-sep", type=float, default=SIGMA)
    main(ap.parse_args())
