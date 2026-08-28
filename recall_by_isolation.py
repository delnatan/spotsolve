"""Which emitters does the pipeline MISS, and why?

At density 0.055 the pipeline finds 68.8 of 84 true emitters. That 18% is a
bigger error than anything in the localization columns, so it decides the
design: if the misses are ISOLATED emitters the acceptance threshold is too
strict and the fix is a threshold; if they are CLOSE PAIRS collapsed into one
detection, the fix (if any) is resolution, which is a different and much harder
problem -- and partly not a problem at all, since below ~1 sigma the pair is
not identifiable from the data.

Each true emitter is classified:

  found      a detection within `radius`
  merged     no detection of its own, but a detection within `radius` of the
             MIDPOINT between it and its nearest neighbour, and that neighbour
             was also not separately found -- the classic two-into-one
  shadowed   not found, nearest neighbour WAS found, and the neighbour's
             detection is nearer to the neighbour than to this emitter
  missed     not found and not explained by either of the above

    python recall_by_isolation.py --seeds 6
"""

import argparse

import numpy as np

import boxsolve
import simulate

SIGMA, GAIN, OFFSET, BG_E = 1.2, 4.23, 100.0, 4.0
AMP = (900.0, 1900.0)


def field(size, density, seed):
    n = max(1, int(round(density * size * size)))
    sim = simulate.simulate(shape=(size, size), n_emitters=n, background=BG_E,
                            amplitude_range=AMP, sigma=SIGMA, border=1.0,
                            seed=seed)
    return sim, sim.image * GAIN + OFFSET


def classify(pred, truth, radius):
    """-> (labels, nn_dist, nn_index). labels in {found, merged, shadowed, missed}."""
    t = np.asarray(truth, float)
    n = len(t)
    dtt = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=-1)
    np.fill_diagonal(dtt, np.inf)
    nn_i = np.argmin(dtt, axis=1)
    nn_d = dtt[np.arange(n), nn_i]

    labels = np.array(["missed"] * n, dtype=object)
    if len(pred) == 0:
        return labels, nn_d, nn_i

    p = np.asarray(pred, float)
    dpt = np.linalg.norm(p[:, None, :] - t[None, :, :], axis=-1)

    # Greedy nearest-first assignment, same rule the other scripts use.
    cand = sorted((dpt[i, j], i, j) for i in range(len(p)) for j in range(n)
                  if dpt[i, j] <= radius)
    up, ut = set(), set()
    for _, i, j in cand:
        if i in up or j in ut:
            continue
        up.add(i)
        ut.add(j)
        labels[j] = "found"

    for j in range(n):
        if labels[j] != "missed":
            continue
        k = nn_i[j]
        mid = 0.5 * (t[j] + t[k])
        if np.min(np.linalg.norm(p - mid, axis=1)) <= radius and labels[k] != "found":
            labels[j] = "merged"
        elif labels[k] == "found":
            labels[j] = "shadowed"
    return labels, nn_d, nn_i


def main(args):
    edges = [(0.0, 1.0 * SIGMA, "nn<1.0s"), (1.0 * SIGMA, 2.0 * SIGMA, "1-2s"),
             (2.0 * SIGMA, 3.0 * SIGMA, "2-3s"), (3.0 * SIGMA, 4.0 * SIGMA, "3-4s"),
             (4.0 * SIGMA, np.inf, "nn>4s")]
    for dens in args.densities:
        rows = []
        for s in range(args.seeds):
            sim, adu = field(args.size, dens, 2000 + s)
            r = boxsolve.detect_boxes(adu, sigma=SIGMA, offset=OFFSET, gain=GAIN,
                                      n_outer=1, k_max=16, verbose=0)
            lab, nn_d, _ = classify(r.positions, sim.positions, args.radius)
            for j in range(len(lab)):
                rows.append(dict(lab=lab[j], nn=nn_d[j],
                                 amp=sim.amplitudes[j], n_est=len(r.positions)))
        n_true = len(rows) / args.seeds
        n_est = np.mean([r["n_est"] for r in rows])
        print(f"\ndensity {dens:.3f}   N_true {n_true:.1f}   N_est {n_est:.1f}   "
              f"({args.seeds} seeds)")
        hdr = (f"{'isolation':>10} {'n true':>7} {'found':>7} {'merged':>7} "
               f"{'shadowed':>9} {'missed':>7} {'recall':>8} {'share of':>9}")
        print(hdr)
        print(f"{'':>10} {'':>7} {'':>7} {'':>7} {'':>9} {'':>7} {'':>8} "
              f"{'all loss':>9}")
        print("-" * len(hdr))
        total_lost = sum(1 for r in rows if r["lab"] != "found")
        for lo, hi, nm in edges:
            g = [r for r in rows if lo <= r["nn"] < hi]
            if not g:
                continue
            c = {k: sum(1 for r in g if r["lab"] == k)
                 for k in ("found", "merged", "shadowed", "missed")}
            lost = len(g) - c["found"]
            print(f"{nm:>10} {len(g):7d} {c['found']:7d} {c['merged']:7d} "
                  f"{c['shadowed']:9d} {c['missed']:7d} "
                  f"{c['found'] / len(g):8.3f} "
                  f"{(lost / total_lost if total_lost else 0):8.1%}")

        # Is what is left over faint, or just crowded?
        lost = [r for r in rows if r["lab"] != "found"]
        if lost:
            print(f"  lost emitters: median amplitude "
                  f"{np.median([r['amp'] for r in lost]):.0f} vs "
                  f"{np.median([r['amp'] for r in rows]):.0f} overall; "
                  f"median nn {np.median([r['nn'] for r in lost]):.2f} px "
                  f"({np.median([r['nn'] for r in lost]) / SIGMA:.2f} sigma)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=39)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--densities", type=float, nargs="*", default=[0.034, 0.055])
    ap.add_argument("--radius", type=float, default=1.5)
    main(ap.parse_args())
