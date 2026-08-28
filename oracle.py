"""Is the close-pair loss in the ESTIMATOR or in the SEARCH?

`efficiency.py` shows the whole CRLB deficit lives in pairs closer than ~2
sigma. That is compatible with two very different diagnoses, which call for
opposite fixes:

  A. the ESTIMATOR cannot do better -- the joint fit of a close pair is
     genuinely hard, the likelihood is banana-shaped, and the reported SE from
     the Fisher matrix understates the real spread. Fix: a better estimator
     (or an honest SE).

  B. the SEARCH hands the estimator the wrong configuration -- wrong N, or a
     starting point in the basin of the wrong local optimum -- and the
     estimator then does exactly what it was asked, precisely. Fix: a better
     proposal/seed, which is a detection problem.

The separation is an ORACLE: run the same `boxsolve.refine` the pipeline uses,
but hand it the TRUE count and the TRUE parameters as its starting point. Any
error that survives is the estimator's floor (A). Any error the pipeline has
ON TOP of that belongs to the search (B).

Three arms, all scored with the same pull statistic:

  oracle-init   refine() from truth, at true N        -- the estimator's floor
  oracle-N      refine() from LoG-ish perturbed truth -- basin sensitivity
  pipeline      the actual detect_boxes output        -- what we ship

`oracle-N` matters because `oracle-init` starts at the answer and a local
optimizer will simply stay there; perturbing the start by a realistic seed
error asks whether the basin is even reachable.

    python oracle.py --seeds 6
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


def nn_distance(t):
    t = np.asarray(t, float)
    if len(t) < 2:
        return np.full(len(t), np.inf)
    d = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


def greedy_match(pred, truth, radius):
    if len(pred) == 0 or len(truth) == 0:
        return []
    d = np.linalg.norm(np.asarray(pred)[:, None, :]
                       - np.asarray(truth)[None, :, :], axis=-1)
    cand = [(d[i, j], i, j) for i in range(d.shape[0]) for j in range(d.shape[1])
            if d[i, j] <= radius]
    cand.sort()
    up, ut, pairs = set(), set(), []
    for dist, i, j in cand:
        if i in up or j in ut:
            continue
        up.add(i)
        ut.add(j)
        pairs.append((i, j, dist))
    return pairs


def score(pos, se, truth, nn, radius, recs, arm, dens, identity=False):
    """Accumulate pull records. `identity=True` skips matching: row i of `pos`
    IS true emitter i, which is what the oracle arms guarantee and what keeps a
    catastrophic oracle failure from being quietly dropped by the matcher."""
    if identity:
        pairs = [(i, i, float(np.linalg.norm(pos[i] - truth[i])))
                 for i in range(len(truth))]
    else:
        pairs = greedy_match(pos, truth, radius)
    for i, j, dist in pairs:
        sy, sx = se[i, 1], se[i, 2]
        recs.append(dict(arm=arm, dens=dens, dist=dist, nn=nn[j],
                         zy=(pos[i, 0] - truth[j, 0]) / sy,
                         zx=(pos[i, 1] - truth[j, 1]) / sx,
                         se=0.5 * (sy + sx)))


def summarize(recs, label):
    z = np.concatenate([[r["zy"] for r in recs], [r["zx"] for r in recs]])
    z = z[np.isfinite(z)]
    d = np.array([r["dist"] for r in recs])
    se = np.array([r["se"] for r in recs])
    if len(z) == 0:
        return f"{label:>22s}  (empty)"
    rsd = 0.7413 * (np.percentile(z, 75) - np.percentile(z, 25))
    return (f"{label:>22s} {len(recs):6d} {np.mean(z):+7.3f} {np.std(z):7.2f} "
            f"{rsd:7.2f} {100 * np.mean(np.abs(z) > 3):6.1f}% "
            f"{np.median(d):8.4f} {np.sqrt(np.mean(d ** 2)):8.4f} "
            f"{np.nanmedian(se):8.4f}")


HDR = (f"{'arm / group':>22s} {'n':>6} {'mean z':>7} {'sd z':>7} {'rsd z':>7} "
       f"{'|z|>3':>7} {'med err':>8} {'RMSE':>8} {'med SE':>8}")


def main(args):
    rng = np.random.default_rng(0)
    recs = []
    counts = {}
    for dens in args.densities:
        for s in range(args.seeds):
            sim, adu = field(args.size, dens, 2000 + s)
            d_e = (adu - OFFSET) / GAIN
            truth, t_amp = sim.positions, sim.amplitudes
            nn = nn_distance(truth)

            # --- arm 1: refine from exact truth, exact N -----------------
            p, a, _, se = boxsolve.refine(d_e, truth.copy(), t_amp.copy(),
                                          SIGMA, BG_E)
            score(p, se, truth, nn, args.radius, recs, "oracle-init", dens,
                  identity=True)

            # --- arm 2: exact N, start perturbed like a real seed --------
            # 0.4 px is the scale of a LoG peak's offset from the true centre
            # (integer pixel argmax on a noisy residual); amplitudes start
            # from a residual read-off, so 20% is generous rather than harsh.
            p0 = truth + rng.normal(0, 0.4, size=truth.shape)
            a0 = t_amp * rng.uniform(0.8, 1.2, size=t_amp.shape)
            p, a, _, se = boxsolve.refine(d_e, p0, a0, SIGMA, BG_E)
            score(p, se, truth, nn, args.radius, recs, "oracle-N", dens,
                  identity=True)

            # --- arm 3: the shipping pipeline ---------------------------
            r = boxsolve.detect_boxes(adu, sigma=SIGMA, offset=OFFSET,
                                      gain=GAIN, n_outer=1, k_max=16, verbose=0)
            score(r.positions, r.se, truth, nn, args.radius, recs,
                  "pipeline", dens)
            counts.setdefault(dens, []).append((len(truth), len(r.positions)))

    print(f"\nfield {args.size}x{args.size}, {args.seeds} seeds/density, "
          f"sigma={SIGMA}\n")
    arms = ["oracle-init", "oracle-N", "pipeline"]

    for dens in args.densities:
        nt, ne = np.mean([c[0] for c in counts[dens]]), np.mean([c[1] for c in counts[dens]])
        print(f"density {dens:.3f}   N_true {nt:.1f}   N_pipeline {ne:.1f}")
        print(HDR)
        print("-" * len(HDR))
        for arm in arms:
            g = [r for r in recs if r["arm"] == arm and r["dens"] == dens]
            if g:
                print(summarize(g, arm))
        print()

    print("=== split by isolation, pooled over density ===")
    edges = [0.0, 2 * SIGMA, 4 * SIGMA, np.inf]
    names = [f"nn<{2 * SIGMA:.1f}", f"nn {2 * SIGMA:.1f}-{4 * SIGMA:.1f}",
             f"nn>{4 * SIGMA:.1f}"]
    print(HDR)
    print("-" * len(HDR))
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        for arm in arms:
            g = [r for r in recs if r["arm"] == arm and lo <= r["nn"] < hi]
            if g:
                print(summarize(g, f"{arm} {nm}"))
        print()

    print("Ideal: mean z = 0, sd z = 1, |z|>3 = 0.3%, RMSE/med SE = 1.414")
    print("oracle-init  = the estimator's floor (started AT the answer)")
    print("oracle-N     = same N, realistic start -- gap to oracle-init is basin loss")
    print("pipeline     = gap to oracle-N is the search's count/seed error")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=39)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--densities", type=float, nargs="*",
                    default=[0.015, 0.034, 0.055])
    ap.add_argument("--radius", type=float, default=1.5)
    main(ap.parse_args())
