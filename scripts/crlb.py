"""How far is `spotsolve` from the CRLB, and how much of the gap is unavoidable?

The pull statistic

    z = (estimate - truth) / reported SE,   per axis

is the acceptance test for localization. An estimator that is at the CRLB with
an honest Fisher matrix gives z with spread 1.00 and 0.3% of values beyond
|z| = 3. Anything above that is either a real deficit or a dishonest SE, and
the two arms below separate them:

    oracle    `refine` started at the TRUE positions and amplitudes, at the
              TRUE N, with the TRUE background. Nothing is searched, so this
              is the estimator's floor -- the best the reported SE could be.
    pipeline  `detect` output matched to truth. The gap to `oracle` is what
              the search costs.

BOTH ARMS EMIT ONE ROW PER TRUE EMITTER. Until 2026-09-02 the oracle did but
the pipeline emitted one per MATCHED detection, so the `<1s` rows compared
n = 114 unconditioned truths against n = 63 truths the search had already
resolved -- and that is why pipeline `<1s` med err read 0.286 against the
oracle's 0.329 while being 4.8x overconfident. The bias runs the wrong way for
anything that helps: resolving more close pairs admits the harder ones, so
`med err` and `rsd z` can degrade while the estimate strictly improves.

A pull needs an SE, so the pull columns are still computable only where there
IS a matched detection. Two columns make that conditioning impossible to miss
rather than removing it:

    match%  the share of the bin's true emitters the pull columns rest on.
            Compare rows only at comparable match%.
    d_nn    distance from each TRUE emitter to the nearest estimate, over ALL
            of them, matched or not. Same denominator in both arms, so this
            one column can be read across arms directly. A merged pair member
            scores about half the separation here; a missed emitter scores
            whatever else is nearby, which is the honest cost of missing it.

Reported per isolation bin, because a frame-level number is dominated by the
easy majority and is blind to the regime still in question.

    python crlb.py --seeds 12
"""

import argparse

import numpy as np

import spotsolve
from spotsolve import backend

import bench

SIGMA, GAIN, OFFSET, BG_E = bench.SIGMA, bench.GAIN, bench.OFFSET, bench.BG_E
BINS = [(0.0, 1.0, "<1s"), (1.0, 2.0, "1-2s"), (2.0, 3.0, "2-3s"),
        (3.0, np.inf, ">3s"), (0.0, np.inf, "all")]


# Four arms, and the pairs between them are the point: `oracle`->`oracle-w`
# prices the extra width parameter at the ESTIMATOR'S FLOOR, where no search is
# involved at all; `fixed`->`mix` prices it end to end; and `mix`->`oracle-w`
# is what the search costs under the shipped model. Comparing a free-width
# pipeline against a fixed-width oracle -- which is what this script did while
# it had only two arms -- charges the extra parameter to the search.
ARMS = ("oracle", "oracle-w", "fixed", "mix")


def _oracle(d_e, sim, truth, amps, impl, free):
    """`refine` from the TRUE positions, N and background. No search."""
    if not free:
        p, a, se = backend.get(impl).refine(
            d_e, truth.copy(), amps.copy(), SIGMA, sim.background, 12,
            spotsolve.REFINE_SWEEPS)
        return p, a, se
    p, a, se, _ = spotsolve.refine(
        d_e, truth.copy(), amps.copy(), SIGMA, sim.background, k_max=12,
        max_sweeps=spotsolve.REFINE_SWEEPS,
        sigmas=np.full(len(amps), SIGMA), slack=spotsolve.SIGMA_SLACK)
    return p, a, se


def collect(size, densities, seeds, radius, bg, impl="py", arms=ARMS):
    recs = []
    for dens in densities:
        for seed in range(seeds):
            sim, raw = bench.field(size, dens, seed, bg)
            d_e = (raw - OFFSET) / GAIN
            truth, amps = sim.positions, sim.amplitudes
            nn = bench.nn_distance(truth)

            for arm in arms:
                if arm.startswith("oracle"):
                    # No search: the estimator's floor, so a deficit can be
                    # attributed to the estimator or to the search instead of
                    # guessed at. Without it "the pull spread is 1.4" is
                    # uninterpretable.
                    p, a, se = _oracle(d_e, sim, truth, amps, impl,
                                       free=arm.endswith("-w"))
                    # One row per truth, in emitter order -- the oracle never
                    # loses one, so no matching is needed or wanted.
                    idx = {j: j for j in range(len(truth))}
                    dist = {j: float(np.linalg.norm(p[j] - truth[j]))
                            for j in range(len(truth))}
                else:
                    kw = (dict(slack=None) if arm == "fixed" else
                          dict(slack=spotsolve.SIGMA_SLACK,
                               band=spotsolve.FOCUS_BAND))
                    r = spotsolve.detect(raw, sigma=SIGMA, offset=OFFSET,
                                         gain=GAIN, verbose=0, impl=impl, **kw)
                    p, a, se = r.positions, r.amplitudes, r.se
                    # `hit` is keyed by the TRUTH index, which is what the
                    # greedy matcher assigns; truths it never reached fall
                    # through as unmatched rows rather than vanishing.
                    hits = bench.greedy_match(p, truth, radius)
                    idx = {j: i for i, j, _ in hits}
                    dist = {j: d for _, j, d in hits}

                dnn = nearest(truth, p)
                for j in range(len(truth)):
                    i = idx.get(j)
                    ok = (i is not None and se is not None
                          and np.all(np.isfinite(se[i, 1:])))
                    recs.append(dict(
                        arm=arm, dens=dens, nn=nn[j], matched=bool(ok),
                        dnn=dnn[j],
                        zy=(p[i, 0] - truth[j, 0]) / se[i, 1] if ok else np.nan,
                        zx=(p[i, 1] - truth[j, 1]) / se[i, 2] if ok else np.nan,
                        za=(a[i] - amps[j]) / se[i, 0] if ok else np.nan,
                        d=dist.get(j, np.nan) if ok else np.nan,
                        se=0.5 * (se[i, 1] + se[i, 2]) if ok else np.nan))
    return recs


def nearest(truth, est):
    """Distance from each true emitter to the nearest estimate, or inf.

    Deliberately not capped at the matching radius: a missed emitter's cost IS
    how far the nearest thing to it is, and capping would fold that back into
    the same conditioning this arm exists to avoid.
    """
    if len(truth) == 0:
        return np.empty(0)
    if len(est) == 0:
        return np.full(len(truth), np.inf)
    return np.min(np.linalg.norm(np.asarray(truth)[:, None, :]
                                 - np.asarray(est)[None, :, :], axis=-1),
                  axis=1)


def summarize(rs, label):
    if not rs:
        return f"{label:>16s}       -"
    z = np.concatenate([[r["zy"] for r in rs], [r["zx"] for r in rs]])
    z = z[np.isfinite(z)]
    za = np.array([r["za"] for r in rs])
    za = za[np.isfinite(za)]
    d = np.array([r["d"] for r in rs], float)
    d = d[np.isfinite(d)]
    se = np.array([r["se"] for r in rs], float)
    dnn = np.array([r["dnn"] for r in rs], float)
    frac = np.mean([r["matched"] for r in rs])
    if len(z) == 0:
        return (f"{label:>16s} {len(rs):6d} {100 * frac:6.1f}% "
                + " " * 48 + f"{np.median(dnn):8.4f}")
    # Robust sd: a heavy tail is a separate failure and must not be allowed to
    # masquerade as inefficiency in the core of the distribution.
    rsd = 0.7413 * (np.percentile(z, 75) - np.percentile(z, 25))
    rsd_a = (0.7413 * (np.percentile(za, 75) - np.percentile(za, 25))
             if len(za) else np.nan)
    return (f"{label:>16s} {len(rs):6d} {100 * frac:6.1f}% "
            f"{np.mean(z):+7.3f} {np.std(z):7.2f} "
            f"{rsd:7.2f} {100 * np.mean(np.abs(z) > 3):6.1f}% "
            f"{rsd_a:7.2f} {np.median(d):8.4f} "
            f"{np.sqrt(np.mean(d ** 2)):8.4f} {np.nanmedian(se):8.4f} "
            f"{np.median(dnn):8.4f}")


# `n` is the bin's TRUE EMITTER count in both arms. Every column between
# `match%` and `med SE` is computed on the matched subset only -- read them
# across arms only at comparable match%. `d_nn` is over all n.
HDR = (f"{'arm / bin':>16s} {'n':>6} {'match%':>7} {'mean z':>7} {'sd z':>7} "
       f"{'rsd z':>7} {'|z|>3':>7} {'rsd zA':>7} {'med err':>8} {'RMSE':>8} "
       f"{'med SE':>8} {'d_nn':>8}")


def main(args):
    recs = collect(args.size, args.densities, args.seeds, args.radius, args.bg,
                   args.impl, args.arms)
    print(f"\n{args.size}x{args.size}, sigma {SIGMA}, background '{args.bg}', "
          f"{args.seeds} seeds/density, match radius {args.radius} px")
    print("pull z = (estimate - truth)/SE per axis; rsd 1.00 = at the CRLB, "
          "|z|>3 ideal 0.3%")
    print("rsd zA is the same statistic on the AMPLITUDE")
    print("n is TRUE EMITTERS in both arms; the pull columns rest on the "
          "match% of them that")
    print("were matched, so compare those across arms only at comparable "
          "match%. d_nn -- the")
    print("distance to the nearest estimate, over all n -- has the same "
          "denominator everywhere.")
    for dens in args.densities:
        print(f"\n=== density {dens} ===")
        print(HDR)
        print("-" * len(HDR))
        for arm in args.arms:
            for lo, hi, nm in BINS:
                sel = [r for r in recs if r["arm"] == arm
                       and r["dens"] == dens
                       and lo * SIGMA <= r["nn"] < hi * SIGMA]
                print(summarize(sel, f"{arm} {nm}"))
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=39)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--densities", type=float, nargs="*",
                    default=[0.034, 0.055])
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--impl", choices=["py", "rs"], default="py",
                    help="which implementation runs the passes "
                         "(see backend.py); both arms use it")
    ap.add_argument("--bg", choices=["flat", "gradient", "blobs"],
                    default="flat")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS,
                    help="oracle/oracle-w are the fixed- and free-width "
                         "estimator floors; fixed/mix are the pipelines")
    main(ap.parse_args())
