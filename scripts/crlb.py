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


def collect(size, densities, seeds, radius, bg, impl="py"):
    recs = []
    for dens in densities:
        for seed in range(seeds):
            sim, raw = bench.field(size, dens, seed, bg)
            d_e = (raw - OFFSET) / GAIN
            truth, amps = sim.positions, sim.amplitudes
            nn = bench.nn_distance(truth)

            # oracle: no search, true N, true background
            # The oracle arm: refine from the TRUE positions at the true N
            # with the true background, so a deficit can be attributed to the
            # estimator or to the search instead of guessed at. Without it "the
            # pull spread is 1.4" is uninterpretable.
            p, a, se = backend.get(impl).refine(
                d_e, truth.copy(), amps.copy(), SIGMA, sim.background, 12,
                spotsolve.REFINE_SWEEPS)
            for j in range(len(truth)):
                if np.all(np.isfinite(se[j, 1:])):
                    recs.append(dict(
                        arm="oracle", dens=dens, nn=nn[j],
                        zy=(p[j, 0] - truth[j, 0]) / se[j, 1],
                        zx=(p[j, 1] - truth[j, 1]) / se[j, 2],
                        za=(a[j] - amps[j]) / se[j, 0],
                        d=float(np.linalg.norm(p[j] - truth[j])),
                        se=0.5 * (se[j, 1] + se[j, 2])))

            r = spotsolve.detect(raw, sigma=SIGMA, offset=OFFSET, gain=GAIN,
                              verbose=0, impl=impl)
            for i, j, dist in bench.greedy_match(r.positions, truth, radius):
                if r.se is None or not np.all(np.isfinite(r.se[i, 1:])):
                    continue
                recs.append(dict(
                    arm="pipeline", dens=dens, nn=nn[j],
                    zy=(r.positions[i, 0] - truth[j, 0]) / r.se[i, 1],
                    zx=(r.positions[i, 1] - truth[j, 1]) / r.se[i, 2],
                    za=(r.amplitudes[i] - amps[j]) / r.se[i, 0],
                    d=dist, se=0.5 * (r.se[i, 1] + r.se[i, 2])))
    return recs


def summarize(rs, label):
    if not rs:
        return f"{label:>16s}       -"
    z = np.concatenate([[r["zy"] for r in rs], [r["zx"] for r in rs]])
    z = z[np.isfinite(z)]
    za = np.array([r["za"] for r in rs])
    za = za[np.isfinite(za)]
    d = np.array([r["d"] for r in rs])
    se = np.array([r["se"] for r in rs])
    if len(z) == 0:
        return f"{label:>16s}       -"
    # Robust sd: a heavy tail is a separate failure and must not be allowed to
    # masquerade as inefficiency in the core of the distribution.
    rsd = 0.7413 * (np.percentile(z, 75) - np.percentile(z, 25))
    rsd_a = (0.7413 * (np.percentile(za, 75) - np.percentile(za, 25))
             if len(za) else np.nan)
    return (f"{label:>16s} {len(rs):6d} {np.mean(z):+7.3f} {np.std(z):7.2f} "
            f"{rsd:7.2f} {100 * np.mean(np.abs(z) > 3):6.1f}% "
            f"{rsd_a:7.2f} {np.median(d):8.4f} "
            f"{np.sqrt(np.mean(d ** 2)):8.4f} {np.nanmedian(se):8.4f}")


HDR = (f"{'arm / bin':>16s} {'n':>6} {'mean z':>7} {'sd z':>7} {'rsd z':>7} "
       f"{'|z|>3':>7} {'rsd zA':>7} {'med err':>8} {'RMSE':>8} {'med SE':>8}")


def main(args):
    recs = collect(args.size, args.densities, args.seeds, args.radius, args.bg,
                   args.impl)
    print(f"\n{args.size}x{args.size}, sigma {SIGMA}, background '{args.bg}', "
          f"{args.seeds} seeds/density, match radius {args.radius} px")
    print("pull z = (estimate - truth)/SE per axis; rsd 1.00 = at the CRLB, "
          "|z|>3 ideal 0.3%")
    print("rsd zA is the same statistic on the AMPLITUDE")
    for dens in args.densities:
        print(f"\n=== density {dens} ===")
        print(HDR)
        print("-" * len(HDR))
        for arm in ("oracle", "pipeline"):
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
    main(ap.parse_args())
