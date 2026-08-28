"""One benchmark, both solvers, identical fields: count, isolation-resolved
recall, CRLB efficiency, false positives, runtime.

The columns are chosen so that no single number can hide a failure behind a
success. In particular recall is reported PER ISOLATION BIN, because a
frame-level recall is dominated by the easy majority and is insensitive to
exactly the regime under test.

    python bench.py --seeds 6
"""

import argparse
import time

import numpy as np
import scipy.ndimage as ndi

import boxsolve
import gsolve
import simulate

SIGMA, GAIN, OFFSET, BG_E = 1.2, 4.23, 100.0, 4.0
AMP = (900.0, 1900.0)

BINS = [(0.0, 1.0, "<1s"), (1.0, 2.0, "1-2s"), (2.0, 3.0, "2-3s"),
        (3.0, np.inf, ">3s")]


def background_surface(shape, kind, seed):
    """Ground-truth background, in photoelectrons.

    `flat` is what `simulate.simulate` produces on its own. The other two exist
    because a flat background cannot show whether modelling the background
    spatially helps -- on a flat field the right answer is a constant, and any
    surface estimator can only add variance. The real frames are not flat (both
    solvers report 110-113% of their own flux there, which is background being
    absorbed into amplitudes), so the structured cases are the ones that decide
    whether the map earns its place.
    """
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W] * 1.0
    if kind == "flat":
        return np.full(shape, BG_E)
    if kind == "gradient":
        # A 4x ramp across the frame, the shape a mis-set illumination or a
        # coverslip tilt actually makes.
        return BG_E * (0.5 + 1.5 * (yy / max(H - 1, 1) + xx / max(W - 1, 1)) / 2)
    if kind == "blobs":
        # Out-of-focus haze: smooth, non-monotone, with no orientation the
        # estimator could exploit.
        rng = np.random.default_rng(10_000 + seed)
        b = rng.normal(size=shape)
        b = ndi.gaussian_filter(b, max(H, W) / 8.0)
        b = (b - b.min()) / max(float(np.ptp(b)), 1e-9)
        return BG_E * (0.5 + 2.0 * b)
    raise ValueError(kind)


def field(size, density, seed, bg="flat"):
    """Emitters on a (possibly structured) background, Poisson-sampled."""
    n = max(1, int(round(density * size * size)))
    sim = simulate.simulate(shape=(size, size), n_emitters=n, background=0.0,
                            amplitude_range=AMP, sigma=SIGMA, border=1.0,
                            seed=seed)
    surf = background_surface((size, size), bg, seed)
    rng = np.random.default_rng(50_000 + seed)
    # `sim.clean` carries the emitters with a zero background, so the sum is
    # re-sampled as one Poisson draw rather than adding two separate ones.
    sim.image = rng.poisson(sim.clean + surf).astype(float)
    sim.background = surf
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
    cand = sorted((d[i, j], i, j) for i in range(d.shape[0])
                  for j in range(d.shape[1]) if d[i, j] <= radius)
    up, ut, pairs = set(), set(), []
    for dist, i, j in cand:
        if i in up or j in ut:
            continue
        up.add(i)
        ut.add(j)
        pairs.append((i, j, dist))
    return pairs


def evaluate(r, sim, radius, pile_sep):
    truth = sim.positions
    nn = nn_distance(truth)
    pairs = greedy_match(r.positions, truth, radius)
    matched_t = {j for _, j, _ in pairs}

    per_bin = {}
    for lo, hi, nm in BINS:
        sel = np.nonzero((nn >= lo * SIGMA) & (nn < hi * SIGMA))[0]
        if len(sel):
            per_bin[nm] = (int(sum(1 for j in sel if j in matched_t)), len(sel))

    zs, dists, damp = [], [], []
    for i, j, dist in pairs:
        dists.append(dist)
        # Relative amplitude error. This is the column the background map is
        # really aimed at: background the model fails to account for has
        # nowhere to go but the emitter amplitudes, and it goes there as a
        # systematic POSITIVE bias that no position statistic can see.
        damp.append((r.amplitudes[i] - sim.amplitudes[j]) / sim.amplitudes[j])
        if r.se is not None and np.all(np.isfinite(r.se[i, 1:])):
            zs.append((r.positions[i, 0] - truth[j, 0]) / r.se[i, 1])
            zs.append((r.positions[i, 1] - truth[j, 1]) / r.se[i, 2])

    pp = np.linalg.norm(r.positions[:, None, :] - r.positions[None, :, :],
                        axis=-1) if len(r.positions) else np.zeros((0, 0))
    if len(pp):
        np.fill_diagonal(pp, np.inf)
    close = int(np.sum(np.triu(pp < pile_sep, 1))) if len(pp) else 0

    return dict(n_true=len(truth), n_est=len(r.positions), matched=len(pairs),
                fp=len(r.positions) - len(pairs), per_bin=per_bin,
                z=np.array(zs), d=np.array(dists), damp=np.array(damp),
                close=close)


def aggregate(rs):
    z = np.concatenate([r["z"] for r in rs]) if rs else np.empty(0)
    z = z[np.isfinite(z)]
    d = np.concatenate([r["d"] for r in rs]) if rs else np.empty(0)
    da = np.concatenate([r["damp"] for r in rs]) if rs else np.empty(0)
    out = dict(
        damp=np.median(da) if len(da) else np.nan,
        n_true=np.mean([r["n_true"] for r in rs]),
        n_est=np.mean([r["n_est"] for r in rs]),
        fp=np.mean([r["fp"] for r in rs]),
        close=np.mean([r["close"] for r in rs]),
        recall=sum(r["matched"] for r in rs) / max(sum(r["n_true"] for r in rs), 1),
        med=np.median(d) if len(d) else np.nan,
        rmse=np.sqrt(np.mean(d ** 2)) if len(d) else np.nan,
        sdz=np.std(z) if len(z) else np.nan,
        rsdz=0.7413 * (np.percentile(z, 75) - np.percentile(z, 25)) if len(z) else np.nan,
        tail=np.mean(np.abs(z) > 3) if len(z) else np.nan,
    )
    for _, _, nm in BINS:
        got = sum(r["per_bin"].get(nm, (0, 0))[0] for r in rs)
        tot = sum(r["per_bin"].get(nm, (0, 0))[1] for r in rs)
        out[nm] = got / tot if tot else np.nan
    return out


def main(args):
    methods = {
        "boxsolve": lambda adu: boxsolve.detect_boxes(
            adu, sigma=SIGMA, offset=OFFSET, gain=GAIN, n_outer=1,
            k_max=16, verbose=0),
        "gsolve-flatbg": lambda adu: gsolve.detect(
            adu, sigma=SIGMA, offset=OFFSET, gain=GAIN, bg_kernel=None,
            verbose=0),
        "gsolve": lambda adu: gsolve.detect(
            adu, sigma=SIGMA, offset=OFFSET, gain=GAIN, verbose=0),
    }
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}

    hdr = (f"{'method':>15} {'Ntrue':>6} {'Nest':>6} {'recall':>7} {'FP':>5} "
           f"{'close':>6} "
           + " ".join(f"{nm:>6}" for _, _, nm in BINS)
           + f" {'med err':>8} {'sd z':>6} {'rsd z':>6} {'|z|>3':>6} "
             f"{'dA/A':>7} {'s/frame':>8}")

    for dens in args.densities:
        print(f"\n=== density {dens:.3f}  background={args.bg} "
              f"({args.seeds} seeds, {args.size}x{args.size}) ===")
        print(hdr)
        print("-" * len(hdr))
        for name, fn in methods.items():
            rs, t0 = [], time.perf_counter()
            for s in range(args.seeds):
                sim, adu = field(args.size, dens, 2000 + s, bg=args.bg)
                rs.append(evaluate(fn(adu), sim, args.radius, args.pile_sep))
            el = (time.perf_counter() - t0) / args.seeds
            a = aggregate(rs)
            print(f"{name:>15} {a['n_true']:6.1f} {a['n_est']:6.1f} "
                  f"{a['recall']:7.3f} {a['fp']:5.2f} {a['close']:6.2f} "
                  + " ".join(f"{a[nm]:6.3f}" for _, _, nm in BINS)
                  + f" {a['med']:8.4f} {a['sdz']:6.2f} {a['rsdz']:6.2f} "
                    f"{100 * a['tail']:5.1f}% {100 * a['damp']:+6.1f}% "
                    f"{el:8.2f}")

    print("\nrecall columns are per true-emitter isolation (nn distance, in sigma)")
    print("sd z / rsd z: pull spread, 1.00 = at the CRLB; |z|>3 ideal 0.3%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=39)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--densities", type=float, nargs="*",
                    default=[0.015, 0.034, 0.055])
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--pile-sep", type=float, default=SIGMA)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--bg", choices=["flat", "gradient", "blobs"],
                    default="flat",
                    help="ground-truth background shape (see background_surface)")
    main(ap.parse_args())
