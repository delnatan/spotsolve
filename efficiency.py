"""How far is boxsolve from the CRLB, honestly?

`compare_precision.py` reports `err/CRLB = RMSE(radial) / median(SE_axis)`.
Those are not the same quantity. If a 2D estimator is exactly efficient, with
per-axis errors N(0, s^2), the radial distance d = sqrt(dy^2 + dx^2) is Rayleigh
with scale s, so

    RMSE(d) = sqrt(2) * s = 1.414 s        median(d) = sqrt(2 ln 2) * s = 1.177 s

An ideal estimator therefore scores 1.414 on that column, not 1.0. Anything
built on top of that number is off by a factor of sqrt(2) before it starts.

What this script measures instead is the PULL,

    z_y = (y_hat - y_true) / SE_y          z_x = (x_hat - x_true) / SE_x

which is N(0,1) for an efficient, unbiased estimator regardless of how bright
the emitter is or how crowded its neighbourhood. That makes emitters directly
comparable and splits the loss into three separable failures:

  * mean(z) != 0        -- BIAS (pull toward/away from a neighbour, or a
                           background error absorbed into position);
  * sd(z) > 1           -- INEFFICIENCY, the estimator is not extracting the
                           information the data contains;
  * excess |z| > 3      -- TAIL, mis-assignment, which no amount of local
                           polishing fixes and which is what dominates RMSE.

Everything is reported both over all matches and split by isolation (distance
to the nearest OTHER true emitter), because "at high density" is precisely the
claim under test and a frame-level average hides it.

    python efficiency.py --seeds 8
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


def greedy_match(pred, truth, radius):
    """Globally-nearest-first greedy matching -> list of (i_pred, j_truth, d)."""
    if len(pred) == 0 or len(truth) == 0:
        return []
    d = np.linalg.norm(np.asarray(pred)[:, None, :]
                       - np.asarray(truth)[None, :, :], axis=-1)
    cand = [(d[i, j], i, j) for i in range(d.shape[0]) for j in range(d.shape[1])
            if d[i, j] <= radius]
    cand.sort()
    used_p, used_t, pairs = set(), set(), []
    for dist, i, j in cand:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        pairs.append((i, j, dist))
    return pairs


def nn_distance(truth):
    """Distance from each true emitter to its nearest other true emitter."""
    t = np.asarray(truth, float)
    if len(t) < 2:
        return np.full(len(t), np.inf)
    d = np.linalg.norm(t[:, None, :] - t[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.min(d, axis=1)


def collect(size, densities, seeds, radius):
    """One row per matched detection, with its pulls and its isolation."""
    recs = []
    for dens in densities:
        for s in range(seeds):
            sim, adu = field(size, dens, 2000 + s)
            r = boxsolve.detect_boxes(adu, sigma=SIGMA, offset=OFFSET, gain=GAIN,
                                      n_outer=1, k_max=16, verbose=0)
            truth, t_amp = sim.positions, sim.amplitudes
            nn = nn_distance(truth)
            pairs = greedy_match(r.positions, truth, radius)
            se = r.se
            for i, j, dist in pairs:
                dy = r.positions[i, 0] - truth[j, 0]
                dx = r.positions[i, 1] - truth[j, 1]
                sy = se[i, 1] if se is not None and np.isfinite(se[i, 1]) else np.nan
                sx = se[i, 2] if se is not None and np.isfinite(se[i, 2]) else np.nan
                recs.append(dict(
                    dens=dens, seed=s, dy=dy, dx=dx, dist=dist,
                    se_y=sy, se_x=sx, se_a=se[i, 0] if se is not None else np.nan,
                    zy=dy / sy, zx=dx / sx,
                    nn=nn[j], amp_true=t_amp[j], amp_est=r.amplitudes[i],
                    n_true=len(truth), n_est=len(r.positions),
                    n_matched=len(pairs)))
    return recs


def summarize(recs, label):
    z = np.concatenate([[r["zy"] for r in recs], [r["zx"] for r in recs]])
    z = z[np.isfinite(z)]
    d = np.array([r["dist"] for r in recs])
    se = np.array([0.5 * (r["se_y"] + r["se_x"]) for r in recs])
    if len(z) == 0:
        return f"{label:>14s}  (no finite pulls)"
    tail = float(np.mean(np.abs(z) > 3.0))
    # Robust sd: the tail is a separate failure and should not be allowed to
    # masquerade as inefficiency in the core of the distribution.
    rsd = 0.7413 * (np.percentile(z, 75) - np.percentile(z, 25))
    return (f"{label:>14s} {len(recs):6d} {np.mean(z):+7.3f} {np.std(z):7.2f} "
            f"{rsd:7.2f} {100 * tail:6.1f}% {np.median(d):8.4f} "
            f"{np.sqrt(np.mean(d ** 2)):8.4f} {np.nanmedian(se):8.4f}")


HDR = (f"{'group':>14s} {'n':>6} {'mean z':>7} {'sd z':>7} {'rsd z':>7} "
       f"{'|z|>3':>7} {'med err':>8} {'RMSE':>8} {'med SE':>8}")


def main(args):
    recs = collect(args.size, args.densities, args.seeds, args.radius)

    print(f"\nfield {args.size}x{args.size}, sigma={SIGMA}, match radius "
          f"{args.radius} px, {args.seeds} seeds/density")
    print("pull z = (estimate - truth) / reported SE, per axis; "
          "N(0,1) iff efficient and unbiased\n")

    print(HDR)
    print("-" * len(HDR))
    for dens in args.densities:
        g = [r for r in recs if r["dens"] == dens]
        print(summarize(g, f"dens {dens:.3f}"))
    print()

    # Isolation split. 2 sigma is roughly where two emitters stop being
    # separately identifiable in the Fisher sense, so the bins bracket it.
    edges = [0.0, 2 * SIGMA, 4 * SIGMA, np.inf]
    names = [f"nn<{2 * SIGMA:.1f}", f"nn {2 * SIGMA:.1f}-{4 * SIGMA:.1f}",
             f"nn>{4 * SIGMA:.1f}"]
    print(HDR)
    print("-" * len(HDR))
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        g = [r for r in recs if lo <= r["nn"] < hi]
        if g:
            print(summarize(g, nm))
    print()

    # Amplitude split: a brightness-dependent pull is a different defect from a
    # crowding-dependent one (it points at the background, not the neighbours).
    a = np.array([r["amp_true"] for r in recs])
    qs = np.percentile(a, [33, 67])
    print(HDR)
    print("-" * len(HDR))
    for lo, hi, nm in [(0, qs[0], "A low"), (qs[0], qs[1], "A mid"),
                       (qs[1], np.inf, "A high")]:
        g = [r for r in recs if lo <= r["amp_true"] < hi]
        if g:
            print(summarize(g, nm))

    print("\nIdeal columns: mean z = 0, sd z = 1, |z|>3 = 0.3%, "
          "RMSE/med SE = 1.414, med err/med SE = 1.177")

    # Amplitude pull too: position and flux share the same Fisher block, so a
    # flux bias and a position bias usually have one cause.
    za = np.array([(r["amp_est"] - r["amp_true"]) / r["se_a"] for r in recs])
    za = za[np.isfinite(za)]
    if len(za):
        print(f"\namplitude pull: mean {np.mean(za):+.3f}  sd {np.std(za):.2f}  "
              f"median relative bias "
              f"{100 * np.median([(r['amp_est'] - r['amp_true']) / r['amp_true'] for r in recs]):+.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=39)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--densities", type=float, nargs="*",
                    default=[0.015, 0.034, 0.055])
    ap.add_argument("--radius", type=float, default=1.5)
    main(ap.parse_args())
