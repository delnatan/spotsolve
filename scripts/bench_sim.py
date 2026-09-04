"""Benchmark `spotsolve` against a psfkit confocal simulation with known z.

What this measures that `bench.py` cannot
-----------------------------------------
`bench.py` renders every emitter at the model's own sigma (or at a spread it
chooses). This dataset renders them through a *real* vectorial spinning-disk
confocal PSF at depths uniform in +/- 0.5 um, so each emitter's width and
brightness are set by physics:

    |z| um   0.00   0.10   0.20   0.30   0.40   0.50
    sigma    0.82   0.84   0.92   1.26   1.96   2.64      px
    flux     1.00   0.91   0.71   0.48   0.31   0.20      of nominal

That is a 3.2x width spread, the axis README section 12 calls the most
damaging in the benchmark -- and here it is not a knob, it is what defocus
does. The out-of-focus population is not noise to be explained away: it is
the real-world background a fixed-sigma detector has to cope with
gracefully. Missing it is fine. Tiling it into five detections is not.

Ground truth carries `z_um`, `photons_in_frame` (what the emitter actually
deposited, after the confocal axial response and any border clipping) and
`peak_photons`, so every row of every table below can be conditioned on
depth, on brightness, or on crowding rather than averaged over them.

Reading the tables
------------------
One row per TRUE emitter, as `crlb.py` does -- `match%` is the share of the
bin the pull columns rest on, and `d_nn` (distance from each truth to the
nearest estimate, over all of them) is the one column with the same
denominator everywhere.

Extra detections are split, because the two have different causes and
different fixes, and the split is made against each truth's OWN width:

    tile    within `--tile-r` x sigma(z) of some true emitter -- one object
            cut into pieces. A truth at |z| = 0.4 um is 1.96 px wide, so its
            tiles land 2-3 px out and a fixed matching radius calls them
            invented. That misreading is the difference between "the detector
            hallucinates" and "the model cannot represent defocus".
    ghost   everything else: no true emitter of any depth is near it.

    python scripts/bench_sim.py --densities sparse
    python scripts/bench_sim.py --methods legacy mix --frames 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import spotsolve

DATA = Path(__file__).resolve().parent.parent / "data" / "sim_out"

# Pixel-integrated-Gaussian sigma of the simulated confocal PSF at focus,
# fitted to the same pixel-binned stamp the simulator deposits (0.818 px =
# 69.5 nm at the 85 nm pixel). Not a guess: see `scripts/psf_sigma_scan.py`.
SIGMA = 0.818        # confocal; widefield is 0.840, see OPTICS

# `|z| <= 0.2 um` is where the free-sigma fit stays within 1.12x of the
# in-focus width, comfortably inside `core.FOCUS_BAND`. These are the
# emitters the fixed-sigma model can actually represent, and the only ones
# whose localization the pipeline is accountable for.
IN_FOCUS_Z = 0.2

# Fitted Gaussian width and deposited-flux fraction of the simulated PSF vs
# |z|, from `scripts/psf_sigma_scan.py --emit`. Cached so this benchmark does
# not need `psfkit` installed; regenerate if the simulation config changes.
#
# TWO TABLES, and the difference is not a detail. A pinhole makes a defocused
# emitter broader AND dimmer; widefield conserves its flux and only spreads it.
# At |z| = 0.5 um the confocal PSF is 3.2x the in-focus width carrying 0.20 of
# the flux, and the widefield one is 4.2x carrying 0.91. Scoring widefield data
# against the confocal table understates every defocused emitter's width, and
# `TILE_R * sigma(z)` then calls genuine tiles "invented".
OPTICS = {
    "confocal": dict(
        sigma0=0.818,
        z=np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                    0.45, 0.50]),
        sigma=np.array([0.818, 0.822, 0.836, 0.864, 0.919, 1.029, 1.256,
                        1.597, 1.960, 2.307, 2.639]),
        flux=np.array([1.0000, 0.9830, 0.9100, 0.8000, 0.7100, 0.5900,
                       0.4800, 0.3860, 0.3120, 0.2500, 0.2030]),
    ),
    # Truncated at 0.80 um, where the fitted width PEAKS at 6.12 px and then
    # FALLS (6.09, 5.68, 5.06, 4.61 out to 1.0 um). That fall is the 21 px
    # stamp, not the optics: past 0.8 um the PSF spreads beyond the stamp and a
    # Gaussian fitted to what is left is narrower. `np.interp` clamps beyond the
    # last point, which is the honest extrapolation -- a wider one would be
    # inventing width the simulator never deposited.
    "widefield": dict(
        sigma0=0.840,
        z=np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]),
        sigma=np.array([0.840, 0.846, 0.865, 0.906, 0.994, 1.207, 1.638,
                        2.127, 2.600, 3.062, 3.524, 3.994, 4.477, 4.968,
                        5.447, 5.864, 6.122]),
        flux=np.array([1.0000, 0.9994, 0.9975, 0.9942, 0.9894, 0.9828,
                       0.9742, 0.9632, 0.9494, 0.9325, 0.9119, 0.8873,
                       0.8584, 0.8251, 0.7872, 0.7452, 0.6995]),
    ),
}
ACTIVE = OPTICS["confocal"]     # `main` selects it from --optics

TILE_R = 3.0
# An extra detection within this many of a true emitter's OWN sigma(z) is a
# piece of it. 3.0 because a fixed-sigma model tiling a source of width S
# spreads its pieces over the source's support, not over the model's: measured
# on the sparse arm, 100% of extra detections land inside 3 sigma(z) of a real
# emitter and 64% of them outside 1 -- there are no invented detections on this
# data at all, only subdivided defocused ones.


def sigma_of_z(z):
    """Fitted Gaussian width (px) of an emitter at depth `z` um."""
    return np.interp(np.abs(np.asarray(z, dtype=float)),
                     ACTIVE["z"], ACTIVE["sigma"])



METHODS = {
    "fixed": lambda raw, a: spotsolve.detect(
        raw, sigma=a.sigma, offset=a.offset, gain=a.gain, verbose=0,
        impl=a.impl, threshold=a.threshold, prune=not a.no_prune,
        slack=None),
    # One uniform width prior over [slack_lo, slack_hi] and no classes: the
    # pipeline as it stood before the mixture. The baseline the `mix` arm is
    # attributable against, which is why its defaults are the OLD (0.95, 2.0).
    "slack": lambda raw, a: spotsolve.detect(
        raw, sigma=a.sigma, offset=a.offset, gain=a.gain, verbose=0,
        threshold=a.threshold, prune=not a.no_prune,
        slack=(a.slack_lo, a.slack_hi), band=None),
    # The two-class mixture: model space `--slack-*`, reporting band
    # `--band-*`, and the class boundary at the band's upper edge.
    "mix": lambda raw, a: spotsolve.detect(
        raw, sigma=a.sigma, offset=a.offset, gain=a.gain, verbose=0,
        threshold=a.threshold, prune=not a.no_prune,
        slack=(a.model_lo, a.model_hi), band=(a.band_lo, a.band_hi),
        width_gamma=a.gamma, prune_tau=a.prune_tau),
    # The single band with the PSF-width proximity veto: the pipeline exactly
    # as it stood, before either change. `slack` differs from it only by the
    # veto radius, which is how that one line is attributed on its own.
    "legacy": lambda raw, a: spotsolve.detect(
        raw, sigma=a.sigma, offset=a.offset, gain=a.gain, verbose=0,
        threshold=a.threshold, prune=not a.no_prune,
        slack=(a.slack_lo, a.slack_hi), band=None, veto_widths=False),
}

Z_BINS = [(0.0, 0.05, "|z|<.05"), (0.05, 0.15, ".05-.15"),
          (0.15, 0.25, ".15-.25"), (0.25, 0.35, ".25-.35"),
          (0.35, 1.0, ">.35")]
# Photons the emitter actually deposited, not its nominal in-focus flux.
F_BINS = [(0.0, 100.0, "<100"), (100.0, 300.0, "100-300"),
          (300.0, 1000.0, "300-1k"), (1000.0, np.inf, ">1k")]
NN_BINS = [(0.0, 1.0, "<1s"), (1.0, 2.0, "1-2s"), (2.0, 3.0, "2-3s"),
           (3.0, np.inf, ">3s")]


def read_truth(path):
    """The truth CSV as a dict of columns. Avoids a polars dependency here so
    the benchmark runs with the library's own numpy/scipy only."""
    rows = np.genfromtxt(path, delimiter=",", names=True)
    return {n: np.atleast_1d(rows[n]) for n in rows.dtype.names}


def greedy_match(est, truth, radius):
    """(est_idx, truth_idx, distance) by increasing distance, one-to-one.

    Greedy rather than Hungarian on purpose: a tile and its parent must not be
    reassigned to two different truths by a global optimum that minimizes
    total cost. The closest pair is the one that is actually the same object.
    """
    if len(est) == 0 or len(truth) == 0:
        return []
    d = np.linalg.norm(np.asarray(est)[:, None, :]
                       - np.asarray(truth)[None, :, :], axis=-1)
    out = []
    ei, ti = np.unravel_index(np.argsort(d, axis=None), d.shape)
    used_e, used_t = set(), set()
    for i, j in zip(ei, ti):
        if d[i, j] > radius:
            break
        if i in used_e or j in used_t:
            continue
        used_e.add(int(i))
        used_t.add(int(j))
        out.append((int(i), int(j), float(d[i, j])))
    return out


def nearest(a, b):
    """Distance from each row of `a` to the nearest row of `b`, or inf."""
    if len(a) == 0:
        return np.empty(0)
    if len(b) == 0:
        return np.full(len(a), np.inf)
    return np.min(np.linalg.norm(np.asarray(a)[:, None, :]
                                 - np.asarray(b)[None, :, :], axis=-1), axis=1)


def nn_distance(pos):
    """Each emitter's distance to its nearest neighbour in the same frame."""
    if len(pos) < 2:
        return np.full(len(pos), np.inf)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def run_frame(raw, truth_pos, truth_z, truth_flux, args, method):
    """Detect on one frame; return per-truth records and frame-level counts."""
    t0 = time.perf_counter()
    r = METHODS[method](raw, args)
    dt = time.perf_counter() - t0

    radius = args.radius * args.sigma
    pairs = greedy_match(r.positions, truth_pos, radius)
    hit = {j: (i, dist) for i, j, dist in pairs}

    nn = nn_distance(truth_pos)
    dnn = nearest(truth_pos, r.positions)

    recs = []
    for j in range(len(truth_pos)):
        i, dist = hit.get(j, (None, np.nan))
        ok = (i is not None and r.se is not None
              and np.all(np.isfinite(r.se[i, 1:])))
        recs.append(dict(
            z=abs(truth_z[j]), flux=truth_flux[j], nn=nn[j], dnn=dnn[j],
            matched=i is not None, se_ok=bool(ok), d=dist,
            zy=(r.positions[i, 0] - truth_pos[j, 0]) / r.se[i, 1] if ok else np.nan,
            zx=(r.positions[i, 1] - truth_pos[j, 1]) / r.se[i, 2] if ok else np.nan,
            a_ratio=(r.amplitudes[i] / truth_flux[j]
                     if i is not None and truth_flux[j] > 0 else np.nan),
            se=0.5 * (r.se[i, 1] + r.se[i, 2]) if ok else np.nan,
            s_fit=(r.sigma_ratio[i] if (i is not None
                                        and r.sigma_ratio is not None)
                   else np.nan),
        ))

    # Extra detections, split by cause against each truth's OWN width. An
    # extra detection 2.5 px from a source that is 1.96 px wide is a piece of
    # that source; the same distance from an in-focus one is invented.
    claimed = {i for i, _, _ in pairs}
    extra = np.array([i for i in range(len(r.positions)) if i not in claimed],
                     dtype=int)
    n_ghost = n_tile = 0
    tile_z = []
    if len(extra) and len(truth_pos):
        d = np.linalg.norm(r.positions[extra][:, None, :]
                           - truth_pos[None, :, :], axis=-1)
        scaled = d / (TILE_R * sigma_of_z(truth_z))[None, :]
        j = np.argmin(scaled, axis=1)
        is_tile = scaled[np.arange(len(extra)), j] <= 1.0
        n_tile = int(is_tile.sum())
        n_ghost = int(len(extra) - n_tile)
        tile_z = np.abs(truth_z[j][is_tile]).tolist()
    elif len(extra):
        n_ghost = len(extra)

    # `wide` counts objects the ROI solver judged unrepresentable at the PSF
    # width and set aside instead of tiling. Zero for the round loop, which
    # has no such move.
    n_wide = 0 if r.aggregates is None else len(r.aggregates)

    frame = dict(n_true=len(truth_pos), n_est=len(r.positions),
                 n_ghost=n_ghost, n_tile=n_tile, n_wide=n_wide,
                 tile_z=tile_z, seconds=dt, rounds=r.n_outer_passes,
                 bg=float(np.median(r.background)),
                 resid_rsd=float(0.7413 * np.subtract(
                     *np.percentile(r.residual
                                    / np.sqrt(np.maximum(r.model_image, 1e-6)),
                                    [75, 25]))))
    return recs, frame


def summarize(rs, label):
    if not rs:
        return f"{label:>12s}      -"
    n = len(rs)
    frac = np.mean([r["matched"] for r in rs])
    z = np.concatenate([[r["zy"] for r in rs], [r["zx"] for r in rs]])
    z = z[np.isfinite(z)]
    d = np.array([r["d"] for r in rs], float)
    d = d[np.isfinite(d)]
    dnn = np.array([r["dnn"] for r in rs], float)
    se = np.array([r["se"] for r in rs], float)
    ar = np.array([r["a_ratio"] for r in rs], float)
    ar = ar[np.isfinite(ar)]
    sf = np.array([r["s_fit"] for r in rs], float)
    sf = sf[np.isfinite(sf)]
    sfit = f"{np.median(sf):6.2f}" if len(sf) else "     -"
    if len(z) == 0:
        return (f"{label:>12s} {n:6d} {100 * frac:6.1f}%"
                + " " * 54 + f"{np.median(dnn):8.3f}")
    rsd = 0.7413 * (np.percentile(z, 75) - np.percentile(z, 25))
    return (f"{label:>12s} {n:6d} {100 * frac:6.1f}% "
            f"{np.mean(z):+6.2f} {rsd:6.2f} {100 * np.mean(np.abs(z) > 3):6.1f}% "
            f"{np.median(d):7.3f} {np.sqrt(np.mean(d ** 2)):7.3f} "
            f"{np.nanmedian(se):7.3f} {np.median(ar):6.2f} {sfit} "
            f"{np.median(dnn):8.3f}")


HDR = (f"{'bin':>12s} {'n':>6} {'match%':>7} {'mn z':>6} {'rsd z':>6} "
       f"{'|z|>3':>7} {'med err':>7} {'RMSE':>7} {'med SE':>7} {'A/A*':>6} "
       f"{'s/s0':>6} {'d_nn':>8}")


def table(recs, bins, key, title, scale=1.0):
    print(f"\n  {title}")
    print("  " + HDR)
    print("  " + "-" * len(HDR))
    for lo, hi, nm in bins:
        sel = [r for r in recs if lo * scale <= r[key] < hi * scale]
        print("  " + summarize(sel, nm))
    print("  " + summarize(recs, "all"))


def main(args):
    global ACTIVE
    ACTIVE = OPTICS[args.optics]
    if args.sigma is None:
        args.sigma = ACTIVE["sigma0"]
    meta = json.loads((args.data / "metadata.json").read_text())
    cam = meta["camera"]
    if args.gain is None:
        args.gain = cam["gain"]
    if args.offset is None:
        args.offset = cam["baseline"]

    print(f"\ndata     : {args.data}")
    print(f"model    : {args.optics}, sigma {args.sigma:.3f} px, "
          f"gain {args.gain:g}, offset {args.offset:g}, impl {args.impl}")
    print(f"matching : radius {args.radius:g} sigma = "
          f"{args.radius * args.sigma:.2f} px; in-focus = |z| <= "
          f"{args.z_focus:g} um; a tile is within {TILE_R:g} sigma(z) "
          f"of a truth")

    for label in args.densities:
        stack = np.load(args.data / f"{label}.npy").astype(float)
        truth = read_truth(args.data / f"{label}_truth.csv")
        nframes = (min(args.frames, stack.shape[0]) if args.frames
                   else stack.shape[0])

        print(f"\n{'=' * 78}\n=== {label}  "
              f"({meta['densities'][label]['density_per_um2']:g}/um^2, "
              f"{nframes} frames)\n{'=' * 78}")

        for method in args.methods:
            recs, frames = [], []
            for f in range(nframes):
                m = truth["frame"] == f
                pos = np.column_stack([truth["y_px"][m], truth["x_px"][m]])
                rr, fr = run_frame(stack[f], pos, truth["z_um"][m],
                                   truth["photons_in_frame"][m], args, method)
                recs += rr
                frames.append(fr)

            n_true = sum(f["n_true"] for f in frames)
            n_focus = sum(1 for r in recs if r["z"] <= args.z_focus)
            tz = [z for f in frames for z in f["tile_z"]]
            print(f"\n--- {method} ---")
            print(f"  truth {n_true / nframes:6.1f}/frame, of which "
                  f"{n_focus / nframes:5.1f} in focus   |   "
                  f"detected {np.mean([f['n_est'] for f in frames]):6.1f}"
                  f"  ghost {np.mean([f['n_ghost'] for f in frames]):5.2f}"
                  f"  tile {np.mean([f['n_tile'] for f in frames]):5.2f}"
                  + (f"  wide {np.mean([f['n_wide'] for f in frames]):5.2f}"
                     if any(f["n_wide"] for f in frames) else "")
                  + (f"   (tiled truths sit at |z| med "
                     f"{np.median(tz):.2f} um)" if tz else ""))
            print(f"  bg {np.mean([f['bg'] for f in frames]):5.2f} e- "
                  f"(true {cam['background']:g})   "
                  f"resid rsd {np.mean([f['resid_rsd'] for f in frames]):5.3f}"
                  f"   {1000 * np.mean([f['seconds'] for f in frames]):6.1f} "
                  f"ms/frame")

            table(recs, Z_BINS, "z",
                  "by |z| (um) -- how gracefully defocus is handled")
            focus = [r for r in recs if r["z"] <= args.z_focus]
            table(focus, F_BINS, "flux",
                  f"by deposited photons, IN-FOCUS ONLY "
                  f"(|z| <= {args.z_focus:g})")
            table(focus, NN_BINS, "nn",
                  "by nearest-neighbour distance, IN-FOCUS ONLY",
                  scale=args.sigma)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--densities", nargs="*",
                    default=["sparse", "moderate", "dense"])
    ap.add_argument("--frames", type=int, default=0, help="0 = all")
    ap.add_argument("--optics", choices=sorted(OPTICS), default="confocal",
                    help="which PSF the data was simulated through; picks the "
                         "sigma(z) table AND the default --sigma")
    ap.add_argument("--sigma", type=float, default=None,
                    help="in-focus PSF sigma, px (default: the optics')")
    ap.add_argument("--gain", type=float, default=None, help="from metadata")
    ap.add_argument("--offset", type=float, default=None)
    ap.add_argument("--radius", type=float, default=2.0, help="in sigma")
    ap.add_argument("--z-focus", type=float, default=IN_FOCUS_Z, dest="z_focus")
    ap.add_argument("--methods", nargs="*", default=["fixed", "slack"],
                    choices=sorted(METHODS))
    ap.add_argument("--model-lo", type=float, dest="model_lo",
                    default=spotsolve.core.SIGMA_SLACK[0])
    ap.add_argument("--model-hi", type=float, dest="model_hi",
                    default=spotsolve.core.SIGMA_SLACK[1])
    ap.add_argument("--prune-tau", type=float, dest="prune_tau",
                    default=spotsolve.PRUNE_TAU,
                    help="the precision/recall dial")
    ap.add_argument("--gamma", type=float,
                    default=spotsolve.prior.FOCUS_WIDTH_GAMMA,
                    help="width prior's Cauchy half-width, in sigma")
    ap.add_argument("--band-lo", type=float, dest="band_lo",
                    default=spotsolve.core.FOCUS_BAND[0])
    ap.add_argument("--band-hi", type=float, dest="band_hi",
                    default=spotsolve.core.FOCUS_BAND[1])
    # Pinned to the historical single-band values, NOT to `SIGMA_SLACK`: this
    # arm is a fixed baseline, and a baseline that moves when the default moves
    # cannot attribute anything.
    ap.add_argument("--slack-lo", type=float, dest="slack_lo", default=0.95)
    ap.add_argument("--slack-hi", type=float, dest="slack_hi", default=2.0)
    ap.add_argument("--threshold", type=float, default=None,
                    help="FIND's seed cut in sd of the LoG null; default "
                         "derives it from the frame and the PSF "
                         "(calibrate.seed_threshold). Pass "
                         f"{spotsolve.CAND_THRESHOLD:.3f} for the historical "
                         "constant")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--impl", choices=["py", "rs"], default="rs")
    main(ap.parse_args())
