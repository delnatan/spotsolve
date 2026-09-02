"""One benchmark for `spotsolve`: count, isolation-resolved recall, CRLB
efficiency, false positives, runtime.

The columns are chosen so that no single number can hide a failure behind a
success. In particular recall is reported PER ISOLATION BIN, because a
frame-level recall is dominated by the easy majority and is insensitive to
exactly the regime under test.

There is a second reason for the column layout: every arm rendered every
emitter at the model's own sigma until 2026-09-02, so no measurement in
`docs/baseline/bench.txt` contained a single mis-widthed source -- and width
mismatch is the one thing the real movies show. `--widths` adds it. See
`simulate.simulate`'s `sigma_spread` for the measurement that sets the arms.

    python bench.py --seeds 6
    python bench.py --widths 0.0 0.2 0.4 --densities 0.015
"""

import argparse
import time

import numpy as np
import scipy.ndimage as ndi

import spotsolve
from spotsolve import core
from spotsolve import backend, simulate

SIGMA, GAIN, OFFSET, BG_E = 1.2, 4.23, 100.0, 4.0

AMP_ARMS = {
    # name:   (lo, hi) total flux in photoelectrons -> peak SNR against
    #         background shot noise, printed in each section header.
    "bright": (900.0, 1900.0),   # peak SNR ~12 -- the ONLY arm until 2026-08-28
    "mid": (200.0, 400.0),       # ~5.3
    "faint": (80.0, 150.0),      # ~3.0
    "dim": (40.0, 70.0),         # ~1.8
}
AMP = AMP_ARMS["bright"]
# `bright` alone hid a recall cliff for years of measurements. FIND's
# `CAND_THRESHOLD` is applied to an UNNORMALIZED LoG response whose null sd is
# the kernel's L2 norm, so at sigma=1.2 the nominal 1.5 is a ~10 sigma cut on
# peak heights -- invisible at SNR 12, and worth 43 points of recall at SNR
# 1.1 (23.1% -> 65.8% as the cut goes 1.5 -> 0.15). Any claim that a stage of
# this pipeline "is not a bottleneck" is a claim about an SNR regime; run at
# least `bright` and `dim` before believing one.


def peak_snr(amp=AMP, sigma=SIGMA, bg=BG_E):
    """Peak SNR of the mean emitter in an arm, against background shot noise."""
    from spotsolve import psf
    pk = float(np.mean(amp)) * psf.peak_factor(sigma)
    return pk / np.sqrt(bg + pk)

BINS = [(0.0, 1.0, "<1s"), (1.0, 2.0, "1-2s"), (2.0, 3.0, "2-3s"),
        (3.0, np.inf, ">3s")]

# Ratio of a true emitter's own width to the model's. The edges are where the
# behaviour changes, not round numbers: tiling is one-sided and switches on at
# about 1.1 (measured, `simulate.simulate`), so 0.95-1.05 is the matched core,
# 1.05-1.25 the shoulder where it starts, and 1.25-1.60 where 85% of emitters
# collect a second detection. Emitters NARROWER than the model never tile,
# which is why everything below 0.95 is one bin.
WIDTH_BINS = [(0.0, 0.95, "<.95"), (0.95, 1.05, ".95-1.05"),
              (1.05, 1.25, "1.05-1.25"), (1.25, 1.60, "1.25-1.6"),
              (1.60, np.inf, ">1.6")]

# Radius, in sigma, within which a detection is attributed to a true emitter
# for the tiling count. Wide enough to catch the pieces a tiled object is cut
# into (they sit inside the object), narrow enough that at the densities run
# here ownership is unambiguous. It is NOT the matching radius: recall in every
# table below uses `greedy_match` at `--radius`, so the two rules never mix.
TILE_R = 2.0


def background_surface(shape, kind, seed):
    """Ground-truth background, in photoelectrons.

    `flat` is what `simulate.simulate` produces on its own. The other two exist
    because a flat background cannot show whether modelling the background
    spatially helps -- on a flat field the right answer is a constant, and any
    surface estimator can only add variance. The real frames are not flat (the
    solver reports 110-113% of its own flux there, which is background being
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


def field(size, density, seed, bg="flat", amp=AMP, sigma_spread=0.0):
    """Emitters on a (possibly structured) background, Poisson-sampled.

    `simulate` draws the widths after the positions and amplitudes, so raising
    `sigma_spread` at a fixed seed gives the SAME field with only the widths
    changed. The width arms are therefore paired with the spread-0 arm rather
    than being an independent sample of it.
    """
    n = max(1, int(round(density * size * size)))
    sim = simulate.simulate(shape=(size, size), n_emitters=n, background=0.0,
                            amplitude_range=amp, sigma=SIGMA, border=1.0,
                            sigma_spread=sigma_spread, seed=seed)
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
                close=close, per_width=per_width(r, sim, matched_t))


def per_width(r, sim, matched_t):
    """(recalled, tiled, n_det, n_emitters) per width bin, or None.

    Two quantities per bin, and they answer different questions:

      recalled  did the emitter get a detection at all -- the SAME
                `greedy_match` verdict the main table reports, only regrouped
                by the emitter's own width instead of by its isolation.
      tiled     did it get MORE than one. This is the failure width causes,
                and no column of the main table isolates it: a tiled object
                is recalled, so recall barely moves while `Nest` and `FP`
                absorb the damage without saying where it came from.
      tiles/det the mean count over emitters that got ANY detection. This is
                the column that decides whether an anti-tiling move works,
                because `dets/em` and `FP` cannot tell "the move merged the
                tiles" from "the seeder never found the object". A method
                that simply misses wide emitters scores well on both of those
                and unchanged on this one.

    Attribution is nearest-truth within `TILE_R`, which is a different rule
    from the matcher and deliberately so -- the extra pieces of a tiled object
    are exactly the detections the 1-1 matcher has no truth left to assign.
    """
    if sim.sigmas is None:
        return None
    truth, ratio = sim.positions, sim.sigmas / sim.sigma
    n_det = np.zeros(len(truth), int)
    if len(r.positions) and len(truth):
        d = np.linalg.norm(r.positions[:, None, :] - truth[None, :, :], axis=-1)
        owner = np.argmin(d, axis=1)
        for i, j in enumerate(owner):
            if d[i, j] <= TILE_R * SIGMA:
                n_det[j] += 1
    out = {}
    for lo, hi, nm in WIDTH_BINS:
        sel = np.nonzero((ratio >= lo) & (ratio < hi))[0]
        if len(sel):
            got = n_det[sel] >= 1
            out[nm] = (int(sum(1 for j in sel if j in matched_t)),
                       int(np.sum(n_det[sel] >= 2)), int(n_det[sel].sum()),
                       len(sel), int(got.sum()))
    return out


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
    out["per_width"] = None
    if rs and rs[0]["per_width"] is not None:
        w = {}
        for _, _, nm in WIDTH_BINS:
            c = np.array([r["per_width"].get(nm, (0, 0, 0, 0, 0))
                          for r in rs]).sum(0)
            if c[3]:
                w[nm] = (c[0] / c[3], c[1] / c[3], c[2] / c[3], int(c[3]),
                         c[2] / c[4] if c[4] else np.nan)
        out["per_width"] = w
    return out


def main(args):
    # The two implementations differ ONLY in the four passes: they share this
    # round loop, `find_candidates`, and `background_map`'s convolutions. So a
    # difference between the `spotsolve` and `spotsolve-rs` columns is a difference in
    # the passes and nowhere else.
    methods = {
        "spotsolve-flatbg": lambda adu: spotsolve.detect(
            adu, sigma=SIGMA, offset=OFFSET, gain=GAIN, bg_kernel=None,
            verbose=0),
        "spotsolve": lambda adu: spotsolve.detect(
            adu, sigma=SIGMA, offset=OFFSET, gain=GAIN, verbose=0),
        "spotsolve-rs": lambda adu: spotsolve.detect(
            adu, sigma=SIGMA, offset=OFFSET, gain=GAIN, verbose=0, impl="rs"),
        "spotsolve-roi": lambda adu: core.detect_local(
            adu, sigma=SIGMA, offset=OFFSET, gain=GAIN, verbose=0),
    }
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}
    elif "rs" not in backend.available():
        # The extension is optional -- `pip install spotsolve` does not build
        # it. Drop its column rather than crash the whole benchmark; asking for
        # it by name still raises, because then it was not a default.
        methods.pop("spotsolve-rs", None)
        print("note: spotsolve_rs not installed, running the Python backend "
              "only\n      (maturin develop --release -m "
              "rust/spotsolve-py/Cargo.toml)\n")

    hdr = (f"{'method':>15} {'Ntrue':>6} {'Nest':>6} {'recall':>7} {'FP':>5} "
           f"{'close':>6} "
           + " ".join(f"{nm:>6}" for _, _, nm in BINS)
           + f" {'med err':>8} {'sd z':>6} {'rsd z':>6} {'|z|>3':>6} "
             f"{'dA/A':>7} {'s/frame':>8}")

    whdr = (f"{'method':>15} {'width bin':>10} {'n':>5} {'recall':>7} "
            f"{'tiled':>7} {'dets/em':>8} {'tiles/det':>10}")

    for arm in args.amps:
        amp = AMP_ARMS[arm]
        for dens in args.densities:
            for spread in args.widths:
                print(f"\n=== {arm} (amp {amp[0]:.0f}-{amp[1]:.0f} e-, peak SNR "
                      f"{peak_snr(amp):.1f})  density {dens:.3f}  "
                      f"background={args.bg}  sigma_spread={spread:.2f} "
                      f"({args.seeds} seeds, {args.size}x{args.size}) ===")
                print(hdr)
                print("-" * len(hdr))
                widths = {}
                for name, fn in methods.items():
                    rs, t0 = [], time.perf_counter()
                    for s in range(args.seeds):
                        sim, adu = field(args.size, dens, 2000 + s, bg=args.bg,
                                         amp=amp, sigma_spread=spread)
                        rs.append(evaluate(fn(adu), sim, args.radius,
                                           args.pile_sep))
                    el = (time.perf_counter() - t0) / args.seeds
                    a = aggregate(rs)
                    widths[name] = a["per_width"]
                    print(f"{name:>15} {a['n_true']:6.1f} {a['n_est']:6.1f} "
                          f"{a['recall']:7.3f} {a['fp']:5.2f} {a['close']:6.2f} "
                          + " ".join(f"{a[nm]:6.3f}" for _, _, nm in BINS)
                          + f" {a['med']:8.4f} {a['sdz']:6.2f} {a['rsdz']:6.2f} "
                            f"{100 * a['tail']:5.1f}% {100 * a['damp']:+6.1f}% "
                            f"{el:8.2f}")
                if spread > 0:
                    print(f"\n{whdr}")
                    print("-" * len(whdr))
                    for name, w in widths.items():
                        for _, _, nm in WIDTH_BINS:
                            if w and nm in w:
                                rc, ti, dp, n, td = w[nm]
                                print(f"{name:>15} {nm:>10} {n:5d} {rc:7.3f} "
                                      f"{100 * ti:6.1f}% {dp:8.2f} "
                                      f"{td:10.2f}")

    print("\nrecall columns are per true-emitter isolation (nn distance, in sigma)")
    print("sd z / rsd z: pull spread, 1.00 = at the CRLB; |z|>3 ideal 0.3%")
    if any(s > 0 for s in args.widths):
        print("width bins are sigma_true/sigma_model; `tiled` is the share of "
              "true emitters")
        print("collecting 2 or more detections, `dets/em` the mean count "
              f"(within {TILE_R:.0f} sigma).")
        print("`tiles/det` is that mean over emitters that got ANY detection "
              "-- the tiling rate")
        print("with the seeder's recall divided out, and the only one of the "
              "three an")
        print("anti-tiling move can improve without simply missing the "
              "object.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=39)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--densities", type=float, nargs="*",
                    default=[0.015, 0.034, 0.055])
    ap.add_argument("--amps", nargs="*", default=["bright"],
                    choices=sorted(AMP_ARMS),
                    help="amplitude arms to run; `bright` alone hides the "
                         "low-SNR recall cliff (see AMP_ARMS)")
    ap.add_argument("--widths", type=float, nargs="*", default=[0.0],
                    help="per-emitter sigma spreads (lognormal sd in log "
                         "space) to run. 0.0 is every arm captured before "
                         "2026-09-02; 0.2 and 0.4 bracket the real bead data "
                         "(see simulate.simulate)")
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--pile-sep", type=float, default=SIGMA)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--bg", choices=["flat", "gradient", "blobs"],
                    default="flat",
                    help="ground-truth background shape (see background_surface)")
    main(ap.parse_args())
