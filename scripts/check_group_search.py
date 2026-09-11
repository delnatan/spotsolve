"""Direct scientific controls for the native local group search.

    python scripts/check_group_search.py
    python scripts/check_group_search.py --case close_pair --trials 40

Run after `maturin develop --release -m rust/spotsolve-py/Cargo.toml`.

These exercise `DenseGroupEngine.search_group` directly, one group transaction
at a time, against known truth. There is no Python implementation of this
search to compare against and none is wanted: what is asserted is recovery,
localization, over-splitting and how often the comparison could not be settled
-- not agreement with anything.

The case list is `docs/RUST_GROUP_SEARCH_PLAN.md` section 6's control table.
Each case fixes its own frame, truth and ENTRY configuration, because half of
what is being measured is whether the search recovers from a bad start in
either direction. Results are reported per case; an average over cases would
hide exactly the cases the gate is about.

The `unresolved` column is the release gate: it is the fraction of transactions
that ended with a comparison the conservative validity policy would not settle.
"""

import argparse
import json
from time import perf_counter

import numpy as np

from spotsolve import backend, calibrate, metrics, prior

SIGMA = 1.2
SLACK = (0.70, 2.2)
BAND = (0.80, 2.0)
MATCH_RADIUS = 1.2


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def frame(shape, truth, background, seed, slope=0.0):
    """A Poisson frame plus the background surface a caller would hand in.

    `truth` is a list of `(y, x, flux, sigma)`. `slope` tilts the background
    across the frame, which is the control for whether the group's free `b`
    and the known additive shape term really do split the surface between them.
    """
    h, w = shape
    yy, _ = np.mgrid[0:h, 0:w]
    bmap = background + slope * (yy - h / 2.0)
    bmap = np.maximum(bmap, 1e-3)
    pos = np.array([[t[0], t[1]] for t in truth], float).reshape(-1, 2)
    amp = np.array([t[2] for t in truth], float)
    sig = np.array([t[3] for t in truth], float)
    mean = calibrate.render_model(pos, amp, sig, shape, 0.0) + bmap
    image = np.random.default_rng(seed).poisson(mean).astype(float)
    return image, np.ascontiguousarray(bmap), pos, amp, sig


def width_prior(lam_focus=0.02, lam_wide=0.002):
    return prior.FocusMixtureWidth(lam_focus, lam_wide, SLACK[0] * SIGMA,
                                   BAND[1] * SIGMA, SLACK[1] * SIGMA, SIGMA)


MAX_REBUILDS = 6
# Transactions a control will run before giving up on a group that keeps asking
# for a rebuild. Each rebuild grows the region -- a bigger position box, or one
# more free neighbour -- so the sequence terminates at the frame; this is a
# backstop against a caller bug, not a convergence criterion.


def run_group(image, bmap, entry, focus, a_s=1400.0, k_max=12, seeds=None,
              **settings):
    """Run transactions until the group stops asking for a rebuilt context.

    `entry` is a list of `(y, x, flux, sigma)`.

    A single call is NOT the operation's contract. `context_rebuild_required`
    means the committed configuration no longer fits the region it was compared
    in -- a source at its position bound, or light inside the region that no
    hypothesis there could place a source on. Answering it is the caller's half of the transaction,
    and a control that ignored it would report a one-shot result for exactly
    the groups that said one shot was not enough.

    What comes back from a transaction is the group's FREE set, not the frame.
    Emitters that were frozen are unchanged and still exist, so they are
    carried forward here by id.
    """
    be = backend.get("rs")
    eng = be.group_engine(image, bmap, SIGMA, SLACK, k_max, width_prior(), a_s,
                          next_id=len(entry))
    pos = np.ascontiguousarray(
        np.array([[e[0], e[1]] for e in entry], float).reshape(-1, 2))
    amp = np.ascontiguousarray(np.array([e[2] for e in entry], float))
    sig = np.ascontiguousarray(np.array([e[3] for e in entry], float))
    ids = np.arange(len(entry), dtype=np.uint32)
    seeds = (np.ascontiguousarray(np.asarray(seeds, float).reshape(-1, 2))
             if seeds is not None else None)

    t0 = perf_counter()
    rounds, total_fits = 0, 0
    for rounds in range(1, MAX_REBUILDS + 1):
        out = eng.search_group(pos, amp, sig, ids, tuple(focus), seeds,
                               **settings)
        total_fits += out["diagnostics"]["n_fits"]
        # Merge the group's committed free set back over the frame's emitters.
        touched = set(int(i) for i in out["ids"]) | set(int(i) for i in out["removed"])
        keep = [i for i, v in enumerate(ids) if int(v) not in touched]
        pos = np.ascontiguousarray(np.vstack([pos[keep], out["positions"]])
                                   if len(keep) else out["positions"])
        amp = np.ascontiguousarray(np.concatenate([amp[keep], out["amplitudes"]]))
        sig = np.ascontiguousarray(np.concatenate([sig[keep], out["sigmas"]]))
        ids = np.ascontiguousarray(
            np.concatenate([ids[keep], out["ids"]]).astype(np.uint32))
        if out["status"] != "context_rebuild_required":
            break
    out = dict(out)
    out["seconds"] = perf_counter() - t0
    out["rounds"] = rounds
    out["total_fits"] = total_fits
    # Report the whole configuration the caller now holds, not just the last
    # group's free set.
    out["positions"], out["amplitudes"], out["sigmas"], out["ids"] = pos, amp, sig, ids
    return out


# ---------------------------------------------------------------------------
# The controls. Each yields dicts of per-trial outcomes.
# ---------------------------------------------------------------------------

def case_empty(trials, rng):
    """Empty region: any emitter at all is a false addition."""
    for t in range(trials):
        image, bmap, *_ = frame((24, 24), [], 6.0, 1000 + t)
        out = run_group(image, bmap, [], (12.0, 12.0))
        yield dict(truth=0, found=len(out["amplitudes"]), out=out,
                   truth_pos=np.empty((0, 2)))


def case_isolated(trials, rng):
    """One in-focus source, entered at truth. Kept, not split, not moved."""
    for t in range(trials):
        flux = rng.uniform(900, 1900)
        y, x = 12.0 + rng.uniform(-0.5, 0.5), 12.0 + rng.uniform(-0.5, 0.5)
        truth = [(y, x, flux, SIGMA)]
        image, bmap, tp, *_ = frame((26, 26), truth, 5.0, 2000 + t)
        out = run_group(image, bmap, [(y, x, flux, SIGMA)], (y, x))
        yield dict(truth=1, found=len(out["amplitudes"]), out=out, truth_pos=tp)


def case_close_pair(trials, rng, equal=True):
    """A pair at 0.5-2.0 sigma, entered as ONE merged source.

    The move the whole design is for: the merged incumbent has no residual peak
    to birth into, so the pair has to be recovered by a split, scored against
    the same incumbent by the same rule that would remove one of them again.
    """
    for t in range(trials):
        sep = rng.uniform(0.5, 2.0) * SIGMA
        ang = rng.uniform(0, np.pi)
        f1 = rng.uniform(900, 1900)
        f2 = f1 if equal else f1 * rng.uniform(0.3, 0.7)
        cy, cx = 14.0, 14.0
        d = 0.5 * sep * np.array([np.cos(ang), np.sin(ang)])
        truth = [(cy + d[0], cx + d[1], f1, SIGMA),
                 (cy - d[0], cx - d[1], f2, SIGMA)]
        image, bmap, tp, *_ = frame((30, 30), truth, 5.0, 3000 + t)
        # Entered merged: one source carrying both fluxes, slightly widened,
        # which is what a fixed-width fit of an unresolved pair produces.
        out = run_group(image, bmap, [(cy, cx, f1 + f2, 1.35)], (cy, cx))
        yield dict(truth=2, found=len(out["amplitudes"]), out=out,
                   truth_pos=tp, sep=sep / SIGMA)


def case_broad(trials, rng):
    """A defocused source. Widening must beat tiling it into pieces."""
    for t in range(trials):
        s = rng.uniform(1.6, 2.1) * SIGMA / SIGMA * SIGMA
        s = rng.uniform(1.7, 2.3)
        flux = rng.uniform(1800, 3200)
        truth = [(15.0, 15.0, flux, s)]
        image, bmap, tp, *_ = frame((34, 34), truth, 5.0, 4000 + t)
        out = run_group(image, bmap, [(15.0, 15.0, flux, SIGMA)], (15.0, 15.0))
        yield dict(truth=1, found=len(out["amplitudes"]), out=out,
                   truth_pos=tp, truth_sigma=s)


def case_narrow_beside_broad(trials, rng):
    """A point source next to a defocused one: the neighbour must survive."""
    for t in range(trials):
        sep = rng.uniform(1.5, 3.0) * SIGMA
        broad_s = rng.uniform(1.8, 2.3)
        truth = [(15.0, 15.0, 3000.0, broad_s),
                 (15.0, 15.0 + sep, rng.uniform(900, 1600), SIGMA)]
        image, bmap, tp, *_ = frame((34, 34), truth, 5.0, 5000 + t)
        # Entered as the broad source alone, already wide enough to have
        # swallowed the neighbour.
        out = run_group(image, bmap, [(15.0, 15.0 + 0.3 * sep, 4200.0, 2.2)],
                        (15.0, 15.0))
        yield dict(truth=2, found=len(out["amplitudes"]), out=out, truth_pos=tp)


def case_degenerate(trials, rng):
    """Zero flux, coincident sources, and a width driven onto its bound.

    What is measured is that these produce an explicit status and no infinite
    score -- not that they produce a particular count.
    """
    for t in range(trials):
        kind = t % 3
        if kind == 0:                      # a source with essentially no flux
            truth = [(14.0, 14.0, 1400.0, SIGMA)]
            entry = [(14.0, 14.0, 1400.0, SIGMA), (18.0, 18.0, 1e-3, SIGMA)]
        elif kind == 1:                    # two sources at the same place
            truth = [(14.0, 14.0, 1600.0, SIGMA)]
            entry = [(14.0, 14.0, 800.0, SIGMA), (14.0, 14.0, 800.0, SIGMA)]
        else:                              # narrower than the model space
            truth = [(14.0, 14.0, 1400.0, 0.60 * SIGMA)]
            entry = [(14.0, 14.0, 1400.0, SLACK[0] * SIGMA * 1.01)]
        image, bmap, tp, *_ = frame((30, 30), truth, 5.0, 6000 + t)
        out = run_group(image, bmap, entry, (14.0, 14.0))
        yield dict(truth=len(truth), found=len(out["amplitudes"]), out=out,
                   truth_pos=tp, sub_case=["zero_flux", "coincident",
                                           "below_width_bound"][kind])


def case_sloped_background(trials, rng):
    """A tilted background: the free level and the known shape must split it."""
    for t in range(trials):
        truth = [(15.0, 15.0, rng.uniform(900, 1900), SIGMA)]
        image, bmap, tp, *_ = frame((30, 30), truth, 6.0, 7000 + t,
                                    slope=rng.uniform(0.15, 0.45))
        out = run_group(image, bmap, [(15.0, 15.0, 1200.0, SIGMA)], (15.0, 15.0))
        yield dict(truth=1, found=len(out["amplitudes"]), out=out, truth_pos=tp)


def case_frozen_neighbour(trials, rng):
    """A bright source just outside the group. Its light is in the halo once."""
    for t in range(trials):
        flux = rng.uniform(900, 1900)
        truth = [(15.0, 15.0, flux, SIGMA), (15.0, 26.0, 6000.0, SIGMA)]
        image, bmap, tp, *_ = frame((34, 40), truth, 5.0, 8000 + t)
        out = run_group(image, bmap,
                        [(15.0, 15.0, flux, SIGMA), (15.0, 26.0, 6000.0, SIGMA)],
                        (15.0, 15.0))
        # Both truth sources are scored: the bright one is expected to be
        # frozen, so it must come back unchanged rather than absent. The number
        # that actually tests the halo is the in-group source's FLUX -- double
        # counting the neighbour would push it down, omitting it would push it
        # up, and neither shows in a position match.
        near = np.argmin(np.linalg.norm(out["positions"] - tp[0], axis=1)) \
            if len(out["positions"]) else None
        yield dict(truth=2, found=len(out["amplitudes"]), out=out,
                   truth_pos=tp,
                   flux_error=(abs(out["amplitudes"][near] - flux) / flux
                               if near is not None else None))


def case_frame_edge(trials, rng):
    """A source against the frame border: less context, still one source."""
    for t in range(trials):
        flux = rng.uniform(1100, 1900)
        y = rng.uniform(1.5, 3.0)
        truth = [(y, 15.0, flux, SIGMA)]
        image, bmap, tp, *_ = frame((26, 30), truth, 5.0, 9000 + t)
        out = run_group(image, bmap, [(y, 15.0, flux, SIGMA)], (y, 15.0))
        yield dict(truth=1, found=len(out["amplitudes"]), out=out, truth_pos=tp)


def case_crowd(trials, rng):
    """More coupled sources than `k_max`: capacity behaviour and identities."""
    for t in range(trials):
        n = 14
        ang = np.arange(n) * 2 * np.pi / n
        r = 3.2
        truth = [(20.0 + r * np.cos(a), 20.0 + r * np.sin(a),
                  rng.uniform(900, 1500), SIGMA) for a in ang]
        image, bmap, tp, *_ = frame((40, 40), truth, 5.0, 10000 + t)
        entry = [(y, x, f, SIGMA) for (y, x, f, _) in truth]
        out = run_group(image, bmap, entry, (truth[0][0], truth[0][1]))
        # Identity: every entering id either survives or is in `removed`.
        entering = set(range(n))
        surviving = set(int(i) for i in out["ids"])
        gone = set(int(i) for i in out["removed"])
        yield dict(truth=n, found=len(out["amplitudes"]), out=out, truth_pos=tp,
                   ids_accounted=bool((surviving | gone) >= entering))


def case_overfit(trials, rng):
    """Entered with more sources than exist: the search must come back down."""
    for t in range(trials):
        flux = rng.uniform(1200, 2000)
        truth = [(15.0, 15.0, flux, SIGMA)]
        image, bmap, tp, *_ = frame((30, 30), truth, 5.0, 11000 + t)
        entry = [(15.0 - 0.7, 15.0, flux / 3, SIGMA),
                 (15.0 + 0.7, 15.0, flux / 3, SIGMA),
                 (15.0, 15.0 + 1.4, flux / 3, SIGMA)]
        out = run_group(image, bmap, entry, (15.0, 15.0))
        yield dict(truth=1, found=len(out["amplitudes"]), out=out, truth_pos=tp)


def case_underfit(trials, rng):
    """Entered empty where three sources exist: the search must come up."""
    for t in range(trials):
        base = np.array([[13.0, 13.0], [16.5, 14.0], [14.5, 17.5]])
        truth = [(y, x, rng.uniform(1100, 1900), SIGMA) for y, x in base]
        image, bmap, tp, *_ = frame((30, 30), truth, 5.0, 12000 + t)
        out = run_group(image, bmap, [], (14.7, 14.8), max_moves=6)
        yield dict(truth=3, found=len(out["amplitudes"]), out=out, truth_pos=tp)


CASES = {
    "empty": case_empty,
    "isolated": case_isolated,
    "close_pair_equal": lambda n, r: case_close_pair(n, r, equal=True),
    "close_pair_unequal": lambda n, r: case_close_pair(n, r, equal=False),
    "broad": case_broad,
    "narrow_beside_broad": case_narrow_beside_broad,
    "degenerate": case_degenerate,
    "sloped_background": case_sloped_background,
    "frozen_neighbour": case_frozen_neighbour,
    "frame_edge": case_frame_edge,
    "crowd": case_crowd,
    "overfit": case_overfit,
    "underfit": case_underfit,
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(name, rows):
    n = len(rows)
    statuses, unsupported = {}, {}
    n_true = n_est = n_match = 0
    sq, fits, secs, moves = [], [], [], []
    unresolved = rebuild = capacity = infinite = 0
    for r in rows:
        out = r["out"]
        statuses[out["status"]] = statuses.get(out["status"], 0) + 1
        for k, v in out["diagnostics"]["unsupported"].items():
            unsupported[k] = unsupported.get(k, 0) + v
        unresolved += out["status"] == "unresolved_comparison"
        rebuild += out["status"] == "context_rebuild_required"
        capacity += bool(out["diagnostics"]["capacity_limited"])
        if out["score"] is not None and not np.isfinite(out["score"]):
            infinite += 1
        fits.append(out.get("total_fits", out["diagnostics"]["n_fits"]))
        moves.append(len(out["trace"]))
        secs.append(out["seconds"])
        tp, ep = r["truth_pos"], out["positions"]
        n_true += len(tp)
        n_est += len(ep)
        if len(tp) and len(ep):
            m = metrics.match(tp, ep, radius=MATCH_RADIUS)
            n_match += m.n_matched
            if m.n_matched and np.isfinite(m.rmse):
                sq.append(m.rmse ** 2 * m.n_matched)
    return dict(
        case=name, trials=n,
        truth=n_true, detected=n_est, matched=n_match,
        recall=round(n_match / n_true, 4) if n_true else None,
        precision=round(n_match / n_est, 4) if n_est else None,
        rmse_px=round(float(np.sqrt(sum(sq) / n_match)), 4) if n_match else None,
        count_exact=round(
            float(np.mean([r["found"] == r["truth"] for r in rows])), 4),
        unresolved=round(unresolved / n, 4),
        rebuild_required=round(rebuild / n, 4),
        capacity_limited=round(capacity / n, 4),
        infinite_scores=infinite,
        median_fits=int(np.median(fits)), p95_fits=int(np.percentile(fits, 95)),
        median_moves=float(np.median(moves)),
        median_rounds=float(np.median([r["out"].get("rounds", 1) for r in rows])),
        median_flux_error=(
            round(float(np.median([r["flux_error"] for r in rows
                                   if r.get("flux_error") is not None])), 4)
            if any(r.get("flux_error") is not None for r in rows) else None),
        incumbent_unsupported=round(float(np.mean(
            [r["out"]["diagnostics"]["incumbent_unsupported"] for r in rows])), 4),
        median_ms=round(float(np.median(secs)) * 1e3, 2),
        p95_ms=round(float(np.percentile(secs, 95)) * 1e3, 2),
        statuses=statuses, unsupported=unsupported,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", nargs="+", default=sorted(CASES))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for name in args.case:
        if name not in CASES:
            raise SystemExit(f"unknown case {name!r}; have {sorted(CASES)}")
    # Warm the extension outside the timings.
    run_group(np.full((12, 12), 5.0), np.full((12, 12), 5.0), [], (6.0, 6.0))
    for name in args.case:
        rng = np.random.default_rng(args.seed)
        rows = list(CASES[name](args.trials, rng))
        print(json.dumps(summarize(name, rows)), flush=True)
