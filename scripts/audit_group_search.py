"""Is a group-search miss the SCORE's fault or the SEARCH's?

    python scripts/audit_group_search.py
    python scripts/audit_group_search.py --sizes 64 --seeds 17 18 19 20 21

Run after `maturin develop --release -m rust/spotsolve-py/Cargo.toml`.

`detect(search="groups")` commits only strict improvements of one
configuration score, so every way its answer can differ from the truth falls
in one of two places, and they need opposite fixes:

* **model failure** -- the found configuration SCORES HIGHER than the truth in
  the same context. The search did its job; the score prefers the wrong
  answer, and no schedule, restart or proposal change can fix that.
* **search failure** -- the truth scores higher. A better configuration exists
  under the score and the search did not reach it.

This script finds every place the two disagree and scores both sides there.
A discrepancy is a cluster of unmatched truth and unmatched detections, linked
within `LINK_PX`; its two configurations are the frame's full found
configuration, and the same with every found emitter within `CLUSTER_PX` of
the cluster replaced by every true one there. Both are fitted and scored by
`DenseGroupEngine.score_state` -- the exact code the search decides with --
under the final epoch's own snapshot (background surface and priors), in one
context whose position box covers both sides.

`gain = score(truth side) - score(found side)`. Within `TIE_NATS` of zero a
cluster is reported as a tie: the two are separated by less than the fits'
own multimodality (`dense_group::SCORE_TOL`'s note measures optima of one
configuration spread over nats), so its sign is not evidence either way.
"""

import argparse

import numpy as np

from spotsolve import backend, detect, metrics
from spotsolve import core
from spotsolve.simulate import simulate

LINK_PX = 3.0      # unmatched items this close are one discrepancy (2.5 sigma)
CLUSTER_PX = 1.8   # found/true emitters this close to a cluster are swapped
MATCH_PX = 1.2     # sigma: the matching radius `check_detect_width` uses
TIE_NATS = 1.0


def full_configuration(r):
    """Every emitter the search holds: reported ones plus the width rejects.

    The reporting split drops out-of-band objects AFTER the search; they were
    modelled, and a comparison without them would put their light back into
    the residual.
    """
    pos, amp, sig = r.positions, r.amplitudes, r.fit_sigma
    if r.width_rejects is not None and len(r.width_rejects):
        w = r.width_rejects
        pos = np.vstack([pos, np.column_stack([w["y"], w["x"]])])
        amp = np.concatenate([amp, w["flux"]])
        sig = np.concatenate([sig, w["sigma"]])
    return pos.reshape(-1, 2), amp, sig


def final_snapshot_engine(r, image, sigma, slack, band, k_max):
    """An engine carrying the final epoch's snapshot, as `detect` built it.

    The final epoch is the one that committed nothing, so its snapshot was
    computed from the final configuration: the returned background surface,
    `A_s`, and the class rates `detect._rates` derives from the widths.
    """
    _, _, sig = full_configuration(r)
    px = float(image.size)
    n_f = int(np.count_nonzero(sig <= band[1] * sigma))
    rates = (max(n_f, 1) / px, max(len(sig) - n_f, 1) / px) if len(sig) \
        else (r.lam, r.lam)
    wp = core._width_prior(slack, band, sigma, *rates)
    return backend.get("rs").group_engine(image, np.ascontiguousarray(r.background),
                                          sigma, slack, k_max, wp, r.A_s,
                                          next_id=len(sig) + 1000)


def clusters(truth_pos, found_pos):
    """Connected groups of unmatched truth and unmatched detections."""
    m = metrics.match(truth_pos, found_pos, radius=MATCH_PX)
    ut = np.setdiff1d(np.arange(len(truth_pos)), m.matched_true_idx)
    uf = np.setdiff1d(np.arange(len(found_pos)), m.matched_est_idx)
    pts = np.vstack([truth_pos[ut], found_pos[uf]]) if len(ut) + len(uf) \
        else np.empty((0, 2))
    kinds = [("truth", i) for i in ut] + [("found", i) for i in uf]
    groups = core._link_groups(pts, LINK_PX) if len(pts) else []
    return [[kinds[j] for j in g] for g in groups]


def score(eng, pos, amp, sig, focus, seeds, lo, hi):
    """Fit and score under the search's own budget policy: the default fit,
    then the escalated budget only if that did not certify a mode."""
    args = (np.ascontiguousarray(pos, float), np.ascontiguousarray(amp, float),
            np.ascontiguousarray(np.clip(sig, lo * 1.0001, hi * 0.9999), float),
            np.arange(len(amp), dtype=np.uint32), focus,
            np.ascontiguousarray(seeds, float))
    r = eng.score_state(*args)
    if r["status"] == "nonstationary":
        r = eng.score_state(*args, max_iter=300, tol_obj=1e-10)
    return r


def audit_frame(size, seed, density, spread):
    sim = simulate(shape=(size, size), density=density,
                   amplitude_range=(900, 1900), background=5,
                   sigma_spread=spread, seed=seed)
    sigma = sim.sigma
    slack, band, k_max = core.SIGMA_SLACK, core.FOCUS_BAND, 12
    r = detect(sim.image, sigma=sigma, gain=1.0, impl="rs", search="groups",
               verbose=0)
    fpos, famp, fsig = full_configuration(r)
    tpos = sim.positions
    tamp = sim.amplitudes
    tsig = sim.sigmas if sim.sigmas is not None else np.full(len(tpos), sigma)
    eng = final_snapshot_engine(r, sim.image, sigma, slack, band, k_max)
    lo, hi = slack[0] * sigma, slack[1] * sigma

    groups = clusters(tpos, fpos)
    rows = []
    for g in groups:
        where = np.array([tpos[i] if k == "truth" else fpos[i] for k, i in g])
        near_f = np.flatnonzero(np.min(np.linalg.norm(
            fpos[:, None] - where[None], axis=2), axis=1) <= CLUSTER_PX)
        near_t = np.flatnonzero(np.min(np.linalg.norm(
            tpos[:, None] - where[None], axis=2), axis=1) <= CLUSTER_PX)
        keep = np.setdiff1d(np.arange(len(fpos)), near_f)
        t_pos = np.vstack([fpos[keep], tpos[near_t]])
        t_amp = np.concatenate([famp[keep], tamp[near_t]])
        t_sig = np.concatenate([fsig[keep], tsig[near_t]])
        focus = tuple(where.mean(axis=0))
        # One position box for both sides: every emitter either side places
        # in the cluster is a seed, and seeds extend the box.
        seeds = np.vstack([fpos[near_f], tpos[near_t]])
        F = score(eng, fpos, famp, fsig, focus, seeds, lo, hi)
        T = score(eng, t_pos, t_amp, t_sig, focus, seeds, lo, hi)
        ok = F["status"] == "supported" and T["status"] == "supported"
        gain = (T["score"] - F["score"]) if ok else np.nan
        if not ok:
            verdict = "unscorable"
        elif gain > TIE_NATS:
            verdict = "search"
        elif gain < -TIE_NATS:
            verdict = "model"
        else:
            verdict = "tie"
        n_missed = sum(1 for k, i in g if k == "truth"
                       and band[0] * sigma <= tsig[i] <= band[1] * sigma)
        rows.append(dict(
            size=size, seed=seed, verdict=verdict, gain=gain,
            n_true=len(near_t), n_found=len(near_f), n_missed=n_missed,
            n_false=sum(1 for k, i in g if k == "found"
                        and band[0] * sigma <= fsig[i] <= band[1] * sigma),
            d_fit=(-T["i_div"] + F["i_div"]) if ok else np.nan,
            d_prior=(T["log_prior"] - F["log_prior"]) if ok else np.nan,
            d_volume=((T["log_volume"] + T["log_box"])
                      - (F["log_volume"] + F["log_box"])) if ok else np.nan,
            found_max_sigma=float(fsig[near_f].max()) / sigma if len(near_f) else np.nan,
            # Closest true pair in the cluster, in PSF widths: below ~0.75
            # the frame carries almost no information about the count, and
            # the prior is an operating point there rather than an estimator.
            sep=(float(np.min(np.linalg.norm(tpos[near_t][:, None] - tpos[near_t][None], axis=2)
                              + np.eye(len(near_t)) * 1e9)) / sigma
                 if len(near_t) > 1 else np.nan),
            statuses=(F["status"], T["status"]),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[39, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19])
    ap.add_argument("--density", type=float, default=0.034)
    ap.add_argument("--spread", type=float, default=0.2)
    args = ap.parse_args()

    rows = []
    print("frame     verdict     gain   true found missed false   d_fit d_prior d_vol  "
          "found max sigma/sigma0")
    for size in args.sizes:
        for seed in args.seeds:
            for row in audit_frame(size, seed, args.density, args.spread):
                rows.append(row)
                print(f"{size:3d}/{seed:<4d} {row['verdict']:10s} {row['gain']:+7.2f}"
                      f"   {row['n_true']:3d}  {row['n_found']:3d}   {row['n_missed']:3d}"
                      f"   {row['n_false']:3d}  {row['d_fit']:+7.2f} {row['d_prior']:+7.2f}"
                      f" {row['d_volume']:+6.2f}   {row['found_max_sigma']:.2f}   sep {row['sep']:.2f}"
                      + ("" if row["verdict"] != "unscorable" else f"   {row['statuses']}"),
                      flush=True)

    print("\nsummary: in-focus truth missed, and false detections, by verdict")
    tot_m = sum(r["n_missed"] for r in rows) or 1
    tot_f = sum(r["n_false"] for r in rows) or 1
    for v in ("model", "search", "tie", "unscorable"):
        rs = [r for r in rows if r["verdict"] == v]
        m = sum(r["n_missed"] for r in rs)
        f = sum(r["n_false"] for r in rs)
        med = np.median([r["gain"] for r in rs]) if rs and v != "unscorable" else np.nan
        print(f"  {v:10s} clusters {len(rs):3d}   missed {m:3d} ({m / tot_m:4.0%})"
              f"   false {f:3d} ({f / tot_f:4.0%})   median gain {med:+.2f}")
    mod = [r for r in rows if r["verdict"] == "model"]
    seps = np.array([r["sep"] for r in mod if np.isfinite(r["sep"])])
    if len(seps):
        print("  model failures by closest true pair (sigma): "
              + "  ".join(f"<{b}: {np.sum(seps < b)}" for b in (0.75, 1.0, 1.5, 2.0))
              + f"  >=2: {np.sum(seps >= 2.0)}   median {np.median(seps):.2f}")
    if mod:
        print("  model failures: median d_fit %+.1f, d_prior %+.1f, d_volume %+.1f;"
              " found side widened past 1.3 sigma0 in %d of %d"
              % (np.median([r["d_fit"] for r in mod]), np.median([r["d_prior"] for r in mod]),
                 np.median([r["d_volume"] for r in mod]),
                 sum(r["found_max_sigma"] > 1.3 for r in mod), len(mod)))


if __name__ == "__main__":
    main()
