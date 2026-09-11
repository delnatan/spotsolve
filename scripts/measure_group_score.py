"""Measure the two numerical constants the group search's comparison rests on.

Neither `dense_group::SCORE_TOL` nor `dense_group::COND_LIMIT` may be chosen:
one is the resolution the configuration score is actually known to, and the
other is where its `logdet` term stops being arithmetic. Both are measured
here, on the matrices and configurations this path really produces.

    python scripts/measure_group_score.py

Run after `maturin develop --release -m rust/spotsolve-py/Cargo.toml`.

Part 1 -- score resolution.
    One configuration, refitted from perturbed starts and scored each time.
    Two effects have to be separated, and conflating them gives a number two
    orders too large:

      * MULTIMODALITY. From starts 1% apart, a crowded K=3 group reaches local
        optima up to 3 nats apart in the data term alone. That is not noise in
        the score -- those are different configurations, and preferring the
        better one is the score working. The search handles it with restarts
        and by keeping the best evaluated state per hypothesis.
      * RESOLUTION. Among fits that reached the SAME optimum, the score still
        varies, because the fit is stationary for `I - log pi` while the score
        also carries `-logdet(F)/2`, whose gradient there is not zero. So a
        returned parameter anywhere inside the stationarity band moves the
        score at FIRST order.

    Only the second is `SCORE_TOL`'s business, so runs are grouped by their
    converged `I` and the spread is taken within a group.

Part 2 -- logdet precision against the scaled condition number.
    Symmetric positive definite matrices at `p = P_MAX_VAR` with a prescribed
    scaled condition number and a KNOWN log-determinant, factorized by the same
    Cholesky the score uses. `COND_LIMIT` must be low enough that the error
    here stays far under `SCORE_TOL`.

Part 3 -- the realized condition numbers.
    What the detector's own groups actually produce, so the limit can be seen
    to be a validity guard rather than a detection rule in disguise.

Part 4 -- the outside-the-box test's operating point.
    `OUTSIDE_PEAK_ALPHA` is a family-wise rate, and the realized rate is not
    the nominal one: local maxima of a smooth field sample its supremum rather
    than `area/win^2` independent draws, which `calibrate.py` records as the
    reason its own seeder runs ~6x its nominal alpha. So both arms are
    measured -- how often a group with nothing outside its box asks for a
    rebuild, and how often a group with a real neighbour out there notices.
"""

import argparse
import json

import numpy as np

import spotsolve_rs
from spotsolve import backend, prior
from spotsolve.simulate import simulate

SIGMA = 1.2
SLACK = (0.70, 2.2)
BAND = (0.80, 2.0)


def width_prior(lam=0.02):
    return prior.FocusMixtureWidth(lam, lam * 0.1, SLACK[0] * SIGMA,
                                   BAND[1] * SIGMA, SLACK[1] * SIGMA, SIGMA)


def engine(image, bmap, a_s=1400.0, k_max=12):
    return backend.get("rs").group_engine(image, bmap, SIGMA, SLACK, k_max,
                                          width_prior(), a_s)


def score_resolution(n_cases=24, n_perturb=16, jitter=1e-2, seed=0,
                     same_optimum=1e-6):
    """Spread of the score among fits that reached the same optimum.

    `same_optimum` is the tolerance, in nats of the data term, within which two
    fits are taken to have found the same mode. It is far tighter than any
    difference the score acts on and far looser than the arithmetic, so the
    grouping is not sensitive to it.
    """
    rng = np.random.default_rng(seed)
    spreads, modes, n_modes = [], [], []
    for case in range(n_cases):
        sim = simulate(shape=(30, 30), density=0.02, amplitude_range=(900, 1900),
                       background=5, sigma_spread=0.2, seed=100 + case)
        if len(sim.positions) == 0:
            continue
        bmap = np.full(sim.image.shape, 5.0)
        eng = engine(sim.image, bmap)
        i = int(rng.integers(len(sim.positions)))
        focus = tuple(sim.positions[i])
        # The group's own free set is chosen natively; hand in every emitter.
        pos = np.ascontiguousarray(sim.positions)
        amp = np.ascontiguousarray(sim.amplitudes)
        sig = np.ascontiguousarray(
            sim.sigmas if sim.sigmas is not None
            else np.full(len(pos), SIGMA))
        ids = np.arange(len(pos), dtype=np.uint32)
        scores, divs = [], []
        for _ in range(n_perturb):
            dp = pos + rng.normal(0.0, jitter, pos.shape)
            da = amp * (1.0 + rng.normal(0.0, jitter, amp.shape))
            ds = sig * (1.0 + rng.normal(0.0, jitter, sig.shape))
            r = eng.score_state(dp, da, ds, ids, focus)
            if r["score"] is not None:
                scores.append(r["score"])
                divs.append(r["i_div"])
        # Group by converged `I`: same mode, different route to it.
        if len(scores) >= 4:
            order = np.argsort(divs)
            sc = np.array(scores)[order]
            iv = np.array(divs)[order]
            start = 0
            for end in range(1, len(iv) + 1):
                if end == len(iv) or iv[end] - iv[start] > same_optimum:
                    if end - start >= 2:
                        spreads.append(float(sc[start:end].max()
                                             - sc[start:end].min()))
                        modes.append(end - start)
                    start = end
            n_modes.append(int(np.sum(np.diff(iv) > same_optimum)) + 1)
    spreads = np.array(spreads)
    return dict(
        part="score_resolution", n=int(len(spreads)), jitter=jitter,
        groups_with_repeats=int(len(spreads)),
        median_modes_per_case=float(np.median(n_modes)) if n_modes else None,
        median=float(np.median(spreads)) if len(spreads) else None,
        p99=float(np.percentile(spreads, 99)) if len(spreads) else None,
        max=float(np.max(spreads)) if len(spreads) else None,
        score_tol=spotsolve_rs.GROUP_SCORE_TOL,
    )


def logdet_precision(p=None, seed=1, repeats=8):
    """|Cholesky logdet - the exact one| against the scaled condition number."""
    p = p or spotsolve_rs.GROUP_P_MAX_VAR
    rng = np.random.default_rng(seed)
    rows = []
    for target in [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]:
        errs = []
        for _ in range(repeats):
            # Eigenvalues log-spaced over the target range, so log|A| is known
            # exactly as their log-sum rather than by a second factorization.
            d = np.exp(np.linspace(0.0, np.log(target), p))[::-1]
            q, _ = np.linalg.qr(rng.normal(size=(p, p)))
            a = (q * d) @ q.T
            a = 0.5 * (a + a.T)
            # Scale to unit diagonal, which is the form the guard measures.
            s = 1.0 / np.sqrt(np.diag(a))
            a = a * s[:, None] * s[None, :]
            exact = float(np.sum(np.log(d)) + 2.0 * np.sum(np.log(s)))
            ld, cond, ok = spotsolve_rs.logdet_cond(np.ascontiguousarray(a))
            if ok:
                errs.append((abs(ld - exact), cond))
        if errs:
            e = np.array(errs)
            rows.append(dict(target_cond=target, p=p,
                             measured_cond=float(np.median(e[:, 1])),
                             max_logdet_error=float(np.max(e[:, 0]))))
    return dict(part="logdet_precision", cond_limit=spotsolve_rs.GROUP_COND_LIMIT,
                score_tol=spotsolve_rs.GROUP_SCORE_TOL, rows=rows)


def realized_conditions(n_frames=6, seed=2):
    """Scaled condition numbers of the curvature on real detector groups."""
    conds, statuses = [], {}
    for f in range(n_frames):
        sim = simulate(shape=(40, 40), density=0.034, amplitude_range=(900, 1900),
                       background=5, sigma_spread=0.2, seed=200 + f + seed)
        if len(sim.positions) == 0:
            continue
        bmap = np.full(sim.image.shape, 5.0)
        eng = engine(sim.image, bmap)
        pos = np.ascontiguousarray(sim.positions)
        amp = np.ascontiguousarray(sim.amplitudes)
        sig = np.ascontiguousarray(
            sim.sigmas if sim.sigmas is not None
            else np.full(len(pos), SIGMA))
        ids = np.arange(len(pos), dtype=np.uint32)
        for i in range(len(pos)):
            r = eng.score_state(pos, amp, sig, ids, tuple(pos[i]))
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
            if np.isfinite(r["cond"]):
                conds.append(r["cond"])
    c = np.array(conds)
    return dict(
        part="realized_conditions", n=int(len(c)), statuses=statuses,
        median=float(np.median(c)) if len(c) else None,
        p99=float(np.percentile(c, 99)) if len(c) else None,
        max=float(np.max(c)) if len(c) else None,
        cond_limit=spotsolve_rs.GROUP_COND_LIMIT,
    )


def outside_box_operating_point(trials=40, seed=3,
                                alphas=(0.05, 0.01, 1e-3, 1e-4, 1e-5)):
    """False rebuild requests against real neighbours found, per alpha."""
    rng = np.random.default_rng(seed)
    # Arm A: an isolated in-focus source. Nothing is outside the box.
    quiet = []
    for t in range(trials):
        flux = rng.uniform(900, 1900)
        truth = [(15.0, 15.0, flux, SIGMA)]
        image, bmap, *_ = _frame((30, 30), truth, 5.0, 20000 + t)
        quiet.append((image, bmap, [(15.0, 15.0, flux, SIGMA)], (15.0, 15.0)))
    # Arm B: a second source in the annulus -- inside the pixel region, outside
    # the admissible position box, so no hypothesis here can propose for it.
    # Swept in flux, because a cut that finds every bright neighbour says
    # nothing about the faint ones it was tightened past.
    neighbour_fluxes = (1500.0, 300.0, 100.0, 60.0, 40.0)
    loud = {f: [] for f in neighbour_fluxes}
    for f2 in neighbour_fluxes:
        for t in range(trials):
            f1 = rng.uniform(1100, 1900)
            truth = [(15.0, 15.0, f1, SIGMA), (15.0, 15.0 + 8.0, f2, SIGMA)]
            image, bmap, *_ = _frame((30, 34), truth, 5.0, 21000 + t)
            loud[f2].append((image, bmap, [(15.0, 15.0, f1, SIGMA)], (15.0, 15.0)))

    def rate(arm, alpha):
        n = sum(bool(_run(*a, outside_peak_alpha=alpha)["diagnostics"]
                     ["residual_peak_outside_box"]) for a in arm)
        return round(n / len(arm), 4)

    rows = []
    for alpha in alphas:
        rows.append(dict(
            alpha=alpha, false_rebuild=rate(quiet, alpha),
            found={f: rate(loud[f], alpha) for f in neighbour_fluxes}))
    return dict(part="outside_box", trials=trials,
                shipped_alpha=spotsolve_rs.GROUP_OUTSIDE_PEAK_ALPHA, rows=rows)


def _frame(shape, truth, background, seed):
    from spotsolve import calibrate
    h, w = shape
    bmap = np.full(shape, float(background))
    pos = np.array([[t[0], t[1]] for t in truth], float).reshape(-1, 2)
    amp = np.array([t[2] for t in truth], float)
    sig = np.array([t[3] for t in truth], float)
    mean = calibrate.render_model(pos, amp, sig, shape, 0.0) + bmap
    return (np.random.default_rng(seed).poisson(mean).astype(float),
            np.ascontiguousarray(bmap), pos, amp, sig)


def _run(image, bmap, entry, focus, **settings):
    eng = engine(image, bmap)
    pos = np.ascontiguousarray(
        np.array([[e[0], e[1]] for e in entry], float).reshape(-1, 2))
    amp = np.ascontiguousarray(np.array([e[2] for e in entry], float))
    sig = np.ascontiguousarray(np.array([e[3] for e in entry], float))
    ids = np.arange(len(entry), dtype=np.uint32)
    return eng.search_group(pos, amp, sig, ids, tuple(focus), **settings)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jitter", type=float, default=1e-7)
    ap.add_argument("--cases", type=int, default=24)
    args = ap.parse_args()
    print(json.dumps(score_resolution(n_cases=args.cases, jitter=args.jitter)))
    print(json.dumps(logdet_precision()))
    print(json.dumps(realized_conditions()))
    print(json.dumps(outside_box_operating_point()))
