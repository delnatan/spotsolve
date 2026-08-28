"""Generate language-neutral golden fixtures for the Rust port.

The Python verification scripts (`verify_*.py`) check this implementation
against analytic truth. They cannot check a *second* implementation against
*this* one, which is what a port needs, and eyeballing `run_box.py` is not a
regression test. This writes JSON the Rust side can read and assert against,
layer by layer, so a port can be validated incrementally rather than only at
the end -- where "the count is different" localizes to nothing.

    python make_fixtures.py            # writes fixtures/*.json

Every float is written with 17 significant digits, which round-trips f64
exactly, so the early layers can be compared bit-for-bit. Read the tolerance
guidance in each fixture's "compare" field: the layers differ in how exactly
they can be reproduced, and pretending otherwise wastes days.
"""

import json
import os

import numpy as np

import boxes as box_mod
import boxsolve
import gsolve
import calibrate
import evidence
import lmga
import msearch
import psf
import score
import simulate

OUT = "fixtures"
SIGMA = 1.2
GAIN, OFFSET = 4.23, 100.0


def enc(o):
    """JSON-encode with full f64 precision."""
    if isinstance(o, (np.floating, float)):
        return float(f"{float(o):.17g}")
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [enc(v) for v in o.tolist()]
    if isinstance(o, (list, tuple)):
        return [enc(v) for v in o]
    if isinstance(o, dict):
        return {k: enc(v) for k, v in o.items()}
    return o


def write(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".json")
    with open(p, "w") as f:
        json.dump(enc(obj), f, indent=1)
    print(f"  wrote {p}")


def field(size, density, seed):
    n = max(1, int(round(density * size * size)))
    sim = simulate.simulate(shape=(size, size), n_emitters=n, background=4.0,
                            amplitude_range=(900.0, 1900.0), sigma=SIGMA,
                            border=1.0, seed=seed)
    return sim, sim.image * GAIN + OFFSET


# ---------------------------------------------------------------- layer 1
def fx_psf():
    """The forward model and its Jacobian. Reproducible to the last ulp
    ONLY if erf agrees; expect ~1e-15 relative, not bit-equality, because
    every libm's erf differs. This is the one layer where a tolerance is
    unavoidable, and it sets the floor for everything above it."""
    cases = []
    for K, seed in ((0, 1), (1, 2), (3, 3), (7, 4)):
        r = np.random.default_rng(seed)
        h, w = 13, 11
        yy, xx = np.mgrid[0:h, 0:w] * 1.0
        theta = (psf.pack(r.uniform(2, 8), r.uniform(300, 1800, K),
                          r.uniform(2, h - 2, K), r.uniform(2, w - 2, K))
                 if K else np.array([r.uniform(2, 8)]))
        m = psf.model(theta, yy, xx, SIGMA)
        J = psf.jac(theta, yy, xx, SIGMA)
        cases.append(dict(K=K, h=h, w=w, sigma=SIGMA, theta=theta,
                          peak_factor=psf.peak_factor(SIGMA),
                          model=m, jac_flat=J.reshape(-1, J.shape[-1])))
    return dict(
        what="psf.model / psf.jac on a (h,w) pixel-centre grid",
        layout="theta = [b, A_0, y_0, x_0, A_1, ...]; jac_flat is "
               "(h*w, 3K+1) in C order, i.e. row-major over (y, x)",
        compare="relative 1e-13 elementwise. NOT bit-exact across libm "
                "implementations -- erf is the reason.",
        cases=cases)


# ---------------------------------------------------------------- layer 2
def fx_lmga():
    """The bounded optimizer. Chaotic in the last digits: a 1-ulp difference
    in the model propagates through ~100 iterations and through the
    accept/reject branch on the gain ratio, which can change the ITERATION
    COUNT. Compare the converged objective, not the path."""
    cases = []
    for K, seed in ((1, 11), (2, 12), (3, 13)):
        r = np.random.default_rng(seed)
        h = w = 13
        yy, xx = np.mgrid[0:h, 0:w] * 1.0
        th = psf.pack(r.uniform(2, 8), r.uniform(300, 1800, K),
                      r.uniform(3, 9, K), r.uniform(3, 9, K))
        d = r.poisson(np.maximum(psf.model(th, yy, xx, SIGMA), 1e-9)).astype(float)
        lo = np.array([0.0] + [1e-4, -0.5, -0.5] * K)
        hi = np.array([max(d.max() * 4, 10.0)]
                      + [8 * max(d.max(), 1) / psf.peak_factor(SIGMA), h - 0.5,
                         w - 0.5] * K)
        t0 = np.clip(th * np.append([1.0], r.uniform(0.7, 1.3, 3 * K)),
                     lo + 1e-9, hi - 1e-9)
        res = lmga.fit(t0, yy, xx, SIGMA, d, lo, hi, max_iter=100)
        cases.append(dict(K=K, h=h, w=w, sigma=SIGMA, data=d, theta0=t0,
                          lower=lo, upper=hi,
                          theta=res.theta, I=res.I, F=res.F,
                          n_iter=res.n_iter, converged=res.converged,
                          stalled=res.stalled,
                          grad_inf_norm=float(np.max(np.abs(
                              _grad(res.theta, yy, xx, SIGMA, d))))))
    return dict(
        what="lmga.fit: bounded Fisher-scoring LM on the Poisson I-divergence",
        compare="I to 1e-8 ABSOLUTE (nats -- the unit decisions are made in); "
                "theta to 1e-6 px / 1e-4 relative on amplitude; F to 1e-9 "
                "relative. Do NOT assert on n_iter: the gain-ratio branch "
                "makes it sensitive to the last ulp of the model.",
        cases=cases)


def _grad(theta, yy, xx, sigma, d):
    m, J = psf.model_and_jac(theta, yy, xx, sigma)
    m = np.maximum(m, 1e-9).reshape(-1)
    J = J.reshape(-1, theta.size)
    return J.T @ ((m - d.reshape(-1)) / m)


# ---------------------------------------------------------------- layer 3
def fx_evidence():
    """The Laplace Bayes factor and its two guards. Pure arithmetic on inputs
    supplied here, so this layer IS bit-reproducible -- the only non-trivial
    part is the Cholesky, and log|F| is a sum of logs of its diagonal."""
    cases = []
    r = np.random.default_rng(21)
    for K in (1, 2, 4):
        p = 3 * K + 1
        Ab = r.normal(size=(p, p)); Fb = Ab @ Ab.T + p * np.eye(p)
        Aa = r.normal(size=(p + 3, p + 3)); Fa = Aa @ Aa.T + (p + 3) * np.eye(p + 3)
        ld_b, cond_b, ok_b = evidence.logdet_cond(Fb)
        ld_a, cond_a, ok_a = evidence.logdet_cond(Fa)
        bf, c = evidence.log_bf_add(120.0, 100.0, Fb, Fa, 900.0, 1500.0,
                                    K, 0.02, 950.0)
        rem = evidence.log_bf_remove(100.0, 120.0, Fa, Fb, 1500.0, 900.0,
                                     K + 1, 0.02, 950.0)
        theta = psf.pack(4.0, r.uniform(200, 1500, K), r.uniform(2, 9, K),
                         r.uniform(2, 9, K))
        cases.append(dict(
            K=K, F_before=Fb, F_after=Fa,
            logdet_before=ld_b, cond_before=cond_b, ok_before=ok_b,
            logdet_after=ld_a, cond_after=cond_a, ok_after=ok_a,
            I_before=120.0, I_after=100.0, sumA_before=900.0,
            sumA_after=1500.0, lam=0.02, A_s=950.0,
            log_bf_add=bf, cond_reported=c, log_bf_remove=rem,
            antisymmetry_residual=float(bf + rem),
            theta=theta,
            amplitudes_resolved=evidence.amplitudes_resolved(theta, Fb)))
    return dict(
        what="evidence.logdet_cond / log_bf_add / log_bf_remove",
        invariant="log_bf_remove is the EXACT negation of log_bf_add on the "
                  "same pair; antisymmetry_residual must be 0.0 exactly, not "
                  "small. If it is not, the two paths have diverged.",
        constants=dict(COND_GUARD=evidence.COND_GUARD,
                       RESOLVED_TAU=evidence.RESOLVED_TAU),
        compare="log BF to 1e-10 absolute; logdet to 1e-12 relative.",
        cases=cases)


# ---------------------------------------------------------------- layer 4
def fx_score():
    """The projected score test and its warm start -- the part of the
    algorithm most likely to be got subtly wrong in a port, because a
    CONDITIONAL rather than MARGINAL denominator still looks plausible and
    still produces peaks in roughly the right places."""
    # The three scenarios that matter, and they must be built so the model
    # handed to `add_context` is NOT the model the data came from -- otherwise
    # the residual is pure noise, no site clears z_min, and the fixture
    # exercises nothing.
    #
    #   "null"  : model == truth. Expect NO candidates. This is the false
    #             -positive check, and the only one of the three that a broken
    #             conditional denominator would still pass.
    #   "missed": data has one emitter the model does not. Expect a candidate
    #             on it -- the BIRTH case.
    #   "pair"  : data has two emitters 1.3 sigma apart, model has one at their
    #             midpoint carrying the combined flux. Expect a candidate --
    #             the SPLIT case, and the one that only works because the
    #             denominator is marginal.
    cases = []
    specs = [("null", 3, 41), ("missed", 3, 42), ("pair", 1, 43)]
    for kind, K, seed in specs:
        r = np.random.default_rng(seed)
        h = w = 15
        yy, xx = np.mgrid[0:h, 0:w] * 1.0
        ay, ax = psf.axes(yy, xx)
        # Draw the model's emitters in the upper-left quadrant only, so the
        # "missed" one at (11.5, 11.5) is guaranteed well separated from all
        # of them. Drawing over the whole patch put it 0.45 px from a
        # neighbour on the first attempt -- an unresolved pair below the
        # identifiability limit, which is a different question entirely and
        # correctly scored z = 1.5.
        th = psf.pack(4.0, r.uniform(600, 1600, K), r.uniform(3.0, 7.5, K),
                      r.uniform(3.0, 7.5, K))
        if kind == "null":
            th_true = th
        elif kind == "missed":
            th_true = psf.pack(4.0, list(th[1::3]) + [1100.0],
                               list(th[2::3]) + [11.5],
                               list(th[3::3]) + [11.5])
        else:                       # "pair"
            cy0, cx0 = 7.5, 7.5
            dsep = 1.3 * SIGMA
            th = psf.pack(4.0, [2200.0], [cy0], [cx0])
            th_true = psf.pack(4.0, [1100.0, 1100.0],
                               [cy0 - dsep / 2, cy0 + dsep / 2],
                               [cx0 - dsep / 2, cx0 + dsep / 2])
        d = r.poisson(np.maximum(psf.model(th_true, yy, xx, SIGMA),
                                 1e-9)).astype(float)
        halo = 0.0
        # The incumbent must be AT its optimum for q = J'Wr to vanish, which
        # is the state `score` is always called in. Fit it first, exactly as
        # `search_patch` does, or the fixture tests a state that never occurs.
        Kf = (len(th) - 1) // 3
        lo, hi = msearch._bounds(Kf, h, w, max(float(d.max()) * 4, 10.0),
                                 8.0 * max(float(d.max()), 1.0)
                                 / psf.peak_factor(SIGMA))
        fit = lmga.fit(np.clip(th, lo + 1e-9, hi - 1e-9), yy, xx, SIGMA, d,
                       lo, hi, max_iter=100)
        th = fit.theta
        ctx = score.add_context(th, ay, ax, SIGMA, d, halo)
        probes = [(3.0, 4.0), (7.5, 7.5), (float(th[2]), float(th[3])),
                  (10.25, 2.75), (11.5, 11.5)]
        zs, ahs = [], []
        for (cy, cx) in probes:
            num, den, _ = score._terms(ctx, [cy], [cx])
            zs.append(float(num[0, 0] / np.sqrt(den[0, 0])))
            ahs.append(float(num[0, 0] / den[0, 0]))
        cand = score.candidates(ctx, z_min=msearch.Z_ADD_MIN, n_max=3,
                                step=0.5)
        ws, A_hat, zw = score.warm_start(ctx, probes[0][0], probes[0][1])
        cases.append(dict(kind=kind, K=(len(th) - 1) // 3, h=h, w=w,
                          sigma=SIGMA, theta=th, theta_true=th_true, data=d,
                          probe_positions=probes, z=zs, A_hat=ahs,
                          candidates=[dict(z=c[0], y=c[1], x=c[2], A_hat=c[3])
                                      for c in cand],
                          warm_start_theta=ws, warm_start_A_hat=A_hat,
                          warm_start_z=zw))
    return dict(
        what="score.py: z = num/sqrt(den), den = g'Wg - u'F^-1 u",
        trap="den is the MARGINAL precision (the Schur complement against the "
             "whole incumbent), not the conditional g'Wg. Dropping the "
             "u'F^-1u term still yields a plausible-looking map, still peaks "
             "near real emitters, and silently destroys the screen -- the raw "
             "conditional score was measured to be exceeded by the post-fit "
             "A/SE in 79% of cases. If only one fixture is ported carefully, "
             "make it this one.",
        constants=dict(Z_ADD_MIN=msearch.Z_ADD_MIN),
        compare="z and A_hat to 1e-9 relative; candidate positions exactly "
                "(they are grid points); warm_start_theta to 1e-9 relative.",
        cases=cases)


# ---------------------------------------------------------------- layer 5
def fx_boxes():
    """The tiling. Pure integer geometry -- assert this EXACTLY. The cores
    must partition the image with no gap and no overlap, and the jitter shift
    must move the boundaries themselves (see README section 6: re-spacing the
    same interval was a rounding no-op that left three seams fixed in both
    phases)."""
    cases = []
    for shape in ((39, 39), (62, 62), (37, 53)):
        for shift in (0, 3):
            bxs = box_mod.tile(shape, SIGMA, offset=(shift, shift))
            cover = np.zeros(shape, dtype=int)
            for b in bxs:
                cover[b.cy0:b.cy1, b.cx0:b.cx1] += 1
            cases.append(dict(
                shape=list(shape), shift=shift, n_boxes=len(bxs),
                core_default=box_mod.default_geometry(SIGMA)[0],
                pad_default=box_mod.default_geometry(SIGMA)[1],
                partition_ok=bool(np.all(cover == 1)),
                boxes=[dict(y0=b.y0, x0=b.x0, y1=b.y1, x1=b.x1, cy0=b.cy0,
                            cx0=b.cx0, cy1=b.cy1, cx1=b.cx1) for b in bxs]))
    return dict(
        what="boxes.tile: fit regions, cores, and the jitter offset",
        invariant="cores partition the image exactly once: partition_ok must "
                  "be true for every case. NOTE this bounds double counting "
                  "only -- see PORTING_NOTES section 13 for why it does not "
                  "bound LOSS.",
        compare="exact integer equality.",
        cases=cases)


# ---------------------------------------------------------------- layer 6
def fx_end_to_end():
    """Whole-pipeline results on deterministic synthetic fields.

    This is the acceptance fixture, and it is built from `gsolve` -- the port
    target. It does NOT expect bit-equality: a port that fans `refine`'s patch
    sweep out across threads (PORTING_NOTES section 14) legitimately lands
    somewhere slightly different. Judge it on N, on the ground-truth match, and
    on the audit, which is the acceptance test for this pipeline anyway
    (README section 12)."""
    import audit
    import metrics
    cases = []
    for size, dens, seed in ((39, 0.034, 1001), (39, 0.055, 1002),
                             (62, 0.047, 1003)):
        sim, adu = field(size, dens, seed)
        res = gsolve.detect(adu, sigma=SIGMA, offset=OFFSET, gain=GAIN,
                            k_max=12, verbose=0)
        d_e = (adu - OFFSET) / GAIN
        a = audit.audit_result(d_e, res.model_image, SIGMA)
        m = metrics.match(sim.positions, res.positions, radius=1.5)
        order = np.lexsort((res.positions[:, 1], res.positions[:, 0]))
        cases.append(dict(
            size=size, density=dens, seed=seed,
            raw_adu=np.round(adu).astype(int),
            truth_positions=sim.positions, truth_amplitudes=sim.amplitudes,
            N=len(res.positions),
            positions=res.positions[order], amplitudes=res.amplitudes[order],
            se=np.nan_to_num(res.se[order], nan=-1.0),
            # `background` is an (H, W) SURFACE in gsolve, not a scalar.
            background=np.asarray(res.background), lam=res.lam, A_s=res.A_s,
            n_passes=len(res.history),
            N_per_pass=[h["N"] for h in res.history],
            audit=dict(n_missed=a["n_missed"], n_piled=a["n_piled"],
                       z_min=a["z_min"], z_max=a["z_max"],
                       z_median=a["z_median"]),
            match=dict(n_true=m.n_true, n_est=m.n_est, precision=m.precision,
                       recall=m.recall, rmse=m.rmse)))
    return dict(
        what="gsolve.detect end to end on synthetic fields with truth",
        compare="NOT bit-exact, deliberately. Accept a port if, per case: "
                "|N - N_expected| <= 1; precision and recall each within 0.03; "
                "rmse within 0.02 px; audit n_missed and n_piled each within "
                "1. Positions are sorted by (y, x) so they can be matched "
                "pairwise for a spot check, but a 1-emitter difference "
                "renumbers everything after it -- match by nearest neighbour, "
                "not by index.",
        settings=dict(sigma=SIGMA, gain=GAIN, offset=OFFSET,
                      k_max=12, camera_offset=OFFSET),
        cases=cases)


def main():
    print("writing fixtures for the Rust port:")
    write("01_psf", fx_psf())
    write("02_lmga", fx_lmga())
    write("03_evidence", fx_evidence())
    # 04 and 05 cover score.py and the box tiling. NEITHER is on the gsolve
    # path -- they are kept only for boxsolve (see BOXSOLVE.md) and because
    # 04_score matters again if the projected score is revived for SPLIT
    # ranking. A gsolve port does not need to pass them.
    write("04_score", fx_score())
    write("05_boxes", fx_boxes())
    write("06_end_to_end", fx_end_to_end())
    print("\nport in this order; each layer is meaningless until the one "
          "below it passes.")


if __name__ == "__main__":
    main()
