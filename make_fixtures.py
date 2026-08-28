"""Generate language-neutral golden fixtures for the Rust port.

The Python verification scripts (`verify_*.py`) check this implementation
against analytic truth. They cannot check a *second* implementation against
*this* one, which is what a port needs, and eyeballing one solved frame is not
a regression test. This writes JSON the Rust side can read and assert against,
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

import gsolve
import calibrate
import evidence
import lmga
import psf
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
            theta=theta))
    return dict(
        what="evidence.logdet_cond / log_bf_add / log_bf_remove",
        invariant="log_bf_remove is the EXACT negation of log_bf_add on the "
                  "same pair; antisymmetry_residual must be 0.0 exactly, not "
                  "small. If it is not, the two paths have diverged.",
        constants=dict(COND_GUARD=evidence.COND_GUARD),
        compare="log BF to 1e-10 absolute; logdet to 1e-12 relative.",
        cases=cases)


# ---------------------------------------------------------------- layer 4
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
    write("04_end_to_end", fx_end_to_end())
    print("\nport in this order; each layer is meaningless until the one "
          "below it passes.")


if __name__ == "__main__":
    main()
