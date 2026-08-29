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
import pathlib

import numpy as np
import scipy.ndimage as ndi

import spotsolve
from spotsolve import calibrate
from spotsolve import evidence
from spotsolve import lmga
from spotsolve import psf
from spotsolve import simulate

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = str(REPO / "tests" / "fixtures")
SIGMA = 1.2
GAIN, OFFSET = 4.23, 100.0


def enc(o):
    """JSON-encode with full f64 precision."""
    # bool BEFORE int: Python's bool subclasses int, so the int branch would
    # claim True and write it as 1. A fixture that says 1 where it means true
    # is ambiguous to the port reading it.
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(f"{float(o):.17g}")
    if isinstance(o, (np.integer, int)):
        return int(o)
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


# ---------------------------------------------------------------- layer 0
def fx_filters():
    """The separable filters and order statistics `detect` needs.

    Numbered after the others because it was written last, but it sits BELOW
    all of them: nothing here depends on the algorithm, and a port that fails
    this layer cannot produce the right candidate list at any level above it.

    Every kernel is recorded as the filter's own response to a unit impulse
    rather than as scipy's private `_gaussian_kernel1d` output. One artifact
    then pins the truncation radius AND the normalization convention, which is
    the single trap in this layer: scipy normalizes the ORDER-0 kernel to sum
    1 and then applies the derivative recurrence, so a truncated order-2
    kernel does NOT sum to zero (the sigma=0.6 case below sums to -6.5e-2).
    A port that "fixes" that by re-normalizing shifts the whole LoG response
    against a fixed threshold of 1.5 and changes what is detected.
    """
    rng = np.random.default_rng(7)
    out = {"what": "separable filters and order statistics, below every "
                   "other layer"}

    # -- 1-D kernels, as impulse responses. n odd, impulse dead centre.
    n = 81
    imp = np.zeros(n)
    imp[n // 2] = 1.0
    kern = []
    for sigma, order in ((1.2, 0), (1.2, 2), (25 / 6.0, 0), (2.0, 1), (0.6, 2)):
        r = ndi.gaussian_filter1d(imp, sigma, order=order, mode="constant")
        nz = np.nonzero(np.abs(r) > 0)[0]
        kern.append(dict(sigma=sigma, order=order,
                         radius=int(4.0 * sigma + 0.5),
                         taps=r[nz.min():nz.max() + 1]))
    out["kernels1d"] = kern

    # -- 2-D filters on a field with structure the boundary modes bite on: a
    # ramp, so `nearest` and `reflect` differ at the edge, plus noise.
    H, W = 17, 23
    yy, xx = np.mgrid[0:H, 0:W] * 1.0
    img = 3.0 + 0.4 * yy - 0.2 * xx + rng.normal(scale=0.7, size=(H, W))
    out["image"] = dict(h=H, w=W, data=img)
    out["filters2d"] = [
        dict(op="gaussian_laplace", sigma=1.2, mode="reflect",
             result=ndi.gaussian_laplace(img, 1.2, mode="reflect")),
        dict(op="maximum_filter", size=5, mode="reflect",
             result=ndi.maximum_filter(img, size=5, mode="reflect")),
        dict(op="uniform_filter", size=25, mode="nearest",
             result=ndi.uniform_filter(img, size=25, mode="nearest")),
        dict(op="uniform_filter", size=3, mode="reflect",
             result=ndi.uniform_filter(img, size=3, mode="reflect")),
        dict(op="gaussian_filter", sigma=25 / 6.0, mode="nearest",
             result=ndi.gaussian_filter(img, 25 / 6.0, mode="nearest")),
    ]
    # A filter WIDER than the array, which `background_map` hits on a small
    # frame (it clamps `kernel` to min(H, W)//3, but the gaussian that follows
    # is not clamped) and which is where a naive index reflection goes wrong.
    small = img[:5, :7].copy()
    out["oversize"] = dict(
        h=5, w=7, data=small,
        uniform_9_nearest=ndi.uniform_filter(small, size=9, mode="nearest"),
        maximum_9_reflect=ndi.maximum_filter(small, size=9, mode="reflect"),
        gaussian_3_nearest=ndi.gaussian_filter(small, 3.0, mode="nearest"))

    # -- order statistics, in numpy's linear-interpolation convention.
    v = rng.normal(size=97) * 10 + 3
    out["stats"] = dict(
        data=v,
        median=float(np.median(v)),
        p10=float(np.percentile(v, 10.0)),
        p15_9=float(np.percentile(v, 15.9)),
        p84_1=float(np.percentile(v, 84.1)),
        q20=float(np.quantile(v, 0.2)),
        # An even count exercises the midpoint average.
        median_even=float(np.median(v[:96])),
        p10_even=float(np.percentile(v[:96], 10.0)))

    # -- the calibrate helpers built on top of them.
    sim, raw = field(48, 0.004, 3)
    d_e = (raw - OFFSET) / GAIN
    out["calibrate"] = dict(
        h=48, w=48, raw=raw, d_e=d_e, sigma=SIGMA,
        positions=sim.positions,
        estimate_gain=float(calibrate.estimate_gain(raw, OFFSET)),
        robust_background_global=float(calibrate.robust_background(d_e)),
        robust_background_masked=float(calibrate.robust_background(
            d_e, sim.positions, SIGMA, 3.0)),
        robust_spread=float(calibrate.robust_spread(d_e - np.median(d_e))))
    out["compare"] = ("kernels and filtered images to 1e-12 absolute; the "
                      "order statistics to 1e-12; `estimate_gain` to 1e-9 "
                      "relative. These are pure arithmetic on identical "
                      "inputs -- a port that only matches to 1e-6 has a "
                      "convention wrong, not a rounding difference.")
    return out


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

    This is the acceptance fixture, and it is built from `spotsolve` -- the port
    target. It does NOT expect bit-equality: a port that fans `refine`'s patch
    sweep out across threads (PORTING_NOTES section 14) legitimately lands
    somewhere slightly different. Judge it on N, on the ground-truth match, and
    on the audit, which is the acceptance test for this pipeline anyway
    (README section 12)."""
    from spotsolve import audit
    from spotsolve import metrics
    cases = []
    for size, dens, seed in ((39, 0.034, 1001), (39, 0.055, 1002),
                             (62, 0.047, 1003)):
        sim, adu = field(size, dens, seed)
        res = spotsolve.detect(adu, sigma=SIGMA, offset=OFFSET, gain=GAIN,
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
            # `background` is an (H, W) SURFACE in spotsolve, not a scalar.
            background=np.asarray(res.background), lam=res.lam, A_s=res.A_s,
            n_passes=len(res.history),
            N_per_pass=[h["N"] for h in res.history],
            audit=dict(n_missed=a["n_missed"], n_piled=a["n_piled"],
                       z_min=a["z_min"], z_max=a["z_max"],
                       z_median=a["z_median"]),
            match=dict(n_true=m.n_true, n_est=m.n_est, precision=m.precision,
                       recall=m.recall, rmse=m.rmse)))
    return dict(
        what="spotsolve.detect end to end on synthetic fields with truth",
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



# ---------------------------------------------------------------- layer 5
def fx_geometry():
    """Patch decomposition, window selection, model rendering and the emitter
    free-mask -- everything that answers "which emitters, over which pixels".

    This layer has no Python verification of its own beyond
    `verify_geometry.py`, and it is the layer the port CHANGES: `cKDTree` +
    `connected_components` become a uniform grid plus union-find, and the
    O(N*H*W) mask loop becomes an O(N*sigma^2) stamp. Same answers, different
    algorithm -- which is exactly when a golden fixture earns its keep."""
    from spotsolve import moves
    from spotsolve import patches as patch_mod
    r = np.random.default_rng(31)
    cases = []
    for H, W, n, seed in ((48, 52, 12, 31), (39, 39, 40, 32), (24, 30, 3, 33)):
        rr = np.random.default_rng(seed)
        pos = np.stack([rr.uniform(2, H - 2, n), rr.uniform(2, W - 2, n)], axis=1)
        amp = rr.uniform(300.0, 1800.0, n)

        ps = patch_mod.build_patches(pos, SIGMA, (H, W),
                                     link_radius_factor=spotsolve.LINK_FACTOR,
                                     halo_radius_factor=spotsolve.HALO_FACTOR,
                                     bbox_pad_factor=spotsolve.BBOX_PAD, k_max=12)
        patches_out = [dict(indices=np.sort(p.indices),
                            frozen_indices=np.sort(p.frozen_indices),
                            y0=p.y0, x0=p.x0, y1=p.y1, x1=p.x1) for p in ps]

        # `_window` at a few probe points: on top of an emitter, between two,
        # and in empty space near the rim (where the bbox clamps).
        probes = [pos[0], 0.5 * (pos[0] + pos[1]), np.array([1.0, 1.0]),
                  np.array([H - 1.5, W - 1.5])]
        windows = []
        for c in probes:
            free, frozen, bbox = spotsolve.core._window(pos, np.asarray(c, float),
                                                SIGMA, (H, W), 12)
            windows.append(dict(cand=c, free=free, frozen=np.sort(frozen),
                                y0=bbox[0], x0=bbox[1], y1=bbox[2], x1=bbox[3]))

        # The emitter free-mask, exactly as background_map and
        # calibrate.robust_background build it. Moves to Rust because it is
        # O(N*H*W) as written; the convolutions around it do not.
        yy, xx = np.mgrid[0:H, 0:W]
        free_mask = np.ones((H, W), dtype=bool)
        r2 = (spotsolve.BG_MASK_RADIUS * SIGMA) ** 2
        for cy, cx in pos:
            free_mask &= ((yy - cy) ** 2 + (xx - cx) ** 2) > r2

        # `moves.residual_axis` on a genuine unresolved pair: render two
        # emitters 1.2 sigma apart along a known axis, fit ONE in their place,
        # and take the residual. That is exactly the state SPLIT exists for --
        # no peak for a LoG filter to find, but a clear quadrupole.
        mv = []
        for ang in (0.0, 0.7, 1.9):
            uy, ux = np.cos(ang), np.sin(ang)
            cy0, cx0 = H / 2.0, W / 2.0
            d = 1.2 * SIGMA
            yy, xx = np.mgrid[0:H, 0:W] * 1.0
            pair = psf.model(psf.pack(0.0, [800.0, 800.0],
                                      [cy0 + 0.5 * d * uy, cy0 - 0.5 * d * uy],
                                      [cx0 + 0.5 * d * ux, cx0 - 0.5 * d * ux]),
                             yy, xx, SIGMA)
            one = psf.pack(0.0, [1600.0], [cy0], [cx0])
            resid = pair - psf.model(one, yy, xx, SIGMA)
            u, strength = moves.residual_axis(one, 0, yy, xx, SIGMA, resid)
            for disp in spotsolve.SPLIT_DISPS:
                mv.append(dict(angle=ang, theta=one, resid=resid,
                               u=u, strength=strength, disp=disp,
                               split=moves.split(one, 0, u, disp * SIGMA)))
        cases.append(dict(
            H=H, W=W, sigma=SIGMA, positions=pos, amplitudes=amp, moves=mv,
            k_max=12, link_factor=spotsolve.LINK_FACTOR,
            halo_factor=spotsolve.HALO_FACTOR, bbox_pad=spotsolve.BBOX_PAD,
            patches=patches_out, windows=windows,
            mask_radius=spotsolve.BG_MASK_RADIUS,
            free_mask=free_mask.astype(int),
            render_truncate=4.0,
            render=calibrate.render_model(pos, amp, SIGMA, (H, W), 0.0)))
    return dict(
        what="patches.build_patches / spotsolve.core._window / the emitter free-mask / "
             "calibrate.render_model",
        moves_note="`moves` holds residual_axis/split on a residual left by "
                   "fitting ONE emitter where two sit 1.2 sigma apart -- the "
                   "state SPLIT exists for. `u` is a principal axis, defined "
                   "only up to SIGN: a port may return -u, in which case its "
                   "`split` lists the two children in the other order. Compare "
                   "|u . u_expected| = 1 and the child positions as a SET.",
        layout="`indices` and `frozen_indices` are SORTED here so a port using "
               "a different traversal order can compare them as sets. "
               "`windows[i].free` is NOT sorted -- it is ordered by distance "
               "to the candidate, and that order is load-bearing: it decides "
               "the theta layout of the fit. free_mask is (H,W) as 0/1.",
        compare="exact. These are integer indices and integer bboxes; there is "
                "no tolerance to spend. `render` is float, relative 1e-13 -- it "
                "truncates each emitter at 4 sigma, so a port must truncate at "
                "the same radius or it will disagree in the 5th digit at the "
                "patch rim, not the 15th.",
        cases=cases)


# ---------------------------------------------------------------- layer 6
def _mid_search_state(size, dens, seed, rounds):
    """Run `detect`'s round loop for `rounds` rounds and hand back its state.

    The passes must be exercised on a state the search actually reaches. On an
    empty model every window is trivial, nothing splits and nothing prunes, so
    a fixture built from round 0 would assert almost nothing."""
    sim, adu = field(size, dens, seed)
    d_e = (adu - OFFSET) / GAIN
    H, W = d_e.shape
    b0 = float(np.percentile(d_e, 10.0))
    bmap = np.full((H, W), max(b0, spotsolve.BG_FLOOR))
    A_s = max(float(d_e.max()) - b0, 10.0) / psf.peak_factor(SIGMA)
    lam = 0.02
    positions, amplitudes = np.empty((0, 2)), np.empty(0)
    for _ in range(rounds):
        model = spotsolve.render(positions, amplitudes, SIGMA, bmap)
        cand, camp, _ = spotsolve.find_candidates(d_e, model, SIGMA, positions,
                                               spotsolve.CAND_THRESHOLD)
        for c, a in zip(cand, camp):
            if len(positions) and np.min(
                    np.linalg.norm(positions - c, axis=1)) <= SIGMA:
                continue
            _, positions, amplitudes = spotsolve.core._try_add(
                d_e, positions, amplitudes, bmap, c, a, SIGMA, lam, A_s, 12)
        positions, amplitudes, _ = spotsolve.refine(
            d_e, positions, amplitudes, SIGMA, bmap, k_max=12, max_sweeps=1)
        lam = max(len(positions) / float(H * W), 1e-6)
        if len(amplitudes):
            A_s = max(float(np.mean(amplitudes)), 1.0)
        bmap = spotsolve.core._update_bg(d_e, positions, amplitudes, SIGMA, bmap,
                                 spotsolve.BG_KERNEL)
    return sim, d_e, positions, amplitudes, bmap, lam, A_s


def fx_passes():
    """The four passes that ARE the port's API boundary: ADD over a candidate
    list, SPLIT, PRUNE, and one REFINE sweep.

    Each is captured from a state one round into a real search, with the full
    input it was handed. This is where a failure has to localize: above it lies
    only `detect`'s round loop, which stays in Python."""
    cases = []
    for size, dens, seed in ((39, 0.055, 1002), (39, 0.034, 1001)):
        sim, d_e, pos, amp, bmap, lam, A_s = _mid_search_state(
            size, dens, seed, rounds=1)
        k_max = 12

        # ADD: the candidate list `detect` would produce at this state, and the
        # positions/amplitudes after the whole pass over it. The in-loop
        # proximity re-check is part of the pass, not of the driver -- it reads
        # the positions an earlier acceptance in the SAME pass wrote.
        model = spotsolve.render(pos, amp, SIGMA, bmap)
        cand, camp, _ = spotsolve.find_candidates(d_e, model, SIGMA, pos,
                                               spotsolve.CAND_THRESHOLD)
        p_add, a_add, n_added = pos.copy(), amp.copy(), 0
        for c, a in zip(cand, camp):
            if len(p_add) and np.min(
                    np.linalg.norm(p_add - c, axis=1)) <= SIGMA:
                continue
            ok, p_add, a_add = spotsolve.core._try_add(d_e, p_add, a_add, bmap, c, a,
                                               SIGMA, lam, A_s, k_max)
            n_added += int(ok)

        # SPLIT: run against the model the adds produced, as `detect` does.
        model_s = spotsolve.render(p_add, a_add, SIGMA, bmap)
        p_spl, a_spl, n_split = spotsolve.core._split_pass(
            d_e, p_add.copy(), a_add.copy(), bmap, SIGMA, lam, A_s, k_max,
            model_s)

        # REFINE: one sweep, which is what the round loop uses.
        p_ref, a_ref, se_ref = spotsolve.refine(
            d_e, p_spl.copy(), a_spl.copy(), SIGMA, bmap, k_max=k_max,
            max_sweeps=1)

        # PRUNE: one pass, faintest first, with write-back onto survivors.
        p_prn, a_prn = spotsolve.core._prune(d_e, p_ref.copy(), a_ref.copy(), bmap,
                                     SIGMA, lam, A_s, k_max)

        cases.append(dict(
            size=size, density=dens, seed=seed, sigma=SIGMA, k_max=k_max,
            lam=lam, A_s=A_s,
            d_e=d_e, bmap=bmap,
            positions=pos, amplitudes=amp,
            add=dict(cand=cand, camp=camp, n_added=n_added,
                     positions=p_add, amplitudes=a_add),
            split=dict(model=model_s, n_split=n_split,
                       positions=p_spl, amplitudes=a_spl),
            refine=dict(max_sweeps=1, positions=p_ref, amplitudes=a_ref,
                        se=np.nan_to_num(se_ref, nan=-1.0)),
            prune=dict(prune_tau=spotsolve.PRUNE_TAU,
                       positions=p_prn, amplitudes=a_prn)))
    return dict(
        what="spotsolve's four passes -- the ADD loop over a candidate list, "
             "_split_pass, one refine sweep, and one _prune pass -- each from "
             "a state one round into a real search",
        order="Run them in the order given: ADD's output is SPLIT's input, "
              "SPLIT's is REFINE's, REFINE's is PRUNE's. That is `detect`'s "
              "own order and the passes are not independent of it.",
        compare="positions to 1e-6 px and amplitudes to 1e-4 relative, per "
                "emitter, ONLY while the counts match. If a count differs the "
                "port has made a different decision somewhere and pairwise "
                "comparison is meaningless -- chase that first, in the layer "
                "below. `se` is -1.0 where the Python reported NaN.",
        cases=cases)


def main():
    print("writing fixtures for the Rust port:")
    write("01_psf", fx_psf())
    write("02_lmga", fx_lmga())
    write("03_evidence", fx_evidence())
    write("04_end_to_end", fx_end_to_end())
    write("05_geometry", fx_geometry())
    write("06_passes", fx_passes())
    write("07_filters", fx_filters())
    print("\nport in this order; each layer is meaningless until the one "
          "below it passes.")


if __name__ == "__main__":
    main()
