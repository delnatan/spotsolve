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

from spotsolve import psf
from spotsolve import simulate
from spotsolve.deprecated import calibrate
from spotsolve.deprecated import core
from spotsolve.deprecated import lmga

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = str(REPO / "tests" / "fixtures")
SIGMA = 1.2
# The confocal simulation's own in-focus width, fitted to the pixel-binned
# stamp it deposits (`scripts/psf_sigma_scan.py`). Not the same instrument as
# `SIGMA` above, which is the Gaussian arm's.
SIM_SIGMA = 0.818
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

    # The 4K+1 layout, which the free width fits and the Rust core already
    # carries as `pack_var` / `model_and_jac_var_sigma_ax`.
    var_cases = []
    for K, seed in ((1, 5), (3, 6)):
        r = np.random.default_rng(seed)
        h, w = 13, 11
        yy, xx = np.mgrid[0:h, 0:w] * 1.0
        sig = r.uniform(0.8, 2.0, K) * SIGMA
        theta = psf.pack_var_sigma(r.uniform(2, 8), r.uniform(300, 1800, K),
                                   r.uniform(2, h - 2, K),
                                   r.uniform(2, w - 2, K), sig)
        m = psf.model_var_sigma(theta, yy, xx)
        J = psf.jac_var_sigma(theta, yy, xx)
        var_cases.append(dict(K=K, h=h, w=w, theta=theta, sigmas=sig,
                              model=m, jac_flat=J.reshape(-1, J.shape[-1])))

    return dict(
        what="psf.model / psf.jac on a (h,w) pixel-centre grid, fixed and "
             "per-emitter width",
        layout="theta = [b, A_0, y_0, x_0, A_1, ...]; jac_flat is "
               "(h*w, 3K+1) in C order, i.e. row-major over (y, x). "
               "var_sigma_cases use [b, A_0, y_0, x_0, s_0, A_1, ...] and "
               "(h*w, 4K+1); the sigma column is the LAST of each emitter's "
               "four, and a port that orders it first will pass this layer's "
               "`model` and fail its `jac_flat`.",
        compare="relative 1e-13 elementwise. NOT bit-exact across libm "
                "implementations -- erf is the reason.",
        cases=cases,
        var_sigma_cases=var_cases)


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
    # Free widths, maximum likelihood.
    var_cases = []
    s0 = SIGMA
    for K, seed in ((1, 31), (2, 32)):
        r = np.random.default_rng(seed)
        h = w = 15
        yy, xx = np.mgrid[0:h, 0:w] * 1.0
        sig = r.uniform(0.95, 1.7, K) * s0
        th = psf.pack_var_sigma(r.uniform(2, 8), r.uniform(300, 1800, K),
                                r.uniform(4, 11, K), r.uniform(4, 11, K), sig)
        d = r.poisson(np.maximum(psf.model_var_sigma(th, yy, xx),
                                 1e-9)).astype(float)
        lo = np.array([0.0] + [1e-4, -0.5, -0.5, 0.70 * s0] * K)
        hi = np.array([max(d.max() * 4, 10.0)]
                      + [8 * max(d.max(), 1) / psf.peak_factor(s0) * 2.2 ** 2,
                         h - 0.5, w - 0.5, 2.2 * s0] * K)
        t0 = np.clip(th * np.append([1.0], np.tile([1.1, 1.0, 1.0, 1.0], K)),
                     lo + 1e-9, hi - 1e-9)
        res = lmga.fit(t0, yy, xx, s0, d, lo, hi, max_iter=100,
                       free_sigma="per_emitter")
        var_cases.append(dict(K=K, h=h, w=w, sigma=s0, data=d, theta0=t0,
                              lower=lo, upper=hi,
                              ml=dict(theta=res.theta, I=res.I, F=res.F,
                                      converged=res.converged,
                                      stalled=res.stalled)))

    return dict(
        what="lmga.fit: bounded Fisher-scoring LM on the Poisson "
             "I-divergence, at fixed and per-emitter width",
        compare="I to 1e-8 ABSOLUTE (nats -- the unit decisions are made in); "
                "theta to 1e-6 px / 1e-4 relative on amplitude; F to 1e-9 "
                "relative. Do NOT assert on n_iter: the gain-ratio branch "
                "makes it sensitive to the last ulp of the model.",
        cases=cases,
        var_sigma_cases=var_cases)


def _grad(theta, yy, xx, sigma, d):
    m, J = psf.model_and_jac(theta, yy, xx, sigma)
    m = np.maximum(m, 1e-9).reshape(-1)
    J = J.reshape(-1, theta.size)
    return J.T @ ((m - d.reshape(-1)) / m)


# ---------------------------------------------------------------- layer 5
def fx_geometry():
    """Patch decomposition, model rendering and the emitter free-mask -- everything that answers "which emitters, over which pixels".

    This layer has no Python verification of its own beyond
    `verify_geometry.py`, and it is the layer the port CHANGES: `cKDTree` +
    `connected_components` become a uniform grid plus union-find, and the
    O(N*H*W) mask loop becomes an O(N*sigma^2) stamp. Same answers, different
    algorithm -- which is exactly when a golden fixture earns its keep."""
    from spotsolve.deprecated import patches as patch_mod
    cases = []
    for H, W, n, seed in ((48, 52, 12, 31), (39, 39, 40, 32), (24, 30, 3, 33)):
        rr = np.random.default_rng(seed)
        pos = np.stack([rr.uniform(2, H - 2, n), rr.uniform(2, W - 2, n)], axis=1)
        amp = rr.uniform(300.0, 1800.0, n)

        ps = patch_mod.build_patches(pos, SIGMA, (H, W),
                                     link_radius_factor=core.LINK_FACTOR,
                                     halo_radius_factor=core.HALO_FACTOR,
                                     bbox_pad_factor=core.BBOX_PAD, k_max=12)
        patches_out = [dict(indices=np.sort(p.indices),
                            frozen_indices=np.sort(p.frozen_indices),
                            y0=p.y0, x0=p.x0, y1=p.y1, x1=p.x1) for p in ps]

        # The emitter free-mask, exactly as background_map and
        # calibrate.robust_background build it. Moves to Rust because it is
        # O(N*H*W) as written; the convolutions around it do not.
        yy, xx = np.mgrid[0:H, 0:W]
        free_mask = np.ones((H, W), dtype=bool)
        r2 = (core.BG_MASK_RADIUS * SIGMA) ** 2
        for cy, cx in pos:
            free_mask &= ((yy - cy) ** 2 + (xx - cx) ** 2) > r2

        cases.append(dict(
            H=H, W=W, sigma=SIGMA, positions=pos, amplitudes=amp,
            k_max=12, link_factor=core.LINK_FACTOR,
            halo_factor=core.HALO_FACTOR, bbox_pad=core.BBOX_PAD,
            patches=patches_out,
            mask_radius=core.BG_MASK_RADIUS,
            free_mask=free_mask.astype(int),
            render_truncate=4.0,
            render=calibrate.render_model(pos, amp, SIGMA, (H, W), 0.0)))
    return dict(
        what="patches.build_patches / the emitter free-mask / "
             "calibrate.render_model",
        layout="`indices` and `frozen_indices` are SORTED here so a port using "
               "a different traversal order can compare them as sets. "
               "free_mask is (H,W) as 0/1.",
        compare="exact. These are integer indices and integer bboxes; there is "
                "no tolerance to spend. `render` is float, relative 1e-13 -- it "
                "truncates each emitter at 4 sigma, so a port must truncate at "
                "the same radius or it will disagree in the 5th digit at the "
                "patch rim, not the 15th.",
        cases=cases)


def main():
    print("writing fixtures for the Rust port:")
    write("01_psf", fx_psf())
    write("02_lmga", fx_lmga())
    write("05_geometry", fx_geometry())
    write("07_filters", fx_filters())
    print("\nport in this order; each layer is meaningless until the one "
          "below it passes.")


if __name__ == "__main__":
    main()
