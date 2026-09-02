"""Post-detect physical PSF-width filtering.

The fixed-sigma detector answers the model-selection question under the
in-focus PSF model. This module then asks a different question of the survivors:
when each emitter is allowed to choose its own sigma, is that width physically
plausible for an in-focus point emitter?
"""

import numpy as np

from . import backend as backend_mod
from . import calibrate, core, evidence, moves, patches, psf
from .structs import DetectResult

INF_FOCUS_SIGMA_RATIO_MIN = 0.8
INF_FOCUS_SIGMA_RATIO_MAX = 1.2
VAR_SIGMA_RATIO_LO = 0.7
VAR_SIGMA_RATIO_HI = 8.0
VAR_SIGMA_MAX_ITER = 180

WIDTH_REJECT_DTYPE = np.dtype([
    ("source_index", np.int64),
    ("y", np.float64),
    ("x", np.float64),
    ("flux", np.float64),
    ("sigma", np.float64),
    ("sigma_ratio", np.float64),
    ("reason", "U12"),
])


def _fit_bounds(k, h, w, b_max, a_max, sigma_lo, sigma_hi):
    a_min = max(moves.A_MIN, core.A_MIN_REL * a_max)
    lo = [0.0]
    hi = [b_max]
    for _ in range(k):
        lo += [a_min, -0.5, -0.5, sigma_lo]
        hi += [a_max, h - 0.5, w - 0.5, sigma_hi]
    return np.asarray(lo), np.asarray(hi)


def _fit_var_sigma(be, sub, halo, theta0, sigma0, sigma_lo, sigma_hi):
    k = (len(theta0) - 1) // 4
    smax = max(float(sub.max()), 1.0)
    b_max = max(smax * 4.0, 10.0)
    a_max = 8.0 * smax / psf.peak_factor(sigma0)
    lo, hi = _fit_bounds(k, sub.shape[0], sub.shape[1], b_max, a_max,
                         sigma_lo, sigma_hi)
    th0 = np.clip(np.asarray(theta0, float), lo + 1e-9, hi - 1e-9)
    return be.fit_var_sigma(th0, sub.shape[0], sub.shape[1], sub, halo, lo, hi,
                            VAR_SIGMA_MAX_ITER)


def _log_bf_remove_var(full, reduced, k_full, sum_a_full, sum_a_reduced,
                       lam, a_s, sigma_width):
    """Positive favours the reduced variable-sigma model."""
    ld_full, ok_full = evidence.logdet(full.F)
    ld_reduced, ok_reduced = evidence.logdet(reduced.F)
    if not ok_reduced:
        return -np.inf
    if not ok_full:
        return np.inf
    add_log_bf = (
        (reduced.I - full.I)
        + np.log(lam)
        - np.log(k_full)
        - np.log(a_s)
        - (sum_a_full - sum_a_reduced) / a_s
        - np.log(sigma_width)
        + 2.0 * np.log(2.0 * np.pi)
        - 0.5 * (ld_full - ld_reduced)
    )
    return float(-add_log_bf)


def _remove_var_emitter(theta, k):
    b, a, cy, cx, sig = psf.unpack_var_sigma(theta)
    keep = np.ones(len(a), dtype=bool)
    keep[k] = False
    return psf.pack_var_sigma(b, a[keep], cy[keep], cx[keep], sig[keep])


def _var_amplitude_se(F, k):
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return None
    v = cov[1 + 4 * k, 1 + 4 * k]
    if not np.isfinite(v) or v <= 0:
        return None
    return float(np.sqrt(v))


def _fit_and_prune_patch(be, d_e, bmap, positions, amplitudes, patch, sigma0,
                         sigma_lo, sigma_hi, lam, a_s):
    yy, xx = patches.patch_grids(patch)
    sub = np.asarray(d_e[patch.y0:patch.y1, patch.x0:patch.x1])
    level, shape = core._window_bg(bmap, patch.y0, patch.x0, patch.y1,
                                   patch.x1)
    halo = (
        patches.build_halo_image(
            positions, amplitudes, patch.frozen_indices, sigma0, yy, xx,
            patch.y0, patch.x0)
        + shape
    )
    loc = positions[patch.indices] - np.array([patch.y0, patch.x0])
    theta = psf.pack_var_sigma(
        level, amplitudes[patch.indices], loc[:, 0], loc[:, 1],
        np.full(len(patch.indices), sigma0))
    fit = _fit_var_sigma(be, sub, halo, theta, sigma0, sigma_lo, sigma_hi)
    sources = [int(i) for i in patch.indices]

    while True:
        _, a, _, _, _ = psf.unpack_var_sigma(fit.theta)
        k = len(a)
        if k <= 1:
            break

        best = None
        for j in range(k):
            theta_reduced = _remove_var_emitter(fit.theta, j)
            reduced = _fit_var_sigma(be, sub, halo, theta_reduced, sigma0,
                                     sigma_lo, sigma_hi)
            _, a_red, _, _, _ = psf.unpack_var_sigma(reduced.theta)
            se_a = _var_amplitude_se(fit.F, j)
            if se_a is None or a[j] < core.PRUNE_TAU * se_a:
                log_bf = np.inf
            else:
                log_bf = _log_bf_remove_var(
                    fit, reduced, k, float(np.sum(a)), float(np.sum(a_red)),
                    lam, a_s, sigma_hi - sigma_lo)
            if log_bf == np.inf:
                best = (log_bf, j, reduced)
                break
            if np.isfinite(log_bf) and log_bf > 0:
                if best is None or log_bf > best[0]:
                    best = (log_bf, j, reduced)
        if best is None:
            break
        del sources[int(best[1])]
        fit = best[2]

    return fit, sources


def _variable_sigma_survivors(d_e, result, sigma_lo, sigma_hi, k_max, be):
    positions = np.atleast_2d(np.asarray(result.positions, float)).reshape(-1, 2)
    amplitudes = np.asarray(result.amplitudes, float).ravel()
    if len(amplitudes) == 0:
        empty = np.empty(0, dtype=int)
        return empty, positions, amplitudes, np.empty(0), empty

    pset = patches.build_patches(
        positions, result.sigma, d_e.shape,
        link_radius_factor=core.LINK_FACTOR,
        halo_radius_factor=core.HALO_FACTOR,
        bbox_pad_factor=core.BBOX_PAD,
        k_max=k_max)
    lam = max(float(result.lam), 1e-6)
    a_s = max(float(result.A_s), 1.0)

    source, pos, amp, sig = [], [], [], []
    for patch in pset:
        fit, sources = _fit_and_prune_patch(
            be, d_e, result.background, positions, amplitudes, patch,
            result.sigma, sigma_lo, sigma_hi, lam, a_s)
        _, a, cy, cx, sigma = psf.unpack_var_sigma(fit.theta)
        for j, src in enumerate(sources):
            source.append(src)
            pos.append((cy[j] + patch.y0, cx[j] + patch.x0))
            amp.append(a[j])
            sig.append(sigma[j])

    source = np.asarray(source, dtype=int)
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    amp = np.asarray(amp, dtype=float)
    sig = np.asarray(sig, dtype=float)
    pruned = np.setdiff1d(np.arange(len(amplitudes), dtype=int), source,
                          assume_unique=False)
    return source, pos, amp, sig, pruned


def _stamp_mask(shape, y, x, sigma, radius_factor):
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for cy, cx, sig in zip(y, x, sigma):
        radius = radius_factor * float(sig)
        r0 = int(np.ceil(radius))
        y0, y1 = max(0, int(cy) - r0), min(shape[0], int(cy) + r0 + 1)
        x0, x1 = max(0, int(cx) - r0), min(shape[1], int(cx) + r0 + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        d2 = ((yy[y0:y1, x0:x1] - cy) ** 2
              + (xx[y0:y1, x0:x1] - cx) ** 2)
        mask[y0:y1, x0:x1] |= d2 <= radius * radius
    return mask


def _render_variable_sigma(rows, shape):
    out = np.zeros(shape, dtype=float)
    for y, x, flux, sigma in rows:
        out += calibrate.render_model(
            np.array([[y, x]], dtype=float), np.array([flux], dtype=float),
            float(sigma), shape, 0.0)
    return out


def _background_for_final_refit(d_e, kept_pos, wide_rows, sigma):
    free = calibrate.emitter_free_mask(d_e.shape, kept_pos, sigma,
                                       core.BG_MASK_RADIUS)
    if len(wide_rows):
        free &= ~_stamp_mask(d_e.shape, wide_rows[:, 0], wide_rows[:, 1],
                             wide_rows[:, 3], core.BG_MASK_RADIUS)
    scalar = max(calibrate.robust_background(
        d_e, kept_pos, sigma, core.BG_MASK_RADIUS, free=free), core.BG_FLOOR)
    return core.background_map(d_e, kept_pos, sigma, fallback=scalar, free=free)


def _aggregate_rows(result):
    rec = result.aggregates
    if rec is None or len(rec) == 0:
        return np.empty((0, 4))
    return np.column_stack([rec["y"], rec["x"], rec["flux"], rec["sigma"]])


def _reject_records(source, pos, amp, sig, reason, sigma0):
    rec = np.empty(len(source), dtype=WIDTH_REJECT_DTYPE)
    rec["source_index"] = source
    rec["y"] = pos[:, 0] if len(pos) else np.empty(0)
    rec["x"] = pos[:, 1] if len(pos) else np.empty(0)
    rec["flux"] = amp
    rec["sigma"] = sig
    rec["sigma_ratio"] = sig / sigma0 if len(sig) else sig
    rec["reason"] = reason
    return rec


def filter_in_focus(data_img, result, offset=0.0, gain=None,
                    sigma_ratio_min=INF_FOCUS_SIGMA_RATIO_MIN,
                    sigma_ratio_max=INF_FOCUS_SIGMA_RATIO_MAX,
                    sigma_lo_ratio=VAR_SIGMA_RATIO_LO,
                    sigma_hi_ratio=VAR_SIGMA_RATIO_HI,
                    k_max=12, impl="rs"):
    """Reject emitters whose fitted sigma is not physically in-focus.

    The detector's fixed-sigma search is left intact. This function is a
    post-detect check:

    1. refit each final patch with one sigma per emitter;
    2. allow Bayes-factor pruning under that wider model;
    3. keep only `sigma_ratio_min <= sigma_fit / result.sigma <=
       sigma_ratio_max`;
    4. refit the surviving emitters at the fixed in-focus sigma.

    Broad rejects are rendered into the final background as a frozen nuisance
    field. Narrow rejects are dropped: rendering them back would preserve the
    pixel-scale overfit texture that made them fail the physical-width test.
    """
    if not (0.0 < sigma_lo_ratio <= sigma_ratio_min
            < sigma_ratio_max <= sigma_hi_ratio):
        raise ValueError(
            "expected 0 < sigma_lo_ratio <= sigma_ratio_min "
            "< sigma_ratio_max <= sigma_hi_ratio")

    be = backend_mod.get(impl)
    raw = np.asarray(data_img, dtype=float)
    g_eff = result.gain if gain is None else float(gain)
    d_e = (raw - offset) / g_eff

    sigma0 = float(result.sigma)
    sigma_lo = float(sigma_lo_ratio) * sigma0
    sigma_hi = float(sigma_hi_ratio) * sigma0
    source, var_pos, var_amp, var_sig, pruned = _variable_sigma_survivors(
        d_e, result, sigma_lo, sigma_hi, k_max, be)

    ratio = var_sig / sigma0 if len(var_sig) else var_sig
    keep = (ratio >= sigma_ratio_min) & (ratio <= sigma_ratio_max)
    too_narrow = ratio < sigma_ratio_min
    too_wide = ratio > sigma_ratio_max

    kept_pos = var_pos[keep]
    kept_amp = var_amp[keep]
    kept_sig = var_sig[keep]
    kept_source = source[keep]
    wide_rows = np.column_stack([
        var_pos[too_wide, 0], var_pos[too_wide, 1],
        var_amp[too_wide], var_sig[too_wide],
    ]) if np.any(too_wide) else np.empty((0, 4))
    frozen_rows = np.vstack([_aggregate_rows(result), wide_rows])

    background = _background_for_final_refit(d_e, kept_pos, frozen_rows, sigma0)
    nuisance = _render_variable_sigma(frozen_rows, raw.shape) if len(frozen_rows) \
        else np.zeros(raw.shape, dtype=float)
    bmap = background + nuisance
    if len(kept_amp):
        pos, amp, se = be.refine(d_e, kept_pos, kept_amp, sigma0, bmap, k_max,
                                 core.REFINE_SWEEPS)
    else:
        pos = np.empty((0, 2))
        amp = np.empty(0)
        se = np.empty((0, 3))

    rejects = [
        _reject_records(source[too_narrow], var_pos[too_narrow],
                        var_amp[too_narrow], var_sig[too_narrow],
                        "too_narrow", sigma0),
        _reject_records(source[too_wide], var_pos[too_wide],
                        var_amp[too_wide], var_sig[too_wide],
                        "too_wide", sigma0),
    ]
    width_rejects = np.concatenate(rejects) if rejects else \
        np.empty(0, dtype=WIDTH_REJECT_DTYPE)

    model = bmap + be.render_model(pos, amp, sigma0, raw.shape, 0.0)
    history = list(result.history)
    history.append(dict(stage="filter_in_focus",
                        n_before=int(len(result.amplitudes)),
                        n_var_pruned=int(len(pruned)),
                        n_too_narrow=int(np.sum(too_narrow)),
                        n_too_wide=int(np.sum(too_wide)),
                        n_after=int(len(amp))))
    return DetectResult(
        positions=pos, amplitudes=amp, sigma=sigma0,
        lam=max(len(amp) / float(raw.size), 1e-6),
        A_s=max(float(np.mean(amp)), 1.0) if len(amp) else result.A_s,
        gain=g_eff, background=bmap, n_outer_passes=result.n_outer_passes,
        model_image=model, residual=d_e - model, se=se, history=history,
        aggregates=result.aggregates,
        aggregate_fraction=result.aggregate_fraction,
        fit_sigma=kept_sig, sigma_ratio=kept_sig / sigma0 if len(kept_sig)
        else kept_sig,
        width_rejects=width_rejects,
        width_filter=dict(
            source_indices=kept_source,
            var_pruned_indices=pruned,
            sigma_ratio_min=float(sigma_ratio_min),
            sigma_ratio_max=float(sigma_ratio_max),
            sigma_lo=float(sigma_lo),
            sigma_hi=float(sigma_hi),
        ),
    )
