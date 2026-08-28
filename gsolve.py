"""Emitter detection and localization: FIND -> SPLIT -> PRUNE, with a Laplace
Bayes factor as the accept rule and a bounded Poisson MLE as the fit.

Pipeline
--------
    d_e = (raw - offset) / gain              photoelectrons; everything below
                                             assumes these units
    repeat until a round changes nothing:
        FIND     LoG peaks on the variance-normalized residual  -> candidates
        ADD      fit K and K+1 on the candidate's window, accept on log BF > 0
        SPLIT    for each emitter, propose two along the residual quadrupole
                 axis, accept on log BF > 0
        REFINE   joint re-fit at fixed N, then re-estimate the background
    REFINE                                   settle before removal is scored
    repeat until nothing is removed:
        PRUNE    per-emitter removal test, faintest first
        REFINE

The loop is monotone in N: ADD and SPLIT only increase it, PRUNE only
decreases it, and they do not alternate. There is no fixed point to chase and
no move that can undo another, so termination is structural.

Layers this depends on
----------------------
    psf         pixel-integrated Gaussian, model and Jacobian
    lmga        bounded Poisson-MLE optimizer (Coleman-Li affine scaling)
    evidence    Laplace log Bayes factor and its conditioning guard
    patches     grouping of emitters into jointly-fittable patches
    moves       proposal constructors (pure theta -> theta transformations)
    calibrate   gain, robust background, model rendering

Model selection is confined to `_try_add`, `_try_split` and `_prune`. `refine`
proposes nothing and is the only place the reported parameters and standard
errors are produced.
"""

import numpy as np
import scipy.ndimage as ndi

import calibrate
import evidence
import lmga
import moves
import msearch
import patches as patch_mod
import psf
from structs import DetectResult

__all__ = ["detect", "find_candidates", "background_map", "refine"]


LINK_FACTOR = 2.5   # emitters within this many sigma are fitted jointly
HALO_FACTOR = 5.0   # ... and beyond it are frozen into the model as a constant
BBOX_PAD = 3.0      # sigma of pixel context around a group's extreme emitters
CAND_THRESHOLD = calibrate.LOG_SEED_THRESHOLD

PRUNE_TAU = 2.0
# A / SE(A), measured on the joint fit, below which `_prune` removes an emitter
# outright instead of scoring it. Two roles:
#
#   - it is the pipeline's precision/recall dial, and the only one left;
#   - it forces removal where the Laplace evidence cannot be computed. A
#     collapsed pair's Fisher matrix is singular along its separation
#     direction, so var(A) diverges and A/SE goes to zero on its own; the
#     Bayes factor there would argue to KEEP the pair, harder the more
#     degenerate it is.
#
# Raising it costs localization as well as recall: removing one member of a
# real close pair leaves the survivor absorbing both fluxes and sitting between
# them. 2.0 is the measured optimum on both bead frames.

SPLIT_DISPS = (1.0, 1.6)
# Displacements, in sigma, at which a split is proposed. Below ~1 sigma a pair
# is not identifiable and the evidence refuses it; beyond ~2 sigma FIND already
# produces a separate LoG peak and the split is redundant.

BG_KERNEL = 25          # side, px, of the window the background surface uses
BG_FLOOR = 1e-3         # W = 1/m is singular at m = 0; far below one e-
BG_MASK_RADIUS = 3.0    # sigma; emitter support excluded from the estimate
BG_MIN_PIXELS = 25      # unmasked pixels a window needs before it is believed

REFINE_SWEEPS = 4       # see `refine`
REFINE_TOL = 1e-3       # px; position shift below which a sweep is a no-op


def background_map(d_e, positions, sigma, kernel=BG_KERNEL,
                   radius_factor=BG_MASK_RADIUS, fallback=None):
    """Local background surface, in photoelectrons, from the pixels no emitter
    reaches.

    Estimated from masked DATA, never from the PSF-subtracted residual: the
    residual route is a feedback loop, because emitters that have absorbed
    background depress it, the surface follows them down, and they must absorb
    more.

    An undetected emitter is not masked, so the mean is taken twice with a
    one-sided Poisson sigma-clip between. The contamination is always positive,
    so clipping the upper tail only is correct; clipping both biases the
    estimate down.

    Windows left with fewer than `BG_MIN_PIXELS` pixels get `fallback`, the
    global robust value. At high density that is most of them -- the surface
    degrades to a scalar rather than to an average over four pixels.
    """
    d = np.asarray(d_e, float)
    H, W = d.shape
    k = int(min(kernel, max(3, min(H, W) // 3)))
    if k % 2 == 0:
        k += 1
    if fallback is None:
        fallback = calibrate.robust_background(d, positions, sigma)

    free = np.ones((H, W), dtype=bool)
    if positions is not None and len(positions):
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = (radius_factor * sigma) ** 2
        for cy, cx in np.atleast_2d(np.asarray(positions, float)):
            free &= ((yy - cy) ** 2 + (xx - cx) ** 2) > r2

    def local_mean(mask):
        num = ndi.uniform_filter(np.where(mask, d, 0.0), size=k, mode="nearest")
        den = ndi.uniform_filter(mask.astype(float), size=k, mode="nearest")
        return np.divide(num, np.maximum(den, 1e-9)), den * k * k

    b, _ = local_mean(free)
    keep = free & (d <= b + 3.0 * np.sqrt(np.maximum(b, BG_FLOOR)))
    b, n = local_mean(keep)

    b = np.where(n >= BG_MIN_PIXELS, b, fallback)
    b = ndi.gaussian_filter(b, k / 6.0, mode="nearest")
    return np.maximum(b, BG_FLOOR)


def render(positions, amplitudes, sigma, bmap):
    """Full model image: the background surface plus every emitter's PSF."""
    bmap = np.asarray(bmap, float)
    return bmap + calibrate.render_model(positions, amplitudes, sigma,
                                         bmap.shape, 0.0)


def _window_bg(bmap, y0, x0, y1, x1):
    """Split this window's background into (level, shape).

    The fit keeps its free scalar `b`, started at `level`; `shape` is the
    background's variation across the window and enters as a known additive
    term, like a frozen emitter.

    The split keeps `b` strictly interior. Folding the whole surface into the
    known term would leave `b` wanting to be 0, which is its lower bound, and
    `lmga`'s Coleman-Li scaling collapses every coordinate's step when any one
    parameter sits on a bound.
    """
    win = bmap[y0:y1, x0:x1]
    level = float(np.median(win))
    return level, win - level


def _refine_sweep(d_e, positions, amplitudes, sigma, bmap, k_max,
                  link_radius_factor, max_iter):
    """One Gauss-Seidel pass over the patch decomposition."""
    se = np.full((len(amplitudes), 3), np.nan)
    out_pos = positions.copy()
    out_amp = amplitudes.copy()
    pset = patch_mod.build_patches(positions, sigma, d_e.shape,
                                   link_radius_factor=link_radius_factor,
                                   k_max=k_max)
    for p in pset:
        yy, xx = patch_mod.patch_grids(p)
        halo = patch_mod.build_halo_image(positions, amplitudes,
                                          p.frozen_indices, sigma, yy, xx,
                                          p.y0, p.x0)
        level, shape_ = _window_bg(bmap, p.y0, p.x0, p.y1, p.x1)
        halo = halo + shape_
        sub = np.asarray(d_e[p.y0:p.y1, p.x0:p.x1])
        loc = positions[p.indices] - np.array([p.y0, p.x0])
        theta0 = psf.pack(level, amplitudes[p.indices], loc[:, 0], loc[:, 1])
        r = _fit_window(sub, yy, xx, sigma, halo, theta0, max_iter=max_iter)
        _, A, cy, cx = psf.unpack(r.theta)
        out_amp[p.indices] = A
        out_pos[p.indices, 0] = cy + p.y0
        out_pos[p.indices, 1] = cx + p.x0
        try:
            var = np.diag(np.linalg.inv(r.F))
        except np.linalg.LinAlgError:
            continue
        v = np.where(var > 0, var, np.nan)
        se[p.indices, 0] = np.sqrt(v[1::3])
        se[p.indices, 1] = np.sqrt(v[2::3])
        se[p.indices, 2] = np.sqrt(v[3::3])
    return out_pos, out_amp, se


def refine(d_e, positions, amplitudes, sigma, bmap, k_max=12,
           link_radius_factor=LINK_FACTOR, max_iter=200,
           max_sweeps=REFINE_SWEEPS, tol=REFINE_TOL):
    """Joint re-fit at fixed N in connected groups, plus per-emitter CRLBs.

    Returns (positions, amplitudes, se) with `se` an (N,3) array of
    (SE_A, SE_y, SE_x) from the Fisher matrix of the fit whose parameters are
    reported. The background is not returned: it is a surface owned by the
    caller, re-estimated between rounds.

    Iterated to a fixed point, not run once. Each patch fit holds its
    out-of-patch neighbours frozen at the positions and amplitudes it was
    handed, so a single pass propagates whatever staleness those carry into
    the emitter it surrounds. Measured on isolated emitters at density 0.055,
    a 0.5 px error in the NEIGHBOURS alone (target started at truth) takes the
    pull sd from 0.96 to 1.98; sweeping to convergence recovers it to 1.27.
    The patch decomposition is rebuilt each sweep, which is what refreshes the
    halo.

    `max_sweeps=1` reproduces the single-pass behaviour and is what the round
    loop uses, since the round loop iterates anyway.
    """
    positions = np.atleast_2d(np.asarray(positions, float))
    amplitudes = np.asarray(amplitudes, float).ravel()
    if len(amplitudes) == 0:
        return positions, amplitudes, np.empty((0, 3))

    se = np.full((len(amplitudes), 3), np.nan)
    for _ in range(max(1, int(max_sweeps))):
        prev = positions
        positions, amplitudes, se = _refine_sweep(
            d_e, positions, amplitudes, sigma, bmap, k_max,
            link_radius_factor, max_iter)
        if np.max(np.abs(positions - prev)) < tol:
            break
    return positions, amplitudes, se


def find_candidates(d_e, model, sigma, positions, threshold=CAND_THRESHOLD):
    """LoG peaks on the variance-normalized residual, brightest first.

    Runs on the residual of the current model, not on the image, so an emitter
    already modelled is not proposed again; the proximity veto below catches
    what survives on a neighbour's wing.

    Normalizing by sqrt(model) before the filter is what makes one threshold
    valid across the frame: under Poisson noise the residual's scale is the
    square root of the mean, so the ratio is on a fixed sigma scale wherever
    the model puts flux.
    """
    resid = d_e - model
    nr = resid / np.sqrt(np.maximum(model, 1e-6))
    log_f = -ndi.gaussian_laplace(nr, sigma)
    win = 2 * int(np.ceil(sigma)) + 1
    peaks = (log_f == ndi.maximum_filter(log_f, size=win)) & (log_f > threshold)
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0)

    cand = np.stack([ys, xs], axis=1).astype(float)
    strength = log_f[ys, xs]
    if len(positions):
        d = np.min(np.linalg.norm(
            cand[:, None, :] - np.atleast_2d(positions)[None, :, :], axis=-1),
            axis=1)
        keep = d > sigma
        cand, strength = cand[keep], strength[keep]
        ys, xs = ys[keep], xs[keep]
    if len(cand) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0)

    amp = np.maximum(resid[ys, xs], 1e-2) / psf.peak_factor(sigma)
    order = np.argsort(-strength)
    return cand[order], amp[order], strength[order]


def _window(positions, cand, sigma, shape, k_max):
    """(free_indices, frozen_indices, bbox) for the local fit around `cand`.

    Free: existing emitters close enough that adding `cand` changes their
    estimates, capped at `k_max - 1` nearest so the joint Fisher matrix stays
    small. Frozen: everything else near enough to contribute flux, folded in
    as a constant. Both radii are `patches.py`'s.

    `BBOX_PAD = 3 sigma` captures 100% of an isolated emitter's position
    information and ~90% of its amplitude information; widening it changes no
    measured outcome and costs runtime linearly in window area.
    """
    n = len(positions)
    if n == 0:
        free = np.empty(0, dtype=int)
    else:
        d = np.linalg.norm(np.atleast_2d(positions) - cand, axis=1)
        near = np.argsort(d)
        free = near[d[near] <= LINK_FACTOR * sigma][:k_max - 1]

    pts = np.vstack([np.atleast_2d(positions)[free], cand[None, :]]) \
        if len(free) else cand[None, :]
    pad = BBOX_PAD * sigma
    y0 = max(0, int(np.floor(pts[:, 0].min() - pad)))
    x0 = max(0, int(np.floor(pts[:, 1].min() - pad)))
    y1 = min(shape[0], int(np.ceil(pts[:, 0].max() + pad)) + 1)
    x1 = min(shape[1], int(np.ceil(pts[:, 1].max() + pad)) + 1)

    if n:
        others = np.setdiff1d(np.arange(n), free)
        py = np.clip(positions[others, 0], y0, y1 - 1)
        px = np.clip(positions[others, 1], x0, x1 - 1)
        d = np.hypot(positions[others, 0] - py, positions[others, 1] - px)
        frozen = others[d <= HALO_FACTOR * sigma]
    else:
        frozen = np.empty(0, dtype=int)
    return free, frozen, (y0, x0, y1, x1)


def _fit_window(sub, yy, xx, sigma, halo, theta0, max_iter=100):
    K = (len(theta0) - 1) // 3
    smax = max(float(sub.max()), 1.0)
    b_max = max(smax * 4.0, 10.0)
    A_max = 8.0 * smax / psf.peak_factor(sigma)
    lo, hi = msearch._bounds(K, sub.shape[0], sub.shape[1], b_max, A_max)
    th0 = np.clip(np.asarray(theta0, float), lo + 1e-9, hi - 1e-9)
    return lmga.fit(th0, yy, xx, sigma, sub, lo, hi, halo=halo,
                    max_iter=max_iter)


def _halo_image(positions, amplitudes, frozen, sigma, yy, xx, y0, x0, base):
    """`base` plus the frozen emitters' contribution, in window coordinates."""
    if not len(frozen):
        return base
    return base + psf.model(psf.pack(0.0, amplitudes[frozen],
                                     positions[frozen, 0] - y0,
                                     positions[frozen, 1] - x0),
                            yy, xx, sigma)


def _try_add(d_e, positions, amplitudes, bmap, cand, camp, sigma,
             lam, A_s, k_max):
    """Score adding one emitter. Returns (accepted, positions, amplitudes).

    Both models are fitted on the SAME pixels with the SAME frozen halo and
    differ only by the one emitter, which is what makes their I-divergences
    differencable into a Bayes factor.

    On acceptance the whole window's refitted parameters are written back:
    adding a source shifts its neighbours, and keeping their stale values
    would leave the model worse than the fit that justified the acceptance.

    The only conditions that can block the move are `log BF <= 0` and
    `COND_GUARD` -- an ill-conditioned Fisher matrix makes the Occam term
    meaningless, so it is not weighed against anything. A pre-fit significance
    screen on A/SE is deliberately absent: measured, it refused 68-79% of every
    true emitter lost inside 2 sigma before the Bayes factor could vote, while
    the Bayes factor itself refused none of them. Degenerate configurations are
    removed by `_prune`, after a joint fit, on more information.
    """
    free, frozen, (y0, x0, y1, x1) = _window(positions, cand, sigma,
                                             d_e.shape, k_max)
    sub = np.asarray(d_e[y0:y1, x0:x1])
    h, w = sub.shape
    yy, xx = np.mgrid[0:h, 0:w] * 1.0

    level, halo = _window_bg(bmap, y0, x0, y1, x1)
    halo = _halo_image(positions, amplitudes, frozen, sigma, yy, xx,
                       y0, x0, halo)

    loc = (positions[free] - np.array([y0, x0])) if len(free) \
        else np.empty((0, 2))
    a0 = amplitudes[free] if len(free) else np.empty(0)

    th_b = psf.pack(level, a0, loc[:, 0], loc[:, 1])
    r_b = _fit_window(sub, yy, xx, sigma, halo, th_b)

    cl = cand - np.array([y0, x0])
    th_a = psf.pack(level, np.append(a0, camp),
                    np.append(loc[:, 0], cl[0]), np.append(loc[:, 1], cl[1]))
    r_a = _fit_window(sub, yy, xx, sigma, halo, th_a)

    log_bf, cond = evidence.log_bf_add(
        r_b.I, r_a.I, r_b.F, r_a.F,
        float(np.sum(psf.unpack(r_b.theta)[1])),
        float(np.sum(psf.unpack(r_a.theta)[1])),
        len(free), lam, A_s)
    if not np.isfinite(log_bf) or log_bf <= 0 or cond > evidence.COND_GUARD:
        return False, positions, amplitudes

    _, A, cy, cx = psf.unpack(r_a.theta)
    new_pos = np.stack([cy + y0, cx + x0], axis=1)
    if len(free):
        positions = positions.copy()
        amplitudes = amplitudes.copy()
        positions[free] = new_pos[:len(free)]
        amplitudes[free] = A[:len(free)]
    positions = np.vstack([positions, new_pos[-1][None, :]]) if len(positions) \
        else new_pos[-1][None, :]
    amplitudes = np.append(amplitudes, A[-1])
    return True, positions, amplitudes


def _try_split(d_e, positions, amplitudes, bmap, gi, sigma,
               lam, A_s, k_max):
    """Score replacing emitter `gi` with two. Returns (accepted, pos, amp).

    The move FIND structurally cannot make. Two emitters closer than about
    1.5 sigma are fitted well by one brighter PSF, so their residual has no
    PEAK -- it has a quadrupole, negative in the middle and positive on two
    lobes along the pair axis. `moves.residual_axis` recovers that axis from
    the second moment of the residual and the split is proposed along it.

    Takes K to K+1 like `_try_add`, so the outer loop stays monotone in N.
    """
    free, frozen, (y0, x0, y1, x1) = _window(positions, positions[gi], sigma,
                                             d_e.shape, k_max)
    lk = int(np.nonzero(free == gi)[0][0]) if gi in free else None
    if lk is None:
        return False, positions, amplitudes

    sub = np.asarray(d_e[y0:y1, x0:x1])
    h, w = sub.shape
    yy, xx = np.mgrid[0:h, 0:w] * 1.0
    level, halo = _window_bg(bmap, y0, x0, y1, x1)
    halo = _halo_image(positions, amplitudes, frozen, sigma, yy, xx,
                       y0, x0, halo)

    loc = positions[free] - np.array([y0, x0])
    th_b = psf.pack(level, amplitudes[free], loc[:, 0], loc[:, 1])
    r_b = _fit_window(sub, yy, xx, sigma, halo, th_b)

    resid = sub - psf.model(r_b.theta, yy, xx, sigma, halo=halo)
    u, _ = moves.residual_axis(r_b.theta, lk, yy, xx, sigma, resid)

    # One incumbent, several proposals: its log-determinant is the same for
    # all of them and is factorized once.
    ld_b = evidence.logdet(r_b.F)
    sumA_b = float(np.sum(psf.unpack(r_b.theta)[1]))
    best = None
    for disp in SPLIT_DISPS:
        th_a = moves.split(r_b.theta, lk, u, disp * sigma)
        r_a = _fit_window(sub, yy, xx, sigma, halo, th_a)
        log_bf, cond = evidence.log_bf_add(
            r_b.I, r_a.I, r_b.F, r_a.F, sumA_b,
            float(np.sum(psf.unpack(r_a.theta)[1])),
            len(free), lam, A_s, before=ld_b)
        if np.isfinite(log_bf) and log_bf > 0 and cond <= evidence.COND_GUARD:
            if best is None or log_bf > best[0]:
                best = (log_bf, r_a)
    if best is None:
        return False, positions, amplitudes

    # `moves.split` keeps the untouched emitters in order and appends the two
    # children, so the fitted vector is [free without gi] + [child0, child1].
    _, A, cy, cx = psf.unpack(best[1].theta)
    new_pos = np.stack([cy + y0, cx + x0], axis=1)
    keep_local = [j for j in range(len(free)) if j != lk]
    positions = positions.copy()
    amplitudes = amplitudes.copy()
    if keep_local:
        positions[free[keep_local]] = new_pos[:len(keep_local)]
        amplitudes[free[keep_local]] = A[:len(keep_local)]
    positions[gi] = new_pos[-2]
    amplitudes[gi] = A[-2]
    positions = np.vstack([positions, new_pos[-1][None, :]])
    amplitudes = np.append(amplitudes, A[-1])
    return True, positions, amplitudes


def _split_pass(d_e, positions, amplitudes, bmap, sigma, lam, A_s,
                k_max, model):
    """Propose a split for every emitter, most pair-like first.

    The ranking is not an optimization detail: a split accepted early changes
    its neighbours, so the order decides which configuration the later
    proposals are scored against.

    The ranking statistic is the residual quadrupole, which background
    curvature also produces. On a strongly structured background the ordering
    degrades and the advantage this move carries shrinks accordingly.
    """
    n0 = len(positions)
    if n0 == 0:
        return positions, amplitudes, 0
    resid_full = d_e - model
    strengths = np.zeros(n0)
    pad = int(np.ceil(BBOX_PAD * sigma))
    for i in range(n0):
        y0 = max(0, int(positions[i, 0]) - pad)
        x0 = max(0, int(positions[i, 1]) - pad)
        y1 = min(d_e.shape[0], int(positions[i, 0]) + pad + 1)
        x1 = min(d_e.shape[1], int(positions[i, 1]) + pad + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1] * 1.0
        th = psf.pack(0.0, [amplitudes[i]], [positions[i, 0]], [positions[i, 1]])
        _, s = moves.residual_axis(th, 0, yy, xx, sigma,
                                   resid_full[y0:y1, x0:x1])
        strengths[i] = s

    n_split = 0
    for gi in np.argsort(-strengths):
        if strengths[gi] <= 0:
            break
        ok, positions, amplitudes = _try_split(
            d_e, positions, amplitudes, bmap, int(gi), sigma,
            lam, A_s, k_max)
        n_split += int(ok)
    return positions, amplitudes, n_split


def _amplitude_var(F, k):
    """var(A_k) from the Fisher matrix, or None if it cannot be trusted.

    None means the Laplace evidence for this configuration cannot be computed
    -- a singular Fisher matrix, or a non-positive amplitude variance -- so
    removal must be forced rather than weighed. Weighing is not an option:
    the quantity that would do the weighing is the thing that has broken.
    """
    try:
        var = np.diag(np.linalg.inv(F))
    except np.linalg.LinAlgError:
        return None
    v = var[1 + 3 * k]
    if not np.isfinite(v) or v <= 0:
        return None
    return float(v)


def _prune(d_e, positions, amplitudes, bmap, sigma, lam, A_s, k_max):
    """One pass of removal tests. Returns (positions, amplitudes).

    Runs outside the add loop and never feeds back into it: an emitter's A/SE
    verdict depends on which neighbours are free in that pass, so letting
    removal drive addition makes the same source get killed and recreated
    indefinitely.
    """
    n = len(positions)
    if n == 0:
        return positions, amplitudes
    # Copied because the survivor write-back below mutates these in place and
    # the caller still holds the pre-prune configuration.
    positions = np.array(positions, dtype=float, copy=True)
    amplitudes = np.array(amplitudes, dtype=float, copy=True)
    # An `alive` mask rather than deleting as we go: the visit order is
    # computed once, and deleting from the arrays inside the loop would shift
    # every later index in that order onto a different emitter.
    alive = np.ones(n, dtype=bool)
    # Faintest first: a spurious emitter is far likelier to be faint, and
    # removing it may make its neighbour's own removal unnecessary.
    for gi in np.argsort(amplitudes):
        gi = int(gi)
        if not alive[gi]:
            continue
        cand = positions[gi]
        others = np.nonzero(alive & (np.arange(n) != gi))[0]
        free, frozen, (y0, x0, y1, x1) = _window(positions[others], cand,
                                                 sigma, d_e.shape, k_max)
        free = others[free]
        frozen = others[frozen]
        sub = np.asarray(d_e[y0:y1, x0:x1])
        h, w = sub.shape
        yy, xx = np.mgrid[0:h, 0:w] * 1.0
        level, halo = _window_bg(bmap, y0, x0, y1, x1)
        halo = _halo_image(positions, amplitudes, frozen, sigma, yy, xx,
                           y0, x0, halo)

        keep_idx = np.append(free, gi).astype(int)
        loc = positions[keep_idx] - np.array([y0, x0])
        th_full = psf.pack(level, amplitudes[keep_idx], loc[:, 0], loc[:, 1])
        r_full = _fit_window(sub, yy, xx, sigma, halo, th_full)

        lr = positions[free] - np.array([y0, x0]) if len(free) \
            else np.empty((0, 2))
        th_red = psf.pack(level, amplitudes[free] if len(free)
                          else np.empty(0), lr[:, 0], lr[:, 1])
        r_red = _fit_window(sub, yy, xx, sigma, halo, th_red)

        # `gi` is appended last in `keep_idx`, so it is the last emitter of
        # the fitted vector.
        k = len(keep_idx) - 1
        v = _amplitude_var(r_full.F, k)
        if v is None or psf.unpack(r_full.theta)[1][k] < PRUNE_TAU * np.sqrt(v):
            log_bf = np.inf                   # forced, not weighed
        else:
            log_bf = evidence.log_bf_remove(
                r_full.I, r_red.I, r_full.F, r_red.F,
                float(np.sum(psf.unpack(r_full.theta)[1])),
                float(np.sum(psf.unpack(r_red.theta)[1])),
                len(keep_idx), lam, A_s)
        if log_bf > 0:
            alive[gi] = False
            # Write the reduced fit back over the survivors. This is what makes
            # the faintest-first cascade correct: when a collapsed pair loses
            # one member the other absorbs its flux, and the next removal test
            # must be scored against that, not against a stale half-amplitude
            # that would make the survivor look removable too.
            if len(free):
                _, A, cy, cx = psf.unpack(r_red.theta)
                positions[free, 0] = cy + y0
                positions[free, 1] = cx + x0
                amplitudes[free] = A
    return positions[alive], amplitudes[alive]


def _update_bg(d_e, positions, amplitudes, sigma, bmap, kernel):
    """Re-estimate the background from the current emitter model.

    `kernel=None` gives one scalar for the frame, from the pixels no emitter
    reaches; any integer estimates a surface on that window.
    """
    scalar = max(calibrate.robust_background(d_e, positions, sigma), BG_FLOOR)
    if kernel is None:
        return np.full(bmap.shape, scalar)
    return background_map(d_e, positions, sigma, kernel=kernel,
                          fallback=scalar)


def detect(data_img, sigma=1.2, offset=0.0, gain=None, lam0=0.02, A_s0=None,
           k_max=12, max_rounds=6, threshold=CAND_THRESHOLD, prune=True,
           split=True, bg_kernel=BG_KERNEL, max_settle=4, verbose=1):
    """Full detection. Returns a `DetectResult`.

    `max_rounds` is a safety stop, not the termination condition: the loop ends
    when a round accepts nothing, which it must eventually do because every
    accepted emitter lowers the residual that produces the candidates.

    `bg_kernel=None` fixes the background at one scalar for the whole frame;
    any integer estimates a surface on that window. The surface is re-estimated
    once per round from the current emitter model, so background and emitters
    are fitted by backfitting rather than simultaneously. It changes the
    reported amplitudes by less than the seed-to-seed noise, because every fit
    already has a free background scalar over a 9-13 px window and a background
    smooth on the scale of the frame is nearly constant over one. What it does
    buy is a cleaner residual on real frames.
    """
    raw = np.asarray(data_img, dtype=float)
    H, W = raw.shape
    g_eff = calibrate.estimate_gain(raw, offset) if gain is None else float(gain)
    d_e = (raw - offset) / g_eff

    b0 = float(np.percentile(d_e, 10.0))
    bmap = np.full((H, W), max(b0, BG_FLOOR))
    if A_s0 is None:
        A_s0 = max(float(d_e.max()) - b0, 10.0) / psf.peak_factor(sigma)
    lam, A_s = lam0, A_s0

    positions = np.empty((0, 2))
    amplitudes = np.empty(0)
    history = []

    for rnd in range(max_rounds):
        model = render(positions, amplitudes, sigma, bmap)
        cand, camp, _ = find_candidates(d_e, model, sigma, positions, threshold)
        n_added = 0
        for c, a in zip(cand, camp):
            # Re-check proximity: an earlier acceptance in THIS round may have
            # already claimed this candidate's flux.
            if len(positions) and np.min(np.linalg.norm(
                    positions - c, axis=1)) <= sigma:
                continue
            ok, positions, amplitudes = _try_add(
                d_e, positions, amplitudes, bmap, c, a, sigma,
                lam, A_s, k_max)
            n_added += int(ok)

        # SPLIT runs on the model the adds just produced: an emitter only looks
        # like an unresolved pair once its neighbourhood is otherwise
        # explained, and splitting against a model still missing a nearby
        # source mostly splits emitters into that source's flux.
        n_split = 0
        if split:
            if n_added:
                model = render(positions, amplitudes, sigma, bmap)
            positions, amplitudes, n_split = _split_pass(
                d_e, positions, amplitudes, bmap, sigma, lam, A_s,
                k_max, model)

        if n_added or n_split:
            # One sweep here; the round loop is the outer iteration.
            positions, amplitudes, _ = refine(
                d_e, positions, amplitudes, sigma, bmap, k_max=k_max,
                max_sweeps=1)
            lam = max(len(positions) / float(H * W), 1e-6)
            if len(amplitudes):
                A_s = max(float(np.mean(amplitudes)), 1.0)
            # Re-estimated only AFTER the emitters have been re-fitted, and
            # only from the emitter model with no background in it.
            bmap = _update_bg(d_e, positions, amplitudes, sigma, bmap,
                              bg_kernel)

        history.append(dict(round=rnd, N=len(positions), added=n_added,
                            split=n_split, candidates=len(cand),
                            background=float(np.median(bmap))))
        if verbose >= 1:
            print(f"  [round {rnd}] {len(cand):3d} candidates, "
                  f"{n_added:3d} added, {n_split:3d} split "
                  f"-> N={len(positions):3d}  "
                  f"bg={np.median(bmap):.3f} "
                  f"[{bmap.min():.2f}, {bmap.max():.2f}]")
        if n_added == 0 and n_split == 0:
            break

    # Refine BEFORE pruning. `refine` re-fits at fixed N with no separation
    # constraint and routinely pulls an accepted pair together; those collapsed
    # pairs are exactly what the removal test is for, and pruning first cannot
    # see them. So the order is settle, prune, settle again at the reduced N.
    positions, amplitudes, se = refine(d_e, positions, amplitudes, sigma, bmap,
                                       k_max=k_max)

    # Settle and prune alternate until the prune removes nothing. One pass is
    # not enough: the re-fit at the reduced N is as free to collapse a pair as
    # the first one was. This cannot cycle -- prune only removes, so N strictly
    # decreases. `max_settle` is a backstop.
    for _ in range(max_settle if prune else 0):
        if not len(positions):
            break
        n_before = len(positions)
        positions, amplitudes = _prune(d_e, positions, amplitudes, bmap,
                                       sigma, lam, A_s, k_max)
        if len(positions) == n_before:
            break
        if verbose >= 1:
            print(f"  [prune] {n_before} -> {len(positions)}")
        positions, amplitudes, se = refine(d_e, positions, amplitudes, sigma,
                                           bmap, k_max=k_max)
    model = render(positions, amplitudes, sigma, bmap)
    if verbose >= 1:
        nr = (d_e - model) / np.sqrt(np.maximum(model, 1e-6))
        print(f"[final] N={len(positions)}  "
              f"bg={np.median(bmap):.2f} [{bmap.min():.2f}, {bmap.max():.2f}]  "
              f"resid median={np.median(nr):+.3f}  "
              f"robust_std={calibrate.robust_spread(nr):.3f}")

    return DetectResult(
        positions=positions, amplitudes=amplitudes, sigma=sigma, lam=lam,
        A_s=A_s, gain=g_eff, n_outer_passes=len(history), model_image=model,
        # `background` is the SURFACE, an (H, W) array, not a scalar. Callers
        # that only want a number should take its median; callers that render
        # or subtract a model want the array.
        residual=d_e - model, background=bmap, se=se, history=history,
    )
