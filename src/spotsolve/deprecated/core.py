"""Shared pieces of the box search's Python reference, `box.localize_boxes`.

The detector is `spotsolve.localize` / `localize_stack`, which runs entirely
in Rust (`boxsearch.rs`). `box.py` is its reference, and this module holds
what that reference is built from:

    find_candidates    FIND: LoG peaks of the variance-normalized residual
    background_map     the smooth background surface
    refine             the polish: block-Jacobi refits at fixed N
    _fit_any           one bounded free-width Poisson fit on a window

and the constants both implementations share (`native._check_constants`
asserts the Rust copies). `flag_aggregates` and `aggregate_report` read a
finished result.

`detect`, the Bayes-factor add/split/prune pipeline this module used to run,
was retired on 2026-09-11 in favour of the box search; the measurements in
the notes below that name its passes were taken under it. Its code is in the
git history before that date.
"""

import numpy as np
import scipy.ndimage as ndi

from . import backend as backend_mod
from . import calibrate
from . import patches as patch_mod
from .. import psf

__all__ = ["refine", "find_candidates", "background_map", "log_kernel_l2"]


LINK_FACTOR = 2.5   # emitters within this many sigma are fitted jointly
HALO_FACTOR = 5.0   # ... and beyond it are frozen into the model as a constant
BBOX_PAD = 3.0      # sigma of pixel context around a group's extreme emitters

SIGMA_SLACK = (0.70, 2.2)
FOCUS_BAND = (0.80, 2.0)
# `SIGMA_SLACK` is the MODEL SPACE: the widths a fit may represent, as
# multiples of the PSF sigma. `FOCUS_BAND` is the REPORTING BAND: the widths
# that count as an in-focus detection. The model space has to cover every
# photon on the sensor or the light it cannot represent gets tiled; the
# reporting band is a downstream contract about what the caller is handed.
# `SIGMA_SLACK[0]` sits below `FOCUS_BAND[0]` so that a broken fit -- nothing
# images narrower than the PSF -- reveals itself instead of being clipped to
# the bound and reported.
#
# Why the model needs slack at all. A fixed-sigma model meets a source it
# cannot represent -- anything out of focus -- by TILING it: two narrow
# Gaussians genuinely do fit a wide blob better than one. Measured on a
# confocal simulation with emitters uniform in +/- 0.5 um, at 1 emitter/um^2
# the fixed-sigma search returned 7.2 extra detections per frame against 13.4
# in-focus emitters, every one within 3 sigma(z) of a real emitter.
#
# The band's upper edge is how far out of focus an emitter may still be
# reported: there a point source images at 1.26x the in-focus width at
# |z| = 0.25 um, 1.95x at 0.35 and 3.2x at 0.50, so 2.0 is |z| < ~0.36 um.
# The slack's upper edge, measured under `detect` (moderate arm, 6 frames,
# band fixed at (0.8, 2.0)):
#
#    hi    recall   med err    RMSE   rsd z   |z|>3   tiles
#   2.2     92.6%     0.076   0.220    1.25    7.0%    2.33
#   2.6     91.1%     0.078   0.201    1.31    8.0%    2.33
#   3.2     90.8%     0.078   0.183    1.27    8.7%    4.00
#   4.0     90.5%     0.078   0.194    1.32    8.1%    2.50
#
# Recall falls monotonically past 2.2: a larger model space buys better
# parameters for the objects it keeps and swallows close neighbours. The box
# search re-measured it as step 4 of its ladder (`box.py`): a bound of 4-6
# sigma lost 3-5 recall points under haze and, on a GEM frame, swallowed
# emitters (N 440 -> 271). 2.2 stands.

BG_KERNEL = 25          # side, px, of the window the background surface uses
BG_FLOOR = 1e-3         # W = 1/m is singular at m = 0; far below one e-
BG_MASK_RADIUS = 3.0    # sigma; emitter support excluded from the estimate
BG_MIN_PIXELS = 25      # unmasked pixels a window needs before it is believed

REFINE_MAX_ITER = 50
# LM iterations one REFINE group fit may take. A BUDGET, not a convergence
# criterion, and deliberately below the 200 this used to be.
#
# A few groups per frame never converge: they are not stalled -- they keep
# taking accepted steps to the iteration cap -- because they are descending a
# direction the data carries almost no information about. At 256x256 they are
# ~1% of emitters and 10-19% of REFINE's LM iterations, and they are NOT
# disposable: 36 of 38 of their emitters survive to the final output.
#
# What that descent is worth is boundable. Each iteration is allowed to
# continue only while it predicts a decrease above REFINE_TOL_OBJ, and a
# decrease of `t` nats moves a parameter about sqrt(2*t) standard errors, so
# the whole truncated tail is worth a small fraction of one SE. Measured
# end-to-end at 256x256, against the same fit run to 400 iterations:
#
#     max_iter   frame s        N   audit   max |dpos|/SE
#           25      7.24     3166   clean          0.0820
#           50      7.80     3166   clean          0.0023   <- here
#          100      8.19     3168   clean          0.0006
#          200      8.73     3169   clean          0.0077
#          400      9.00     3168   clean          0.0006
#
# N wanders by +-3 at EVERY budget, 400 included, so it is the usual churn at
# marginal decisions and not a signal; the audit is clean throughout. 50 is
# where the agreement band is tightest for the least work, and on a sweep with
# no stuck group it is bit-identical to a budget of 4000.
#
REFINE_SWEEPS = 4       # see `refine`
REFINE_TOL = 1e-3       # px; position shift below which a sweep is a no-op

REFINE_TOL_OBJ = 1e-6
# nats of predicted decrease, for the polish. Near the optimum
# I(t) ~ I_min + 0.5 dt' F dt, so stopping at a predicted decrease of `tol`
# leaves a parameter about sqrt(2*tol) standard errors short. Measured under
# `detect` (256 px, N=3169, py-vs-rs band in SE):
#
#     tol_obj    frame s        N     max |dpos|/SE
#       1e-8        9.78     3169            0.0010
#       1e-6        8.66     3169            0.0015   <- here
#       1e-5        7.68     3168            0.1466
#       1e-4        7.08     3161            0.2828
#
# 1e-6 is the last value that leaves N and the agreement band where 1e-8 does.

A_MIN = 1e-4
A_MIN_REL = 1e-6
# The amplitude floor a fit may not go below, as a fraction of the window's own
# `A_max`; `A_MIN` is the absolute backstop. The floor has to be relative
# because
# what it protects is a RATIO. An emitter's position block of the Fisher
# matrix scales as A^2, so at bead fluxes of ~2000 e- an amplitude of 1e-4
# puts those entries at ~5.7e-12 against a largest diagonal of ~768 -- a ratio
# of 3e-14, about 130x float64 epsilon. At that point log|F| is numerical
# noise, the Occam term of every Bayes factor built on it is noise with it,
# and the LM step along that direction is unbounded: traced on such a patch,
# the fit predicted a 5.2e4 nat decrease, delivered 1.05e-3, and crawled for
# 3000+ iterations still 364 nats above the optimum.
#
# Measured ratio of smallest to largest diag(F) with a second emitter parked
# at the floor:
#
#     floor / A_max     min/max diag(F)
#       0 (1e-4 abs)        3.0e-14      <- float64 noise
#           1e-6            6.4e-10
#           1e-4            4.8e-07      (saturates; a different parameter
#           3e-3            4.8e-07       becomes the smallest)
#
# 1e-6 buys six orders of margin over epsilon while remaining physically
# negligible -- on a bead patch it is a floor of ~0.02 e- of total flux. A
# larger floor would start to express an opinion about how faint an emitter
# may be, and that decision belongs to the search, not to a numerical guard.
# The absolute floor `A_MIN` is only a backstop for a window whose `A_max` is
# itself tiny.

def background_map(d_e, positions, sigma, kernel=BG_KERNEL,
                   radius_factor=BG_MASK_RADIUS, fallback=None, free=None):
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
    # `free` may be supplied by the caller -- `_update_bg` does, because the
    # scalar fallback needs the same mask. The convolutions below stay in scipy
    # either way: they are per-round and profile at 0.1-0.3%.
    if free is None:
        free = calibrate.emitter_free_mask((H, W), positions, sigma,
                                           radius_factor)
    if fallback is None:
        fallback = calibrate.robust_background(d, positions, sigma,
                                               radius_factor, free=free)

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


def _refine_sweep(d_e, positions, amplitudes, sigmas, sigma, bmap, k_max,
                  link_radius_factor, max_iter, dirty=None, se=None,
                  move_eps=0.0, slack=SIGMA_SLACK, fit_backend=None):
    """One block-Jacobi pass over the patch decomposition.

    Jacobi, not Gauss-Seidel: the halo is built from the sweep's INPUT state
    and results go to a separate output array, so every patch in a sweep reads
    the same frozen neighbourhood regardless of visit order. Within a patch the
    emitters are fitted jointly, so the blocks are solved exactly. That
    order-independence is what makes the pass safe to parallelize, and it is
    worth preserving deliberately.

    Note the map is not continuous: `build_patches` is rebuilt from the current
    positions, so an emitter that drifts far enough to merge or split a patch
    changes which joint fit it belongs to, and jumps. A fixed point need not
    exist -- see PORTING_NOTES section 20.
    """
    if se is None:
        se = np.full((len(amplitudes), 3), np.nan)
    out_pos = positions.copy()
    out_amp = amplitudes.copy()
    out_sig = sigmas.copy()
    moved = np.zeros(len(amplitudes), dtype=bool)
    n_fitted = 0
    pset = patch_mod.build_patches(positions, sigma, d_e.shape,
                                   link_radius_factor=link_radius_factor,
                                   k_max=k_max)
    for p in pset:
        # A group's fit reads its own emitters and its frozen halo and nothing
        # else. If none of those moved last pass, refitting reproduces its own
        # answer to within the optimizer's noise -- and that noise is not free,
        # because moving the group dirties its neighbours. Scheduling on the
        # decomposition's own locality is what lets quiet regions drop out.
        if dirty is not None:
            touched = np.concatenate([p.indices, p.frozen_indices]) \
                if len(p.frozen_indices) else p.indices
            if not dirty[np.asarray(touched, dtype=int)].any():
                continue
        n_fitted += 1
        yy, xx = patch_mod.patch_grids(p)
        halo = patch_mod.build_halo_image(positions, amplitudes,
                                          p.frozen_indices, sigmas, yy, xx,
                                          p.y0, p.x0)
        level, shape_ = _window_bg(bmap, p.y0, p.x0, p.y1, p.x1)
        halo = halo + shape_
        sub = np.asarray(d_e[p.y0:p.y1, p.x0:p.x1])
        loc = positions[p.indices] - np.array([p.y0, p.x0])
        r, _, A, cy, cx, sg = _fit_any(
            sub, yy, xx, sigma, halo, level, amplitudes[p.indices],
            loc[:, 0], loc[:, 1], sigmas[p.indices], slack,
            max_iter=max_iter, tol_obj=REFINE_TOL_OBJ, fit_backend=fit_backend)
        out_amp[p.indices] = A
        out_sig[p.indices] = sg
        out_pos[p.indices, 0] = cy + p.y0
        out_pos[p.indices, 1] = cx + p.x0
        moved[p.indices] = np.hypot(
            out_pos[p.indices, 0] - positions[p.indices, 0],
            out_pos[p.indices, 1] - positions[p.indices, 1]) > move_eps
        try:
            var = np.diag(np.linalg.inv(r.F))
        except np.linalg.LinAlgError:
            continue
        v = np.where(var > 0, var, np.nan)
        # (A, y, x) are the first three of each emitter's four; the width is
        # reported separately, not folded into `se`.
        se[p.indices, 0] = np.sqrt(v[1::4])
        se[p.indices, 1] = np.sqrt(v[2::4])
        se[p.indices, 2] = np.sqrt(v[3::4])
    return out_pos, out_amp, out_sig, se, moved, n_fitted


def refine(d_e, positions, amplitudes, sigma, bmap, k_max=12,
           link_radius_factor=LINK_FACTOR, max_iter=None,
           max_sweeps=REFINE_SWEEPS, tol=REFINE_TOL, sigmas=None,
           slack=SIGMA_SLACK, fit_backend=None):
    """Joint re-fit at fixed N in connected groups, plus per-emitter CRLBs.

    Returns (positions, amplitudes, se, sigmas) with `se` an (N,3) array of
    (SE_A, SE_y, SE_x) from the Fisher matrix of the fit whose parameters are
    reported, and `sigmas` the per-emitter widths, fitted within `slack`. The
    background is not returned: it is a surface owned by the caller.

    Scheduled group-wise, not globally. A group is refitted only when an
    emitter it reads -- its own or one in its frozen halo -- moved more than
    `tol` in the previous pass, so quiet regions drop out and the loop has a
    termination condition it can reach. It replaces a global
    `max |dpos| < tol` break that could not fire: measured on a 906-emitter
    frame, 80% of emitters were still moving past `tol` after eight sweeps.

    Do not expect the queue to drain on a crowded field. It does not, at any
    threshold up to 0.1 px, because halos overlap and dirtiness percolates from
    a handful of degenerate groups -- collapsed pairs at nn-distances of
    0.001 px that never converge. Neither Gauss-Seidel nor pinning the
    decomposition fixes that (both were measured; see PORTING_NOTES section
    20), because it is not a slow-iteration problem; `max_sweeps` bounds it.

    Iterated to a fixed point, not run once. Each patch fit holds its
    out-of-patch neighbours frozen at the positions and amplitudes it was
    handed, so a single pass propagates whatever staleness those carry into
    the emitter it surrounds. Measured on isolated emitters at density 0.055,
    a 0.5 px error in the NEIGHBOURS alone (target started at truth) takes the
    pull sd from 0.96 to 1.98; sweeping to convergence recovers it to 1.27.
    The patch decomposition is rebuilt each sweep, which is what refreshes the
    halo.

    `fit_backend` optionally selects the numerical backend for variable-width
    fits; the default uses the Python reference.

    `max_sweeps=1` is a single pass.
    """
    if max_iter is None:
        max_iter = REFINE_MAX_ITER
    positions = np.atleast_2d(np.asarray(positions, float))
    amplitudes = np.asarray(amplitudes, float).ravel()
    sigmas = (np.full(len(amplitudes), float(sigma)) if sigmas is None
              else np.asarray(sigmas, float).ravel())
    if len(amplitudes) == 0:
        return positions, amplitudes, np.empty((0, 3)), sigmas

    se = np.full((len(amplitudes), 3), np.nan)
    dirty = np.ones(len(amplitudes), dtype=bool)
    for _ in range(max(1, int(max_sweeps))):
        positions, amplitudes, sigmas, se, moved, n_fitted = _refine_sweep(
            d_e, positions, amplitudes, sigmas, sigma, bmap, k_max,
            link_radius_factor, max_iter, dirty=dirty, se=se, move_eps=tol,
            slack=slack, fit_backend=fit_backend)
        if n_fitted == 0 or not moved.any():
            break
        dirty = moved
    return positions, amplitudes, se, sigmas


_LOG_L2_CACHE = {}


def log_kernel_l2(sigma):
    """L2 norm of the 2-D LoG kernel -- the null sd of `find_candidates`'s
    filter response, and therefore its unit.

    `nr = (d - m)/sqrt(m)` is unit-variance per pixel under a correct model,
    so a linear filter of it has sd `||w||_2`. Dividing by this turns the LoG
    response into a z-score that means the same thing at every sigma.

    Closed form rather than a filtered impulse. The 2-D kernel is
    `w = g2 (x) g0 + g0 (x) g2` for 1-D Gaussian derivative kernels `g0`, `g2`,
    so

        ||w||^2 = 2 ||g0||^2 ||g2||^2 + 2 (g0 . g2)^2

    which is exact (checked bit-identical to `sqrt(sum(gaussian_laplace(
    impulse)**2))` at sigma 0.8 to 3.0) and costs O(taps) instead of a 2-D
    filter. Cached: the reference calls it once per placement.
    """
    key = float(sigma)
    if key not in _LOG_L2_CACHE:
        n = 201
        imp = np.zeros(n)
        imp[n // 2] = 1.0
        r = int(4.0 * key + 0.5)
        sl = slice(n // 2 - r, n // 2 + r + 1)
        g0 = ndi.gaussian_filter1d(imp, key, order=0, mode="constant")[sl]
        g2 = ndi.gaussian_filter1d(imp, key, order=2, mode="constant")[sl]
        _LOG_L2_CACHE[key] = float(np.sqrt(
            2.0 * (g0 @ g0) * (g2 @ g2) + 2.0 * (g0 @ g2) ** 2))
    return _LOG_L2_CACHE[key]


def find_candidates(d_e, model, sigma, threshold):
    """LoG peaks on the variance-normalized residual `d_e - model`, brightest
    first. Returns `(positions, amplitude guesses, strengths)`.

    Two normalizations, and they do different jobs. Dividing by `sqrt(model)`
    before the filter is what makes one threshold valid across the FRAME:
    under Poisson noise the residual's scale is the square root of the mean,
    so the ratio is on a fixed sigma scale wherever the model puts flux.
    Dividing by `log_kernel_l2` after it is what makes one threshold valid
    across SIGMA: the filter's own null sd is its kernel's L2 norm, which
    scales as sigma^-3, so without this the same numeric threshold is a
    1.8-sigma cut at sigma=0.8 and a 100-sigma cut at sigma=3.0.

    `threshold` is therefore a count of standard deviations, on pixels. See
    `calibrate.seed_threshold`.
    """
    resid = d_e - model
    nr = resid / np.sqrt(np.maximum(model, 1e-6))
    log_f = -ndi.gaussian_laplace(nr, sigma) / log_kernel_l2(sigma)
    win = 2 * int(np.ceil(sigma)) + 1
    peaks = (log_f == ndi.maximum_filter(log_f, size=win)) & (log_f > threshold)
    ys, xs = np.nonzero(peaks)
    if len(ys) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0)

    cand = np.stack([ys, xs], axis=1).astype(float)
    strength = log_f[ys, xs]
    amp = np.maximum(resid[ys, xs], 1e-2) / psf.peak_factor(sigma)
    order = np.argsort(-strength)
    return cand[order], amp[order], strength[order]



def _bounds(K, h, w, b_max, A_max, sigma_bounds=None):
    """Box constraints for one window's fit, in LOCAL coordinates.

    Positions are confined to the sub-image. Letting a centre leave the frame
    was tried -- it lets the fit put rim flux where it actually came from --
    and measurably lost real detections elsewhere, so the bounds stay closed.

    `sigma_bounds` adds a fourth parameter per emitter, its own width, bounded
    to that (lo, hi) in pixels. Omit it for the fixed-width layout.
    """
    a_min = max(A_MIN, A_MIN_REL * A_max)
    lo = [0.0]
    hi = [b_max]
    for _ in range(K):
        lo += [a_min, -0.5, -0.5]
        hi += [A_max, h - 0.5, w - 0.5]
        if sigma_bounds is not None:
            lo.append(sigma_bounds[0])
            hi.append(sigma_bounds[1])
    return np.asarray(lo), np.asarray(hi)



def _fit_any(sub, yy, xx, sigma, halo, b, A, cy, cx, sig, slack, max_iter,
             tol_obj, fit_backend=None):
    """One free-width window fit, taken and returned in UNPACKED parameters.

    Returns `(r, b, A, cy, cx, sig)`; the caller may read `I`, `F` and
    `converged` off `r`.

    `A_max` is raised by `slack[1]**2`: it is derived from the window's peak
    through `peak_factor(sigma)`, and a source at `n` times the PSF width
    carries the same flux at `1/n^2` of the peak, so the in-focus bound would
    clip exactly the defocused emitters the slack exists for.
    """
    K = len(A)
    smax = max(float(sub.max()), 1.0)
    b_max = max(smax * 4.0, 10.0)
    A_max = 8.0 * smax / psf.peak_factor(sigma) * slack[1] ** 2
    sb = (slack[0] * sigma, slack[1] * sigma)
    lo, hi = _bounds(K, sub.shape[0], sub.shape[1], b_max, A_max, sb)
    th0 = np.clip(psf.pack_var_sigma(b, A, cy, cx, np.clip(sig, *sb)),
                  lo + 1e-9, hi - 1e-9)
    if fit_backend is None:
        fit_backend = backend_mod.get("py")
    r = fit_backend.fit_var_sigma(th0, *sub.shape, sub, halo, lo, hi,
                                  max_iter, tol_obj=tol_obj)
    return (r,) + psf.unpack_var_sigma(r.theta)


def _model_any(b, A, cy, cx, sig, yy, xx, sigma, slack, halo=0.0):
    """The window model of a free-width fit's parameters."""
    return psf.model_var_sigma(psf.pack_var_sigma(b, A, cy, cx, sig),
                               yy, xx, halo=halo)
