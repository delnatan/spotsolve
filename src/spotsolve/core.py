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

from . import backend as backend_mod
from . import calibrate
from . import evidence
from . import lmga
from . import moves
from . import patches as patch_mod
from . import psf
from .structs import DetectResult

__all__ = ["detect", "find_candidates", "background_map", "refine",
           "log_kernel_l2", "find_aggregates", "flag_aggregates",
           "aggregate_report"]


LINK_FACTOR = 2.5   # emitters within this many sigma are fitted jointly
HALO_FACTOR = 5.0   # ... and beyond it are frozen into the model as a constant
BBOX_PAD = 3.0      # sigma of pixel context around a group's extreme emitters
CAND_THRESHOLD = calibrate.LOG_SEED_Z

PRUNE_TAU = 2.0
# A / SE(A), measured on the joint fit, below which `_prune` removes an emitter
# outright instead of scoring it. Three roles:
#
#   - it is the pipeline's precision/recall dial, and the only one left;
#   - it forces removal where the Laplace evidence cannot be computed. A
#     collapsed pair's Fisher matrix is singular along its separation
#     direction, so var(A) diverges and A/SE goes to zero on its own; the
#     Bayes factor there would argue to KEEP the pair, harder the more
#     degenerate it is.
#   - it keeps the Laplace approximation inside its domain of validity. That
#     is a claim about the approximation, not a significance test: the Laplace
#     form integrates the added dimensions against an UNBOUNDED Gaussian of
#     width SE(A), while the true posterior is truncated at A >= 0. When the
#     mode sits less than a few SE from that boundary the Gaussian spills
#     across it and the posterior volume -- hence the evidence -- is
#     overstated.
#
# That overstatement is not a bounded nuisance, it diverges. The emitter's 3x3
# block of F has F_AA = O(1) but F_yy, F_xx proportional to A^2, so |F| ~ A^4
# and the Laplace volume |F|^-1/2 ~ A^-2. Holding a second emitter at a fixed
# amplitude and shrinking it (one real emitter, 11x11 patch, bg 4 e-):
#
#     A_2      dI      -0.5 dlog|F|    log BF for ADDING it
#    30.0    3.771         2.042             -2.77
#     3.0    0.727         6.237             -1.60
#     1.0    0.045        10.302             +1.78
#     0.1    0.025        13.191             +4.65
#     0.01   0.003        17.699             +9.14
#
# dI goes to zero -- the emitter explains nothing -- while the Occam term,
# whose whole job is to charge for complexity, PAYS about 4.5 nats per decade
# for making it fainter. Any greedy search with an honest optimizer walks
# straight into that, which is why the guard cannot be dropped.
#
# Calibrated against exact 4-D numerical integration of the same posterior,
# binned by A/SE(A) (mean signed error of the Laplace log BF, and the largest
# absolute error in the bin):
#
#     A/SE(A)     n     mean err    max |err|
#       0-1       9      -0.59        3.22
#       1-2      11      -0.80        1.70
#       2-3       7      -0.74        1.22
#       3-4       4      -0.26        0.61
#       4-6       4      -0.22        0.28
#       6-10      4      +0.01        0.06
#      10+       31      -0.14        1.70
#
# The approximation is trustworthy from about 3 SE outward and degrades below
# it. Validity alone would argue for 3.0; raising it that far costs
# localization as well as recall, because removing one member of a real close
# pair leaves the survivor absorbing both fluxes and sitting between them.
# 2.0 is the measured optimum on both bead frames.

SPLIT_DISPS = (1.0, 1.6)
# Displacements, in sigma, at which a split is proposed. Below ~1 sigma a pair
# is not identifiable and the evidence refuses it; beyond ~2 sigma FIND already
# produces a separate LoG peak and the split is redundant.

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
# Note the evidence fits keep 100 (`_fit_window`'s default). They start further
# from their optimum than REFINE does and their objective IS differenced into a
# Bayes factor, so truncating them biases model selection [P10]. This budget is
# for the polish step only.
REFINE_SWEEPS = 4       # see `refine`
REFINE_TOL = 1e-3       # px; position shift below which a sweep is a no-op

# The two objective tolerances, in NATS of predicted decrease. They differ
# because the fits they stop are answering different questions.
EVIDENCE_TOL_OBJ = 1e-8
# For any fit whose I-divergence is differenced into a log Bayes factor:
# `_try_add`, `_try_split`, `_prune`. Tight on purpose. A proposal fit starts
# further from its optimum than the incumbent it is scored against, so a loose
# tolerance does not add symmetric noise -- it leaves the proposal's objective
# systematically too high and biases model selection toward the smaller model.
# The margins this has to resolve are real: on a 906-emitter frame the closest
# prune decision sat at log BF = +0.022.
REFINE_TOL_OBJ = 1e-6
# For `refine`. No Bayes factor is built from a refine fit, so the asymmetry
# argument above does not apply to it directly and its own natural unit is not
# nats but PIXELS: near the optimum I(t) ~ I_min + 0.5 dt' F dt, so stopping at
# a predicted decrease of `tol` leaves a parameter about sqrt(2*tol) standard
# errors short. That is the argument for loosening it, and it is measured --
# `rust/spotsolve-core/examples/refine_tol.rs` reproduces the whole table.
#
# But refine does not stand alone: its output is what `_prune` re-tests and
# what the next round's candidates are scored against, so a sloppier refine
# does eventually move DETECTIONS. That is what fixes the value here, and it
# was measured rather than guessed (256 px, N=3169, py-vs-rs band in SE):
#
#     tol_obj    frame s        N     max |dpos|/SE
#       1e-8        9.78     3169            0.0010
#       1e-6        8.66     3169            0.0015   <- here
#       1e-5        7.68     3168            0.1466
#       1e-4        7.08     3161            0.2828
#
# 1e-6 is the last value that leaves N and the two-implementation agreement
# band exactly where 1e-8 does, and it still removes ~11% of frame time and
# ~40% of refine's LM iterations. Below it both move together, which is the
# signature of a numerical tolerance that has started making decisions -- and
# no accuracy column in `bench.py` rewards paying for that.


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
                  link_radius_factor, max_iter, dirty=None, se=None,
                  move_eps=0.0):
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
                                          p.frozen_indices, sigma, yy, xx,
                                          p.y0, p.x0)
        level, shape_ = _window_bg(bmap, p.y0, p.x0, p.y1, p.x1)
        halo = halo + shape_
        sub = np.asarray(d_e[p.y0:p.y1, p.x0:p.x1])
        loc = positions[p.indices] - np.array([p.y0, p.x0])
        theta0 = psf.pack(level, amplitudes[p.indices], loc[:, 0], loc[:, 1])
        r = _fit_window(sub, yy, xx, sigma, halo, theta0, max_iter=max_iter,
                        tol_obj=REFINE_TOL_OBJ)
        _, A, cy, cx = psf.unpack(r.theta)
        out_amp[p.indices] = A
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
        se[p.indices, 0] = np.sqrt(v[1::3])
        se[p.indices, 1] = np.sqrt(v[2::3])
        se[p.indices, 2] = np.sqrt(v[3::3])
    return out_pos, out_amp, se, moved, n_fitted


def refine(d_e, positions, amplitudes, sigma, bmap, k_max=12,
           link_radius_factor=LINK_FACTOR, max_iter=None,
           max_sweeps=REFINE_SWEEPS, tol=REFINE_TOL):
    """Joint re-fit at fixed N in connected groups, plus per-emitter CRLBs.

    Returns (positions, amplitudes, se) with `se` an (N,3) array of
    (SE_A, SE_y, SE_x) from the Fisher matrix of the fit whose parameters are
    reported. The background is not returned: it is a surface owned by the
    caller, re-estimated between rounds.

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
    20), because it is not a slow-iteration problem. `_prune` is the fix, which
    is why the pipeline order is settle, prune, settle.

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
    if max_iter is None:
        max_iter = REFINE_MAX_ITER
    positions = np.atleast_2d(np.asarray(positions, float))
    amplitudes = np.asarray(amplitudes, float).ravel()
    if len(amplitudes) == 0:
        return positions, amplitudes, np.empty((0, 3))

    se = np.full((len(amplitudes), 3), np.nan)
    dirty = np.ones(len(amplitudes), dtype=bool)
    for _ in range(max(1, int(max_sweeps))):
        positions, amplitudes, se, moved, n_fitted = _refine_sweep(
            d_e, positions, amplitudes, sigma, bmap, k_max,
            link_radius_factor, max_iter, dirty=dirty, se=se, move_eps=tol)
        if n_fitted == 0 or not moved.any():
            break
        dirty = moved
    return positions, amplitudes, se


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
    filter. Cached because `detect` calls this once per round at a fixed
    sigma.
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


def find_candidates(d_e, model, sigma, positions, threshold=CAND_THRESHOLD):
    """LoG peaks on the variance-normalized residual, brightest first.

    Runs on the residual of the current model, not on the image, so an emitter
    already modelled is not proposed again; the proximity veto below catches
    what survives on a neighbour's wing.

    Two normalizations, and they do different jobs. Dividing by `sqrt(model)`
    before the filter is what makes one threshold valid across the FRAME:
    under Poisson noise the residual's scale is the square root of the mean,
    so the ratio is on a fixed sigma scale wherever the model puts flux.
    Dividing by `log_kernel_l2` after it is what makes one threshold valid
    across SIGMA: the filter's own null sd is its kernel's L2 norm, which
    scales as sigma^-3, so without this the same numeric threshold is a
    1.8-sigma cut at sigma=0.8 and a 100-sigma cut at sigma=3.0.

    `threshold` is therefore a count of standard deviations, on pixels. See
    `calibrate.LOG_SEED_Z`.
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


AGG_FLUX_RATIO = 10.0   # over-wide cut: fitted flux / the median candidate's
AGG_SIGMA_LO = 0.9      # fitted sigma / PSF sigma: plausibility band, not a
AGG_SIGMA_HI = 5.0      # discriminator -- see `find_aggregates`
AGG_MASK_RADIUS = 3.0   # sigma_fit of excluded support around each aggregate
AGG_FIT_PAD = 7         # px half-window for the free-sigma fit
AGG_SIGMA_MAX = 8.0     # upper bound on the free sigma; also the widest object
                        # this test can describe


AGGREGATE_DTYPE = np.dtype([("y", float), ("x", float), ("sigma", float),
                            ("flux", float), ("radius", float)])


def find_aggregates(d_e, sigma, b0, threshold=CAND_THRESHOLD,
                    flux_ratio=AGG_FLUX_RATIO, sigma_lo=AGG_SIGMA_LO,
                    sigma_hi=AGG_SIGMA_HI, mask_radius=AGG_MASK_RADIUS,
                    pad=AGG_FIT_PAD):
    """OVER-WIDE objects -- wider than the PSF can represent -- and their mask.

    Returns `(records, mask)`: a structured array of `AGGREGATE_DTYPE` and a
    boolean `(H, W)` array that is True where such an object's support lies.

    This is one of three regimes, and the ONLY one that needs a pre-search
    pass. See README section 10b for the vocabulary:

        ordinary point source   sigma_fit ~ sigma,  flux ~ median
        OVER-BRIGHT             sigma_fit ~ sigma,  flux >> median
                                -> representable; `flag_aggregates`, post hoc
        OVER-WIDE               sigma_fit >> sigma
                                -> not representable; THIS function

    Note the word "aggregate" does not decide which one you have -- optics
    does. A sub-diffraction aggregate is over-BRIGHT, not over-wide, and
    belongs to `flag_aggregates`.

    Why this runs BEFORE the search
    -------------------------------
    The model has a FIXED sigma, so a genuinely wider object cannot be one
    wide emitter -- the search tiles it with PSF-sized ones. Measured on three
    over-wide objects (flux 20-60k e-, sigma 2.4-4.0): **84 detections**, and the
    pieces are not identifiable afterwards by any per-detection statistic.
    Their amplitudes point the WRONG WAY (median 1249 e- against 1370 for
    ordinary emitters, with 69 of 84 dimmer than the brightest real one), and
    a post-hoc free-sigma refit gives 1.22 against 1.20 -- because locally a
    tile IS a PSF-sized bump, and the neighbours are frozen into its halo.

    There is also frame-level damage that no downstream filter can undo:
    those three objects inflated `lam`, the density prior inside every
    Bayes factor, by 2.3x (0.00662 -> 0.01552), making the detector more
    permissive across the WHOLE frame. The information needed to reject an
    over-wide object exists only while it is still one object.

    The test
    --------
    A free-sigma fit at each round-0 candidate on the raw frame, accepted as
    an aggregate when

        flux_fit > flux_ratio * median(flux_fit over candidates)   and
        sigma_lo * sigma  <  sigma_fit  <  sigma_hi * sigma

    **Flux is the discriminator; sigma is only a plausibility band.** That
    ordering is the opposite of the obvious one and it comes from real data.
    On `hyp7gem_wt_crop.tif` (sigma 1.45) the four visible bright objects fit
    at sigma 1.47-1.65 -- 1.01 to 1.14 times the PSF width, not wider in any
    useful sense, i.e. over-BRIGHT and not over-wide -- while carrying 65 to
    137 times the median candidate's flux:

        peak e-   sigma_fit   flux/median
          6449      1.59         137
          5078      1.50          95
          4591      1.65          97
          3483      1.47          65

    An aggregate of a few thousand fluorophores is still a sub-diffraction
    object, so it images at the PSF width and is merely BRIGHT. It only looks
    wide on screen because the display saturates. An earlier version of this
    test required `sigma_fit > 1.5 * sigma` and found none of these.

    The sigma band exists to reject fits that are not describing an object at
    all. On the same frame 79 of 246 fits ran to a bound over dim diffuse
    regions, where a wide Gaussian accumulates large flux without any compact
    source under it -- and one of those, not any real aggregate, was what the
    earlier version reported. Bound-hitting fits are discarded outright and
    the band catches the rest.

    KNOWN FAILURE: a dense cluster reads as one over-wide object
    ------------------------------------------------------------
    **This test misfires on crowded fields, and `beads_60x_still_02.tif` is
    one.** There it reports 2 objects and masks 24.8% of the frame; the masked
    region contains 37 real beads whose amplitudes (median 978 e-) are
    indistinguishable from those outside it (939 e-). The larger record fits
    sigma=6.99 over a region the residual audit independently flags as
    piled-up PSFs.

    That frame provably contains nothing this pass should fire on: its flux
    distribution is unimodal, max/median 1.58, with NO detection above 2x the
    median (measured 2026-08-29; `beads_60x_still.tif` gives 1.34, likewise
    none above 2x). There are no aggregates in the bead data -- a bright
    diffraction-limited spot there is one bead, or beads sitting very close
    together, which would show as an over-BRIGHT detection at ~2x and never as
    a wide one.

    The cause is the identifiability wall this whole pipeline lives against: a
    single wide Gaussian fits "many close point sources" exactly as well as
    "one genuinely wide object", and the flux condition cannot separate them
    either, because 37 beads at ~950 e- carry an over-wide object's total flux.

    So this is safe for ISOLATED over-wide objects on a sparse-to-moderate
    field (measured: 3/3 found, parameters recovered to within 5%, clean fields
    a strict no-op) and NOT safe on a dense field without a further test. The
    discriminator that should work, and is not implemented: fit the wide
    Gaussian, then look for PSF-scale structure in what is left. A genuinely
    extended object leaves a smooth residual; a bead cluster leaves point-like
    peaks that `audit.score_map` already detects. Until that exists, read
    `result.aggregates` and check it against the image before trusting a run
    with this enabled -- which is why it reports rather than silently drops,
    and why it is off by default.
    """
    d_e = np.asarray(d_e, float)
    H, W = d_e.shape
    empty = np.empty(0, dtype=AGGREGATE_DTYPE)
    cand, _, _ = find_candidates(d_e, np.full((H, W), b0), sigma,
                                 np.empty((0, 2)), threshold)
    if len(cand) == 0:
        return empty, np.zeros((H, W), bool)

    rows = []
    for cy, cx in cand:
        y0, y1 = max(0, int(cy) - pad), min(H, int(cy) + pad + 1)
        x0, x1 = max(0, int(cx) - pad), min(W, int(cx) + pad + 1)
        if y1 - y0 < 6 or x1 - x0 < 6:
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        sub = d_e[y0:y1, x0:x1]
        a0 = max(float(sub.max()) - b0, 1.0) / psf.peak_factor(sigma)
        theta0 = np.concatenate([psf.pack(b0, [a0], [cy], [cx]), [sigma]])
        lo = np.array([1e-3, 1e-3, cy - 1.5, cx - 1.5, 0.3])
        hi = np.array([max(b0 * 20.0, 20.0), a0 * 50.0, cy + 1.5, cx + 1.5,
                       AGG_SIGMA_MAX])
        res = lmga.fit(np.clip(theta0, lo, hi), gy.astype(float),
                       gx.astype(float), sigma, sub, lo, hi, halo=0.0,
                       max_iter=80, free_sigma=True)
        if not res.converged:
            continue
        # A fit that walked to its POSITION bound was not describing the
        # object under the candidate -- it was sliding down an aggregate's
        # skirt towards the real centre, and its sigma is a measure of how far
        # it got, not of any object's width. Those fits are what produced
        # spurious wide records on the flanks of genuine aggregates.
        at_bound = (abs(res.theta[2] - cy) >= 1.5 - 1e-6
                    or abs(res.theta[3] - cx) >= 1.5 - 1e-6
                    or res.theta[-1] >= AGG_SIGMA_MAX - 1e-6)
        if at_bound:
            continue
        rows.append((res.theta[2], res.theta[3], res.theta[-1], res.theta[1]))
    if not rows:
        return empty, np.zeros((H, W), bool)

    rows = np.array(rows)                     # (M, 4): y, x, sigma_fit, flux
    med_flux = float(np.median(rows[:, 3]))
    hit = ((rows[:, 3] > flux_ratio * med_flux)
           & (rows[:, 2] > sigma_lo * sigma) & (rows[:, 2] < sigma_hi * sigma))
    if not hit.any():
        return empty, np.zeros((H, W), bool)

    # One aggregate raises several LoG maxima, so several candidates fit the
    # SAME object. Merge them brightest-first: a hit inside an already-accepted
    # record's support is that record, not a second aggregate. Without this a
    # 3-aggregate frame reported 8 records and masked 29% of itself.
    keep = rows[hit]
    keep = keep[np.argsort(-keep[:, 3])]
    merged = []
    for row in keep:
        ry, rx, rs = row[0], row[1], row[2]
        if any((ry - m[0]) ** 2 + (rx - m[1]) ** 2
               <= (mask_radius * max(rs, m[2])) ** 2 for m in merged):
            continue
        merged.append(row)
    keep = np.array(merged)

    rec = np.empty(len(keep), dtype=AGGREGATE_DTYPE)
    rec["y"], rec["x"] = keep[:, 0], keep[:, 1]
    rec["sigma"], rec["flux"] = keep[:, 2], keep[:, 3]
    rec["radius"] = mask_radius * keep[:, 2]

    mask = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    for r in rec:
        # Stamped over the disc's own bounding box, not the whole frame.
        r0 = int(np.ceil(r["radius"]))
        y0, y1 = max(0, int(r["y"]) - r0), min(H, int(r["y"]) + r0 + 1)
        x0, x1 = max(0, int(r["x"]) - r0), min(W, int(r["x"]) + r0 + 1)
        d2 = ((yy[y0:y1, x0:x1] - r["y"]) ** 2
              + (xx[y0:y1, x0:x1] - r["x"]) ** 2)
        mask[y0:y1, x0:x1] |= d2 <= r["radius"] ** 2
    return rec, mask


AGG_AMP_RATIO = 20.0    # over-bright cut: flux / the frame's median detection
AGG_LINK = 3.0          # sigma; flagged detections within this are one object


AGGREGATE_FLAG_DTYPE = np.dtype([("y", float), ("x", float), ("flux", float),
                                 ("ratio", float), ("n", int)])


def flag_aggregates(result, ratio=AGG_AMP_RATIO, min_flux=None,
                    link=AGG_LINK):
    """Flag OVER-BRIGHT detections in a finished result, by flux alone.

    Returns `(mask, objects)`: a per-detection boolean over
    `result.positions`, and a structured array of `AGGREGATE_FLAG_DTYPE` with
    one row per linked object -- flux-weighted centroid, summed flux, the
    brightest member's ratio to the frame median, and how many detections it
    absorbed.

    An over-bright detection is one at the PSF width (sigma_fit ~ sigma)
    carrying flux far above the frame median. That is a statement about the
    PICTURE, not about the object -- see "what over-bright can mean" below.
    The other regime, sigma_fit >> sigma, is `find_aggregates`.

    Why flux is the ONLY signal
    ---------------------------
    Sigma cannot work here, and not because the fit is poor. The diffraction
    limit is ~200 nm, so **every object smaller than that images at exactly
    the PSF sigma**: a single GFP is ~3 nm, an aggregate of thousands of them
    may still be ~200 nm, and two beads 100 nm apart are a 200 nm object --
    all three are the same width on the camera. Size information below the
    limit is not attenuated, it is absent. What differs is how many
    fluorophores are in the spot, and that is flux.

    Measured on `hyp7gem_wt_crop.tif` (sigma 1.45): the visible aggregates fit
    sigma 1.47-1.65, i.e. 1.01-1.14x the PSF, while their detected cores run
    33000-63000 e- against a median detection of 378 -- a separation of about
    100-170x in flux and essentially none in width.

    What over-bright can mean, and what this function does NOT decide
    ----------------------------------------------------------------
    At least two physically different situations give the same picture -- a
    PSF-width spot with n times the usual flux:

      * an UNRESOLVED MULTIPLE: n ordinary sources within ~1 sigma, fitted as
        one. Ratio ~ n, a small integer, readable as a count only when the
        population is near-monodisperse (beads, or one fluorophore species).
      * a SUB-DIFFRACTION AGGREGATE: one object below the limit holding many
        fluorophores. Ratio in the tens to hundreds.

    Nothing here separates them, and no per-detection statistic can: a pair
    closer than 1 sigma is not identifiable. The MAGNITUDE of the ratio is the
    only evidence -- hyp7gem's 100-170x is no coincidence of ordinary sources;
    a 2x is almost certainly two of them (and is the flux-side view of the
    open problem in README section 15, where the survivor of an unresolved
    pair reports a confident SE and has absorbed both fluxes).

    The `ratio=20` default is set to catch aggregates. It will NOT find
    unresolved multiples, which sit at ~2x, inside the ordinary flux spread of
    most frames. On the bead data the question does not arise: measured
    2026-08-29, both frames are unimodal with max/median 1.34 and 1.58 and
    nothing above 2x, so there are no aggregates AND no unresolved multiples
    there, and this function correctly flags nothing on either.

    Why this is post hoc, unlike `find_aggregates`
    ----------------------------------------------
    A PSF-width object is REPRESENTABLE by the fixed-sigma model, so the
    search fits it as one bright emitter (plus a few small neighbours) and it
    stays identifiable afterwards. Nothing needs to be excluded before the
    search, and excluding it hurts: freezing a sigma~1.5 object into `bmap`
    took the residual audit from z in [-7.0, 10.3] to [-26.6, 10.3] and raised
    N from 689 to 715, because the frozen object double-counts against
    emitters fitted beside it.

    `find_aggregates` remains the right tool for the other case -- an object
    genuinely WIDER than the PSF, which the model cannot represent and
    therefore tiles into ~25 pieces that no post-hoc statistic can recover.
    Choose by measuring: `sigma_fit / sigma` near 1 means use this function;
    much greater than 1 means use `find_aggregates`.

    `ratio` is against the frame's own median detection, so the cut is
    unitless and transfers between exposures and datasets. `min_flux` sets an
    absolute floor instead, when the frame's median is itself unreliable (very
    few detections, or a frame that is mostly aggregate).
    """
    pos = np.atleast_2d(np.asarray(result.positions, float))
    amp = np.asarray(result.amplitudes, float).ravel()
    empty = np.empty(0, dtype=AGGREGATE_FLAG_DTYPE)
    if amp.size == 0:
        return np.zeros(0, bool), empty

    cut = float(min_flux) if min_flux is not None \
        else ratio * float(np.median(amp))
    mask = amp > cut
    if not mask.any():
        return mask, empty

    idx = np.nonzero(mask)[0]
    # One aggregate can raise several flagged detections -- the bright core
    # plus a neighbour or two. Link them so the count is objects, not
    # detections, which is what a per-frame quality metric wants.
    groups = _link_groups(pos[idx], link * float(result.sigma))
    med = float(np.median(amp))
    objs = np.empty(len(groups), dtype=AGGREGATE_FLAG_DTYPE)
    for k, g in enumerate(groups):
        sel = idx[g]
        w = amp[sel]
        objs[k] = (float(np.average(pos[sel, 0], weights=w)),
                   float(np.average(pos[sel, 1], weights=w)),
                   float(w.sum()), float(w.max() / max(med, 1e-12)), len(sel))
    objs = objs[np.argsort(-objs["flux"])]
    return mask, objs


def _link_groups(points, radius):
    """Single-linkage groups of `points` within `radius`, as index lists."""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [np.array([0])]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return [np.array([i]) for i in range(n)]
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                     shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    return [np.nonzero(lab == c)[0] for c in np.unique(lab)]


def aggregate_report(result, ratio=AGG_AMP_RATIO, min_flux=None,
                     link=AGG_LINK):
    """`flag_aggregates` reduced to per-frame quality numbers.

    `flux_fraction` is the share of ALL detected flux sitting in aggregates,
    which is the number that says whether a frame's emitter statistics mean
    anything: a frame with 2% of its flux in aggregates is a frame to analyse,
    one with 60% is a frame to look at.
    """
    mask, objs = flag_aggregates(result, ratio, min_flux, link)
    amp = np.asarray(result.amplitudes, float).ravel()
    total = float(amp.sum())
    return dict(n_aggregates=len(objs),
                n_detections_flagged=int(mask.sum()),
                n_detections=int(amp.size),
                flux_in_aggregates=float(amp[mask].sum()) if mask.any() else 0.0,
                flux_fraction=(float(amp[mask].sum()) / total
                               if total > 0 and mask.any() else 0.0),
                median_flux=float(np.median(amp)) if amp.size else float("nan"),
                cut=(float(min_flux) if min_flux is not None
                     else ratio * float(np.median(amp)) if amp.size
                     else float("nan")),
                objects=objs)


def render_aggregates(rec, shape):
    """The fitted aggregates as an image, to be FROZEN into the model.

    Masking an aggregate's candidates is not enough on its own: its flux is
    still in the image and still unexplained, so SPLIT subdivides it inward
    from the mask boundary and ADD re-seeds wherever the residual pokes out.
    Measured before this existed -- 68 of 120 detections were still inside the
    mask, carrying 112742 e- against the aggregates' true 110000. The search
    was faithfully modelling the aggregate, one PSF at a time, exactly as it
    is designed to.

    Adding this to `bmap` instead makes the aggregate a KNOWN additive term.
    Every window fit already splits `bmap` into a free level and a frozen
    shape (`_window_bg`), so the aggregate is accounted for by every fit with
    no change to any pass -- and the residual over it is flat, so nothing is
    proposed there in the first place.
    """
    H, W = int(shape[0]), int(shape[1])
    out = np.zeros((H, W))
    if rec is None or len(rec) == 0:
        return out
    for a in rec:
        # Each aggregate has its OWN sigma, so they cannot be rendered in one
        # call -- `calibrate.render_model` takes a single sigma for all.
        out += calibrate.render_model(np.array([[a["y"], a["x"]]]),
                                      np.array([a["flux"]]), float(a["sigma"]),
                                      (H, W), 0.0)
    return out


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


A_MIN_REL = 1e-6
# The amplitude floor a fit may not go below, as a fraction of the window's own
# `A_max`; `moves.A_MIN` is the absolute backstop. The floor has to be relative
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
# may be, and that decision belongs to `_prune` and the Bayes factor, not to a
# numerical guard. The absolute floor is only a backstop for a window whose
# `A_max` is itself tiny.
#
# Note this cannot be enforced downstream in `evidence.logdet_cond` instead:
# no test on F alone distinguishes an uninformed parameter from a well-posed
# matrix in badly scaled units. Here the flux scale is known, so it can.


def _bounds(K, h, w, b_max, A_max):
    """Box constraints for one window's fit, in LOCAL coordinates.

    Positions are confined to the sub-image. Letting a centre leave the frame
    was tried -- it lets the fit put rim flux where it actually came from --
    and measurably lost real detections elsewhere, so the bounds stay closed.
    """
    a_min = max(moves.A_MIN, A_MIN_REL * A_max)
    lo = [0.0]
    hi = [b_max]
    for _ in range(K):
        lo += [a_min, -0.5, -0.5]
        hi += [A_max, h - 0.5, w - 0.5]
    return np.asarray(lo), np.asarray(hi)


def _fit_window(sub, yy, xx, sigma, halo, theta0, max_iter=100,
                tol_obj=EVIDENCE_TOL_OBJ):
    K = (len(theta0) - 1) // 3
    smax = max(float(sub.max()), 1.0)
    b_max = max(smax * 4.0, 10.0)
    A_max = 8.0 * smax / psf.peak_factor(sigma)
    lo, hi = _bounds(K, sub.shape[0], sub.shape[1], b_max, A_max)
    th0 = np.clip(np.asarray(theta0, float), lo + 1e-9, hi - 1e-9)
    return lmga.fit(th0, yy, xx, sigma, sub, lo, hi, halo=halo,
                    max_iter=max_iter, tol_obj=tol_obj)


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


def _add_pass(d_e, bmap, positions, amplitudes, cand, camp, sigma, lam, A_s,
              k_max):
    """One ADD pass over a candidate list. Returns (pos, amp, n_added).

    The proximity re-check is part of the pass rather than of `detect`: it reads
    the positions an earlier acceptance in THIS pass has already written, so an
    emitter accepted a moment ago can claim a later candidate's flux.
    """
    n_added = 0
    for c, a in zip(cand, camp):
        if len(positions) and np.min(np.linalg.norm(
                positions - c, axis=1)) <= sigma:
            continue
        ok, positions, amplitudes = _try_add(
            d_e, positions, amplitudes, bmap, c, a, sigma, lam, A_s, k_max)
        n_added += int(ok)
    return positions, amplitudes, n_added


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


def _update_bg(d_e, positions, amplitudes, sigma, bmap, kernel, be=None,
               exclude=None):
    """Re-estimate the background from the current emitter model.

    `kernel=None` gives one scalar for the frame, from the pixels no emitter
    reaches; any integer estimates a surface on that window.
    """
    # One mask, two consumers. `robust_background` and `background_map` mask on
    # the same radius, so stamping it once here removes a full O(N*H*W) sweep
    # per round -- 41% of a 512x512 frame before this, and the largest single
    # cost left in the Python that stayed behind the port [P9].
    if be is None:
        free = calibrate.emitter_free_mask(bmap.shape, positions, sigma,
                                           BG_MASK_RADIUS)
    else:
        free = be.emitter_free_mask(positions, sigma, bmap.shape,
                                    BG_MASK_RADIUS)
    if exclude is not None:
        # An aggregate's pixels are not background and not a fitted emitter.
        # Leaving them in pulls the local surface up towards the aggregate,
        # which is exactly the flux we are trying to keep out of the model.
        free = free & ~exclude
    scalar = max(calibrate.robust_background(d_e, positions, sigma,
                                             BG_MASK_RADIUS, free=free),
                 BG_FLOOR)
    if kernel is None:
        return np.full(bmap.shape, scalar)
    return background_map(d_e, positions, sigma, kernel=kernel,
                          fallback=scalar, free=free)


def detect(data_img, sigma=1.2, offset=0.0, gain=None, lam0=0.02, A_s0=None,
           k_max=12, max_rounds=6, threshold=CAND_THRESHOLD, prune=True,
           split=True, bg_kernel=BG_KERNEL, max_settle=4, verbose=1,
           impl="py", reject_aggregates=False,
           agg_flux_ratio=AGG_FLUX_RATIO, agg_sigma_lo=AGG_SIGMA_LO,
           agg_sigma_hi=AGG_SIGMA_HI, agg_mask_radius=AGG_MASK_RADIUS):
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

    `impl` selects which implementation runs the four passes: "py" for the
    reference in this file, "rs" for the Rust extension. Everything else -- this
    round loop, `find_candidates`, and `background_map`'s convolutions -- is the
    same code either way, which is what makes the two comparable. See
    `backend.py`.

    `reject_aggregates` runs `find_aggregates` once before the search and
    excludes what it finds from candidates, from the background estimate and
    from `lam`'s area. What it excluded is REPORTED in `result.aggregates`,
    never silently dropped. It is **off by default**: the test flags about 6%
    of ordinary candidate peaks as well (close pairs, which one wide Gaussian
    also fits), so turning it on trades a little recall for immunity to an
    error that is otherwise unbounded -- and that trade should be a decision,
    not a default. Turn it on only for frames that actually contain an
    OVER-WIDE object (sigma_fit >> sigma at the bright sites) -- never for
    over-bright, PSF-width ones, where it does measurable harm; see
    `flag_aggregates`. Read `result.aggregate_fraction` to see how much of the
    frame went.
    """
    be = backend_mod.get(impl)
    raw = np.asarray(data_img, dtype=float)
    H, W = raw.shape
    g_eff = calibrate.estimate_gain(raw, offset) if gain is None else float(gain)
    d_e = (raw - offset) / g_eff

    b0 = float(np.percentile(d_e, 10.0))

    # BEFORE the search: an aggregate is still one object here. After it, the
    # fixed-sigma model will have tiled it into pieces no per-detection
    # statistic can identify. See `find_aggregates`.
    agg_rec, agg_mask = None, None
    if reject_aggregates:
        agg_rec, agg_mask = find_aggregates(
            d_e, sigma, max(b0, BG_FLOOR), threshold=threshold,
            flux_ratio=agg_flux_ratio, sigma_lo=agg_sigma_lo,
            sigma_hi=agg_sigma_hi, mask_radius=agg_mask_radius)
        if not agg_mask.any():
            agg_mask = None
        if verbose >= 1 and agg_rec is not None and len(agg_rec):
            frac = 0.0 if agg_mask is None else float(agg_mask.mean())
            print(f"  [aggregates] {len(agg_rec)} found, {100*frac:.1f}% of "
                  f"the frame masked; sigma "
                  f"{agg_rec['sigma'].min():.2f}-{agg_rec['sigma'].max():.2f}, "
                  f"flux {agg_rec['flux'].max():.0f} max")

    # lam is emitters per usable px^2, so the masked area must come out of the
    # denominator -- otherwise excluding a region lowers the density prior and
    # quietly makes the Bayes factor stricter everywhere else.
    usable_px = float(H * W if agg_mask is None else (~agg_mask).sum())
    usable_px = max(usable_px, 1.0)

    # Frozen, and kept SEPARATE from the estimated background so that
    # `_update_bg` re-estimating the surface each round cannot wipe it.
    agg_model = render_aggregates(agg_rec, (H, W))

    bmap = np.full((H, W), max(b0, BG_FLOOR)) + agg_model
    if A_s0 is None:
        A_s0 = max(float(d_e.max()) - b0, 10.0) / psf.peak_factor(sigma)
    lam, A_s = lam0, A_s0

    positions = np.empty((0, 2))
    amplitudes = np.empty(0)
    history = []

    for rnd in range(max_rounds):
        model = bmap + be.render_model(positions, amplitudes, sigma,
                                       bmap.shape, 0.0)
        cand, camp, _ = find_candidates(d_e, model, sigma, positions, threshold)
        if agg_mask is not None and len(cand):
            inside = agg_mask[cand[:, 0].astype(int), cand[:, 1].astype(int)]
            cand, camp = cand[~inside], camp[~inside]
        positions, amplitudes, n_added = be.add_pass(
            d_e, bmap, positions, amplitudes, cand, camp, sigma, lam, A_s,
            k_max)

        # SPLIT runs on the model the adds just produced: an emitter only looks
        # like an unresolved pair once its neighbourhood is otherwise
        # explained, and splitting against a model still missing a nearby
        # source mostly splits emitters into that source's flux.
        n_split = 0
        if split:
            if n_added:
                model = bmap + be.render_model(positions, amplitudes, sigma,
                                               bmap.shape, 0.0)
            positions, amplitudes, n_split = be.split_pass(
                d_e, bmap, positions, amplitudes, model, sigma, lam, A_s,
                k_max)

        if n_added or n_split:
            # One sweep here; the round loop is the outer iteration.
            positions, amplitudes, _ = be.refine(
                d_e, positions, amplitudes, sigma, bmap, k_max, 1)
            lam = max(len(positions) / usable_px, 1e-6)
            if len(amplitudes):
                A_s = max(float(np.mean(amplitudes)), 1.0)
            # Re-estimated only AFTER the emitters have been re-fitted, and
            # only from the emitter model with no background in it.
            bmap = _update_bg(d_e, positions, amplitudes, sigma, bmap,
                              bg_kernel, be=be, exclude=agg_mask) + agg_model

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
    positions, amplitudes, se = be.refine(d_e, positions, amplitudes, sigma,
                                          bmap, k_max, REFINE_SWEEPS)

    # Settle and prune alternate until the prune removes nothing. One pass is
    # not enough: the re-fit at the reduced N is as free to collapse a pair as
    # the first one was. This cannot cycle -- prune only removes, so N strictly
    # decreases. `max_settle` is a backstop.
    for _ in range(max_settle if prune else 0):
        if not len(positions):
            break
        n_before = len(positions)
        positions, amplitudes, _ = be.prune(d_e, bmap, positions, amplitudes,
                                            sigma, lam, A_s, k_max)
        if len(positions) == n_before:
            break
        if verbose >= 1:
            print(f"  [prune] {n_before} -> {len(positions)}")
        positions, amplitudes, se = be.refine(d_e, positions, amplitudes,
                                              sigma, bmap, k_max,
                                              REFINE_SWEEPS)
    model = bmap + be.render_model(positions, amplitudes, sigma, bmap.shape,
                                   0.0)
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
        aggregates=agg_rec,
        aggregate_fraction=(0.0 if agg_mask is None else float(agg_mask.mean())),
    )
