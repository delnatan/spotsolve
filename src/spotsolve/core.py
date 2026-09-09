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
from . import prior as prior_mod
from . import psf
from .structs import DetectResult, width_reject_records

__all__ = ["detect", "refine", "find_candidates", "background_map",
           "log_kernel_l2", "flag_aggregates", "aggregate_report"]


LINK_FACTOR = 2.5   # emitters within this many sigma are fitted jointly
HALO_FACTOR = 5.0   # ... and beyond it are frozen into the model as a constant
BBOX_PAD = 3.0      # sigma of pixel context around a group's extreme emitters
SEED_ALPHA = calibrate.SEED_ALPHA
CAND_THRESHOLD = calibrate.LOG_SEED_Z
# The historical constant. `detect(threshold=None)` DERIVES the cut from the
# frame and the PSF instead -- see `calibrate.seed_threshold`, and note that it
# is not a constant at all: it scales with frame size and sigma, which is why
# carrying one number silently gave every instrument a different detector.
# This name survives so the old behaviour stays reachable and attributable.

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
# 2.0 was the measured optimum on both bead frames -- under the fixed-width,
# single-class, ML-fitted pipeline, all three of which have since changed.
# RE-SWEPT on the confocal simulation under the current algorithm (moderate,
# 6 frames):
#
#   tau   recall   med err    RMSE   rsd z   |z|>3   tiles
#   1.5    93.4%     0.077   0.217    1.30    6.1%    3.33
#   2.0    92.6%     0.076   0.220    1.25    7.0%    2.33
#   2.5    92.3%     0.076   0.220    1.29    8.1%    2.83
#   3.0    89.4%     0.079   0.224    1.39   10.4%    1.67
#
# 2.0 survives: best pull spread, fewest tiles, and within a point of 1.5's
# recall. 3.0 is clearly wrong. This matters beyond the dial itself --
# `backend.RustBackend` asserts on this constant at load, so a stale value
# would be frozen into a second implementation.

SIGMA_SLACK = (0.70, 2.2)
FOCUS_BAND = (0.80, 2.0)
# `SIGMA_SLACK` is the MODEL SPACE: the widths a fit may represent, as
# multiples of the PSF sigma. `FOCUS_BAND` is the REPORTING BAND: the widths
# that count as an in-focus detection. `None` for the slack fixes every emitter
# at the PSF width, which is what the pipeline did before 2026-09-03.
#
# Until 2026-09-03 these were ONE constant, `(0.95, 2.0)`, and that conflation
# is what made the detector tile defocused sources. The model space has to
# cover every photon on the sensor or the light it cannot represent gets tiled;
# the reporting band is a downstream contract about what the caller is handed.
# Clipping the first to the second manufactured the tiles. What makes the two
# affordable at once is the MIXTURE width prior -- see `prior.WidthPrior`,
# which carries the argument and the arithmetic. In short: under one uniform
# prior over [0.95, 8.0] every in-focus add would pay 1.9 nats for the
# enlargement, and a wide fit swallowing a real neighbour would pay almost
# nothing for the privilege; under the mixture an in-focus add pays exactly
# what it paid before, and a wide one pays against its own rarer rate.
#
# The band's edges are asymmetric in KIND, which is why they are enforced in
# different places. `FOCUS_BAND[1]` is a class boundary between two populations
# that are both real, so it lives in the prior and is decided during the
# search, while the flux is still undivided. `FOCUS_BAND[0]` is not a class:
# nothing images narrower than the PSF, so a fit below it is a fit that has
# BROKEN, and there is no information about that which the search destroys.
# It is a post-fit check on the survivors, and `SIGMA_SLACK[0]` sits below it
# so that a broken fit can reveal itself instead of being clipped to the bound
# and reported. A post-hoc filter used to make the same two cuts with the same
# two values, after the search rather than during it; see README section 13 for
# why that could not work and where it went.
#
# Why the model needs slack at all. A fixed-sigma model meets a source it
# cannot represent -- anything out of focus -- by TILING it: the residual a
# single narrow PSF leaves on a broad blob has lobes, SPLIT proposes into
# them, and the evidence correctly prefers two narrow Gaussians to one,
# because two narrow Gaussians genuinely do fit a wide blob better. The
# decision rule is right and the model space is wrong, so no threshold fixes
# it. Measured on a confocal simulation with emitters uniform in +/- 0.5 um
# (`scripts/bench_sim.py`), at 1 emitter/um^2 the fixed-sigma search returned
# 7.2 extra detections per frame against 13.4 in-focus emitters -- and every
# one of them sat within 3 sigma(z) of a REAL emitter. Not one was invented.
#
# The band's UPPER edge is how far out of focus an emitter may still be
# reported as a detection. In this simulation a point source images at 1.26x
# the in-focus width at |z| = 0.25 um, 1.95x at 0.35 and 3.2x at 0.50, so 2.0
# is everything inside |z| ~ 0.36 um.
#
# The MODEL's upper bound is a different question -- what "one object" can
# still mean. Beyond that the thing is not a defocused point source and
# belongs to the background, or to `find_aggregates` if it is bright enough to
# be worth excluding by hand.
#
# The sweep below is why those two cannot be the same number, and it is the
# measurement that used to be read as an argument for the hard bound at 2.0.
# It was taken with a SINGLE uniform width prior, where raising the bound
# charges a wide fit only log(2.25/1.05) = 0.76 nats for the whole enlargement
# -- so a wide emitter swallowing a genuine close pair was charged essentially
# nothing, and the residual degraded monotonically past 2.0 for that reason
# and not because 2.0 is where the optics stop. 4 frames per density, sweeping
# the bound alone (`residrsd` is the normalized residual's robust spread, 1.0
# being a model that explains the frame):
#
#            sparse                 moderate               dense
#   hi    tile  recall residrsd  tile recall residrsd  tile recall residrsd
#  fixed  9.50   98.0%   1.165   42.0  96.3%   1.240   79.2  92.1%   1.153
#   1.3   4.75   98.0%   1.144   20.2  92.3%   1.235   27.8  85.8%   1.315
#   1.6   2.00   98.0%   1.142   10.8  93.9%   1.229   14.5  85.2%   1.372
#   2.0   1.75   96.1%   1.128    8.8  93.5%   1.211   15.5  85.1%   1.288
#   2.5   1.75   98.0%   1.138    6.2  93.9%   1.239   10.8  81.9%   1.420
#   3.2   1.00   96.1%   1.170    3.5  91.9%   1.318   12.0  79.8%   1.680
#
# 2.0 was the best residual at every density and the best pull tail at two of
# three. Read again with the mixture prior in hand, what this table measures
# is the cost of an UNPRICED enlargement, not the location of a physical edge:
# the tiling column falls monotonically all the way to 3.2 -- the model space
# doing its job -- while the residual turns over at 2.0, which is the wide
# class going unpaid for.
#
# RE-SWEPT under the MAP width fit, which is the pipeline this constant now
# lives in (moderate arm, 6 frames, band fixed at (0.8, 2.0)):
#
#    hi    recall   med err    RMSE   rsd z   |z|>3   tiles
#   2.2     92.6%     0.076   0.220    1.25    7.0%    2.33
#   2.6     91.1%     0.078   0.201    1.31    8.0%    2.33
#   3.2     90.8%     0.078   0.183    1.27    8.7%    4.00
#   4.0     90.5%     0.078   0.194    1.32    8.1%    2.50
#
# 2.2 survives, and the prediction that the MAP fit would let the bound rise
# toward the optics' own 3.2 was WRONG: recall falls monotonically past 2.2
# while RMSE improves, which is a larger model space buying better parameters
# for the objects it keeps and losing the close neighbours it swallows. The
# trade does not reverse; it just gets priced honestly.
#
# Note what the sweep also says, and section 15 records: most of the recall
# cost at high density is paid by the FIRST step away from a fixed width --
# 1.3 already costs 6 points at 10 emitters/um^2 while only halving the
# tiling. That is not the bound's fault, it is the search order's; see
# "A widened emitter can hide its own neighbour".
#
# The trade is priced by the evidence rather than assumed: the extra width
# parameter pays a Laplace dimension, an Occam factor and its own width
# prior's mass on every emitter that carries it.

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
    """Full model image: the background surface plus every emitter's PSF.

    `sigma` is a scalar or one width per emitter."""
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


def _refine_sweep(d_e, positions, amplitudes, sigmas, sigma, bmap, k_max,
                  link_radius_factor, max_iter, dirty=None, se=None,
                  move_eps=0.0, slack=None, wprior=None):
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
    stride = _stride(slack)
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
            max_iter=max_iter, tol_obj=REFINE_TOL_OBJ, wprior=wprior)
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
        # (A, y, x) are the first three of each emitter's block in BOTH
        # layouts, so only the stride changes; a free width is the fourth and
        # is reported separately, not folded into `se`.
        se[p.indices, 0] = np.sqrt(v[1::stride])
        se[p.indices, 1] = np.sqrt(v[2::stride])
        se[p.indices, 2] = np.sqrt(v[3::stride])
    return out_pos, out_amp, out_sig, se, moved, n_fitted


def refine(d_e, positions, amplitudes, sigma, bmap, k_max=12,
           link_radius_factor=LINK_FACTOR, max_iter=None,
           max_sweeps=REFINE_SWEEPS, tol=REFINE_TOL, sigmas=None, slack=None,
           wprior=None):
    """Joint re-fit at fixed N in connected groups, plus per-emitter CRLBs.

    Returns (positions, amplitudes, se, sigmas) with `se` an (N,3) array of
    (SE_A, SE_y, SE_x) from the Fisher matrix of the fit whose parameters are
    reported, and `sigmas` the per-emitter widths -- all at `sigma` unless
    `slack` let them move. `wprior` makes those widths MAP estimates under
    that prior instead of ML ones, and is where most of the free width's cost
    to a close pair is repaid -- see `_WidthPenalty`. The background is not
    returned: it is a surface owned by the caller, re-estimated between
    rounds.

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
            slack=slack, wprior=wprior)
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


def veto_radius(sigma, sigmas, band):
    """How close a candidate may come to an incumbent before it is refused.

    The incumbent's OWN width, because a defocused source fitted at 2 sigma
    reaches twice as far and a candidate 1.2 px off its centre is a piece of
    it -- but CAPPED at the reporting band's upper edge, which is not a detail.
    The veto exists to stop one object being proposed twice, and that is a
    question about RESOLVABILITY. Past the band the object is not a point
    source any more, and its centre has no claim on a peak that a linear
    detector still resolves on top of it: measured at 5 emitters/um^2, letting
    the radius run to the model's full 8 sigma cost 8.3 points of in-focus
    recall, because a wide object vetoed a disc big enough to hold several real
    emitters. Nothing is lost by capping it -- what suppresses tiles is the
    model being able to REPRESENT the broad source, not the veto.
    """
    if sigmas is None:
        return sigma
    r = np.asarray(sigmas, dtype=float)
    return r if band is None else np.minimum(r, band[1] * sigma)


def find_candidates(d_e, model, sigma, positions,
                    threshold=CAND_THRESHOLD, sigmas=None, band=None):
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
        d = np.linalg.norm(
            cand[:, None, :] - np.atleast_2d(positions)[None, :, :], axis=-1)
        radius = veto_radius(sigma, sigmas, band)
        keep = ~np.any(d <= np.atleast_1d(radius)[None, :], axis=1)
        cand, strength = cand[keep], strength[keep]
        ys, xs = ys[keep], xs[keep]
    if len(cand) == 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0)

    amp = np.maximum(resid[ys, xs], 1e-2) / psf.peak_factor(sigma)
    order = np.argsort(-strength)
    return cand[order], amp[order], strength[order]


AGG_MASK_RADIUS = 3.0   # sigma_fit of excluded support around each aggregate
                        # this test can describe


AGGREGATE_DTYPE = np.dtype([("y", float), ("x", float), ("sigma", float),
                            ("flux", float), ("radius", float)])


def _wide_records(pos, amp, sig):
    """The nuisance class as `AGGREGATE_DTYPE` rows.

    Wide objects go back in the same field `find_aggregates` uses, because a
    caller asking "what did this frame contain that is not a point emitter"
    wants one answer, not two. What differs is only WHEN each was decided,
    which `history` records.
    """
    rec = np.zeros(len(sig), dtype=AGGREGATE_DTYPE)
    if len(sig):
        rec["y"], rec["x"] = pos[:, 0], pos[:, 1]
        rec["sigma"], rec["flux"] = sig, amp
        rec["radius"] = AGG_MASK_RADIUS * sig
    return rec


def _window(positions, cand, sigma, shape, k_max, sigmas=None):
    """(free_indices, frozen_indices, bbox) for the local fit around `cand`.

    Free: existing emitters close enough that adding `cand` changes their
    estimates, capped at `k_max - 1` nearest so the joint Fisher matrix stays
    small. Frozen: everything else near enough to contribute flux, folded in
    as a constant. Both radii are `patches.py`'s.

    `BBOX_PAD = 3 sigma` captures 100% of an isolated emitter's position
    information and ~90% of its amplitude information; widening it changes no
    measured outcome and costs runtime linearly in window area.

    `sigmas` is the per-emitter width array when widths are free; every radius
    then scales with the width of the emitter it is measured from, since that
    is what sets how far its flux actually reaches.
    """
    n = len(positions)
    widths = (np.full(n, float(sigma)) if sigmas is None
              else np.asarray(sigmas, dtype=float))
    if n == 0:
        free = np.empty(0, dtype=int)
    else:
        d = np.linalg.norm(np.atleast_2d(positions) - cand, axis=1)
        near = np.argsort(d)
        # Each radius uses the LARGER of the two widths involved. A defocused
        # emitter reaches further, so grouping it at the in-focus width would
        # leave its flux out of both the joint fit and the frozen halo -- an
        # unmodelled pedestal, which section 6 measured as the one thing the
        # halo radius may not do.
        link = LINK_FACTOR * np.maximum(widths[near], sigma)
        free = near[d[near] <= link][:k_max - 1]

    pts = np.vstack([np.atleast_2d(positions)[free], cand[None, :]]) \
        if len(free) else cand[None, :]
    pad = BBOX_PAD * (max(float(widths[free].max()), float(sigma))
                      if len(free) else float(sigma))
    y0 = max(0, int(np.floor(pts[:, 0].min() - pad)))
    x0 = max(0, int(np.floor(pts[:, 1].min() - pad)))
    y1 = min(shape[0], int(np.ceil(pts[:, 0].max() + pad)) + 1)
    x1 = min(shape[1], int(np.ceil(pts[:, 1].max() + pad)) + 1)

    if n:
        others = np.setdiff1d(np.arange(n), free)
        py = np.clip(positions[others, 0], y0, y1 - 1)
        px = np.clip(positions[others, 1], x0, x1 - 1)
        d = np.hypot(positions[others, 0] - py, positions[others, 1] - px)
        frozen = others[d <= HALO_FACTOR * np.maximum(widths[others], sigma)]
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


def _bounds(K, h, w, b_max, A_max, sigma_bounds=None):
    """Box constraints for one window's fit, in LOCAL coordinates.

    Positions are confined to the sub-image. Letting a centre leave the frame
    was tried -- it lets the fit put rim flux where it actually came from --
    and measurably lost real detections elsewhere, so the bounds stay closed.

    `sigma_bounds` adds a fourth parameter per emitter, its own width, bounded
    to that (lo, hi) in pixels. Omit it for the fixed-width layout.
    """
    a_min = max(moves.A_MIN, A_MIN_REL * A_max)
    lo = [0.0]
    hi = [b_max]
    for _ in range(K):
        lo += [a_min, -0.5, -0.5]
        hi += [A_max, h - 0.5, w - 0.5]
        if sigma_bounds is not None:
            lo.append(sigma_bounds[0])
            hi.append(sigma_bounds[1])
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


def _width_prior(slack, band, sigma, lam_focus, lam_wide,
                 gamma=prior_mod.FOCUS_WIDTH_GAMMA):
    """The count-and-width prior for the current configuration, or None at a
    fixed width.

    Neither rate is a knob: `detect` re-estimates both from the frame each
    round, as it already does for `lam` and `A_s`.

    `band=None` is the SINGLE-BAND pipeline: one class, one uniform width prior
    over the whole model space. It is what ran before the mixture existed, kept
    for the reason `prior.ExponentialFlux` is kept -- so a change in detections
    is attributable to the mixture's shape rather than to the rewrite around
    it. `slack=None` is the fixed-width pipeline and has no widths to price at
    all, which is the one case `evidence` is told by a bare `None`.

    This is the ONLY thing `evidence` needs to be told about the layout: it
    sets the extra Laplace dimension and the whole count-and-width prior.
    """
    if slack is None:
        return None
    if band is None:
        return prior_mod.UniformWidth(lam_focus, slack[0] * sigma,
                                      slack[1] * sigma)
    if not slack[0] < band[1] < slack[1]:
        raise ValueError(
            f"FOCUS_BAND's upper edge {band[1]} must lie strictly inside "
            f"SIGMA_SLACK {slack}: it is the class boundary, and a boundary "
            f"on a bound leaves one of the two classes empty by construction")
    return prior_mod.FocusMixtureWidth(
        lam_focus, lam_wide,
        slack[0] * sigma, band[1] * sigma, slack[1] * sigma, sigma, gamma)


class _WidthPenalty:
    """`lmga`'s MAP penalty for the free widths: `-log pi(sigma_k)`, summed.

    Only the width slots of `theta` are touched, so everything is a strided
    view -- the amplitude and position priors are flat and contribute nothing,
    and the background's uniform prior cancels in every comparison.

    This is what makes the FIT and the EVIDENCE use the same prior. Before it,
    `evidence` priced an emitter's width while `refine` chose that width by
    maximum likelihood, and the two disagreed exactly where it mattered: at
    1-2 sigma separation the ML fit is nearly indifferent between two narrow
    emitters and one wide one, and with nothing to break the tie it took the
    wide one and lost the neighbour.
    """

    __slots__ = ("wprior", "_sl")

    def __init__(self, wprior, k):
        self.wprior = wprior
        self._sl = slice(4, 1 + 4 * k, 4)      # theta = [b, (A,y,x,s) * k]

    def _sigmas(self, theta):
        return theta[self._sl]

    def value(self, theta):
        return -float(np.sum(self.wprior.logpdf(self._sigmas(theta))))

    def grad(self, theta):
        g = np.zeros_like(theta)
        s = self._sigmas(theta)
        # -d/dsigma log pi. Central difference: the priors here are cheap
        # scalar functions and an analytic gradient per prior class would be
        # one more thing each must keep consistent with its own `logpdf`.
        h = 1e-6 * np.maximum(np.abs(s), 1.0)
        g[self._sl] = -(self.wprior.logpdf(s + h)
                        - self.wprior.logpdf(s - h)) / (2.0 * h)
        return g

    def hess_diag(self, theta):
        d = np.zeros_like(theta)
        d[self._sl] = self.wprior.curvature(self._sigmas(theta))
        return d


def _fit_any(sub, yy, xx, sigma, halo, b, A, cy, cx, sig, slack,
             max_iter=100, tol_obj=EVIDENCE_TOL_OBJ, wprior=None):
    """One window fit, taken and returned in UNPACKED parameters.

    Returns `(r, b, A, cy, cx, sig)`. The caller never sees a theta, which is
    the point: the fixed-width layout is `3K+1` and the free-width one is
    `4K+1`, and every pass above this line is written once for both. What the
    caller MAY read off `r` is `I`, `F` and `converged` -- `r.F` is in the
    layout that was fitted, and `evidence` is told which by its width prior.

    `A_max` is raised by `slack[1]**2` when widths are free: it is derived
    from the window's peak through `peak_factor(sigma)`, and a source at `n`
    times the PSF width carries the same flux at `1/n^2` of the peak, so the
    in-focus bound would clip exactly the defocused emitters this exists for.

    `wprior` makes the free-width fit a MAP fit under that prior rather than an
    ML fit -- see `_WidthPenalty`. It is the SAME object `evidence` scores the
    move with, which is the point: a fit and an evidence that disagree about
    what a width costs will disagree about what exists.
    """
    K = len(A)
    smax = max(float(sub.max()), 1.0)
    b_max = max(smax * 4.0, 10.0)
    if slack is None:
        A_max = 8.0 * smax / psf.peak_factor(sigma)
        lo, hi = _bounds(K, sub.shape[0], sub.shape[1], b_max, A_max)
        th0 = np.clip(psf.pack(b, A, cy, cx), lo + 1e-9, hi - 1e-9)
        r = lmga.fit(th0, yy, xx, sigma, sub, lo, hi, halo=halo,
                     max_iter=max_iter, tol_obj=tol_obj)
        b_o, A_o, cy_o, cx_o = psf.unpack(r.theta)
        return r, b_o, A_o, cy_o, cx_o, np.full(K, float(sigma))

    A_max = 8.0 * smax / psf.peak_factor(sigma) * slack[1] ** 2
    sb = (slack[0] * sigma, slack[1] * sigma)
    lo, hi = _bounds(K, sub.shape[0], sub.shape[1], b_max, A_max, sb)
    th0 = np.clip(psf.pack_var_sigma(b, A, cy, cx, np.clip(sig, *sb)),
                  lo + 1e-9, hi - 1e-9)
    r = lmga.fit(th0, yy, xx, sigma, sub, lo, hi, halo=halo,
                 max_iter=max_iter, tol_obj=tol_obj,
                 free_sigma="per_emitter",
                 penalty=None if (wprior is None or K == 0
                                  or wprior.is_flat)
                 else _WidthPenalty(wprior, K))
    return (r,) + psf.unpack_var_sigma(r.theta)


def _model_any(b, A, cy, cx, sig, yy, xx, sigma, slack, halo=0.0):
    """The window model for either layout."""
    if slack is None:
        return psf.model(psf.pack(b, A, cy, cx), yy, xx, sigma, halo=halo)
    return psf.model_var_sigma(psf.pack_var_sigma(b, A, cy, cx, sig),
                               yy, xx, halo=halo)


def _stride(slack):
    """Parameters per emitter in the fitted layout: 3 fixed, 4 free-width."""
    return 3 if slack is None else 4


def _halo_image(positions, amplitudes, frozen, sigma, yy, xx, y0, x0, base):
    """`base` plus the frozen emitters' contribution, in window coordinates.

    `sigma` is a scalar or one width per emitter over all of `positions`.
    """
    if not len(frozen):
        return base
    return base + patch_mod.build_halo_image(
        positions, amplitudes, np.asarray(frozen, dtype=int), sigma,
        yy, xx, y0, x0)


def _try_add(d_e, positions, amplitudes, sigmas, bmap, cand, camp, sigma,
             lam, A_s, k_max, slack, wprior=None):
    """Score adding one emitter. Returns (accepted, positions, amplitudes,
    sigmas).

    Both models are fitted on the SAME pixels with the SAME frozen halo and
    differ only by the one emitter, which is what makes their I-divergences
    differencable into a Bayes factor.

    On acceptance the whole window's refitted parameters are written back:
    adding a source shifts its neighbours, and keeping their stale values
    would leave the model worse than the fit that justified the acceptance.
    With free widths that includes the neighbours' WIDTHS, for the same
    reason -- a width is a fitted parameter like any other.

    The candidate starts at the PSF width. It is a LoG peak at that width, so
    that is what has actually been seen; letting it start wide would let a
    proposal begin by claiming its neighbour's flux.

    The only conditions that can block the move are `log BF <= 0` and
    `COND_GUARD` -- an ill-conditioned Fisher matrix makes the Occam term
    meaningless, so it is not weighed against anything. A pre-fit significance
    screen on A/SE is deliberately absent: measured, it refused 68-79% of every
    true emitter lost inside 2 sigma before the Bayes factor could vote, while
    the Bayes factor itself refused none of them. Degenerate configurations are
    removed by `_prune`, after a joint fit, on more information.
    """
    free, frozen, (y0, x0, y1, x1) = _window(positions, cand, sigma,
                                             d_e.shape, k_max, sigmas)
    sub = np.asarray(d_e[y0:y1, x0:x1])
    h, w = sub.shape
    yy, xx = np.mgrid[0:h, 0:w] * 1.0

    level, halo = _window_bg(bmap, y0, x0, y1, x1)
    halo = _halo_image(positions, amplitudes, frozen, sigmas, yy, xx,
                       y0, x0, halo)

    loc = (positions[free] - np.array([y0, x0])) if len(free) \
        else np.empty((0, 2))
    a0 = amplitudes[free] if len(free) else np.empty(0)
    s0 = sigmas[free] if len(free) else np.empty(0)

    r_b, _, A_b, _, _, s_b = _fit_any(sub, yy, xx, sigma, halo, level, a0,
                                      loc[:, 0], loc[:, 1], s0, slack,
                                      wprior=wprior)

    cl = cand - np.array([y0, x0])
    r_a, _, A, cy, cx, sg = _fit_any(
        sub, yy, xx, sigma, halo, level, np.append(a0, camp),
        np.append(loc[:, 0], cl[0]), np.append(loc[:, 1], cl[1]),
        np.append(s0, sigma), slack, wprior=wprior)

    log_bf, cond = evidence.log_bf_add(
        r_b.I, r_a.I, r_b.F, r_a.F, A_b, A, len(free), lam, A_s,
        widths=None if wprior is None else (wprior, s_b, sg))
    if not np.isfinite(log_bf) or log_bf <= 0 or cond > evidence.COND_GUARD:
        return False, positions, amplitudes, sigmas

    new_pos = np.stack([cy + y0, cx + x0], axis=1)
    if len(free):
        positions = positions.copy()
        amplitudes = amplitudes.copy()
        sigmas = sigmas.copy()
        positions[free] = new_pos[:len(free)]
        amplitudes[free] = A[:len(free)]
        sigmas[free] = sg[:len(free)]
    positions = np.vstack([positions, new_pos[-1][None, :]]) if len(positions) \
        else new_pos[-1][None, :]
    amplitudes = np.append(amplitudes, A[-1])
    sigmas = np.append(sigmas, sg[-1])
    return True, positions, amplitudes, sigmas


def _try_split(d_e, positions, amplitudes, sigmas, bmap, gi, sigma,
               lam, A_s, k_max, slack, wprior=None):
    """Score replacing emitter `gi` with two. Returns (accepted, pos, amp, sig).

    The move FIND structurally cannot make. Two emitters closer than about
    1.5 sigma are fitted well by one brighter PSF, so their residual has no
    PEAK -- it has a quadrupole, negative in the middle and positive on two
    lobes along the pair axis. `moves.residual_axis_var` recovers that axis
    from the second moment of the residual and the split is proposed along it.

    Takes K to K+1 like `_try_add`, so the outer loop stays monotone in N.

    With free widths this move is no longer the pipeline's answer to defocus.
    A quadrupole is what an unresolved PAIR leaves; a source merely broader
    than the model leaves a ROTATIONALLY SYMMETRIC residual, and once the
    incumbent can widen to absorb it there is no residual left to propose
    into. That is the whole mechanism: the tiling this move used to do was
    never a bad decision, it was the right decision inside a model space that
    could not hold the answer.
    """
    free, frozen, (y0, x0, y1, x1) = _window(positions, positions[gi], sigma,
                                             d_e.shape, k_max, sigmas)
    lk = int(np.nonzero(free == gi)[0][0]) if gi in free else None
    if lk is None:
        return False, positions, amplitudes, sigmas

    sub = np.asarray(d_e[y0:y1, x0:x1])
    h, w = sub.shape
    yy, xx = np.mgrid[0:h, 0:w] * 1.0
    level, halo = _window_bg(bmap, y0, x0, y1, x1)
    halo = _halo_image(positions, amplitudes, frozen, sigmas, yy, xx,
                       y0, x0, halo)

    loc = positions[free] - np.array([y0, x0])
    r_b, b_b, A_b, cy_b, cx_b, s_b = _fit_any(
        sub, yy, xx, sigma, halo, level, amplitudes[free],
        loc[:, 0], loc[:, 1], sigmas[free], slack, wprior=wprior)

    th_b = psf.pack_var_sigma(b_b, A_b, cy_b, cx_b, s_b)
    resid = sub - _model_any(b_b, A_b, cy_b, cx_b, s_b, yy, xx, sigma, slack,
                             halo=halo)
    u, _ = moves.residual_axis_var(th_b, lk, yy, xx, resid)

    # One incumbent, several proposals: its log-determinant is the same for
    # all of them and is factorized once.
    ld_b = evidence.logdet(r_b.F)
    best = None
    for disp in SPLIT_DISPS:
        th_a = moves.split_var(th_b, lk, u, disp * sigma)
        _, A0, cy0, cx0, s0 = psf.unpack_var_sigma(th_a)
        r_a, b_a, A_a, cy_a, cx_a, s_a = _fit_any(
            sub, yy, xx, sigma, halo, float(th_a[0]), A0, cy0, cx0, s0, slack,
            wprior=wprior)
        log_bf, cond = evidence.log_bf_add(
            r_b.I, r_a.I, r_b.F, r_a.F, A_b, A_a, len(free), lam, A_s,
            before=ld_b,
            widths=None if wprior is None else (wprior, s_b, s_a))
        if np.isfinite(log_bf) and log_bf > 0 and cond <= evidence.COND_GUARD:
            if best is None or log_bf > best[0]:
                best = (log_bf, A_a, cy_a, cx_a, s_a)
    if best is None:
        return False, positions, amplitudes, sigmas

    # `moves.split_var` keeps the untouched emitters in order and appends the
    # two children, so the fitted vector is [free without gi] + [child0, child1].
    _, A, cy, cx, sg = best
    new_pos = np.stack([cy + y0, cx + x0], axis=1)
    keep_local = [j for j in range(len(free)) if j != lk]
    positions = positions.copy()
    amplitudes = amplitudes.copy()
    sigmas = sigmas.copy()
    if keep_local:
        positions[free[keep_local]] = new_pos[:len(keep_local)]
        amplitudes[free[keep_local]] = A[:len(keep_local)]
        sigmas[free[keep_local]] = sg[:len(keep_local)]
    positions[gi] = new_pos[-2]
    amplitudes[gi] = A[-2]
    sigmas[gi] = sg[-2]
    positions = np.vstack([positions, new_pos[-1][None, :]])
    amplitudes = np.append(amplitudes, A[-1])
    sigmas = np.append(sigmas, sg[-1])
    return True, positions, amplitudes, sigmas


def _add_pass(d_e, bmap, positions, amplitudes, sigmas, cand, camp, sigma,
              lam, A_s, k_max, slack=None, wprior=None, band=None,
              veto_widths=True):
    """One ADD pass over a candidate list. Returns (pos, amp, sig, n_added).

    The proximity re-check is part of the pass rather than of `detect`: it reads
    the positions an earlier acceptance in THIS pass has already written, so an
    emitter accepted a moment ago can claim a later candidate's flux.
    """
    n_added = 0
    for c, a in zip(cand, camp):
        # `veto_radius` again -- here reading the widths THIS pass has already
        # written, so an emitter that just widened claims the candidates it now
        # covers.
        if len(positions) and np.any(
                np.linalg.norm(positions - c, axis=1)
                <= veto_radius(sigma, sigmas if veto_widths else None, band)):
            continue
        ok, positions, amplitudes, sigmas = _try_add(
            d_e, positions, amplitudes, sigmas, bmap, c, a, sigma, lam, A_s,
            k_max, slack, wprior)
        n_added += int(ok)
    return positions, amplitudes, sigmas, n_added


def _split_pass(d_e, positions, amplitudes, sigmas, bmap, sigma, lam, A_s,
                k_max, model, slack=None, wprior=None):
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
        return positions, amplitudes, sigmas, 0
    resid_full = d_e - model
    strengths = np.zeros(n0)
    for i in range(n0):
        # Padded by the emitter's OWN width: the quadrupole of a wide source
        # lives outside a box drawn at the in-focus width.
        pad = int(np.ceil(BBOX_PAD * max(sigmas[i], sigma)))
        y0 = max(0, int(positions[i, 0]) - pad)
        x0 = max(0, int(positions[i, 1]) - pad)
        y1 = min(d_e.shape[0], int(positions[i, 0]) + pad + 1)
        x1 = min(d_e.shape[1], int(positions[i, 1]) + pad + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1] * 1.0
        th = psf.pack_var_sigma(0.0, [amplitudes[i]], [positions[i, 0]],
                                [positions[i, 1]], [sigmas[i]])
        _, s = moves.residual_axis_var(th, 0, yy, xx,
                                       resid_full[y0:y1, x0:x1])
        strengths[i] = s

    n_split = 0
    for gi in np.argsort(-strengths):
        if strengths[gi] <= 0:
            break
        ok, positions, amplitudes, sigmas = _try_split(
            d_e, positions, amplitudes, sigmas, bmap, int(gi), sigma,
            lam, A_s, k_max, slack, wprior)
        n_split += int(ok)
    return positions, amplitudes, sigmas, n_split


def _amplitude_var(F, k, stride=3):
    """var(A_k) from the Fisher matrix, or None if it cannot be trusted.

    None means the Laplace evidence for this configuration cannot be computed
    -- a singular Fisher matrix, or a non-positive amplitude variance -- so
    removal must be forced rather than weighed. Weighing is not an option:
    the quantity that would do the weighing is the thing that has broken.

    `stride` is the parameters per emitter in the layout `F` was built in.
    Note this is the MARGINAL variance, from the inverse: with free widths it
    therefore already carries the amplitude-width correlation, which is real
    and large for a faint broad source -- flux and width trade off against
    each other -- and which is exactly what `PRUNE_TAU` should be reading.
    """
    try:
        var = np.diag(np.linalg.inv(F))
    except np.linalg.LinAlgError:
        return None
    v = var[1 + stride * k]
    if not np.isfinite(v) or v <= 0:
        return None
    return float(v)


def _prune(d_e, positions, amplitudes, sigmas, bmap, sigma, lam, A_s, k_max,
           slack=None, wprior=None, tau=PRUNE_TAU):
    """One pass of removal tests. Returns (positions, amplitudes, sigmas).

    Runs outside the add loop and never feeds back into it: an emitter's A/SE
    verdict depends on which neighbours are free in that pass, so letting
    removal drive addition makes the same source get killed and recreated
    indefinitely.
    """
    n = len(positions)
    if n == 0:
        return positions, amplitudes, sigmas
    # Copied because the survivor write-back below mutates these in place and
    # the caller still holds the pre-prune configuration.
    positions = np.array(positions, dtype=float, copy=True)
    amplitudes = np.array(amplitudes, dtype=float, copy=True)
    sigmas = np.array(sigmas, dtype=float, copy=True)
    stride = _stride(slack)
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
                                                 sigma, d_e.shape, k_max,
                                                 sigmas[others])
        free = others[free]
        frozen = others[frozen]
        sub = np.asarray(d_e[y0:y1, x0:x1])
        h, w = sub.shape
        yy, xx = np.mgrid[0:h, 0:w] * 1.0
        level, halo = _window_bg(bmap, y0, x0, y1, x1)
        halo = _halo_image(positions, amplitudes, frozen, sigmas, yy, xx,
                           y0, x0, halo)

        keep_idx = np.append(free, gi).astype(int)
        loc = positions[keep_idx] - np.array([y0, x0])
        r_full, _, A_full, _, _, s_full = _fit_any(
            sub, yy, xx, sigma, halo, level, amplitudes[keep_idx],
            loc[:, 0], loc[:, 1], sigmas[keep_idx], slack, wprior=wprior)

        lr = positions[free] - np.array([y0, x0]) if len(free) \
            else np.empty((0, 2))
        r_red, _, A_red, cy_r, cx_r, s_r = _fit_any(
            sub, yy, xx, sigma, halo, level,
            amplitudes[free] if len(free) else np.empty(0),
            lr[:, 0], lr[:, 1],
            sigmas[free] if len(free) else np.empty(0), slack, wprior=wprior)

        # `gi` is appended last in `keep_idx`, so it is the last emitter of
        # the fitted vector.
        k = len(keep_idx) - 1
        v = _amplitude_var(r_full.F, k, stride)
        if v is None or A_full[k] < tau * np.sqrt(v):
            log_bf = np.inf                   # forced, not weighed
        else:
            # `(reduced, full)`: removal is the negation of the ADD whose
            # "before" is the reduced configuration.
            log_bf = evidence.log_bf_remove(
                r_full.I, r_red.I, r_full.F, r_red.F, A_full, A_red,
                len(keep_idx), lam, A_s,
                widths=None if wprior is None else (wprior, s_r, s_full))
        if log_bf > 0:
            alive[gi] = False
            # Write the reduced fit back over the survivors. This is what makes
            # the faintest-first cascade correct: when a collapsed pair loses
            # one member the other absorbs its flux, and the next removal test
            # must be scored against that, not against a stale half-amplitude
            # that would make the survivor look removable too.
            if len(free):
                positions[free, 0] = cy_r + y0
                positions[free, 1] = cx_r + x0
                amplitudes[free] = A_red
                sigmas[free] = s_r
    return positions[alive], amplitudes[alive], sigmas[alive]


def _update_bg(d_e, positions, amplitudes, sigma, bmap, kernel, be=None,
               exclude=None):
    """Re-estimate the background from the current emitter model.

    `kernel=None` gives one scalar for the frame, from the pixels no emitter
    reaches; any integer estimates a surface on that window.

    `sigma` may be one width per emitter, in which case each emitter's support
    is masked at its own width and `be` must be None -- the Rust mask takes a
    scalar.
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
           k_max=12, max_rounds=6, threshold=None, prune=True,
           split=True, bg_kernel=BG_KERNEL, max_settle=4, verbose=1,
           impl="py", prior=None,
           slack=SIGMA_SLACK, band=FOCUS_BAND,
           width_gamma=prior_mod.FOCUS_WIDTH_GAMMA, prune_tau=PRUNE_TAU,
           veto_widths=True):
    """Full detection. Returns a `DetectResult`.

    `max_rounds` is a safety stop, not the termination condition: the loop ends
    when a round accepts nothing, which it must eventually do because every
    accepted emitter lowers the residual that produces the candidates.

    `threshold=None` derives FIND's seed cut from the frame size and the PSF
    width (`calibrate.seed_threshold`) rather than carrying a constant, which
    cannot be right on two frame sizes at once. Pass a float to pin it; pass
    `CAND_THRESHOLD` for the pre-2026-09-03 value.

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

    `slack` gives every emitter its own width, bounded to that (lo, hi)
    multiple of `sigma`; `None` fixes them all at `sigma`. See `SIGMA_SLACK`
    for why the model needs it -- in short, a fixed-width model answers a
    source it cannot represent by tiling it, and no threshold repairs a model
    space that does not contain the answer.

    `impl` is IGNORED while `slack` is on, and the Python passes run: the Rust
    core implements the fixed-width layout only. Pass `slack=None` for the
    fast path. This older full-frame path remains separate from the calibrated
    native local inference and the single-pass sparse localizer.
    """
    if slack is not None and impl != "py":
        # Silently ignoring a performance argument is worse than being slow.
        print(f"note: impl={impl!r} ignored -- the Rust core implements the "
              f"fixed-width passes only.\n      Pass slack=None for it, or "
              f"accept the Python passes.")
    be = backend_mod.get("py" if slack is not None else impl)
    raw = np.asarray(data_img, dtype=float)
    H, W = raw.shape
    g_eff = calibrate.estimate_gain(raw, offset) if gain is None else float(gain)
    d_e = (raw - offset) / g_eff

    # FIND's cut is DERIVED from the frame and the PSF unless the caller
    # pins it; a constant cannot be right on two frame sizes at once.
    if threshold is None:
        threshold = calibrate.seed_threshold(d_e.shape, sigma)

    b0 = float(np.percentile(d_e, 10.0))

    # BEFORE the search: an aggregate is still one object here. After it, the
    # fixed-sigma model will have tiled it into pieces no per-detection
    # statistic can identify. See `find_aggregates`.
    # lam is emitters per px^2.
    usable_px = max(float(H * W), 1.0)

    bmap = np.full((H, W), max(b0, BG_FLOOR))
    if A_s0 is None:
        A_s0 = max(float(d_e.max()) - b0, 10.0) / psf.peak_factor(sigma)
    lam, A_s = lam0, A_s0

    def _in_focus(sig):
        """The reporting band, as a mask over the working arrays."""
        if slack is None or band is None or not len(sig):
            return np.ones(len(sig), dtype=bool)
        return (sig >= band[0] * sigma) & (sig <= band[1] * sigma)

    def _rates(sig):
        """(lam_focus, lam_wide) from the current configuration.

        Empirical Bayes, exactly as `lam` and `A_s` already are: how rare a
        defocused object is comes off the frame, not off a constant. Both are
        floored at a count of one -- a frame may a priori hold one object of
        either class -- which at a typical 4096 usable px opens the wide class
        at about 3 nats against and stops mattering once one is found. Before
        anything is detected there is nothing to estimate from, and both fall
        back to `lam`, as the single-class pipeline does.
        """
        if slack is None or band is None or not len(sig):
            return lam, lam
        n_f = int(np.count_nonzero(sig <= band[1] * sigma))
        return (max(n_f, 1) / usable_px, max(len(sig) - n_f, 1) / usable_px)

    positions = np.empty((0, 2))
    amplitudes = np.empty(0)
    sigmas = np.empty(0)
    history = []

    # The four passes carry per-emitter widths; the backend contract does not,
    # so with `slack` on they are called directly rather than through `be`.
    # `find_candidates`, `background_map` and this loop are the same code
    # either way, which is what has always made the backends comparable.
    def _render(pos, amp, sig):
        return bmap + (be.render_model(pos, amp, sigma, bmap.shape, 0.0)
                       if slack is None
                       else calibrate.render_model(pos, amp, sig, bmap.shape,
                                                   0.0))

    def _refine(pos, amp, sig, sweeps):
        if slack is None:
            pos, amp, se_ = be.refine(d_e, pos, amp, sigma, bmap, k_max,
                                      sweeps)
            return pos, amp, se_, np.full(len(amp), float(sigma))
        # The SAME prior the round's moves were scored with. Refining under a
        # different one would let the estimation half undo what the model
        # selection half decided.
        return refine(d_e, pos, amp, sigma, bmap, k_max=k_max,
                      max_sweeps=sweeps, sigmas=sig, slack=slack,
                      wprior=_width_prior(slack, band, sigma, *_rates(sig),
                                          gamma=width_gamma))

    def _prune_once(pos, amp, sig):
        if slack is None:
            pri = A_s if prior is None else prior
            pos, amp, n = be.prune(d_e, bmap, pos, amp, sigma, lam, pri, k_max)
            return pos, amp, np.full(len(amp), float(sigma))
        return _prune(d_e, pos, amp, sig, bmap, sigma, lam,
                      A_s if prior is None else prior,
                      k_max, slack,
                      _width_prior(slack, band, sigma, *_rates(sig),
                                   gamma=width_gamma),
                      tau=prune_tau)

    for rnd in range(max_rounds):
        model = _render(positions, amplitudes, sigmas)
        veto_sig = sigmas if veto_widths else None
        cand, camp, _ = find_candidates(d_e, model, sigma, positions,
                                        threshold, veto_sig, band)
        pri = A_s if prior is None else prior
        wprior = _width_prior(slack, band, sigma, *_rates(sigmas),
                              gamma=width_gamma)
        if slack is None:
            positions, amplitudes, n_added = be.add_pass(
                d_e, bmap, positions, amplitudes, cand, camp, sigma, lam, pri,
                k_max)
            sigmas = np.full(len(amplitudes), float(sigma))
        else:
            positions, amplitudes, sigmas, n_added = _add_pass(
                d_e, bmap, positions, amplitudes, sigmas, cand, camp, sigma,
                lam, pri, k_max, slack, wprior, band, veto_widths)

        # SPLIT runs on the model the adds just produced: an emitter only looks
        # like an unresolved pair once its neighbourhood is otherwise
        # explained, and splitting against a model still missing a nearby
        # source mostly splits emitters into that source's flux.
        n_split = 0
        if split:
            if n_added:
                model = _render(positions, amplitudes, sigmas)
            if slack is None:
                positions, amplitudes, n_split = be.split_pass(
                    d_e, bmap, positions, amplitudes, model, sigma, lam, pri,
                    k_max)
                sigmas = np.full(len(amplitudes), float(sigma))
            else:
                positions, amplitudes, sigmas, n_split = _split_pass(
                    d_e, positions, amplitudes, sigmas, bmap, sigma, lam, pri,
                    k_max, model, slack, wprior)

        if n_added or n_split:
            # One sweep here; the round loop is the outer iteration.
            positions, amplitudes, _, sigmas = _refine(
                positions, amplitudes, sigmas, 1)
            # Both estimated from the IN-FOCUS class alone. Counting the
            # nuisance objects in `lam` would be a positive feedback loop --
            # more objects raises the count prior, which makes the next add
            # easier -- and averaging their fluxes into `A_s` drags the
            # amplitude prior towards the dim, broad ones, which are dim
            # because of the confocal axial response and not because the
            # emitter population is faint.
            foc = _in_focus(sigmas)
            lam = max(int(np.count_nonzero(foc)) / usable_px, 1e-6)
            if foc.any():
                A_s = max(float(np.mean(amplitudes[foc])), 1.0)
            # Re-estimated only AFTER the emitters have been re-fitted, and
            # only from the emitter model with no background in it. The mask
            # uses each emitter's OWN width, so a defocused source's larger
            # footprint is excluded from its own background window.
            bmap = _update_bg(d_e, positions, amplitudes,
                              sigma if slack is None else sigmas, bmap,
                              bg_kernel, be=be if slack is None else None,
                              )

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
    positions, amplitudes, se, sigmas = _refine(positions, amplitudes, sigmas,
                                               REFINE_SWEEPS)

    # Settle and prune alternate until the prune removes nothing. One pass is
    # not enough: the re-fit at the reduced N is as free to collapse a pair as
    # the first one was. This cannot cycle -- prune only removes, so N strictly
    # decreases. `max_settle` is a backstop.
    for _ in range(max_settle if prune else 0):
        if not len(positions):
            break
        n_before = len(positions)
        positions, amplitudes, sigmas = _prune_once(positions, amplitudes,
                                                    sigmas)
        if len(positions) == n_before:
            break
        if verbose >= 1:
            print(f"  [prune] {n_before} -> {len(positions)}")
        positions, amplitudes, se, sigmas = _refine(positions, amplitudes,
                                                    sigmas, REFINE_SWEEPS)
    model = _render(positions, amplitudes, sigmas)

    # The reporting split, and the ONLY place it happens. Everything above this
    # line worked on the whole configuration -- which is the point: a defocused
    # object has to be fitted, refined and pruned like any other or its flux
    # goes back into the residual and refills FIND's candidate list, which is
    # the tiling this exists to stop. `model_image` and `residual` therefore
    # contain the nuisance objects too; they are what produced them.
    focus = _in_focus(sigmas)
    narrow = (~focus & (sigmas < band[0] * sigma)
              if slack is not None and band is not None
              else np.zeros(len(sigmas), dtype=bool))
    wide = ~focus & ~narrow
    idx = np.arange(len(sigmas))
    width_rejects = np.concatenate([
        width_reject_records(idx[narrow], positions[narrow],
                             amplitudes[narrow], sigmas[narrow], sigma,
                             "too_narrow"),
        width_reject_records(idx[wide], positions[wide], amplitudes[wide],
                             sigmas[wide], sigma, "too_wide"),
    ])
    wide_rec = _wide_records(positions[wide], amplitudes[wide], sigmas[wide])


    if verbose >= 1:
        nr = (d_e - model) / np.sqrt(np.maximum(model, 1e-6))
        print(f"[final] N={int(focus.sum())}"
              + (f" (+{int(wide.sum())} wide, {int(narrow.sum())} narrow)"
                 if len(width_rejects) else "")
              + f"  bg={np.median(bmap):.2f} "
              f"[{bmap.min():.2f}, {bmap.max():.2f}]  "
              f"resid median={np.median(nr):+.3f}  "
              f"robust_std={calibrate.robust_spread(nr):.3f}")

    positions, amplitudes = positions[focus], amplitudes[focus]
    se = se[focus] if se is not None and len(se) else se
    sigmas = sigmas[focus]

    return DetectResult(
        positions=positions, amplitudes=amplitudes, sigma=sigma, lam=lam,
        A_s=A_s, gain=g_eff, n_outer_passes=len(history), model_image=model,
        # `background` is the SURFACE, an (H, W) array, not a scalar. Callers
        # that only want a number should take its median; callers that render
        # or subtract a model want the array.
        residual=d_e - model, background=bmap, se=se, history=history,
        aggregates=(wide_rec if len(wide_rec) else None),
        width_rejects=(width_rejects if len(width_rejects) else None),
        width_filter=dict(band=None if band is None else tuple(band),
                          slack=None if slack is None else tuple(slack)),
        # The fitted widths are a RESULT, not a diagnostic afterthought: they
        # are what the model used for these positions, amplitudes and CRLBs,
        # and `sigma_ratio` is a per-emitter defocus readout the fixed-width
        # pipeline could not produce. Asking the same question AFTER the
        # search, on a fit whose neighbours have been frozen into its halo,
        # does not work -- README section 10b measured that the answer is no
        # longer there by then, and section 13 records the removal.
        fit_sigma=sigmas,
        sigma_ratio=(sigmas / sigma if len(sigmas) else sigmas),
    )


# ---------------------------------------------------------------------------
# Post-hoc reporting -- NOT part of the port
# ---------------------------------------------------------------------------
#
# Everything above this line runs inside the search and is the port target.
# What follows reads a finished `DetectResult` and says something about it. It
# is here rather than in `loctable` only because `loctable` needs `polars` and
# this does not; nothing below is on any hot path and none of it should cross
# into Rust.

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
