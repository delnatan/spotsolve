"""Box-local localization: in each box, the best fit the data support wins.

Pipeline
--------
    d_e = (raw - offset) / gain + read_noise^2
    FIND     LoG peaks on the image, once, inside `roi`     -> candidates
    BMAP     smooth background surface, candidates masked   -> each box's
             known background shape; its level stays free
    BOXES    candidates whose footprints touch share a box, <= k_max each
    SWEEPS times, for each box, brightest first:
        fit the background alone (K = 0)
        FORWARD   place one emitter at the strongest residual peak the box
                  owns that passes FIND's z-threshold, refit all K+1
                  jointly, keep it iff the Poisson deviance falls by more
                  than ADD_NATS
        BACKWARD  score every emitter's removal; drop the cheapest while it
                  costs less than ADD_NATS
        the box's emitters replace what it held in every other box's halo
    POLISH   one fixed-K joint refit (`core.refine`) for final estimates/SEs

Each emitter is decided in the box that owns it, by one comparison rule.
There is no SPLIT move, no separate removal pass, no forced-removal threshold,
no conditioning guard and no width prior: a collapsed or redundant emitter
explains almost no deviance, so it cannot pay ADD_NATS on the way in and costs
almost nothing on the way out.

The design is settled: the variable-width fitter, the read-noise model, the
ROI and the smooth background map. Two treatments of 3-8 px haze were
measured and rejected (the notes below).

This module is the REFERENCE. The fast path is its native port,
`spotsolve.localize` / `localize_stack` (`native.py`, over Rust's
`boxsearch.rs`), held to statistical parity with it by `tests/test_localize.py`.
Change the algorithm here first, measure it here, then port.
"""

import numpy as np
import scipy.ndimage as ndi

from . import backend as backend_mod
from . import calibrate
from . import core
from . import psf
from .patches import build_patches
from .structs import DetectResult, width_reject_records

__all__ = ["localize_boxes"]


ADD_NATS = 10.0
# The whole decision rule: an emitter exists iff it lowers the Poisson
# deviance (I-divergence) of its box by more than this many nats.
#
# Measured 2026-09-10 by replacing every Laplace Bayes factor in `detect` with
# `dI - c` (COND_GUARD, PRUNE_TAU and the MAP width fit unchanged), 64x64
# `simulate` fields, three seeds per cell, recall / precision / tiles per
# frame:
#
#   density spread     Laplace BF          c = 10            c = 14
#    0.015   0.4    .888 .920  2.3    .888 .938  2.0    .898 .969  1.0
#    0.034   0.4    .832 .807 13.3    .815 .868  8.3    .805 .899  6.0
#    0.055   0.4    .721 .797 20.0    .704 .833 15.3    .688 .867 11.3
#
# c = 10 matched the Bayes factor within seed noise in all six cells; c = 14
# trades 2-3 recall points for about half the tiles. The Bayes factor's
# priors, log-determinants and empirical-Bayes rates were an operating point.

OWN_RADIUS = 3.0
# sigma. A box may place an emitter only on pixels within this distance of one
# of its own candidates, and nearer to its own than to any other box's. The
# nearest-candidate rule is what stops two boxes claiming the same light; the
# radius stops a box reaching across empty space. Measured on the six 64x64
# frames at density 0.034, spread 0.2 (one sweep, MAP widths):
#
#   radius    recall   prec   tiles/frame
#     2.0     .783     .890      8.2
#     3.0     .823     .886      8.8
#     4.0     .842     .868     10.8
#     inf     .844     .863     11.2
#
# At 2.0 a partner 2-3 sigma from the candidate it hides behind was outside
# the box's reach -- those pairs were 31 of the misses, against 3 for
# `detect`. Past 3.0 recall rises about as fast as tiles do.

SWEEPS = 2
# Box decisions are conditional on the neighbours in the halo, and on the
# first sweep a neighbour not yet decided is only its FIND seed. A mismatched
# seed leaves light that the current box claims, and the neighbour's source
# ends up split across two boxes. The second sweep re-decides every box from
# K = 0 against neighbours that have all been fitted. Same six frames:
#
#   sweeps    recall   prec   tiles/frame   search+polish fits
#     1       .823     .886      8.8             504
#     2       .821     .927      5.8             725
#     3       .827     .935      5.5             942

FIT_TOL_OBJ = 1e-6
# nats. A decision here only has to resolve dI against ADD_NATS, where
# `core.EVIDENCE_TOL_OBJ`'s 1e-8 served a score differenced to +0.02 nats.
# Measured 2026-09-11 against 1e-8 and 1e-4 on the eight referee cells (flat
# and haze, three seeds): recall and precision agreed to within one emitter
# per cell at all three. On the real frames, 1e-8 moved positions by
# 0.006 px (glycerol f0) and 0.036 px (GEM f0), and 1e-4 by 0.03-0.04 px.

FIT_MAX_ITER = 100

EDGE_MARGIN = 1.0
# sigma. An out-of-band fit whose centre lies within this distance of the
# frame border is reported as "edge". On beads_60x_still (in-focus beads on
# the coverslip, sigma0 1.0 px) the fits cut off by the border sat 0-0.5 px
# from it; the two narrow interior fits sat 1.9 px and further in.

# Read noise: the shifted-Poisson approximation. A pixel's variance is
# m + sigma_r^2, not m, and adding s = sigma_r^2 (e-^2) to both the data and
# the model gives a Poisson likelihood with exactly that variance and mean
# m + s. Here it is a shift of `d_e` alone: every box model already carries
# a free constant background, which absorbs `s`, and FIND's `sqrt(model)`
# normalization then divides by the right standard deviation. `s` is taken
# back off the reported background and model.
#
# Without it, at low background a read-noise spike reads as a significant
# single-pixel source. Measured 2026-09-10 on 64x64 `simulate` frames with
# Gaussian read noise added, false detections per frame on EMPTY frames (six
# seeds), then recall / precision at density 0.015, flux U(150, 500) e-
# (three seeds):
#
#   bg e-  sigma_r   empty: Poisson  shifted    Poisson        shifted
#    1.0     1.6             1.2       0.0     .837 .937      .830 .983
#    1.0     2.5            13.3       0.0     .823 .811      .823 .983
#    3.0     2.5             5.7       0.0     .837 .922      .816 .975
#   10.0     2.5             0.5       0.0     .773 .965      .730 .963
#
# With no read noise the two are the same pipeline. The recall the shifted
# model gives up where sigma_r^2 rivals the background is not yet traced to
# individual emitters.

# Background: `core.background_map`, a masked 25 px local mean, is each box's
# known shape, with the box's level free. It was chosen over a plane per box
# (b + slopes as free parameters) and over a free constant alone. Measured
# 2026-09-11 on the referee frames (64x64, flux U(900, 1900) e-, bg 20 e-,
# seeds 17-19), recall / precision / invented per frame. Haze `L, H` is white
# noise Gaussian-filtered at L px, scaled to span [0, H] e-:
#
#   cell               constant           plane              map
#   flat  0.015 0.4    .702 .980  0.7     .660 .949  1.7     .695 .970  1.0
#   flat  0.055 0.4    .424 .855 12.3     .426 .856 12.3     .438 .873 11.0
#   haze  L15 H60      .701 .957  3.3     .707 .978  1.7     .729 .992  0.7
#   haze  L5  H60      .682 .952  3.7     .670 .960  3.0     .704 .966  2.7
#   haze  L15 H20      .710 .983  1.3     .698 .978  1.7     .698 .961  3.0
#
# The plane's two extra parameters cost sparse fields; the map costs no fit
# parameters or time. Weak haze (H20) is the map's one worse cell, by 3-6
# events over three frames. Re-estimating the map from the fitted emitters
# after the first sweep moved nothing beyond seed noise.

# No width prior. Every fit here is a plain ML fit over the width bounds
# `slack`. `detect`'s MAP width penalty pulls every width toward sigma0; in
# this search that leaves the wings of a broad emitter unexplained, and the
# next placement lands on them as a faint satellite. Removing it, six frames
# per cell, recall / precision / tiles per frame:
#
#   density spread     MAP width           flat width
#    0.015   0.4    .876 .893  3.3    .866 .940  1.7
#    0.034   0.2    .825 .929  5.7    .813 .949  4.2
#    0.034   0.4    .783 .823 11.5    .776 .858  9.0
#    0.055   0.2    .745 .908 10.3    .737 .941  6.7


def _render(pos, amp, sig, yy, xx):
    """Sum of emitters at global `pos`, on local grids already offset."""
    if not len(amp):
        return 0.0
    th = psf.pack_var_sigma(0.0, amp, pos[:, 0], pos[:, 1], sig)
    return psf.model_var_sigma(th, yy, xx)


def _near_rect(pos, sig, sigma, y0, x0, y1, x1):
    """Mask of sources close enough to a box to put flux in it."""
    if not len(pos):
        return np.zeros(0, dtype=bool)
    py = np.clip(pos[:, 0], y0, y1 - 1)
    px = np.clip(pos[:, 1], x0, x1 - 1)
    d = np.hypot(pos[:, 0] - py, pos[:, 1] - px)
    return d <= core.HALO_FACTOR * np.maximum(sig, sigma)


class _Box:
    """One box's data, grids, halo and ownership mask."""

    def __init__(self, d_e, patch, own, other, halo, sigma, bg, roi=None):
        self.y0, self.x0 = patch.y0, patch.x0
        self.sub = np.asarray(d_e[patch.y0:patch.y1, patch.x0:patch.x1])
        h, w = self.sub.shape
        self.yy, self.xx = np.mgrid[0:h, 0:w] * 1.0
        self.halo = halo
        self.bg = bg
        # Ownership, in local coordinates.
        gy, gx = self.yy + self.y0, self.xx + self.x0
        d_own = np.min(np.hypot(gy[..., None] - own[:, 0],
                                gx[..., None] - own[:, 1]), axis=-1)
        self.owned = d_own <= OWN_RADIUS * sigma
        if len(other):
            d_oth = np.min(np.hypot(gy[..., None] - other[:, 0],
                                    gx[..., None] - other[:, 1]), axis=-1)
            self.owned &= d_own <= d_oth
        if roi is not None:
            self.owned &= roi[patch.y0:patch.y1, patch.x0:patch.x1]


class _Search:
    """Fits and the forward/backward rule for one box."""

    def __init__(self, box, sigma, slack, be, threshold):
        self.box, self.sigma, self.slack = box, sigma, slack
        self.threshold = threshold
        self.be = be
        self.n_fits = 0

    def fit(self, b, A, cy, cx, sg):
        box = self.box
        self.n_fits += 1
        return core._fit_any(box.sub, box.yy, box.xx, self.sigma, box.halo, b,
                             np.asarray(A, float), np.asarray(cy, float),
                             np.asarray(cx, float), np.asarray(sg, float),
                             self.slack, max_iter=FIT_MAX_ITER,
                             tol_obj=FIT_TOL_OBJ,
                             fit_backend=self.be)

    def placement(self, state):
        """(y, x, A0) of the strongest owned residual peak that passes FIND's
        own significance test, or None.

        The test is not optional. The deviance of the best of many placements
        is a maximum over positions, and ADD_NATS was measured only on
        placements FIND had already screened; without the screen every bright
        emitter collected faint satellites on its wings (130 emitters for 107
        true on seed 17, precision 0.74)."""
        _, b, A, cy, cx, sg = state
        box = self.box
        model = core._model_any(b, A, cy, cx, sg, box.yy, box.xx, self.sigma,
                                self.slack, halo=box.halo)
        resid = box.sub - model
        nr = resid / np.sqrt(np.maximum(model, 1e-6))
        log_f = (-ndi.gaussian_laplace(nr, self.sigma, mode="nearest")
                 / core.log_kernel_l2(self.sigma))
        log_f = np.where(box.owned, log_f, -np.inf)
        iy, ix = np.unravel_index(int(np.argmax(log_f)), log_f.shape)
        if not log_f[iy, ix] > self.threshold:
            return None
        return (float(iy), float(ix),
                max(float(resid[iy, ix]), 1e-2) / psf.peak_factor(self.sigma))

    def run(self, k_max):
        box = self.box
        e = np.empty(0)
        state = self.fit(box.bg, e, e, e, e)
        # FORWARD: one placement at a time, all K+1 refit jointly.
        while len(state[2]) < k_max:
            p = self.placement(state)
            if p is None:
                break
            _, b, A, cy, cx, sg = state
            trial = self.fit(b, np.append(A, p[2]), np.append(cy, p[0]),
                             np.append(cx, p[1]), np.append(sg, self.sigma))
            if not state[0].I - trial[0].I > ADD_NATS:
                break
            state = trial
        # BACKWARD elimination: every emitter's removal is scored against the
        # current fit, and the cheapest goes while it costs < ADD_NATS. Not
        # at K = 1: that removal is the K = 0 fit FORWARD already beat by more
        # than ADD_NATS. Measured: output bit-identical, 11-25% fewer fits.
        while len(state[2]) > 1:
            r, b, A, cy, cx, sg = state
            best = None
            for k in range(len(A)):
                keep = np.arange(len(A)) != k
                reduced = self.fit(b, A[keep], cy[keep], cx[keep], sg[keep])
                if best is None or reduced[0].I < best[0].I:
                    best = reduced
            if not best[0].I - r.I < ADD_NATS:
                break
            state = best
        return state


def localize_boxes(data_img, sigma=1.2, offset=0.0, gain=None, read_noise=0.0,
                   roi=None, k_max=12, threshold=None, slack=core.SIGMA_SLACK,
                   band=core.FOCUS_BAND, impl="rs", sweeps=SWEEPS, polish=True,
                   verbose=0):
    """Box-local localization. Returns a `DetectResult`.

    `slack` and `band` mean what they mean in `detect`: the widths a fit may
    represent, and the widths reported as in-focus detections. Everything
    outside `band` is still fitted, and returned in `width_rejects`.
    `history[0]` records the box and fit counts.

    `read_noise` is the camera's read noise in e- rms, from the same
    calibration as `gain` and `offset`; see the read-noise note above. With
    `gain=None` the gain estimate does not account for it.

    `roi`, a boolean array shaped like the frame, confines the search:
    candidates outside it are dropped, so no box forms there, and a box
    places emitters only on ROI pixels. A box still fits every pixel of its
    rectangle, so an emitter on the ROI's edge keeps its whole PSF. Fitted
    positions are not clipped to the ROI.
    """
    be = backend_mod.get(impl)
    raw = np.asarray(data_img, dtype=float)
    H, W = raw.shape
    if roi is not None:
        roi = np.asarray(roi, dtype=bool)
        if roi.shape != raw.shape:
            raise ValueError(f"roi has shape {roi.shape}, frame {raw.shape}")
    g_eff = calibrate.estimate_gain(raw, offset) if gain is None else float(gain)
    shift = float(read_noise) ** 2
    d_e = (raw - offset) / g_eff + shift
    if threshold is None:
        threshold = calibrate.seed_threshold(d_e.shape, sigma)
    b0 = max(float(np.percentile(d_e, 10.0)), core.BG_FLOOR)
    bmap = np.full((H, W), b0)

    cand, camp, strength = core.find_candidates(d_e, bmap, sigma,
                                                np.empty((0, 2)), threshold)
    # The smooth background, from the pixels no candidate reaches -- every
    # candidate, the ROI's or not, so light outside the ROI stays masked. It
    # is a known shape in every box and in the polish; each keeps a free
    # level. See the note above `_render`.
    bmap = core.background_map(d_e, cand, sigma)
    if roi is not None and len(cand):
        inside = roi[cand[:, 0].astype(int), cand[:, 1].astype(int)]
        cand, camp, strength = cand[inside], camp[inside], strength[inside]
    boxes = build_patches(cand, sigma, d_e.shape,
                          link_radius_factor=core.LINK_FACTOR,
                          halo_radius_factor=core.HALO_FACTOR,
                          bbox_pad_factor=core.BBOX_PAD, k_max=k_max)
    # Brightest first, so the strongest light is already fitted when its
    # neighbours read it through their halos.
    boxes.sort(key=lambda p: -float(strength[p.indices].max()))
    # What each box currently holds: until a box is first decided, its own
    # candidates stand in for it as in-focus seeds.
    held = [(cand[p.indices], camp[p.indices],
             np.full(len(p.indices), float(sigma))) for p in boxes]

    n_fits = 0
    for _ in range(sweeps):
        for i, patch in enumerate(boxes):
            own = cand[patch.indices]
            other = cand[np.setdiff1d(np.arange(len(cand)), patch.indices)]
            # The halo: what every OTHER box holds right now, near enough to
            # put flux in this one.
            rest = [held[j] for j in range(len(boxes)) if j != i]
            src_pos = np.vstack([np.empty((0, 2))] + [r[0] for r in rest])
            src_amp = np.concatenate([np.empty(0)] + [r[1] for r in rest])
            src_sig = np.concatenate([np.empty(0)] + [r[2] for r in rest])
            near = _near_rect(src_pos, src_sig, sigma, patch.y0, patch.x0,
                              patch.y1, patch.x1)
            h, w = patch.y1 - patch.y0, patch.x1 - patch.x0
            yy, xx = np.mgrid[0:h, 0:w] * 1.0
            halo = _render(src_pos[near] - [patch.y0, patch.x0],
                           src_amp[near], src_sig[near], yy, xx) + 0.0 * yy
            level, shape_ = core._window_bg(bmap, patch.y0, patch.x0,
                                            patch.y1, patch.x1)
            halo = halo + shape_
            box = _Box(d_e, patch, own, other, halo, sigma, level, roi)
            search = _Search(box, sigma, slack, be, threshold)
            _, _, A, cy, cx, sg = search.run(k_max)
            n_fits += search.n_fits
            held[i] = (np.stack([cy + patch.y0, cx + patch.x0], 1)
                       if len(A) else np.empty((0, 2)), A, sg)

    pos = np.vstack([np.empty((0, 2))] + [h_[0] for h_ in held])
    amp = np.concatenate([np.empty(0)] + [h_[1] for h_ in held])
    sig = np.concatenate([np.empty(0)] + [h_[2] for h_ in held])

    se = np.full((len(amp), 3), np.nan)
    if polish and len(amp):
        pos, amp, se, sig = core.refine(d_e, pos, amp, sigma, bmap, k_max=k_max,
                                        sigmas=sig, slack=slack,
                                        fit_backend=be)

    model = bmap + calibrate.render_model(pos, amp, sig, bmap.shape, 0.0)
    history = [dict(boxes=len(boxes), candidates=len(cand), search_fits=n_fits,
                    N=int(len(sig)))]
    res = _result(pos, amp, sig, se, bmap, model, d_e, shift, g_eff, sigma,
                  band, slack, history)
    if verbose:
        wr = res.width_rejects
        count = (lambda r: 0 if wr is None else int((wr["reason"] == r).sum()))
        print(f"[boxes] {len(boxes)} boxes, {len(cand)} candidates, "
              f"{n_fits} search fits -> N={len(res.amplitudes)} "
              f"(+{count('too_wide')} wide, {count('too_narrow')} narrow)")
    return res


def _result(pos, amp, sig, se, bmap, model, d_e, shift, g_eff, sigma, band,
            slack, history):
    """Classify every fitted emitter and assemble the `DetectResult`.

    Shared with the native path in `native.py`, so both report by one rule.
    `bmap`, `model` and `d_e` are in shifted units, and `shift` is taken
    back off. `model=None` skips the model and residual images.
    """
    H, W = bmap.shape
    focus = ((sig >= band[0] * sigma) & (sig <= band[1] * sigma)
             if band is not None else np.ones(len(sig), dtype=bool))
    # A source the frame border cuts is not an interior width measurement, so
    # an out-of-band fit there is reported as "edge", never "too_narrow" or
    # "too_wide": a width flag always means a broken interior fit.
    border = np.minimum.reduce([pos[:, 0] + 0.5, H - 0.5 - pos[:, 0],
                                pos[:, 1] + 0.5, W - 0.5 - pos[:, 1]]) \
        if len(sig) else np.empty(0)
    edge = ~focus & (border <= EDGE_MARGIN * sigma)
    narrow = ~focus & ~edge & ((sig < band[0] * sigma) if band is not None
                               else True)
    wide = ~focus & ~edge & ~narrow
    idx = np.arange(len(sig))
    rejects = np.concatenate([
        width_reject_records(idx[m], pos[m], amp[m], sig[m], sigma, reason)
        for m, reason in ((narrow, "too_narrow"), (wide, "too_wide"),
                          (edge, "edge"))])
    return DetectResult(
        positions=pos[focus], amplitudes=amp[focus], sigma=sigma,
        lam=float(focus.sum()) / max(H * W, 1), A_s=float(np.mean(amp[focus]))
        if focus.any() else 0.0, gain=g_eff, background=bmap - shift,
        n_outer_passes=1,
        model_image=None if model is None else model - shift,
        residual=None if model is None else d_e - model,
        se=se[focus], history=history,
        width_rejects=rejects if len(rejects) else None,
        width_filter=dict(band=None if band is None else tuple(band),
                          slack=None if slack is None else tuple(slack)),
        fit_sigma=sig[focus], sigma_ratio=sig[focus] / sigma)
