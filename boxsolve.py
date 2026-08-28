"""Box-sequential detection: solve "how many emitters are in this box?" one
box at a time, commit only what the box owns, then refine everything jointly.

Why this replaces the sweep in `detect.sweep_patches`
----------------------------------------------------
The old loop made model selection a GLOBAL fixed-point problem. Patches were
connected components of the current emitter set, so accepting one move
redrew every group; each group's Fisher matrix, and therefore the guards built
on it, changed with the redrawing rather than with the data. Traced on
beads_60x_still.tif: 0 of 192 sweeps accepted nothing, 0 of 16 rounds reached a
fixed point, and N oscillated with period 2 between 65 and 118 for the whole
run. 2710 of 2725 accepted deaths were forced by `A/SE(A) >= 3` failing -- a
test whose answer depends on which neighbours happen to be free that sweep, not
on the image.

Here the decomposition is a fixed tiling of the IMAGE (boxes.py). Three rules
make each box's answer its own:

  1. A box sees its whole fit region -- core plus pad -- and is free to place
     emitters anywhere in it. It has to be: flux from a neighbour just outside
     the core lands on the core's pixels, and a model that cannot represent it
     will explain it with a spurious core emitter instead.

  2. A box COMMITS only the emitters in its core. Cores partition the image, so
     every emitter has AT MOST one owner. An emitter the box puts in the pad
     ring is truncated by an artificial boundary -- at the box edge it loses
     half its support -- so this box is not entitled to an opinion about it. It
     is discarded when the box finishes, and the box that owns that ring
     decides for itself with a truncation-free view.

     "At most one" is the whole guarantee: the partition rules out double
     counting, but NOT loss. Ownership is tested on each box's own fitted
     position, and the two boxes sharing a seam estimate that position from
     different data. An emitter within their disagreement of a seam is placed
     just past it by both, and both discard it. `_adopt_orphans` closes that
     gap at the end of each pass.

  3. The box's residual is built FROM SCRATCH: the committed emitters inside
     the fit region are re-fit as free parameters (initialized from their
     committed values), and only the committed emitters OUTSIDE it contribute
     a frozen halo. Nothing carries over from the previous box's arithmetic.

Boxes are visited in sequence and each sees what the previous ones committed
(Gauss-Seidel), not a frozen snapshot (Jacobi). That is the other half of why
this converges: with Jacobi, two boxes could both claim the same flux in the
same pass and both be surprised in the next one.

Count first, then CRLB
----------------------
`solve()` decides HOW MANY. Its per-box fits exist to make that decision, not
to be the final answer: a box's emitter is fitted against a boundary the image
does not have, and its background is a local nuisance parameter. Once N is
stable, `refine()` re-fits everything at FIXED N in connected groups spanning
box boundaries, which is where the parameter estimates and their CRLBs come
from. Model selection and estimation are separate problems and are answered by
separate fits on purpose.
"""

import numpy as np
import scipy.ndimage as ndi

import boxes as box_mod
import calibrate
import lmga
import msearch
import patches as patch_mod
import psf
import tracer

__all__ = ["solve", "refine", "detect_boxes"]


SEED_THRESHOLD = calibrate.LOG_SEED_THRESHOLD


HALO_TRUNCATE = 4.0
# Distance, in sigma, beyond which a committed emitter is not rendered into a
# box's frozen halo. Matched deliberately to `calibrate.render_model`'s own
# `truncate=4.0`: without it, the model each box FITS AGAINST and the model
# finally REPORTED are not the same function.
#
# It is also the pipeline's only super-linear term. `_box_halo` used to render
# every committed emitter outside the fit region, with no cutoff, so its cost
# was O(N) per box and O(N * B) = O(area^2) per pass. Measured at the real bead
# density (0.047/px^2), emitters rendered per box and the share of total
# runtime it took:
#
#     size     N     boxes   emitters/box   halo share
#      39      64      206        53.0          0.5%
#      78     244     1060       222.8          1.7%
#     117     549     2128       506.5          3.3%
#     156     983     3039       902.8          5.0%
#     234    2191     8980      2054.4         11.6%
#
# At 234x234 only ~1.5% of those emitters are within 4 sigma of their box; the
# rest each contribute under 1e-4 e-. The share grows linearly with area, so on
# a 512x512 field this term alone would have been most of the runtime.


def _box_halo(positions, amplitudes, keep_out, sigma, yy, xx, y0, x0):
    """Frozen contribution of the committed emitters OUTSIDE the fit region.

    Only those within `HALO_TRUNCATE * sigma` of the box are rendered; see the
    constant above for why the cutoff exists and why it is 4.
    """
    idx = np.nonzero(~keep_out)[0]
    if idx.size == 0:
        return 0.0
    pos = np.asarray(positions)[idx]
    ly = pos[:, 0] - y0
    lx = pos[:, 1] - x0
    # Distance from the emitter to the box rectangle, per axis. The rejection
    # is a cheap comparison; what it saves is the erf evaluation over the whole
    # box grid that rendering each emitter costs.
    rad = HALO_TRUNCATE * sigma
    h, w = yy.shape
    dy = np.maximum(np.maximum(-0.5 - ly, ly - (h - 0.5)), 0.0)
    dx = np.maximum(np.maximum(-0.5 - lx, lx - (w - 0.5)), 0.0)
    near = (dy <= rad) & (dx <= rad)
    if not near.any():
        return 0.0
    return psf.model(psf.pack(0.0, np.asarray(amplitudes)[idx][near],
                              ly[near], lx[near]), yy, xx, sigma)


def _seed_box(sub, base_model, sigma, existing_local, threshold=SEED_THRESHOLD):
    """LoG candidates on this box's own residual.

    The search's ADD move would find these one at a time, each costing a full
    fit and a full proposal round. Seeding hands the search a configuration
    that is already roughly right and leaves ADD to mop up what the seeding
    missed, which is what it is good at.

    Seeding from the projected score of `msearch`/`score.py` instead -- so that
    seeds and proposals use one statistic -- was tried and is much WORSE, for a
    reason worth recording: one score map places one emitter. The map is a
    statement about adding a single emitter to the CURRENT model, so reading
    several peaks off it at once double-counts every neighbourhood where two
    peaks compete for the same flux. Measured on beads_60x_still.tif at
    z >= 4.0: 618 seeds against this function's 140, and 391735 LM iterations
    against 35031 (31.2 s against 2.8 s) for the same audit -- the surplus
    seeds all being pruned again by DEATH. A LoG filter has no such problem
    because it never claims to be a fit. Raising LOG_SEED_THRESHOLD does not
    help either (2.5 gives 134 seeds and slightly MORE work), so this
    threshold is left where it is.
    """
    resid = sub - base_model
    nr = resid / np.sqrt(np.maximum(base_model, 1e-6))
    log_f = -ndi.gaussian_laplace(nr, sigma)
    win = 2 * int(np.ceil(sigma)) + 1
    peaks = (log_f == ndi.maximum_filter(log_f, size=win)) & (log_f > threshold)
    ys, xs = np.nonzero(peaks)
    cand = np.stack([ys, xs], axis=1).astype(float)
    if len(cand) and len(existing_local):
        d = np.min(np.linalg.norm(
            cand[:, None, :] - np.atleast_2d(existing_local)[None, :, :], axis=-1), axis=1)
        cand = cand[d > sigma]
    if len(cand) == 0:
        return cand, np.empty((0,)), log_f, nr
    amps = np.maximum(resid[cand[:, 0].astype(int), cand[:, 1].astype(int)], 1e-2)
    return cand, amps / psf.peak_factor(sigma), log_f, nr


def solve(
    d_e, sigma, lam, A_s, background,
    core=None, pad=None, k_max=16, n_passes=8, jitter=True, n_stable=2,
    max_iter=40, out_margin=None, verbose=1,
):
    """Decide the emitter count and rough parameters, box by box.

    Returns (positions, amplitudes, background, history) where `history` is one
    dict per pass: N, how many emitters each pass committed and discarded, and
    how far the committed set moved.

    Convergence is declared on N alone, held for `n_stable` passes OF THE SAME
    LATTICE PHASE (see the stopping rule below). `n_stable=2` -- one repeat per
    phase -- because with `jitter` on, a phase repeating its own count is
    already a full lattice cycle of agreement; measured on both bead frames the
    answer is final by pass 4 and this rule stops at 6.
    The positions are deliberately not part of the test: `jitter` moves the
    core lattice between passes, so consecutive passes fit each emitter against
    a different box boundary and a sub-pixel wobble of 0.1-0.4 px between them
    is the expected behaviour of a correct solver, not a failure to converge.
    What must stop moving is the COUNT -- that is the question the boxes are
    answering. The parameters are settled afterwards by `refine`, at fixed N,
    in groups that ignore box boundaries entirely.
    """
    H, W = d_e.shape
    if out_margin is None:
        # OFF by default, on measurement rather than on principle.
        #
        # The principle says it should help: flux from a bead centred outside
        # the frame is real, it lands on the rim pixels, and with positions
        # clamped to the data the only way to explain it is to rail an emitter
        # at the bound (traced: x = 15.50 against a bound of 15.5), where the
        # Laplace evidence is not valid and the search kills and re-creates it
        # on alternate passes. Letting the centre leave the frame does fix
        # that locally -- on the 16x16 crop it took the over-modelled rim peaks
        # from 6 to 2.
        #
        # Globally it is a clear loss, because an emitter outside the frame
        # lies in no core and is therefore never committed. A rim bead whose
        # true centre is at y = +0.2 is fitted to y = -0.6 often enough that it
        # is thrown away with its flux, and the global model ends up explaining
        # LESS of the rim than before. Full frame, gain pinned at 4.23:
        #
        #     out_margin   N    audit (whole frame)   z sd   converged
        #        0 px      67   7 missed / 3 piled    1.80   yes, 13 passes
        #        4 px      57  13 missed / 8 piled    3.77   no, N cycles 57<->60
        #
        # Pass a non-zero value when the image is a CROP of a larger field,
        # where the region outside really is full of emitters and none of the
        # rim detections were going to be trustworthy anyway.
        out_margin = 0.0
    positions = np.empty((0, 2))
    amplitudes = np.empty((0,))
    history = []

    for p in range(n_passes):
        # Shifting the core lattice on alternate passes moves the seams. A box
        # boundary that falls between two overlapping emitters puts them in
        # different cores, and neither box ever sees the pair whole; on the
        # next pass that seam is somewhere else.
        c_px = core if core is not None else box_mod.default_geometry(sigma)[0]
        shift = c_px // 2 if (jitter and p % 2 == 1) else 0
        bxs = box_mod.tile((H, W), sigma, core=core, pad=pad,
                           offset=(shift, shift))

        if tracer.active():
            tracer.emit("sweep_start", positions=tracer.snap(positions),
                        amplitudes=tracer.snap(amplitudes), background=background,
                        lam=lam, A_s=A_s, k_max=k_max,
                        patches=[dict(y0=b.y0, x0=b.x0, y1=b.y1, x1=b.x1,
                                      indices=np.empty(0, int),
                                      frozen=np.empty(0, int)) for b in bxs])

        prev_pos, prev_amp = positions, amplitudes
        n_commit = n_discard = 0
        n_adopt = 0
        bg_hat = []
        dropped = []

        for ib, b in enumerate(bxs):
            tracer.set_context(patch=ib, y0=b.y0, x0=b.x0, y1=b.y1, x1=b.x1,
                               cy0=b.cy0, cx0=b.cx0, cy1=b.cy1, cx1=b.cx1)
            sub = np.asarray(d_e[b.y0:b.y1, b.x0:b.x1])
            h, w = sub.shape
            yy, xx = np.mgrid[0:h, 0:w] * 1.0

            free_mask = b.inside(positions) if len(positions) else np.zeros(0, bool)
            halo = _box_halo(positions, amplitudes, free_mask, sigma, yy, xx, b.y0, b.x0)
            loc = (positions[free_mask] - np.array([b.y0, b.x0])
                   if free_mask.any() else np.empty((0, 2)))
            amp0 = amplitudes[free_mask] if free_mask.any() else np.empty((0,))

            base = np.asarray(halo, float) + background
            if len(loc):
                base = base + psf.model(psf.pack(0.0, amp0, loc[:, 0], loc[:, 1]),
                                        yy, xx, sigma)
            seeds, samp, log_f, nr = _seed_box(sub, np.maximum(base, 1e-6),
                                               sigma, loc)
            if tracer.active():
                tracer.emit("box_seed", sub=tracer.snap(sub), nr=tracer.snap(nr),
                            log_f=tracer.snap(log_f), seeds=tracer.snap(seeds),
                            existing=tracer.snap(loc), sigma=sigma,
                            threshold=SEED_THRESHOLD)

            init_pos = np.vstack([loc, seeds]) if len(loc) or len(seeds) else np.empty((0, 2))
            init_amp = np.concatenate([amp0, samp])

            # At the IMAGE border there is no neighbouring box to own the
            # outside, so the box MAY be allowed to place emitters beyond the
            # frame (see msearch._bounds and `out_margin` above; off by
            # default). Interior box edges never get that slack: what lies
            # beyond them is already in the halo.
            plo = (-0.5 - (out_margin if b.y0 == 0 else 0.0),
                   -0.5 - (out_margin if b.x0 == 0 else 0.0))
            phi = (h - 0.5 + (out_margin if b.y1 == H else 0.0),
                   w - 0.5 + (out_margin if b.x1 == W else 0.0))
            res = msearch.search_patch(
                sub, sigma, init_pos, init_amp, lam=lam, A_s=A_s, halo=halo,
                b0=background, k_max=k_max, max_iter=max_iter,
                pos_lo=plo, pos_hi=phi, verbose=max(0, verbose - 2),
            )
            bg_hat.append(res.background)

            # --- commit only what this box owns -----------------------------
            gpos = res.positions + np.array([b.y0, b.x0]) if len(res.positions) \
                else np.empty((0, 2))
            keep = b.owns(gpos) if len(gpos) else np.zeros(0, bool)
            n_commit += int(keep.sum())
            n_discard += int((~keep).sum())
            for q in np.nonzero(~keep)[0]:
                dropped.append((gpos[q, 0], gpos[q, 1], res.amplitudes[q], ib,
                                _view_margin(b, gpos[q])))

            stay = ~b.owns(positions) if len(positions) else np.zeros(0, bool)
            positions = (np.vstack([positions[stay], gpos[keep]])
                         if stay.any() or keep.any() else np.empty((0, 2)))
            amplitudes = np.concatenate([amplitudes[stay], res.amplitudes[keep]])

            if tracer.active():
                tracer.emit("commit", committed=tracer.snap(gpos[keep]),
                            discarded=tracer.snap(gpos[~keep]),
                            positions=tracer.snap(positions),
                            amplitudes=tracer.snap(amplitudes),
                            background=background, accepted=list(res.accepted))

        # An emitter within the two boxes' disagreement of a shared core seam
        # can be placed on the far side of it by BOTH and so discarded twice.
        # Cores partitioning the image prevents double counting; it does not
        # prevent loss. See `_adopt_orphans`.
        positions, amplitudes, n_adopt = _adopt_orphans(
            positions, amplitudes, dropped, sigma)
        n_commit += n_adopt
        n_discard -= n_adopt
        if tracer.active() and n_adopt:
            # Whole-frame context: this decision belongs to the pass, not to
            # whichever box happened to run last.
            tracer.set_context(patch=-1, y0=0, x0=0, y1=H, x1=W,
                               cy0=0, cx0=0, cy1=H, cx1=W)
            tracer.emit("adopt", positions=tracer.snap(positions),
                        amplitudes=tracer.snap(amplitudes),
                        adopted=tracer.snap(positions[-n_adopt:]),
                        background=background)

        # Background from the pixels no emitter reaches, not from the median of
        # the boxes' free `b`: a box in a dense region has no such pixels, and
        # its `b` has absorbed whatever the model did not explain.
        background = calibrate.robust_background(d_e, positions, sigma)

        moved = _max_shift(prev_pos, positions)
        rec = dict(pass_=p, N=len(positions), committed=n_commit,
                   discarded=n_discard, background=background, max_shift=moved,
                   dN=len(positions) - len(prev_pos), n_boxes=len(bxs),
                   adopted=n_adopt)
        history.append(rec)
        if verbose >= 1:
            shift_txt = "n/a (N changed)" if not np.isfinite(moved) else f"{moved:.3f} px"
            adopt_txt = f" adopted={n_adopt}" if n_adopt else ""
            print(f"  [box pass {p}] N={len(positions):3d} (dN={rec['dN']:+d})  "
                  f"committed={n_commit} discarded={n_discard}{adopt_txt}  "
                  f"bg={background:.3f}  shift={shift_txt}")
        if tracer.active():
            tracer.emit("sweep_end", positions=tracer.snap(positions),
                        amplitudes=tracer.snap(amplitudes), background=background,
                        changed=bool(rec["dN"] or moved > 0.05), n_patches=len(bxs))

        # Convergence is PER LATTICE PHASE. `jitter` alternates the core
        # lattice every pass, so consecutive passes answer the question from
        # different box boundaries and the two phases legitimately disagree
        # about the handful of emitters that sit near a seam. Asking for N to
        # hold over CONSECUTIVE passes therefore cannot succeed while jitter is
        # on -- measured on both bead frames it never once fired, and the loop
        # ran the full `n_passes` every time printing a warning about a
        # two-cycle that is the expected behaviour of a correct solver:
        #
        #     FOV1  [65, 67, 65, 68, 65, 68, 65, 68]
        #     FOV2  [128, 125, 127, 126, 127, 126, 127, 126]
        #
        # Both are converged by pass 6 in the only sense available: each phase
        # has stopped changing. Comparing same-phase passes (stride `period`)
        # detects that and stops, and still reduces to the old consecutive test
        # when jitter is off.
        #
        # NOTE the count returned is the one belonging to the phase the loop
        # happens to stop on (68 not 65, 126 not 127). The phases disagree
        # about emitters near a seam and each is right about different ones;
        # that disagreement is a real open problem, not something this
        # stopping rule decides.
        period = 2 if jitter else 1
        Ns = [h["N"] for h in history]
        if len(Ns) >= period * n_stable:
            stable = all(
                len(set(Ns[-1 - q :: -period][:n_stable])) == 1
                for q in range(period)
            )
            if stable:
                if verbose >= 1:
                    phases = [Ns[-1 - q] for q in range(period)][::-1]
                    print(f"  [box] N stable per lattice phase at "
                          f"{phases if period > 1 else phases[0]} for "
                          f"{n_stable} passes -- converged after {p + 1}")
                break
    else:
        if verbose >= 1 and history:
            print(f"  [box] WARNING: N never settled in {n_passes} passes; "
                  f"trajectory {[h['N'] for h in history]}")

    return positions, amplitudes, background, history


def _view_margin(b, gpos):
    """How far inside `b`'s FIT region the point `gpos` (global) sits, in px.

    Pixel i covers [i-0.5, i+0.5), so the fit region spans [y0-0.5, y1-0.5).
    Larger is better: it is the amount of the emitter's support the box can
    actually see, and it is what decides whose estimate to keep when two boxes
    both fitted the same emitter (see `_adopt_orphans`).
    """
    return float(min(gpos[0] - (b.y0 - 0.5), (b.y1 - 0.5) - gpos[0],
                     gpos[1] - (b.x0 - 0.5), (b.x1 - 0.5) - gpos[1]))


def _adopt_orphans(positions, amplitudes, dropped, sigma, min_sep_factor=1.0):
    """Recover emitters that fell through a core seam. Returns
    (positions, amplitudes, n_adopted).

    Cores partition the image, so an emitter has at most one owner -- that is
    what makes double counting impossible. It does NOT make loss impossible,
    because ownership is tested on each box's OWN fitted position and the two
    boxes sharing a seam estimate that position from different data. They
    routinely disagree by ~0.1 px. An emitter that close to a seam is therefore
    placed just past it by BOTH boxes, each concludes it belongs to the other,
    and both discard it:

        box core x [12.5, 19.5)   fitted the emitter at x = 19.53  -> discarded
        box core x [19.5, 25.5)   fitted the same one at x = 19.45  -> discarded

    Nothing else in the pass can recover it. Traced on beads_60x_still.tif this
    lost one bead of A = 987 at (34.1, 19.5), which is the whole of that frame's
    remaining interior residual: a score-test peak of z = +70 before `refine`,
    which `refine` then smears into four positive and one negative finding by
    dragging the surrounding emitters at fixed N -- one of them 5 px, onto its
    patch bound. FOV2 lost a bead of A = 1386 the same way. The defect was
    masked until the optimizer was fixed (see lmga._to_interior): while every
    fit stalled near its integer seed, positions never moved close enough to a
    seam for the two boxes to straddle it.

    A discarded emitter is adopted only when the discard was an ownership
    artifact and not a judgement:

      1. no committed emitter within `min_sep_factor * sigma` of it -- if the
         owning box committed one there, nothing was lost;
      2. at least two DIFFERENT boxes fitted it. A box's pad ring is truncated
         and can hold spurious emitters, and the box that owns that ground had
         a truncation-free view and is entitled to say there is nothing there.
         Mutual agreement between the neighbours is what separates "both
         thought it was the other's" from "one of them was wrong".

    The estimate kept is the one from the box that saw the emitter with the
    largest `_view_margin`, i.e. the least truncated view of it.
    """
    if not dropped:
        return positions, amplitudes, 0
    min_sep = min_sep_factor * sigma
    d = np.asarray([(y, x) for y, x, _, _, _ in dropped], dtype=float)

    # Greedy single-link grouping at min_sep; the groups are pairs or triples of
    # near-identical estimates of one emitter, so nothing subtler is warranted.
    order = np.argsort([-a for _, _, a, _, _ in dropped])
    label = np.full(len(dropped), -1, dtype=int)
    n_groups = 0
    for i in order:
        if label[i] >= 0:
            continue
        near = np.nonzero((np.linalg.norm(d - d[i], axis=1) <= min_sep)
                          & (label < 0))[0]
        label[near] = n_groups
        n_groups += 1

    add_pos, add_amp = [], []
    committed = np.atleast_2d(np.asarray(positions, float))
    for g in range(n_groups):
        members = np.nonzero(label == g)[0]
        if len({dropped[i][3] for i in members}) < 2:
            continue                       # only one box ever saw it
        c = d[members].mean(axis=0)
        if len(committed) and np.min(np.linalg.norm(committed - c, axis=1)) <= min_sep:
            continue                       # the owner committed one after all
        best = max(members, key=lambda i: dropped[i][4])
        add_pos.append([dropped[best][0], dropped[best][1]])
        add_amp.append(dropped[best][2])

    if not add_pos:
        return positions, amplitudes, 0
    positions = np.vstack([positions, np.asarray(add_pos, float)]) \
        if len(positions) else np.asarray(add_pos, float)
    amplitudes = np.concatenate([amplitudes, np.asarray(add_amp, float)])
    return positions, amplitudes, len(add_pos)


def _max_shift(a, b):
    """Largest distance any emitter moved, matching nearest neighbours. Returns
    inf when the counts differ, which the caller treats as 'not converged'."""
    a = np.atleast_2d(np.asarray(a, float))
    b = np.atleast_2d(np.asarray(b, float))
    if len(a) != len(b):
        return np.inf
    if len(a) == 0:
        return 0.0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(np.max(np.min(d, axis=1)))


# ------------------------------------------------------------- refinement


def refine(d_e, positions, amplitudes, sigma, background, k_max=12,
           link_radius_factor=2.5, max_iter=200):
    """Joint re-fit at FIXED N in connected groups, plus per-emitter CRLBs.

    This is the estimation half, and it is deliberately a different fit from
    the ones `solve` used to decide the count. Groups here are connected
    components of the committed set (patches.py), so a pair that a box boundary
    separated is fitted jointly at last; no move is proposed, so nothing here
    can change the answer to "how many"; and the standard errors come from the
    Fisher matrix of THIS fit, which is the one whose parameters are reported.

    Returns (positions, amplitudes, background, se) with `se` an (N,3) array of
    (SE_A, SE_y, SE_x) in flux and pixels -- the CRLB at the fitted point.
    """
    positions = np.atleast_2d(np.asarray(positions, float))
    amplitudes = np.asarray(amplitudes, float).ravel()
    if len(amplitudes) == 0:
        return positions, amplitudes, background, np.empty((0, 3))

    se = np.full((len(amplitudes), 3), np.nan)
    out_pos = positions.copy()
    out_amp = amplitudes.copy()
    pset = patch_mod.build_patches(positions, sigma, d_e.shape,
                                   link_radius_factor=link_radius_factor,
                                   k_max=k_max)
    for p in pset:
        yy, xx = patch_mod.patch_grids(p)
        halo = patch_mod.build_halo_image(positions, amplitudes, p.frozen_indices,
                                          sigma, yy, xx, p.y0, p.x0)
        sub = np.asarray(d_e[p.y0:p.y1, p.x0:p.x1])
        K = len(p.indices)
        loc = positions[p.indices] - np.array([p.y0, p.x0])
        theta0 = psf.pack(background, amplitudes[p.indices], loc[:, 0], loc[:, 1])
        A_max = 8.0 * max(float(sub.max()), 1.0) / psf.peak_factor(sigma)
        lo, hi = msearch._bounds(K, sub.shape[0], sub.shape[1],
                                 max(float(sub.max()) * 4.0, 10.0), A_max)
        r = lmga.fit(np.clip(theta0, lo + 1e-9, hi - 1e-9), yy, xx, sigma, sub,
                     lo, hi, halo=halo, max_iter=max_iter)
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

    background = calibrate.robust_background(d_e, out_pos, sigma)
    return out_pos, out_amp, background, se


# ------------------------------------------------------------ entry point


def detect_boxes(
    data_img, sigma=1.2, offset=0.0, gain=None, lam0=0.02, A_s0=None,
    n_outer=1, refine_gain=False, core=None, pad=None, k_max=16, n_passes=8,
    jitter=True, out_margin=None, verbose=1,
):
    """Full detection through the box-sequential solver, returning a
    DetectResult so it is a drop-in comparison against `detect.detect`.

    The outer loop is the empirical-Bayes one and nothing else: it re-estimates
    (lambda, A_s, background, gain) from a CONVERGED configuration and runs the
    box solver again. It is not a substitute for convergence -- `solve` reaches
    its own fixed point inside each pass, and if it does not, that is a result
    to look at rather than something to iterate past.

    `n_outer=1` because a second pass was measured to change nothing. On both
    bead frames, gain pinned, the empirical-Bayes estimates are already at
    their fixed point after pass 0 (FOV2: lam 0.0328, A_s 945.5 then 945.6),
    and the second pass reproduces the result exactly while costing half the
    runtime:

        frame  n_outer   N    audit          z range         time
        FOV1     2      68   3 missed/0 piled  -5.0 .. +26.6   89.8 s
        FOV1     1      68   3 missed/0 piled  -5.0 .. +26.6   44.9 s
        FOV2     2     126   6 missed/3 piled -13.0 .. +43.4  143.4 s
        FOV2     1     126   6 missed/3 piled -13.0 .. +43.4   70.8 s

    Raise it if you are NOT pinning the gain, since that is the one estimate
    that genuinely needs a second look -- but see calibrate.py for why pinning
    a measured gain is the better answer.

    `refine_gain=False` because the gain cannot improve the fit -- only the
    threshold -- and a residual-based estimate of it is inflated by lack of
    fit. See the module docstring of calibrate.py for the exact law.
    """
    raw = np.asarray(data_img, dtype=float)
    H, W = raw.shape
    g_eff = calibrate.estimate_gain(raw, offset) if gain is None else float(gain)
    d_e = (raw - offset) / g_eff
    background = float(np.percentile(d_e, 10.0))
    if A_s0 is None:
        A_s0 = max(float(d_e.max()) - background, 10.0) / psf.peak_factor(sigma)
    lam, A_s = lam0, A_s0

    positions = np.empty((0, 2))
    amplitudes = np.empty((0,))
    history = []

    if tracer.active():
        tracer.emit("init", d_e=tracer.snap(d_e), raw=tracer.snap(raw), sigma=sigma,
                    gain=g_eff, offset=offset, background=background, lam=lam,
                    A_s=A_s, k_max=k_max, border_margin=0.0, n_outer=n_outer)

    for outer in range(n_outer):
        tracer.set_context(outer=outer, rnd=0, sweep=-1)
        positions, amplitudes, background, hist = solve(
            d_e, sigma, lam, A_s, background, core=core, pad=pad, k_max=k_max,
            n_passes=n_passes, jitter=jitter, out_margin=out_margin,
            verbose=verbose,
        )
        for h in hist:
            h["outer"] = outer
        history += hist

        lam = max(len(positions) / float(H * W), 1e-6)
        if len(amplitudes):
            A_s = max(float(np.mean(amplitudes)), 1.0)
        model = calibrate.render_model(positions, amplitudes, sigma, (H, W), background)
        ratio = calibrate.gain_ratio_from_residual(d_e, model, seed=outer)
        if tracer.active():
            tracer.emit("eb_update", positions=tracer.snap(positions),
                        amplitudes=tracer.snap(amplitudes), model=tracer.snap(model),
                        background=background, lam=lam, A_s=A_s, gain=g_eff,
                        gain_ratio=ratio)
        if verbose >= 1:
            print(f"[pass {outer}] N={len(positions):3d}  lam={lam:.4f}  "
                  f"A_s={A_s:.1f}  g_eff={g_eff:.3f}  gain_ratio={ratio:.3f}")
        if refine_gain and np.isfinite(ratio) and ratio > 0:
            f = float(np.clip(ratio ** 2, 0.8, 1.25))
            g_eff *= f
            d_e = (raw - offset) / g_eff
            amplitudes = amplitudes / f
            background /= f
            A_s /= f

    # ---- estimation, at fixed N ----
    tracer.set_context(outer=n_outer, rnd=-2, sweep=-1)
    positions, amplitudes, background, se = refine(
        d_e, positions, amplitudes, sigma, background, k_max=k_max)

    model = calibrate.render_model(positions, amplitudes, sigma, (H, W), background)
    if tracer.active():
        tracer.emit("final", positions=tracer.snap(positions),
                    amplitudes=tracer.snap(amplitudes), model=tracer.snap(model),
                    d_e=tracer.snap(d_e), background=background, gain=g_eff,
                    lam=lam, A_s=A_s, sigma=sigma)
    if verbose >= 1:
        nr = (d_e - model) / np.sqrt(np.maximum(model, 1e-6))
        print(f"[final] N={len(positions)}  g_eff={g_eff:.3f}  bg={background:.2f}  "
              f"resid median={np.median(nr):+.3f}  "
              f"robust_std={calibrate.robust_spread(nr):.3f}")
    from structs import DetectResult
    return DetectResult(
        positions=positions, amplitudes=amplitudes, sigma=sigma, lam=lam, A_s=A_s,
        gain=g_eff, n_outer_passes=n_outer, model_image=model,
        residual=d_e - model, background=background, se=se, history=history,
    )
