"""Greedy evidence-maximizing search over emitter count, for one patch.

Each iteration proposes every enabled move (SPLIT / BIRTH / DEATH / MERGE),
fits each proposal to convergence, and accepts the single BEST one if its
log Bayes factor clears the threshold. Taking the best rather than the first
acceptable move is what makes the loop terminate: log-evidence increases
strictly on every accepted step, so no configuration can repeat.

Proposals come from moves.py, scoring from evidence.py, fitting from
lmga.py. This module contains only the loop.
"""

import numpy as np

import psf
import lmga
import moves
import evidence
import tracer
from structs import SearchResult

__all__ = ["search_patch", "fit_theta", "MOVES_ALL", "MOVES_WITH_MERGE"]


MOVES_ALL = ("split", "birth", "death")
MOVES_WITH_MERGE = ("split", "birth", "death", "merge")

# MERGE is not in the default set, on measurement and on an argument.
#
# The argument: `evidence.log_bf_remove` is the exact negation of
# `log_bf_add`. The search only ever accepts a SPLIT whose log BF is > 0, so
# merging that pair back on the very configuration the split produced has log
# BF < 0 by construction. MERGE can therefore only fire on a configuration
# inherited from an earlier pass whose (lam, A_s, background) have since moved
# -- a narrow window.
#
# The measurement, over a full run on the second bead frame:
#
#     move     proposed   accepted   blocked   max log BF
#     split      1844        56       1605       +512.47
#     birth       466        46        367       +512.56
#     death       932        34          0         -0.20
#     merge       819         0          0         -2.14
#
# 819 fits, no acceptance, and never within 2 nats of one. Since a proposal
# fit is the unit of cost in this pipeline (98% of runtime is lmga.fit, and
# 92% of fits are proposals), that is a fifth of the total work. Dropping it
# leaves both frames bit-identical:
#
#     frame   merge    N    audit             z range          time
#     FOV1     on     68   3 missed/0 piled   -5.0 .. +26.6    26.9 s
#     FOV1     off    68   3 missed/0 piled   -5.0 .. +26.6    21.7 s
#     FOV2     on    126   6 missed/3 piled  -13.0 .. +43.4    39.1 s
#     FOV2     off   126   6 missed/3 piled  -13.0 .. +43.4    31.3 s
#
# The move itself is correct and is kept: pass `enable=MOVES_WITH_MERGE` to
# a search that might inherit an over-split configuration.


A_MIN_REL = 1e-6
# The amplitude floor, as a fraction of this patch's A_max. `moves.A_MIN`
# (1e-4) is an ABSOLUTE floor and is kept only as a backstop for a patch whose
# A_max is itself tiny.
#
# The floor has to be relative because what it protects is a RATIO. An
# emitter's position block of the Fisher matrix scales as A^2, so at bead
# fluxes of ~2000 e- an amplitude of 1e-4 puts those entries at ~5.7e-12
# against a largest diagonal of ~768 -- a ratio of 3e-14, about 130x float64
# epsilon. At that point log|F| is numerical noise, the Occam term of every
# Bayes factor built on it is noise with it, and the LM step along that
# direction is unbounded: traced on such a patch, the fit predicted a 5.2e4
# nat decrease, delivered 1.05e-3, and crawled for 3000+ iterations still
# 364 nats above the optimum.
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
# 1e-6 is chosen rather than something larger because it buys six orders of
# margin over epsilon while remaining physically negligible -- on a bead patch
# it is a floor of ~0.02 e- of total flux. A larger floor would start to
# express an opinion about how faint an emitter may be, and that decision
# belongs to DEATH and the Bayes factor, not to a numerical guard.
#
# Note this cannot be enforced downstream in evidence.logdet_cond instead:
# no test on F alone distinguishes an uninformed parameter from a well-posed
# matrix in badly scaled units. Here the flux scale is known, so it can.


def _bounds(K, h, w, b_max, A_max, pos_lo=None, pos_hi=None):
    """Box constraints for one fit. `pos_lo`/`pos_hi` are (y, x) limits in
    LOCAL coordinates, defaulting to the sub-image itself.

    A caller may widen them past the sub-image, and at the IMAGE border it
    should. Flux from an emitter centred outside the frame is real and lands on
    the frame's edge pixels, but with positions clamped to the data the fit can
    only explain it by railing an emitter at the bound -- where half its support
    is missing, its position sits exactly on the boundary of the parameter
    space, and the Fisher matrix that every Bayes factor is built from is not
    valid. Traced on the bead crop, that railed emitter (x = 15.50 against a
    bound of 15.5) is what the search kills and re-creates on alternate passes,
    and it is what leaves the rim over-modelled. Allowing the centre to leave
    the frame lets the fit put the flux where it actually came from; the caller
    then declines to COMMIT anything outside the image, so nothing
    unidentifiable enters the result.
    """
    a_min = max(moves.A_MIN, A_MIN_REL * A_max)
    ylo, xlo = (-0.5, -0.5) if pos_lo is None else pos_lo
    yhi, xhi = (h - 0.5, w - 0.5) if pos_hi is None else pos_hi
    lo = [0.0]
    hi = [b_max]
    for _ in range(K):
        lo += [a_min, ylo, xlo]
        hi += [A_max, yhi, xhi]
    return np.asarray(lo), np.asarray(hi)


def fit_theta(theta0, yy, xx, sigma, sub, halo, b_max, A_max, max_iter=100,
              pos_lo=None, pos_hi=None):
    """Fit one candidate configuration to convergence under box constraints."""
    K = (len(theta0) - 1) // 3
    lo, hi = _bounds(K, sub.shape[0], sub.shape[1], b_max, A_max, pos_lo, pos_hi)
    th0 = np.clip(np.asarray(theta0, dtype=float), lo + 1e-9, hi - 1e-9)
    return lmga.fit(th0, yy, xx, sigma, sub, lo, hi, halo=halo, max_iter=max_iter)


def _sumA(theta):
    return float(np.sum(moves.unpack(theta)[1]))


def _unresolved_indices(theta, F, tau=evidence.RESOLVED_TAU):
    """Indices of emitters whose amplitude is within `tau` standard errors of
    the A >= 0 boundary, where the Laplace evidence is not usable."""
    K = (len(theta) - 1) // 3
    if K == 0:
        return []
    try:
        var = np.diag(np.linalg.inv(F))[1::3]
    except np.linalg.LinAlgError:
        return list(range(K))
    A = np.asarray(theta)[1::3]
    bad = ~(var > 0) | ~np.isfinite(var) | (A < tau * np.sqrt(np.abs(var)))
    return [int(k) for k in np.nonzero(bad)[0]]


def _guard(cond, rr):
    """The conditioning number a proposal must clear, raised to +inf when the
    proposal's own amplitudes are too close to the A >= 0 boundary for the
    Laplace evidence to mean anything (see evidence.RESOLVED_TAU).

    Routing this through the SAME channel as the conditioning guard is
    deliberate: both are statements that the approximation behind the Bayes
    factor does not hold for this configuration, and both must therefore
    block the move rather than be weighed against it.
    """
    if not evidence.amplitudes_resolved(rr.theta, rr.F):
        return np.inf
    return cond


def _propose(name, res, ctx):
    """Yield (log_bf, label, FitResult) for every proposal of one move type."""
    K = (len(res.theta) - 1) // 3
    fit = lambda t: fit_theta(t, ctx["yy"], ctx["xx"], ctx["sigma"], ctx["sub"],
                              ctx["halo"], ctx["b_max"], ctx["A_max"],
                              pos_lo=ctx["pos_lo"], pos_hi=ctx["pos_hi"])
    lam, A_s = ctx["lam"], ctx["A_s"]

    if name == "split" and K < ctx["k_max"]:
        # Only emitters whose residual actually looks like an unresolved pair
        # are worth splitting. `residual_axis` returns the quadrupole strength
        # alongside the axis; a genuinely single emitter has none, and paying
        # for four fits per emitter regardless is what made the proposal count
        # combinatorial in K.
        cand = []
        for k in range(K):
            u, strength = moves.residual_axis(res.theta, k, ctx["yy"], ctx["xx"],
                                              ctx["sigma"], ctx["resid"])
            cand.append((strength, k, u))
        cand.sort(reverse=True)
        for strength, k, u in cand[:ctx["max_split_cand"]]:
            for direction, disp in ((u, ctx["split_disp"][0]),
                                    (u, ctx["split_disp"][1])):
                rr = fit(moves.split(res.theta, k, direction, disp * ctx["sigma"]))
                bf, cond = evidence.log_bf_add(
                    res.I, rr.I, res.F, rr.F,
                    _sumA(res.theta), _sumA(rr.theta), K, lam, A_s)
                yield bf, f"split[{k}]", rr, _guard(cond, rr)

    elif name == "birth" and K < ctx["k_max"]:
        m, r = ctx["model"], ctx["resid"]
        py, px = np.unravel_index(
            int(np.argmax(r / np.sqrt(np.maximum(m, 1e-9)))), r.shape)
        A0 = max(float(r[py, px]), 1e-3) / psf.peak_factor(ctx["sigma"])
        rr = fit(moves.birth(res.theta, float(py), float(px), A0))
        bf, cond = evidence.log_bf_add(
            res.I, rr.I, res.F, rr.F,
            _sumA(res.theta), _sumA(rr.theta), K, lam, A_s)
        yield bf, "birth", rr, _guard(cond, rr)

    elif name == "death" and K > 0:
        # Rank by amplitude and test only the faintest few. This is a
        # PROPOSAL ordering, not a significance screen -- whichever emitters
        # are proposed still face the full Bayes factor. (The screen this
        # replaces, `A/SE_A < 5`, was a statistical test in disguise and only
        # ever reached emitters below ~12 ADU.)
        _, A_cur, _, _ = moves.unpack(res.theta)
        order = list(np.argsort(A_cur)[:ctx["max_death_cand"]])

        # An emitter whose amplitude is not resolved from the A >= 0 boundary
        # makes the CURRENT configuration's evidence unusable, by the same
        # argument that blocks a proposal from creating one (see _guard and
        # evidence.RESOLVED_TAU). It must always be offered for removal, even
        # if it is not among the faintest few, and removing it is not a
        # judgement call -- the evidence for keeping it cannot be computed.
        # This is the same convention log_bf_remove already uses when the full
        # model's Fisher matrix is not positive definite.
        unresolved = _unresolved_indices(res.theta, res.F)
        for k in unresolved:
            if k not in order:
                order.insert(0, k)

        for k in order:
            rr = fit(moves.drop(res.theta, int(k)))
            if k in unresolved:
                yield np.inf, f"death[{k}]", rr, 0.0
                continue
            bf = evidence.log_bf_remove(
                res.I, rr.I, res.F, rr.F,
                _sumA(res.theta), _sumA(rr.theta), K, lam, A_s)
            yield bf, f"death[{k}]", rr, 0.0

    elif name == "merge" and K > 1:
        # Only each emitter's nearest neighbour is a plausible merge, so this
        # is O(K) pairs rather than O(K^2).
        _, _, cy, cx = moves.unpack(res.theta)
        d = np.hypot(cy[:, None] - cy[None, :], cx[:, None] - cx[None, :])
        np.fill_diagonal(d, np.inf)
        pairs = {tuple(sorted((i, int(np.argmin(d[i]))))) for i in range(K)
                 if d[i].min() <= 3.0 * ctx["sigma"]}
        for i, j in sorted(pairs):
            rr = fit(moves.merge(res.theta, i, j))
            bf = evidence.log_bf_remove(
                res.I, rr.I, res.F, rr.F,
                _sumA(res.theta), _sumA(rr.theta), K, lam, A_s)
            yield bf, f"merge[{i},{j}]", rr, 0.0


def search_patch(
    sub,
    sigma,
    init_pos,
    init_amp,
    lam,
    A_s,
    halo=0.0,
    b0=None,
    b_max=None,
    log_bf_threshold=0.0,
    cond_guard=evidence.COND_GUARD,
    k_max=8,
    max_iter=30,
    split_disp=(1.0, 2.0),
    max_split_cand=2,
    max_death_cand=2,
    enable=MOVES_ALL,
    pos_lo=None,
    pos_hi=None,
    verbose=0,
):
    """Search for the best-supported emitter configuration in one patch.

    `sub` is (h,w) in PHOTOELECTRONS; `init_pos` is (K0,2) in local pixel
    coordinates; `init_amp` is (K0,) total flux. Returns a SearchResult.

    verbose: 0 silent, 1 accepted moves, 2 every proposal considered.
    """
    sub = np.asarray(sub, dtype=float)
    h, w = sub.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ctx = {
        "yy": yy * 1.0, "xx": xx * 1.0, "sigma": sigma, "sub": sub, "halo": halo,
        "lam": lam, "A_s": A_s, "k_max": k_max, "split_disp": split_disp,
        "max_split_cand": max_split_cand, "max_death_cand": max_death_cand,
        "b_max": b_max if b_max is not None else max(float(sub.max()) * 4.0, 10.0),
        "A_max": 8.0 * max(float(sub.max()), 1.0) / psf.peak_factor(sigma),
        "pos_lo": pos_lo, "pos_hi": pos_hi,
    }
    if b0 is None:
        b0 = max(float(np.percentile(sub, 10)), 1e-3)

    init_pos = np.atleast_2d(np.asarray(init_pos, float)) if len(init_pos) else np.empty((0, 2))
    init_amp = np.asarray(init_amp, float).ravel()
    theta0 = (psf.pack(b0, init_amp, init_pos[:, 0], init_pos[:, 1])
              if len(init_pos) else np.asarray([b0]))
    res = fit_theta(theta0, ctx["yy"], ctx["xx"], sigma, sub, halo,
                    ctx["b_max"], ctx["A_max"], pos_lo=pos_lo, pos_hi=pos_hi)

    if tracer.active():
        tracer.emit("search_start", sub=tracer.snap(sub),
                    halo=tracer.snap(np.broadcast_to(np.asarray(halo, float), sub.shape)),
                    init_pos=tracer.snap(init_pos), init_amp=tracer.snap(init_amp),
                    theta=tracer.snap(res.theta), I=res.I, sigma=sigma,
                    converged=res.converged, stalled=res.stalled, n_iter=res.n_iter,
                    k_max=k_max, lam=lam, A_s=A_s)

    accepted, best_refused = [], -np.inf
    stop_reason = "max_iter"

    for _step in range(max_iter):
        # The incumbent's model and residual, rendered ONCE per step. SPLIT
        # ranks every emitter against the residual and BIRTH seeds from its
        # largest normalized peak; neither depends on which proposal is being
        # scored, so this used to be re-rendered K+1 times per step.
        ctx["model"] = psf.model(res.theta, ctx["yy"], ctx["xx"],
                                 ctx["sigma"], ctx["halo"])
        ctx["resid"] = ctx["sub"] - ctx["model"]

        cands = []
        considered = []
        for name in enable:
            for bf, label, rr, cond in _propose(name, res, ctx):
                blocked = cond >= cond_guard
                considered.append((label, float(bf), float(cond), blocked,
                                   tracer.snap(rr.theta) if tracer.active() else None))
                if verbose >= 2:
                    print(f"      {label:14s} log BF = {bf:+9.2f}"
                          f"{'  BLOCKED (cond %.1e)' % cond if blocked else ''}")
                if blocked:
                    best_refused = max(best_refused, bf)
                else:
                    cands.append((bf, label, rr))
        if not cands:
            stop_reason = "no proposal survived the conditioning guard" if considered \
                else "no move was applicable"
            if tracer.active():
                tracer.emit("proposals", step=_step, theta=tracer.snap(res.theta),
                            considered=considered, chosen=None, accepted=False)
            break
        bf, label, rr = max(cands, key=lambda c: c[0])
        if tracer.active():
            tracer.emit("proposals", step=_step, theta=tracer.snap(res.theta),
                        considered=considered, chosen=label,
                        accepted=bool(bf > log_bf_threshold), log_bf=float(bf))
        if bf <= log_bf_threshold:
            best_refused = max(best_refused, bf)
            stop_reason = f"best move {label} log BF {bf:+.2f} <= threshold"
            if verbose >= 1:
                print(f"      stop: best move {label} log BF = {bf:+.2f}")
            break
        if verbose >= 1:
            print(f"      accept {label:14s} log BF = {bf:+9.2f}   "
                  f"K -> {(len(rr.theta) - 1) // 3}")
        if tracer.active():
            tracer.emit("move", step=_step, label=label, log_bf=float(bf),
                        theta_before=tracer.snap(res.theta),
                        theta_after=tracer.snap(rr.theta),
                        I_before=res.I, I_after=rr.I,
                        converged=rr.converged, stalled=rr.stalled, n_iter=rr.n_iter)
        res = rr
        accepted.append(label)

    b, A, cy, cx = moves.unpack(res.theta)
    order = np.argsort(-A)
    if tracer.active():
        tracer.emit("search_end", theta=tracer.snap(res.theta), I=res.I,
                    accepted=list(accepted), rejected_best=float(best_refused),
                    stop_reason=stop_reason, K=len(A))
    return SearchResult(
        positions=np.stack([cy[order], cx[order]], axis=1),
        amplitudes=A[order],
        background=b,
        theta=res.theta,
        I=res.I,
        F=res.F,
        accepted=accepted,
        rejected_best=best_refused,
    )
