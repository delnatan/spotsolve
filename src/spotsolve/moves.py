"""Proposal constructors: pure transformations of a parameter vector.

Each takes a `theta` (see structs.py for the layout) and returns a new
`theta` with one more emitter. Neither fits anything, scores anything, or
decides anything -- they only propose. Scoring lives in evidence.py and the
accept/reject loop in core.py.

Keeping proposals separate matters because they are where the method's
domain knowledge sits: a good proposal is what lets the optimizer reach the
two-emitter basin at all, and it is the part most worth testing on its own.
"""

import numpy as np

from . import psf

__all__ = ["A_MIN", "residual_axis", "split",
           "residual_axis_var", "split_var"]


A_MIN = 1e-4
# A hard zero amplitude makes that emitter's position block of the Fisher
# matrix identically zero, so F is singular and every evidence term built
# from it is meaningless. A small positive floor keeps the fit well posed;
# an emitter that does not want to be there is removed by `core._prune`,
# on the evidence, rather than by silently collapsing.


def _axis(A_k, cy_k, cx_k, width, yy, xx, resid):
    """The quadrupole axis and its strength, given one emitter's parameters.

    Layout-free so the fixed-sigma and per-emitter-sigma entry points below
    are two unpackings of one calculation rather than two calculations.
    `width` is the scale the second moment is weighted on -- the emitter's own
    width, which is the PSF sigma when widths are not free.
    """
    dy = np.asarray(yy) - cy_k
    dx = np.asarray(xx) - cx_k
    w = np.exp(-(dy**2 + dx**2) / (2.0 * (1.5 * width) ** 2))
    wr = w * resid
    M = np.array([[np.sum(wr * dy * dy), np.sum(wr * dy * dx)],
                  [np.sum(wr * dy * dx), np.sum(wr * dx * dx)]])
    if not np.all(np.isfinite(M)):
        return np.array([1.0, 0.0]), 0.0
    vals, vecs = np.linalg.eigh(M)
    i = int(np.argmax(vals))
    u = vecs[:, i]
    n = np.linalg.norm(u)
    return (u / n if n > 0 else np.array([1.0, 0.0]),
            float(vals[i]) / max(float(A_k), 1e-12))


def residual_axis(theta, k, yy, xx, sigma, resid):
    """Unit vector along the residual quadrupole around emitter k.

    `resid` is the incumbent's residual, sub - model(theta), and the CALLER
    renders it. It does not depend on k, while this function is called once per
    emitter to rank split candidates -- rendering it here made the model cost
    O(K) per search step for a quantity that is the same every time.

    Two emitters closer than about 1.5 sigma are fitted well by one brighter
    PSF. What gives them away is the SHAPE of the residual, not its peak:
    negative at the centre, positive on two opposite lobes along the pair
    axis. The eigenvector of the PSF-weighted second moment of that residual
    recovers the axis, which is where a split should be proposed.

    Returns ((uy, ux), strength), where `strength` is the largest eigenvalue
    of the moment normalized by the emitter's flux -- a scale-free measure of
    how much this emitter looks like an unresolved pair. Callers use it to
    rank which emitters are worth proposing a split for.
    """
    _, A, cy, cx = psf.unpack(theta)
    return _axis(A[k], cy[k], cx[k], sigma, yy, xx, resid)


def residual_axis_var(theta, k, yy, xx, resid):
    """`residual_axis` on a per-emitter-sigma theta, weighted at emitter k's
    OWN width.

    The weight decides which pixels count as "around this emitter". Weighting
    a defocused source at the in-focus width sees only its core, which is
    exactly the region where a broadened PSF and an unresolved pair look most
    alike -- so the ranking this feeds would put defocused singles at the top,
    which is the failure the free width exists to remove.
    """
    _, A, cy, cx, sig = psf.unpack_var_sigma(theta)
    return _axis(A[k], cy[k], cx[k], sig[k], yy, xx, resid)


def _children(A, cy, cx, k, u, disp):
    """(keep, A, cy, cx) for replacing emitter k with two at +/- disp/2 along u."""
    keep = [j for j in range(len(A)) if j != k]
    half = max(A[k] / 2.0, A_MIN)
    return (keep,
            list(A[keep]) + [half, half],
            list(cy[keep]) + [cy[k] + 0.5 * disp * u[0],
                              cy[k] - 0.5 * disp * u[0]],
            list(cx[keep]) + [cx[k] + 0.5 * disp * u[1],
                              cx[k] - 0.5 * disp * u[1]])


def split(theta, k, u, disp):
    """Replace emitter k with two at c_k +/- (disp/2)*u, each of half its flux.

    Total flux is conserved by construction, which is what makes the
    amplitude-prior term in evidence.log_bf_add collapse to -log(A_s).
    """
    b, A, cy, cx = psf.unpack(theta)
    _, A2, cy2, cx2 = _children(A, cy, cx, k, u, disp)
    return psf.pack(b, A2, cy2, cx2)


def split_var(theta, k, u, disp):
    """`split` on a per-emitter-sigma theta. Both children inherit the
    PARENT's width.

    Not the in-focus width: the proposal and the incumbent it is scored
    against must start in the same basin, because the Bayes factor is a
    difference of their two I-divergences and a proposal started worse is
    under-credited -- the bias `lmga.fit`'s convergence note describes.
    """
    b, A, cy, cx, sig = psf.unpack_var_sigma(theta)
    keep, A2, cy2, cx2 = _children(A, cy, cx, k, u, disp)
    return psf.pack_var_sigma(b, A2, cy2, cx2,
                              list(sig[keep]) + [sig[k], sig[k]])
