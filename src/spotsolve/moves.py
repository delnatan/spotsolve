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

__all__ = ["A_MIN", "residual_axis", "split"]


A_MIN = 1e-4
# A hard zero amplitude makes that emitter's position block of the Fisher
# matrix identically zero, so F is singular and every evidence term built
# from it is meaningless. A small positive floor keeps the fit well posed;
# an emitter that does not want to be there is removed by `core._prune`,
# on the evidence, rather than by silently collapsing.


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
    _, _, cy, cx = psf.unpack(theta)
    r = resid
    dy = np.asarray(yy) - cy[k]
    dx = np.asarray(xx) - cx[k]
    w = np.exp(-(dy**2 + dx**2) / (2.0 * (1.5 * sigma) ** 2))
    wr = w * r
    M = np.array([[np.sum(wr * dy * dy), np.sum(wr * dy * dx)],
                  [np.sum(wr * dy * dx), np.sum(wr * dx * dx)]])
    if not np.all(np.isfinite(M)):
        return np.array([1.0, 0.0]), 0.0
    vals, vecs = np.linalg.eigh(M)
    i = int(np.argmax(vals))
    u = vecs[:, i]
    n = np.linalg.norm(u)
    _, A, _, _ = psf.unpack(theta)
    strength = float(vals[i]) / max(float(A[k]), 1e-12)
    return (u / n if n > 0 else np.array([1.0, 0.0])), strength


def split(theta, k, u, disp):
    """Replace emitter k with two at c_k +/- (disp/2)*u, each of half its flux.

    Total flux is conserved by construction, which is what makes the
    amplitude-prior term in evidence.log_bf_add collapse to -log(A_s).
    """
    b, A, cy, cx = psf.unpack(theta)
    keep = [j for j in range(len(A)) if j != k]
    half = max(A[k] / 2.0, A_MIN)
    return psf.pack(
        b,
        list(A[keep]) + [half, half],
        list(cy[keep]) + [cy[k] + 0.5 * disp * u[0], cy[k] - 0.5 * disp * u[0]],
        list(cx[keep]) + [cx[k] + 0.5 * disp * u[1], cx[k] - 0.5 * disp * u[1]],
    )
