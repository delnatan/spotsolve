"""Where should the next emitter go? -- the projected score test.

This module answers, without fitting anything, the question every ADD proposal
needs answered first: *if one more emitter were added at (y, x), how many
standard errors would its amplitude be from zero?*

That is the same quantity `evidence.RESOLVED_TAU` reads off the CONVERGED fit
and uses to block a proposal, which is why it is worth computing beforehand:
measured on beads_60x_still.tif, 80% of all add proposals were fitted to
convergence -- at ~75 LM iterations each, half of them exhausting the iteration
cap -- only to be discarded because that number came out below 3.

The statistic
-------------
Linearize about the incumbent fit: design J = d(model)/d(theta) (n x p),
weights W = diag(1/m), residual r = d - m. Adding a unit-flux PSF `g` at (y, x)
appends one column, and the ML amplitude of that column with EVERY existing
parameter free is the least-squares coefficient on the part of `g` that the
current model cannot already produce,

    g_perp = g - J F^-1 J^T W g ,      F = J^T W J

so, writing u = J^T W g and q = J^T W r (which is -grad, and zero at a
converged fit),

    den   = g^T W g - u^T F^-1 u          <- 1 / Var(A_new), MARGINAL
    num   = r^T W g - u^T F^-1 q
    A_hat = num / den ,   SE = den^-1/2 ,   z = num / sqrt(den)

`den` is the Schur complement of the new amplitude against the whole incumbent
model. Marginal, not conditional: it already charges for the flux a neighbour
could have absorbed instead, which is exactly what makes this comparable to the
post-fit A/SE(A) rather than merely correlated with it.

Why this replaces BOTH birth and split
--------------------------------------
The projection is what makes one move do the work of two. Near an existing
emitter, `g` is largely reproducible by that emitter's own amplitude and
position columns, and the projection removes precisely that part; what survives
is the residual quadrupole an unresolved pair leaves behind. So a peak of `z`
sitting on top of an emitter IS the split proposal, found from the data instead
of from a fixed displacement along an eigenvector, and a peak in open ground is
the birth proposal. The two moves differed only in how they guessed; they never
differed in what they claimed.

`evidence.log_bf_add` needs no change to accept this, because it takes the
amplitude-prior term from the actual flux difference: a genuine birth raises
total flux and is charged A_new/A_s, a split redistributes it and is not.

The warm start comes free
-------------------------
The same solve gives the whole linearized step, not just the new amplitude. The
augmented Gauss-Newton system at the incumbent (where the existing gradient is
zero and A_new starts at zero) is

    [ F   u ] [ dtheta  ]   [  0  ]
    [ u^T c ] [ A_new   ] = [ r^T W g ]

whose solution is A_new = A_hat and dtheta = -A_hat * F^-1 u -- the existing
emitters' rebalancing, and F^-1 u is already computed for `den`. So the proposal
starts at the joint linearized optimum rather than at a guess, which is what
takes the fit's iteration count down.

Cost
----
The whole map is four small matmuls plus one p x p solve, vectorized over
candidate positions, because the PSF is separable: with Ey (h x ny) and
Ex (w x nx),

    r^T W g  at all positions  =  Ey^T (r*W) Ex

and likewise for g^T W g and for each column of u. On a 15x15 box with K=16
that is ~2 Mflop per search step, against ~0.5 Mflop for a single LM iteration
of a fit that would have run for 75 of them.
"""

import numpy as np

import psf

__all__ = ["AddContext", "add_context", "candidates", "warm_start"]


class AddContext:
    """Everything about one incumbent fit that the score map needs.

    Built once per search step and reused for every candidate position.
    """

    __slots__ = ("theta", "ay", "ax", "sigma", "m", "W", "r", "J", "Cinv",
                 "q", "Cq", "h", "w", "ok")

    def __init__(self, theta, ay, ax, sigma, sub, halo):
        self.theta = np.asarray(theta, float)
        self.ay, self.ax, self.sigma = ay, ax, sigma
        m, J = psf.model_and_jac_ax(self.theta, ay, ax, sigma, halo)
        m = np.maximum(m, 1e-9)
        self.h, self.w = m.shape
        self.m = m
        self.W = 1.0 / m
        self.r = np.asarray(sub, float) - m
        p = self.theta.size
        J2 = J.reshape(-1, p)
        self.J = J
        F = J2.T @ (self.W.reshape(-1)[:, None] * J2)
        self.q = J2.T @ (self.W.reshape(-1) * self.r.reshape(-1))
        try:
            self.Cinv = np.linalg.inv(F)
            self.Cq = self.Cinv @ self.q
            self.ok = np.all(np.isfinite(self.Cinv))
        except np.linalg.LinAlgError:
            self.Cinv, self.Cq, self.ok = None, None, False


def add_context(theta, ay, ax, sigma, sub, halo):
    return AddContext(theta, ay, ax, sigma, sub, halo)


DEN_REL_FLOOR = 1e-8
# `den` is a Schur complement of a positive definite matrix, so it is positive
# in exact arithmetic. It is NOT reliably positive in floating point: it is
# formed as the difference g'Wg - u'F^-1 u, and when the candidate PSF is
# nearly in the span of the incumbent's columns those two terms very nearly
# cancel. With an ill-conditioned F the difference can come out at the level of
# its own rounding error, or negative.
#
# Clamping it at a small ABSOLUTE floor -- which this did at first -- is the
# worst thing to do, because z = num/sqrt(den) then reports a colossal score
# exactly where the model has the LEAST information about a new emitter. A
# golden-fixture case with three emitters clustered in a 4.5 px box produced
# z = 4.1e12 and two candidates at garbage positions, on a residual that was
# pure noise. The screen is supposed to be an upper bound on A/SE; an
# unguarded division makes it unbounded in the wrong direction.
#
# The floor is therefore RELATIVE to g'Wg, which is the natural scale (it is
# what `den` would be if the incumbent explained none of `g`). den/g'Wg is the
# fraction of the candidate's Fisher information that the existing model
# cannot already produce. Below ~1e-8 of it the new amplitude is
# unidentifiable, so the honest answer is not "infinitely significant" but
# "this adds nothing": z = 0, A_hat = 0.


def _terms(ctx, ys, xs):
    """(num, den, U) on the outer product grid ys x xs.

    U is (p, ny, nx), the J^T W g vector at every candidate; the caller needs
    it for the warm start, and it is the expensive part, so it is returned
    rather than recomputed.

    Positions where the projection is numerically degenerate are returned as
    num = 0, den = g'Wg, so that z and A_hat are exactly zero there rather
    than enormous. See DEN_REL_FLOOR.
    """
    ys = np.atleast_1d(np.asarray(ys, float))
    xs = np.atleast_1d(np.asarray(xs, float))
    Ey = psf._shape(ctx.ay, ys, ctx.sigma)          # (h, ny)
    Ex = psf._shape(ctx.ax, xs, ctx.sigma)          # (w, nx)

    rW = ctx.r * ctx.W
    num = Ey.T @ rW @ Ex                            # (ny, nx)
    gWg = (Ey * Ey).T @ ctx.W @ (Ex * Ex)

    gWg = np.maximum(gWg, 1e-30)
    if not ctx.ok:
        return num, gWg, None

    # u[c] = Ey^T (J_c * W) Ex, batched over the p columns of J.
    JW = ctx.J * ctx.W[:, :, None]                  # (h, w, p)
    U = np.einsum("ia,ijc,jb->cab", Ey, JW, Ex, optimize=True)
    p, ny, nx = U.shape
    Uf = U.reshape(p, -1)
    V = ctx.Cinv @ Uf                               # F^-1 u
    den = gWg - np.einsum("ck,ck->k", Uf, V).reshape(ny, nx)
    num = num - (Uf.T @ ctx.Cq).reshape(ny, nx)

    # Degenerate projection: the incumbent already spans this candidate to
    # within rounding, so the added amplitude is unidentifiable here. Report
    # no information rather than a division by a cancellation residue.
    bad = ~(den > DEN_REL_FLOOR * gWg) | ~np.isfinite(den) | ~np.isfinite(num)
    if bad.any():
        num = np.where(bad, 0.0, num)
        den = np.where(bad, gWg, den)
    return num, den, U


def score_map(ctx, step=0.5, pos_lo=None, pos_hi=None):
    """(z, A_hat, ys, xs) on a grid of candidate positions over the patch."""
    ylo = -0.5 if pos_lo is None else max(pos_lo[0], -0.5)
    xlo = -0.5 if pos_lo is None else max(pos_lo[1], -0.5)
    yhi = ctx.h - 0.5 if pos_hi is None else min(pos_hi[0], ctx.h - 0.5)
    xhi = ctx.w - 0.5 if pos_hi is None else min(pos_hi[1], ctx.w - 0.5)
    ys = np.arange(0.0, ctx.h - 1e-9, step)
    xs = np.arange(0.0, ctx.w - 1e-9, step)
    ys = ys[(ys >= ylo) & (ys <= yhi)]
    xs = xs[(xs >= xlo) & (xs <= xhi)]
    if ys.size == 0 or xs.size == 0:
        return (np.empty((0, 0)), np.empty((0, 0)), ys, xs)
    num, den, _ = _terms(ctx, ys, xs)
    return num / np.sqrt(den), num / den, ys, xs


def candidates(ctx, z_min, n_max=2, min_sep=None, step=0.5,
               pos_lo=None, pos_hi=None):
    """Local maxima of the projected score, best first.

    Returns a list of (z, cy, cx, A_hat). A candidate is kept only if its
    `z` clears `z_min`, which is what makes this a screen and not merely a
    ranking: `z` is (to first order, and measured to within a few percent) an
    upper bound on the post-fit A/SE(A) of the emitter the proposal would
    create, so a site below the evidence guard's own threshold cannot produce
    a proposal the guard would let through.
    """
    z, Ah, ys, xs = score_map(ctx, step=step, pos_lo=pos_lo, pos_hi=pos_hi)
    if z.size == 0:
        return []
    if min_sep is None:
        min_sep = ctx.sigma
    out = []
    flat = np.argsort(-z, axis=None)
    for idx in flat[: max(64, 8 * n_max)]:
        i, j = np.unravel_index(idx, z.shape)
        if z[i, j] < z_min:
            break
        cy, cx = float(ys[i]), float(xs[j])
        if any(np.hypot(cy - o[1], cx - o[2]) < min_sep for o in out):
            continue
        out.append((float(z[i, j]), cy, cx, float(Ah[i, j])))
        if len(out) >= n_max:
            break
    return out


def warm_start(ctx, cy, cx):
    """The augmented Gauss-Newton step: a full theta with one emitter added at
    (cy, cx) AND the existing emitters rebalanced for it.

    Returns (theta_new, A_hat, z). The rebalancing is what a SPLIT used to
    approximate by halving the parent's flux -- here it is the actual
    linearized solution, at no extra cost, because F^-1 u is already formed.
    """
    num, den, U = _terms(ctx, [cy], [cx])
    A_hat = float(num[0, 0] / den[0, 0])
    z = float(num[0, 0] / np.sqrt(den[0, 0]))
    th = ctx.theta.copy()
    if U is not None and ctx.Cinv is not None:
        th = th - A_hat * (ctx.Cinv @ U.reshape(U.shape[0], -1)[:, 0])
    return np.concatenate([th, [A_hat, cy, cx]]), A_hat, z
