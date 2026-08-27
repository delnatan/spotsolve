"""Fixed geometric tiling: which box OWNS a pixel, and which box may SEE it.

This is the piece the patch decomposition in `patches.py` cannot provide.
`build_patches` groups emitters by connected components of the CURRENT
position set, so the grouping is a function of the answer: accept one move and
the next sweep's groups are different. Every quantity computed inside a group
inherits that instability. Measured on beads_60x_still.tif, the amplitude
resolution test `A / SE(A) >= 3` that gates every proposal reads 20.6 for a
pair of bright emitters 1.5 px apart when the neighbour is frozen in the halo
and 3.5 when the same neighbour is free in the same group -- so the same
physical pair is "resolved" on one sweep and "unresolved" on the next, and the
search oscillates with period 2 forever (0 of 192 sweeps reached a fixed
point).

A tiling fixes that by being a function of the IMAGE, not of the model. Two
regions per box:

    core   the pixels this box owns. Cores PARTITION the image exactly: every
           point belongs to the core of exactly one box, so an emitter has
           exactly one owner and can be neither lost nor double-counted.
    box    the core grown by `pad` on every side, clipped to the image. This
           is what the box gets to see and fit.

Only core emitters are committed to the global model. An emitter that the
search places in the pad ring is a working variable: it is there to explain
flux that genuinely belongs to the box's data, and it is discarded when the
box is done, because the box that owns that ring is entitled to a truncation-
free view of it and will decide for itself.

Choosing `pad`
--------------
`pad` is what buys the core a truncation-free view. A core emitter sits at
least `pad` px inside the box on every side, so the fraction of its
pixel-integrated flux the box actually contains is erf((pad+0.5)/(sigma*sqrt2))^2:

    pad / sigma      pad px (sigma=1.2)     flux seen by a core emitter
        2.0                2.4                      0.9905
        2.5                3.0                      0.9970
        3.0                3.6                      0.9996
        4.0                4.8                      0.99999

The default is 3*sigma: at 4 parts in 10^4 the truncation is far below the
Poisson noise on any emitter this pipeline can detect, and it keeps the box
small enough that the joint fit stays cheap. An emitter in the pad RING, by
contrast, can be truncated arbitrarily badly -- one sitting on the box edge
loses half its flux -- which is exactly why the ring is not committed.

Choosing `core`
---------------
`core` looks like a pure cost knob -- the joint fit inside a box is O(K^3) in
its free emitter count, so a large core in a dense field should be expensive --
but it is also an ACCURACY knob, and that turns out to dominate. Every core
boundary is a place where the count decision is made twice, from two different
data windows, and the two answers need not agree. Fewer seams means fewer such
places. Measured on beads_60x_still.tif (gain 4.23, one outer pass):

    core  pad   box    N    time    audit             z range        z sd
      5    4    13x13  66   65 s    7 missed/4 piled  -8.4 .. +62.8  1.73
      5    6    17x17  67  110 s    5 missed/4 piled  -5.6 .. +63.1  1.45
      7    4    15x15  68   45 s    3 missed/0 piled  -5.0 .. +26.6  1.27
      7    6    19x19  66   90 s    3 missed/3 piled  -5.3 .. +63.1  1.37
      9    6    21x21  68   43 s    2 missed/1 piled  -5.0 .. +26.6  1.20

Widening the PAD alone does not buy this -- (5,6) is a 17x17 box and is still
worse than (7,4) at 15x15 -- so it is the seam count that matters, not the
amount of context each box has. It is also faster: fewer, larger boxes beat more,
smaller ones because the per-box overhead (seeding, halo rendering, the first
fit) is paid once per box regardless of size.

The default is therefore 6*sigma (7 px at sigma=1.2), not the 4*sigma this
started with. Raising it further is worth trying on denser data; the joint fit
is the eventual limit, and `k_max` should rise with it.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["Box", "tile", "default_geometry"]


@dataclass
class Box:
    """One tile. (y0,x0,y1,x1) is the fit region, (cy0,cx0,cy1,cx1) the core.

    Both are half-open pixel-index ranges. Cores tile the image exactly; fit
    regions overlap and are clipped at the image border.
    """

    y0: int
    x0: int
    y1: int
    x1: int
    cy0: int
    cx0: int
    cy1: int
    cx1: int

    @property
    def shape(self):
        return (self.y1 - self.y0, self.x1 - self.x0)

    def owns(self, positions):
        """Boolean mask of which of `positions` (global, (N,2), continuous)
        this box's core owns.

        Pixel i covers [i-0.5, i+0.5), so a core spanning pixels [c0, c1)
        covers [c0-0.5, c1-0.5). Written this way the cores partition the
        plane with no gap and no overlap -- an emitter exactly on a boundary
        goes to the higher box, and to exactly one of them.
        """
        p = np.atleast_2d(np.asarray(positions, dtype=float))
        if p.size == 0:
            return np.zeros(0, dtype=bool)
        return (
            (p[:, 0] >= self.cy0 - 0.5) & (p[:, 0] < self.cy1 - 0.5)
            & (p[:, 1] >= self.cx0 - 0.5) & (p[:, 1] < self.cx1 - 0.5)
        )

    def inside(self, positions, margin=0.0):
        """Mask of positions lying within the FIT region (optionally shrunk by
        `margin`). Used to decide which committed emitters become free
        variables of this box rather than part of its frozen halo."""
        p = np.atleast_2d(np.asarray(positions, dtype=float))
        if p.size == 0:
            return np.zeros(0, dtype=bool)
        return (
            (p[:, 0] >= self.y0 - 0.5 + margin) & (p[:, 0] < self.y1 - 0.5 - margin)
            & (p[:, 1] >= self.x0 - 0.5 + margin) & (p[:, 1] < self.x1 - 0.5 - margin)
        )


def default_geometry(sigma, core_factor=6.0, pad_factor=3.0):
    """(core, pad) in pixels. See the module docstring for the trade-offs."""
    core = max(2, int(round(core_factor * sigma)))
    pad = max(2, int(np.ceil(pad_factor * sigma)))
    return core, pad


def tile(shape, sigma, core=None, pad=None, offset=(0, 0)):
    """Tile `shape` into overlapping boxes with exactly partitioning cores.

    `offset` shifts the core lattice by (dy, dx) pixels. Nothing in the solver
    requires it, but a box boundary that cuts through a close pair splits a
    joint fit that should have been joint, and shifting the lattice between
    passes moves that seam somewhere else -- so a pair that one pass could not
    fit jointly gets a pass where it can.
    """
    H, W = shape
    if core is None or pad is None:
        c, p = default_geometry(sigma)
        core = c if core is None else core
        pad = p if pad is None else pad

    dy, dx = offset
    ys, xs = _edges(H, core, dy), _edges(W, core, dx)

    out = []
    for cy0, cy1 in zip(ys[:-1], ys[1:]):
        for cx0, cx1 in zip(xs[:-1], xs[1:]):
            out.append(Box(
                y0=max(0, cy0 - pad), x0=max(0, cx0 - pad),
                y1=min(H, cy1 + pad), x1=min(W, cx1 + pad),
                cy0=cy0, cx0=cx0, cy1=cy1, cx1=cx1,
            ))
    return out


def _edges(n, core, shift=0):
    """Core boundaries along one axis: `n` split into nearly equal pieces of
    about `core` px, so no tile is left a 1-px sliver by an unlucky remainder.
    `shift` displaces every interior boundary, shortening only the first piece.

    The shift has to be applied to the boundaries THEMSELVES, not to the range
    they are spread over. Re-running linspace from -shift with the same piece
    count merely re-spaces the same interval, and integer rounding then snaps
    the later boundaries back onto the unshifted ones. Measured on the 39x39
    bead frame with core=5, that is not a subtle loss -- it left y = 24, 29 and
    34 as seams in BOTH lattice phases:

        phase 0   0   5  10  15  20  24  29  34  39
        phase 1   0   3   8  13  18  24  29  34  39     <- last four identical

    so a pair straddling y = 34 was never fitted jointly by any box in any
    pass, no matter how many passes ran. That is exactly the failure `jitter`
    exists to prevent, and it showed up as the one strong interior defect left
    on the frame: two emitters at (33.30, 26.06) and (33.57, 24.62), the second
    0.43 px from the seam, with an over-modelled score of z = -12.5 beside an
    unexplained z = +27.1 across the boundary. Cropping the same region to 17
    rows made the rounding land elsewhere, the seam moved, and the pair came
    out clean -- which is why the region looked solvable in isolation and was
    not solvable in place.
    """
    k = max(1, int(round(n / float(core))))
    e = np.round(np.linspace(0, n, k + 1)).astype(int)
    s = int(shift) % core
    if s:
        e = np.unique(np.clip(np.concatenate([[0], e[:-1] + s, [n]]), 0, n))
        # A boundary leaving a piece shorter than half a core buys nothing and
        # costs a box; drop it rather than emit a sliver core.
        min_piece = max(2, core // 2)
        keep = [int(e[0])]
        for v in e[1:-1]:
            if v - keep[-1] >= min_piece and n - v >= min_piece:
                keep.append(int(v))
        keep.append(int(e[-1]))
        e = np.asarray(keep)
    return e
