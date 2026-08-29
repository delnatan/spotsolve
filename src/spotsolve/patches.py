"""Decompose a set of emitters into small, jointly-fittable patches.

Emitters whose ~sigma-scale supports touch are grouped into one patch and
fit jointly. Emitters just outside a patch (within a further halo radius)
are held fixed and folded into the model as a constant contribution, so
flux is not double-counted at patch borders and the joint Hessian stays
small (bounded by K_max).

Choosing halo_radius_factor
---------------------------
An emitter that is neither free in a patch nor in its frozen halo is, from
that patch's point of view, not in the model at all -- and the patch's free
background `b` is the only parameter that can absorb it. So the halo radius
has to be large enough that what it excludes is genuinely negligible against
the BACKGROUND, not merely small against a bead peak.

Measured on random fields with bead-matched brightness (peaks 94-198 e-,
background 4 e-), worst-case leak into a patch from emitters outside its
halo:

    halo_radius_factor   max leak   p99 leak   frozen emitters per patch
            3.0            3.10       2.40              2.8
            4.0            0.155      0.086             3.7
            5.0            0.001      0.001             4.6

At the old 3.0 a patch could be handed an unmodelled pedestal of 3.1 e- on
a 4 e- background. That is not a small perturbation: with the halo removed
entirely the same patch fit drives `b` to 57 e- and its precision and
recall fall from 1.00/0.75 to 0.62/0.62. 5.0 buys the margin for about two
extra frozen emitters per patch, and frozen emitters cost only one rendered
constant each -- they do not enter the Hessian. It also keeps the halo at
least as wide as the 4-sigma truncation `detect.render_model` uses, so the
halo never omits flux the global model includes.
"""

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import psf
from .structs import Patch

__all__ = ["build_patches", "build_halo_image", "patch_grids"]


def _bbox_for(ys, xs, pad, img_shape):
    y0 = max(0, int(np.floor(ys.min() - pad)))
    x0 = max(0, int(np.floor(xs.min() - pad)))
    y1 = min(img_shape[0], int(np.ceil(ys.max() + pad)) + 1)
    x1 = min(img_shape[1], int(np.ceil(xs.max() + pad)) + 1)
    return y0, x0, y1, x1


def build_patches(
    positions,
    sigma,
    img_shape,
    link_radius_factor=2.5,
    halo_radius_factor=5.0,
    bbox_pad_factor=3.0,
    k_max=12,
):
    """positions: (N,2) array of (y,x). Returns a list of Patch."""
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    if n == 0:
        return []

    tree = cKDTree(positions)
    link_r = link_radius_factor * sigma
    pairs = tree.query_pairs(r=link_r, output_type="ndarray")

    if pairs.shape[0] == 0:
        adj = np.zeros((n, n), dtype=bool)
    else:
        adj = np.zeros((n, n), dtype=bool)
        adj[pairs[:, 0], pairs[:, 1]] = True
        adj[pairs[:, 1], pairs[:, 0]] = True

    n_comp, labels = connected_components(adj, directed=False)

    patches = []
    halo_r = halo_radius_factor * sigma
    bbox_pad = bbox_pad_factor * sigma

    for c in range(n_comp):
        idx = np.nonzero(labels == c)[0]
        if idx.shape[0] > k_max:
            for sub in _split_component(idx, positions, k_max):
                patches.append(
                    _finalize_patch(sub, positions, tree, halo_r, bbox_pad, img_shape)
                )
        else:
            patches.append(
                _finalize_patch(idx, positions, tree, halo_r, bbox_pad, img_shape)
            )
    return patches


def _finalize_patch(idx, positions, tree, halo_r, bbox_pad, img_shape):
    ys, xs = positions[idx, 0], positions[idx, 1]
    y0, x0, y1, x1 = _bbox_for(ys, xs, bbox_pad, img_shape)

    # frozen halo: emitters not in idx, within halo_r of the bbox
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    diag = np.hypot(y1 - y0, x1 - x0) / 2.0
    cand = tree.query_ball_point([cy, cx], diag + halo_r)
    cand = np.array([i for i in cand if i not in set(idx.tolist())], dtype=int)
    if cand.size:
        # keep only those actually within halo_r of the bbox rectangle
        py = np.clip(positions[cand, 0], y0, y1 - 1)
        px = np.clip(positions[cand, 1], x0, x1 - 1)
        d = np.hypot(positions[cand, 0] - py, positions[cand, 1] - px)
        cand = cand[d <= halo_r]

    return Patch(indices=idx, frozen_indices=cand, y0=y0, x0=x0, y1=y1, x1=x1)


def _split_component(idx, positions, k_max):
    """Recursively bisect an oversized component by a spatial median split
    (along its longer axis) until every piece is <= k_max. This is a proxy
    for "cut the weakest graph edge": splitting along the axis of greatest
    spread tends to cut through the sparsest part of a spatially clustered
    component."""
    if idx.shape[0] <= k_max:
        return [idx]
    pts = positions[idx]
    spread = pts.max(axis=0) - pts.min(axis=0)
    axis = int(np.argmax(spread))
    order = idx[np.argsort(pts[:, axis])]
    mid = order.shape[0] // 2
    left, right = order[:mid], order[mid:]
    return _split_component(left, positions, k_max) + _split_component(
        right, positions, k_max
    )


def build_halo_image(positions, amplitudes, frozen_indices, sigma, yy, xx, y0, x0):
    """Render the constant (parameter-free) contribution from frozen
    (out-of-patch) emitters. The patch's own background `b` is a free
    parameter handled separately in the patch's theta vector.

    `yy, xx` are the patch's LOCAL grids (see patch_grids); `positions` are
    global, so they are converted to local coordinates here via (y0, x0)
    -- the same origin passed to patch_grids for this patch.
    """
    if frozen_indices.size == 0:
        return 0.0
    A = np.asarray(amplitudes)[frozen_indices]
    pos = np.asarray(positions)[frozen_indices]
    theta = psf.pack(0.0, A, pos[:, 0] - y0, pos[:, 1] - x0)
    return psf.model(theta, yy, xx, sigma)


def patch_grids(patch):
    """Local (0-based) pixel-center coordinate grids for this patch, i.e.
    grid[0,0] corresponds to global pixel (patch.y0, patch.x0) -- matching
    both `data_img[patch.y0:patch.y1, patch.x0:patch.x1]` (sub-image
    extraction) and how theta positions are built/stored (local coords,
    with patch.y0/x0 added back only when writing into the global
    positions array). Do not switch this to global coordinates without
    also removing every local<->global offset elsewhere."""
    h, w = patch.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return yy * 1.0, xx * 1.0
