"""Over-bright detections in a finished result: aggregates, flagged by flux.

Reads any result with `positions`, `amplitudes` and `sigma` -- what
`spotsolve.localize` returns -- and says something about it. Nothing here
runs inside the search.
"""

import numpy as np

__all__ = ["AGG_AMP_RATIO", "AGG_LINK", "AGGREGATE_FLAG_DTYPE",
           "flag_aggregates", "aggregate_report"]


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
    The other regime, sigma_fit well above the reporting band, the search
    itself returns in `rejects` as "too_wide".

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
    a 2x is almost certainly two of them, fitted as one that has absorbed
    both fluxes and reports a confident SE.

    The `ratio=20` default is set to catch aggregates. It will NOT find
    unresolved multiples, which sit at ~2x, inside the ordinary flux spread of
    most frames. On the bead data the question does not arise: measured
    2026-08-29, both frames are unimodal with max/median 1.34 and 1.58 and
    nothing above 2x, so there are no aggregates AND no unresolved multiples
    there, and this function correctly flags nothing on either.

    Why this is post hoc
    --------------------
    A PSF-width object is REPRESENTABLE by the model, so the search fits it
    as one bright emitter (plus a few small neighbours) and it stays
    identifiable afterwards. Nothing needs to be excluded before the search,
    and excluding it hurts: freezing a sigma~1.5 object into the background
    took the residual audit from z in [-7.0, 10.3] to [-26.6, 10.3] and raised
    N from 689 to 715, because the frozen object double-counts against
    emitters fitted beside it (measured under the retired `detect`).

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
