"""Flag bright detections and group nearby flags after localization."""

import numpy as np

__all__ = ["AGG_AMP_RATIO", "AGG_LINK", "AGGREGATE_FLAG_DTYPE",
           "flag_aggregates", "aggregate_report"]


AGG_AMP_RATIO = 20.0    # over-bright cut: flux / the frame's median detection
AGG_LINK = 3.0          # sigma; flagged detections within this are one object


AGGREGATE_FLAG_DTYPE = np.dtype([("y", float), ("x", float), ("flux", float),
                                 ("ratio", float), ("n", int)])


def flag_aggregates(result, ratio=AGG_AMP_RATIO, min_flux=None,
                    link=AGG_LINK):
    """Return (mask, objects) for detections above a flux threshold.

    The threshold is `ratio` times the frame's median flux, or `min_flux`
    when supplied. Flagged detections within `link * result.sigma` form
    single-linkage groups. Each object records its flux-weighted centroid,
    summed flux, brightest member's ratio to the median, and member count.

    This brightness flag does not distinguish aggregates from unresolved
    sources or other bright objects. It does not change the detector fit.
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
    # Nearby flags count as one object in frame summaries.
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
    """Summarize flagged objects, including their share of total detected flux."""
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
