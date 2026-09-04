"""Truth-based metrics that preserve focused versus nuisance semantics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_positions(estimated, truth, max_distance):
    """Return matched indices and distances under a one-to-one assignment."""
    estimated = np.asarray(estimated, dtype=float).reshape(-1, 2)
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    if len(estimated) == 0 or len(truth) == 0:
        return (np.empty(0, dtype=int), np.empty(0, dtype=int),
                np.empty(0, dtype=float))
    distance = np.linalg.norm(estimated[:, None, :] - truth[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distance)
    keep = distance[rows, cols] <= float(max_distance)
    return rows[keep], cols[keep], distance[rows[keep], cols[keep]]


def focused_metrics(estimated, scenario, max_distance=None):
    """Score only reported focused emitters against focused truth.

    A localization on a wide source or haze remains a false focused emitter;
    proximity to real nuisance light never promotes it to a true positive.
    """
    estimated = np.asarray(estimated, dtype=float).reshape(-1, 2)
    truth = scenario.focused_positions
    if max_distance is None:
        max_distance = scenario.sigma
    rows, cols, distances = match_positions(estimated, truth, max_distance)
    tp = int(len(rows))
    fp = int(len(estimated) - tp)
    fn = int(len(truth) - tp)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "exact_count": bool(len(estimated) == len(truth) and tp == len(truth)),
        "rmse": (float(np.sqrt(np.mean(distances ** 2)))
                 if len(distances) else np.nan),
        "matched_distances": distances,
    }


def summarize(records):
    """Aggregate dictionaries returned by :func:`focused_metrics`."""
    records = list(records)
    if not records:
        raise ValueError("at least one metric record is required")
    distances = [r["matched_distances"] for r in records
                 if len(r["matched_distances"])]
    joined = np.concatenate(distances) if distances else np.empty(0)
    return {
        "frames": len(records),
        "focused_false_emitters_per_frame": float(np.mean(
            [r["false_positive"] for r in records])),
        "exact_count_rate": float(np.mean([r["exact_count"] for r in records])),
        "recall": float(sum(r["true_positive"] for r in records)
                        / max(sum(r["true_positive"] + r["false_negative"]
                                  for r in records), 1)),
        "localization_rmse": (float(np.sqrt(np.mean(joined ** 2)))
                              if len(joined) else np.nan),
    }
