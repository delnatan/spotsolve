"""Matching and scoring of detected emitters against ground truth."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    precision: float
    recall: float
    f1: float
    n_true: int
    n_est: int
    n_matched: int
    rmse: float  # over matched pairs
    matched_true_idx: np.ndarray
    matched_est_idx: np.ndarray


def match(true_pos, est_pos, radius=1.0):
    """Hungarian matching at a fixed radius; unmatched pairs excluded via a
    high-cost cutoff (radius) so the assignment never links a true emitter
    to an estimate farther than `radius` even if it's the best available."""
    true_pos = np.atleast_2d(true_pos)
    est_pos = np.atleast_2d(est_pos)
    n_true = true_pos.shape[0] if true_pos.size else 0
    n_est = est_pos.shape[0] if est_pos.size else 0

    if n_true == 0 or n_est == 0:
        return MatchResult(
            precision=0.0 if n_est else 1.0,
            recall=0.0 if n_true else 1.0,
            f1=0.0 if (n_true or n_est) else 1.0,
            n_true=n_true,
            n_est=n_est,
            n_matched=0,
            rmse=np.nan,
            matched_true_idx=np.array([], dtype=int),
            matched_est_idx=np.array([], dtype=int),
        )

    d = np.linalg.norm(true_pos[:, None, :] - est_pos[None, :, :], axis=-1)
    cost = np.where(d <= radius, d, radius * 10.0 + d)  # discourage far links
    row, col = linear_sum_assignment(cost)
    keep = d[row, col] <= radius
    row, col = row[keep], col[keep]

    n_matched = row.shape[0]
    precision = n_matched / n_est
    recall = n_matched / n_true
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    rmse = float(np.sqrt(np.mean(d[row, col] ** 2))) if n_matched else np.nan

    return MatchResult(
        precision=precision,
        recall=recall,
        f1=f1,
        n_true=n_true,
        n_est=n_est,
        n_matched=n_matched,
        rmse=rmse,
        matched_true_idx=row,
        matched_est_idx=col,
    )
