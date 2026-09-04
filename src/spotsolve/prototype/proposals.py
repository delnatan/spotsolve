"""Cheap full-frame proposals for the calibrated local-model prototype.

The score is the standardized Poisson score for adding a non-negative,
fixed-width emitter at each pixel. It is a proposal statistic only: local
maxima are not detections and carry no final false-positive interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from scipy.special import erf


@dataclass(frozen=True)
class ProposalOptions:
    # Intentionally permissive: this threshold controls compute, not the
    # scientific false-positive rate. The full proposal search is calibrated
    # downstream as a maximum over the frame.
    score_threshold: float = 2.0
    nms_radius_ratio: float = 1.0
    dedup_radius_ratio: float = 0.75
    component_link_ratio: float = 3.0
    roi_pad_ratio: float = 4.0
    max_proposals: int = 64
    background_smooth_ratio: float = 2.0
    background_clip_sigma: float = 3.0
    background_iterations: int = 2


@dataclass(frozen=True)
class Proposal:
    centre: np.ndarray
    score: float
    flux_score: float
    pair_axis: float
    axis_anisotropy: float


@dataclass(frozen=True)
class ProposalComponent:
    proposal_indices: tuple[int, ...]
    bounds: tuple[int, int, int, int]


@dataclass(frozen=True)
class ProposalResult:
    proposals: tuple[Proposal, ...]
    components: tuple[ProposalComponent, ...]
    score_map: np.ndarray
    flux_score_map: np.ndarray
    background_map: np.ndarray
    local_maxima_above_threshold: int
    budget_exhausted: bool


@lru_cache(maxsize=64)
def _axis_kernel(sigma):
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    radius = max(2, int(np.ceil(4.0 * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=float)
    scale = sigma * np.sqrt(2.0)
    kernel = 0.5 * (
        erf((axis + 0.5) / scale) - erf((axis - 0.5) / scale))
    kernel /= kernel.sum()
    return kernel


def poisson_score_map(data, background, sigma):
    """Return standardized source-score and one-step flux-estimate maps."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or min(data.shape) < 3:
        raise ValueError("data must be a two-dimensional image at least 3x3")
    background = np.asarray(background, dtype=float)
    if background.ndim == 0:
        background = np.full(data.shape, float(background))
    if background.shape != data.shape:
        raise ValueError("background must be scalar or match data shape")
    if np.any(~np.isfinite(data)) or np.any(data < 0.0):
        raise ValueError("Poisson data must be finite and non-negative")
    if np.any(~np.isfinite(background)) or np.any(background <= 0.0):
        raise ValueError("background must be finite and strictly positive")

    kernel = _axis_kernel(float(sigma))
    residual_over_mean = (data - background) / background
    score = ndi.correlate1d(
        ndi.correlate1d(residual_over_mean, kernel, axis=0,
                        mode="constant", cval=0.0),
        kernel, axis=1, mode="constant", cval=0.0)
    inv_mean = 1.0 / background
    kernel2 = kernel * kernel
    information = ndi.correlate1d(
        ndi.correlate1d(inv_mean, kernel2, axis=0,
                        mode="constant", cval=0.0),
        kernel2, axis=1, mode="constant", cval=0.0)
    information = np.maximum(information, np.finfo(float).tiny)
    standardized = score / np.sqrt(information)
    flux = np.maximum(score / information, 0.0)
    return standardized, flux


def estimate_background(data, sigma, *, options=None):
    """Estimate a proposal-only smooth mean with iterative upper clipping.

    This is deliberately a cheap field estimate, not the scientific
    background model used for final selection. Positive compact residuals are
    clipped before repeated smoothing so bright emitters do not erase their
    own proposal score.
    """
    options = ProposalOptions() if options is None else options
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or min(data.shape) < 3:
        raise ValueError("data must be a two-dimensional image at least 3x3")
    if np.any(~np.isfinite(data)) or np.any(data < 0.0):
        raise ValueError("Poisson data must be finite and non-negative")
    smooth_sigma = options.background_smooth_ratio * float(sigma)
    if smooth_sigma <= 0.0 or options.background_iterations < 0:
        raise ValueError("background smoothing settings are invalid")
    estimate = ndi.gaussian_filter(data, smooth_sigma, mode="reflect")
    floor = max(float(np.percentile(data, 5.0)) * 0.1, 1e-3)
    estimate = np.maximum(estimate, floor)
    for _ in range(options.background_iterations):
        ceiling = (estimate + options.background_clip_sigma
                   * np.sqrt(np.maximum(estimate, floor)))
        clipped = np.minimum(data, ceiling)
        estimate = np.maximum(
            ndi.gaussian_filter(clipped, smooth_sigma, mode="reflect"), floor)
    return estimate


def _subpixel_peak(values, y, x):
    h, w = values.shape
    if y == 0 or x == 0 or y == h - 1 or x == w - 1:
        return np.array([float(y), float(x)])

    def offset(left, middle, right):
        denominator = left - 2.0 * middle + right
        if denominator >= -1e-12:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))

    return np.array([
        y + offset(values[y - 1, x], values[y, x], values[y + 1, x]),
        x + offset(values[y, x - 1], values[y, x], values[y, x + 1]),
    ])


def _pair_axis(values, y, x):
    h, w = values.shape
    if y == 0 or x == 0 or y == h - 1 or x == w - 1:
        return 0.0, 0.0
    centre = values[y, x]
    hyy = values[y - 1, x] - 2.0 * centre + values[y + 1, x]
    hxx = values[y, x - 1] - 2.0 * centre + values[y, x + 1]
    hyx = 0.25 * (
        values[y + 1, x + 1] - values[y + 1, x - 1]
        - values[y - 1, x + 1] + values[y - 1, x - 1])
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.array([[hyy, hyx], [hyx, hxx]], dtype=float))
    # The broad/weak-curvature direction of an elongated maximum is the pair
    # separation axis. For a round maximum its orientation is intentionally
    # marked as uninformative through a near-zero anisotropy.
    index = int(np.argmax(eigenvalues))
    vector = eigenvectors[:, index]
    magnitude = np.abs(eigenvalues)
    anisotropy = float(
        abs(magnitude[1] - magnitude[0]) / max(magnitude.sum(), 1e-12))
    angle = float(np.mod(np.arctan2(vector[0], vector[1]), np.pi))
    return angle, anisotropy


def _deduplicate(points, scores, radius, limit):
    order = np.argsort(-scores, kind="stable")
    kept = []
    for index in order:
        if all(np.linalg.norm(points[index] - points[old]) > radius
               for old in kept):
            kept.append(int(index))
        if len(kept) >= limit:
            break
    return kept


def _components(proposals, shape, sigma, options):
    count = len(proposals)
    if count == 0:
        return ()
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    centres = np.stack([proposal.centre for proposal in proposals])
    link = options.component_link_ratio * sigma
    for left in range(count):
        for right in range(left + 1, count):
            if np.linalg.norm(centres[left] - centres[right]) <= link:
                union(left, right)
    groups = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    pad = options.roi_pad_ratio * sigma
    result = []
    for indices in groups.values():
        points = centres[indices]
        y0 = max(0, int(np.floor(points[:, 0].min() - pad)))
        x0 = max(0, int(np.floor(points[:, 1].min() - pad)))
        y1 = min(shape[0], int(np.ceil(points[:, 0].max() + pad)) + 1)
        x1 = min(shape[1], int(np.ceil(points[:, 1].max() + pad)) + 1)
        result.append(ProposalComponent(tuple(indices), (y0, x0, y1, x1)))
    return tuple(result)


def generate_proposals(data, sigma, *, background=None, options=None):
    """Generate ranked, deduplicated proposal components for one frame."""
    options = ProposalOptions() if options is None else options
    if options.max_proposals < 1:
        raise ValueError("max_proposals must be positive")
    if background is None:
        background = estimate_background(data, sigma, options=options)
    else:
        background = np.asarray(background, dtype=float)
        if background.ndim == 0:
            background = np.full(np.asarray(data).shape, float(background))
    score, flux = poisson_score_map(data, background, sigma)
    radius = max(1, int(np.ceil(options.nms_radius_ratio * sigma)))
    size = 2 * radius + 1
    maxima = ((score == ndi.maximum_filter(score, size=size, mode="constant",
                                            cval=-np.inf))
              & (score >= options.score_threshold))
    ys, xs = np.nonzero(maxima)
    raw_count = len(ys)
    if raw_count == 0:
        return ProposalResult((), (), score, flux, background, 0, False)

    integer_points = np.stack([ys, xs], axis=1).astype(float)
    strengths = score[ys, xs]
    keep = _deduplicate(
        integer_points, strengths,
        options.dedup_radius_ratio * sigma, options.max_proposals)
    proposals = []
    for index in keep:
        y, x = int(ys[index]), int(xs[index])
        axis, anisotropy = _pair_axis(score, y, x)
        proposals.append(Proposal(
            centre=_subpixel_peak(score, y, x),
            score=float(score[y, x]),
            flux_score=float(flux[y, x]),
            pair_axis=axis,
            axis_anisotropy=anisotropy,
        ))
    proposals = tuple(proposals)
    return ProposalResult(
        proposals=proposals,
        components=_components(proposals, np.asarray(data).shape, sigma, options),
        score_map=score,
        flux_score_map=flux,
        background_map=background,
        local_maxima_above_threshold=raw_count,
        budget_exhausted=raw_count > options.max_proposals,
    )
