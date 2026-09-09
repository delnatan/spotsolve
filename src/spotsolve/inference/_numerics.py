"""Shared numerical primitives; no detector or experimental-model imports."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ROIGrid:
    """Pixel centers for one local ROI."""

    yy: np.ndarray
    xx: np.ndarray

    @classmethod
    def from_shape(cls, shape):
        h, w = map(int, shape)
        if h < 3 or w < 3:
            raise ValueError("an ROI must be at least 3 by 3 pixels")
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        return cls(yy=yy, xx=xx)

    @property
    def shape(self):
        return self.yy.shape


def _poisson_objective(data, mean):
    positive = data > 0
    safe_data = np.where(positive, data, 1.0)
    term = np.where(positive, safe_data * np.log(safe_data / mean), 0.0)
    return float(np.sum(term - (data - mean)))


def _weighted_geometry(data, background):
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    weight = np.maximum(np.asarray(data, dtype=float) - background, 0.0)
    total = float(weight.sum())
    if total <= 1e-12:
        y, x = np.unravel_index(int(np.argmax(data)), data.shape)
        return np.array([float(y), float(x)]), 0.0, 1.0
    centre = np.array([(weight * yy).sum() / total,
                       (weight * xx).sum() / total])
    dy, dx = yy - centre[0], xx - centre[1]
    covariance = np.array([
        [(weight * dy * dy).sum(), (weight * dy * dx).sum()],
        [(weight * dy * dx).sum(), (weight * dx * dx).sum()],
    ]) / total
    values, vectors = np.linalg.eigh(covariance)
    vector = vectors[:, int(np.argmax(values))]
    angle = float(np.mod(np.arctan2(vector[0], vector[1]), np.pi))
    return centre, angle, total


def _projected_gradient(theta, gradient, lower, upper):
    gradient = np.array(gradient, copy=True)
    gradient[(theta <= lower + 1e-8) & (gradient > 0)] = 0
    gradient[(theta >= upper - 1e-8) & (gradient < 0)] = 0
    return float(np.max(np.abs(gradient)))
