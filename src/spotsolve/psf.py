"""Pixel-integrated Gaussian PSFs, for simulation and model images.

Each emitter contributes A * E(y; cy, sigma) * E(x; cx, sigma), where E
integrates a unit Gaussian over one pixel and A is total flux. Fixed-width
parameters are [background, A, y, x, ...]; per-emitter widths add sigma
after each emitter's x coordinate. The detector's own model and
derivatives are in Rust (`spotsolve_core::psf`).
"""

import math

import numpy as np
from scipy.special import erf

SQRT2 = math.sqrt(2.0)

__all__ = ["peak_factor", "model", "pack", "unpack",
           "model_var_sigma", "pack_var_sigma"]


def peak_factor(sigma):
    """Central-pixel signal per unit flux for scalar or array `sigma`.

    For an emitter centered on a pixel, `peak = flux * peak_factor(sigma)`.
    """
    return erf(0.5 / (np.asarray(sigma, float) * SQRT2)) ** 2


def _axes(yy, xx):
    """The 1-D pixel-center axes of a separable mgrid pair."""
    return np.asarray(yy)[:, 0].astype(float), np.asarray(xx)[0, :].astype(float)


def _shape(ax, c, sigma):
    """Pixel integrals for one axis, shape (len(ax), K)."""
    k = 1.0 / (sigma * SQRT2)
    return 0.5 * (erf((ax[:, None] - c[None, :] + 0.5) * k)
                  - erf((ax[:, None] - c[None, :] - 0.5) * k))


def model(theta, yy, xx, sigma):
    """Render `[b, A, y, x, ...]` at one width on the grids, shape (h, w)."""
    ay, ax = _axes(yy, xx)
    b, A, cy, cx = unpack(theta)
    if A.size == 0:
        return np.full((ay.size, ax.size), b)
    return b + np.einsum("k,ik,jk->ij", A, _shape(ay, cy, sigma), _shape(ax, cx, sigma))


def model_var_sigma(theta, yy, xx):
    """Render `[b, A, y, x, sigma, ...]`, one width per emitter."""
    ay, ax = _axes(yy, xx)
    theta = np.asarray(theta, dtype=float)
    rest = theta[1:].reshape(-1, 4)
    out = np.full((ay.size, ax.size), float(theta[0]))
    for a, y, x, s in rest:
        out += a * _shape(ay, np.array([y]), s)[:, 0][:, None] * _shape(ax, np.array([x]), s)[:, 0][None, :]
    return out


def pack(b, A, cy, cx):
    """Build `[b, A, y, x, ...]` from b (scalar) and per-emitter arrays."""
    A, cy, cx = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (A, cy, cx))
    return np.concatenate([[float(b)], np.stack([A, cy, cx], axis=-1).ravel()])


def pack_var_sigma(b, A, cy, cx, sigma):
    """Build `[b, A, y, x, sigma, ...]`, one width per emitter."""
    A, cy, cx, sigma = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (A, cy, cx, sigma))
    return np.concatenate([[float(b)], np.stack([A, cy, cx, sigma], axis=-1).ravel()])


def unpack(theta):
    """(b, A, cy, cx) of `[b, A, y, x, ...]` as plain arrays."""
    theta = np.asarray(theta, dtype=float)
    rest = theta[1:].reshape(-1, 3)
    return float(theta[0]), rest[:, 0], rest[:, 1], rest[:, 2]
