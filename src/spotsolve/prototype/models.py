"""Mean models and analytic Jacobians for local hypothesis competition.

The source hypotheses use a positive log-plane. Its two slope coordinates describe
the log-background change across a half ROI, which keeps their numerical scale
near one and guarantees a positive Poisson mean without coupled constraints.
For the small slopes expected inside a PSF-sized ROI this is locally the same
as an additive plane to first order. Hsmooth is a separate positive
log-quadratic nuisance hypothesis for low-frequency curvature.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .. import psf
from .parameterize import pair_components


class Hypothesis(str, Enum):
    H0 = "background"
    HSMOOTH = "smooth"
    H1 = "focus"
    H2 = "pair"
    HWIDE = "wide"


@dataclass(frozen=True)
class ROIGrid:
    """Pixel centers and normalized coordinates for one local ROI."""

    yy: np.ndarray
    xx: np.ndarray
    uy: np.ndarray
    ux: np.ndarray

    @classmethod
    def from_shape(cls, shape):
        h, w = map(int, shape)
        if h < 3 or w < 3:
            raise ValueError("an ROI must be at least 3 by 3 pixels")
        yy, xx = np.mgrid[0:h, 0:w].astype(float)
        cy, cx = (h - 1.0) / 2.0, (w - 1.0) / 2.0
        sy, sx = max(cy, 1.0), max(cx, 1.0)
        return cls(yy=yy, xx=xx, uy=(yy - cy) / sy, ux=(xx - cx) / sx)

    @property
    def shape(self):
        return self.yy.shape


def _background(theta, grid):
    z = theta[0] + theta[1] * grid.uy + theta[2] * grid.ux
    background = np.exp(z)
    jac = np.stack([background, background * grid.uy, background * grid.ux], axis=-1)
    return background, jac


def _smooth_background(theta, grid):
    basis = np.stack([
        np.ones(grid.shape), grid.uy, grid.ux,
        grid.uy * grid.uy, grid.uy * grid.ux, grid.ux * grid.ux,
    ], axis=-1)
    background = np.exp(np.einsum("ijk,k->ij", basis, theta))
    return background, background[:, :, None] * basis


def _unit_fixed(grid, y, x, sigma):
    packed = psf.pack(0.0, [1.0], [y], [x])
    model, jac = psf.model_and_jac(packed, grid.yy, grid.xx, sigma)
    return model, jac[:, :, 2], jac[:, :, 3]


def _unit_wide(grid, y, x, sigma):
    packed = psf.pack_var_sigma(0.0, [1.0], [y], [x], [sigma])
    jac = psf.jac_var_sigma(packed, grid.yy, grid.xx)
    return jac[:, :, 1], jac[:, :, 2], jac[:, :, 3], jac[:, :, 4]


def evaluate(hypothesis, theta, grid, sigma):
    """Return ``(mean, dmean/dtheta)`` for one hypothesis.

    Layouts are:

    - H0: ``log_b, gy, gx``
    - Hsmooth: ``log_b, gy, gx, cyy, cyx, cxx``
    - H1: H0 + ``log_A, y, x``
    - H2: H0 + ``log_F, cy, cx, separation, angle, q``
    - Hwide: H0 + ``log_A, y, x, log(width/focus_width)``
    """
    hypothesis = Hypothesis(hypothesis)
    theta = np.asarray(theta, dtype=float)
    expected = {Hypothesis.H0: 3, Hypothesis.HSMOOTH: 6, Hypothesis.H1: 6,
                Hypothesis.H2: 9, Hypothesis.HWIDE: 7}[hypothesis]
    if theta.shape != (expected,):
        raise ValueError(f"{hypothesis.value} theta must have {expected} entries")

    if hypothesis is Hypothesis.HSMOOTH:
        return _smooth_background(theta, grid)

    background, jb = _background(theta, grid)
    if hypothesis is Hypothesis.H0:
        return background, jb

    if hypothesis is Hypothesis.H1:
        amplitude = float(np.exp(theta[3]))
        g, gy, gx = _unit_fixed(grid, theta[4], theta[5], sigma)
        mean = background + amplitude * g
        jac = np.empty(grid.shape + (6,), dtype=float)
        jac[:, :, :3] = jb
        jac[:, :, 3] = amplitude * g
        jac[:, :, 4] = amplitude * gy
        jac[:, :, 5] = amplitude * gx
        return mean, jac

    if hypothesis is Hypothesis.HWIDE:
        amplitude = float(np.exp(theta[3]))
        width = float(sigma * np.exp(theta[6]))
        g, gy, gx, gs = _unit_wide(grid, theta[4], theta[5], width)
        mean = background + amplitude * g
        jac = np.empty(grid.shape + (7,), dtype=float)
        jac[:, :, :3] = jb
        jac[:, :, 3] = amplitude * g
        jac[:, :, 4] = amplitude * gy
        jac[:, :, 5] = amplitude * gx
        jac[:, :, 6] = amplitude * gs * width
        return mean, jac

    fluxes, positions = pair_components(theta)
    packed = psf.pack(0.0, [1.0, 1.0], positions[:, 0], positions[:, 1])
    _, jp = psf.model_and_jac(packed, grid.yy, grid.xx, sigma)
    g0, gy0, gx0 = jp[:, :, 1], jp[:, :, 2], jp[:, :, 3]
    g1, gy1, gx1 = jp[:, :, 4], jp[:, :, 5], jp[:, :, 6]
    f0, f1 = fluxes
    q = float(theta[8])
    separation, angle = theta[6:8]
    uy, ux = np.sin(angle), np.cos(angle)
    vy, vx = separation * uy, separation * ux
    day, dax = separation * ux, -separation * uy

    source = f0 * g0 + f1 * g1
    mean = background + source
    jac = np.empty(grid.shape + (9,), dtype=float)
    jac[:, :, :3] = jb
    jac[:, :, 3] = source
    jac[:, :, 4] = f0 * gy0 + f1 * gy1
    jac[:, :, 5] = f0 * gx0 + f1 * gx1
    directional_y = f0 * (1.0 - q) * gy0 - f1 * q * gy1
    directional_x = f0 * (1.0 - q) * gx0 - f1 * q * gx1
    jac[:, :, 6] = directional_y * uy + directional_x * ux
    jac[:, :, 7] = directional_y * day + directional_x * dax
    jac[:, :, 8] = (
        (f0 + f1) * (g0 - g1)
        - vy * (f0 * gy0 + f1 * gy1)
        - vx * (f0 * gx0 + f1 * gx1)
    )
    return mean, jac


def decode(hypothesis, theta, sigma):
    """Convert a fitted vector into common physical quantities."""
    hypothesis = Hypothesis(hypothesis)
    theta = np.asarray(theta, dtype=float)
    result = {
        "background_center": float(np.exp(theta[0])),
        "background_log_slopes": theta[1:3].copy(),
    }
    if hypothesis in (Hypothesis.H0, Hypothesis.HSMOOTH):
        result.update(fluxes=np.empty(0), positions=np.empty((0, 2)))
    elif hypothesis is Hypothesis.H2:
        fluxes, positions = pair_components(theta)
        result.update(fluxes=fluxes, positions=positions,
                      separation=float(theta[6]),
                      angle=float(np.mod(theta[7], 2.0 * np.pi)),
                      flux_fraction=float(theta[8]))
    else:
        result.update(fluxes=np.array([np.exp(theta[3])]),
                      positions=theta[None, 4:6].copy())
        if hypothesis is Hypothesis.HWIDE:
            result["width"] = float(sigma * np.exp(theta[6]))
    return result
