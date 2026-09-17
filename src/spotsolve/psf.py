"""Pixel-integrated Gaussian PSFs and analytic derivatives.

Each emitter contributes A * E(y; cy, sigma) * E(x; cx, sigma), where E
integrates a unit Gaussian over one pixel and A is total flux. Fixed-width
parameters are [background, A, y, x, ...]; variable-width parameters add
sigma after each emitter's x coordinate.

Models use separable pixel-center grids. The `_ax` variants accept their
1-D axes directly; `halo` adds fixed neighboring light or background shape.
"""

import math

import numpy as np
from scipy.special import erf

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
SQRTPI = math.sqrt(math.pi)

__all__ = ["peak_factor", "axes", "model", "model_ax", "jac",
           "model_and_jac", "model_and_jac_ax",
           "model_free_sigma", "jac_free_sigma",
           "pack_var_sigma", "unpack_var_sigma",
           "model_var_sigma", "jac_var_sigma", "pack", "unpack"]


def peak_factor(sigma):
    """Central-pixel signal per unit flux for scalar or array `sigma`.

    For an emitter centered on a pixel, `peak = flux * peak_factor(sigma)`.
    """
    return erf(0.5 / (np.asarray(sigma, float) * SQRT2)) ** 2


def axes(yy, xx):
    """Extract the 1-D pixel-center axes from a separable mgrid pair."""
    yy = np.asarray(yy)
    xx = np.asarray(xx)
    return yy[:, 0].astype(float), xx[0, :].astype(float)


_axes = axes


def _unpack(theta):
    theta = np.asarray(theta, dtype=float)
    b = float(theta[0])
    rest = theta[1:].reshape(-1, 3)
    return b, rest[:, 0], rest[:, 1], rest[:, 2]


def _shape(ax, c, sigma):
    """Pixel integrals for one axis, shape (len(ax), K)."""
    k = 1.0 / (sigma * SQRT2)
    return 0.5 * (erf((ax[:, None] - c[None, :] + 0.5) * k)
                  - erf((ax[:, None] - c[None, :] - 0.5) * k))


def _factors(ax, c, sigma):
    """Return (E, dE_dc), each shaped (len(ax), K).

    Width derivatives are computed separately by `_factors_sigma`.
    """
    k = 1.0 / (sigma * SQRT2)
    up = (ax[:, None] - c[None, :] + 0.5) * k
    um = (ax[:, None] - c[None, :] - 0.5) * k
    E = 0.5 * (erf(up) - erf(um))
    dE_dc = -(np.exp(-up * up) - np.exp(-um * um)) / (sigma * SQRT2PI)
    return E, dE_dc


def _factors_sigma(ax, c, sigma):
    """(E, dE_dc, dE_dsigma) for one axis; the free-sigma path only."""
    k = 1.0 / (sigma * SQRT2)
    up = (ax[:, None] - c[None, :] + 0.5) * k
    um = (ax[:, None] - c[None, :] - 0.5) * k
    ep, em = np.exp(-up * up), np.exp(-um * um)
    E = 0.5 * (erf(up) - erf(um))
    dE_dc = -(ep - em) / (sigma * SQRT2PI)
    dE_dsig = (um * em - up * ep) / (sigma * SQRTPI)
    return E, dE_dc, dE_dsig


def model_ax(theta, ay, ax, sigma, halo=0.0):
    """`model` on pre-extracted 1-D axes (see `axes`)."""
    b, A, cy, cx = _unpack(theta)
    if A.size == 0:
        return np.full((ay.size, ax.size), b) + halo
    Ey = _shape(ay, cy, sigma)
    Ex = _shape(ax, cx, sigma)
    return b + np.einsum("k,ik,jk->ij", A, Ey, Ex) + halo


def model(theta, yy, xx, sigma, halo=0.0):
    """Render the model image, shape (h, w)."""
    ay, ax = axes(yy, xx)
    return model_ax(theta, ay, ax, sigma, halo)


def jac(theta, yy, xx, sigma, halo=0.0):
    """d(model)/d(theta), shape (h, w, 3K+1)."""
    ay, ax = _axes(yy, xx)
    b, A, cy, cx = _unpack(theta)
    h, w, K = ay.size, ax.size, A.size
    J = np.empty((h, w, 3 * K + 1))
    J[:, :, 0] = 1.0
    if K == 0:
        return J
    Ey, dEy = _factors(ay, cy, sigma)
    Ex, dEx = _factors(ax, cx, sigma)
    J[:, :, 1::3] = Ey[:, None, :] * Ex[None, :, :]
    J[:, :, 2::3] = A * dEy[:, None, :] * Ex[None, :, :]
    J[:, :, 3::3] = A * Ey[:, None, :] * dEx[None, :, :]
    return J


def model_and_jac_ax(theta, ay, ax, sigma, halo=0.0):
    """`model_and_jac` on pre-extracted 1-D axes (see `axes`).

    This is the optimizer's inner loop. Take the axes once per fit and call
    this, rather than re-deriving them from the 2-D grids on every one of the
    ~160 evaluations a single fit makes.
    """
    b, A, cy, cx = _unpack(theta)
    h, w, K = ay.size, ax.size, A.size
    J = np.empty((h, w, 3 * K + 1))
    J[:, :, 0] = 1.0
    if K == 0:
        return np.full((h, w), b) + halo, J
    Ey, dEy = _factors(ay, cy, sigma)
    Ex, dEx = _factors(ax, cx, sigma)
    EyEx = Ey[:, None, :] * Ex[None, :, :]
    J[:, :, 1::3] = EyEx
    J[:, :, 2::3] = A * dEy[:, None, :] * Ex[None, :, :]
    J[:, :, 3::3] = A * Ey[:, None, :] * dEx[None, :, :]
    return b + EyEx @ A + halo, J


def model_and_jac(theta, yy, xx, sigma, halo=0.0):
    """(model, jacobian) in one pass.

    The optimizer always needs both at the same theta, and the erf/exp
    factors are the whole cost of either. Computing them separately
    evaluates `_factors` four times per LM iteration instead of two.
    """
    ay, ax = axes(yy, xx)
    return model_and_jac_ax(theta, ay, ax, sigma, halo)


def model_free_sigma(theta, yy, xx, halo=0.0):
    """Same as `model`, but sigma is the last entry of theta."""
    return model(theta[:-1], yy, xx, float(np.asarray(theta)[-1]), halo=halo)


def jac_free_sigma(theta, yy, xx, halo=0.0):
    """d(model)/d(theta) with sigma as the last parameter, shape (h,w,3K+2)."""
    theta = np.asarray(theta, dtype=float)
    sigma = float(theta[-1])
    ay, ax = axes(yy, xx)
    b, A, cy, cx = _unpack(theta[:-1])
    h, w, K = ay.size, ax.size, A.size
    J = np.empty((h, w, 3 * K + 2))
    J[:, :, :-1] = jac(theta[:-1], yy, xx, sigma)
    if K == 0:
        J[:, :, -1] = 0.0
        return J
    Ey, _, dEy_s = _factors_sigma(ay, cy, sigma)
    Ex, _, dEx_s = _factors_sigma(ax, cx, sigma)
    J[:, :, -1] = np.einsum(
        "k,ik,jk->ij", A, dEy_s, Ex
    ) + np.einsum("k,ik,jk->ij", A, Ey, dEx_s)
    return J


def pack_var_sigma(b, A, cy, cx, sigma):
    """Build theta with one sigma per emitter.

    Layout is [b, A0, y0, x0, sigma0, A1, y1, x1, sigma1, ...]. This is used by
    post-hoc diagnostics, not by the fixed-sigma detector.
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    cy = np.atleast_1d(np.asarray(cy, dtype=float))
    cx = np.atleast_1d(np.asarray(cx, dtype=float))
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    return np.concatenate([[float(b)], np.stack([A, cy, cx, sigma], axis=-1).ravel()])


def unpack_var_sigma(theta):
    """(b, A, cy, cx, sigma) for per-emitter-sigma theta."""
    theta = np.asarray(theta, dtype=float)
    rest = theta[1:].reshape(-1, 4)
    return float(theta[0]), rest[:, 0], rest[:, 1], rest[:, 2], rest[:, 3]


def model_var_sigma(theta, yy, xx, halo=0.0):
    """Render a model with one sigma per emitter."""
    ay, ax = axes(yy, xx)
    b, A, cy, cx, sigma = unpack_var_sigma(theta)
    out = np.full((ay.size, ax.size), b) + halo
    for a, y, x, s in zip(A, cy, cx, sigma):
        Ey = _shape(ay, np.array([y]), float(s))[:, 0]
        Ex = _shape(ax, np.array([x]), float(s))[:, 0]
        out += a * Ey[:, None] * Ex[None, :]
    return out


def jac_var_sigma(theta, yy, xx, halo=0.0):
    """d(model)/d(theta) for one sigma per emitter, shape (h,w,4K+1)."""
    ay, ax = axes(yy, xx)
    _, A, cy, cx, sigma = unpack_var_sigma(theta)
    h, w, K = ay.size, ax.size, A.size
    J = np.empty((h, w, 4 * K + 1))
    J[:, :, 0] = 1.0
    for k, (a, y, x, s) in enumerate(zip(A, cy, cx, sigma)):
        Ey, dEy, dEy_s = _factors_sigma(ay, np.array([y]), float(s))
        Ex, dEx, dEx_s = _factors_sigma(ax, np.array([x]), float(s))
        Ey = Ey[:, 0]
        dEy = dEy[:, 0]
        dEy_s = dEy_s[:, 0]
        Ex = Ex[:, 0]
        dEx = dEx[:, 0]
        dEx_s = dEx_s[:, 0]
        j0 = 1 + 4 * k
        J[:, :, j0] = Ey[:, None] * Ex[None, :]
        J[:, :, j0 + 1] = a * dEy[:, None] * Ex[None, :]
        J[:, :, j0 + 2] = a * Ey[:, None] * dEx[None, :]
        J[:, :, j0 + 3] = (
            a * dEy_s[:, None] * Ex[None, :]
            + a * Ey[:, None] * dEx_s[None, :]
        )
    return J


def pack(b, A, cy, cx):
    """Build a flat theta from b (scalar) and per-emitter arrays."""
    A = np.atleast_1d(np.asarray(A, dtype=float))
    cy = np.atleast_1d(np.asarray(cy, dtype=float))
    cx = np.atleast_1d(np.asarray(cx, dtype=float))
    return np.concatenate([[float(b)], np.stack([A, cy, cx], axis=-1).ravel()])


def unpack(theta):
    """(b, A, cy, cx) as plain arrays."""
    return _unpack(theta)
