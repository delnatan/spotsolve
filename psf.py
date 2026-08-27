"""Pixel-integrated 2D Gaussian PSF model, with analytic derivatives.

Model
-----
    m[i,j] = b + sum_k A_k * ey_k[i] * ex_k[j]
    ey_k[i] = 0.5*(erf((i - cy_k + 0.5)/(sigma*sqrt2))
                 - erf((i - cy_k - 0.5)/(sigma*sqrt2)))

A_k is TOTAL FLUX: the pixel-integrated Gaussian sums to A over all pixels,
so a peak-height guess must be divided by `peak_factor(sigma)` to become an
A. See structs.py for the full theta layout.

Why numpy and not JAX
---------------------
This was originally jax.jit + jax.jacfwd. On the array sizes this pipeline
actually uses -- patches of ~81 to ~200 pixels -- JAX's tracing and dispatch
cost dominates the arithmetic completely, and the model search changes the
parameter-vector length on every proposal, so each new (3K+1, h, w)
combination triggers a fresh compile costing ~40 ms against ~0.02 ms for a
warm call. Analytic derivatives in numpy have no compile step, no shape
specialization, and no dispatch overhead.

The derivatives are elementary. With u_pm = (i - cy_k +- 0.5)/(sigma*sqrt2),

    d(ey)/d(cy)    = -(1/(sigma*sqrt(2*pi))) * (exp(-u_+^2) - exp(-u_-^2))
    d(ey)/d(sigma) =  (1/(sigma*sqrt(pi)))   * (u_- * exp(-u_-^2)
                                              - u_+ * exp(-u_+^2))

and the model separates, so ey (h x K) and ex (w x K) are built as 1-D
factors and combined by outer products rather than evaluated on the full
2-D grid per emitter.

Grid convention
---------------
`yy, xx` are the 2-D pixel-centre grids from numpy/jax mgrid. Only their
separable axes are used (yy[:,0] and xx[0,:]), which is what mgrid provides;
do not pass non-separable coordinate arrays.
"""

import math

import numpy as np
from scipy.special import erf

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
SQRTPI = math.sqrt(math.pi)

__all__ = ["peak_factor", "axes", "model", "model_ax", "jac",
           "model_and_jac", "model_and_jac_ax",
           "model_free_sigma", "jac_free_sigma", "pack", "unpack"]


def peak_factor(sigma):
    """Ratio of an on-pixel-centre peak height to the amplitude parameter A:
    peak = A * peak_factor(sigma). Roughly 0.104 at sigma=1.2, so an observed
    peak-minus-background must be divided by ~0.104 (multiplied by ~9.6) to
    become an initial guess for A."""
    return math.erf(0.5 / (sigma * SQRT2)) ** 2


def axes(yy, xx):
    """The two 1-D pixel-centre axes of a separable mgrid pair.

    The model separates, so only `yy[:,0]` and `xx[0,:]` are ever read. A
    caller that evaluates repeatedly on ONE patch -- which is every fit --
    should hoist this out of its loop and use the `_ax` entry points below;
    `lmga.fit` calls the model ~160 times per fit and the grid never changes.
    """
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
    """E only, for one axis; (len(ax), K). What `model` needs and no more."""
    k = 1.0 / (sigma * SQRT2)
    return 0.5 * (erf((ax[:, None] - c[None, :] + 0.5) * k)
                  - erf((ax[:, None] - c[None, :] - 0.5) * k))


def _factors(ax, c, sigma):
    """(E, dE_dc) for one axis; each (len(ax), K).

    dE_dsigma is deliberately NOT computed here -- see `_factors_sigma`. It is
    read only by `jac_free_sigma`, i.e. by the free-sigma diagnostic, while
    this function is on the fixed-sigma fitting path that runs ~1.4M times per
    frame. Computing a derivative nothing reads cost two exps, a multiply and
    a divide on every one of those calls.
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


def pack(b, A, cy, cx):
    """Build a flat theta from b (scalar) and per-emitter arrays."""
    A = np.atleast_1d(np.asarray(A, dtype=float))
    cy = np.atleast_1d(np.asarray(cy, dtype=float))
    cx = np.atleast_1d(np.asarray(cx, dtype=float))
    return np.concatenate([[float(b)], np.stack([A, cy, cx], axis=-1).ravel()])


def unpack(theta):
    """(b, A, cy, cx) as plain arrays."""
    return _unpack(theta)
