"""Retained calibrated model: focused emitters + one defocused emitter + haze.

Layout: 16 bicubic background coefficients (10 photons/pixel units), then
(flux/1000, y, x, depth_um) for defocus, then (flux/1000, y, x) per focus.
Background pixel rates and source fluxes are nonnegative. Polynomial
coefficients may be signed. Calibration normalization is never ROI-dependent.
"""
from dataclasses import dataclass, field
from math import comb

import numpy as np

from ._numerics import ROIGrid
from .psf_bank import PixelPSFBank
from .types import FLUX_UNIT


def evaluate_component(count, theta, model):
    return model.evaluate(count, theta)


def parameter_bounds(data, model, count):
    return model.parameter_bounds(data, count)


@dataclass(frozen=True)
class FocusedModel:
    shape: tuple[int, int]
    focus_bounds: tuple[float, float, float, float]
    psf: PixelPSFBank = field(repr=False, compare=False)
    defocus_bounds_um: tuple[float, float] = (.25, .55)
    # This scale affects initial source proposals only, never the fitted PSF.
    seed_sigma: float = 1.0
    grid: ROIGrid = field(init=False, repr=False, compare=False)
    smooth_basis: np.ndarray = field(init=False, repr=False, compare=False)
    _constraints: tuple = field(init=False, repr=False, compare=False)

    background_size = 16
    nuisance_size = 20
    broad_index = 16

    def __post_init__(self):
        if (len(self.shape) != 2 or any(not isinstance(v, (int, np.integer)) or v < 3
                                      for v in self.shape)):
            raise ValueError("shape must contain two integers >=3")
        bounds = np.asarray(self.focus_bounds, float)
        if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
            raise ValueError("focus_bounds must contain four finite values")
        h, w = self.shape
        y0, x0, y1, x1 = bounds
        if not (-.5 <= y0 < y1 <= h-.5 and -.5 <= x0 < x1 <= w-.5):
            raise ValueError("focus_bounds must lie inside the observed patch")
        if not np.isfinite(self.seed_sigma) or self.seed_sigma <= 0:
            raise ValueError("seed_sigma must be finite and positive")
        if not isinstance(self.psf, PixelPSFBank):
            raise ValueError("psf must be a PixelPSFBank")
        if not self.psf.depth_um[0] <= 0 <= self.psf.depth_um[-1]:
            raise ValueError("PSF calibration requires a zero-depth plane")
        z = np.asarray(self.defocus_bounds_um, float)
        if (z.shape != (2,) or not np.all(np.isfinite(z)) or not 0 < z[0] < z[1]
                or not self.psf.depth_um[0] <= z[0] < z[1] <= self.psf.depth_um[-1]):
            raise ValueError("positive defocus bounds must lie inside calibration")
        extent = max(h, w)-.5
        if self.psf.offsets_px[0] > -extent or self.psf.offsets_px[-1] < extent:
            raise ValueError("PSF table must cover all ROI offsets at allowed centers")
        object.__setattr__(self, "grid", ROIGrid.from_shape(self.shape))
        def bernstein(length):
            t = np.linspace(0, 1, length)
            basis = np.stack([comb(3, k)*t**k*(1-t)**(3-k) for k in range(4)], axis=-1)
            return basis/basis.sum(axis=1, keepdims=True)
        by, bx = bernstein(h), bernstein(w)
        basis = (by[:, None, :, None]*bx[None, :, None, :]).reshape(h, w, 16)
        basis.setflags(write=False)
        object.__setattr__(self, "smooth_basis", basis)
        constraints = []
        for count in range(3):
            matrix = np.zeros((h*w, self.nuisance_size+3*count))
            matrix[:, :16] = basis.reshape(-1, 16)
            matrix.setflags(write=False)
            constraints.append(matrix)
        object.__setattr__(self, "_constraints", tuple(constraints))

    @property
    def broad_starts(self):
        return np.linspace(*self.defocus_bounds_um, 3)

    def broad_unit(self, y, x, parameter, grid=None):
        return self.psf.evaluate(self.grid if grid is None else grid, y, x, parameter)

    def affine_indices(self, count):
        return np.r_[np.arange(17), self.nuisance_size+3*np.arange(count)].astype(int)

    def amplitude_geometry(self, count):
        return [(16, (17, 18, 19))]+[(20+3*k, (21+3*k, 22+3*k)) for k in range(count)]

    def rate_constraints(self, count):
        return self._constraints[count]

    def nuisance_starts(self, background, excess, centres, midpoint):
        starts = [np.r_[np.full(16, background/10), max(excess, 1)/FLUX_UNIT, point, z]
                  for point in centres for z in self.broad_starts]
        starts.append(np.r_[np.full(16, background/10), 0, midpoint, self.broad_starts[0]])
        return starts

    def parameter_bounds(self, data, count):
        h, w = self.shape
        flux_max = max(float(data.sum())*5, FLUX_UNIT)/FLUX_UNIT
        background_max = max(float(data.max())*10, 100)/10
        y0, x0, y1, x1 = self.focus_bounds
        lo = [-background_max]*16+[0, -.5, -.5, self.defocus_bounds_um[0]]+[0, y0, x0]*count
        hi = [background_max]*16+[flux_max, h-.5, w-.5, self.defocus_bounds_um[1]]+[flux_max, y1, x1]*count
        return np.asarray(lo), np.asarray(hi)

    def evaluate(self, count, theta):
        theta = np.asarray(theta, float)
        if count not in (0, 1, 2) or theta.shape != (self.nuisance_size+3*count,):
            raise ValueError("theta must contain 20+3*K parameters, K in {0,1,2}")
        jac = np.empty(self.shape+(len(theta),))
        jac[..., :16] = 10*self.smooth_basis
        mean = 1e-4+np.sum(jac[..., :16]*theta[:16], axis=-1)
        unit, dy, dx, dz = self.broad_unit(*theta[17:20])
        mean = mean+FLUX_UNIT*theta[16]*unit
        jac[..., 16] = FLUX_UNIT*unit
        for j, derivative in zip((17, 18, 19), (dy, dx, dz)):
            jac[..., j] = FLUX_UNIT*theta[16]*derivative
        for k in range(count):
            offset = 20+3*k
            amplitude, y, x = theta[offset:offset+3]
            unit, dy, dx, _ = self.psf.evaluate(self.grid, y, x, 0.)
            mean += FLUX_UNIT*amplitude*unit
            jac[..., offset] = FLUX_UNIT*unit
            jac[..., offset+1] = FLUX_UNIT*amplitude*dy
            jac[..., offset+2] = FLUX_UNIT*amplitude*dx
        return mean, jac
