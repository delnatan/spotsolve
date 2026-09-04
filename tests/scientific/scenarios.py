"""Deterministic local-ROI scenarios with semantic source truth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from spotsolve import psf


@dataclass(frozen=True)
class Scenario:
    image: np.ndarray
    mean: np.ndarray
    sigma: float
    focused_positions: np.ndarray
    focused_fluxes: np.ndarray
    nuisance_kind: str | None = None
    nuisance_positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=float))
    metadata: dict = field(default_factory=dict)


def _sample(mean, sigma, focused_positions, focused_fluxes, *, seed,
            nuisance_kind=None, nuisance_positions=None, metadata=None,
            poisson=True):
    rng = np.random.default_rng(seed)
    image = rng.poisson(mean).astype(float) if poisson else np.asarray(mean, float).copy()
    nuisance_positions = (np.empty((0, 2), dtype=float) if nuisance_positions is None
                          else np.atleast_2d(np.asarray(nuisance_positions, float)))
    return Scenario(
        image=image,
        mean=np.asarray(mean, dtype=float),
        sigma=float(sigma),
        focused_positions=np.asarray(focused_positions, dtype=float).reshape(-1, 2),
        focused_fluxes=np.asarray(focused_fluxes, dtype=float).reshape(-1),
        nuisance_kind=nuisance_kind,
        nuisance_positions=nuisance_positions,
        metadata={} if metadata is None else dict(metadata),
    )


def _grid(shape):
    return np.mgrid[0:shape[0], 0:shape[1]].astype(float)


def _centre(shape, centre):
    if centre is None:
        return np.array([(shape[0] - 1.0) / 2.0,
                         (shape[1] - 1.0) / 2.0])
    value = np.asarray(centre, dtype=float)
    if value.shape != (2,):
        raise ValueError("centre must be a (y, x) pair")
    return value


def blank(*, shape=(13, 13), background=4.0, sigma=1.2, seed=0,
          poisson=True):
    mean = np.full(shape, float(background))
    return _sample(mean, sigma, [], [], seed=seed, poisson=poisson,
                   metadata={"background": float(background)})


def focused_single(*, shape=(13, 13), background=4.0, sigma=1.2,
                   photons=900.0, centre=None, seed=0, poisson=True):
    centre = _centre(shape, centre)
    yy, xx = _grid(shape)
    theta = psf.pack(background, [photons], [centre[0]], [centre[1]])
    mean = psf.model(theta, yy, xx, sigma)
    return _sample(mean, sigma, [centre], [photons], seed=seed, poisson=poisson,
                   metadata={"background": float(background),
                             "photons": float(photons),
                             "centre": centre.tolist()})


def focused_pair(*, shape=(13, 13), background=4.0, sigma=1.2,
                 bright_photons=900.0, flux_ratio=1.0,
                 separation_ratio=1.0, angle=0.0, centre=None, seed=0,
                 poisson=True):
    if flux_ratio < 1:
        raise ValueError("flux_ratio is bright/dim and must be >= 1")
    centre = _centre(shape, centre)
    direction = np.array([np.sin(angle), np.cos(angle)])
    vector = float(separation_ratio) * sigma * direction
    positions = np.stack([centre + 0.5 * vector, centre - 0.5 * vector])
    fluxes = np.array([bright_photons, bright_photons / flux_ratio], dtype=float)
    yy, xx = _grid(shape)
    theta = psf.pack(background, fluxes, positions[:, 0], positions[:, 1])
    mean = psf.model(theta, yy, xx, sigma)
    return _sample(
        mean, sigma, positions, fluxes, seed=seed, poisson=poisson,
        metadata={"background": float(background),
                  "bright_photons": float(bright_photons),
                  "flux_ratio": float(flux_ratio),
                  "separation_ratio": float(separation_ratio),
                  "angle": float(angle),
                  "centre": centre.tolist()},
    )


def wide_source(*, shape=(13, 13), background=4.0, sigma=1.2,
                photons=900.0, width_ratio=2.0, centre=None, seed=0,
                poisson=True):
    centre = _centre(shape, centre)
    yy, xx = _grid(shape)
    width = float(width_ratio) * sigma
    theta = psf.pack_var_sigma(background, [photons], [centre[0]], [centre[1]],
                               [width])
    mean = psf.model_var_sigma(theta, yy, xx)
    return _sample(
        mean, sigma, [], [], seed=seed, nuisance_kind="wide",
        nuisance_positions=[centre], poisson=poisson,
        metadata={"background": float(background), "photons": float(photons),
                  "width_ratio": float(width_ratio),
                  "centre": centre.tolist()},
    )


def smooth_haze(*, shape=(25, 25), background=4.0, sigma=1.2,
                peak_above_background=8.0, correlation_length=5.0, seed=0,
                poisson=True):
    """A positive, non-Gaussian low-frequency background field."""
    rng = np.random.default_rng(seed)
    raw = gaussian_filter(rng.normal(size=shape), correlation_length,
                          mode="reflect")
    raw -= raw.min()
    scale = float(raw.max())
    haze = np.zeros(shape) if scale <= 0 else raw / scale * peak_above_background
    mean = float(background) + haze
    # Use an independent stream for the photon draw while retaining deterministic
    # haze geometry for this seed.
    return _sample(
        mean, sigma, [], [], seed=seed + 10_000_019, poisson=poisson,
        nuisance_kind="smooth_haze",
        metadata={"background": float(background),
                  "peak_above_background": float(peak_above_background),
                  "correlation_length": float(correlation_length)},
    )
