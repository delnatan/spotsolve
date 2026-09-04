"""Parameter conversions for the explicit local hypotheses."""

from __future__ import annotations

import numpy as np


def pair_components(theta):
    """Return ``(fluxes, positions)`` for an H2 parameter vector.

    H2 uses ``[log_b, gy, gx, log_F, cy, cx, d, angle, q]``.  The first
    component is defined as the brighter one, so ``q >= 0.5`` removes the
    otherwise exact label-swapping duplicate.  ``(cy, cx)`` remains the flux
    centroid for every separation and flux ratio.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (9,):
        raise ValueError("H2 theta must have 9 entries")
    total = float(np.exp(theta[3]))
    centre = theta[4:6]
    separation, angle = theta[6:8]
    vector = separation * np.array([np.sin(angle), np.cos(angle)])
    q = float(theta[8])
    positions = np.stack([
        centre + (1.0 - q) * vector,
        centre - q * vector,
    ])
    return np.array([q * total, (1.0 - q) * total]), positions


def pair_summary(theta):
    """Human- and metric-facing H2 parameters without changing fit geometry."""
    fluxes, positions = pair_components(theta)
    theta = np.asarray(theta, dtype=float)
    return {
        "centroid": theta[4:6].copy(),
        "total_flux": float(fluxes.sum()),
        "flux_fraction": float(theta[8]),
        "separation": float(theta[6]),
        "angle": float(np.mod(theta[7], 2.0 * np.pi)),
        "fluxes": fluxes,
        "positions": positions,
    }
