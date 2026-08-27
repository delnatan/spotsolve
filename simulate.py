"""Synthetic Poisson emitter images with known ground truth, for validating
the detection pipeline (recovery RMSE vs CRLB, F1, N_est-N_true, achieved
false-positive rate) independently of the un-truthed bead data.
"""

from dataclasses import dataclass

import numpy as np

import psf


@dataclass
class SimResult:
    image: np.ndarray  # Poisson-sampled counts, (H,W)
    clean: np.ndarray  # noiseless model image
    positions: np.ndarray  # (N,2) true (y,x)
    amplitudes: np.ndarray  # (N,) true peak amplitudes
    background: float
    sigma: float


def simulate(
    shape=(64, 64),
    n_emitters=None,
    density=None,
    amplitude_range=(150.0, 600.0),
    background=20.0,
    sigma=1.2,
    border=4.0,
    min_separation=0.0,
    seed=None,
):
    """Render a Poisson-noise emitter image.

    Exactly one of n_emitters or density (emitters/px^2, over the interior
    area excluding `border`) must be given.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    interior_area = max(H - 2 * border, 1.0) * max(W - 2 * border, 1.0)

    if (n_emitters is None) == (density is None):
        raise ValueError("specify exactly one of n_emitters or density")
    if density is not None:
        n_emitters = max(1, int(round(density * interior_area)))

    positions = np.empty((0, 2))
    amplitudes = np.empty((0,))
    if n_emitters > 0:
        if min_separation > 0:
            positions, amplitudes = _sample_min_separated(
                rng, n_emitters, shape, border, min_separation, amplitude_range
            )
        else:
            ys = rng.uniform(border, H - border, size=n_emitters)
            xs = rng.uniform(border, W - border, size=n_emitters)
            positions = np.stack([ys, xs], axis=1)
            amplitudes = rng.uniform(*amplitude_range, size=n_emitters)

    yy, xx = np.mgrid[0:H, 0:W] * 1.0
    if positions.shape[0] > 0:
        theta = psf.pack(background, amplitudes, positions[:, 0], positions[:, 1])
        clean = psf.model(theta, yy, xx, sigma)
    else:
        clean = np.full(shape, background)

    image = rng.poisson(clean).astype(float)

    return SimResult(
        image=image,
        clean=clean,
        positions=positions,
        amplitudes=amplitudes,
        background=background,
        sigma=sigma,
    )


def _sample_min_separated(rng, n, shape, border, min_sep, amp_range, max_tries=2000):
    H, W = shape
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        y = rng.uniform(border, H - border)
        x = rng.uniform(border, W - border)
        if all(np.hypot(y - py, x - px) >= min_sep for py, px in pts):
            pts.append((y, x))
        tries += 1
    positions = np.array(pts)
    amplitudes = rng.uniform(*amp_range, size=positions.shape[0])
    return positions, amplitudes
