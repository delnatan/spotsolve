"""Synthetic Poisson emitter images with known ground truth, for validating
the detection pipeline (recovery RMSE vs CRLB, F1, N_est-N_true, achieved
false-positive rate) independently of the un-truthed bead data.
"""

from dataclasses import dataclass

import numpy as np

from . import psf


@dataclass
class SimResult:
    image: np.ndarray  # Poisson-sampled counts, (H,W)
    clean: np.ndarray  # noiseless model image
    positions: np.ndarray  # (N,2) true (y,x)
    amplitudes: np.ndarray  # (N,) true peak amplitudes
    background: float
    sigma: float
    sigmas: np.ndarray = None
    """(N,) per-emitter true width, or None when every emitter is at `sigma`.

    Recorded rather than derived because it is the only truth column a
    width-mismatch arm can be scored against, and the solver never sees it.
    """


def simulate(
    shape=(64, 64),
    n_emitters=None,
    density=None,
    amplitude_range=(150.0, 600.0),
    background=20.0,
    sigma=1.2,
    sigma_spread=0.0,
    border=4.0,
    min_separation=0.0,
    seed=None,
):
    """Render a Poisson-noise emitter image.

    Exactly one of n_emitters or density (emitters/px^2, over the interior
    area excluding `border`) must be given.

    `sigma_spread` is the sd, IN LOG SPACE, of a per-emitter lognormal width
    drawn around `sigma`. It exists because a field where every emitter sits at
    the model's own width cannot exhibit the failure the real data does: on
    `beads_80pct-glycerol` the per-object sigma_ratio sd is 0.194 and the
    per-localization sd 0.442, so 0.2 and 0.4 bracket that data. Measured at
    the benchmark's easiest arm (bright, density 0.015, flat background, where
    recall is 0.978 and FP 0.00), width spread alone drives N_est/N_true:

        spread   0.0    0.10   0.20   0.40
        Nest/Nt  0.95   1.07   1.26   1.56

    and the excess is one-sided tiling, monotone in each emitter's OWN width:
    at sigma_true/sigma <= 1.05 no emitter collects a second detection, at
    1.05-1.25 38% do, at 1.25-1.60 85% do. Emitters NARROWER than the model
    never tile.

    `sigma_spread = 0` draws no random numbers and renders through `psf.model`,
    so every existing seed reproduces its field byte for byte.
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

    # Drawn only when asked for, and only after positions and amplitudes, so a
    # spread of 0 leaves the random stream exactly where it was.
    sigmas = None
    if sigma_spread > 0 and positions.shape[0] > 0:
        sigmas = sigma * np.exp(
            rng.normal(0.0, sigma_spread, size=positions.shape[0]))

    yy, xx = np.mgrid[0:H, 0:W] * 1.0
    if positions.shape[0] == 0:
        clean = np.full(shape, background)
    elif sigmas is None:
        theta = psf.pack(background, amplitudes, positions[:, 0], positions[:, 1])
        clean = psf.model(theta, yy, xx, sigma)
    else:
        theta = psf.pack_var_sigma(background, amplitudes, positions[:, 0],
                                   positions[:, 1], sigmas)
        clean = psf.model_var_sigma(theta, yy, xx)

    image = rng.poisson(clean).astype(float)

    return SimResult(
        image=image,
        clean=clean,
        positions=positions,
        amplitudes=amplitudes,
        background=background,
        sigma=sigma,
        sigmas=sigmas,
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
