"""The in-focus PSF width, measured from the data: `calibrate_sigma`."""

from dataclasses import dataclass

import numpy as np

from .native import localize_stack

__all__ = ["SigmaCalibration", "calibrate_sigma"]


TOL = 0.002  # relative change in sigma
MAX_ROUNDS = 8

# Disable width reporting to avoid truncating the calibration sample.
# Use all fitted widths: selecting only bright spots favors merged pairs.


@dataclass(frozen=True)
class SigmaCalibration:
    """What `calibrate_sigma` measured."""

    sigma: float
    """The in-focus PSF width, px: the median fitted width."""
    ci: tuple
    """95% bootstrap interval of that median, over spots."""
    n_spots: int
    converged: bool
    """False if `max_rounds` ran out before sigma moved less than `tol`."""
    guesses: tuple
    """The sigma each round was run at, first to last."""
    widths: np.ndarray
    """Every fitted width of the final round, px."""


def calibrate_sigma(frames, sigma_guess, *, offset=0.0, roi=None, tol=TOL,
                    max_rounds=MAX_ROUNDS,
                    n_boot=2000, seed=0, n_threads=None):
    """Estimate PSF width from one frame or a (T, H, W) stack.

    Refit with the reporting band disabled, update sigma to the median fitted
    width, and repeat until its relative change is below `tol`. `offset` and
    `roi` follow `localize`. The returned interval bootstraps spots from the
    final round; broad or out-of-focus populations can bias the median upward.
    """
    stack = np.asarray(frames, dtype=float)
    if stack.ndim == 2:
        stack = stack[None]
    if stack.ndim != 3:
        raise ValueError(f"expected a frame or a (T, H, W) stack, got {stack.shape}")
    guess = float(sigma_guess)
    guesses, converged, widths = [], False, np.empty(0)
    for _ in range(int(max_rounds)):
        guesses.append(guess)
        results = localize_stack(stack, guess, offset=offset, roi=roi,
                                 band=None,
                                 n_threads=n_threads)
        widths = np.concatenate([r.fit_sigma for r in results])
        if len(widths) == 0:
            raise ValueError("no spots were found to calibrate on")
        estimate = float(np.median(widths))
        if abs(estimate / guess - 1.0) < tol:
            converged = True
            break
        guess = estimate
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(widths, (int(n_boot), len(widths))), axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return SigmaCalibration(sigma=estimate, ci=(float(lo), float(hi)),
                            n_spots=len(widths), converged=converged,
                            guesses=tuple(guesses), widths=widths)
