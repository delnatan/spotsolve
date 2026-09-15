"""The in-focus PSF width, measured from the data: `calibrate_sigma`."""

from dataclasses import dataclass

import numpy as np

from .native import localize_stack

__all__ = ["SigmaCalibration", "calibrate_sigma"]


TOL = 0.002
# Relative change in sigma below which the iteration stops. From a guess 0.8x
# or 1.25x the truth it converges to the same value either way, in 2-6 rounds
# (simulated 96x96 fields, sigma 1.0 / 1.3 / 1.6, densities 0.005-0.04 px^-2).

MAX_ROUNDS = 8

# How the estimate is formed, and why. Measured 2026-09-11 on simulated
# fields of one true width (three 96x96 frames per cell), converged
# estimate / truth:
#
#   The reporting band must be OFF while calibrating. With it on, a guess
#   25% too large puts the true in-focus spots at 0.8x the guess -- the band's
#   lower edge -- and loses half of them: estimates 1.02-1.13x truth. Off, the
#   same starts give 1.00x on bright fields.
#
#   The MEDIAN of all fitted widths, not of the brightest:
#
#     field                         all fits    brightest 50%   brightest 25%
#     sparse, bright                1.00-1.01   1.00-1.01       1.00-1.01
#     dense or dim (0.02-0.04)      1.00-1.03   1.01-1.06       1.02-1.07
#
#   In crowded fields the brightest spots include merged pairs, which fit
#   wide. A half-sample MODE targets the narrowest peak instead but was
#   noisier (-2% to +5% on single-width fields) and read 0.80-0.85x when the
#   widths genuinely vary; it was measured and not chosen.
#
# What the median does NOT do is separate populations: any wider
# sub-population pulls it up. On the 80% glycerol bead frames 0-4 (measured
# with the camera's gain 1.93 and read noise 2.41 e-, before the detector
# stopped taking them) it read 1.309 px [1.298, 1.326], where the mode of the
# same widths is 1.184 -- the difference is the out-of-focus beads. For a
# sample like that, `SigmaCalibration.widths` holds every fitted width.


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
    """Measure the in-focus PSF width from one frame or a `(T, H, W)` stack.

    Localizes every frame at the current guess with the reporting band off,
    takes the median of all fitted widths, and repeats at that value until it
    moves by less than `tol` (relative). The guess only needs to be within
    ~25% of the truth. `offset` and `roi` are as in `localize`; the more
    spots, the tighter `ci`.
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
