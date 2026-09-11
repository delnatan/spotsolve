"""`calibrate_sigma`: the in-focus PSF width, from the data."""

import numpy as np
import pytest

import spotsolve
from spotsolve.simulate import simulate

pytest.importorskip("spotsolve_rs")


def _stack(sigma, density=0.005, seeds=(17, 18, 19)):
    return np.stack([simulate(shape=(96, 96), density=density,
                              amplitude_range=(900.0, 1900.0), sigma=sigma,
                              seed=s).image for s in seeds])


@pytest.mark.parametrize("guess", [1.04, 1.3, 1.625])      # 0.8x, 1x, 1.25x
def test_converges_to_the_true_width_from_either_side(guess):
    c = spotsolve.calibrate_sigma(_stack(1.3), guess, gain=1.0)
    assert c.converged and len(c.guesses) <= 6
    # Measured at 1.010x on this field; the median's own bias, not noise.
    assert c.sigma == pytest.approx(1.3, rel=0.02)
    assert c.ci[0] <= c.sigma <= c.ci[1]
    assert c.n_spots == len(c.widths) > 50


def test_a_single_frame_is_accepted_and_empty_frames_refused():
    frame = _stack(1.0, seeds=(17,))[0]
    assert spotsolve.calibrate_sigma(frame, 1.0, gain=1.0).sigma == pytest.approx(1.0, rel=0.02)
    with pytest.raises(ValueError):
        spotsolve.calibrate_sigma(np.full((40, 40), 20.0), 1.2, gain=1.0)


def test_localizations_report_every_fit_once():
    img = _stack(1.3, density=0.02, seeds=(18,))[0]
    locs = spotsolve.localize(img, 1.3, gain=1.0)
    every = spotsolve.localize(img, 1.3, gain=1.0, band=None)
    # band=None reports every fit as a detection; the default splits the
    # same fits between detections and rejects.
    assert len(locs) + len(locs.rejects) == len(every)
    assert set(locs.rejects["reason"]) <= {"too_narrow", "too_wide", "edge"}
    assert np.all((locs.sigma_ratio >= spotsolve.BAND[0]) & (locs.sigma_ratio <= spotsolve.BAND[1]))
    assert locs.info["search_fits"] > 0 and locs.read_noise == 0.0
