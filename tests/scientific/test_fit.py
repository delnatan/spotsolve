import numpy as np

from spotsolve.prototype import FitOptions, Hypothesis, StartGrid, fit_hypotheses
from tests.scientific.scenarios import (
    focused_pair,
    focused_single,
    smooth_haze,
    wide_source,
)


FAST_OPTIONS = FitOptions(max_iter=180, screen_iter=18, keep_screened=2)


def test_noiseless_single_is_recovered():
    scenario = focused_single(photons=700.0, centre=(6.15, 5.72), poisson=False)
    fits = fit_hypotheses(
        scenario.image, scenario.sigma, centre=(6.0, 6.0),
        start_grid=StartGrid(separation_ratios=(1.0,), flux_ratios=(1.0,),
                             angle_offsets=(0.0,)),
        options=FAST_OPTIONS,
    )
    h1 = fits[Hypothesis.H1]
    np.testing.assert_allclose(h1.physical["positions"][0],
                               scenario.focused_positions[0], atol=2e-3)
    np.testing.assert_allclose(h1.physical["fluxes"],
                               scenario.focused_fluxes, rtol=2e-3)
    assert h1.objective < 1e-5


def test_noiseless_pair_beats_single_and_recovers_sources():
    scenario = focused_pair(bright_photons=900.0, flux_ratio=2.0,
                            separation_ratio=1.0, angle=0.37, poisson=False)
    fits = fit_hypotheses(
        scenario.image, scenario.sigma,
        start_grid=StartGrid(separation_ratios=(0.8, 1.0, 1.3),
                             flux_ratios=(1.0, 2.0, 4.0),
                             angle_offsets=(0.0, np.pi / 2.0)),
        options=FAST_OPTIONS,
    )
    h1, h2 = fits[Hypothesis.H1], fits[Hypothesis.H2]
    assert h2.objective + 0.5 < h1.objective
    assert abs(h2.physical["separation"] - 1.2) < 0.02
    assert not h2.collapsed


def test_noiseless_wide_source_is_explained_by_nuisance_model():
    scenario = wide_source(photons=900.0, width_ratio=2.0, poisson=False)
    fits = fit_hypotheses(
        scenario.image, scenario.sigma,
        start_grid=StartGrid(separation_ratios=(1.0,), flux_ratios=(1.0,),
                             angle_offsets=(0.0,)),
        options=FAST_OPTIONS,
    )
    h1, hwide = fits[Hypothesis.H1], fits[Hypothesis.HWIDE]
    assert hwide.objective + 1.0 < h1.objective
    assert abs(hwide.physical["width"] / scenario.sigma - 2.0) < 0.02
    assert hwide.objective < 1e-5


def test_pair_separation_is_radially_bounded():
    scenario = focused_pair(separation_ratio=4.0, poisson=False)
    fits = fit_hypotheses(scenario.image, scenario.sigma)
    pair = fits[Hypothesis.H2]
    assert pair.physical["separation"] <= (
        fits.options.pair_max_separation_ratio * scenario.sigma + 1e-9)
    assert pair.at_boundary


def test_smooth_background_can_explain_correlated_haze():
    scenario = smooth_haze(shape=(13, 13), poisson=False, seed=27)
    fits = fit_hypotheses(scenario.image, scenario.sigma)
    assert (fits[Hypothesis.HSMOOTH].objective
            < fits[Hypothesis.H0].objective)


def test_default_pair_starts_match_dense_grid_on_difficult_noise_draw():
    # This seed exposed a 0.86-nat local-minimum loss when the orientation
    # screen contained only the moment axis and its perpendicular.
    scenario = focused_pair(bright_photons=900.0, flux_ratio=1.0,
                            separation_ratio=0.75, seed=7131)
    default = fit_hypotheses(scenario.image, scenario.sigma)
    dense = fit_hypotheses(
        scenario.image, scenario.sigma,
        start_grid=StartGrid(
            separation_ratios=(0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.9, 2.3),
            flux_ratios=(1.0, 2.0, 4.0, 8.0),
            angle_offsets=(0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0),
        ),
    )
    loss = (default[Hypothesis.H2].objective
            - dense[Hypothesis.H2].objective)
    assert loss < 1e-6
