import numpy as np

from spotsolve.prototype import (
    ProposalOptions,
    generate_proposals,
    poisson_score_map,
)
from tests.scientific.scenarios import focused_pair, focused_single, smooth_haze


def _axial_error(left, right):
    return 0.5 * abs(np.angle(np.exp(2j * (left - right))))


def test_poisson_score_is_zero_when_data_equal_background():
    background = np.linspace(2.0, 8.0, 17 * 19).reshape(17, 19)
    score, flux = poisson_score_map(background, background, sigma=1.2)
    np.testing.assert_allclose(score, 0.0, atol=1e-14)
    np.testing.assert_allclose(flux, 0.0, atol=1e-14)


def test_noiseless_single_proposal_is_subpixel_and_has_flux_scale():
    scenario = focused_single(
        shape=(33, 33), photons=900.0, centre=(16.2, 15.7), poisson=False)
    result = generate_proposals(
        scenario.image, scenario.sigma, background=4.0)
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert np.linalg.norm(proposal.centre - scenario.focused_positions[0]) < 0.1
    assert abs(proposal.flux_score / 900.0 - 1.0) < 0.05


def test_pair_hessian_axis_tracks_close_pair_orientation():
    scenario = focused_pair(
        shape=(33, 33), bright_photons=900.0, flux_ratio=1.0,
        separation_ratio=1.0, angle=0.7, centre=(16.2, 15.7),
        poisson=False)
    result = generate_proposals(
        scenario.image, scenario.sigma, background=4.0)
    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert _axial_error(proposal.pair_axis, 0.7) < 0.2
    np.testing.assert_allclose(
        proposal.centre, np.mean(scenario.focused_positions, axis=0), atol=0.1)


def test_internal_background_keeps_faint_source_and_bounds_haze_candidates():
    single = focused_single(
        shape=(33, 33), photons=150.0, centre=(16.2, 15.7), seed=4)
    haze = smooth_haze(shape=(33, 33), seed=3)
    single_result = generate_proposals(single.image, single.sigma)
    haze_result = generate_proposals(haze.image, haze.sigma)
    assert any(np.linalg.norm(proposal.centre - single.focused_positions[0]) < 1.0
               for proposal in single_result.proposals)
    assert len(haze_result.proposals) <= 8


def test_proposal_budget_and_component_grouping_are_explicit():
    image = np.full((41, 41), 4.0)
    for y in range(5, 38, 8):
        for x in range(5, 38, 8):
            image[y, x] = 100.0
    options = ProposalOptions(
        score_threshold=1.0, max_proposals=3,
        component_link_ratio=100.0)
    result = generate_proposals(
        image, sigma=0.8, background=4.0, options=options)
    assert result.local_maxima_above_threshold > 3
    assert len(result.proposals) == 3
    assert result.budget_exhausted
    assert len(result.components) == 1
    assert result.components[0].proposal_indices == (0, 1, 2)
