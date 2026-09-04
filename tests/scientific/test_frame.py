import numpy as np

from spotsolve.prototype import (
    Hypothesis,
    ProposalOptions,
    fit_proposal_components,
    generate_proposals,
)
from tests.scientific.scenarios import focused_single


def test_proposal_fit_recovers_global_single_position():
    scenario = focused_single(
        shape=(33, 33), photons=900.0, centre=(16.2, 15.7), poisson=False)
    proposals = generate_proposals(
        scenario.image, scenario.sigma, background=4.0)
    collection = fit_proposal_components(
        scenario.image, scenario.sigma, proposals)
    assert len(collection.fitted) == 1
    proposed = collection.fitted[0]
    local = proposed.fits[Hypothesis.H1].physical["positions"][0]
    global_position = local + np.asarray(proposed.origin)
    np.testing.assert_allclose(
        global_position, scenario.focused_positions[0], atol=2e-3)


def test_edge_and_component_budget_are_reported():
    image = np.full((33, 33), 4.0)
    image[1, 1] = 200.0
    image[16, 16] = 200.0
    image[27, 27] = 200.0
    proposals = generate_proposals(
        image, sigma=1.2, background=4.0,
        options=ProposalOptions(component_link_ratio=1.0))
    all_components = fit_proposal_components(
        image, 1.2, proposals, max_components=64)
    assert all_components.skipped_edge_components == 0
    assert len(all_components.fitted) == len(proposals.components)
    collection = fit_proposal_components(
        image, 1.2, proposals, max_components=2)
    assert collection.skipped_budget_components >= 1
    assert len(collection.fitted) <= 2
