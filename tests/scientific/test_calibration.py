import numpy as np
import pytest

from spotsolve.prototype import (
    BootstrapCalibration,
    CalibrationSpec,
    CorrelatedHazeNull,
    FitOptions,
    Hypothesis,
    NullModel,
    ProposalBootstrapCalibration,
    ProposalOptions,
    StartGrid,
    calibrate_local,
    calibrate_proposal_search,
    fit_hypotheses,
    select_local,
    select_proposed_frame,
)
from tests.scientific.scenarios import (
    blank,
    focused_pair,
    focused_single,
    smooth_haze,
    wide_source,
)


OPTIONS = FitOptions(max_iter=120, screen_iter=12, keep_screened=2)
GRID = StartGrid(separation_ratios=(0.6, 1.2), flux_ratios=(1.0, 4.0),
                 angle_offsets=(0.0, np.pi / 2.0))


def _fits(scenario):
    return fit_hypotheses(scenario.image, scenario.sigma,
                          options=OPTIONS, start_grid=GRID)


def _null_models():
    scenarios = {
        Hypothesis.H0: blank(poisson=False),
        Hypothesis.HSMOOTH: smooth_haze(shape=(13, 13), poisson=False),
        Hypothesis.H1: focused_single(poisson=False),
        Hypothesis.HWIDE: wide_source(width_ratio=2.0, poisson=False),
    }
    return tuple(
        NullModel.from_fit(_fits(scenario)[kind], scenario.image.shape,
                           scenario.sigma)
        for kind, scenario in scenarios.items()
    )


def test_bootstrap_has_class_conditional_nulls_and_roundtrips(tmp_path):
    calibration = calibrate_local(
        _null_models(), 3, seed=41, options=OPTIONS, start_grid=GRID)
    assert calibration.focus_null[Hypothesis.H0].shape == (1, 3)
    assert calibration.focus_null[Hypothesis.HSMOOTH].shape == (1, 3)
    assert calibration.focus_null[Hypothesis.HWIDE].shape == (1, 3)
    assert calibration.pair_null[Hypothesis.H0].shape == (1, 3)
    assert calibration.pair_null[Hypothesis.HSMOOTH].shape == (1, 3)
    assert calibration.pair_null[Hypothesis.H1].shape == (1, 3)
    assert calibration.pair_null[Hypothesis.HWIDE].shape == (1, 3)

    # The plus-one rule never reports a zero p-value; with three draws its
    # smallest possible value is 1/4.
    assert calibration.p_value("pair", np.inf, Hypothesis.H1) == 0.25
    conservative = calibration.conservative_p(
        "pair", 0.0, (Hypothesis.H0, Hypothesis.H1, Hypothesis.HWIDE))
    assert conservative == max(
        calibration.p_value("pair", 0.0, kind)
        for kind in (Hypothesis.H0, Hypothesis.H1, Hypothesis.HWIDE))

    path = tmp_path / "calibration.json"
    calibration.save(path)
    restored = BootstrapCalibration.load(path)
    assert restored.fingerprint == calibration.fingerprint
    for kind, values in calibration.pair_null.items():
        np.testing.assert_array_equal(restored.pair_null[kind], values)

    damaged = calibration.to_dict()
    damaged["pair_null"][Hypothesis.H1.value][0][0] += 1.0
    with pytest.raises(ValueError, match="content fingerprint"):
        BootstrapCalibration.from_dict(damaged)


def test_multiple_nuisance_cells_are_not_pooled():
    spec = CalibrationSpec.create((13, 13), 1.2, OPTIONS, GRID)
    calibration = BootstrapCalibration(
        spec=spec,
        focus_null={
            Hypothesis.H0: np.zeros((1, 3)),
            Hypothesis.HSMOOTH: np.zeros((1, 3)),
            Hypothesis.HWIDE: np.array([[0.0, 0.0, 0.0],
                                        [10.0, 10.0, 10.0]]),
        },
        pair_null={
            Hypothesis.H0: np.zeros((1, 3)),
            Hypothesis.HSMOOTH: np.zeros((1, 3)),
            Hypothesis.H1: np.zeros((1, 3)),
            Hypothesis.HWIDE: np.zeros((2, 3)),
        },
        null_models=(), draws_per_model=3, seed=0,
    )
    # At an observed gain of five, a pooled distribution would report 1/2.
    # The difficult wide-source cell correctly forces the conservative p to 1.
    assert calibration.p_value("focus", 5.0, Hypothesis.HWIDE) == 1.0


def test_correlated_haze_null_resamples_field_and_roundtrips():
    null = CorrelatedHazeNull(
        shape=(9, 9), sigma=1.2, background=4.0,
        peak_above_background=8.0, correlation_length=3.0)
    first = null.sample(np.random.default_rng(123))
    repeated = null.sample(np.random.default_rng(123))
    different = null.sample(np.random.default_rng(124))
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)

    spec = CalibrationSpec.create((9, 9), 1.2, OPTIONS, GRID)
    values = np.zeros((1, 2))
    calibration = BootstrapCalibration(
        spec=spec,
        focus_null={Hypothesis.H0: values, Hypothesis.HSMOOTH: values,
                    Hypothesis.HWIDE: values},
        pair_null={Hypothesis.H0: values, Hypothesis.HSMOOTH: values,
                   Hypothesis.H1: values, Hypothesis.HWIDE: values},
        null_models=(null,), draws_per_model=2, seed=8,
    )
    restored = BootstrapCalibration.from_dict(calibration.to_dict())
    assert isinstance(restored.null_models[0], CorrelatedHazeNull)
    assert restored.fingerprint == calibration.fingerprint


def test_calibration_fingerprint_rejects_different_fit_configuration():
    calibration = calibrate_local(
        _null_models(), 1, seed=9, options=OPTIONS, start_grid=GRID)
    fits = fit_hypotheses(
        focused_single(seed=12).image, 1.2,
        options=OPTIONS,
        start_grid=StartGrid(separation_ratios=(0.5,), flux_ratios=(1.0,),
                             angle_offsets=(0.0,)),
    )
    with pytest.raises(ValueError, match="does not match calibration"):
        select_local(fits, calibration, alpha_focus=0.5, alpha_pair=0.5)


def test_sequential_selection_distinguishes_wide_single_and_pair():
    spec = CalibrationSpec.create((13, 13), 1.2, OPTIONS, GRID)
    zeros = np.zeros(9)
    pair_null = np.ones(9)
    calibration = BootstrapCalibration(
        spec=spec,
        focus_null={Hypothesis.H0: zeros, Hypothesis.HSMOOTH: zeros,
                    Hypothesis.HWIDE: zeros},
        pair_null={Hypothesis.H0: pair_null,
                   Hypothesis.HSMOOTH: pair_null, Hypothesis.H1: pair_null,
                   Hypothesis.HWIDE: pair_null},
        null_models=(), draws_per_model=9, seed=0,
    )

    single = select_local(_fits(focused_single(poisson=False)), calibration,
                          alpha_focus=0.11, alpha_pair=0.11)
    pair = select_local(_fits(focused_pair(separation_ratio=1.25, poisson=False)),
                        calibration, alpha_focus=0.11, alpha_pair=0.11)
    wide = select_local(_fits(wide_source(width_ratio=2.0, poisson=False)),
                        calibration, alpha_focus=0.11, alpha_pair=0.11)
    assert single.hypothesis is Hypothesis.H1
    assert single.n_focus == 1
    assert pair.hypothesis is Hypothesis.H2
    assert pair.n_focus == 2
    assert wide.hypothesis is Hypothesis.HWIDE
    assert wide.n_focus == 0


def test_proposal_search_calibration_roundtrips_and_selects_frame(tmp_path):
    proposal_options = ProposalOptions(
        score_threshold=1.5, max_proposals=4)
    calibration = calibrate_proposal_search(
        _null_models(), 1, roi_size=9, max_components=4, seed=52,
        fit_options=OPTIONS, start_grid=GRID,
        proposal_options=proposal_options)
    assert calibration.focus_null[Hypothesis.H0].shape == (1, 1)
    assert calibration.pair_null[Hypothesis.H1].shape == (1, 1)
    assert calibration.proposal_counts[Hypothesis.HWIDE].shape == (1, 1)

    path = tmp_path / "proposal-calibration.json"
    calibration.save(path)
    restored = ProposalBootstrapCalibration.load(path)
    assert restored.fingerprint == calibration.fingerprint
    with pytest.raises(ValueError, match="pipeline does not match"):
        restored.check_pipeline(
            (13, 13), 1.2,
            ProposalOptions(score_threshold=2.0, max_proposals=4),
            9, 4)

    scenario = focused_single(shape=(13, 13), photons=900.0, seed=73)
    selected = select_proposed_frame(
        scenario.image, scenario.sigma, restored,
        alpha_focus=0.51, alpha_pair=0.51)
    assert len(selected.focused_positions) >= 1


def test_boundary_railed_pair_is_not_reported_as_resolved():
    spec = CalibrationSpec.create((13, 13), 1.2, FitOptions(), StartGrid())
    zeros = np.zeros(19)
    calibration = BootstrapCalibration(
        spec=spec,
        focus_null={Hypothesis.H0: zeros, Hypothesis.HSMOOTH: zeros,
                    Hypothesis.HWIDE: zeros},
        pair_null={Hypothesis.H0: zeros, Hypothesis.HSMOOTH: zeros,
                   Hypothesis.H1: zeros, Hypothesis.HWIDE: zeros},
        null_models=(), draws_per_model=19, seed=0,
    )
    scenario = focused_pair(separation_ratio=4.0, poisson=False)
    fits = fit_hypotheses(scenario.image, scenario.sigma)
    assert fits[Hypothesis.H2].at_boundary
    decision = select_local(
        fits, calibration, alpha_focus=0.1, alpha_pair=0.1)
    assert not decision.statistics.pair_eligible
    assert decision.hypothesis is not Hypothesis.H2
