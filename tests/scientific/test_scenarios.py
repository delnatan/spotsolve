import numpy as np

from tests.scientific.metrics import focused_metrics, match_positions, summarize
from tests.scientific.scenarios import focused_pair, smooth_haze, wide_source


def test_scenarios_are_deterministic_and_keep_nuisance_semantics():
    a = focused_pair(seed=17)
    b = focused_pair(seed=17)
    np.testing.assert_array_equal(a.image, b.image)

    wide = wide_source(seed=2)
    haze = smooth_haze(seed=3)
    assert len(wide.focused_positions) == 0
    assert len(haze.focused_positions) == 0
    assert focused_metrics(wide.nuisance_positions, wide)["false_positive"] == 1


def test_matching_is_one_to_one_and_aggregates():
    truth = focused_pair(poisson=False)
    estimated = np.vstack([truth.focused_positions, truth.focused_positions[0]])
    rows, cols, distances = match_positions(
        estimated, truth.focused_positions, max_distance=0.2)
    assert len(rows) == len(cols) == len(distances) == 2
    record = focused_metrics(estimated, truth, max_distance=0.2)
    assert record["true_positive"] == 2
    assert record["false_positive"] == 1
    summary = summarize([record])
    assert summary["recall"] == 1.0
    assert summary["focused_false_emitters_per_frame"] == 1.0
