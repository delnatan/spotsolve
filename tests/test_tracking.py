"""The linker: its table contract, and what it recovers from known truth.

Optimality of each frame's assignment is checked against enumeration in the
Rust unit tests (`spotsolve_core::track`). Checked here: the boundary Python
owns (columns in, one column out, row order) and an accuracy floor on
simulated trajectories, so a regression shows up as a worse answer.
"""

import numpy as np
import polars as pl
import pytest

import spotsolve

pytest.importorskip("spotsolve_rs")

# The GEM dataset's regime: D = 0.3 um^2/s at 104 nm per px and 22 ms per
# frame is 0.61 px^2/frame, and 29 nm of localization error is 0.28 px.
D_PX = 0.61
SE_PX = 0.28
# Three times the rms 2-D step of a mobile particle.
MAX_STEP = 3 * np.sqrt(4 * D_PX + 4 * SE_PX**2)


def movie(seed, n_particles=120, n_frames=20, field=128.0, p_detect=1.0,
          immobile=0.3, se=SE_PX, se_spread=0.3):
    """Brownian particles with known identity. -> (locs, true particle id)."""
    rng = np.random.default_rng(seed)
    d = np.where(rng.random(n_particles) < immobile, 0.0, D_PX)
    pos = rng.uniform(0, field, (n_particles, 2))
    rows = []
    for f in range(n_frames):
        if f:
            pos = pos + rng.normal(0.0, np.sqrt(2 * d)[:, None], pos.shape)
        seen = rng.random(n_particles) < p_detect
        k = int(seen.sum())
        s = se * np.exp(rng.normal(0.0, se_spread, k))
        obs = pos[seen] + rng.normal(0.0, s[:, None], (k, 2))
        rows.append((np.full(k, f), obs, s, np.flatnonzero(seen)))
    frame = np.concatenate([r[0] for r in rows])
    obs = np.concatenate([r[1] for r in rows])
    s = np.concatenate([r[2] for r in rows])
    who = np.concatenate([r[3] for r in rows])
    locs = pl.DataFrame({
        "loc_id": np.arange(len(frame), dtype=np.uint32),
        "frame": frame.astype(np.uint32),
        "y": obs[:, 0], "x": obs[:, 1],
        "se_y": s, "se_x": s,
    })
    return locs, who


def switch_rate(tracks, who):
    """Switches per 100 links: of the links reported, the share that join two
    different particles. Fragmenting a trajectory is recoverable; switching
    its identity is not."""
    tid = tracks["track_id"].to_numpy()
    n_links = wrong = 0
    for t in np.unique(tid):
        rows = np.flatnonzero(tid == t)
        n_links += len(rows) - 1
        wrong += int(np.sum(who[rows[:-1]] != who[rows[1:]]))
    return 100.0 * wrong / max(n_links, 1), n_links


def links(tracks):
    """Consecutive `loc_id` pairs within each track."""
    df = tracks.sort("track_id", "frame")
    tid, lid = df["track_id"].to_numpy(), df["loc_id"].to_numpy()
    same = tid[1:] == tid[:-1]
    return set(zip(lid[:-1][same], lid[1:][same]))


def test_link_returns_the_table_plus_one_column():
    locs, _ = movie(1)
    tracks = spotsolve.link(locs, MAX_STEP)
    assert tracks.columns == locs.columns + ["track_id"]
    assert tracks["track_id"].dtype == pl.UInt32
    assert tracks.drop("track_id").equals(locs)
    relinked = spotsolve.link(tracks, MAX_STEP)
    assert relinked.columns == tracks.columns
    assert relinked.equals(tracks)


def test_row_order_does_not_change_the_linking():
    locs, _ = movie(2)
    shuffled = locs.sample(fraction=1.0, shuffle=True, seed=7)
    assert links(spotsolve.link(locs, MAX_STEP)) == \
        links(spotsolve.link(shuffled, MAX_STEP))


def test_missing_columns_and_bad_steps_are_refused():
    locs, _ = movie(4)
    with pytest.raises(ValueError, match="missing"):
        spotsolve.link(locs.drop("y"), MAX_STEP)
    for bad in (0.0, -1.0, float("nan"), float("inf"), True, "far"):
        with pytest.raises(ValueError, match="max_step"):
            spotsolve.link(locs, bad)
    nan = locs.with_columns(
        pl.when(pl.col("loc_id") == 3).then(float("nan"))
        .otherwise(pl.col("x")).alias("x"))
    with pytest.raises(ValueError, match="non-finite"):
        spotsolve.link(nan, MAX_STEP)


def test_no_link_is_longer_than_max_step_and_a_missed_frame_ends_a_track():
    locs, _ = movie(5, p_detect=0.9)
    tracks = spotsolve.link(locs, MAX_STEP).sort("track_id", "frame")
    same = np.diff(tracks["track_id"].to_numpy()) == 0
    step = np.hypot(np.diff(tracks["y"].to_numpy()), np.diff(tracks["x"].to_numpy()))
    assert np.all(step[same] < MAX_STEP)
    assert np.all(np.diff(tracks["frame"].to_numpy().astype(int))[same] == 1)


# Measured when the thresholds were set (three seeds, MAX_STEP = 5.0 px,
# 30% immobile); floors carry about 30% headroom:
#
#   regime                    step/NN  switches/100   links/detection
#   sparse, every frame seen    0.11       0.57            0.950
#   GEM-like, 95% detected      0.31       3.48            0.906
#   dense, 90% detected         0.54      13.00            0.877
@pytest.mark.parametrize("name,n,field,p_detect,max_switch,min_links", [
    ("sparse", 80, 256.0, 1.0, 1.0, 0.92),
    ("gem_like", 160, 128.0, 0.95, 4.5, 0.88),
    ("dense", 280, 96.0, 0.90, 17.0, 0.85),
])
def test_switch_rate_on_known_trajectories(name, n, field, p_detect,
                                           max_switch, min_links):
    rates, dens = [], []
    for seed in (11, 12, 13):
        locs, who = movie(seed, n_particles=n, field=field, p_detect=p_detect)
        r, n_links = switch_rate(spotsolve.link(locs, MAX_STEP), who)
        rates.append(r)
        dens.append(n_links / len(locs))
    assert np.mean(rates) <= max_switch, f"{name}: {np.mean(rates):.2f} switches/100"
    assert np.mean(dens) >= min_links, f"{name}: {np.mean(dens):.2f} links/detection"
