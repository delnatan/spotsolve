"""The linker: its table contract, and what it recovers from known truth.

The row-for-row comparison against `tracksolve`, the Python reference this was
ported from, lives in `rust/spotsolve-core/tests/layer6_track.rs` against the
frozen fixture `tests/fixtures/08_track.json`. What is checked here is the
boundary Python owns -- which columns go in, what comes out, and in what order
-- plus an end-to-end accuracy floor on simulated trajectories, so a
regression in the native code shows up as a worse answer and not merely as a
changed one.
"""

import numpy as np
import polars as pl
import pytest

import spotsolve

pytest.importorskip("spotsolve_rs")

# The GEM dataset's regime, in the canonical units: D = 0.3 um^2/s at 104 nm
# per px and 22 ms per frame is 0.61 px^2/frame, and 29 nm of localization
# error is 0.28 px.
D_PX = 0.61
SE_PX = 0.28


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
        "flux": np.full(len(frame), 1000.0),
    })
    return locs, who


def switch_rate(tracks, who):
    """Switches per 100 links: of the links reported, the share that join two
    different particles. Identity, not track count, is the thing that matters
    -- fragmenting a trajectory is recoverable and switching it is not."""
    tid = tracks["track_id"].to_numpy()
    n_links = wrong = 0
    for t in np.unique(tid):
        rows = np.flatnonzero(tid == t)
        n_links += len(rows) - 1
        wrong += int(np.sum(who[rows[:-1]] != who[rows[1:]]))
    return 100.0 * wrong / max(n_links, 1), n_links


def test_link_returns_the_table_plus_one_column():
    locs, _ = movie(1)
    tracks = spotsolve.link(locs)
    assert tracks.columns == locs.columns + ["track_id"]
    assert tracks["track_id"].dtype == pl.UInt32
    assert tracks.drop("track_id").equals(locs)          # row order preserved
    # Every detection lands in some track; one that never links is length 1.
    assert len(tracks) == len(locs)
    assert tracks["track_id"].n_unique() <= len(locs)


def test_row_order_does_not_change_the_linking():
    locs, who = movie(2)
    a = spotsolve.link(locs)
    shuffled = locs.sample(fraction=1.0, shuffle=True, seed=7)
    b = spotsolve.link(shuffled).sort("loc_id")
    # Track ids are assigned in frame order, so they are only equal up to
    # relabelling; the partition into trajectories must be identical.
    def pairs(df):
        tid, lid = df["track_id"].to_numpy(), df["loc_id"].to_numpy()
        out = set()
        for t in np.unique(tid):
            r = lid[tid == t]
            r.sort()
            out |= set(zip(r[:-1], r[1:]))
        return out
    assert pairs(a) == pairs(b)


def test_a_fitted_parameter_set_can_be_reused():
    locs, _ = movie(3)
    p = spotsolve.fit_link_params(locs)
    assert spotsolve.link(locs, p).equals(spotsolve.link(locs))
    assert [s["label"] for s in p.trajectory] == [
        "initialize", "iter1", "iter2", "iter3"]
    assert p.d_grid[0] == 0.0 and np.all(np.diff(p.d_grid) > 0)
    assert np.isclose(np.exp(p.d_logprior).sum(), 1.0)


def test_missing_columns_and_bad_errors_are_refused():
    locs, _ = movie(4)
    with pytest.raises(ValueError, match="missing"):
        spotsolve.link(locs.drop("se_y"))
    nan = locs.with_columns(
        pl.when(pl.col("loc_id") == 3).then(float("nan"))
        .otherwise(pl.col("se_y")).alias("se_y"))
    with pytest.raises(ValueError, match="finite and positive"):
        spotsolve.link(nan)


def test_flux_is_not_read():
    """Intensity is real evidence about identity and is deliberately unused:
    it is also what merge/split inference needs, and a partial dependency now
    would make the two harder to separate later."""
    locs, _ = movie(5)
    scrambled = locs.with_columns(
        pl.col("flux").shuffle(seed=1), pl.col("loc_id").alias("loc_id"))
    assert spotsolve.link(scrambled)["track_id"].to_list() == \
        spotsolve.link(locs)["track_id"].to_list()


# Measured at the port (2026-09-11), seeds 11-13, with the parameters fitted
# per movie. The linker is exact given its scores, so these move only if the
# model or the fit changes; the thresholds carry about 30% headroom over the
# measurement so a different BLAS or a new seed cannot fail them on noise.
#
#   regime                    step/NN  switches/100 (range)   links/detection
#   sparse, every frame seen    0.11      0.26 (0.00-0.53)         0.950
#   GEM-like, 95% detected      0.31      3.06 (2.39-3.50)         0.906
#   dense, 90% detected         0.54     10.77 (10.62-10.99)       0.866
#
# The ratio is what governs the difficulty: at 0.11 the nearest detection in
# the next frame is almost always the right one, and by 0.55 it often is not.
# The real GEM movie sits near 0.3.
@pytest.mark.parametrize("name,n,field,p_detect,max_switch,min_links", [
    ("sparse", 80, 256.0, 1.0, 1.0, 0.93),
    ("gem_like", 160, 128.0, 0.95, 4.0, 0.88),
    ("dense", 280, 96.0, 0.90, 13.0, 0.84),
])
def test_switch_rate_on_known_trajectories(name, n, field, p_detect,
                                           max_switch, min_links):
    rates, dens = [], []
    for seed in (11, 12, 13):
        locs, who = movie(seed, n_particles=n, field=field, p_detect=p_detect)
        tracks = spotsolve.link(locs)
        r, n_links = switch_rate(tracks, who)
        rates.append(r)
        dens.append(n_links / len(locs))
    assert np.mean(rates) <= max_switch, f"{name}: {np.mean(rates):.2f} switches/100"
    assert np.mean(dens) >= min_links, f"{name}: {np.mean(dens):.2f} links/detection"


def test_the_fit_recovers_the_motion_it_was_shown():
    """No dials: D, the immobile fraction and the continuation probability
    are estimated from the data, so they have to come back."""
    locs, _ = movie(21, n_particles=200, n_frames=25, field=160.0,
                    p_detect=0.9, immobile=0.3)
    p = spotsolve.fit_link_params(locs)
    # Mean D over the population is 0.7 * D_PX; the grid puts sub-floor
    # mobility on points near zero, so compare the mean, not the zero weight.
    assert 0.5 * 0.7 * D_PX < p.d_mean < 1.6 * 0.7 * D_PX, p
    assert 0.80 < p.p_cont < 0.95, p
    assert p.se_inflate < 1.6, p
