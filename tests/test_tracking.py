"""The linker: its table contract, and what it recovers from known truth.

The assignment's exactness is checked against enumeration in `lap.rs`, and
what the linker reads from the movie (the detector gap, the continuation
fraction, the density) against truth in `track.rs`. Checked here: the
boundary Python owns (columns in, one column out, row order), and accuracy
on simulated movies that mix mobilities. For each mobility class, at the
recall the linker reaches, its identity switches may not exceed those of
Crocker & Grier's least squared displacement at the radius best for that
class.
"""

import numpy as np
import polars as pl
import pytest
from scipy.optimize import linear_sum_assignment

import spotsolve

pytest.importorskip("spotsolve_rs")

# The GEM regime: 104 nm per px and 22 ms per frame, so 1 um^2/s is
# 2.03 px^2/frame, and 29 nm of localization error is 0.28 px.
UM2S = 0.61 / 0.3
SE_PX = 0.28
CLASSES = ("immobile", "slow", "fast")
# Nine confining discs of radius 28 px on a 192 px field.
CELL_R = 28.0
CELL_C = np.array([(y, x) for y in (32.0, 96.0, 160.0) for x in (32.0, 96.0, 160.0)])
GEM_LIKE = 160 / 128.0**2
DENSE = 280 / 96.0**2


def rstep(d):
    """Three times the rms 2-D step at D = d px^2/frame."""
    return 3 * np.sqrt(4 * d + 4 * SE_PX**2)


def movie(seed, density, d_fast_um, n_frames=100, gap=2.5):
    """Particles confined to the discs, `density` per px^2 of disc: 30%
    immobile, 50% at 0.3 um^2/s, 20% at `d_fast_um`. Each is seen with its
    own probability, uniform on (0.4, 0.99), and same-frame detections
    closer than `gap` are reported as one, as a detector would.
    -> (locs, class per row, the particles each row stands for)"""
    rng = np.random.default_rng(seed)
    n = int(round(density * len(CELL_C) * np.pi * CELL_R**2))
    cls = rng.choice(3, n, p=(0.3, 0.5, 0.2))
    d = np.array([0.0, 0.3 * UM2S, d_fast_um * UM2S])[cls]
    p_seen = rng.uniform(0.4, 0.99, n)
    home = CELL_C[rng.integers(len(CELL_C), size=n)]
    r, th = CELL_R * np.sqrt(rng.random(n)), 2 * np.pi * rng.random(n)
    pos = home + np.c_[r * np.cos(th), r * np.sin(th)]
    frames, obs, errs, who, members = [], [], [], [], []
    for f in range(n_frames):
        if f:
            pos = pos + rng.normal(0.0, np.sqrt(2 * d)[:, None], pos.shape)
            q = pos - home
            rr = np.hypot(q[:, 0], q[:, 1])
            out = rr > CELL_R
            back = np.minimum(np.maximum(2 * CELL_R - rr[out], 0.0), CELL_R)
            pos[out] = home[out] + q[out] * (back / rr[out])[:, None]
        seen = np.flatnonzero(rng.random(n) < p_seen)
        s = SE_PX * np.exp(rng.normal(0.0, 0.3, len(seen)))
        o = pos[seen] + rng.normal(0.0, s[:, None], (len(seen), 2))
        keep, merged = [], {}
        for k in rng.permutation(len(seen)):
            if keep:
                dist = np.hypot(*(o[keep] - o[k]).T)
                j = int(np.argmin(dist))
                if dist[j] < gap:
                    merged.setdefault(keep[j], []).append(seen[k])
                    continue
            keep.append(k)
        keep = sorted(keep)
        frames.append(np.full(len(keep), f))
        obs.append(o[keep])
        errs.append(s[keep])
        who.append(seen[keep])
        members.extend(frozenset([seen[k], *merged.get(k, [])]) for k in keep)
    frame, xy, se, who = (np.concatenate(v) for v in (frames, obs, errs, who))
    locs = pl.DataFrame({
        "loc_id": np.arange(len(frame), dtype=np.uint32),
        "frame": frame.astype(np.uint32),
        "y": xy[:, 0], "x": xy[:, 1],
        "se_y": se, "se_x": se,
    })
    return locs, cls[who], members


def crocker_grier(locs, r):
    """Least squared displacement per frame pair, each track end costing
    r^2: the reference linker. -> track id per row."""
    frame = locs["frame"].to_numpy().astype(int)
    pos = np.c_[locs["y"].to_numpy(), locs["x"].to_numpy()]
    track = -np.ones(len(frame), int)
    nt, prev = 0, None
    for f in range(frame.max() + 1):
        cur = np.flatnonzero(frame == f)
        if prev is not None and len(prev) and len(cur):
            d2 = ((pos[prev][:, None] - pos[cur][None]) ** 2).sum(-1)
            gain = np.where(d2 < r * r, r * r - d2, 0.0)
            for i, j in zip(*linear_sum_assignment(-gain)):
                if gain[i, j] > 0:
                    track[cur[j]] = track[prev[i]]
        for j in cur:
            if track[j] < 0:
                track[j] = nt
                nt += 1
        prev = cur
    return track


def score(locs, track, cls, members):
    """Per class: (recall, switches per 100 links). A link is right when its
    two detections share a particle; recall counts each particle seen in
    consecutive frames once."""
    frame = locs["frame"].to_numpy().astype(int)
    track = np.asarray(track)
    at = {}
    for i, (f, ws) in enumerate(zip(frame.tolist(), members)):
        for w in ws:
            at[(f, w)] = i
    nxt = np.full(len(frame), -1)
    o = np.lexsort((frame, track))
    same = track[o][1:] == track[o][:-1]
    nxt[o[:-1][same]] = o[1:][same]
    out = {}
    for c, name in enumerate(CLASSES):
        rows = np.flatnonzero(cls == c)
        true = right = linked = wrong = 0
        for i in rows:
            ws, n = members[i], nxt[i]
            has_next = any((frame[i] + 1, w) in at for w in ws)
            ok = n >= 0 and not ws.isdisjoint(members[n])
            true += has_next
            right += has_next and ok
            linked += n >= 0
            wrong += n >= 0 and not ok
        out[name] = (right / max(true, 1), 100.0 * wrong / max(linked, 1))
    return out


def frontier(points):
    """Least switches the reference reaches at recall >= r, interpolated
    along its radius curve; past its highest recall, its value there."""
    pts = sorted(points)
    top = pts[-1][0]

    def f(r):
        r = min(r, top)
        best = min([s for rr, s in pts if rr >= r])
        for (r0, s0), (r1, s1) in zip(pts[:-1], pts[1:]):
            if r0 <= r <= r1 and r1 > r0:
                best = min(best, s0 + (s1 - s0) * (r - r0) / (r1 - r0))
        return best
    return f


def test_link_returns_the_table_plus_one_column():
    locs, _, _ = movie(1, GEM_LIKE, 2.5, n_frames=15)
    tracks = spotsolve.link(locs, 13.6)
    assert tracks.columns == locs.columns + ["track_id"]
    assert tracks["track_id"].dtype == pl.UInt32
    assert tracks.drop("track_id").equals(locs)
    relinked = spotsolve.link(tracks, 13.6)
    assert relinked.columns == tracks.columns
    assert relinked.equals(tracks)


def test_row_order_does_not_change_the_linking():
    locs, _, _ = movie(2, GEM_LIKE, 2.5, n_frames=15)

    def links(t):
        t = t.sort("track_id", "frame")
        tid, lid = t["track_id"].to_numpy(), t["loc_id"].to_numpy()
        same = tid[1:] == tid[:-1]
        return set(zip(lid[:-1][same], lid[1:][same]))

    shuffled = locs.sample(fraction=1.0, shuffle=True, seed=7)
    assert links(spotsolve.link(locs, 13.6)) == links(spotsolve.link(shuffled, 13.6))


def test_missing_columns_and_bad_inputs_are_refused():
    locs, _, _ = movie(4, GEM_LIKE, 2.5, n_frames=5)
    for column in ("y", "se_x"):
        with pytest.raises(ValueError, match="missing"):
            spotsolve.link(locs.drop(column), 13.6)
    for bad in (0.0, -1.0, float("nan"), float("inf"), True, "far"):
        with pytest.raises(ValueError, match="max_step"):
            spotsolve.link(locs, bad)
    for column, value, message in (("x", float("nan"), "non-finite"),
                                   ("se_y", 0.0, "localization error")):
        bad = locs.with_columns(
            pl.when(pl.col("loc_id") == 3).then(value)
            .otherwise(pl.col(column)).alias(column))
        with pytest.raises(ValueError, match=message):
            spotsolve.link(bad, 13.6)


def test_no_link_is_longer_than_max_step_and_a_missed_frame_ends_a_track():
    locs, _, _ = movie(5, GEM_LIKE, 2.5, n_frames=30)
    tracks = spotsolve.link(locs, 13.6).sort("track_id", "frame")
    same = np.diff(tracks["track_id"].to_numpy()) == 0
    step = np.hypot(np.diff(tracks["y"].to_numpy()), np.diff(tracks["x"].to_numpy()))
    assert np.all(step[same] < 13.6)
    assert np.all(np.diff(tracks["frame"].to_numpy().astype(int))[same] == 1)


RADII = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.5, 10.0, 12.0, 14.0)


@pytest.mark.parametrize("density,d_fast_um", [(GEM_LIKE, 2.5), (DENSE, 1.0)])
def test_no_class_switches_more_than_crocker_grier_at_its_best_radius(density, d_fast_um):
    """Per seed, the linker's switches at its own recall minus the
    reference's least at that recall, for each class; the mean over seeds
    may not be positive."""
    max_step = rstep(d_fast_um * UM2S)
    excess = {k: [] for k in CLASSES}
    for seed in range(300, 309):
        locs, cls, members = movie(seed, density, d_fast_um)
        ref = [score(locs, crocker_grier(locs, r), cls, members) for r in RADII if r <= max_step]
        ours = score(locs, spotsolve.link(locs, max_step)["track_id"].to_numpy(), cls, members)
        for k in CLASSES:
            f = frontier([s[k] for s in ref])
            excess[k].append(ours[k][1] - f(ours[k][0]))
    for k in CLASSES:
        assert np.mean(excess[k]) <= 0.0, f"{k}: {np.round(excess[k], 2)} switches/100 over the reference"
