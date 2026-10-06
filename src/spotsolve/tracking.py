"""Link localization tables frame to frame, preserving all input rows.

Each track predicts its next step from its own diffusion coefficient,
inferred from its steps so far, so immobile, slow and fast particles in one
movie are linked on their own scales. No step longer than `max_step` is
linked; a missed frame ends a track. See docs/TRACKING.md for the model and
its validation.
"""

import numpy as np

try:
    import spotsolve_rs as _rs
except ImportError as error:          # pragma: no cover - build problem
    raise ImportError(
        "spotsolve needs its bundled Rust extension; reinstall a compatible wheel "
        "or run `maturin develop --release` from the repository root") from error

__all__ = ["link", "LINK_COLUMNS"]

LINK_COLUMNS = ("frame", "y", "x", "se_y", "se_x")
"""Columns the linker reads."""


def link(locs, max_step):
    """Link a localization table into trajectories. -> the table + `track_id`.

    `max_step` is the largest distance, in the table's position units
    (pixels for `loctable` output), a particle may move between consecutive
    frames: about three times the rms step of the fastest particles of
    interest. A larger value costs slow particles little, because each
    track is held to its own step size. `se_y` and `se_x` are each
    detection's localization errors in the same units.

    The result is the input with one `UInt32` column added, in the input's
    row order; a `track_id` column already present is replaced. Every row
    belongs to a track, including single-frame ones.
    """
    import polars as pl

    missing = [c for c in LINK_COLUMNS if c not in locs.columns]
    if missing:
        raise ValueError(f"the localization table is missing {missing}; "
                         f"linking needs {list(LINK_COLUMNS)}")
    if isinstance(max_step, (bool, np.bool_)):
        raise ValueError("max_step must be positive and finite")
    try:
        max_step = float(max_step)
    except (TypeError, ValueError) as error:
        raise ValueError("max_step must be positive and finite") from error
    frame = np.ascontiguousarray(locs["frame"].to_numpy(), dtype=np.int64)
    pos = np.ascontiguousarray(
        np.stack([locs["y"].to_numpy(), locs["x"].to_numpy()], axis=1), dtype=float)
    err = np.ascontiguousarray(
        np.stack([locs["se_y"].to_numpy(), locs["se_x"].to_numpy()], axis=1), dtype=float)
    ids = _rs.track_link(frame, pos, err, max_step)
    return locs.drop([c for c in ("track_id",) if c in locs.columns]).with_columns(
        pl.Series("track_id", ids, dtype=pl.UInt32))
