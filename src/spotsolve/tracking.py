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


def _frame_numbers(column):
    """The `frame` column as int64, refused unless every row holds a whole
    number within 64 bits."""
    frame = column.to_numpy() if column.dtype.is_numeric() and not column.null_count() else None
    whole = frame is not None and (np.issubdtype(frame.dtype, np.integer)
                                   or bool(np.all(np.isfinite(frame) & (frame == np.round(frame)))))
    if not whole or (frame.size and not -2**63 <= frame.min() <= frame.max() < 2**63):
        raise ValueError("`frame` must hold a whole number, within 64 bits, on every row")
    return np.ascontiguousarray(frame, dtype=np.int64)


def link(locs, max_step, strength="detect_z", link_p=True):
    """Link a localization table into trajectories. -> the table + `track_id`
    (and `link_p`).

    `max_step` is the largest distance, in the table's position units
    (pixels for `loctable` output), a particle may move between consecutive
    frames: about three times the rms step of the fastest particles of
    interest. A larger value costs slow particles little, because each
    track is held to its own step size. `se_y` and `se_x` are each
    detection's localization errors in the same units, and `frame` a whole
    number on every row.

    `strength` names a column of each detection's strength, the detector's
    `detect_z` by default. When the table has it, the chance that a
    detection is seen again in the next frame is read from the movie per
    strength class rather than once for the whole movie, and a track's end
    is priced by its own detection's class and the new detection's: a
    bright in-focus particle is seen again almost surely, a dim one near the
    detection bar often is not. `None`, or a table without the column, reads
    one continuation for the movie.

    With `link_p`, each row also gets the probability that the link into
    it, from the previous row of its track, joins one particle's
    detections: null where a track starts. It annotates the links and never
    changes them; filter or weight steps by it downstream. On simulated
    movies its mean matches the share of links that are right to within a
    point (docs/TRACKING.md).

    The result is the input with a `UInt32` `track_id` column (and a
    `Float64` `link_p` column) added, in the input's row order; columns of
    those names already present are replaced. Every row belongs to a track,
    including single-frame ones.
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
    frame = _frame_numbers(locs["frame"])
    pos = np.ascontiguousarray(
        np.stack([locs["y"].to_numpy(), locs["x"].to_numpy()], axis=1), dtype=float)
    err = np.ascontiguousarray(
        np.stack([locs["se_y"].to_numpy(), locs["se_x"].to_numpy()], axis=1), dtype=float)
    z = None
    if strength is not None and strength in locs.columns:
        z = np.ascontiguousarray(locs[strength].cast(pl.Float64).to_numpy(), dtype=float)
    ids, p = _rs.track_link(frame, pos, err, max_step, z, link_p=True)
    cols = [pl.Series("track_id", ids, dtype=pl.UInt32)]
    if link_p:
        cols.append(pl.Series("link_p", p, dtype=pl.Float64, nan_to_null=True))
    return locs.drop([c for c in ("track_id", "link_p") if c in locs.columns]).with_columns(cols)
