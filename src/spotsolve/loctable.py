"""The standard localization table: `spotsolve` results as `polars` DataFrames.

`core.detect` answers a question about ONE image. Everything downstream --
trajectory linking first among them -- asks questions about a MOVIE, and needs
the per-frame results in one table with a fixed, documented set of columns.
This module defines that table and nothing else: no detection, no linking, no
plotting. It is the contract between the localizer and whatever consumes it.

Three tables come out of a run, because there are three different row units
and forcing them into one would mean either duplicating frame-level facts on
every detection or hiding them:

  `locs`        one row per detection          -- LOCALIZATION_SCHEMA
  `frames`      one row per frame              -- FRAME_SCHEMA
  `aggregates`  one row per flagged object     -- AGGREGATE_SCHEMA

Units
-----
Pixels and frames are CANONICAL; micrometres and seconds are derived columns,
present only because the linker's motion model lives in physical units and
converting in two places invites two conventions. `flux` is in
photoelectrons, as everywhere downstream of `detect` (README section 2) --
never ADU, and never a peak height.

What the linker actually needs from this table
----------------------------------------------
Position and frame are the obvious part. The part that is not obvious, and is
the reason a table from this localizer is worth more to a multiple-hypothesis
tracker than one from a centroid finder, is `se_y`/`se_x`: the per-detection
CRLB, which `refine` reaches (README section 12). A gate built on them is a
real Mahalanobis distance rather than a hand-tuned radius, and it is
*heteroscedastic* -- a dim emitter in a crowd carries a genuinely wider gate
than a bright isolated one, which is exactly the distinction that decides
hard assignments in a dense field. The PSF is isotropic and the position
block of the Fisher matrix is diagonal to the precision that matters here, so
the two numbers are the whole covariance.

`flux` and `se_flux` are the second linking cue: a real trajectory's flux is
continuous frame to frame, and `flux_snr` (= A/SE(A), the same statistic the
search's own acceptance veto uses) says how much to trust it.

Filtering
---------
`is_aggregate` is a FLAG, never a deletion. `filter_aggregates` returns the
filtered view and leaves the full table intact, because how much of a frame
went into aggregates is a fact about the frame that the tracker should be
able to read -- a movie whose flux is half aggregate is not a movie to report
diffusion coefficients from without saying so.
"""

import numpy as np
import polars as pl

from . import core

LOCALIZATION_SCHEMA = {
    "loc_id": pl.UInt32,      # unique over the whole movie; a stable handle
    "frame": pl.UInt32,       # 0-based index into the stack
    "t": pl.Float64,          # seconds since the first frame
    "y": pl.Float64,          # px, image coordinates (row)
    "x": pl.Float64,          # px, image coordinates (column)
    "y_um": pl.Float64,
    "x_um": pl.Float64,
    "se_y": pl.Float64,       # px, CRLB standard error
    "se_x": pl.Float64,
    "se_pos": pl.Float64,     # px, hypot(se_y, se_x); one number for reports
    "se_y_um": pl.Float64,
    "se_x_um": pl.Float64,
    "flux": pl.Float64,       # photoelectrons, total, background-free
    "se_flux": pl.Float64,
    "flux_snr": pl.Float64,   # flux / se_flux; the search's own A/SE statistic
    "bg": pl.Float64,         # photoelectrons/px, background surface here
    "sigma": pl.Float64,      # px, the PSF sigma the fit was held at
    "flux_ratio": pl.Float64,  # flux / this frame's median detection
    "is_aggregate": pl.Boolean,
}

FRAME_SCHEMA = {
    "frame": pl.UInt32,
    "t": pl.Float64,
    "n_locs": pl.UInt32,
    "n_aggregates": pl.UInt32,        # linked objects, not detections
    "n_locs_flagged": pl.UInt32,      # detections inside those objects
    "median_flux": pl.Float64,
    "agg_flux_fraction": pl.Float64,  # share of detected flux in aggregates
    "median_se_pos": pl.Float64,
    "background": pl.Float64,         # median of the background surface
    "lam": pl.Float64,                # fitted emitters per px^2
    "gain": pl.Float64,
    "n_rounds": pl.UInt32,
    "seconds": pl.Float64,            # wall clock for this frame's detect()
}

AGGREGATE_SCHEMA = {
    "frame": pl.UInt32,
    "t": pl.Float64,
    "y": pl.Float64,          # flux-weighted centroid of the linked detections
    "x": pl.Float64,
    "y_um": pl.Float64,
    "x_um": pl.Float64,
    "flux": pl.Float64,       # summed over the object's detections
    "ratio": pl.Float64,      # brightest member / the frame's median detection
    "n_locs": pl.UInt32,      # detections this object absorbed
}


def _sample_background(bmap, positions):
    """The background surface read at each emitter's own pixel."""
    if not len(positions):
        return np.empty(0)
    h, w = bmap.shape
    yi = np.clip(np.rint(positions[:, 0]).astype(int), 0, h - 1)
    xi = np.clip(np.rint(positions[:, 1]).astype(int), 0, w - 1)
    return bmap[yi, xi]


def frame_tables(result, frame, t=0.0, pixel_size=1.0, agg_ratio=None,
                 seconds=float("nan"), loc_id0=0):
    """One `DetectResult` -> its (locs, frame_row, aggregates) tables.

    `agg_ratio` is the over-bright cut as a multiple of THIS frame's median
    detection; `None` uses `core.AGG_AMP_RATIO`. Being relative to the
    frame's own median is what lets one number serve a whole movie whose
    illumination and bleaching drift -- see `core.flag_aggregates`.
    """
    ratio = core.AGG_AMP_RATIO if agg_ratio is None else float(agg_ratio)
    pos = np.atleast_2d(np.asarray(result.positions, float)).reshape(-1, 2)
    amp = np.asarray(result.amplitudes, float).ravel()
    n = len(amp)

    # `se` is (N,3) = (SE_A, SE_y, SE_x), or None if nothing was fitted.
    if result.se is not None and len(result.se) == n:
        se = np.asarray(result.se, float)
    else:
        se = np.full((n, 3), np.nan)

    mask, objs = core.flag_aggregates(result, ratio=ratio)
    med = float(np.median(amp)) if n else float("nan")
    ps = float(pixel_size)

    locs = pl.DataFrame(
        {
            "loc_id": np.arange(loc_id0, loc_id0 + n),
            "frame": np.full(n, frame),
            "t": np.full(n, t),
            "y": pos[:, 0], "x": pos[:, 1],
            "y_um": pos[:, 0] * ps, "x_um": pos[:, 1] * ps,
            "se_y": se[:, 1], "se_x": se[:, 2],
            "se_pos": np.hypot(se[:, 1], se[:, 2]),
            "se_y_um": se[:, 1] * ps, "se_x_um": se[:, 2] * ps,
            "flux": amp, "se_flux": se[:, 0],
            "flux_snr": np.divide(amp, se[:, 0],
                                  out=np.full(n, np.nan),
                                  where=np.isfinite(se[:, 0]) & (se[:, 0] > 0)),
            "bg": _sample_background(result.background, pos),
            "sigma": np.full(n, float(result.sigma)),
            "flux_ratio": amp / med if n else amp,
            "is_aggregate": mask,
        },
        schema=LOCALIZATION_SCHEMA,
    )

    aggs = pl.DataFrame(
        {
            "frame": np.full(len(objs), frame),
            "t": np.full(len(objs), t),
            "y": objs["y"], "x": objs["x"],
            "y_um": objs["y"] * ps, "x_um": objs["x"] * ps,
            "flux": objs["flux"], "ratio": objs["ratio"],
            "n_locs": objs["n"],
        },
        schema=AGGREGATE_SCHEMA,
    )

    flux_total = float(amp.sum())
    row = pl.DataFrame(
        {
            "frame": [frame], "t": [t], "n_locs": [n],
            "n_aggregates": [len(objs)],
            "n_locs_flagged": [int(mask.sum())],
            "median_flux": [med],
            "agg_flux_fraction": [float(amp[mask].sum()) / flux_total
                                  if flux_total > 0 and mask.any() else 0.0],
            "median_se_pos": [float(np.nanmedian(np.hypot(se[:, 1], se[:, 2])))
                              if n else float("nan")],
            "background": [float(np.median(result.background))],
            "lam": [float(result.lam)], "gain": [float(result.gain)],
            "n_rounds": [int(result.n_outer_passes)],
            "seconds": [float(seconds)],
        },
        schema=FRAME_SCHEMA,
    )
    return locs, row, aggs


def concat(parts):
    """Stack per-frame tables, keeping the schema even when a frame is empty."""
    if not parts:
        raise ValueError("nothing to concatenate")
    return pl.concat(parts, how="vertical")


def filter_aggregates(locs, keep_flagged=False):
    """The point-emitter localizations, with over-bright detections removed.

    A view, not a mutation: `locs` still holds everything, which is what makes
    `agg_flux_fraction` auditable afterwards.
    """
    return locs if keep_flagged else locs.filter(~pl.col("is_aggregate"))


def link_input(locs, units="um"):
    """The minimal columns a tracker needs, in one consistent unit system.

    Returned as `loc_id, frame, t, y, x, se_y, se_x, flux, se_flux` with the
    position columns in `units` ("um" or "px"). Kept deliberately narrow: a
    linker that reads only these cannot accidentally come to depend on a
    detector-internal column, so the two stay separable.
    """
    if units == "um":
        cols = [pl.col("y_um").alias("y"), pl.col("x_um").alias("x"),
                pl.col("se_y_um").alias("se_y"),
                pl.col("se_x_um").alias("se_x")]
    elif units == "px":
        cols = [pl.col("y"), pl.col("x"), pl.col("se_y"), pl.col("se_x")]
    else:
        raise ValueError(f"units must be 'um' or 'px', got {units!r}")
    return locs.select([pl.col("loc_id"), pl.col("frame"), pl.col("t")]
                       + cols + [pl.col("flux"), pl.col("se_flux")])
