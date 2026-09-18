"""Convert per-frame results to Polars localization and summary tables.

`LOCALIZATION_SCHEMA` has one row per detection; `FRAME_SCHEMA` one per
frame; `AGGREGATE_SCHEMA` one per linked bright object; `REJECT_SCHEMA` one
per fit outside the reporting band.

Coordinates and errors use pixels, with derived micrometre columns for
tracking. Flux is total signal in ADU above the camera offset; divide by
gain for photoelectrons. `peak` is the model's central-pixel signal above
background, derived from flux and fitted width.

The linker uses position errors and optionally flux errors. Tables retain
only marginal position errors, not the full joint-fit covariance.
Aggregate flags preserve all rows; `filter_aggregates` returns a filtered
table. `filter_quality` excludes unusable coordinates and optionally applies
precision/flux-significance cuts without adding columns.
"""

from numbers import Real

import numpy as np
import polars as pl

from . import aggregates, psf
from .results import REJECT_DTYPE

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
    "flux": pl.Float64,       # ADU above offset, total, background-free
    "se_flux": pl.Float64,
    "flux_snr": pl.Float64,   # flux / se_flux
    "peak": pl.Float64,       # ADU, on-centre model pixel value; read vs `bg`
    "bg": pl.Float64,         # ADU/px above offset, background surface here
    "sigma": pl.Float64,      # px, the in-focus PSF sigma the search ran at
    "fit_sigma": pl.Float64,  # px, this emitter's own fitted width
    "sigma_ratio": pl.Float64,  # fit_sigma / sigma; a per-emitter defocus readout
    "flux_ratio": pl.Float64,  # flux / this frame's median detection
    "is_aggregate": pl.Boolean,
}

FRAME_SCHEMA = {
    "frame": pl.UInt32,
    "t": pl.Float64,
    "n_locs": pl.UInt32,
    "n_too_narrow": pl.UInt32,
    "n_too_wide": pl.UInt32,
    "n_edge": pl.UInt32,
    "n_aggregates": pl.UInt32,        # linked objects, not detections
    "n_locs_flagged": pl.UInt32,      # detections inside those objects
    "median_flux": pl.Float64,
    "agg_flux_fraction": pl.Float64,  # share of detected flux in aggregates
    "median_se_pos": pl.Float64,
    "background": pl.Float64,         # median of the background surface
    "dispersion": pl.Float64,         # measured variance per unit signal, ADU
    "seconds": pl.Float64,            # wall clock for this frame's search
}

REJECT_SCHEMA = {
    "frame": pl.UInt32,
    "t": pl.Float64,
    "y": pl.Float64,
    "x": pl.Float64,
    "y_um": pl.Float64,
    "x_um": pl.Float64,
    "flux": pl.Float64,
    "peak": pl.Float64,
    "fit_sigma": pl.Float64,
    "sigma_ratio": pl.Float64,
    "reason": pl.String,
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
    """Return (localizations, frame summary, aggregates) as DataFrames.

    `agg_ratio` is the flux threshold relative to the frame's median detection;
    None uses `aggregates.AGG_AMP_RATIO`. `pixel_size` is micrometres per pixel,
    `t` is seconds, and `loc_id0` starts the consecutive localization IDs.
    """
    ratio = aggregates.AGG_AMP_RATIO if agg_ratio is None else float(agg_ratio)
    pos = np.asarray(result.positions, float).reshape(-1, 2)
    amp = np.asarray(result.amplitudes, float).ravel()
    se = np.asarray(result.se, float).reshape(-1, 3)
    n = len(amp)
    mask, objs = aggregates.flag_aggregates(result, ratio=ratio)
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
            "peak": np.asarray(result.peak, float),
            "bg": _sample_background(result.background, pos),
            "sigma": np.full(n, float(result.sigma)),
            "fit_sigma": np.asarray(result.fit_sigma, float),
            "sigma_ratio": np.asarray(result.sigma_ratio, float),
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
    reason = result.rejects["reason"]
    row = pl.DataFrame(
        {
            "frame": [frame], "t": [t], "n_locs": [n],
            "n_too_narrow": [int(np.sum(reason == "too_narrow"))],
            "n_too_wide": [int(np.sum(reason == "too_wide"))],
            "n_edge": [int(np.sum(reason == "edge"))],
            "n_aggregates": [len(objs)],
            "n_locs_flagged": [int(mask.sum())],
            "median_flux": [med],
            "agg_flux_fraction": [float(amp[mask].sum()) / flux_total
                                  if flux_total > 0 and mask.any() else 0.0],
            "median_se_pos": [float(np.nanmedian(np.hypot(se[:, 1], se[:, 2])))
                              if n else float("nan")],
            "background": [float(np.median(result.background))],
            "dispersion": [float(result.dispersion)],
            "seconds": [float(seconds)],
        },
        schema=FRAME_SCHEMA,
    )
    return locs, row, aggs


def reject_table(result, frame, t=0.0, pixel_size=1.0):
    """The fits outside the reporting band, as a table."""
    rec = result.rejects
    ps = float(pixel_size)
    return pl.DataFrame(
        {
            "frame": np.full(len(rec), frame),
            "t": np.full(len(rec), t),
            "y": rec["y"], "x": rec["x"],
            "y_um": rec["y"] * ps, "x_um": rec["x"] * ps,
            "flux": rec["flux"],
            "peak": rec["flux"] * psf.peak_factor(rec["sigma"]),
            "fit_sigma": rec["sigma"],
            "sigma_ratio": rec["sigma_ratio"],
            "reason": rec["reason"],
        },
        schema=REJECT_SCHEMA,
    )


def concat(parts):
    """Stack per-frame tables, keeping the schema even when a frame is empty."""
    if not parts:
        raise ValueError("nothing to concatenate")
    return pl.concat(parts, how="vertical")


def filter_aggregates(locs, keep_flagged=False):
    """Return rows without aggregate flags, or all rows if `keep_flagged`."""
    return locs if keep_flagged else locs.filter(~pl.col("is_aggregate"))


def filter_quality(locs, *, max_se_pos=None, min_flux_snr=None):
    """Keep usable coordinates, optionally requiring precision/flux support.

    Always require finite x/y and finite, positive se_x/se_y. `max_se_pos`
    limits hypot(se_y, se_x) in the table's coordinate units (pixels for
    `frame_tables`). `min_flux_snr` optionally requires positive finite flux
    and se_flux and a flux/se_flux ratio at least this large. Both cutoffs
    must be positive and finite; neither has a calibrated universal default.

    Derived values are computed from the base columns, so stale `se_pos` or
    `flux_snr` columns cannot affect filtering. No columns are added or changed;
    retained rows keep their order and IDs. Keep the original table to inspect
    rejected detections, and filter before linking: removing a row may end a
    trajectory. This is a usability filter, not a probability of being real.
    """
    for name, value in (("max_se_pos", max_se_pos), ("min_flux_snr", min_flux_snr)):
        if value is not None:
            try:
                valid = (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                         and np.isfinite(float(value)) and value > 0)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be positive and finite")

    keep = pl.all_horizontal(
        *[pl.col(c).is_finite() for c in ("y", "x", "se_y", "se_x")],
        pl.col("se_y") > 0, pl.col("se_x") > 0,
    )
    if max_se_pos is not None:
        # hypot handles extreme finite values without overflow/underflow.
        precision = pl.Series(np.hypot(locs["se_y"].to_numpy(), locs["se_x"].to_numpy()))
        keep &= precision <= float(max_se_pos)
    if min_flux_snr is not None:
        keep &= pl.all_horizontal(
            pl.col("flux").is_finite(), pl.col("se_flux").is_finite(),
            pl.col("flux") > 0, pl.col("se_flux") > 0,
            pl.col("flux") / pl.col("se_flux") >= float(min_flux_snr),
        )
    return locs.filter(keep.fill_null(False))


def link_input(locs, units="um"):
    """Select loc_id, frame, t, y, x, se_y, se_x, flux and se_flux.

    Positions and their errors use `units` ("um" or "px").
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
