"""Convert all fitted emitters to Polars tables without filtering rows.

Positions and errors use pixels, with derived micrometre columns. Flux and
background use ADU above offset. Optional quality cuts are post-processing.
"""

from numbers import Real

import numpy as np
import polars as pl


LOCALIZATION_SCHEMA = {
    "loc_id": pl.UInt32,      # unique over the whole movie; a stable handle
    "frame": pl.UInt32,       # 0-based index into the stack
    "t": pl.Float64,          # seconds since the first frame
    "y": pl.Float64,          # px, image coordinates (row)
    "x": pl.Float64,          # px, image coordinates (column)
    "y_um": pl.Float64,
    "x_um": pl.Float64,
    "se_y": pl.Float64,       # px, model-based standard error
    "se_x": pl.Float64,
    "se_pos": pl.Float64,     # px, hypot(se_y, se_x); one number for reports
    "se_y_um": pl.Float64,
    "se_x_um": pl.Float64,
    "flux": pl.Float64,       # ADU above offset, total, background-free
    "se_flux": pl.Float64,
    "flux_snr": pl.Float64,   # flux / se_flux
    "peak": pl.Float64,       # ADU, on-centre model pixel value; read vs `bg`
    "bg": pl.Float64,         # ADU/px, fitted background at the emitter pixel
    "sigma": pl.Float64,      # px, reference search width
    "fit_sigma": pl.Float64,  # px, this emitter's own fitted width
    "sigma_se": pl.Float64,
    "sigma_ratio": pl.Float64,  # fit_sigma / sigma
    "flags": pl.UInt8,
    "fisher_flux": pl.Float64,
    "fisher_y": pl.Float64,
    "fisher_x": pl.Float64,
    "fisher_sigma": pl.Float64,
}

FRAME_SCHEMA = {
    "frame": pl.UInt32,
    "t": pl.Float64,
    "n_locs": pl.UInt32,
    "n_flagged": pl.UInt32,
    "median_flux": pl.Float64,
    "median_se_pos": pl.Float64,
    "background": pl.Float64,         # median of the background surface
    "dispersion": pl.Float64,         # measured variance per unit signal, ADU
    "seconds": pl.Float64,            # wall clock for this frame's search
}


def _finite_median(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _sample_background(bmap, positions):
    """The background surface read at each emitter's own pixel."""
    if not len(positions):
        return np.empty(0)
    h, w = bmap.shape
    yi = np.clip(np.rint(positions[:, 0]).astype(int), 0, h - 1)
    xi = np.clip(np.rint(positions[:, 1]).astype(int), 0, w - 1)
    return bmap[yi, xi]


def frame_tables(result, frame, t=0.0, pixel_size=1.0,
                 seconds=float("nan"), loc_id0=0):
    """Return (localizations, frame summary), retaining every result row.

    `pixel_size` is micrometres per pixel, `t` is seconds, and `loc_id0`
    starts consecutive localization IDs. Unavailable diagnostics are NaN.
    """
    pos = np.asarray(result.positions, float).reshape(-1, 2)
    amp = np.asarray(result.amplitudes, float).ravel()
    se = np.asarray(result.se, float).reshape(-1, 3)
    n = len(amp)
    fraction = np.asarray(result.info.get("fisher_fraction", np.full((n, 4), np.nan)))
    background = result.info.get("fitted_background")
    if background is None:
        background = _sample_background(result.background, pos)
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
            "bg": np.asarray(background, float),
            "sigma": np.full(n, float(result.sigma)),
            "fit_sigma": np.asarray(result.fit_sigma, float),
            "sigma_ratio": np.asarray(result.sigma_ratio, float),
            "sigma_se": np.asarray(result.sigma_se, float),
            "flags": np.asarray(result.flags, np.uint8),
            **{f"fisher_{name}": fraction[:, j]
               for j, name in enumerate(("flux", "y", "x", "sigma"))},
        },
        schema=LOCALIZATION_SCHEMA,
    )

    row = pl.DataFrame(
        {
            "frame": [frame], "t": [t], "n_locs": [n],
            "n_flagged": [int(np.count_nonzero(result.flags))],
            "median_flux": [med],
            "median_se_pos": [_finite_median(np.hypot(se[:, 1], se[:, 2]))
                              if n else float("nan")],
            "background": [_finite_median(result.background)],
            "dispersion": [float(result.dispersion)],
            "seconds": [float(seconds)],
        },
        schema=FRAME_SCHEMA,
    )
    return locs, row


def concat(parts):
    """Stack per-frame tables, keeping the schema even when a frame is empty."""
    if not parts:
        raise ValueError("nothing to concatenate")
    return pl.concat(parts, how="vertical")


def filter_quality(locs, *, max_se_pos=None, min_flux_snr=None):
    """Keep usable coordinates, optionally requiring precision/flux support.

    Flags are left for the caller to inspect or select explicitly.
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
