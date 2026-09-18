"""Linking localizations into trajectories: the table in, `track_id` out.

`localize` answers a question about one image. This module answers the one
question a MOVIE asks that a single frame cannot: which detections are the
same particle. It reads the standard localization table (`loctable`) and
returns it with one column added, so the result is a `polars` DataFrame with
`frame` and `track_id` -- napari's convention for a Tracks layer -- and every
detector column still beside it.

    import polars as pl
    import spotsolve
    from spotsolve import loctable, tracking

    locs = loctable.concat([...])                      # one row per detection
    locs = loctable.filter_aggregates(locs)            # point emitters only
    tracks = tracking.link(locs)
    tracks.select("track_id", "frame", "y", "x")

Frame-to-frame linking only -- stage 1 of Jaqaman et al. (2008, *Nat. Methods*
5:695). A missed detection ENDS a track: fragmenting a trajectory is a safe
failure and switching its identity is not, so gaps are left for a later stage
to close rather than guessed at here. Every detection ends up in some track,
and one that never links is a track of length 1.

What it reads, and what it does not
-----------------------------------
`frame`, `y`, `x`, `se_y`, `se_x`, in pixels and frames. The errors are the
point of it: they are spotsolve's per-detection CRLB, so the gate around a
bright, precisely localized spot is genuinely tighter than the one around a
dim one, and the cost of a link is a likelihood ratio rather than a distance.
The measurements behind that choice sit in `rust/spotsolve-core/src/track.rs`.

`flux` and `se_flux` are read only with `link(locs, brightness=True)`. Then
each track also carries its log-flux level, and a link is scored on how well
a detection's brightness matches it, so a bright particle keeps its identity
among dimmer, faster ones: spiked into real GEM frames, a mobile bright spot
was taken over by a dim neighbour half as often (56 -> 29 steals). It is not
the default because real GEM flux flickers by a factor of about 2 per frame,
and on that movie about 2% more tracks end as single detections, without
truth to say whether those breaks are right. The measurement sits in
`rust/spotsolve-core/src/track.rs` at `FluxModel`.

`is_aggregate` is not read either: this links what it is given. Aggregates are
flagged by `loctable`, not deleted, and whether to drop them is the caller's
decision -- `loctable.filter_aggregates(locs)` before linking.

Units
-----
Pixels and frames throughout, which is what the table holds. `D` therefore
comes back in px^2/frame; multiply by `pixel_size**2 / dt` for um^2/s. The
scores are log likelihood ratios, which are invariant to the choice of units,
so this produces the same links either way -- the conversion is a reporting
convenience and never enters a decision.

Motion parameters and acceptance
--------------------------------
Motion-model parameters are estimated from the data by `fit_link_params`, which `link`
calls for you if you do not pass one. Fit once and reuse it when you are
linking several movies of the same sample, or to read what was estimated:
`params.trajectory` records the empirical-Bayes loop iterate by iterate, so a
fit that goes wrong says so rather than quietly returning a worse answer.

Acceptance is a separate choice: `min_link_margin` can end tracks at ambiguous
assignments, and `min_track_length` flags segments with enough consecutive
frames for analysis. Neither supplies D or changes the population fit. Short
segments remain in the output. A margin is a score difference, not a calibrated
probability of a correct link; `diagnostics=True` exposes it for inspection.
"""

from dataclasses import dataclass, field

import numpy as np

try:
    import spotsolve_rs as _rs
except ImportError as error:          # pragma: no cover - build problem
    raise ImportError(
        "spotsolve needs its Rust extension; build it with `maturin develop "
        "--release -m rust/spotsolve-py/Cargo.toml`") from error

__all__ = ["LinkParams", "fit_link_params", "link", "LINK_COLUMNS"]

LINK_COLUMNS = ("frame", "y", "x", "se_y", "se_x")
"""The only columns the linker reads. Kept deliberately narrow: a linker that
reads nothing else cannot come to depend on a detector-internal column."""


@dataclass(frozen=True)
class LinkParams:
    """What the linker needs, measured from the movie rather than supplied.

    `d_grid` is the grid of candidate diffusion coefficients (px^2/frame,
    starting at an exact zero for genuinely immobile particles) and
    `d_logprior` the fitted population distribution over it. Every track
    carries its own posterior over that grid, which is why an immobile
    particle ends up with a tighter gate than a mobile one.
    """

    d_grid: np.ndarray
    d_logprior: np.ndarray
    p_cont: float
    """Per-frame P(detected AND still alive). One number, not two: with no gap
    hypotheses a bleached particle and an undetected one produce the same
    observation, so the data cannot separate them."""
    lam_birth: float
    """New tracks per px^2 within one frame."""
    se_inflate: float
    """Factor the reported CRLB variance is scaled by, >= 1. Estimated from
    the lag-1 displacement covariance, which is the part of the motion that
    localization error explains and diffusion does not."""
    trajectory: tuple = field(default_factory=tuple)
    """One entry per iteration of the fit, for the record."""

    @property
    def d_mean(self):
        """Population mean of D under the fitted prior, px^2/frame."""
        return float(np.sum(np.exp(self.d_logprior) * self.d_grid))

    @property
    def d_immobile(self):
        """Fitted fraction of the population that is immobile."""
        return float(np.exp(self.d_logprior)[0])

    def __repr__(self):
        return (f"LinkParams(D_mean={self.d_mean:.4g} px^2/frame, "
                f"immobile={self.d_immobile:.3f}, p_cont={self.p_cont:.3f}, "
                f"lam_birth={self.lam_birth:.3g}/px^2, "
                f"se_inflate={self.se_inflate:.3f})")


def _arrays(locs):
    """The five columns the linker reads, as the native call wants them."""
    missing = [c for c in LINK_COLUMNS if c not in locs.columns]
    if missing:
        raise ValueError(
            f"the localization table is missing {missing}; linking needs "
            f"{list(LINK_COLUMNS)} -- pass a `loctable` locs frame")
    frame = np.ascontiguousarray(locs["frame"].to_numpy(), dtype=np.int64)
    pos = np.ascontiguousarray(
        np.stack([locs["y"].to_numpy(), locs["x"].to_numpy()], axis=1),
        dtype=float)
    se = np.ascontiguousarray(
        np.stack([locs["se_y"].to_numpy(), locs["se_x"].to_numpy()], axis=1),
        dtype=float)
    return frame, pos, se


def fit_link_params(locs):
    """Estimate every linking parameter from the localizations. -> LinkParams.

    Nothing here is asked of the caller, because nothing here is a matter of
    opinion: how far a particle moves between frames, how often the detector
    misses one and how many new particles appear are all properties of this
    movie, and a person guessing at them is guessing at something the data
    already knows.

    The first estimate forms no association at all -- it fits a mixture to the
    distance from each detection to the nearest one in the following frame,
    which estimates the population distribution of D directly. Linking to
    estimate a step size and then using that step size to link would be
    circular. Three rounds of link and re-estimate follow, damped and clamped
    to that link-free anchor.
    """
    out = _rs.track_fit(*_arrays(locs))
    return LinkParams(
        d_grid=out["d_grid"], d_logprior=out["d_logprior"],
        p_cont=float(out["p_cont"]), lam_birth=float(out["lam_birth"]),
        se_inflate=float(out["se_inflate"]),
        trajectory=tuple(out["trajectory"]))


def _brightness(locs):
    """Per-row log flux and its variance, from `flux` and `se_flux`. A row
    whose flux error is missing or not positive gets a variance of 1 (a
    factor of e), which leaves its brightness nearly uninformative."""
    missing = [c for c in ("flux", "se_flux") if c not in locs.columns]
    if missing:
        raise ValueError(f"brightness=True needs the columns {missing}")
    flux = locs["flux"].to_numpy().astype(float)
    se = locs["se_flux"].to_numpy().astype(float)
    ok = np.isfinite(flux) & (flux > 0) & np.isfinite(se) & (se > 0)
    lf = np.log(np.where(ok, flux, 1.0))
    if ok.any():
        lf[~ok] = np.median(lf[ok])
    var = np.where(ok, (se / np.where(ok, flux, 1.0)) ** 2, 1.0)
    return [float(v) for v in lf], [float(v) for v in var]


def link(locs, params=None, brightness=False, *, min_link_margin=0.0,
         min_track_length=None, diagnostics=False):
    """Link a localization table into trajectories. -> the table + `track_id`.

    `params` is fitted from `locs` when omitted. The returned frame is the
    input with one `UInt32` column added, in the input's row order; a
    `track_id` column already present is replaced. Previous tracking
    diagnostics and acceptance columns are removed or recomputed on re-linking.

    `brightness=True` also reads `flux` and `se_flux`, so that a particle's
    brightness helps keep its identity; its noise is estimated from the movie.
    See the module docstring for what that buys and costs.

    `min_link_margin` is a finite non-negative cutoff in natural-log score
    units (nats). Each proposed link is compared with the best whole-frame
    assignment forbidding that link, including competitors and termination.
    Links with a smaller margin are cut before updating the track's state;
    no second-choice reassignment is made. Zero preserves the original
    behavior, including ties. This is not a probability threshold.

    `min_track_length=N` adds `track_length` (UInt32) and `track_accepted`
    (Boolean). N must be a positive integer, counting consecutive detected
    frames, not links. Acceptance is retrospective over each complete segment;
    it never encourages extending a track to reach N, and no rows are dropped.
    Omit it to preserve the original output schema.

    `diagnostics=True` adds `link_margin` (nullable Float64) and `link_rejected`
    (Boolean), describing the proposed incoming link at each detection. A
    rejected proposal has its measured margin and starts a new track; a birth
    with no proposal has a null margin and False. Margins are conditional on
    track histories committed before this frame, not whole-movie confidence.
    """
    import polars as pl
    from numbers import Integral

    if isinstance(min_link_margin, (bool, np.bool_)):
        raise ValueError("min_link_margin must be finite and non-negative")
    try:
        min_link_margin = float(min_link_margin)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("min_link_margin must be finite and non-negative") from error
    if not np.isfinite(min_link_margin) or min_link_margin < 0:
        raise ValueError("min_link_margin must be finite and non-negative")
    if min_track_length is not None:
        if (isinstance(min_track_length, (bool, np.bool_))
                or not isinstance(min_track_length, Integral)
                or min_track_length < 1):
            raise ValueError("min_track_length must be a positive integer")
        min_track_length = int(min_track_length)

    if params is None:
        params = fit_link_params(locs)
    frame, pos, se = _arrays(locs)
    ids, _, diag = _rs.track_link_scored(
        frame, pos, se,
        np.ascontiguousarray(params.d_grid, dtype=float),
        np.ascontiguousarray(params.d_logprior, dtype=float),
        float(params.p_cont), float(params.lam_birth),
        float(params.se_inflate),
        brightness=_brightness(locs) if brightness else None,
        min_link_margin=min_link_margin, diagnostics=diagnostics)
    metadata = {"track_length", "track_accepted", "link_margin", "link_rejected"}
    out = locs.drop([c for c in locs.columns if c in metadata]).with_columns(
        pl.Series("track_id", ids, dtype=pl.UInt32))
    if min_track_length is not None:
        out = out.with_columns(pl.len().over("track_id").cast(pl.UInt32).alias("track_length"))
        # A requested N larger than the table cannot be met; this also avoids
        # passing an arbitrary-size Python integer into a fixed-width literal.
        accepted = (pl.col("track_length") >= min_track_length
                    if min_track_length <= len(out) else pl.lit(False))
        out = out.with_columns(accepted.alias("track_accepted"))
    if diagnostics:
        margin, rejected = diag
        out = out.with_columns(
            pl.Series("link_margin", margin, dtype=pl.Float64),
            pl.Series("link_rejected", rejected, dtype=pl.Boolean))
    return out
