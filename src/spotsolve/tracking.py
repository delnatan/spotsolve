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

`flux` is deliberately not read, though a bright particle staying bright is
real evidence. Intensity is also what merge/split inference needs, and taking
a partial dependency on it now would make the two harder to separate later.

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

No dials
--------
Every parameter is estimated from the data by `fit_link_params`, which `link`
calls for you if you do not pass one. Fit once and reuse it when you are
linking several movies of the same sample, or to read what was estimated:
`params.trajectory` records the empirical-Bayes loop iterate by iterate, so a
fit that goes wrong says so rather than quietly returning a worse answer.
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


def link(locs, params=None):
    """Link a localization table into trajectories. -> the table + `track_id`.

    `params` is fitted from `locs` when omitted. The returned frame is the
    input with one `UInt32` column added, in the input's row order; a
    `track_id` column already present is replaced.
    """
    import polars as pl

    if params is None:
        params = fit_link_params(locs)
    frame, pos, se = _arrays(locs)
    ids = _rs.track_link(
        frame, pos, se,
        np.ascontiguousarray(params.d_grid, dtype=float),
        np.ascontiguousarray(params.d_logprior, dtype=float),
        float(params.p_cont), float(params.lam_birth),
        float(params.se_inflate))
    return locs.with_columns(pl.Series("track_id", ids, dtype=pl.UInt32))
