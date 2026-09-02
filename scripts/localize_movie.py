"""Detect emitters frame by frame and write the standard localization table.

A worked example of driving `spotsolve` over a movie, and of the one filtering
decision real data forces: separating point emitters from aggregates.

    python localize_movie.py                    # first 10 frames of hyp7gem
    python localize_movie.py --frames 0 49 --out hyp7_all

Detection is strictly per frame -- nothing here uses the previous frame's
answer, so the output is an honest input for a tracker rather than something
already smoothed in time. Linking is the next stage and is not this script's
business.

Which aggregate test, and why this one
--------------------------------------
`spotsolve` has two, for two physically different objects (README section 10b):

  * OVER-WIDE (sigma_fit >> sigma): the fixed-sigma model cannot represent it,
    so the search tiles it into dozens of pieces that no later statistic can
    recover. It must be excluded BEFORE the search -- `detect(...,
    reject_aggregates=True)`.
  * OVER-BRIGHT (sigma_fit ~ sigma, flux >> the frame median): representable,
    so the search fits it correctly as one bright emitter and it stays
    identifiable. Filter it AFTER -- `spotsolve.flag_aggregates`.

hyp7gem is the second kind, measured: its aggregates fit sigma 1.47-1.65
against a PSF sigma of 1.45 -- 1.01-1.14x, no width signal at all -- while
carrying 100-176x the median flux. They look wide on screen only because the
display saturates. So this script uses the post-hoc filter and leaves
`reject_aggregates` off; turning it on here does measurable harm, because a
PSF-width object frozen into the background double-counts against the
emitters fitted beside it.

The consequence to read, not to hide: on frame 0, 47% of all detected flux is
in 8 aggregates. That is a fact about the sample, and it is why the flag is
carried in the table instead of being applied silently.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import tifffile

import spotsolve
from spotsolve import loctable

# The hyp7gem crop, and the acquisition it was cut from. Pixel size and frame
# interval come from the source .nd2 (65 nm, 20.005 ms); sigma and gain are
# the values calibrated for this dataset in README sections 10b and 13.
DEFAULT_IMAGE = "/Users/delnatan/Projects/github/sfwloc/data/hyp7gem_wt_crop.tif"
DEFAULT_SIGMA = 1.45      # px
DEFAULT_GAIN = 2.401      # ADU per photoelectron
DEFAULT_PIXEL_SIZE = 0.065   # um
DEFAULT_INTERVAL = 0.020005  # s
CAMERA_OFFSET = 100.0     # ADU


def load_stack(path, first, last):
    img = tifffile.imread(path)
    if img.ndim == 2:
        img = img[None]
    elif img.ndim != 3:
        raise SystemExit(f"expected a 2-D image or 3-D stack, got {img.shape}")
    last = len(img) if last is None else min(last, len(img))
    if not 0 <= first < last:
        raise SystemExit(f"--frames {first} {last} empty for a stack of "
                         f"{len(img)}")
    return img[first:last], first


def main(args):
    stack, first = load_stack(args.image, *args.frames)
    print(f"{args.image}: frames {first}..{first + len(stack) - 1}, "
          f"{stack.shape[1]}x{stack.shape[2]} px, sigma={args.sigma}, "
          f"gain={args.gain}, backend={args.impl}")

    locs_parts, frame_parts, agg_parts, width_parts = [], [], [], []
    loc_id = 0
    for k, raw in enumerate(stack):
        frame = first + k
        t0 = time.time()
        res = spotsolve.detect(raw, sigma=args.sigma, offset=CAMERA_OFFSET,
                            gain=args.gain, k_max=args.k_max, impl=args.impl,
                            verbose=0)
        if args.filter_in_focus:
            res = spotsolve.filter_in_focus(
                raw, res, offset=CAMERA_OFFSET, gain=args.gain,
                sigma_ratio_min=args.sigma_ratio_min,
                sigma_ratio_max=args.sigma_ratio_max,
                sigma_lo_ratio=args.var_sigma_lo_ratio,
                sigma_hi_ratio=args.var_sigma_hi_ratio,
                k_max=args.k_max, impl=args.impl)
        dt = time.time() - t0

        locs, row, aggs = loctable.frame_tables(
            res, frame=frame, t=frame * args.interval,
            pixel_size=args.pixel_size, agg_ratio=args.agg_ratio,
            seconds=dt, loc_id0=loc_id)
        width_rejects = loctable.width_reject_table(
            res, frame=frame, t=frame * args.interval,
            pixel_size=args.pixel_size)
        loc_id += locs.height
        locs_parts.append(locs)
        frame_parts.append(row)
        agg_parts.append(aggs)
        width_parts.append(width_rejects)

        r = row.row(0, named=True)
        print(f"  frame {frame:3d}  N={r['n_locs']:4d}  "
              f"median flux {r['median_flux']:7.0f} e-  "
              f"median SE(pos) {r['median_se_pos']:.3f} px  "
              f"width pruned/reject "
              f"{r['n_width_pruned']:3d}/"
              f"{r['n_width_too_narrow'] + r['n_width_too_wide']:3d}  "
              f"aggregates {r['n_aggregates']:2d} "
              f"({100 * r['agg_flux_fraction']:4.1f}% of flux)  {dt:.1f}s")

    locs = loctable.concat(locs_parts)
    frames = loctable.concat(frame_parts)
    aggregates = loctable.concat(agg_parts)
    width_rejects = loctable.concat(width_parts)
    clean = loctable.filter_aggregates(locs)

    print(f"\n{locs.height} localizations over {frames.height} frames; "
          f"{locs.height - clean.height} flagged as aggregate "
          f"({aggregates.height} objects), {clean.height} point emitters kept")
    print(f"detected flux in aggregates: "
          f"{100 * float(locs.filter(pl.col('is_aggregate'))['flux'].sum()) / float(locs['flux'].sum()):.1f}%")

    print("\nlocalization table (point emitters, first rows):")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print(clean.select("loc_id", "frame", "t", "y", "x", "se_y", "se_x",
                           "flux", "se_flux", "flux_snr",
                           "flux_ratio").head(6))
        print("\nper-frame summary:")
        print(frames.select("frame", "n_locs", "median_flux", "median_se_pos",
                            "n_width_pruned", "n_width_too_narrow",
                            "n_width_too_wide",
                            "n_aggregates", "agg_flux_fraction", "seconds"))
        print("\naggregates, brightest first:")
        print(aggregates.sort("flux", descending=True)
              .select("frame", "y", "x", "flux", "ratio", "n_locs").head(10))

    # Precision, which is the column a tracker's gate is built on.
    q = clean["se_pos"].quantile
    print(f"\nposition CRLB over kept emitters, px (10/50/90): "
          f"{q(0.1):.3f} / {q(0.5):.3f} / {q(0.9):.3f}   "
          f"= {1000 * q(0.5) * args.pixel_size:.0f} nm at the median")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    locs.write_parquet(out / "localizations.parquet")
    frames.write_parquet(out / "frames.parquet")
    aggregates.write_parquet(out / "aggregates.parquet")
    width_rejects.write_parquet(out / "width_rejects.parquet")
    # Pixel size, interval and gain are not in the parquet -- they are
    # properties of the acquisition, not of any row -- but every physical
    # quantity a tracker computes depends on them, so they travel alongside.
    (out / "meta.json").write_text(json.dumps({
        "image": str(Path(args.image).resolve()),
        "frames": [first, first + len(stack)],
        "sigma_px": args.sigma, "gain": args.gain,
        "camera_offset_adu": CAMERA_OFFSET,
        "pixel_size_um": args.pixel_size, "frame_interval_s": args.interval,
        "k_max": args.k_max, "impl": args.impl,
        "filter_in_focus": args.filter_in_focus,
        "sigma_ratio_min": args.sigma_ratio_min if args.filter_in_focus else None,
        "sigma_ratio_max": args.sigma_ratio_max if args.filter_in_focus else None,
        "var_sigma_lo_ratio": args.var_sigma_lo_ratio if args.filter_in_focus else None,
        "var_sigma_hi_ratio": args.var_sigma_hi_ratio if args.filter_in_focus else None,
        "agg_ratio": args.agg_ratio if args.agg_ratio is not None
        else spotsolve.AGG_AMP_RATIO,
        "flux_units": "photoelectrons", "position_units": "px (y, x)",
    }, indent=2) + "\n")
    print(f"\nwrote {out}/localizations.parquet, frames.parquet, "
          f"aggregates.parquet, width_rejects.parquet, meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--frames", type=int, nargs=2, default=(0, 10),
                    metavar=("FIRST", "LAST"),
                    help="half-open frame range (default: 0 10)")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--gain", type=float, default=DEFAULT_GAIN,
                    help="ADU per photoelectron; the fit is invariant to it, "
                         "the acceptance threshold is not")
    ap.add_argument("--pixel-size", type=float, default=DEFAULT_PIXEL_SIZE,
                    help="um per px, for the derived physical columns")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                    help="seconds per frame")
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--impl", choices=["py", "rs"], default="rs")
    ap.add_argument("--agg-ratio", type=float, default=None,
                    help=f"over-bright cut, flux / the frame's median "
                         f"detection (default {spotsolve.AGG_AMP_RATIO:.0f})")
    ap.add_argument("--filter-in-focus", action="store_true",
                    help="post-detect variable-sigma physical-width filter")
    ap.add_argument("--sigma-ratio-min", type=float,
                    default=spotsolve.INF_FOCUS_SIGMA_RATIO_MIN,
                    help="minimum fitted sigma / in-focus sigma to keep")
    ap.add_argument("--sigma-ratio-max", type=float,
                    default=spotsolve.INF_FOCUS_SIGMA_RATIO_MAX,
                    help="maximum fitted sigma / in-focus sigma to keep")
    ap.add_argument("--var-sigma-lo-ratio", type=float,
                    default=spotsolve.VAR_SIGMA_RATIO_LO,
                    help="lower variable-sigma fit bound, relative to sigma")
    ap.add_argument("--var-sigma-hi-ratio", type=float,
                    default=spotsolve.VAR_SIGMA_RATIO_HI,
                    help="upper variable-sigma fit bound, relative to sigma")
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parent.parent
                                / "data" / "hyp7_locs"))
    main(ap.parse_args())
