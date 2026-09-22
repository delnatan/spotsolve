"""Localize a movie and save all measurements and diagnostic flags.

    python localize_movie.py --frames 0 50 --out hyp7_all
    python localize_movie.py --selection bic --count-penalty 2 --threads 5
"""

import argparse
import json
import time
from pathlib import Path

import polars as pl
import tifffile

import spotsolve
from spotsolve import loctable

# The hyp7gem crop, and the acquisition it was cut from. Pixel size and frame
# interval come from the source .nd2 (65 nm, 20.005 ms); sigma is the value
# calibrated for this dataset in section 10b of
# docs/archive/ALGORITHM_HISTORY.md. The crop is not tracked (see .gitignore).
DEFAULT_IMAGE = str(Path(__file__).resolve().parent.parent
                    / "data" / "hyp7gem_wt_crop.tif")
DEFAULT_SIGMA = 1.45      # px
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
          f"{stack.shape[1]}x{stack.shape[2]} px, sigma={args.sigma}")

    # One native call for the whole range: frames run in parallel threads.
    t0 = time.time()
    results = spotsolve.localize_stack(stack, sigma=args.sigma,
                                       offset=CAMERA_OFFSET,
                                       k_max=args.k_max,
                                       selection=args.selection,
                                       count_penalty=args.count_penalty,
                                       n_threads=args.threads)
    dt = (time.time() - t0) / len(stack)
    print(f"  {len(stack)} frames in {dt * len(stack):.2f} s "
          f"({1 / dt:.1f} frames/s)")

    locs_parts, frame_parts = [], []
    loc_id = 0
    for k, res in enumerate(results):
        frame = first + k

        locs, row = loctable.frame_tables(
            res, frame=frame, t=frame * args.interval,
            pixel_size=args.pixel_size,
            seconds=dt, loc_id0=loc_id)
        loc_id += locs.height
        locs_parts.append(locs)
        frame_parts.append(row)

        r = row.row(0, named=True)
        print(f"  frame {frame:3d}  N={r['n_locs']:4d}  "
              f"median flux {r['median_flux']:7.0f} ADU  "
              f"median SE(pos) {r['median_se_pos']:.3f} px  "
              f"flagged {r['n_flagged']:3d}")

    locs = loctable.concat(locs_parts)
    frames = loctable.concat(frame_parts)
    print(f"\n{locs.height} localizations over {frames.height} frames")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print(locs.select("loc_id", "frame", "y", "x", "se_y", "se_x",
                          "flux", "se_flux", "fit_sigma", "sigma_se", "flags").head(6))
        print(frames)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    locs.write_parquet(out / "localizations.parquet")
    frames.write_parquet(out / "frames.parquet")
    # Pixel size and interval are not in the parquet -- they are
    # properties of the acquisition, not of any row -- but every physical
    # quantity a tracker computes depends on them, so they travel alongside.
    (out / "meta.json").write_text(json.dumps({
        "image": str(Path(args.image).resolve()),
        "frames": [first, first + len(stack)],
        "sigma_px": args.sigma,
        "camera_offset_adu": CAMERA_OFFSET,
        "pixel_size_um": args.pixel_size, "frame_interval_s": args.interval,
        "k_max": args.k_max,
        "selection": args.selection,
        "count_penalty": args.count_penalty,
        "threads": args.threads,
        "fit_flags": {flag.name: int(flag) for flag in spotsolve.FitFlag},
        "flux_units": "ADU above offset", "position_units": "px (y, x)",
    }, indent=2) + "\n")
    print(f"\nwrote {out}/localizations.parquet, frames.parquet, "
          f"meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--frames", type=int, nargs=2, default=(0, 10),
                    metavar=("FIRST", "LAST"),
                    help="half-open frame range (default: 0 10)")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--pixel-size", type=float, default=DEFAULT_PIXEL_SIZE,
                    help="um per px, for the derived physical columns")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                    help="seconds per frame")
    ap.add_argument("--k-max", type=int, default=spotsolve.K_MAX)
    ap.add_argument("--selection", choices=("fixed", "bic"), default="fixed")
    ap.add_argument("--count-penalty", type=float, default=0.0,
                    help="extra cost per emitter (default: 0)")
    ap.add_argument("--threads", type=int, default=None,
                    help="parallel frame workers (default: machine's cores)")
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parent.parent
                                / "data" / "hyp7_locs"))
    main(ap.parse_args())
