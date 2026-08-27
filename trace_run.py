"""Run the detector under the tracer and render the step-by-step movie.

    python trace_run.py                                # whole bead frame
    python trace_run.py --crop 10 26 12 28             # one region, on its own
    python trace_run.py --crop 10 26 12 28 --out patch_A

`--crop y0 y1 x0 x1` runs the WHOLE pipeline on that sub-image alone, which is
the point of it: a region cropped out of the frame is a much smaller problem
(seconds, not minutes), so a hypothesis about it can be tested in one edit-run
cycle. Note that cropping changes the problem in one honest way -- the emitters
just outside the crop are gone rather than modelled -- so a crop should be taken
wide enough that its own border is quiet in the score map.

Outputs, all under `--out` (default `trace_beads/`):

    trace.pkl        the raw event list (tracer.Recorder.load)
    trace_log.txt    one line per decision -- read this first
    storyboard.png   one row per checkpoint, the whole run on a page
    frames/          every frame of the movie
    movie.mp4        the frames assembled
"""

import argparse
import os
import time

import numpy as np
import tifffile

import audit
import boxsolve
import tracer
import trace_view

CAMERA_OFFSET = 100.0


def main(args):
    img = tifffile.imread(args.image)
    if args.crop:
        y0, y1, x0, x1 = args.crop
        img = img[y0:y1, x0:x1]
        print(f"cropped to y {y0}:{y1}  x {x0}:{x1}  -> {img.shape}")
    print(f"image: {img.shape} dtype={img.dtype} min={img.min()} max={img.max()}")

    os.makedirs(args.out, exist_ok=True)
    tracer.start(image=args.image, crop=args.crop, sigma=args.sigma)
    t0 = time.time()
    res = boxsolve.detect_boxes(img, sigma=args.sigma, offset=CAMERA_OFFSET,
                                gain=args.gain,
                                refine_gain=args.refine_gain,
                                n_outer=args.n_outer, k_max=args.k_max, verbose=1)
    dt = time.time() - t0
    rec = tracer.stop()
    print(f"\ndetection took {dt:.1f}s;  trace: {rec.summary()}")

    d_e = (img.astype(float) - CAMERA_OFFSET) / res.gain
    a = audit.audit_result(d_e, res.model_image, res.sigma, z_thresh=5.0)
    print()
    print(audit.format_report(a, label=f"N={len(res.positions)}"))

    print()
    print(trace_view.summarize(rec))

    rec.save(os.path.join(args.out, "trace.pkl"))
    trace_view.write_log(rec, os.path.join(args.out, "trace_log.txt"))
    tl = trace_view.Timeline(rec)
    print(f"\ntimeline: {len(tl)} frames")
    trace_view.storyboard(tl, os.path.join(args.out, "storyboard.png"))
    if not args.no_movie:
        fdir = os.path.join(args.out, "frames")
        paths = trace_view.render_frames(
            tl, fdir, dpi=args.dpi, stride=args.stride, limit=args.limit,
            focus=tuple(args.focus) if args.focus else None,
            kinds=tuple(args.kinds) if args.kinds else None)
        print(f"rendered {len(paths)} frames -> {fdir}")
        mv = trace_view.write_movie(fdir, os.path.join(args.out, "movie.mp4"),
                                    fps=args.fps)
        print(f"wrote {mv}")
    print(f"wrote {args.out}/trace_log.txt and {args.out}/storyboard.png")
    return rec, tl, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="beads_60x_still.tif")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"),
                    default=None, help="run the pipeline on this sub-image alone")
    ap.add_argument("--out", default="trace_beads")
    ap.add_argument("--sigma", type=float, default=1.2)
    ap.add_argument("--n-outer", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--refine-gain", action="store_true",
                    help="off by default; see calibrate.py")
    ap.add_argument("--no-movie", action="store_true",
                    help="log + storyboard only; skip frame rendering")
    ap.add_argument("--focus", type=float, nargs=3, metavar=("Y", "X", "R"),
                    default=None,
                    help="movie only of decisions taken within R px of (Y, X)")
    ap.add_argument("--kinds", nargs="+", default=None,
                    metavar="KIND",
                    help="event kinds to keep as frames (seed sweep_start move "
                         "proposals sweep_end eb_update final)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--limit", type=int, default=None, help="cap the frame count")
    ap.add_argument("--dpi", type=int, default=90)
    ap.add_argument("--fps", type=int, default=4)
    main(ap.parse_args())
