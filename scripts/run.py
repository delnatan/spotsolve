"""Run `spotsolve` on one image and report the residual audit.

The acceptance test is the audit panel (yellow = missed, cyan = over-modelled),
not N and not the residual spread. Everything shown comes from the model
`spotsolve` itself accepted; nothing here re-fits or culls afterwards.

    python run.py --image beads_60x_still_02.tif --gain 4.23 --read-noise 1.6
"""

import argparse
import pathlib
import time

import numpy as np
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spotsolve
from spotsolve import audit

CAMERA_OFFSET = 100.0


def main(args):
    img = tifffile.imread(args.image)
    if img.ndim == 3:
        # A stack: this tool reports ONE frame. Detection is per-frame, so
        # picking a frame here is the honest thing rather than projecting,
        # which would blur emitters across their own motion.
        if not 0 <= args.frame < len(img):
            raise SystemExit(f"--frame {args.frame} out of range for a stack "
                             f"of {len(img)}")
        print(f"stack of {len(img)}; using frame {args.frame}")
        img = img[args.frame]
    elif img.ndim != 2:
        raise SystemExit(f"expected a 2-D image or a 3-D stack, got shape "
                         f"{img.shape}")
    if args.crop:
        y0, y1, x0, x1 = args.crop
        img = img[y0:y1, x0:x1]
    print(f"{args.image}  {img.shape}  gain={args.gain}")

    t = time.time()
    res = spotsolve.localize(img, sigma=args.sigma, offset=CAMERA_OFFSET,
                             gain=args.gain, read_noise=args.read_noise,
                             k_max=args.k_max)
    dt = time.time() - t

    d_e = (img.astype(float) - CAMERA_OFFSET) / res.gain
    a = audit.audit_result(d_e, res.model_image, res.sigma)
    nr = (d_e - res.model_image) / np.sqrt(np.maximum(res.model_image, 1e-6))

    print(f"\nN={len(res.positions)}  gain={res.gain:.3f}  "
          f"bg={np.median(res.background):.2f}  {dt:.3f}s  {res.history[0]}")
    print(audit.format_report(a, label=args.image))
    if res.se is not None and len(res.se) and np.isfinite(res.se).any():
        sp = np.nanmedian(np.hypot(res.se[:, 1], res.se[:, 2]))
        print(f"median position CRLB: {sp:.3f} px")
    if len(res.amplitudes):
        q = np.percentile(res.amplitudes, [5, 50, 95])
        print(f"amplitude e- (5/50/95): {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}")
    rep = spotsolve.aggregate_report(res, ratio=args.agg_ratio)
    if rep["n_aggregates"]:
        print(f"\naggregates (post-hoc, flux > {args.agg_ratio:.0f}x the "
              f"median detection of {rep['median_flux']:.0f} e-): "
              f"{rep['n_aggregates']} objects from "
              f"{rep['n_detections_flagged']} detections, "
              f"{100*rep['flux_fraction']:.1f}% of all detected flux")
        print(f"{'y':>8} {'x':>8} {'flux e-':>11} {'x median':>9} {'ndet':>5}")
        for o in rep["objects"]:
            print(f"{o['y']:8.2f} {o['x']:8.2f} {o['flux']:11.0f} "
                  f"{o['ratio']:9.1f} {o['n']:5d}")

    rej = res.width_rejects
    wide = (rej[rej["reason"] == "too_wide"] if rej is not None
            else np.empty(0, dtype=spotsolve.WIDTH_REJECT_DTYPE))
    if len(wide):
        print(f"\nwide objects (modelled, not reported as detections): "
              f"{len(wide)}")
        print(f"{'y':>8} {'x':>8} {'sigma':>7} {'flux e-':>10}")
        for g in np.sort(wide, order="flux")[::-1]:
            print(f"{g['y']:8.2f} {g['x']:8.2f} {g['sigma']:7.2f} "
                  f"{g['flux']:10.0f}")

    fig, ax = plt.subplots(1, 4, figsize=(15, 4.0))
    ax[0].imshow(img, cmap="gray")
    if len(res.positions):
        ax[0].plot(res.positions[:, 1], res.positions[:, 0], "r+", ms=8, mew=1.3)
    # Wide objects drawn at twice their fitted width, so what the model
    # absorbed as a non-point-source is visible rather than merely tabulated.
    for g in wide:
        ax[0].add_patch(plt.Circle((g["x"], g["y"]), 2 * g["sigma"],
                                   fill=False, ec="orange", lw=1.4, ls="--"))
    ttl = f"{args.image}\nN={len(res.positions)}"
    if len(wide):
        ttl += f"  (+{len(wide)} wide)"
    ax[0].set_title(ttl, fontsize=9)
    ax[1].imshow(res.model_image, cmap="gray")
    ax[1].set_title("model", fontsize=9)
    ax[2].imshow(nr, cmap="RdBu_r", vmin=-5, vmax=5)
    ax[2].set_title(f"norm. residual  (med {np.median(nr):+.2f}, "
                    f"sd {0.5*(np.percentile(nr,84.1)-np.percentile(nr,15.9)):.2f})",
                    fontsize=9)
    z = audit.score_map(d_e, res.model_image, res.sigma)
    ax[3].imshow(z, cmap="RdBu_r", vmin=-8, vmax=8)
    if len(a["positive"]):
        ax[3].scatter(a["positive"][:, 1], a["positive"][:, 0], s=110,
                      facecolors="none", edgecolors="yellow", lw=1.3)
    if len(a["negative"]):
        ax[3].scatter(a["negative"][:, 1], a["negative"][:, 0], s=110,
                      facecolors="none", edgecolors="cyan", lw=1.3)
    ax[3].set_title(f"score z: {a['n_missed']} missed / {a['n_piled']} piled\n"
                    f"|z| max {max(abs(a['z_max']), abs(a['z_min'])):.1f}", fontsize=9)
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image",
                    default=str(pathlib.Path(__file__).resolve().parent.parent
                                / "data" / "beads_60x_still_02.tif"))
    ap.add_argument("--crop", type=int, nargs=4, default=None,
                    metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--frame", type=int, default=0,
                    help="which frame, if the file is a stack")
    ap.add_argument("--sigma", type=float, default=1.2)
    ap.add_argument("--gain", type=float, default=4.23,
                    help="ADU per photoelectron; omit to estimate (calibrate.py)")
    ap.add_argument("--read-noise", type=float, default=0.0,
                    help="camera read noise, e- rms")
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--agg-ratio", type=float, default=spotsolve.AGG_AMP_RATIO,
                    help="post-hoc aggregate flag: flux as a multiple of the "
                         "frame's median detection (reported, never removed)")
    ap.add_argument("--out", default="result.png")
    main(ap.parse_args())
