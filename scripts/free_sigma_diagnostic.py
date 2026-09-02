"""Post-hoc free-sigma diagnostic for fixed-sigma spotsolve detections.

This is an experiment, not a replacement detector. It starts from a localization
table produced by the fixed-sigma pipeline, rebuilds the same style of local
patches, and refits each patch with one shared sigma parameter. The result is a
width/flux diagnostic table and, optionally, a visual panel for one frame.
"""

import argparse
import json
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tifffile

from spotsolve import calibrate, core, lmga, patches, psf


def load_stack(path):
    img = tifffile.imread(path)
    if img.ndim == 2:
        img = img[None]
    elif img.ndim != 3:
        raise SystemExit(f"expected a 2-D image or 3-D stack, got {img.shape}")
    return img


def make_background(d_e, positions, sigma):
    free = calibrate.emitter_free_mask(
        d_e.shape, positions, sigma, core.BG_MASK_RADIUS
    )
    scalar = max(
        calibrate.robust_background(
            d_e, positions, sigma, core.BG_MASK_RADIUS, free=free
        ),
        core.BG_FLOOR,
    )
    return core.background_map(d_e, positions, sigma, fallback=scalar, free=free)


def fit_patch(d_e, bmap, positions, amplitudes, patch, sigma0, sigma_lo, sigma_hi):
    yy, xx = patches.patch_grids(patch)
    sub = np.asarray(d_e[patch.y0 : patch.y1, patch.x0 : patch.x1])
    level, shape = core._window_bg(bmap, patch.y0, patch.x0, patch.y1, patch.x1)
    halo = (
        patches.build_halo_image(
            positions,
            amplitudes,
            patch.frozen_indices,
            sigma0,
            yy,
            xx,
            patch.y0,
            patch.x0,
        )
        + shape
    )

    loc = positions[patch.indices] - np.array([patch.y0, patch.x0])
    theta0 = psf.pack(level, amplitudes[patch.indices], loc[:, 0], loc[:, 1])

    fixed = core._fit_window(sub, yy, xx, sigma0, halo, theta0, max_iter=100)

    lo, hi = core._bounds(
        len(patch.indices),
        sub.shape[0],
        sub.shape[1],
        max(float(sub.max()), 1.0) * 4.0,
        8.0 * max(float(sub.max()), 1.0) / psf.peak_factor(sigma0),
    )
    lo = np.append(lo, sigma_lo)
    hi = np.append(hi, sigma_hi)
    theta_fs0 = np.append(fixed.theta, sigma0)
    free = lmga.fit(
        np.clip(theta_fs0, lo + 1e-9, hi - 1e-9),
        yy,
        xx,
        sigma0,
        sub,
        lo,
        hi,
        halo=halo,
        max_iter=160,
        tol_obj=core.EVIDENCE_TOL_OBJ,
        free_sigma=True,
    )
    return fixed, free


def sigma_se_and_corr(F, n_emitters):
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return float("nan"), np.full(n_emitters, np.nan), float("nan")
    v_sig = cov[-1, -1]
    sigma_se = np.sqrt(v_sig) if np.isfinite(v_sig) and v_sig > 0 else float("nan")
    corr = np.full(n_emitters, np.nan)
    if np.isfinite(sigma_se) and sigma_se > 0:
        for k in range(n_emitters):
            v_a = cov[1 + 3 * k, 1 + 3 * k]
            if np.isfinite(v_a) and v_a > 0:
                corr[k] = cov[1 + 3 * k, -1] / (np.sqrt(v_a) * sigma_se)
    try:
        cond = float(np.linalg.cond(F))
    except np.linalg.LinAlgError:
        cond = float("nan")
    return float(sigma_se), corr, cond


def render_free_model(shape, bmap, patch_rows):
    H, W = shape
    out = np.array(bmap, dtype=float, copy=True)
    for row in patch_rows:
        y = row["fit_y"]
        x = row["fit_x"]
        a = row["fit_flux"]
        sigma = row["fit_sigma"]
        out += calibrate.render_model(
            np.array([[y, x]], dtype=float),
            np.array([a], dtype=float),
            float(sigma),
            (H, W),
            0.0,
        )
    return out


def frame_diagnostic(raw, locs, frame, sigma0, gain, offset, args):
    d_e = (raw.astype(float) - offset) / gain
    positions = locs.select("y", "x").to_numpy()
    amplitudes = locs["flux"].to_numpy()
    bmap = make_background(d_e, positions, sigma0)
    pset = patches.build_patches(
        positions,
        sigma0,
        d_e.shape,
        link_radius_factor=core.LINK_FACTOR,
        halo_radius_factor=core.HALO_FACTOR,
        bbox_pad_factor=core.BBOX_PAD,
        k_max=args.k_max,
    )

    rows = []
    patch_summary = []
    t0 = time.time()
    for patch_id, p in enumerate(pset):
        fixed, free = fit_patch(
            d_e, bmap, positions, amplitudes, p, sigma0, args.sigma_lo, args.sigma_hi
        )
        _, A, cy, cx = psf.unpack(free.theta[:-1])
        sigma_fit = float(free.theta[-1])
        sigma_se, corr_a_sigma, cond = sigma_se_and_corr(free.F, len(p.indices))
        delta_I = float(fixed.I - free.I)
        patch_summary.append(
            {
                "frame": frame,
                "patch_id": patch_id,
                "n_emitters": len(p.indices),
                "sigma": sigma_fit,
                "sigma_se": sigma_se,
                "delta_I": delta_I,
                "converged": free.converged,
                "stalled": free.stalled,
                "n_iter": free.n_iter,
                "cond": cond,
            }
        )
        for j, loc_id in enumerate(p.indices):
            loc_id = int(loc_id)
            rows.append(
                {
                    "loc_id": int(locs["loc_id"][loc_id]),
                    "frame": frame,
                    "patch_id": patch_id,
                    "patch_n": len(p.indices),
                    "orig_y": float(positions[loc_id, 0]),
                    "orig_x": float(positions[loc_id, 1]),
                    "orig_flux": float(amplitudes[loc_id]),
                    "fit_y": float(cy[j] + p.y0),
                    "fit_x": float(cx[j] + p.x0),
                    "fit_flux": float(A[j]),
                    "fit_sigma": sigma_fit,
                    "sigma_ratio": sigma_fit / sigma0,
                    "sigma_se": sigma_se,
                    "corr_flux_sigma": float(corr_a_sigma[j]),
                    "delta_I": delta_I,
                    "converged": free.converged,
                    "stalled": free.stalled,
                    "n_iter": free.n_iter,
                    "cond": cond,
                    "dim": float(A[j]) < args.flux_cut,
                    "wide": sigma_fit > args.wide_sigma,
                }
            )
    seconds = time.time() - t0
    return rows, patch_summary, bmap, seconds


def plot_frame(raw, d_e, bmap, rows, args, out_png):
    y = np.array([r["fit_y"] for r in rows])
    x = np.array([r["fit_x"] for r in rows])
    flux = np.array([r["fit_flux"] for r in rows])
    sig = np.array([r["fit_sigma"] for r in rows])
    bad = (flux < args.flux_cut) | (sig > args.wide_sigma)
    model = render_free_model(raw.shape, bmap, rows)
    nr = (d_e - model) / np.sqrt(np.maximum(model, 1e-6))

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax = ax.ravel()
    ax[0].imshow(raw, cmap="gray")
    ax[0].scatter(x[~bad], y[~bad], s=7, c="#00d17a", linewidths=0, alpha=0.8)
    ax[0].scatter(x[bad], y[bad], s=9, c="#ff4d8d", linewidths=0, alpha=0.8)
    ax[0].set_title(
        f"free-sigma diagnostic: {bad.sum()} flagged / {len(rows)} "
        f"(flux < {args.flux_cut:g} or sigma > {args.wide_sigma:g})",
        fontsize=9,
    )

    ax[1].imshow(raw, cmap="gray")
    sc = ax[1].scatter(x, y, s=8, c=sig, cmap="viridis", vmin=args.sigma_lo, vmax=args.sigma_hi)
    ax[1].set_title("fitted patch sigma", fontsize=9)
    fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].scatter(sig, flux, s=8, c=np.where(bad, "#ff4d8d", "#1f77b4"), alpha=0.45, linewidths=0)
    ax[2].axhline(args.flux_cut, color="black", lw=1, ls="--")
    ax[2].axvline(args.sigma0, color="gray", lw=1)
    ax[2].axvline(args.wide_sigma, color="black", lw=1, ls="--")
    ax[2].set_xlabel("PSF sigma (px)")
    ax[2].set_ylabel("flux (photoelectrons)")
    ax[2].set_title("flux vs width", fontsize=9)
    ax[2].grid(True, color="0.9", linewidth=0.6)

    ax[3].imshow(nr, cmap="RdBu_r", vmin=-5, vmax=5)
    ax[3].set_title(
        f"diagnostic residual, med {np.median(nr):+.2f}, "
        f"robust sd {calibrate.robust_spread(nr):.2f}",
        fontsize=9,
    )
    for a in (ax[0], ax[1], ax[3]):
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(args):
    loc_dir = pathlib.Path(args.locs)
    meta = json.loads((loc_dir / "meta.json").read_text())
    sigma0 = args.sigma0 if args.sigma0 is not None else float(meta["sigma_px"])
    gain = args.gain if args.gain is not None else float(meta["gain"])
    offset = args.offset if args.offset is not None else float(meta["camera_offset_adu"])
    args.sigma0 = sigma0

    stack = load_stack(args.image or meta["image"])
    locs_all = pl.read_parquet(loc_dir / "localizations.parquet")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    patch_rows = []
    for frame in range(args.frames[0], args.frames[1]):
        frame_locs = locs_all.filter(pl.col("frame") == frame).sort("loc_id")
        raw = stack[frame]
        print(
            f"frame {frame}: {frame_locs.height} fixed detections, "
            f"free sigma bounds [{args.sigma_lo}, {args.sigma_hi}]"
        )
        rows, patches_, bmap, seconds = frame_diagnostic(
            raw, frame_locs, frame, sigma0, gain, offset, args
        )
        all_rows.extend(rows)
        patch_rows.extend(patches_)
        print(
            f"  {len(patches_)} patches in {seconds:.1f}s; "
            f"median sigma {np.median([r['fit_sigma'] for r in rows]):.3f}, "
            f"flux<{args.flux_cut:g}: {sum(r['fit_flux'] < args.flux_cut for r in rows)}"
        )
        if args.plot_frame is not None and frame == args.plot_frame:
            d_e = (raw.astype(float) - offset) / gain
            plot_frame(raw, d_e, bmap, rows, args, out / f"frame_{frame:03d}_diagnostic.png")

    diag = pl.DataFrame(all_rows)
    patch_diag = pl.DataFrame(patch_rows)
    diag.write_parquet(out / "free_sigma_localizations.parquet")
    patch_diag.write_parquet(out / "free_sigma_patches.parquet")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "image": str(pathlib.Path(args.image or meta["image"]).resolve()),
                "source_locs": str(loc_dir.resolve()),
                "frames": list(args.frames),
                "fixed_sigma_px": sigma0,
                "gain": gain,
                "offset": offset,
                "sigma_bounds": [args.sigma_lo, args.sigma_hi],
                "flux_cut": args.flux_cut,
                "wide_sigma": args.wide_sigma,
                "note": "Post-hoc diagnostic: one shared free sigma per final fixed-sigma patch.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--locs", required=True, help="directory from scripts/localize_movie.py")
    ap.add_argument("--image", default=None)
    ap.add_argument("--frames", type=int, nargs=2, default=(0, 1), metavar=("FIRST", "LAST"))
    ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--offset", type=float, default=None)
    ap.add_argument("--sigma-lo", type=float, default=0.6)
    ap.add_argument("--sigma-hi", type=float, default=8.0)
    ap.add_argument("--wide-sigma", type=float, default=1.6)
    ap.add_argument("--flux-cut", type=float, default=450.0)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--plot-frame", type=int, default=0)
    ap.add_argument("--out", default="data/free_sigma_diagnostic")
    main(ap.parse_args())
