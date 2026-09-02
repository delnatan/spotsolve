"""Variable-sigma prune experiment for fixed-sigma spotsolve detections.

The fixed-sigma detector first finds an overcomplete explanation of the frame.
This post-hoc experiment then refits each final patch with one sigma per
emitter and allows Bayes-factor pruning under that wider model. The intended
use is visual assessment of whether out-of-focus structure collapses into
fewer, broader PSFs before width-based filtering.
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

from spotsolve import calibrate, core, evidence, lmga, patches, psf
from spotsolve import moves
from spotsolve.structs import FitResult


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


def var_bounds(K, h, w, b_max, A_max, sigma_lo, sigma_hi):
    a_min = max(moves.A_MIN, core.A_MIN_REL * A_max)
    lo = [0.0]
    hi = [b_max]
    for _ in range(K):
        lo += [a_min, -0.5, -0.5, sigma_lo]
        hi += [A_max, h - 0.5, w - 0.5, sigma_hi]
    return np.asarray(lo), np.asarray(hi)


def fit_var(sub, yy, xx, halo, theta0, sigma0, sigma_lo, sigma_hi, max_iter=180, impl="py"):
    K = (len(theta0) - 1) // 4
    smax = max(float(sub.max()), 1.0)
    b_max = max(smax * 4.0, 10.0)
    A_max = 8.0 * smax / psf.peak_factor(sigma0)
    lo, hi = var_bounds(K, sub.shape[0], sub.shape[1], b_max, A_max, sigma_lo, sigma_hi)
    th0 = np.clip(np.asarray(theta0, float), lo + 1e-9, hi - 1e-9)
    if impl == "rs":
        import spotsolve_rs

        theta, I, F, n_iter, converged, stalled = spotsolve_rs.lmcl_fit_var_sigma(
            np.ascontiguousarray(th0, dtype=float),
            int(sub.shape[0]),
            int(sub.shape[1]),
            np.ascontiguousarray(sub, dtype=float),
            np.ascontiguousarray(halo, dtype=float),
            np.ascontiguousarray(lo, dtype=float),
            np.ascontiguousarray(hi, dtype=float),
            max_iter,
        )
        return FitResult(theta=theta, I=I, F=F, n_iter=n_iter, converged=converged, stalled=stalled)
    if impl != "py":
        raise ValueError(f"impl must be 'py' or 'rs', got {impl!r}")
    return lmga.fit(
        th0,
        yy,
        xx,
        sigma0,
        sub,
        lo,
        hi,
        halo=halo,
        max_iter=max_iter,
        tol_obj=core.EVIDENCE_TOL_OBJ,
        free_sigma="per_emitter",
    )


def log_bf_remove_var(full, reduced, K_full, sumA_full, sumA_reduced, lam, A_s, sigma_width):
    """Positive favours the reduced model, mirroring evidence.log_bf_remove."""
    ld_full, ok_full = evidence.logdet(full.F)
    ld_reduced, ok_reduced = evidence.logdet(reduced.F)
    if not ok_reduced:
        return -np.inf
    if not ok_full:
        return np.inf
    add_log_bf = (
        (reduced.I - full.I)
        + np.log(lam)
        - np.log(K_full)
        - np.log(A_s)
        - (sumA_full - sumA_reduced) / A_s
        - np.log(sigma_width)
        + 2.0 * np.log(2.0 * np.pi)
        - 0.5 * (ld_full - ld_reduced)
    )
    return float(-add_log_bf)


def remove_emitter(theta, k):
    b, A, cy, cx, sig = psf.unpack_var_sigma(theta)
    keep = np.ones(len(A), dtype=bool)
    keep[k] = False
    return psf.pack_var_sigma(b, A[keep], cy[keep], cx[keep], sig[keep])


def var_se(F, K):
    """Per-emitter standard errors for variable-sigma theta."""
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return np.full((K, 4), np.nan)
    out = np.full((K, 4), np.nan)
    for k in range(K):
        for j in range(4):
            v = cov[1 + 4 * k + j, 1 + 4 * k + j]
            if np.isfinite(v) and v > 0:
                out[k, j] = np.sqrt(v)
    return out


def fit_and_prune_patch(
    d_e,
    bmap,
    positions,
    amplitudes,
    patch,
    sigma0,
    sigma_lo,
    sigma_hi,
    lam,
    A_s,
    tau,
    impl,
):
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
    theta = psf.pack_var_sigma(
        level,
        amplitudes[patch.indices],
        loc[:, 0],
        loc[:, 1],
        np.full(len(patch.indices), sigma0),
    )
    fit = fit_var(sub, yy, xx, halo, theta, sigma0, sigma_lo, sigma_hi, impl=impl)
    kept_sources = [int(i) for i in patch.indices]
    n_start = len(patch.indices)
    removed = 0
    decisions = []

    while True:
        _, A, _, _, _ = psf.unpack_var_sigma(fit.theta)
        K = len(A)
        if K <= 1:
            break
        sigma_width = sigma_hi - sigma_lo
        ld_full, ok_full = evidence.logdet(fit.F)

        forced = []
        if ok_full:
            try:
                cov = np.linalg.inv(fit.F)
                for k in range(K):
                    v = cov[1 + 4 * k, 1 + 4 * k]
                    if np.isfinite(v) and v > 0 and A[k] < tau * np.sqrt(v):
                        forced.append(k)
            except np.linalg.LinAlgError:
                forced = list(range(K))
        else:
            forced = list(range(K))

        best = None
        for k in range(K):
            theta_reduced = remove_emitter(fit.theta, k)
            reduced = fit_var(sub, yy, xx, halo, theta_reduced, sigma0, sigma_lo, sigma_hi, impl=impl)
            _, A_red, _, _, _ = psf.unpack_var_sigma(reduced.theta)
            if k in forced:
                log_bf = np.inf
            else:
                log_bf = log_bf_remove_var(
                    fit,
                    reduced,
                    K,
                    float(np.sum(A)),
                    float(np.sum(A_red)),
                    lam,
                    A_s,
                    sigma_width,
                )
            if np.isfinite(log_bf) and log_bf > 0:
                if best is None or log_bf > best[0]:
                    best = (log_bf, k, reduced)
            elif log_bf == np.inf:
                best = (log_bf, k, reduced)
                break
        if best is None:
            break
        decisions.append({"removed_local": int(best[1]), "log_bf_remove": float(best[0])})
        del kept_sources[int(best[1])]
        fit = best[2]
        removed += 1

    return fit, n_start, removed, decisions, kept_sources


def frame_experiment(raw, locs, frame, sigma0, gain, offset, args):
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
    lam = max(float(locs.height) / float(raw.size), 1e-6)
    A_s = max(float(np.mean(amplitudes)), 1.0)

    rows = []
    patch_rows = []
    t0 = time.time()
    for patch_id, p in enumerate(pset):
        fit, n_start, n_removed, decisions, kept_sources = fit_and_prune_patch(
            d_e,
            bmap,
            positions,
            amplitudes,
            p,
            sigma0,
            args.sigma_lo,
            args.sigma_hi,
            lam,
            A_s,
            args.prune_tau,
            args.impl,
        )
        _, A, cy, cx, sig = psf.unpack_var_sigma(fit.theta)
        se = var_se(fit.F, len(A))
        patch_rows.append(
            {
                "frame": frame,
                "patch_id": patch_id,
                "n_start": n_start,
                "n_final": len(A),
                "n_removed": n_removed,
                "converged": fit.converged,
                "stalled": fit.stalled,
                "n_iter": fit.n_iter,
                "I": float(fit.I),
                "remove_log_bf_max": max(
                    [d["log_bf_remove"] for d in decisions], default=float("nan")
                ),
            }
        )
        for j in range(len(A)):
            source_idx = int(kept_sources[j])
            se_log_sigma = se[j, 3] / sig[j] if sig[j] > 0 else float("nan")
            log_sigma_ratio = float(np.log(sig[j] / sigma0))
            rows.append(
                {
                    "source_loc_id": int(locs["loc_id"][source_idx]),
                    "frame": frame,
                    "patch_id": patch_id,
                    "patch_n_start": n_start,
                    "patch_n_final": len(A),
                    "y": float(cy[j] + p.y0),
                    "x": float(cx[j] + p.x0),
                    "flux": float(A[j]),
                    "se_flux": float(se[j, 0]),
                    "sigma": float(sig[j]),
                    "sigma_ratio": float(sig[j] / sigma0),
                    "log_sigma_ratio": log_sigma_ratio,
                    "se_sigma": float(se[j, 3]),
                    "se_log_sigma": float(se_log_sigma),
                    "width_z": float(log_sigma_ratio / se_log_sigma)
                    if np.isfinite(se_log_sigma) and se_log_sigma > 0
                    else float("nan"),
                    "wide": bool(sig[j] > args.wide_sigma),
                    "converged": fit.converged,
                    "stalled": fit.stalled,
                }
            )
    return rows, patch_rows, bmap, time.time() - t0


def render_var_model(shape, bmap, rows):
    out = np.array(bmap, dtype=float, copy=True)
    for r in rows:
        out += calibrate.render_model(
            np.array([[r["y"], r["x"]]], dtype=float),
            np.array([r["flux"]], dtype=float),
            float(r["sigma"]),
            shape,
            0.0,
        )
    return out


def plot_frame(raw, d_e, bmap, rows, fixed_locs, args, out_png):
    y = np.array([r["y"] for r in rows])
    x = np.array([r["x"] for r in rows])
    flux = np.array([r["flux"] for r in rows])
    sig = np.array([r["sigma"] for r in rows])
    wide = sig > args.wide_sigma
    kept_source_ids = {int(r["source_loc_id"]) for r in rows}
    fixed_pruned = fixed_locs.filter(~pl.col("loc_id").is_in(list(kept_source_ids)))
    model = render_var_model(raw.shape, bmap, rows)
    nr = (d_e - model) / np.sqrt(np.maximum(model, 1e-6))

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    ax = ax.ravel()
    ax[0].imshow(raw, cmap="gray")
    ax[0].scatter(x[~wide], y[~wide], s=8, c="#00d17a", linewidths=0, alpha=0.8)
    ax[0].scatter(x[wide], y[wide], s=18, facecolors="none", edgecolors="#ff4d8d", linewidths=0.8)
    if fixed_pruned.height:
        ax[0].scatter(
            fixed_pruned["x"].to_numpy(),
            fixed_pruned["y"].to_numpy(),
            s=18,
            marker="x",
            c="#ffb000",
            linewidths=0.8,
            alpha=0.8,
        )
    ax[0].set_title(
        f"variable-sigma prune: {len(rows)} kept, {fixed_pruned.height} pruned, {wide.sum()} wide",
        fontsize=9,
    )

    ax[1].imshow(raw, cmap="gray")
    sc = ax[1].scatter(
        x,
        y,
        s=10,
        c=sig,
        cmap="viridis",
        vmin=args.sigma_lo,
        vmax=args.sigma_hi,
        linewidths=0,
    )
    ax[1].set_title("per-emitter fitted sigma", fontsize=9)
    fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].scatter(sig, flux, s=8, c=np.where(wide, "#ff4d8d", "#1f77b4"), alpha=0.45, linewidths=0)
    ax[2].axvline(args.sigma0, color="gray", lw=1)
    ax[2].axvline(args.wide_sigma, color="black", lw=1, ls="--")
    ax[2].set_xlabel("PSF sigma (px)")
    ax[2].set_ylabel("flux (photoelectrons)")
    ax[2].set_title("flux vs width after pruning", fontsize=9)
    ax[2].grid(True, color="0.9", linewidth=0.6)

    ax[3].imshow(nr, cmap="RdBu_r", vmin=-5, vmax=5)
    ax[3].set_title(
        f"residual, med {np.median(nr):+.2f}, "
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
    all_patch_rows = []
    for frame in range(args.frames[0], args.frames[1]):
        frame_locs = locs_all.filter(pl.col("frame") == frame).sort("loc_id")
        raw = stack[frame]
        print(f"frame {frame}: starting from {frame_locs.height} fixed-sigma detections")
        rows, patch_rows, bmap, seconds = frame_experiment(raw, frame_locs, frame, sigma0, gain, offset, args)
        all_rows.extend(rows)
        all_patch_rows.extend(patch_rows)
        removed = frame_locs.height - len(rows)
        wide = sum(r["wide"] for r in rows)
        print(
            f"  kept {len(rows)}, pruned {removed}, wide {wide}, "
            f"median sigma {np.median([r['sigma'] for r in rows]):.3f}, {seconds:.1f}s"
        )
        if args.plot_frame is not None and frame == args.plot_frame:
            d_e = (raw.astype(float) - offset) / gain
            plot_frame(
                raw,
                d_e,
                bmap,
                rows,
                frame_locs,
                args,
                out / f"frame_{frame:03d}_var_sigma_prune.png",
            )

    pl.DataFrame(all_rows).write_parquet(out / "var_sigma_localizations.parquet")
    pl.DataFrame(all_patch_rows).write_parquet(out / "var_sigma_patches.parquet")
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
                "wide_sigma": args.wide_sigma,
                "prune_tau": args.prune_tau,
                "impl": args.impl,
                "note": "Post-hoc experiment: per-emitter sigma fit with BF pruning inside fixed-sigma patches.",
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
    ap.add_argument("--sigma-lo", type=float, default=0.7)
    ap.add_argument("--sigma-hi", type=float, default=8.0)
    ap.add_argument("--wide-sigma", type=float, default=1.6)
    ap.add_argument("--prune-tau", type=float, default=2.0)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--plot-frame", type=int, default=0)
    ap.add_argument("--impl", choices=["py", "rs"], default="py")
    ap.add_argument("--out", default="data/var_sigma_prune")
    main(ap.parse_args())
