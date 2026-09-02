"""Apply the variable-sigma out-of-focus filter to localization parquet files.

Inputs:
  * fixed-sigma localization directory from scripts/localize_movie.py
  * variable-sigma prune directory from scripts/var_sigma_prune.py

Outputs:
  * localizations.parquet            kept in-focus variable-sigma survivors
  * out_of_focus.parquet             variable-sigma survivors rejected by width
  * pruned.parquet                   fixed-sigma detections removed by BF pruning
  * all_variable_sigma.parquet       all post-prune variable-sigma survivors
  * frames.parquet                   per-frame accounting summary
  * meta.json                        cutoff and provenance
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tifffile

from spotsolve import calibrate, core


def load_stack(path):
    img = tifffile.imread(path)
    if img.ndim == 2:
        img = img[None]
    elif img.ndim != 3:
        raise SystemExit(f"expected a 2-D image or 3-D stack, got {img.shape}")
    return img


def add_fixed_columns(var, fixed):
    fixed_small = fixed.select(
        pl.col("loc_id").alias("source_loc_id"),
        pl.col("y").alias("fixed_y"),
        pl.col("x").alias("fixed_x"),
        pl.col("flux").alias("fixed_flux"),
        pl.col("se_y").alias("fixed_se_y"),
        pl.col("se_x").alias("fixed_se_x"),
        pl.col("se_pos").alias("fixed_se_pos"),
        pl.col("bg").alias("fixed_bg"),
    )
    return var.join(fixed_small, on="source_loc_id", how="left")


def anti_join_fixed(fixed, var):
    surviving_ids = var.select("source_loc_id").unique()
    return fixed.join(
        surviving_ids,
        left_on="loc_id",
        right_on="source_loc_id",
        how="anti",
    )


def stamp_mask(shape, y, x, sigma, radius_factor=3.0):
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for cy, cx, sig in zip(y, x, sigma):
        radius = radius_factor * float(sig)
        r0 = int(np.ceil(radius))
        y0, y1 = max(0, int(cy) - r0), min(shape[0], int(cy) + r0 + 1)
        x0, x1 = max(0, int(cx) - r0), min(shape[1], int(cx) + r0 + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        d2 = (yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2
        mask[y0:y1, x0:x1] |= d2 <= radius * radius
    return mask


def render_variable_sigma(rows, shape):
    model = np.zeros(shape, dtype=float)
    for row in rows.iter_rows(named=True):
        model += calibrate.render_model(
            np.array([[row["y"], row["x"]]], dtype=float),
            np.array([row["flux"]], dtype=float),
            float(row["sigma"]),
            shape,
            0.0,
        )
    return model


def background_excluding_nuisance(d_e, kept, rejected, sigma_focus):
    pos = kept.select("y", "x").to_numpy() if kept.height else np.empty((0, 2))
    free = calibrate.emitter_free_mask(
        d_e.shape, pos, sigma_focus, core.BG_MASK_RADIUS
    )
    if rejected.height:
        free &= ~stamp_mask(
            d_e.shape,
            rejected["y"].to_numpy(),
            rejected["x"].to_numpy(),
            rejected["sigma"].to_numpy(),
            core.BG_MASK_RADIUS,
        )
    scalar = max(
        calibrate.robust_background(
            d_e, pos, sigma_focus, core.BG_MASK_RADIUS, free=free
        ),
        core.BG_FLOOR,
    )
    return core.background_map(d_e, pos, sigma_focus, fallback=scalar, free=free)


def sample_background(bmap, positions):
    if len(positions) == 0:
        return np.empty(0)
    h, w = bmap.shape
    yi = np.clip(np.rint(positions[:, 0]).astype(int), 0, h - 1)
    xi = np.clip(np.rint(positions[:, 1]).astype(int), 0, w - 1)
    return bmap[yi, xi]


def final_refine_frame(raw, kept, nuisance, sigma_focus, gain, offset, k_max, impl):
    if kept.height == 0:
        return kept
    d_e = (raw.astype(float) - offset) / gain
    positions = kept.select("y", "x").to_numpy()
    amplitudes = kept["flux"].to_numpy()
    background = background_excluding_nuisance(d_e, kept, nuisance, sigma_focus)
    nuisance_model = render_variable_sigma(nuisance, raw.shape) if nuisance.height else 0.0
    bmap = background + nuisance_model

    if impl == "rs":
        try:
            import spotsolve_rs

            pos, amp, se = spotsolve_rs.refine(
                np.ascontiguousarray(d_e, dtype=float),
                np.ascontiguousarray(positions, dtype=float),
                np.ascontiguousarray(amplitudes, dtype=float),
                float(sigma_focus),
                np.ascontiguousarray(bmap, dtype=float),
                int(k_max),
                int(core.REFINE_MAX_ITER),
                int(core.REFINE_SWEEPS),
                float(core.REFINE_TOL),
            )
        except ImportError:
            pos, amp, se = core.refine(
                d_e, positions, amplitudes, sigma_focus, bmap, k_max=k_max
            )
    else:
        pos, amp, se = core.refine(
            d_e, positions, amplitudes, sigma_focus, bmap, k_max=k_max
        )

    out = kept.rename(
        {
            "y": "var_y",
            "x": "var_x",
            "flux": "var_flux",
            "sigma": "var_sigma",
        }
    ).with_columns(
        pl.Series("y", pos[:, 0]),
        pl.Series("x", pos[:, 1]),
        pl.Series("flux", amp),
        pl.Series("se_flux", se[:, 0]),
        pl.Series("se_y", se[:, 1]),
        pl.Series("se_x", se[:, 2]),
        pl.Series("se_pos", np.hypot(se[:, 1], se[:, 2])),
        pl.Series("bg", sample_background(background, pos)),
        pl.lit(float(sigma_focus)).alias("sigma"),
    )
    return out


def final_refine(stack, kept, rejected, sigma_focus, gain, offset, k_max, impl):
    parts = []
    for frame in kept["frame"].unique().sort():
        k = kept.filter(pl.col("frame") == frame)
        # Broad rejected spots are physical out-of-focus nuisance structure.
        # Narrow rejected spots are usually over-fit pixel-scale texture/noise,
        # so rendering them back would preserve the false positive.
        r = rejected.filter(
            (pl.col("frame") == frame) & (pl.col("reject_reason") == "too_wide")
        )
        parts.append(
            final_refine_frame(
                stack[int(frame)],
                k,
                r,
                sigma_focus,
                gain,
                offset,
                k_max,
                impl,
            )
        )
    return pl.concat(parts, how="diagonal") if parts else kept


def frame_summary(fixed, var, kept, rejected, pruned):
    frames = fixed.select("frame").unique().sort("frame")
    parts = []
    for row in frames.iter_rows(named=True):
        frame = row["frame"]
        f = fixed.filter(pl.col("frame") == frame)
        v = var.filter(pl.col("frame") == frame)
        k = kept.filter(pl.col("frame") == frame)
        r = rejected.filter(pl.col("frame") == frame)
        p = pruned.filter(pl.col("frame") == frame)
        parts.append(
            {
                "frame": int(frame),
                "fixed_n": f.height,
                "var_pruned_n": p.height,
                "var_survivor_n": v.height,
                "out_of_focus_n": r.height,
                "kept_n": k.height,
                "kept_fraction_of_fixed": k.height / max(f.height, 1),
                "out_of_focus_fraction_of_survivors": r.height / max(v.height, 1),
                "median_sigma_ratio": float(v["sigma_ratio"].median()) if v.height else float("nan"),
                "q95_sigma_ratio": float(v["sigma_ratio"].quantile(0.95)) if v.height else float("nan"),
                "max_sigma_ratio": float(v["sigma_ratio"].max()) if v.height else float("nan"),
                "median_kept_flux": float(k["flux"].median()) if k.height else float("nan"),
                "median_kept_se_pos": float(k["se_pos"].median())
                if k.height and "se_pos" in k.columns
                else float("nan"),
                "median_rejected_flux": float(r["flux"].median()) if r.height else float("nan"),
            }
        )
    return pl.DataFrame(parts)


def plot_frame(stack, frame, kept, rejected, pruned, sigma_ratio_min, sigma_ratio_max, out_png, crop=None):
    raw = stack[frame]
    k = kept.filter(pl.col("frame") == frame)
    r = rejected.filter(pl.col("frame") == frame)
    r_lo = r.filter(pl.col("reject_reason") == "too_narrow")
    r_hi = r.filter(pl.col("reject_reason") == "too_wide")
    p = pruned.filter(pl.col("frame") == frame)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    ax[0].imshow(raw, cmap="gray")
    if k.height:
        ax[0].scatter(k["x"], k["y"], s=8, c="#00d17a", linewidths=0, alpha=0.8)
    if r_hi.height:
        ax[0].scatter(
            r_hi["x"],
            r_hi["y"],
            s=18,
            facecolors="none",
            edgecolors="#ff4d8d",
            linewidths=0.8,
            alpha=0.9,
        )
    if r_lo.height:
        ax[0].scatter(
            r_lo["x"],
            r_lo["y"],
            s=18,
            facecolors="none",
            edgecolors="#00a6ff",
            linewidths=0.8,
            alpha=0.9,
        )
    if p.height:
        ax[0].scatter(p["x"], p["y"], s=18, marker="x", c="#ffb000", linewidths=0.8, alpha=0.8)
    ax[0].set_title(
        f"frame {frame}: kept {k.height}, narrow {r_lo.height}, wide {r_hi.height}, pruned {p.height}",
        fontsize=9,
    )

    ax[1].imshow(raw, cmap="gray")
    if k.height or r.height:
        v = pl.concat([k, r], how="diagonal")
        sc = ax[1].scatter(
            v["x"],
            v["y"],
            s=10,
            c=v["sigma_ratio"],
            cmap="viridis",
            vmin=0.7,
            vmax=max(3.0, float(v["sigma_ratio"].quantile(0.99))),
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].set_title("sigma / in-focus sigma", fontsize=9)

    if k.height:
        ax[2].scatter(k["sigma_ratio"], k["flux"], s=8, c="#1f77b4", alpha=0.35, linewidths=0, label="kept")
    if r_lo.height:
        ax[2].scatter(r_lo["sigma_ratio"], r_lo["flux"], s=10, c="#00a6ff", alpha=0.55, linewidths=0, label="too narrow")
    if r_hi.height:
        ax[2].scatter(r_hi["sigma_ratio"], r_hi["flux"], s=10, c="#ff4d8d", alpha=0.55, linewidths=0, label="too wide")
    ax[2].axvline(sigma_ratio_min, color="black", lw=1, ls="--")
    ax[2].axvline(sigma_ratio_max, color="black", lw=1, ls="--")
    ax[2].set_xlabel("sigma / in-focus sigma")
    ax[2].set_ylabel("flux (photoelectrons)")
    ax[2].set_title("width filter", fontsize=9)
    ax[2].grid(True, color="0.9", linewidth=0.6)
    ax[2].legend(frameon=False)

    for a in (ax[0], ax[1]):
        a.set_xticks([])
        a.set_yticks([])
        if crop is not None:
            x0, y0, w, h = crop
            a.set_xlim(x0, x0 + w)
            a.set_ylim(y0 + h, y0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(args):
    fixed_dir = pathlib.Path(args.fixed_locs)
    var_dir = pathlib.Path(args.var_locs)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fixed_meta = json.loads((fixed_dir / "meta.json").read_text())
    var_meta = json.loads((var_dir / "meta.json").read_text())
    fixed = pl.read_parquet(fixed_dir / "localizations.parquet")
    var = pl.read_parquet(var_dir / "var_sigma_localizations.parquet")
    if "source_loc_id" not in var.columns:
        raise SystemExit("variable-sigma parquet must include source_loc_id; rerun scripts/var_sigma_prune.py")

    var = add_fixed_columns(var, fixed)
    var = var.with_columns(
        pl.when(pl.col("sigma_ratio") < args.sigma_ratio_min)
        .then(pl.lit("too_narrow"))
        .when(pl.col("sigma_ratio") > args.sigma_ratio_max)
        .then(pl.lit("too_wide"))
        .otherwise(pl.lit("kept"))
        .alias("reject_reason")
    )
    kept = var.filter(pl.col("reject_reason") == "kept")
    rejected = var.filter(pl.col("reject_reason") != "kept")
    pruned = anti_join_fixed(fixed, var)
    if args.final_refine:
        stack = load_stack(fixed_meta["image"])
        kept = final_refine(
            stack,
            kept,
            rejected,
            float(fixed_meta["sigma_px"]),
            float(fixed_meta["gain"]),
            float(fixed_meta["camera_offset_adu"]),
            args.k_max,
            args.final_refine_impl,
        )
    frames = frame_summary(fixed, var, kept, rejected, pruned)

    kept.write_parquet(out / "localizations.parquet")
    rejected.write_parquet(out / "out_of_focus.parquet")
    pruned.write_parquet(out / "pruned.parquet")
    var.write_parquet(out / "all_variable_sigma.parquet")
    frames.write_parquet(out / "frames.parquet")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "fixed_locs": str(fixed_dir.resolve()),
                "var_sigma_locs": str(var_dir.resolve()),
                "image": fixed_meta["image"],
                "frames": fixed_meta["frames"],
                "fixed_sigma_px": fixed_meta["sigma_px"],
                "sigma_ratio_min": args.sigma_ratio_min,
                "sigma_ratio_max": args.sigma_ratio_max,
                "sigma_min_px": args.sigma_ratio_min * fixed_meta["sigma_px"],
                "sigma_max_px": args.sigma_ratio_max * fixed_meta["sigma_px"],
                "gain": fixed_meta["gain"],
                "offset": fixed_meta["camera_offset_adu"],
                "var_sigma_bounds": var_meta["sigma_bounds"],
                "final_refine": args.final_refine,
                "final_refine_impl": args.final_refine_impl if args.final_refine else None,
                "rule": "keep variable-sigma survivors with sigma_ratio_min <= sigma_fit / fixed_sigma <= sigma_ratio_max",
            },
            indent=2,
        )
        + "\n"
    )

    if args.plot_frame is not None:
        stack = load_stack(fixed_meta["image"])
        plot_frame(
            stack,
            args.plot_frame,
            kept,
            rejected,
            pruned,
            args.sigma_ratio_min,
            args.sigma_ratio_max,
            out / f"frame_{args.plot_frame:03d}_physical_width_filter.png",
            crop=args.crop,
        )

    print(frames.select(pl.sum("fixed_n"), pl.sum("var_pruned_n"), pl.sum("out_of_focus_n"), pl.sum("kept_n")))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixed-locs", required=True)
    ap.add_argument("--var-locs", required=True)
    ap.add_argument("--sigma-ratio-min", type=float, default=0.8)
    ap.add_argument("--sigma-ratio-max", "--sigma-ratio-cut", dest="sigma_ratio_max", type=float, default=1.2)
    ap.add_argument("--plot-frame", type=int, default=0)
    ap.add_argument("--crop", type=float, nargs=4, metavar=("X0", "Y0", "W", "H"))
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--final-refine", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--final-refine-impl", choices=["py", "rs"], default="rs")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
