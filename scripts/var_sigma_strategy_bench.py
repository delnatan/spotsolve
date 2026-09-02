"""Synthetic benchmark for variable-sigma out-of-focus filtering strategies.

The benchmark creates mixed fields with known in-focus and broad out-of-focus
emitters, runs the ordinary fixed-sigma detector, then runs the post-hoc
variable-sigma prune stage. It compares width-only classifier scores that are
intended to transfer across optical conditions:

  sigma_ratio       sigma_fit / sigma_focus
  width_z           log(sigma_fit / sigma_focus) / SE_log_sigma
  logbf_width       Laplace approximation for freeing log-sigma vs fixed sigma
  credible_ratio    lower confidence bound on sigma_ratio
"""

import argparse
import pathlib
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import spotsolve
from spotsolve import metrics, psf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import var_sigma_prune as vsp  # noqa: E402


GAIN = 2.3
OFFSET = 100.0
BACKGROUND = 26.0


def sample_points(rng, n, shape, border):
    H, W = shape
    return np.stack(
        [rng.uniform(border, H - border, n), rng.uniform(border, W - border, n)],
        axis=1,
    )


def simulate_mixed(
    seed,
    shape,
    sigma_focus,
    n_focus,
    n_broad,
    focus_amp,
    broad_amp,
    broad_ratio,
):
    rng = np.random.default_rng(seed)
    focus_pos = sample_points(rng, n_focus, shape, border=5.0 * sigma_focus)
    broad_pos = sample_points(rng, n_broad, shape, border=8.0 * sigma_focus)
    focus_A = rng.uniform(*focus_amp, n_focus)
    broad_A = rng.uniform(*broad_amp, n_broad)
    broad_sigma = sigma_focus * rng.uniform(*broad_ratio, n_broad)

    pos = np.vstack([focus_pos, broad_pos])
    amp = np.concatenate([focus_A, broad_A])
    sig = np.concatenate([np.full(n_focus, sigma_focus), broad_sigma])

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]] * 1.0
    clean = psf.model_var_sigma(
        psf.pack_var_sigma(BACKGROUND, amp, pos[:, 0], pos[:, 1], sig),
        yy,
        xx,
    )
    image_e = rng.poisson(clean).astype(float)
    adu = image_e * GAIN + OFFSET
    return {
        "adu": adu,
        "image_e": image_e,
        "focus_pos": focus_pos,
        "broad_pos": broad_pos,
        "broad_sigma": broad_sigma,
        "truth_pos": pos,
        "truth_sigma": sig,
        "truth_label": np.array(["focus"] * n_focus + ["broad"] * n_broad),
    }


def label_rows(rows, sim, sigma_focus):
    out = []
    if not rows:
        return out
    est = np.array([[r["y"], r["x"]] for r in rows])
    truth = sim["truth_pos"]
    truth_sig = sim["truth_sigma"]
    truth_label = sim["truth_label"]
    d = np.linalg.norm(est[:, None, :] - truth[None, :, :], axis=-1)
    gate = np.where(truth_label == "focus", 1.5 * sigma_focus, 1.5 * truth_sig)
    norm = d / gate[None, :]
    nearest = np.argmin(norm, axis=1)
    best = norm[np.arange(len(rows)), nearest]
    for r, j, nd in zip(rows, nearest, best):
        nr = dict(r)
        if nd <= 1.0:
            nr["truth_label"] = str(truth_label[j])
            nr["truth_sigma"] = float(truth_sig[j])
            nr["truth_dist_gate"] = float(nd)
        else:
            nr["truth_label"] = "unmatched"
            nr["truth_sigma"] = float("nan")
            nr["truth_dist_gate"] = float(nd)
        se_log = nr.get("se_log_sigma", float("nan"))
        width_z = nr.get("width_z", float("nan"))
        prior_width = np.log(8.0 / 0.7)
        if np.isfinite(se_log) and se_log > 0 and np.isfinite(width_z):
            nr["logbf_width"] = (
                0.5 * width_z * width_z
                + 0.5 * np.log(2.0 * np.pi)
                + np.log(se_log / prior_width)
            )
            nr["credible_ratio_90"] = float(
                np.exp(nr["log_sigma_ratio"] - 1.2815515655446004 * se_log)
            )
            nr["credible_ratio_95"] = float(
                np.exp(nr["log_sigma_ratio"] - 1.6448536269514722 * se_log)
            )
        else:
            nr["logbf_width"] = float("nan")
            nr["credible_ratio_90"] = float("nan")
            nr["credible_ratio_95"] = float("nan")
        out.append(nr)
    return out


def var_prune_rows(adu, fixed, sigma_focus):
    locs = pl.DataFrame(
        {
            "loc_id": np.arange(len(fixed.positions)),
            "frame": np.zeros(len(fixed.positions), dtype=int),
            "y": fixed.positions[:, 0] if len(fixed.positions) else np.empty(0),
            "x": fixed.positions[:, 1] if len(fixed.positions) else np.empty(0),
            "flux": fixed.amplitudes,
        }
    )
    args = SimpleNamespace(
        sigma_lo=0.7 * sigma_focus,
        sigma_hi=8.0 * sigma_focus,
        wide_sigma=1.6 * sigma_focus,
        prune_tau=2.0,
        k_max=12,
        impl="py",
    )
    rows, patch_rows, _, _ = vsp.frame_experiment(
        adu, locs, 0, sigma_focus, GAIN, OFFSET, args
    )
    return rows, patch_rows


def rates(df, pred_expr):
    focus = df.filter(pl.col("truth_label") == "focus")
    broad = df.filter(pl.col("truth_label") == "broad")
    unmatched = df.filter(pl.col("truth_label") == "unmatched")
    return {
        "focus_removed": float(focus.filter(pred_expr).height / max(focus.height, 1)),
        "broad_removed": float(broad.filter(pred_expr).height / max(broad.height, 1)),
        "unmatched_removed": float(unmatched.filter(pred_expr).height / max(unmatched.height, 1)),
        "focus_n": focus.height,
        "broad_n": broad.height,
        "unmatched_n": unmatched.height,
    }


def sweep(df):
    rows = []
    ratio_grid = np.linspace(1.1, 3.0, 39)
    z_grid = np.linspace(0.5, 5.0, 37)
    bf_grid = np.linspace(-3.0, 8.0, 45)
    for t in ratio_grid:
        r = rates(df, pl.col("sigma_ratio") > t)
        rows.append({"method": "sigma_ratio", "threshold": t, **r})
    for t in z_grid:
        r = rates(df, pl.col("width_z") > t)
        rows.append({"method": "width_z", "threshold": t, **r})
    for t in bf_grid:
        r = rates(df, (pl.col("log_sigma_ratio") > 0) & (pl.col("logbf_width") > t))
        rows.append({"method": "logbf_width", "threshold": t, **r})
    for t in ratio_grid:
        r = rates(df, pl.col("credible_ratio_90") > t)
        rows.append({"method": "credible_ratio_90", "threshold": t, **r})
    for t in ratio_grid:
        r = rates(df, pl.col("credible_ratio_95") > t)
        rows.append({"method": "credible_ratio_95", "threshold": t, **r})
    return pl.DataFrame(rows)


def best_at_fpr(sw, max_focus_removed):
    return (
        sw.filter(pl.col("focus_removed") <= max_focus_removed)
        .sort(["broad_removed", "unmatched_removed"], descending=[True, True])
        .group_by("method", maintain_order=True)
        .head(1)
        .sort("broad_removed", descending=True)
    )


def plot(df, sw, out_png):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {"focus": "#1f77b4", "broad": "#d62728", "unmatched": "#777777"}
    for label, color in colors.items():
        sub = df.filter(pl.col("truth_label") == label)
        ax[0].scatter(
            sub["sigma_ratio"],
            sub["flux"],
            s=8,
            alpha=0.35,
            linewidths=0,
            c=color,
            label=label,
        )
    ax[0].set_xlabel("sigma / in-focus sigma")
    ax[0].set_ylabel("flux (photoelectrons)")
    ax[0].set_title("normalized width vs flux")
    ax[0].legend(frameon=False)
    ax[0].grid(True, color="0.9", linewidth=0.6)

    for m in sw["method"].unique():
        sub = sw.filter(pl.col("method") == m).sort("focus_removed")
        ax[1].plot(sub["focus_removed"], sub["broad_removed"], label=m)
    ax[1].set_xlabel("in-focus removed")
    ax[1].set_ylabel("out-of-focus removed")
    ax[1].set_title("classifier tradeoff")
    ax[1].grid(True, color="0.9", linewidth=0.6)

    for label, color in colors.items():
        sub = df.filter(pl.col("truth_label") == label)
        ax[2].hist(
            sub["sigma_ratio"].to_numpy(),
            bins=np.linspace(0.5, 5.0, 70),
            histtype="step",
            color=color,
            label=label,
            density=True,
        )
    ax[2].set_xlabel("sigma / in-focus sigma")
    ax[2].set_ylabel("density")
    ax[2].set_title("width score populations")
    ax[2].grid(True, color="0.9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    patch_rows_all = []
    run_rows = []

    for sigma_focus in args.sigmas:
        n_focus = int(round(args.focus_density * args.size * args.size))
        n_broad = int(round(args.broad_density * args.size * args.size))
        for seed in range(args.seeds):
            sim = simulate_mixed(
                seed=10_000 + 100 * int(10 * sigma_focus) + seed,
                shape=(args.size, args.size),
                sigma_focus=sigma_focus,
                n_focus=n_focus,
                n_broad=n_broad,
                focus_amp=tuple(args.focus_amp),
                broad_amp=tuple(args.broad_amp),
                broad_ratio=tuple(args.broad_ratio),
            )
            fixed = spotsolve.detect(
                sim["adu"],
                sigma=sigma_focus,
                offset=OFFSET,
                gain=GAIN,
                impl="py",
                verbose=0,
            )
            var_rows, patch_rows = var_prune_rows(sim["adu"], fixed, sigma_focus)
            labeled = label_rows(var_rows, sim, sigma_focus)
            for r in labeled:
                r["sigma_focus"] = sigma_focus
                r["seed"] = seed
            rows.extend(labeled)
            for p in patch_rows:
                p["sigma_focus"] = sigma_focus
                p["seed"] = seed
            patch_rows_all.extend(patch_rows)

            m_fixed_focus = metrics.match(sim["focus_pos"], fixed.positions, radius=1.5 * sigma_focus)
            m_var_focus = metrics.match(
                sim["focus_pos"],
                np.array([[r["y"], r["x"]] for r in labeled if r["truth_label"] == "focus"]),
                radius=1.5 * sigma_focus,
            )
            run_rows.append(
                {
                    "sigma_focus": sigma_focus,
                    "seed": seed,
                    "true_focus": n_focus,
                    "true_broad": n_broad,
                    "fixed_n": len(fixed.positions),
                    "var_n": len(var_rows),
                    "fixed_focus_recall": m_fixed_focus.recall,
                    "var_labeled_focus_recall": m_var_focus.recall,
                    "var_broad_labeled": sum(1 for r in labeled if r["truth_label"] == "broad"),
                    "var_unmatched": sum(1 for r in labeled if r["truth_label"] == "unmatched"),
                }
            )
            print(
                f"sigma {sigma_focus:.2f} seed {seed}: fixed {len(fixed.positions)}, "
                f"var {len(var_rows)}, labels "
                f"focus={sum(r['truth_label']=='focus' for r in labeled)} "
                f"broad={sum(r['truth_label']=='broad' for r in labeled)} "
                f"unmatched={sum(r['truth_label']=='unmatched' for r in labeled)}"
            )

    df = pl.DataFrame(rows)
    patches = pl.DataFrame(patch_rows_all)
    runs = pl.DataFrame(run_rows)
    sw = sweep(df)
    best05 = best_at_fpr(sw, 0.05)
    best10 = best_at_fpr(sw, 0.10)

    df.write_parquet(out / "classified_localizations.parquet")
    patches.write_parquet(out / "patches.parquet")
    runs.write_parquet(out / "runs.parquet")
    sw.write_parquet(out / "classifier_sweep.parquet")
    best05.write_csv(out / "best_at_5pct_focus_removed.csv")
    best10.write_csv(out / "best_at_10pct_focus_removed.csv")
    plot(df, sw, out / "strategy_benchmark.png")

    print("\nrun summary")
    print(runs.select(pl.col("fixed_n").mean(), pl.col("var_n").mean(), pl.col("fixed_focus_recall").mean()))
    print("\nbest at <=5% in-focus removed")
    print(best05)
    print("\nbest at <=10% in-focus removed")
    print(best10)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/var_sigma_strategy_bench")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--sigmas", type=float, nargs="*", default=[1.0, 1.2, 1.6])
    ap.add_argument("--focus-density", type=float, default=0.010)
    ap.add_argument("--broad-density", type=float, default=0.003)
    ap.add_argument("--focus-amp", type=float, nargs=2, default=[700.0, 1800.0])
    ap.add_argument("--broad-amp", type=float, nargs=2, default=[1200.0, 4500.0])
    ap.add_argument("--broad-ratio", type=float, nargs=2, default=[1.8, 4.0])
    main(ap.parse_args())
