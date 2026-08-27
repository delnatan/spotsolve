"""Bead-matched simulations: what perfect recall LOOKS like at this density.

Generates fields with the same shape, gain, offset, background and brightness
distribution as beads_60x_still.tif, sweeps the emitter density through and
past the real one, and renders detections against known truth. The point is
calibration by eye: if the simulated panels at the real density show clean
recovery and a structureless residual, then whatever the real data does
differently is a property of the DATA, not of the algorithm.

Matched to beads_60x_still.tif:
    39x39, gain 4.23 ADU/e-, offset 100, background ~4 e-,
    peak brightness ~100-200 e-, density ~0.034 emitters/px^2 (N~51)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile

import psf
import audit
import boxsolve
import metrics
import simulate

SHAPE = (39, 39)
GAIN = 4.23
OFFSET = 100.0
BG_E = 4.0
SIGMA = 1.2
AMP_RANGE = (900.0, 1900.0)      # flux -> peak ~94-198 e-, matching the beads
DENSITIES = [0.015, 0.034, 0.055]  # sparse / the real density / crowded


def simulate_beadlike(density, seed):
    n = max(1, int(round(density * SHAPE[0] * SHAPE[1])))
    # border=1.0, not 3.0. The old value was numerically identical to the
    # detector's own `border_margin` (2.5*sigma = 3 px at sigma 1.2), so the
    # simulation placed emitters only where the detector was already willing
    # to look. That made the one failure the real beads show most plainly --
    # clean, well-separated PSFs 1-2 px from the frame going undetected --
    # structurally impossible to reproduce here. A calibration image must be
    # allowed to populate the region the detector is suspected of mishandling.
    sim = simulate.simulate(shape=SHAPE, n_emitters=n, background=BG_E,
                            amplitude_range=AMP_RANGE, sigma=SIGMA,
                            border=1.0, seed=seed)
    return sim, sim.image * GAIN + OFFSET      # raw ADU, like the camera


def _mark_audit(ax, a):
    """Ring every place the residual audit says the model is wrong: yellow for
    unexplained flux (a missed emitter), cyan for over-modelled flux (piled-up
    PSFs). An empty panel is the result to want."""
    if len(a["positive"]):
        ax.scatter(a["positive"][:, 1], a["positive"][:, 0], s=110,
                   facecolors="none", edgecolors="yellow", linewidths=1.4)
    if len(a["negative"]):
        ax.scatter(a["negative"][:, 1], a["negative"][:, 0], s=110,
                   facecolors="none", edgecolors="cyan", linewidths=1.4)


def main(seed=1):
    real = tifffile.imread("beads_60x_still.tif").astype(float)
    nrow = len(DENSITIES) + 1
    fig, ax = plt.subplots(nrow, 3, figsize=(11.5, 3.5 * nrow))
    rows = []

    for r, dens in enumerate(DENSITIES):
        sim, adu = simulate_beadlike(dens, seed + r)
        res = boxsolve.detect_boxes(adu, sigma=SIGMA, offset=OFFSET, verbose=0)
        m = metrics.match(sim.positions, res.positions, radius=1.5)
        d_e = (adu - OFFSET) / res.gain
        nr = (d_e - res.model_image) / np.sqrt(np.maximum(res.model_image, 1e-6))
        a = audit.audit_result(d_e, res.model_image, SIGMA, z_thresh=5.0)
        rows.append((dens, m, res, nr, a))

        ax[r, 0].imshow(adu, cmap="gray")
        ax[r, 0].set_ylabel(f"density {dens:.3f}\nN_true={m.n_true}", fontsize=9)
        ax[r, 0].set_title("simulated data (ADU)" if r == 0 else "")

        ax[r, 1].imshow(adu, cmap="gray")
        ax[r, 1].scatter(sim.positions[:, 1], sim.positions[:, 0], s=70,
                         facecolors="none", edgecolors="lime", linewidths=1.1,
                         label="truth")
        ax[r, 1].scatter(res.positions[:, 1], res.positions[:, 0], s=28,
                         marker="+", c="red", linewidths=1.1, label="detected")
        ax[r, 1].set_title("truth (o) vs detected (+)" if r == 0 else "")
        ax[r, 1].text(0.02, 0.98,
                      f"P={m.precision:.2f} R={m.recall:.2f}\nRMSE={m.rmse:.3f}px",
                      transform=ax[r, 1].transAxes, va="top", fontsize=8,
                      color="yellow",
                      bbox=dict(fc="black", alpha=0.55, ec="none", pad=2))
        if r == 0:
            ax[r, 1].legend(loc="lower right", fontsize=7, framealpha=0.6)

        im = ax[r, 2].imshow(nr, cmap="RdBu_r", vmin=-4, vmax=4)
        ax[r, 2].set_title("normalized residual" if r == 0 else "")
        ax[r, 2].text(0.02, 0.98,
                      f"med={np.median(nr):+.2f}\nrstd={0.5*(np.percentile(nr,84.1)-np.percentile(nr,15.9)):.2f}\n"
                      f"audit: {a['n_missed']} missed, {a['n_piled']} piled",
                      transform=ax[r, 2].transAxes, va="top", fontsize=8,
                      color="k", bbox=dict(fc="white", alpha=0.7, ec="none", pad=2))
        _mark_audit(ax[r, 2], a)
        plt.colorbar(im, ax=ax[r, 2], fraction=0.046)

    # ---- the real data on the same axes, for comparison ----
    res = boxsolve.detect_boxes(real, sigma=SIGMA, offset=OFFSET, verbose=0)
    d_e = (real - OFFSET) / res.gain
    nr = (d_e - res.model_image) / np.sqrt(np.maximum(res.model_image, 1e-6))
    a_real = audit.audit_result(d_e, res.model_image, SIGMA, z_thresh=5.0)
    r = len(DENSITIES)
    ax[r, 0].imshow(real, cmap="gray")
    ax[r, 0].set_ylabel(f"REAL BEADS\nN_est={len(res.positions)}", fontsize=9,
                        color="darkred")
    ax[r, 1].imshow(real, cmap="gray")
    ax[r, 1].scatter(res.positions[:, 1], res.positions[:, 0], s=28, marker="+",
                     c="red", linewidths=1.1)
    ax[r, 1].text(0.02, 0.98, "no ground truth", transform=ax[r, 1].transAxes,
                  va="top", fontsize=8, color="yellow",
                  bbox=dict(fc="black", alpha=0.55, ec="none", pad=2))
    im = ax[r, 2].imshow(nr, cmap="RdBu_r", vmin=-4, vmax=4)
    ax[r, 2].text(0.02, 0.98,
                  f"med={np.median(nr):+.2f}\nrstd={0.5*(np.percentile(nr,84.1)-np.percentile(nr,15.9)):.2f}\n"
                  f"audit: {a_real['n_missed']} missed, {a_real['n_piled']} piled",
                  transform=ax[r, 2].transAxes, va="top", fontsize=8, color="k",
                  bbox=dict(fc="white", alpha=0.7, ec="none", pad=2))
    _mark_audit(ax[r, 2], a_real)
    plt.colorbar(im, ax=ax[r, 2], fraction=0.046)

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Bead-matched simulation vs the real data  "
                 "(39x39, gain 4.23, bg 4 e-, peaks 94-198 e-)\n"
                 "residual panel: yellow = unexplained flux (missed), "
                 "cyan = over-modelled flux (piled-up PSFs)", fontsize=10)
    fig.tight_layout()
    fig.savefig("bead_like_simulation.png", dpi=150)

    print("\n%-12s %7s %7s %6s %6s %8s %9s %9s" %
          ("density", "N_true", "N_est", "P", "R", "RMSE", "resid med", "resid std"))
    for dens, m, rr, nrz, a in rows:
        print("%-12.3f %7d %7d %6.2f %6.2f %8.3f %9.2f %9.2f   audit: %d missed, %d piled" %
              (dens, m.n_true, m.n_est, m.precision, m.recall, m.rmse,
               np.median(nrz),
               0.5*(np.percentile(nrz,84.1)-np.percentile(nrz,15.9)),
               a["n_missed"], a["n_piled"]))
    print("%-12s %7s %7d %6s %6s %8s %9.2f %9.2f   audit: %d missed, %d piled" %
          ("REAL BEADS", "?", len(res.positions), "-", "-", "-",
           np.median(nr), 0.5*(np.percentile(nr,84.1)-np.percentile(nr,15.9)),
           a_real["n_missed"], a_real["n_piled"]))
    print("\nsaved bead_like_simulation.png")


if __name__ == "__main__":
    main()
