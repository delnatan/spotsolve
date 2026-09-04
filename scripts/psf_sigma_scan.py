"""What Gaussian sigma does the simulated confocal PSF present, at each depth?

`data/sim_out` was rendered by `psfkit`'s vectorial spinning-disk confocal
PSF, not by a Gaussian. Two numbers are needed before that data can be used
to judge `spotsolve`, and neither may be guessed:

    sigma(0)    the fixed sigma the model should be run at;
    sigma(z)    how wide each defocused emitter actually is, which is what
                decides whether an extra detection near it is a TILE of a
                real object or an invented one.

Both come from the same place: rebuild the simulator's PSF bank, bin it to
camera pixels exactly as it does, and fit the pipeline's own
pixel-integrated Gaussian to the centred stamp at each z.

    python scripts/psf_sigma_scan.py            # prints the table
    python scripts/psf_sigma_scan.py --emit     # prints it as Python source

Requires `psfkit`. The result is cached in `bench_sim.SIGMA_Z` so the
benchmark does not need it installed.
"""

import argparse

import numpy as np
from scipy.optimize import least_squares

from spotsolve import psf as spsf

# The simulator's settings; keep in sync with the arm's metadata.json.
PIXEL = 0.085
OVERSAMPLE = 11
STAMP = 21
DZ = 0.05
Z_RANGE = 0.5

WIDEFIELD_PINHOLE_AU = 20.0
# Widefield expressed as a confocal with the pinhole wide open, matching
# `psfkit`'s `examples/simulate_spinning_disk_spots.py --instrument widefield`.
# The two optics need SEPARATE width tables and it is not a small correction:
# a pinhole makes a defocused emitter broader AND dimmer, widefield only
# broader. Using the confocal table on widefield data understates every
# defocused emitter's width, which makes `bench_sim`'s tile/ghost split call
# genuine tiles "invented".


def build_stamps(optics="confocal", z_range=Z_RANGE):
    """(stamps, z): centred pixel-integrated PSF stamps, normalized to the
    in-focus total, on the simulator's own z grid."""
    from psfkit import ConfocalOptics, compute_confocal_psf

    common = dict(wavelength_exc=0.488, wavelength_em=0.525, na=1.4,
                  ni=1.515, ns=1.334)
    opt = (ConfocalOptics.from_andor_bc43(magnification=100.0, **common)
           if optics == "confocal"
           else ConfocalOptics(pinhole_radius_au=WIDEFIELD_PINHOLE_AU,
                               **common))
    k, p = OVERSAMPLE, STAMP
    nz = 2 * int(np.ceil(z_range / DZ)) + 1
    fine = compute_confocal_psf(
        opt, shape=(nz, k * (p + 2), k * (p + 2)),
        spacing=(DZ, PIXEL / k, PIXEL / k), vectorial=True, normalize=None)
    # Sub-pixel shift zero is the simulator's `bank.lookup(iz, 0, 0)` window.
    win = fine[:, k:k + p * k, k:k + p * k]
    stamps = win.reshape(nz, p, k, p, k).mean(axis=(2, 4))
    stamps /= stamps[nz // 2].sum()
    return stamps, (np.arange(nz) - nz // 2) * DZ


def fit_sigma(img):
    """Free-sigma pixel-integrated Gaussian fit; returns (sigma, rel_resid)."""
    p = img.shape[0]
    yy, xx = np.mgrid[0:p, 0:p]
    ay, ax = spsf.axes(yy, xx)
    c = (p - 1) / 2.0

    def resid(q):
        b, A, cy, cx, s = q
        return (spsf.model_ax(np.array([b, A, cy, cx]), ay, ax, s)
                - img).ravel()

    r = least_squares(resid, [0.0, img.sum(), c, c, 1.0],
                      bounds=([-np.inf, 0, c - 2, c - 2, 0.2],
                              [np.inf, np.inf, c + 2, c + 2, 8.0]))
    return float(r.x[4]), float(np.sqrt(np.mean(r.fun ** 2)) / img.max())


def main(args):
    stamps, z = build_stamps(args.optics, args.z_range)
    sig, frac, res = [], [], []
    for i in range(len(z)):
        s, rr = fit_sigma(stamps[i])
        sig.append(s)
        frac.append(float(stamps[i].sum()))
        res.append(rr)

    if args.emit:
        print(f"# {args.optics}, z +/- {args.z_range:g} um")
        half = len(z) // 2
        print("SIGMA_Z = (")
        print("    np.array([" + ", ".join(f"{v:.2f}" for v in z[half:])
              + "]),")
        print("    np.array([" + ", ".join(f"{v:.3f}" for v in sig[half:])
              + "]),")
        print(")")
        print("FLUX_Z = np.array(["
              + ", ".join(f"{v:.4f}" for v in frac[half:]) + "])")
        return

    print(f"\n{'z (um)':>7} {'sigma (px)':>11} {'sigma/sigma0':>13} "
          f"{'flux frac':>10} {'fit resid':>10}")
    print("-" * 55)
    for i in range(len(z)):
        print(f"{z[i]:+7.2f} {sig[i]:11.3f} {sig[i] / sig[len(z) // 2]:13.2f} "
              f"{frac[i]:10.4f} {100 * res[i]:9.2f}%")
    print(f"\nin-focus sigma = {sig[len(z) // 2]:.3f} px "
          f"({sig[len(z) // 2] * PIXEL * 1000:.1f} nm at "
          f"{PIXEL * 1000:.0f} nm pixels)")
    print("The Gaussian is a 0.3% fit at focus and a 5% fit at |z| = 0.5 um, "
          "so\n'sigma(z)' is a real width down to about |z| = 0.4 and a "
          "scale beyond it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--optics", choices=("confocal", "widefield"),
                    default="confocal",
                    help="which instrument to scan; they need separate tables")
    ap.add_argument("--z-range", type=float, default=Z_RANGE, dest="z_range",
                    help="um; scan +/- this depth")
    ap.add_argument("--emit", action="store_true",
                    help="print the table as Python source for bench_sim.py")
    main(ap.parse_args())
