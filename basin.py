"""Is a bad close-pair fit a WRONG BASIN or the STATISTICAL FLOOR?

`oracle.py` shows that at fixed, correct N, starting the joint fit 0.4 px from
truth wrecks close pairs (sd z 7.2, 35% beyond 3 sigma) while starting it AT
truth does not (sd z 1.33, 3.2%). Two incompatible readings:

  WRONG BASIN     the perturbed start converges to a worse point than the
                  truth-start does. The likelihood is multimodal and the local
                  optimizer picked the wrong mode. `I_perturbed > I_truth`.
                  Fixable by search: restarts, better seeds, a convexifying
                  homotopy.

  STATISTICAL     the perturbed start converges to a BETTER point than truth.
  FLOOR           The data's maximum-likelihood configuration genuinely is not
                  at the true positions -- Poisson noise moved it there. No
                  optimizer recovers that; it is the information limit, and the
                  only honest response is to report it in the SE.
                  `I_perturbed <= I_truth`.

The sign of `I_perturbed - I_truth` decides it, per group, and it is a fair
test because both arms fit the same pixels with the same N and the same model.

The perturbation is also swept, because "how far can a seed be before the basin
is lost" is exactly the tolerance any seeding scheme has to meet.

    python basin.py
"""

import argparse

import numpy as np

import lmga
import msearch
import patches as patch_mod
import psf
import simulate

SIGMA, GAIN, OFFSET, BG_E = 1.2, 4.23, 100.0, 4.0
AMP = (900.0, 1900.0)


def field(size, density, seed):
    n = max(1, int(round(density * size * size)))
    sim = simulate.simulate(shape=(size, size), n_emitters=n, background=BG_E,
                            amplitude_range=AMP, sigma=SIGMA, border=1.0,
                            seed=seed)
    return sim, sim.image


def fit_group(d_e, p, positions, amplitudes, background, max_iter=400):
    """Fit one patch exactly the way boxsolve.refine does. Returns
    (positions, amplitudes, I, se, converged)."""
    yy, xx = patch_mod.patch_grids(p)
    halo = patch_mod.build_halo_image(positions, amplitudes, p.frozen_indices,
                                      SIGMA, yy, xx, p.y0, p.x0)
    sub = np.asarray(d_e[p.y0:p.y1, p.x0:p.x1])
    K = len(p.indices)
    loc = positions[p.indices] - np.array([p.y0, p.x0])
    theta0 = psf.pack(background, amplitudes[p.indices], loc[:, 0], loc[:, 1])
    A_max = 8.0 * max(float(sub.max()), 1.0) / psf.peak_factor(SIGMA)
    lo, hi = msearch._bounds(K, sub.shape[0], sub.shape[1],
                             max(float(sub.max()) * 4.0, 10.0), A_max)
    r = lmga.fit(np.clip(theta0, lo + 1e-9, hi - 1e-9), yy, xx, SIGMA, sub,
                 lo, hi, halo=halo, max_iter=max_iter)
    _, A, cy, cx = psf.unpack(r.theta)
    out_p = np.stack([cy + p.y0, cx + p.x0], axis=1)
    se = np.full((K, 2), np.nan)
    try:
        var = np.diag(np.linalg.inv(r.F))
        v = np.where(var > 0, var, np.nan)
        se[:, 0] = np.sqrt(v[2::3])
        se[:, 1] = np.sqrt(v[3::3])
    except np.linalg.LinAlgError:
        pass
    return out_p, A, r.I, se, r.converged


def run(size, densities, seeds, sigmas_pert):
    rng = np.random.default_rng(0)
    rows = []
    for dens in densities:
        for s in range(seeds):
            sim, d_e = field(size, dens, 3000 + s)
            truth, t_amp = sim.positions, sim.amplitudes
            if len(truth) < 2:
                continue
            dmat = np.linalg.norm(truth[:, None, :] - truth[None, :, :], axis=-1)
            np.fill_diagonal(dmat, np.inf)
            nn = np.min(dmat, axis=1)

            # Patches are built ONCE from truth so that every perturbation arm
            # fits the identical pixels with the identical grouping; otherwise a
            # regrouping difference would masquerade as a basin difference.
            pset = patch_mod.build_patches(truth, SIGMA, d_e.shape,
                                           link_radius_factor=2.5, k_max=12)
            for p in pset:
                idx = p.indices
                if len(idx) == 0:
                    continue
                base_p, base_a, I0, se0, ok0 = fit_group(
                    d_e, p, truth.copy(), t_amp.copy(), BG_E)
                err0 = np.linalg.norm(base_p - truth[idx], axis=1)

                for ps in sigmas_pert:
                    pos0 = truth.copy()
                    pos0[idx] += rng.normal(0, ps, size=(len(idx), 2))
                    amp0 = t_amp.copy()
                    amp0[idx] *= rng.uniform(0.8, 1.2, size=len(idx))
                    gp, ga, I1, se1, ok1 = fit_group(d_e, p, pos0, amp0, BG_E)
                    err1 = np.linalg.norm(gp - truth[idx], axis=1)
                    for k, gi in enumerate(idx):
                        rows.append(dict(
                            dens=dens, pert=ps, nn=nn[gi], K=len(idx),
                            dI=I1 - I0, err0=err0[k], err1=err1[k],
                            se0=np.nanmean(se0[k]), se1=np.nanmean(se1[k]),
                            conv0=ok0, conv1=ok1))
    return rows


def main(args):
    rows = run(args.size, args.densities, args.seeds, args.pert)

    print(f"\nfield {args.size}x{args.size}, {args.seeds} seeds/density, "
          f"fixed N = N_true, patches fixed from truth\n")
    print("dI = I(perturbed start) - I(truth start), in nats, per GROUP.")
    print("dI > 0  => the perturbed start converged to a WORSE point "
          "(wrong basin, fixable).")
    print("dI < 0  => it found a BETTER point than truth "
          "(the ML answer is not at truth; statistical floor).\n")

    hdr = (f"{'pert':>6} {'group':>12} {'n':>6} {'wrong basin':>12} "
           f"{'>1 nat worse':>13} {'better than':>12} {'med err trut':>13} "
           f"{'med err pert':>13} {'med SE':>8}")
    print(hdr)
    print("-" * len(hdr))
    edges = [(0.0, 2 * SIGMA, f"nn<{2 * SIGMA:.1f}"),
             (2 * SIGMA, 4 * SIGMA, f"nn {2 * SIGMA:.1f}-{4 * SIGMA:.1f}"),
             (4 * SIGMA, np.inf, f"nn>{4 * SIGMA:.1f}")]
    for ps in args.pert:
        for lo, hi, nm in edges:
            g = [r for r in rows if r["pert"] == ps and lo <= r["nn"] < hi]
            if not g:
                continue
            dI = np.array([r["dI"] for r in g])
            print(f"{ps:6.2f} {nm:>12} {len(g):6d} "
                  f"{100 * np.mean(dI > 1e-6):11.1f}% "
                  f"{100 * np.mean(dI > 1.0):12.1f}% "
                  f"{100 * np.mean(dI < -1e-6):11.1f}% "
                  f"{np.median([r['err0'] for r in g]):13.4f} "
                  f"{np.median([r['err1'] for r in g]):13.4f} "
                  f"{np.nanmedian([r['se0'] for r in g]):8.4f}")
        print()

    # The floor itself: how well can ANY estimator do, given that the ML point
    # is where it is? err0 is the truth-started fit, which is the best case.
    print("=== the truth-started fit alone (the estimator's floor) ===")
    print(f"{'group':>12} {'n':>6} {'med err':>9} {'p90 err':>9} {'RMSE':>9} "
          f"{'med SE':>9} {'med/SE':>8} {'conv':>7}")
    for lo, hi, nm in edges:
        g = [r for r in rows if r["pert"] == args.pert[0] and lo <= r["nn"] < hi]
        if not g:
            continue
        e = np.array([r["err0"] for r in g])
        se = np.nanmedian([r["se0"] for r in g])
        print(f"{nm:>12} {len(g):6d} {np.median(e):9.4f} "
              f"{np.percentile(e, 90):9.4f} {np.sqrt(np.mean(e ** 2)):9.4f} "
              f"{se:9.4f} {np.median(e) / se:8.2f} "
              f"{100 * np.mean([r['conv0'] for r in g]):6.0f}%")
    print("\n(med/SE = 1.177 for an efficient 2D estimator)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=39)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--densities", type=float, nargs="*", default=[0.034, 0.055])
    ap.add_argument("--pert", type=float, nargs="*",
                    default=[0.1, 0.2, 0.4, 0.8])
    main(ap.parse_args())
