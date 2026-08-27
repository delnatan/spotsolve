"""Validation for msearch.py: can the move set recover a close pair?

All experiments use a KNOWN gain and offset, so the Poisson weighting is
exactly correct and any failure is attributable to the move mechanics
rather than to the noise model.

  1. Separation sweep  -- does SPLIT recover two emitters, and from what
                          separation onward?
  2. Split vs birth    -- is SPLIT actually necessary, or would a
                          residual-seeded BIRTH have found the pair?
  3. LRT calibration   -- is 2*dI distributed as chi2(3) under H0? (No.)
  4. False split rate  -- how often does a single true emitter get split?
"""

import numpy as np
import jax.numpy as jnp

import psf
import msearch
import moves

GAIN = 4.7        # ADU per photoelectron -- known by construction
OFFSET = 100.0    # ADU
SIGMA = 1.2
H = W = 15
BG = 20.0         # photoelectrons per pixel

_yy, _xx = jnp.mgrid[0:H, 0:W]
_yy, _xx = _yy * 1.0, _xx * 1.0


def make_pair(sep_sigma, flux, rng, angle=None):
    """Two emitters separated by sep_sigma*SIGMA, rendered to raw ADU."""
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    if angle is None:
        angle = rng.uniform(0, np.pi)
    s = sep_sigma * SIGMA
    dy, dx = 0.5 * s * np.cos(angle), 0.5 * s * np.sin(angle)
    pos = np.array([[cy + dy, cx + dx], [cy - dy, cx - dx]])
    if sep_sigma == 0.0:
        pos = pos[:1]
        amps = np.array([2.0 * flux])
    else:
        amps = np.array([flux, flux])
    theta = psf.pack(BG, amps, pos[:, 0], pos[:, 1])
    clean = np.asarray(psf.model(theta, _yy, _xx, SIGMA))
    electrons = rng.poisson(clean).astype(float)
    return electrons * GAIN + OFFSET, pos, amps


def to_electrons(adu):
    return (adu - OFFSET) / GAIN


def run_search(adu, lam, A_s, enable=("split", "birth", "death", "merge"), thr=0.0):
    sub = to_electrons(adu)
    # deliberately naive start: ONE emitter at the brightest pixel.
    py, px = np.unravel_index(int(np.argmax(sub)), sub.shape)
    b0 = float(np.percentile(sub, 10))
    A0 = max(float(sub[py, px] - b0), 1.0) / psf.peak_factor(SIGMA)
    return msearch.search_patch(
        sub, SIGMA, np.array([[float(py), float(px)]]), np.array([A0]),
        lam=lam, A_s=A_s, log_bf_threshold=thr, enable=enable,
    )


def match_rmse(true_pos, est_pos):
    from scipy.optimize import linear_sum_assignment
    if len(est_pos) == 0 or len(true_pos) == 0:
        return np.nan
    d = np.linalg.norm(true_pos[:, None, :] - est_pos[None, :, :], axis=-1)
    r, c = linear_sum_assignment(d)
    return float(np.sqrt(np.mean(d[r, c] ** 2)))


def crlb_separation(flux, sep_sigma):
    """CRLB position sigma for one emitter of a pair, for context."""
    s = sep_sigma * SIGMA
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    pos = np.array([[cy + s / 2, cx], [cy - s / 2, cx]])
    th = psf.pack(BG, [flux, flux], pos[:, 0], pos[:, 1])
    m = np.asarray(psf.model(th, _yy, _xx, SIGMA)).reshape(-1)
    J = np.asarray(psf.jac(jnp.array(th), _yy, _xx, SIGMA)).reshape(-1, 7)
    F = J.T @ ((1.0 / m)[:, None] * J)
    try:
        return float(np.sqrt(np.diag(np.linalg.inv(F))[2]))
    except np.linalg.LinAlgError:
        return np.inf


def exp1_separation_sweep(flux=1500.0, n_trials=40, seed=0):
    print("=" * 92)
    print("EXPERIMENT 1 -- separation sweep. Truth: TWO emitters, %g photons each." % flux)
    print("Search starts from ONE emitter at the brightest pixel; gain/offset known.")
    print("=" * 92)
    lam, A_s = 0.02, flux
    print("%9s %10s %10s %12s %12s %10s" %
          ("sep/sigma", "sep (px)", "K=2 found", "median K", "RMSE (px)", "CRLB"))
    out = []
    for sep in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        rng = np.random.default_rng(seed + int(sep * 100))
        ks, rmses = [], []
        for t in range(n_trials):
            adu, pos, _ = make_pair(sep, flux, rng)
            r = run_search(adu, lam, A_s)
            ks.append(len(r.amplitudes))
            if len(r.amplitudes) == 2:
                rmses.append(match_rmse(pos, r.positions))
        ks = np.array(ks)
        frac2 = float(np.mean(ks == 2))
        rm = float(np.median(rmses)) if rmses else np.nan
        crlb = crlb_separation(flux, sep)
        out.append((sep, frac2, rm))
        print("%9.2f %10.2f %9.0f%% %12.1f %12s %10.3f" %
              (sep, sep * SIGMA, 100 * frac2, np.median(ks),
               "  n/a" if not np.isfinite(rm) else "%.3f" % rm, crlb))
    return out


def exp2_split_vs_birth(flux=1500.0, n_trials=40, seed=1):
    print()
    print("=" * 92)
    print("EXPERIMENT 2 -- is SPLIT necessary? Same fields, move set ablated.")
    print("=" * 92)
    lam, A_s = 0.02, flux
    print("%9s %26s %26s" % ("sep/sigma", "birth+death+merge", "split+birth+death+merge"))
    for sep in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
        row = []
        for enable in (("birth", "death", "merge"),
                       ("split", "birth", "death", "merge")):
            rng = np.random.default_rng(seed + int(sep * 100))
            hits = 0
            for t in range(n_trials):
                adu, pos, _ = make_pair(sep, flux, rng)
                r = run_search(adu, lam, A_s, enable=enable)
                hits += (len(r.amplitudes) == 2)
            row.append(100.0 * hits / n_trials)
        print("%9.2f %25.0f%% %25.0f%%" % (sep, row[0], row[1]))


def exp3_lrt_calibration(flux=1500.0, n_trials=120, n_restart=25, seed=7):
    """Why this project uses a Bayes factor and not an F-test / LRT.

    The likelihood-ratio statistic 2*(I_1 - I_2) is often assumed to be
    chi2(3) -- three extra parameters. It is not, for two reasons that are
    structural to emitter detection rather than incidental:

      * under H0 the extra emitter lies on the BOUNDARY of the parameter
        space (A -> 0), and
      * exactly there, its position (y, x) is UNIDENTIFIABLE -- the
        likelihood is flat in two of the three added directions.

    Both violate the regularity conditions for Wilks' theorem. The
    practical consequence measured below is worse than a fixed offset: the
    null distribution DEPENDS ON HOW HARD THE OPTIMIZER SEARCHES, because
    the statistic is a maximum over the 2-emitter space and a better search
    finds a larger one. A test whose false-positive rate is a function of
    your restart count is not a calibrated test.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    b_max_f = lambda s_: max(float(s_.max()) * 4.0, 10.0)
    A_max_f = lambda s_: 8.0 * max(float(s_.max()), 1.0) / psf.peak_factor(SIGMA)

    def best_2emitter(sub, n_rs):
        best = None
        py, px = np.unravel_index(int(np.argmax(sub)), sub.shape)
        for _ in range(n_rs):
            p_ = np.array([py, px]) + rng.normal(0, 1.5, size=(2, 2))
            A_ = np.abs(rng.normal(1.0, 0.4, size=2)) * A_max_f(sub) / 8.0
            t0 = np.asarray(psf.pack(float(np.percentile(sub, 10)), A_, p_[:, 0], p_[:, 1]))
            rr = msearch.fit_theta(t0, _yy, _xx, SIGMA, sub, 0.0, b_max_f(sub), A_max_f(sub))
            if best is None or rr.I < best:
                best = rr.I
        return best

    print()
    print("=" * 92)
    print("EXPERIMENT 3 -- is the likelihood-ratio statistic chi2(3) under H0?")
    print("Truth: ONE emitter. Statistic: 2*(I_1 - I_2), maximized over 2-emitter space.")
    print("=" * 92)
    print("%-12s %8s %8s %8s %8s %10s" % ("restarts", "p50", "p90", "p95", "p99", "FP@5%"))
    crit = stats.chi2.ppf(0.95, 3)
    for n_rs in (4, n_restart):
        rng = np.random.default_rng(seed)
        stat = []
        for t in range(n_trials):
            adu, _, _ = make_pair(0.0, flux, rng)
            sub = to_electrons(adu)
            py, px = np.unravel_index(int(np.argmax(sub)), sub.shape)
            A0 = max(float(sub[py, px] - np.percentile(sub, 10)), 1.0) / psf.peak_factor(SIGMA)
            t1 = np.asarray(psf.pack(float(np.percentile(sub, 10)), [A0], [float(py)], [float(px)]))
            r1 = msearch.fit_theta(t1, _yy, _xx, SIGMA, sub, 0.0, b_max_f(sub), A_max_f(sub))
            stat.append(2.0 * (r1.I - best_2emitter(sub, n_rs)))
        stat = np.array(stat)
        print("%-12d %8.2f %8.2f %8.2f %8.2f %9.0f%%" % (
            n_rs, *np.percentile(stat, [50, 90, 95, 99]), 100 * np.mean(stat > crit)))
    print("%-12s %8.2f %8.2f %8.2f %8.2f %9.0f%%" % (
        "chi2(3)", *[stats.chi2.ppf(v / 100, 3) for v in (50, 90, 95, 99)], 5))
    print()
    print("The nominal 5%% test rejects at %.2f. With a weak search the observed rate" % crit)
    print("comes in UNDER nominal (the statistic is under-maximized); with a thorough")
    print("search it comes in OVER. Neither is 5%, and the gap between them is pure")
    print("optimizer effort. The Bayes factor has no such dependence: it integrates")
    print("the added dimensions against a proper prior instead of maximizing over")
    print("them, so the Occam factor pays for the search volume automatically.")


def exp4_false_split(flux=1500.0, n_trials=150, seed=3):
    print()
    print("=" * 92)
    print("EXPERIMENT 4 -- false split rate. Truth: ONE emitter. Bayes factor decides.")
    print("=" * 92)
    lam, A_s = 0.02, flux
    print("%18s %14s %14s" % ("log BF threshold", "K=1 (correct)", "K>1 (false)"))
    for thr in (0.0, 3.0, 5.0):
        rng = np.random.default_rng(seed)
        ks = []
        for t in range(n_trials):
            adu, pos, _ = make_pair(0.0, flux, rng)
            r = run_search(adu, lam, A_s, thr=thr)
            ks.append(len(r.amplitudes))
        ks = np.array(ks)
        print("%18.1f %13.0f%% %13.0f%%" %
              (thr, 100 * np.mean(ks == 1), 100 * np.mean(ks > 1)))


if __name__ == "__main__":
    exp1_separation_sweep()
    exp2_split_vs_birth()
    exp3_lrt_calibration()
    exp4_false_split()
