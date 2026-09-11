"""Referee for the box-truncated Laplace score of the native group search.

    python scripts/measure_box_laplace.py
    python scripts/measure_box_laplace.py --trials 20 --samples 400000

Run after `maturin develop --release -m rust/spotsolve-py/Cargo.toml`.

`dense_group::fit_and_score` scores a configuration by integrating a quadratic
model of its log posterior. The regular Laplace expression integrates that
model over all of R^p; the group score restricts it to the prior's support --
widths in their class interval, positions in the transaction's box, positive
amplitudes -- one coordinate at a time (`dense_group::log_box_mass`).

This script estimates the integral BOTH are approximations to, with neither
approximation in it:

    log Z = log int_support exp(-I(theta) + log p(theta)) dtheta

-- the exact Poisson I-divergence against the same observations and halo, the
exact configuration prior (count, class, flux and width terms), over exactly
the support the score uses -- by importance sampling from a multivariate
Student-t centred on the mode. It reports each Laplace expression's error
against that, per configuration and per removal GAIN, which is the number a
decision actually reads.

Each configuration is scored in the context built around it, so an incumbent
and its removals can have slightly different position boxes. That does not
touch what is measured -- each Laplace estimate is compared against the
referee on its OWN support -- but it means a `gain` row reports the error of a
difference of two approximations, not the engine's gain itself.

What the referee measures is the mass of the mode the fit found. It does not
sum over other modes: an over-fitted group has many, and the score does not
claim to integrate them either. `ess` is the effective sample size; a row
with `ess < 200` is reported but is not evidence.
"""

import argparse
import sys
from math import lgamma
from pathlib import Path

import numpy as np
from scipy.special import erf, gammaln, logsumexp

sys.path.insert(0, str(Path(__file__).parent))
import check_group_search as C  # noqa: E402  -- the controls' frames and prior

from spotsolve import backend  # noqa: E402

SQRT2 = np.sqrt(2.0)
T_DF = 5.0
T_SCALE = 1.25
# Student-t with 5 degrees of freedom, scaled 25% wider than the curvature
# says: heavier-tailed than the posterior in every direction the quadratic
# model describes, which is the condition an importance sampler needs. A
# coordinate whose marginal SD exceeds half its support is shrunk to that, so
# a phantom's unidentified position still lands inside the box.


# ---------------------------------------------------------------------------
# The exact integrand
# ---------------------------------------------------------------------------

def log_prior(S, wp, a_s):
    """`log p(configuration)` at each row of `S`, as `dense_group` defines it.

    Count, class and labeling terms, the exponential flux density and the
    Cauchy width density -- the same expression `PriorSnapshot::log_config`
    evaluates, written again here so the two can disagree.
    """
    A, sig = S[:, 1::4], S[:, 4::4]
    k = A.shape[1]
    if k == 0:
        return np.zeros(len(S))
    flux = np.where(A > 0, -np.log(a_s) - A / a_s, -np.inf).sum(1)
    u = (sig - wp.sigma0) / wp.scale
    dens = (-np.log1p(u * u) - wp._logZ).sum(1)
    k_f = (sig <= wp.mid).sum(1)
    k_w = k - k_f
    lf = np.array([lgamma(n + 1) for n in range(k + 1)])
    count = (k_f * np.log(wp.lam_focus) - lf[k_f]
             + k_w * np.log(wp.lam_wide) - lf[k_w])
    return flux + dens + count


def neg_idiv(S, r):
    """`-I(theta)`: the data-only Poisson I-divergence, as `lmcl::idiv` has it."""
    obs, halo = r["obs"], r["halo"]
    h, w = obs.shape
    ay, ax = np.arange(h, dtype=float), np.arange(w, dtype=float)
    b = S[:, 0]
    A, y, x, sig = S[:, 1::4], S[:, 2::4], S[:, 3::4], S[:, 4::4]
    k = 1.0 / (sig[:, :, None] * SQRT2)
    Ey = 0.5 * (erf((ay - y[:, :, None] + 0.5) * k) - erf((ay - y[:, :, None] - 0.5) * k))
    Ex = 0.5 * (erf((ax - x[:, :, None] + 0.5) * k) - erf((ax - x[:, :, None] - 0.5) * k))
    m = b[:, None, None] + halo[None] + np.einsum("nk,nkh,nkw->nhw", A, Ey, Ex)
    m = np.maximum(m, 1e-9)
    pos = obs > 0
    d_safe = np.where(pos, obs, 1.0)
    term = np.where(pos, d_safe * np.log(d_safe / m), 0.0)
    return -(term - (obs - m)).sum((1, 2))


def log_post(S, r, wp, a_s):
    out = np.full(len(S), -np.inf)
    inside = np.all((S >= r["score_lo"]) & (S <= r["score_hi"]), axis=1)
    if inside.any():
        Si = S[inside]
        out[inside] = neg_idiv(Si, r) + log_prior(Si, wp, a_s)
    return out


# ---------------------------------------------------------------------------
# Importance sampling
# ---------------------------------------------------------------------------

WEAK_Z = 5.0
# An emitter whose amplitude is fewer than this many marginal standard errors
# from zero gets the defensive component below. Such an emitter's posterior is
# not a bump at the mode: at small A its position is unidentified and spreads
# over the whole box, which a mode-centred proposal cannot reach.


class StudentT:
    """Multivariate Student-t on a subset of coordinates, shrunk to the box."""

    def __init__(self, centre, cov, width):
        sd = np.sqrt(np.diag(cov))
        shrink = np.minimum(1.0, 0.5 * width / sd)
        cov = cov * shrink[:, None] * shrink[None, :]
        self.c, self.L = centre, np.linalg.cholesky(cov)
        self.cinv = np.linalg.inv(cov)
        p = len(centre)
        self.p = p
        self.norm = (gammaln((T_DF + p) / 2) - gammaln(T_DF / 2)
                     - 0.5 * p * np.log(T_DF * np.pi)
                     - np.log(np.diag(self.L)).sum())

    def sample(self, m, rng):
        z = rng.standard_normal((m, self.p)) @ self.L.T
        return self.c + z / np.sqrt(rng.chisquare(T_DF, m) / T_DF)[:, None]

    def logpdf(self, X):
        dev = X - self.c
        maha = np.einsum("ni,ij,nj->n", dev, self.cinv, dev)
        return self.norm - 0.5 * (T_DF + self.p) * np.log1p(maha / T_DF)


def referee(r, wp, a_s, n, rng, chunk=4000):
    """`(log Z, standard error of log Z, effective sample size)`.

    A two-component defensive mixture when the configuration has a weak
    emitter: half the samples from a Student-t on every coordinate, half from
    a Student-t on everything EXCEPT the weakest emitter, whose position and
    width are drawn uniformly over their support and whose amplitude from an
    exponential on the scale of its own standard error -- the shape its
    posterior takes when it is invisible. Every sample is weighted by the
    MIXTURE density, so the estimate is unbiased whichever component drew it.
    """
    theta = np.asarray(r["theta"])
    p = len(theta)
    lo, hi = np.asarray(r["score_lo"]), np.asarray(r["score_hi"])
    width = hi - lo
    cov = np.linalg.inv(np.asarray(r["curvature"])) * T_SCALE ** 2
    full = StudentT(theta, cov, width)

    k = (p - 1) // 4
    z = [theta[1 + 4 * e] / np.sqrt(cov[1 + 4 * e, 1 + 4 * e]) * T_SCALE
         for e in range(k)]
    weak = int(np.argmin(z)) if k and min(z) < WEAK_Z else None
    if weak is not None:
        blk = np.arange(1 + 4 * weak, 5 + 4 * weak)
        rest = np.setdiff1d(np.arange(p), blk)
        part = StudentT(theta[rest], cov[np.ix_(rest, rest)], width[rest])
        a_prop = max(theta[blk[0]], np.sqrt(cov[blk[0], blk[0]]))

        def logq_part(S):
            A, pos = S[:, blk[0]], S[:, blk[1:]]
            inside = np.all((pos >= lo[blk[1:]]) & (pos <= hi[blk[1:]]), axis=1) & (A > 0)
            lq = (part.logpdf(S[:, rest]) - np.log(width[blk[1:]]).sum()
                  - np.log(a_prop) - A / a_prop)
            return np.where(inside, lq, -np.inf)

        def sample_part(m):
            S = np.empty((m, p))
            S[:, rest] = part.sample(m, rng)
            S[:, blk[0]] = rng.exponential(a_prop, m)
            S[:, blk[1:]] = lo[blk[1:]] + rng.uniform(size=(m, 3)) * width[blk[1:]]
            return S

    logw = []
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        if weak is None:
            S = full.sample(m, rng)
            logq = full.logpdf(S)
        else:
            S = np.vstack([full.sample(m // 2, rng), sample_part(m - m // 2)])
            logq = np.logaddexp(np.log(0.5) + full.logpdf(S), np.log(0.5) + logq_part(S))
        logw.append(log_post(S, r, wp, a_s) - logq)
    logw = np.concatenate(logw)
    lz = logsumexp(logw) - np.log(n)
    wn = np.exp(logw - logw.max())
    ess = wn.sum() ** 2 / (wn ** 2).sum()
    # Delta method: var(log mean w) = var(w) / (n * mean(w)^2).
    se = np.sqrt(np.var(wn) / (n * np.mean(wn) ** 2))
    return lz, se, ess


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

def score(eng, ents, focus):
    pos = np.ascontiguousarray(
        np.array([[e[0], e[1]] for e in ents], float).reshape(-1, 2))
    amp = np.ascontiguousarray(np.array([e[2] for e in ents], float))
    sig = np.ascontiguousarray(np.array([e[3] for e in ents], float))
    ids = np.arange(len(ents), dtype=np.uint32)
    # The engine's own budget policy: the default fit, then a continuation at
    # the escalated budget if that did not certify a mode. The referee should
    # be asked about the modes the search actually scores.
    r = eng.score_state(pos, amp, sig, ids, tuple(focus))
    if r["status"] == "nonstationary":
        again = score_from(eng, r, focus)
        if again["status"] == "supported" or again["objective"] < r["objective"]:
            r = again
    return r


def score_from(eng, r, focus):
    ems = emitters_of(r)
    pos = np.ascontiguousarray(np.array([[e[0], e[1]] for e in ems], float).reshape(-1, 2))
    amp = np.ascontiguousarray(np.array([e[2] for e in ems], float))
    sig = np.ascontiguousarray(np.array([e[3] for e in ems], float))
    return eng.score_state(pos, amp, sig, np.arange(len(ems), dtype=np.uint32),
                           tuple(focus), max_iter=300, tol_obj=1e-10)


def laplace_terms(r):
    """`(regular, boxed)`: the two Laplace estimates of `log Z`."""
    base = -r["i_div"] + r["log_prior"]
    return base + r["log_volume"], base + r["log_volume"] + r["log_box"]


def emitters_of(r):
    """The fitted configuration back as global `(y, x, flux, sigma)` rows."""
    y0, x0, _, _ = r["region"]
    th = np.asarray(r["theta"])
    return [(th[2 + 4 * e] + y0, th[3 + 4 * e] + x0, th[1 + 4 * e], th[4 + 4 * e])
            for e in range((len(th) - 1) // 4)]


def cases(trials, rng):
    """`(name, image, bmap, [configurations], focus)`.

    Each case yields the configurations whose log Z is refereed. For the
    over-fitted group that is the fitted K=3 incumbent AND each single removal
    from it, because the decision reads the difference.
    """
    for t in range(trials):
        f = rng.uniform(900, 1900)
        tr = [(12.0 + rng.uniform(-.5, .5), 12.0 + rng.uniform(-.5, .5), f, C.SIGMA)]
        img, bm, *_ = C.frame((26, 26), tr, 5.0, 2000 + t)
        yield "isolated", img, bm, [[(tr[0][0], tr[0][1], f, C.SIGMA)]], tr[0][:2]

        tr = [(14.0, 14.0, 1400.0, 0.60 * C.SIGMA)]
        img, bm, *_ = C.frame((30, 30), tr, 5.0, 6002 + 3 * t)
        yield ("below_width_bound", img, bm,
               [[(14.0, 14.0, 1400.0, C.SLACK[0] * C.SIGMA * 1.01)]], (14.0, 14.0))

        tr = [(12.0, 12.0, 1500.0, C.SIGMA)]
        img, bm, *_ = C.frame((26, 26), tr, 5.0, 13000 + t)
        yield ("faint_neighbour_on_background", img, bm,
               [[(12.0, 12.0, 1500.0, C.SIGMA)],
                [(12.0, 12.0, 1500.0, C.SIGMA), (15.5, 15.5, 12.0, C.SIGMA)]],
               (13.5, 13.5))

        f = rng.uniform(1200, 2000)
        img, bm, *_ = C.frame((30, 30), [(15.0, 15.0, f, C.SIGMA)], 5.0, 11000 + t)
        entry = [(14.3, 15.0, f / 3, C.SIGMA), (15.7, 15.0, f / 3, C.SIGMA),
                 (15.0, 16.4, f / 3, C.SIGMA)]
        yield "overfit", img, bm, [entry], (15.0, 15.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    wp = C.width_prior()
    a_s = 1400.0
    be = backend.get("rs")

    print("case                          K act  status          logZ(IS)    se     ess"
          "   regular-IS  boxed-IS")
    rows = []
    for name, img, bm, configs, focus in cases(args.trials, rng):
        eng = be.group_engine(img, bm, C.SIGMA, C.SLACK, 12, wp, a_s, next_id=16)
        fitted = [score(eng, c, focus) for c in configs]
        # For the multi-emitter cases, also referee each single removal from
        # the LARGEST fitted configuration: those are the gains the search
        # decides with.
        big = fitted[-1]
        ems = emitters_of(big)
        family = [("incumbent", big)]
        if len(ems) > 1:
            for j in range(len(ems)):
                family.append((f"minus {j}", score(eng, ems[:j] + ems[j + 1:], focus)))
        for c, r in zip(configs[:-1], fitted[:-1]):
            family.append(("given", r))
        refs = {}
        for tag, r in family:
            reg, box = laplace_terms(r)
            # Sanity: the referee's integrand must BE the score's integrand at
            # the mode, or nothing below means anything.
            th = np.asarray(r["theta"])[None]
            at_mode = log_post(th, r, wp, a_s)[0]
            assert abs(at_mode - (-r["i_div"] + r["log_prior"])) < 1e-6, (
                name, tag, at_mode, -r["i_div"] + r["log_prior"])
            lz, se, ess = referee(r, wp, a_s, args.samples, rng)
            refs[tag] = (lz, reg, box, r, ess)
            k = (len(th[0]) - 1) // 4
            print(f"{name[:22]:22s} {tag[:8]:8s} {k:d} {r['n_active']:3d}  {r['status'][:14]:14s}"
                  f" {lz:10.3f} {se:6.3f} {ess:7.0f}   {reg - lz:+8.3f}  {box - lz:+8.3f}",
                  flush=True)
            rows.append(dict(case=name, tag=tag, k=k, active=r["n_active"],
                             status=r["status"], lz=lz, se=se, ess=ess,
                             reg=reg - lz, box=box - lz))
        # Gains against the incumbent: what a removal decision reads.
        if "incumbent" in refs:
            lz0, reg0, box0, r0, ess0 = refs["incumbent"]
            for tag, (lz, reg, box, r, ess) in refs.items():
                if tag == "incumbent":
                    continue
                # A gain is evidence only when both of its ends are: supported
                # scores, and referee estimates with a usable sample size.
                if min(ess, ess0) < 200 or "supported" not in (r["status"], r0["status"]) \
                        or r["status"] != r0["status"]:
                    continue
                g_is = lz - lz0
                print(f"{'':22s}   gain {tag:10s}  IS {g_is:+8.3f}   regular err {reg - reg0 - g_is:+7.3f}"
                      f"   boxed err {box - box0 - g_is:+7.3f}", flush=True)
                rows.append(dict(case=name, tag="gain " + tag, gain_is=g_is,
                                 reg=reg - reg0 - g_is, box=box - box0 - g_is))

    print("\nsummary (ess >= 200 only; a gain needs both ends supported and ess >= 200)")
    for name in dict.fromkeys(r["case"] for r in rows):
        conf = [r for r in rows if r["case"] == name and "lz" in r and r["ess"] >= 200]
        gains = [r for r in rows if r["case"] == name and "gain_is" in r]
        for label, rs in (("config", conf), ("gain", gains)):
            if not rs:
                continue
            reg = np.array([r["reg"] for r in rs])
            box = np.array([r["box"] for r in rs])
            print(f"  {name:30s} {label:6s} n={len(rs):3d}  |regular err| median {np.median(np.abs(reg)):.3f}"
                  f" max {np.abs(reg).max():.3f}   |boxed err| median {np.median(np.abs(box)):.3f}"
                  f" max {np.abs(box).max():.3f}")


if __name__ == "__main__":
    main()
