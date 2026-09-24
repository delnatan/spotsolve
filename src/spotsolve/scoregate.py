"""PROTOTYPE (2026-09-23): score-gated box search. Not exported.

One statistic decides every change of count: the efficient score z for
adding one reference-width emitter at a pixel, given everything already
fitted in the window (level and emitters, with their parameters free).

1. Detect: the K = 0 score over the frame (a zero-mean matched filter).
   Each local maximum with z > u seeds its own window.
2. Add: in a seed's window, place at the owned pixel of highest z, only
   if z > u, starting from the one-step amplitude S / I_eff. Keep the
   emitter only if the refit gains u^2 / 2 dispersion-scaled nats.
3. Windows are decided once, brightest seed first; each sees the emitters
   of windows decided before it as fixed light (halo).

z > u is the score test, and u^2 / 2 nats is its likelihood-ratio
equivalent. The knob is `fp_per_mpx`, the expected false emitters per 10^6
pixels of pure noise; u is solved from it for the given sigma (`threshold`).
Widths are estimated within
`slack * sigma`; the score always uses the reference width.

The background map is a BG_WIN median filter and phi a scalar
fourth-difference estimate (`estimate_background`). Detection only needs
them for the null variance and each window's background shape; the level
is fitted per window. Fits use the production LM fitter
(`lmcl_fit_var_sigma`).

Pruned 2026-09-23 after ablation (output/scoregate/ablation.json): a
backward removal pass (removed 3 of 4687 additions on GEM, 13 of 13892 on
beads, for 26-35% of all fits), a second sweep over windows (doubled fits;
synthetic results identical, real counts within 3%) and a final joint
polish (position RMSE already matched production). Production's masked
background and noise maps were replaced (output/scoregate/preprocess.json):
counts on GEM and beads moved under 2%, phi under 1%, and the median map
had the fewest false positives on synthetic haze. The efficient
projection is not optional: a level-only score resolved 1% of 2-sigma
pairs against 64%.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

import spotsolve_rs as _rs

from . import psf

FP_PER_MPX = 16.0
"""Default target: false emitters per 10^6 noise pixels. At sigma 1.45 this
is u = 4.0, which had the best isolated recall at matched false positives
(output/scoregate/u_sweep.json); it is ~0.3 per 128x128 GEM frame."""
RFT_C = 0.80 * 1e6 / (2.0 * (2.0 * np.pi) ** 1.5)
"""Per Mpx: false emitters = RFT_C * u * exp(-u^2 / 2) / sigma^2, the
Euler-characteristic density of a Gaussian-smoothed field (Lambda =
1 / (2 sigma^2)). 0.80 is measured: accepted noise emitters over 4 Mpx at
sigma 1.45, u 4.0-4.47, fell 0.80x below the continuous formula."""
SLACK = (0.70, 2.2)
OWN = 4.0
"""sigma. A window places emitters within this of its seed, on pixels no
nearer another seed. On a dense synthetic field (output/scoregate/
geometry_tied.json) recall rose to 0.729 at 4 and only 0.733 at 6."""
SUPPORT = 3.0
"""sigma. Context beyond the placement radius, so an emitter placed at its
edge keeps its 3-sigma support: window half-side = (OWN + SUPPORT) sigma.
With the window edge at OWN instead, precision was 0.907 against 0.955."""
REACH = 3.0
"""Emitter widths. Neighbouring light within this of a window enters its
halo; 5 widths gave identical results (output/scoregate/geometry.json)."""
K_MAX = 12
"""Safety cap per window; a cap of 4 changed GEM by 2%."""
FIT_MAX_ITER = 100
FIT_TOL = 1e-6


def _axis_kernel(sigma, r):
    t = np.arange(-r, r + 1, dtype=float)
    return psf._shape(t, np.zeros(1), sigma)[:, 0]


def psf_kernel(sigma):
    """Centred pixel-integrated unit-flux PSF, radius ceil(4 sigma)."""
    k = _axis_kernel(sigma, int(np.ceil(4 * sigma)))
    return np.outer(k, k)


def detection_map(r, var, sigma):
    """Frame-wide K = 0 score z, blind to a local constant level.

    `r` is data minus background, `var` the null pixel variance (ADU^2).
    The zero-mean kernel is the efficient score when the level is a free
    nuisance and the variance is locally flat.
    """
    g = psf_kernel(sigma)
    k = g - g.mean()
    num = ndi.correlate(r, k, mode="reflect")
    den = ndi.correlate(var, k * k, mode="reflect")
    return num / np.sqrt(np.maximum(den, 1e-12))


def find_seeds(z, sigma, u):
    win = 2 * int(np.ceil(sigma)) + 1
    peak = (z == ndi.maximum_filter(z, size=win, mode="reflect")) & (z > u)
    idx = np.argwhere(peak)
    order = np.argsort(-z[peak])
    return idx[order].astype(float), z[peak][order]


@dataclass
class Window:
    y0: int
    x0: int
    d: np.ndarray
    shape: np.ndarray      # background map minus its median: fixed
    level: float           # background map median: where the free level starts
    owned: np.ndarray
    phi: float
    halo: np.ndarray = None

    @property
    def hw(self):
        return self.d.shape


@dataclass
class Fit:
    i_div: float
    theta: np.ndarray

    @property
    def k(self):
        return (len(self.theta) - 1) // 4

    def ems(self):
        return self.theta[1:].reshape(-1, 4)


@dataclass
class Stats:
    fits: int = 0
    score_evals: int = 0
    adds: int = 0
    lr_fail: int = 0        # passed the score gate, failed the LR
    accepted: list = field(default_factory=list)   # (z, dI/phi) per add


def _bounds(win, k, sigma, slack):
    h, w = win.hw
    smax = max(float(win.d.max()), 1.0)
    a_max = 8.0 * smax / float(psf.peak_factor(sigma)) * slack[1] ** 2
    a_min = max(1e-4, 1e-6 * a_max)
    lo = [0.0] + [a_min, -0.5, -0.5, slack[0] * sigma] * k
    hi = [max(4.0 * smax, 10.0)] + [a_max, h - 0.5, w - 0.5, slack[1] * sigma] * k
    return np.array(lo), np.array(hi)


def fit(win, theta0, sigma, slack, stats):
    k = (len(theta0) - 1) // 4
    lo, hi = _bounds(win, k, sigma, slack)
    th = np.clip(np.asarray(theta0, float), lo + 1e-9, hi - 1e-9)
    h, w = win.hw
    theta, i_div, *_ = _rs.lmcl_fit_var_sigma(
        th, h, w, win.d, win.halo, lo, hi, FIT_MAX_ITER, tol_obj=FIT_TOL)
    stats.fits += 1
    return Fit(float(i_div), np.asarray(theta))


def efficient_score(win, state, g, stats):
    """(z, S / I_eff) at every pixel for one more reference-width emitter.

    S = sum g_p (d - m) / (phi m); I_eff is sum g_p^2 / (phi m) less its
    projection onto the current parameters' Fisher information.
    """
    stats.score_evals += 1
    h, w = win.hw
    yy, xx = np.mgrid[:h, :w].astype(float)
    m = psf.model_var_sigma(state.theta, yy, xx, win.halo)
    wt = 1.0 / (win.phi * np.maximum(m, 1e-3))
    J = psf.jac_var_sigma(state.theta, yy, xx)
    P = J.shape[-1]
    def corr(a, k):
        return ndi.correlate(a, k, mode="constant")

    S = corr((win.d - m) * wt, g)
    igg = corr(wt, g * g)
    C = np.stack([corr(J[..., q] * wt, g).ravel() for q in range(P)])
    Jf = J.reshape(-1, P)
    F = Jf.T @ (Jf * wt.reshape(-1, 1))
    X = np.linalg.lstsq(F, C, rcond=1e-12)[0]
    ieff = igg.ravel() - np.einsum("qn,qn->n", C, X)
    ieff = np.maximum(ieff, 1e-12).reshape(h, w)
    return S / np.sqrt(ieff), S / ieff


def search(win, sigma, u, slack, k_max, stats):
    """Score-gated additions from K = 0, each confirmed by the LR."""
    g = psf_kernel(sigma)
    gain = 0.5 * u * u * win.phi
    state = fit(win, [win.level], sigma, slack, stats)
    while state.k < k_max:
        z, a = efficient_score(win, state, g, stats)
        z = np.where(win.owned, z, -np.inf)
        p = np.unravel_index(np.argmax(z), z.shape)
        if not z[p] > u:
            break
        trial = fit(win, np.r_[state.theta, a[p], p[0], p[1], sigma], sigma, slack, stats)
        d_i = state.i_div - trial.i_div
        if not d_i > gain:
            stats.lr_fail += 1
            break
        stats.adds += 1
        stats.accepted.append((float(z[p]), d_i / win.phi))
        state = trial
    return state


CHI2_1_MEDIAN = 0.4549364231195728
BG_WIN = 25
"""px. Background and dispersion windows (production BG_KERNEL / NOISE_WIN)."""


def estimate_dispersion(d):
    """Scalar phi = pixel variance / mean from the separable fourth
    difference, whose squared output has median var * 70^2 * CHI2_1_MEDIAN
    for white noise. The median pixel is taken to be background."""
    k = np.array([1.0, -4.0, 6.0, -4.0, 1.0])
    f = ndi.correlate1d(ndi.correlate1d(d, k, axis=0, mode="reflect"), k, axis=1, mode="reflect")
    var = np.median(f[2:-2, 2:-2] ** 2) / (CHI2_1_MEDIAN * 70.0 ** 2)
    return float(var / max(np.median(d), 1e-6))


def median_background(d):
    """BG_WIN median of `d` rounded to whole ADU, reflected at the edges.

    Rounding (error <= 0.5 ADU, far below the noise) makes the median exact
    for a sliding-histogram implementation, which the Rust port uses.
    """
    return ndi.median_filter(np.rint(d), size=BG_WIN, mode="reflect")


def estimate_background(d):
    """(BG_WIN median-filtered background, scalar phi) of an offset-free frame.

    Masking this detector's own seeds out of a local mean did no better,
    and a flat background did worse on haze (output/scoregate/preprocess.json).
    """
    return median_background(d), estimate_dispersion(d)


def _render(ems, y0, x0, h, w):
    """Light of global emitters `(A, y, x, s)` on a window, no background."""
    if len(ems) == 0:
        return np.zeros((h, w))
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(float)
    th = psf.pack_var_sigma(0.0, ems[:, 0], ems[:, 1], ems[:, 2], ems[:, 3])
    return psf.model_var_sigma(th, yy, xx)


@dataclass
class Result:
    positions: np.ndarray
    amplitudes: np.ndarray
    fit_sigma: np.ndarray
    background: np.ndarray
    dispersion: float
    seeds: np.ndarray
    seed_z: np.ndarray
    z0: np.ndarray
    stats: Stats
    u: float


def threshold(sigma, fp_per_mpx=FP_PER_MPX):
    """Score threshold u giving `fp_per_mpx` expected noise false emitters.

    Solves RFT_C * u * exp(-u^2 / 2) / sigma^2 = fp_per_mpx on u >= 1,
    where the left side decreases monotonically.
    """
    if not fp_per_mpx > 0:
        raise ValueError("fp_per_mpx must be positive")
    rate = lambda u: RFT_C * u * np.exp(-0.5 * u * u) / sigma ** 2  # noqa: E731
    lo, hi = 1.0, 40.0
    if rate(lo) <= fp_per_mpx:
        return lo
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rate(mid) > fp_per_mpx else (lo, mid)
    return 0.5 * (lo + hi)


def localize(frame, sigma, *, offset=0.0, fp_per_mpx=FP_PER_MPX, u=None, slack=SLACK, k_max=K_MAX,
             background=None, dispersion=None):
    """Score-gated localization of one frame. Returns `Result`.

    `u` overrides the threshold solved from `fp_per_mpx`. `background`
    (ADU above offset, frame-shaped) and `dispersion` override the
    estimates of `estimate_background`.
    """
    u = threshold(sigma, fp_per_mpx) if u is None else float(u)
    d = np.ascontiguousarray(frame, dtype=float) - offset
    H, W = d.shape
    if background is None:
        background = median_background(d)
    phi = estimate_dispersion(d) if dispersion is None else float(dispersion)
    var = phi * np.maximum(background, 1e-3)
    z0 = detection_map(d - background, var, sigma)
    seeds, seed_z = find_seeds(z0, sigma, u)
    pad = int(np.ceil((OWN + SUPPORT) * sigma))
    wins = []
    for i, (sy, sx) in enumerate(seeds.astype(int)):
        y0, x0 = max(sy - pad, 0), max(sx - pad, 0)
        y1, x1 = min(sy + pad + 1, H), min(sx + pad + 1, W)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d_own = np.hypot(yy - sy, xx - sx)
        others = np.delete(seeds, i, axis=0)
        near = others[(np.abs(others[:, 0] - sy) <= 2 * pad) & (np.abs(others[:, 1] - sx) <= 2 * pad)]
        d_oth = (np.min(np.hypot(yy[..., None] - near[:, 0], xx[..., None] - near[:, 1]), -1)
                 if len(near) else np.full(yy.shape, np.inf))
        bg = background[y0:y1, x0:x1]
        level = float(np.median(bg))
        wins.append(Window(y0, x0, np.ascontiguousarray(d[y0:y1, x0:x1]), bg - level, level,
                           (d_own <= OWN * sigma) & (d_own <= d_oth), phi))
    # Seeds come strongest first, so bright light is fitted before the dim
    # windows that read it as halo.
    held = []
    stats = Stats()
    reach = REACH * slack[1] * sigma
    for win in wins:
        h, w = win.hw
        nb = np.concatenate(held) if held else np.zeros((0, 4))
        if len(nb):
            dy = nb[:, 1] - np.clip(nb[:, 1], win.y0, win.y0 + h - 1)
            dx = nb[:, 2] - np.clip(nb[:, 2], win.x0, win.x0 + w - 1)
            nb = nb[np.hypot(dy, dx) <= reach]
        win.halo = np.ascontiguousarray(win.shape + _render(nb, win.y0, win.x0, h, w))
        e = search(win, sigma, u, slack, k_max, stats).ems().copy()
        e[:, 1] += win.y0
        e[:, 2] += win.x0
        held.append(e)
    ems = np.concatenate(held) if held else np.zeros((0, 4))
    return Result(ems[:, 1:3], ems[:, 0], ems[:, 3], background, phi,
                  seeds, seed_z, z0, stats, u)
