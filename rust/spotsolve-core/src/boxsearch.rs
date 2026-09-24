//! Score-gated spot detection in ADU above the camera offset.
//!
//! One statistic decides every change of count: the efficient score z for
//! adding one reference-width emitter at a pixel, given everything already
//! fitted in the window (level and emitters, all parameters free).
//!
//! 1. Background: a [`BG_WIN`] median of the frame rounded to whole ADU;
//!    dispersion `phi`: one scalar from the fourth difference.
//! 2. Seeds: local maxima of the K = 0 score (a zero-mean matched filter)
//!    with z > u, strongest first. u is solved from `fp_per_mpx`, the
//!    expected false emitters per 10^6 noise pixels ([`threshold`]).
//! 3. Each seed owns a window, visited strongest first; emitters of earlier
//!    windows are fixed light (halo). A window adds an emitter at the owned
//!    pixel of highest z, only if z > u, and keeps it only if the refit
//!    gains u^2 / 2 dispersion-scaled nats. Widths float within `slack`.
//!
//! The design and every constant were measured on the Python prototype
//! (commit 7b7dfe7, `src/spotsolve/scoregate.py`); the numbers are in
//! `output/scoregate/*.json` and `tests/fixtures/10_scoregate.json` pins its
//! outputs. The LoG/10-nat search this replaced is in git history.

use crate::filters::{self, Mode};
use crate::joint;
use crate::linalg::Chol;
use crate::lmcl::{self, Bounds, FitOpts, FitWorkspace};
use crate::patches;
use crate::psf;
use crate::statistics;

/// Optimization bounds for fitted widths, as multiples of `sigma`.
pub const SLACK: (f64, f64) = (0.70, 2.2);
/// Default count knob: expected false emitters per 10^6 pixels of pure
/// noise. At sigma 1.45 this is u = 4.0, which had the best isolated recall
/// at matched false positives (output/scoregate/u_sweep.json).
pub const FP_PER_MPX: f64 = 16.0;
/// Per Mpx: false emitters = RFT_C * u * exp(-u^2/2) / sigma^2, the
/// Euler-characteristic density of a Gaussian-smoothed field (Lambda =
/// 1 / (2 sigma^2)). 0.80 is measured: accepted noise emitters over 4 Mpx at
/// sigma 1.45, u 4.0-4.47, fell 0.80x below the continuous formula.
pub const RFT_C: f64 = 0.80 * 1e6 / (2.0 * 15.749_609_945_722_419);
/// sigma. A window places emitters within this of its seed, on pixels no
/// nearer another seed. Dense-field recall was 0.729 at 4, 0.733 at 6.
pub const OWN: f64 = 4.0;
/// sigma. Context beyond the placement radius, so an emitter placed at its
/// edge keeps its support; also the edge-diagnostic distance. With the
/// window edge at OWN instead, dense-field precision was 0.907 vs 0.955.
pub const SUPPORT: f64 = 3.0;
/// Emitter widths. Light of earlier windows' emitters within this of a
/// window enters its halo; 5 widths gave identical results.
pub const REACH: f64 = 3.0;
/// Safety cap on emitters per window; a cap of 4 changed GEM counts by 2%.
pub const K_MAX: usize = 12;
/// px. Side of the median background window.
pub const BG_WIN: usize = 25;
/// ADU. `W = 1/m` is singular at `m = 0`; this is far below one count.
pub const BG_FLOOR: f64 = 1e-3;
/// Search-fit iteration budget and objective tolerance (nats).
pub const FIT_MAX_ITER: usize = 100;
pub const FIT_TOL_OBJ: f64 = 1e-6;
/// Numerical amplitude floor: `max(A_MIN, A_MIN_REL * A_max)`.
pub const A_MIN: f64 = 1e-4;
pub const A_MIN_REL: f64 = 1e-6;
/// Relative tolerance for reporting an active optimization bound.
pub const BOUND_TOL: f64 = 1e-6;

/// Per-emitter diagnostics; flags never remove a fitted emitter.
pub const FLAG_EDGE: u8 = 1;
pub const FLAG_NOT_CONVERGED: u8 = 2;
pub const FLAG_STALLED: u8 = 4;
pub const FLAG_COVARIANCE: u8 = 8;
pub const FLAG_BOUND: u8 = 16;

/// What a caller chooses per frame.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// In-focus PSF width, px.
    pub sigma: f64,
    /// Expected false emitters per 10^6 noise pixels; sets u.
    pub fp_per_mpx: f64,
    /// Widths a fit may take, as multiples of `sigma`.
    pub slack: (f64, f64),
}

/// All emitters with diagnostics. Positions are global `(y, x)`.
#[derive(Clone, Debug, Default)]
pub struct Output {
    /// `2N`, row-major `(y, x)`.
    pub pos: Vec<f64>,
    pub amp: Vec<f64>,
    pub sig: Vec<f64>,
    /// `3N`: SE of `(A, y, x)` from the window's final Fisher matrix, scaled
    /// by the dispersion; NaN without.
    pub se: Vec<f64>,
    /// `N`: SE of each fitted width, likewise.
    pub se_sig: Vec<f64>,
    /// `4N`: conditional/marginal variance ratios for `(A, y, x, sigma)`:
    /// `1 / (F_qq * (F^-1)_qq)`. Small values indicate parameter confounding.
    pub fisher_fraction: Vec<f64>,
    /// `N`: bitwise combination of `FLAG_*` diagnostics.
    pub flags: Vec<u8>,
    /// Fitted background at each emitter's nearest pixel, excluding neighbours.
    pub fitted_background: Vec<f64>,
    /// Scalar dispersion: pixel variance per unit of signal, ADU.
    pub dispersion: f64,
    /// `H*W`, ADU above the offset.
    pub background: Vec<f64>,
    /// The score threshold used.
    pub u: f64,
    pub n_seeds: usize,
    /// All fits, one-pass and joint.
    pub fits: usize,
    /// Additions that passed the score test but failed the LR confirmation.
    pub lr_fail: usize,
    /// Joint model: additions and removals after convergence, outer rounds,
    /// and the empirical-null scale of the last add round.
    pub adds: usize,
    pub removed: usize,
    pub outer: usize,
    pub kappa: f64,
}

/// Reusable per-thread storage: one per worker, never shared.
pub struct Workspace {
    pub(crate) fit: FitWorkspace,
    pub(crate) f: psf::Factors,
    pub(crate) chol: Chol,
    pub(crate) scratch: Vec<f64>,
    pub(crate) model: Vec<f64>,
    pub(crate) jac: Vec<f64>,
}

impl Workspace {
    pub fn new() -> Self {
        Self {
            fit: FitWorkspace::new(),
            f: psf::Factors::new(1, 1, 1),
            chol: Chol::new(1),
            scratch: Vec::new(),
            model: Vec::new(),
            jac: Vec::new(),
        }
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Score threshold u giving `fp_per_mpx` expected noise false emitters:
/// solves `RFT_C * u * exp(-u^2/2) / sigma^2 = fp_per_mpx` on `u >= 1`,
/// where the left side decreases, by bisection.
pub fn threshold(sigma: f64, fp_per_mpx: f64) -> f64 {
    let rate = |u: f64| RFT_C * u * (-0.5 * u * u).exp() / (sigma * sigma);
    let (mut lo, mut hi) = (1.0, 40.0);
    if rate(lo) <= fp_per_mpx {
        return lo;
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if rate(mid) > fp_per_mpx {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

pub(crate) fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 0 {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    } else {
        s[n / 2]
    }
}

/// `scipy.ndimage.median_filter(np.rint(d), size=BG_WIN, mode="reflect")`.
///
/// Rounding to whole ADU (error <= 0.5 ADU, far below the noise) lets a
/// sliding histogram keep the median exact at O(BG_WIN) per pixel. Values
/// spanning more than 2^22 ADU fall back to a per-pixel selection.
pub fn median_background(d: &[f64], h: usize, w: usize) -> Vec<f64> {
    let v: Vec<f64> = d.iter().map(|x| x.round_ties_even()).collect();
    let r = (BG_WIN / 2) as isize;
    let n_win = BG_WIN * BG_WIN;
    let k = n_win / 2;
    let (lo, hi) = v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    let mut out = vec![0.0; h * w];
    let ry = |y: usize, dy: isize| Mode::Reflect.index(y as isize + dy, h);
    let rx = |x: usize, dx: isize| Mode::Reflect.index(x as isize + dx, w);
    if !(hi - lo <= (1u64 << 22) as f64) {
        let mut buf = Vec::with_capacity(n_win);
        for y in 0..h {
            for x in 0..w {
                buf.clear();
                for dy in -r..=r {
                    for dx in -r..=r {
                        buf.push(v[ry(y, dy) * w + rx(x, dx)]);
                    }
                }
                let (_, m, _) = buf.select_nth_unstable_by(k, f64::total_cmp);
                out[y * w + x] = *m;
            }
        }
        return out;
    }
    let bins: Vec<usize> = v.iter().map(|&x| (x - lo) as usize).collect();
    let mut hist = vec![0u32; (hi - lo) as usize + 1];
    let mut med = 0usize;
    let mut rows = Vec::with_capacity(BG_WIN);
    for y in 0..h {
        rows.clear();
        rows.extend((-r..=r).map(|dy| ry(y, dy) * w));
        for dx in -r..=r {
            let c = rx(0, dx);
            for &row in &rows {
                hist[bins[row + c]] += 1;
            }
        }
        let mut lt = 0;
        for dx in -r..=r {
            let c = rx(0, dx);
            lt += rows.iter().filter(|&&row| bins[row + c] < med).count();
        }
        for x in 0..w {
            if x > 0 {
                let (gone, came) = (rx(x - 1, -r), rx(x, r));
                for &row in &rows {
                    let (bg, bc) = (bins[row + gone], bins[row + came]);
                    hist[bg] -= 1;
                    lt -= (bg < med) as usize;
                    hist[bc] += 1;
                    lt += (bc < med) as usize;
                }
            }
            // med is the smallest value with more than k samples at or below it.
            while lt > k {
                med -= 1;
                lt -= hist[med] as usize;
            }
            while lt + hist[med] as usize <= k {
                lt += hist[med] as usize;
                med += 1;
            }
            out[y * w + x] = lo + med as f64;
        }
        for dx in -r..=r {
            let c = rx(w - 1, dx);
            for &row in &rows {
                hist[bins[row + c]] -= 1;
            }
        }
    }
    out
}

/// Scalar `phi = pixel variance / mean`. The separable fourth difference
/// `[1, -4, 6, -4, 1]` of white noise has variance `var * 70^2`, and its
/// square has median `var * 70^2 * CHI2_1_MEDIAN`; the median pixel is taken
/// to be background. Frames too small to filter are taken as Poisson.
pub fn dispersion(d: &[f64], h: usize, w: usize) -> f64 {
    if h < 5 || w < 5 {
        return 1.0;
    }
    const K: [f64; 5] = [1.0, -4.0, 6.0, -4.0, 1.0];
    let mut a = vec![0.0; h * w];
    let mut b = vec![0.0; h * w];
    filters::convolve1d(d, &mut a, h, w, &K, 0, Mode::Reflect);
    filters::convolve1d(&a, &mut b, h, w, &K, 1, Mode::Reflect);
    let sq: Vec<f64> = (2..h - 2)
        .flat_map(|r| (2..w - 2).map(move |c| (r, c)))
        .map(|(r, c)| b[r * w + c] * b[r * w + c])
        .collect();
    let var = median(&sq) / (statistics::CHI2_1_MEDIAN * 70.0 * 70.0);
    var / median(d).max(1e-6)
}

/// The centred pixel-integrated unit-flux PSF along one axis, radius
/// `ceil(4 sigma)`; the 2-D kernel is its outer product.
pub(crate) fn psf_kernel1d(sigma: f64) -> Vec<f64> {
    let r = (4.0 * sigma).ceil() as isize;
    let t: Vec<f64> = (-r..=r).map(|v| v as f64).collect();
    let mut k = vec![0.0; t.len()];
    psf::shape_axis(&t, &[0.0], sigma, &mut k);
    k
}

/// Separable 2-D correlation `ky (x) kx` with symmetric kernels.
fn sep(src: &[f64], h: usize, w: usize, ky: &[f64], kx: &[f64], mode: Mode) -> Vec<f64> {
    let mut a = vec![0.0; h * w];
    let mut b = vec![0.0; h * w];
    filters::convolve1d(src, &mut a, h, w, ky, 0, mode);
    filters::convolve1d(&a, &mut b, h, w, kx, 1, mode);
    b
}

/// The same with zeros outside the array, into `out`.
fn sep_zero(src: &[f64], h: usize, w: usize, k: &[f64], tmp: &mut Vec<f64>, out: &mut [f64]) {
    let r = (k.len() / 2) as isize;
    tmp.clear();
    tmp.resize(h * w, 0.0);
    for i in 0..h {
        for j in 0..w {
            let mut acc = 0.0;
            for (t, &kv) in k.iter().enumerate() {
                let y = i as isize + t as isize - r;
                if y >= 0 && (y as usize) < h {
                    acc += src[y as usize * w + j] * kv;
                }
            }
            tmp[i * w + j] = acc;
        }
    }
    for i in 0..h {
        for j in 0..w {
            let mut acc = 0.0;
            for (t, &kv) in k.iter().enumerate() {
                let x = j as isize + t as isize - r;
                if x >= 0 && (x as usize) < w {
                    acc += tmp[i * w + x as usize] * kv;
                }
            }
            out[i * w + j] = acc;
        }
    }
}

/// Frame-wide K = 0 score z, blind to a local constant level: correlation
/// of `r = d - background` with the zero-mean kernel `g - mean(g)`, over
/// the square root of `var` correlated with its square. Both expand into
/// separable filters and box sums.
pub fn detection_map(r: &[f64], var: &[f64], h: usize, w: usize, sigma: f64) -> Vec<f64> {
    let k1 = psf_kernel1d(sigma);
    let k2: Vec<f64> = k1.iter().map(|v| v * v).collect();
    let ones = vec![1.0; k1.len()];
    let s1: f64 = k1.iter().sum();
    let c = s1 * s1 / (k1.len() * k1.len()) as f64;
    let rg = sep(r, h, w, &k1, &k1, Mode::Reflect);
    let rb = sep(r, h, w, &ones, &ones, Mode::Reflect);
    let vg2 = sep(var, h, w, &k2, &k2, Mode::Reflect);
    let vg = sep(var, h, w, &k1, &k1, Mode::Reflect);
    let vb = sep(var, h, w, &ones, &ones, Mode::Reflect);
    (0..h * w)
        .map(|i| {
            let num = rg[i] - c * rb[i];
            let den = vg2[i] - 2.0 * c * vg[i] + c * c * vb[i];
            num / den.max(1e-12).sqrt()
        })
        .collect()
}

/// Local maxima of `z` over a `2 ceil(sigma) + 1` window with z > u, as
/// pixel indices, strongest first.
pub fn find_seeds(z: &[f64], h: usize, w: usize, sigma: f64, u: f64) -> Vec<usize> {
    let win = 2 * sigma.ceil() as usize + 1;
    let mx = filters::maximum_filter(z, h, w, win, Mode::Reflect);
    let mut s: Vec<usize> = (0..h * w).filter(|&i| z[i] == mx[i] && z[i] > u).collect();
    s.sort_by(|&a, &b| z[b].total_cmp(&z[a]));
    s
}

/// Whether the emitter's support crosses a physical frame edge.
fn edge_truncated(y: f64, x: f64, sigma: f64, h: usize, w: usize) -> bool {
    let border = (y + 0.5).min(h as f64 - 0.5 - y)
        .min(x + 0.5).min(w as f64 - 0.5 - x);
    border < SUPPORT * sigma
}

pub(crate) fn at_bound(value: f64, lo: f64, hi: f64) -> bool {
    let margin = 2.0 * lmcl::INTERIOR_FRAC * (hi - lo).max(1e-12);
    value - lo <= margin + BOUND_TOL * (1.0 + lo.abs())
        || hi - value <= margin + BOUND_TOL * (1.0 + hi.abs())
}

/// Window half-side, px.
fn half_side(sigma: f64) -> usize {
    ((OWN + SUPPORT) * sigma).ceil() as usize
}

/// Context needed around an ROI: a window around any seed, plus the median
/// window and score kernel that seed's background and z read.
fn crop_margin(sigma: f64) -> usize {
    let kernel = (4.0 * sigma).ceil() as usize + sigma.ceil() as usize;
    half_side(sigma).max(kernel) + 1 + BG_WIN / 2 + 2
}

/// Widen `[lo, hi)` to at least `want` pixels without leaving `[0, n)`, and
/// without ever giving up ground it already held.
fn widen(lo: usize, hi: usize, want: usize, n: usize) -> (usize, usize) {
    let want = want.min(n);
    if hi - lo >= want {
        return (lo, hi);
    }
    let mid = (lo + hi) / 2;
    let start = mid.saturating_sub(want / 2).min(n - want);
    (start.min(lo), (start + want).max(hi))
}

/// The ROI's bounding box plus [`crop_margin`], at least `3 * BG_WIN` a
/// side where the frame allows, so the scalar dispersion has pixels to read.
/// `None` when the ROI selects no pixel.
fn roi_crop(roi: &[bool], h: usize, w: usize, sigma: f64) -> Option<patches::BBox> {
    let (mut y0, mut y1) = (usize::MAX, 0usize);
    let (mut x0, mut x1) = (usize::MAX, 0usize);
    for r in 0..h {
        for c in 0..w {
            if roi[r * w + c] {
                y0 = y0.min(r);
                y1 = y1.max(r + 1);
                x0 = x0.min(c);
                x1 = x1.max(c + 1);
            }
        }
    }
    if y0 == usize::MAX {
        return None;
    }
    let m = crop_margin(sigma);
    let side = 3 * BG_WIN;
    let (y0, y1) = widen(y0.saturating_sub(m), (y1 + m).min(h), side, h);
    let (x0, x1) = widen(x0.saturating_sub(m), (x1 + m).min(w), side, w);
    Some(patches::BBox { y0, x0, y1, x1 })
}

fn crop<T: Copy>(v: &[T], w: usize, bb: &patches::BBox) -> Vec<T> {
    let mut out = Vec::with_capacity(bb.n_pixels());
    for r in bb.y0..bb.y1 {
        out.extend_from_slice(&v[r * w + bb.x0..r * w + bb.x1]);
    }
    out
}

/// An emitter: `[A, y, x, sigma]`.
pub type Em = [f64; 4];

/// A window's pixels, its fixed light, and where it may place.
pub(crate) struct Window {
    pub(crate) y0: usize,
    pub(crate) x0: usize,
    pub(crate) h: usize,
    pub(crate) w: usize,
    pub(crate) sub: Vec<f64>,
    /// Earlier windows' emitters plus the background's shape.
    pub(crate) halo: Vec<f64>,
    /// Background median here; where the free level starts.
    pub(crate) level: f64,
    /// Background minus `level`.
    pub(crate) shape: Vec<f64>,
    pub(crate) owned: Vec<bool>,
    pub(crate) phi: f64,
}

impl Window {
    /// Pixels of `bb` in an `fw`-wide frame `d` and background `bmap`.
    fn new(d: &[f64], fw: usize, bmap: &[f64], bb: &patches::BBox, owned: Vec<bool>, phi: f64) -> Self {
        let (h, w) = (bb.h(), bb.w());
        let sub = crop(d, fw, bb);
        let bg = crop(bmap, fw, bb);
        let level = median(&bg);
        let shape: Vec<f64> = bg.iter().map(|v| v - level).collect();
        Self { y0: bb.y0, x0: bb.x0, h, w, sub, halo: shape.clone(), level, shape, owned, phi }
    }

    /// `halo <- (emitters, rendered here) + shape`. `ems` are global.
    fn set_halo(&mut self, ems: &[Em], f: &mut psf::Factors) {
        let n = self.h * self.w;
        self.halo.clear();
        self.halo.resize(n, 0.0);
        if !ems.is_empty() {
            let mut theta = Vec::with_capacity(4 * ems.len() + 1);
            theta.push(0.0);
            for e in ems {
                theta.extend_from_slice(&[e[0], e[1] - self.y0 as f64, e[2] - self.x0 as f64, e[3]]);
            }
            let (ay, ax) = (psf::local_axis(self.h), psf::local_axis(self.w));
            f.ensure(self.h, self.w, ems.len());
            psf::model_var_sigma_ax(&theta, &ay, &ax, None, f, &mut self.halo);
        }
        for (v, s) in self.halo.iter_mut().zip(&self.shape) {
            *v += s;
        }
    }
}

/// One window fit: data-only I-divergence and parameters in local coordinates.
#[derive(Clone)]
pub(crate) struct Fitted {
    pub(crate) i_div: f64,
    pub(crate) b: f64,
    pub(crate) em: Vec<Em>,
    pub(crate) flags: Vec<u8>,
}

impl Fitted {
    fn theta(&self) -> Vec<f64> {
        let mut t = Vec::with_capacity(4 * self.em.len() + 1);
        t.push(self.b);
        for e in &self.em {
            t.extend_from_slice(e);
        }
        t
    }
}

/// Bounded free-width ML fit, with a strictly interior starting point.
/// Scale the peak-derived flux bound by `slack.1^2` to allow broad sources.
/// Positions remain inside the window.
fn fit_window(
    ws: &mut Workspace,
    win: &Window,
    b: f64,
    em: &[Em],
    s: &Settings,
    max_iter: usize,
    tol_obj: f64,
) -> Fitted {
    let k = em.len();
    let smax = win.sub.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v)).max(1.0);
    let b_max = (4.0 * smax).max(10.0);
    let a_max = 8.0 * smax / psf::peak_factor(s.sigma) * s.slack.1 * s.slack.1;
    let a_min = A_MIN.max(A_MIN_REL * a_max);
    let (s_lo, s_hi) = (s.slack.0 * s.sigma, s.slack.1 * s.sigma);
    let mut lo = Vec::with_capacity(4 * k + 1);
    let mut hi = Vec::with_capacity(4 * k + 1);
    lo.push(0.0);
    hi.push(b_max);
    for _ in 0..k {
        lo.extend_from_slice(&[a_min, -0.5, -0.5, s_lo]);
        hi.extend_from_slice(&[a_max, win.h as f64 - 0.5, win.w as f64 - 0.5, s_hi]);
    }
    let mut th0 = Vec::with_capacity(4 * k + 1);
    th0.push(b);
    for e in em {
        th0.extend_from_slice(&[e[0], e[1], e[2], e[3].clamp(s_lo, s_hi)]);
    }
    for q in 0..th0.len() {
        th0[q] = th0[q].clamp(lo[q] + 1e-9, hi[q] - 1e-9);
    }
    let bounds = Bounds::new(&lo, &hi);
    let info = lmcl::fit_var_sigma(
        &mut ws.fit,
        &th0,
        win.h,
        win.w,
        &win.sub,
        &bounds,
        Some(&win.halo),
        FitOpts {
            max_iter,
            tol_obj,
            ..Default::default()
        },
    );
    let t = ws.fit.theta();
    Fitted {
        i_div: info.i_div,
        b: t[0],
        flags: (0..k).map(|j| {
            let mut flag = 0;
            if !info.converged { flag |= FLAG_NOT_CONVERGED; }
            if info.stalled { flag |= FLAG_STALLED; }
            if at_bound(t[0], lo[0], hi[0])
                || (1 + 4*j..5 + 4*j).any(|q| at_bound(t[q], lo[q], hi[q])) {
                flag |= FLAG_BOUND;
            }
            flag
        }).collect(),
        em: (0..k)
            .map(|j| [t[1 + 4 * j], t[2 + 4 * j], t[3 + 4 * j], t[4 + 4 * j]])
            .collect(),
    }
}

/// Model, Jacobian (parameter-major) and Poisson Fisher matrix
/// `J^T diag(1/m) J` of `state` in `win`, into the workspace.
pub(crate) fn information(ws: &mut Workspace, win: &Window, state: &Fitted) -> Vec<f64> {
    let n = win.h * win.w;
    let theta = state.theta();
    let p = theta.len();
    let (ay, ax) = (psf::local_axis(win.h), psf::local_axis(win.w));
    ws.f.ensure(win.h, win.w, state.em.len().max(1));
    ws.model.clear();
    ws.model.resize(n, 0.0);
    ws.jac.clear();
    ws.jac.resize(p * n, 0.0);
    psf::model_and_jac_var_sigma_ax(&theta, &ay, &ax, Some(&win.halo), &mut ws.f, &mut ws.model, &mut ws.jac);
    let mut f = vec![0.0; p * p];
    for q in 0..p {
        for r in 0..=q {
            let mut acc = 0.0;
            for i in 0..n {
                acc += ws.jac[q * n + i] * ws.jac[r * n + i] / ws.model[i].max(BG_FLOOR);
            }
            f[q * p + r] = acc;
            f[r * p + q] = acc;
        }
    }
    f
}

/// Best owned pixel for one more reference-width emitter:
/// `(z, local index, one-step amplitude S / I_eff)`.
///
/// `S = sum g_p (d - m) / (phi m)`; `I_eff` is `sum g_p^2 / (phi m)` less
/// its projection onto the current parameters' information. Without that
/// projection a fitted neighbour's light leaks into the test: a level-only
/// score resolved 1% of 2-sigma pairs against 64%.
pub(crate) fn efficient_score(ws: &mut Workspace, win: &Window, state: &Fitted, k1: &[f64]) -> Option<(f64, usize, f64)> {
    let n = win.h * win.w;
    let fp = information(ws, win, state);
    let p = state.em.len() * 4 + 1;
    let (h, w) = (win.h, win.w);
    let wt: Vec<f64> = ws.model.iter().map(|m| 1.0 / (win.phi * m.max(BG_FLOOR))).collect();
    let k2: Vec<f64> = k1.iter().map(|v| v * v).collect();
    let mut tmp = Vec::new();
    let mut s = vec![0.0; n];
    let resid: Vec<f64> = (0..n).map(|i| (win.sub[i] - ws.model[i]) * wt[i]).collect();
    sep_zero(&resid, h, w, k1, &mut tmp, &mut s);
    let mut igg = vec![0.0; n];
    sep_zero(&wt, h, w, &k2, &mut tmp, &mut igg);
    // C[q][i] = sum_j g_i(j) J_q(j) wt(j); F = J^T diag(wt) J = Fisher / phi.
    let mut c = vec![0.0; p * n];
    let mut col = vec![0.0; n];
    for q in 0..p {
        for i in 0..n {
            col[i] = ws.jac[q * n + i] * wt[i];
        }
        sep_zero(&col, h, w, k1, &mut tmp, &mut c[q * n..(q + 1) * n]);
    }
    let f: Vec<f64> = fp.iter().map(|v| v / win.phi).collect();
    ws.chol.ensure(p);
    let proj = ws.chol.factor(&f, p);
    let mut best: Option<(f64, usize, f64)> = None;
    let mut cq = vec![0.0; p];
    let mut x = vec![0.0; p];
    for i in 0..n {
        if !win.owned[i] {
            continue;
        }
        let mut ieff = igg[i];
        if proj {
            for q in 0..p {
                cq[q] = c[q * n + i];
            }
            ws.chol.solve(&cq, &mut x);
            ieff -= cq.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>();
        }
        let ieff = ieff.max(1e-12);
        let z = s[i] / ieff.sqrt();
        if best.is_none_or(|(bz, _, _)| z > bz) {
            best = Some((z, i, s[i] / ieff));
        }
    }
    best
}

/// Score-gated additions from K = 0, each confirmed by the LR.
/// Returns the accepted state, the fits spent and LR failures.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn search(ws: &mut Workspace, win: &Window, s: &Settings, u: f64, k1: &[f64]) -> (Fitted, usize, usize) {
    let gain = 0.5 * u * u * win.phi;
    let mut state = fit_window(ws, win, win.level, &[], s, FIT_MAX_ITER, FIT_TOL_OBJ);
    let (mut fits, mut lr_fail) = (1, 0);
    while state.em.len() < K_MAX {
        let Some((z, i, a)) = efficient_score(ws, win, &state, k1) else { break };
        if !(z > u) {
            break;
        }
        let mut em = state.em.clone();
        em.push([a, (i / win.w) as f64, (i % win.w) as f64, s.sigma]);
        let trial = fit_window(ws, win, state.b, &em, s, FIT_MAX_ITER, FIT_TOL_OBJ);
        fits += 1;
        if !(state.i_div - trial.i_div > gain) {
            lr_fail += 1;
            break;
        }
        state = trial;
    }
    (state, fits, lr_fail)
}

/// The one-pass search on a frame (or crop) `d`: its emitters in `d`'s
/// coordinates, the median background it used, and its counters.
#[derive(Clone, Debug, Default)]
pub struct OnePass {
    pub ems: Vec<Em>,
    pub background: Vec<f64>,
    pub dispersion: f64,
    pub u: f64,
    pub n_seeds: usize,
    pub fits: usize,
    pub lr_fail: usize,
}

/// Seeds, then each seed's window decided once, strongest first, against
/// the emitters of earlier windows as fixed light. `roi` (same shape as `d`)
/// limits seeds and placements. This is the joint model's starting point.
pub fn one_pass(d: &[f64], h: usize, w: usize, roi: Option<&[bool]>, s: &Settings, ws: &mut Workspace) -> OnePass {
    assert_eq!(d.len(), h * w);
    let u = threshold(s.sigma, s.fp_per_mpx);
    let bmap = median_background(d, h, w);
    let phi = dispersion(d, h, w);
    let resid: Vec<f64> = d.iter().zip(&bmap).map(|(a, b)| a - b).collect();
    let var: Vec<f64> = bmap.iter().map(|b| phi * b.max(BG_FLOOR)).collect();
    let z = detection_map(&resid, &var, h, w, s.sigma);
    let seeds: Vec<(usize, usize)> = find_seeds(&z, h, w, s.sigma, u)
        .into_iter()
        .map(|i| (i / w, i % w))
        .filter(|&(y, x)| roi.is_none_or(|m| m[y * w + x]))
        .collect();

    let pad = half_side(s.sigma);
    let own = OWN * s.sigma;
    let reach = REACH * s.slack.1 * s.sigma;
    let k1 = psf_kernel1d(s.sigma);
    let mut out = OnePass { dispersion: phi, u, n_seeds: seeds.len(), ..OnePass::default() };
    let mut ems: Vec<Em> = Vec::new();
    for (i, &(sy, sx)) in seeds.iter().enumerate() {
        let wb = patches::BBox {
            y0: sy.saturating_sub(pad),
            x0: sx.saturating_sub(pad),
            y1: (sy + pad + 1).min(h),
            x1: (sx + pad + 1).min(w),
        };
        let near: Vec<(f64, f64)> = seeds
            .iter()
            .enumerate()
            .filter(|&(j, &(y, x))| j != i && y.abs_diff(sy) <= 2 * pad && x.abs_diff(sx) <= 2 * pad)
            .map(|(_, &(y, x))| (y as f64, x as f64))
            .collect();
        let mut owned = Vec::with_capacity(wb.n_pixels());
        for py in wb.y0..wb.y1 {
            for px in wb.x0..wb.x1 {
                let (fy, fx) = (py as f64, px as f64);
                let d_own = (fy - sy as f64).hypot(fx - sx as f64);
                let d_oth = near.iter().map(|&(y, x)| (fy - y).hypot(fx - x)).fold(f64::INFINITY, f64::min);
                owned.push(d_own <= own && d_own <= d_oth && roi.is_none_or(|m| m[py * w + px]));
            }
        }
        let mut win = Window::new(d, w, &bmap, &wb, owned, phi);
        let (y0, y1) = (wb.y0 as f64, (wb.y1 - 1) as f64);
        let (x0, x1) = (wb.x0 as f64, (wb.x1 - 1) as f64);
        ems.clear();
        ems.extend(out.ems.iter().filter(|e| {
            (e[1] - e[1].clamp(y0, y1)).hypot(e[2] - e[2].clamp(x0, x1)) <= reach
        }));
        win.set_halo(&ems, &mut ws.f);
        let (state, fits, lr_fail) = search(ws, &win, s, u, &k1);
        out.fits += fits;
        out.lr_fail += lr_fail;
        out.ems.extend(state.em.iter().map(|e| [e[0], e[1] + y0, e[2] + x0, e[3]]));
    }
    out.background = bmap;
    out
}

/// Localize an offset-subtracted frame in ADU: [`one_pass`], then the joint
/// model ([`joint::Joint`]), then uncertainties from each final group's
/// Fisher matrix. An ROI limits seeds and additions; everything runs on its
/// bounding box plus enough context for every filter, window and group.
/// Returns global coordinates and a full-frame background map (the fitted
/// node surface). An empty ROI returns no detections.
pub fn localize(
    d: &[f64],
    h: usize,
    w: usize,
    roi: Option<&[bool]>,
    s: &Settings,
    ws: &mut Workspace,
) -> Output {
    assert_eq!(d.len(), h * w);
    let bb = match roi {
        None => patches::BBox { y0: 0, x0: 0, y1: h, x1: w },
        Some(m) => match roi_crop(m, h, w, s.sigma) {
            Some(bb) => bb,
            None => {
                return Output {
                    background: vec![BG_FLOOR; h * w],
                    dispersion: f64::NAN,
                    u: threshold(s.sigma, s.fp_per_mpx),
                    kappa: 1.0,
                    ..Output::default()
                }
            }
        },
    };
    let whole = bb.h() == h && bb.w() == w;
    let (ch, cw) = (bb.h(), bb.w());
    let (dsub, rsub) = if whole {
        (Vec::new(), None)
    } else {
        (crop(d, w, &bb), roi.map(|m| crop(m, w, &bb)))
    };
    let dc: &[f64] = if whole { d } else { &dsub };
    let rc: Option<&[bool]> = if whole { roi } else { rsub.as_deref() };

    let first = one_pass(dc, ch, cw, rc, s, ws);
    let mut jm = joint::Joint::new(dc, ch, cw, &first.ems, &first.background, first.dispersion, first.u, s, rc, ws);
    jm.run(ws);
    let mut out = report(&jm, ws, bb.y0, bb.x0, h, w);
    out.u = first.u;
    out.n_seeds = first.n_seeds;
    out.fits = first.fits + jm.stats.fits;
    out.lr_fail = first.lr_fail + jm.stats.lr_fail;
    // Outside the crop nothing was estimated: the map carries a fill there.
    out.background = if whole {
        jm.bg
    } else {
        let fill = median(&jm.bg);
        let mut full = vec![fill; h * w];
        for r in 0..ch {
            full[(bb.y0 + r) * w + bb.x0..(bb.y0 + r) * w + bb.x1].copy_from_slice(&jm.bg[r * cw..(r + 1) * cw]);
        }
        full
    };
    out
}

/// Output rows for the joint model's emitters, in its order, shifted by
/// `(oy, ox)` into an `h x w` frame. SEs come from each final group's
/// Fisher matrix with a free local level: the group fit locks the level
/// to the nodes, but the nodes are themselves estimated, and a free level
/// is the conservative stand-in for that.
fn report(jm: &joint::Joint, ws: &mut Workspace, oy: usize, ox: usize, h: usize, w: usize) -> Output {
    let n = jm.ems.len();
    let mut out = Output {
        dispersion: jm.phi,
        adds: jm.stats.adds,
        removed: jm.stats.removed,
        outer: jm.stats.outer,
        kappa: jm.stats.kappa,
        se: vec![f64::NAN; 3 * n],
        se_sig: vec![f64::NAN; n],
        fisher_fraction: vec![f64::NAN; 4 * n],
        ..Output::default()
    };
    let mut se4 = vec![f64::NAN; 4 * n];
    let mut var_q = Vec::new();
    for g in jm.groups(&jm.model(), ws) {
        let bb = jm.group_box(&g);
        let (win, _) = jm.window(&g, &bb, ws);
        let state = Fitted {
            i_div: f64::NAN,
            b: 0.0,
            em: g.iter().map(|&i| {
                let e = jm.ems[i];
                [e[0], e[1] - bb.y0 as f64, e[2] - bb.x0 as f64, e[3]]
            }).collect(),
            flags: Vec::new(),
        };
        let fisher = information(ws, &win, &state);
        let p = fisher.len().isqrt();
        var_q.clear();
        var_q.resize(p, f64::NAN);
        ws.chol.ensure(p);
        if ws.chol.factor(&fisher, p) {
            ws.chol.inv_diag(&mut var_q, &mut ws.scratch);
        }
        let idx: Vec<u32> = g.iter().map(|&i| i as u32).collect();
        store_uncertainties(&fisher, &var_q, &idx, &mut se4, &mut out.fisher_fraction);
    }
    let scale = jm.phi.sqrt();
    for (i, e) in jm.ems.iter().enumerate() {
        let (gy, gx) = (e[1] + oy as f64, e[2] + ox as f64);
        out.pos.extend_from_slice(&[gy, gx]);
        out.amp.push(e[0]);
        out.sig.push(e[3]);
        for c in 0..3 {
            out.se[3 * i + c] = se4[4 * i + c] * scale;
        }
        out.se_sig[i] = se4[4 * i + 3] * scale;
        let mut flag = jm.flags[i];
        if edge_truncated(gy, gx, e[3], h, w) {
            flag |= FLAG_EDGE;
        }
        if se4[4 * i..4 * i + 4].iter().any(|v| !v.is_finite() || *v <= 0.0) {
            flag |= FLAG_COVARIANCE;
        }
        out.flags.push(flag);
        let py = (e[1].round().max(0.0) as usize).min(jm.h - 1);
        let px = (e[2].round().max(0.0) as usize).min(jm.w - 1);
        out.fitted_background.push(jm.bg[py * jm.w + px]);
    }
    out
}

/// Conditional variance is `1/F_qq`; marginal variance includes all fitted
/// nuisance parameters, including background. Their ratio is dimensionless,
/// invariant to parameter ordering and diagonal rescaling, and costs O(p)
/// once the inverse diagonal needed for SEs is available. It measures local
/// confounding, not absolute precision or the probability an emitter is real.
fn store_uncertainties(
    fisher: &[f64],
    var: &[f64],
    indices: &[u32],
    se: &mut [f64],
    fraction: &mut [f64],
) {
    let p = var.len();
    for (j, &i) in indices.iter().enumerate() {
        for c in 0..4 {
            let q = 1 + 4 * j + c;
            let dst = 4 * i as usize + c;
            let (v, f) = (var[q], fisher[q * p + q]);
            se[dst] = if v.is_finite() && v > 0.0 { v.sqrt() } else { f64::NAN };
            fraction[dst] = if v.is_finite() && v > 0.0 && f.is_finite() && f > 0.0 {
                // Sequential division avoids overflow in F_qq * var_q.
                // The exact ratio is <= 1; clip roundoff at that endpoint.
                (1.0 / f / v).clamp(0.0, 1.0)
            } else {
                f64::NAN
            };
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Localize one raw frame using `d = raw - offset` in ADU.
pub fn localize_raw(
    raw: &[f64],
    h: usize,
    w: usize,
    offset: f64,
    roi: Option<&[bool]>,
    s: &Settings,
    ws: &mut Workspace,
    d: &mut Vec<f64>,
) -> Output {
    d.clear();
    d.extend(raw.iter().map(|&r| r - offset));
    localize(d, h, w, roi, s, ws)
}

/// Localize every frame of a stack on `n_threads` workers. Frames are
/// independent, so each worker takes the next undone frame and keeps its own
/// [`Workspace`]; the output is in frame order whatever the scheduling.
///
/// `raw` is `n*H*W`; each frame goes through [`localize_raw`].
#[allow(clippy::too_many_arguments)]
pub fn localize_stack(
    raw: &[f64],
    n: usize,
    h: usize,
    w: usize,
    offset: f64,
    roi: Option<&[bool]>,
    s: &Settings,
    n_threads: usize,
) -> Vec<Output> {
    assert_eq!(raw.len(), n * h * w);
    crate::frames::map(n, n_threads,
        || (Workspace::new(), Vec::with_capacity(h * w)),
        |t, (ws, d)| {
            let frame = &raw[t * h * w..(t + 1) * h * w];
            localize_raw(frame, h, w, offset, roi, s, ws, d)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic standard normals: an LCG through Box-Muller.
    fn normals(n: usize, mut state: u64) -> Vec<f64> {
        let mut uni = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| (-2.0 * uni().ln()).sqrt() * (2.0 * std::f64::consts::PI * uni()).cos())
            .collect()
    }

    fn settings(sigma: f64) -> Settings {
        Settings { sigma, fp_per_mpx: FP_PER_MPX, slack: SLACK }
    }

    #[test]
    fn the_histogram_median_is_the_reflected_window_median() {
        // Smaller and larger than the window, with ties and negatives.
        for (h, w) in [(7, 9), (40, 31)] {
            let d: Vec<f64> = normals(h * w, 5).iter().map(|v| (8.0 * v).round() - 3.0).collect();
            let got = median_background(&d, h, w);
            let r = (BG_WIN / 2) as isize;
            for y in 0..h {
                for x in 0..w {
                    let mut v = Vec::new();
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let yy = Mode::Reflect.index(y as isize + dy, h);
                            let xx = Mode::Reflect.index(x as isize + dx, w);
                            v.push(d[yy * w + xx]);
                        }
                    }
                    v.sort_by(f64::total_cmp);
                    assert_eq!(got[y * w + x], v[v.len() / 2], "({y}, {x}) in {h}x{w}");
                }
            }
        }
    }

    #[test]
    fn the_threshold_solves_the_calibrated_rate() {
        for (sigma, fp) in [(1.0, 2.0), (1.45, 16.0), (2.0, 100.0)] {
            let u = threshold(sigma, fp);
            let rate = RFT_C * u * (-0.5 * u * u).exp() / (sigma * sigma);
            assert!((rate / fp - 1.0).abs() < 1e-9, "sigma {sigma}: rate {rate}");
        }
        assert!((threshold(1.45, FP_PER_MPX) - 4.003).abs() < 1e-3);
    }

    #[test]
    fn the_dispersion_reads_white_noise_and_scales_with_the_image() {
        let (h, w) = (200, 180);
        // Mean 50, sd 3: phi = 9 / 50.
        let d: Vec<f64> = normals(h * w, 7).iter().map(|v| 50.0 + 3.0 * v).collect();
        let phi = dispersion(&d, h, w);
        assert!((phi - 0.18).abs() < 0.01, "phi {phi}");
        let d7: Vec<f64> = d.iter().map(|v| 7.0 * v).collect();
        assert!((dispersion(&d7, h, w) - 7.0 * phi).abs() < 1e-9 * phi);
    }

    #[test]
    fn fisher_fraction_measures_coupling_independently_of_units_and_order() {
        // Two emitter fluxes correlated through the information matrix;
        // everything else is independent. Each flux retains 1-rho^2 of its
        // conditional information after the other flux is allowed to vary.
        let p = 9;
        let rho = 0.99;
        for order in [[0u32, 1], [1, 0]] {
            for scale in [[1.0; 9], [0.01, 1e-3, 2.0, 3.0, 4.0, 1e3, 5.0, 6.0, 7.0]] {
                let mut f = vec![0.0; p * p];
                for q in 0..p { f[q * p + q] = scale[q] * scale[q]; }
                f[p + 5] = rho * scale[1] * scale[5];
                f[5 * p + 1] = f[p + 5];
                let mut chol = Chol::new(p);
                assert!(chol.factor(&f, p));
                let mut var = vec![0.0; p];
                chol.inv_diag(&mut var, &mut Vec::new());
                let mut se = vec![f64::NAN; 12];
                let mut fraction = se.clone();
                store_uncertainties(&f, &var, &order, &mut se, &mut fraction);
                for i in 0..2 {
                    assert!((fraction[4 * i] - (1.0 - rho * rho)).abs() < 1e-12);
                    for c in 1..4 { assert!((fraction[4 * i + c] - 1.0).abs() < 1e-12); }
                }
                assert!(fraction[8..].iter().all(|v| v.is_nan()));
                var.fill(f64::NAN);
                store_uncertainties(&f, &var, &[1, 2], &mut se, &mut fraction);
                assert!(se[..4].iter().all(|v| v.is_finite()));
                assert!(se[4..].iter().chain(&fraction[4..]).all(|v| v.is_nan()));
            }
        }
    }

    #[test]
    fn fit_flags_distinguish_iteration_limits_and_active_bounds() {
        let (h, w) = (17, 17);
        let mut data = vec![0.0; h*w];
        psf::model_var_sigma_ax(
            &[20.0, 1500.0, 8.2, 8.4, 2.8], &psf::local_axis(h), &psf::local_axis(w),
            None, &mut psf::Factors::new(h, w, 1), &mut data,
        );
        let win = Window::new(&data, w, &vec![20.0; h*w],
            &patches::BBox { y0: 0, x0: 0, y1: h, x1: w }, vec![true; h * w], 1.0);
        let s = settings(1.0);
        let mut ws = Workspace::new();
        let start = [[1200.0, 8.0, 8.0, 1.5]];
        let unfinished = fit_window(&mut ws, &win, 20.0, &start, &s, 0, FIT_TOL_OBJ);
        assert_eq!(unfinished.flags, vec![FLAG_NOT_CONVERGED]);
        let bounded = fit_window(&mut ws, &win, 20.0, &start, &s, 200, FIT_TOL_OBJ);
        assert!(bounded.flags[0] & FLAG_BOUND != 0);
        assert!(bounded.flags[0] & FLAG_NOT_CONVERGED == 0);
        assert!((bounded.em[0][3] - SLACK.1).abs() < 1e-5);
        let wider = Settings { slack: (0.7, 3.5), ..s };
        let recovered = fit_window(&mut ws, &win, 20.0, &start, &wider, 200, FIT_TOL_OBJ);
        assert_eq!(recovered.flags, vec![0]);
        assert!((recovered.em[0][3] - 2.8).abs() < 1e-3);
        assert!((recovered.em[0][0] - 1500.0).abs() < 1.0);
    }

    #[test]
    fn active_bounds_include_the_optimizers_interior_margin() {
        let bounds = Bounds::new(&[100.0], &[1e8]);
        let theta = lmcl::Interior::new(&[0.0], &bounds);
        assert!(at_bound(theta.as_slice()[0], 100.0, 1e8));
        assert!(!at_bound(10000.0, 100.0, 1e8));
    }

    #[test]
    fn edge_diagnostic_tracks_fitted_support_and_physical_pixel_edges() {
        assert!(edge_truncated(2.0, 20.0, 1.0, 40, 40));
        assert!(!edge_truncated(3.0, 20.0, 1.0, 40, 40));
        assert!(edge_truncated(3.0, 20.0, 2.0, 40, 40));
        assert!(edge_truncated(20.0, 37.0, 1.0, 40, 40));
    }

    #[test]
    fn an_isolated_emitter_is_found_once_where_it_is() {
        let (h, w, sigma) = (41, 43, 1.2);
        let theta = [0.0, 1500.0, 19.3, 21.6, 1.3];
        let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
        let mut f = psf::Factors::new(h, w, 1);
        let mut d = vec![0.0; h * w];
        psf::model_var_sigma_ax(&theta, &ay, &ax, None, &mut f, &mut d);
        // Gaussian noise at the Poisson variance on a background of 10.
        for (v, z) in d.iter_mut().zip(normals(h * w, 11)) {
            *v += 10.0;
            *v += v.sqrt() * z;
        }
        let o = localize(&d, h, w, None, &settings(sigma), &mut Workspace::new());
        assert_eq!(o.amp.len(), 1, "found {:?} amp {:?}", o.pos, o.amp);
        assert_eq!(o.flags[0], 0);
        assert!(o.se.iter().chain(&o.se_sig).all(|v| v.is_finite() && *v > 0.0));
        assert!((o.dispersion - 1.0).abs() < 0.3, "dispersion {}", o.dispersion);
        assert!((o.pos[0] - 19.3).abs() < 3.0 * o.se[1], "y {}", o.pos[0]);
        assert!((o.pos[1] - 21.6).abs() < 3.0 * o.se[2], "x {}", o.pos[1]);
        assert!((o.amp[0] - 1500.0).abs() < 3.0 * o.se[0], "A {}", o.amp[0]);
        assert!((o.sig[0] - 1.3).abs() < 3.0 * o.se_sig[0], "sigma {}", o.sig[0]);
    }
}
