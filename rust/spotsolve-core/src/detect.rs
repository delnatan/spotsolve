//! Spot detection in ADU above the camera offset.
//!
//! 1. Background: a [`BG_WIN`] median of the frame rounded to whole ADU;
//!    dispersion `phi`: one scalar from the fourth difference.
//! 2. Seeds: local maxima over position and width of the score for one
//!    emitter against the nodes fitted to that background ([`score_map`]),
//!    on a bank of widths spanning the fit's bounds ([`widths`]), above u
//!    less what that grid can miss of a maximum ([`proposal_factor`]). u is
//!    solved from `fp_per_mpx`, the expected false emitters per 10^6 noise
//!    pixels ([`threshold`]). Each seed becomes one emitter ([`start`]).
//! 3. The joint model ([`Model`]) fits them together with a node
//!    background, removes those not worth `u^2 / 2` nats and adds where the
//!    residual asks for more.
//! 4. Components wider than `slack.1 * sigma` are out of focus: their
//!    light joins the background and they are not reported ([`widest`]).
//! 5. Uncertainties from each final group's Fisher information.

use crate::filters::{self, Mode};
use crate::linalg::Chol;
use crate::model::{self, Em, Model, Rect};
use crate::psf;
use crate::statistics;

/// Widths of reported emitters, as multiples of `sigma`: the in-focus
/// width up to the edge of the depth of focus. Wider components, up to
/// [`widest`], are fitted as background.
pub const SLACK: (f64, f64) = (1.0, 2.25);
/// Default count knob: expected false emitters per 10^6 pixels of pure noise.
pub const FP_PER_MPX: f64 = 16.0;
/// Width ratio of adjacent seed templates: an emitter between two keeps at
/// least `2 sqrt(r) / (1 + r)` = 98% of its score.
pub const WIDTH_STEP: f64 = 1.5;
/// sigma. An add round places emitters within this of a group member.
pub const OWN: f64 = 4.0;
/// sigma. Context beyond the placement radius, so an emitter placed at its
/// edge keeps its support; also the edge-diagnostic distance.
pub const SUPPORT: f64 = 3.0;
/// Safety cap on additions per group and add round.
pub const K_MAX: usize = 12;
/// px. Side of the median background window.
pub const BG_WIN: usize = 25;
/// ADU. `W = 1/m` is singular at `m = 0`; this is far below one count.
pub const BG_FLOOR: f64 = 1e-3;
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
    /// Fitted background at each emitter's nearest pixel, excluding
    /// neighbours: the nodes plus out-of-focus light.
    pub fitted_background: Vec<f64>,
    /// Scalar dispersion: pixel variance per unit of signal, ADU.
    pub dispersion: f64,
    /// `H*W`, ADU above the offset: the nodes plus out-of-focus light.
    pub background: Vec<f64>,
    /// The score threshold used.
    pub u: f64,
    pub n_seeds: usize,
    pub fits: usize,
    /// Additions that passed the score test but failed the LR confirmation.
    pub lr_fail: usize,
    /// Joint model: additions and removals after convergence, outer rounds,
    /// and the empirical-null scale of the last add round.
    pub adds: usize,
    pub removed: usize,
    pub outer: usize,
    pub kappa: f64,
    /// Out-of-focus components fitted and returned as background.
    pub out_of_focus: usize,
}

/// px. The widest component the model fits: half the node spacing
/// ([`model::tile`]), where the nodes carry most of a blob's light wherever
/// it sits. Out-of-focus light narrower than this would otherwise have to
/// be explained by emitters.
pub fn widest(s: &Settings) -> f64 {
    0.5 * model::tile(s) as f64
}

/// The seed templates' widths: geometric from `slack.0 * sigma` to
/// [`widest`], adjacent ones at most [`WIDTH_STEP`] apart.
pub fn widths(s: &Settings) -> Vec<f64> {
    let (lo, hi) = (s.slack.0 * s.sigma, widest(s));
    let n = ((hi / lo).ln() / WIDTH_STEP.ln() - 1e-9).ceil().max(0.0) as usize + 1;
    (0..n).map(|j| if n == 1 { lo } else { lo * (hi / lo).powf(j as f64 / (n - 1) as f64) }).collect()
}

/// Score threshold u giving `fp_per_mpx` expected noise false emitters.
/// An emitter survives on noise when its likelihood ratio, maximized over
/// position and width with the background nodes profiled out, reaches
/// `u^2 / 2`: a maximum above u of that ratio's signed root, a Gaussian
/// field over position and log width. Their expected number is the field's
/// expected Euler characteristic ([`statistics::lkc`]), so the count is at
/// most `fp_per_mpx`. Solved on `u >= 1`, where the rate decreases, by
/// bisection.
pub fn threshold(s: &Settings) -> f64 {
    let l = statistics::lkc(s.slack.0 * s.sigma, widest(s), Some(model::tile(s)));
    let fp_per_mpx = s.fp_per_mpx;
    let rate = |u: f64| 1e6 * statistics::expected_ec(u, &l);
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

pub(crate) use crate::prefilter::median;

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

pub use crate::prefilter::dispersion;

pub(crate) use crate::prefilter::psf_kernel1d;

/// Frame-wide score z of one emitter of width `sigma` against the bilinear
/// background on nodes `tile` px apart: the correlation of `r` with the
/// profile `g`, over `sqrt(var |g - P g|^2)`, `P` the projection on the
/// nodes' tents. `r` must be orthogonal to the tents (the data less its
/// node fit, or a fitted model's residual): then `<g, r> = <g - P g, d>`,
/// the signed root of the likelihood ratio with the nodes profiled out, the
/// statistic the count decisions test, taking `var` as constant across the
/// profile. No estimate of the background enters its mean. The frame has
/// no pixels beyond its edge.
pub fn score_map(r: &[f64], var: &[f64], h: usize, w: usize, sigma: f64, tile: usize) -> Vec<f64> {
    let k1 = psf_kernel1d(sigma);
    let c = model::blur(r, h, w, &k1, (0, h, 0, w));
    let (ny, py) = axis_norms(&k1, h, tile);
    let (nx, px) = axis_norms(&k1, w, tile);
    (0..h * w)
        .map(|i| {
            let (y, x) = (i / w, i % w);
            let n = (ny[y] * nx[x] - py[y] * px[x]).max(1e-12);
            c[i] / (var[i] * n).max(1e-12).sqrt()
        })
        .collect()
}

/// Along an axis of `n` pixels, for the profile `k1` centred on each pixel
/// and cut at the edges: `|e|^2` and `e^T P e`, `P` the projection on the
/// tents of nodes `tile` apart ([`model::Nodes`]).
fn axis_norms(k1: &[f64], n: usize, tile: usize) -> (Vec<f64>, Vec<f64>) {
    let rad = k1.len() / 2;
    let nn = (n.max(1) - 1).div_ceil(tile) + 1;
    let tent = |j: usize, t: usize| if nn == 1 { 1.0 } else { (1.0 - (t as f64 / tile as f64 - j as f64).abs()).max(0.0) };
    let mut gram = vec![0.0; nn * nn];
    for t in 0..n {
        let j = (t / tile).min(nn.saturating_sub(2));
        for a in j..(j + 2).min(nn) {
            for b in j..(j + 2).min(nn) {
                gram[a * nn + b] += tent(a, t) * tent(b, t);
            }
        }
    }
    let mut chol = Chol::new(nn);
    let ok = chol.factor(&gram, nn);
    let (mut norm, mut proj) = (vec![0.0; n], vec![0.0; n]);
    let (mut tv, mut x) = (vec![0.0; nn], vec![0.0; nn]);
    for ctr in 0..n {
        tv.fill(0.0);
        let (t0, t1) = (ctr.saturating_sub(rad), (ctr + rad + 1).min(n));
        for t in t0..t1 {
            let v = k1[t + rad - ctr];
            norm[ctr] += v * v;
            let j = (t / tile).min(nn.saturating_sub(2));
            for a in j..(j + 2).min(nn) {
                tv[a] += tent(a, t) * v;
            }
        }
        if ok {
            chol.solve(&tv, &mut x);
            proj[ctr] = tv.iter().zip(&x).map(|(a, b)| a * b).sum();
        }
    }
    (norm, proj)
}

/// Fraction of its continuous maximum the score keeps where the template
/// grid samples it worst: half a pixel off in y and x, which to second
/// order loses `Lambda / 4` of it (`Lambda = 1 / (2 s^2)` the field's
/// curvature, `s^2` the profile variance), and midway between two widths
/// [`WIDTH_STEP`] apart. Proposals are taken at `u` times this, so none a
/// test at `u` would keep is missed; the likelihood ratio decides.
pub fn proposal_factor(s: f64) -> f64 {
    let lambda = 0.5 / (s * s + 1.0 / 12.0);
    (1.0 - 0.25 * lambda) * 2.0 * WIDTH_STEP.sqrt() / (1.0 + WIDTH_STEP)
}

/// Local maxima over position and width of the score maps `z` (one per
/// width of `widths`) with z above `u` times [`proposal_factor`], as
/// `(pixel, width index)`, strongest first. A maximum at width `j` tops its `2 ceil(s_j) + 1` window and the
/// adjacent widths' windows at the same pixel.
pub fn find_seeds(z: &[Vec<f64>], widths: &[f64], h: usize, w: usize, u: f64) -> Vec<(usize, usize)> {
    let mx: Vec<Vec<f64>> = z
        .iter()
        .zip(widths)
        .map(|(zj, s)| filters::maximum_filter(zj, h, w, 2 * s.ceil() as usize + 1, Mode::Reflect))
        .collect();
    let n = z.len();
    let mut seeds: Vec<(usize, usize)> = (0..n)
        .flat_map(|j| (0..h * w).map(move |i| (i, j)))
        .filter(|&(i, j)| {
            let v = z[j][i];
            v > u * proposal_factor(widths[j]) && v == mx[j][i] && (j == 0 || v >= mx[j - 1][i]) && (j + 1 == n || v >= mx[j + 1][i])
        })
        .collect();
    seeds.sort_by(|a, b| z[b.1][b.0].total_cmp(&z[a.1][a.0]));
    seeds
}

/// Whether the emitter's support crosses a physical frame edge.
fn edge_truncated(y: f64, x: f64, sigma: f64, h: usize, w: usize) -> bool {
    let border = (y + 0.5).min(h as f64 - 0.5 - y)
        .min(x + 0.5).min(w as f64 - 0.5 - x);
    border < SUPPORT * sigma
}

/// Whether `value` sits on a bound of `[lo, hi]`, allowing for the fit's
/// interior margin.
pub(crate) fn at_bound(value: f64, lo: f64, hi: f64) -> bool {
    let margin = 2.0 * model::INTERIOR_FRAC * (hi - lo).max(1e-12);
    value - lo <= margin + BOUND_TOL * (1.0 + lo.abs()) || hi - value <= margin + BOUND_TOL * (1.0 + hi.abs())
}

/// Context needed around an ROI: an add-round box around any seed
/// (`(OWN + SUPPORT) sigma`), plus the median window and the widest score
/// kernel that seed's background and z read.
fn crop_margin(s: &Settings) -> usize {
    let wide = widest(s);
    let kernel = (4.0 * wide).ceil() as usize + wide.ceil() as usize;
    let pad = ((OWN + SUPPORT) * s.sigma).ceil() as usize;
    pad.max(kernel) + 1 + BG_WIN / 2 + 2
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
fn roi_crop(roi: &[bool], h: usize, w: usize, s: &Settings) -> Option<Rect> {
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
    let m = crop_margin(s);
    let side = 3 * BG_WIN;
    let (y0, y1) = widen(y0.saturating_sub(m), (y1 + m).min(h), side, h);
    let (x0, x1) = widen(x0.saturating_sub(m), (x1 + m).min(w), side, w);
    Some(Rect { r0: y0, r1: y1, c0: x0, c1: x1 })
}

fn crop<T: Copy>(v: &[T], w: usize, rc: &Rect) -> Vec<T> {
    let mut out = Vec::with_capacity((rc.r1 - rc.r0) * (rc.c1 - rc.c0));
    for r in rc.r0..rc.r1 {
        out.extend_from_slice(&v[r * w + rc.c0..r * w + rc.c1]);
    }
    out
}

/// The joint model's starting point on a frame (or crop) `d`.
#[derive(Clone, Debug, Default)]
pub struct Start {
    /// One reference-width emitter per seed, flux from the seed pixel's
    /// excess over the background.
    pub ems: Vec<Em>,
    pub background: Vec<f64>,
    pub dispersion: f64,
    pub u: f64,
}

/// Seeds as emitters: local maxima over position and width of the score
/// ([`score_map`]) with the nodes profiled out, above `u` times
/// [`proposal_factor`], strongest first, each at its template's width. `roi` (same shape as `d`) limits seeds. Seeds are
/// proposals; the joint model's removal test decides them.
pub fn start(d: &[f64], h: usize, w: usize, roi: Option<&[bool]>, s: &Settings) -> Start {
    assert_eq!(d.len(), h * w);
    let u = threshold(s);
    let bmap = median_background(d, h, w);
    let phi = dispersion(d, h, w);
    // The score's mean needs the data less its projection on the tents; its
    // variance, the background level, from the nodes fitted to the median,
    // which the emitters barely move.
    let tile = model::tile(s);
    let mut nodes = model::Nodes::new(h, w, tile);
    nodes.fit_map(d);
    let resid: Vec<f64> = d.iter().zip(&nodes.surface()).map(|(a, b)| a - b).collect();
    nodes.fit_map(&bmap);
    let var: Vec<f64> = nodes.surface().iter().map(|b| phi * b.max(BG_FLOOR)).collect();
    let ws = widths(s);
    let z: Vec<Vec<f64>> = ws.iter().map(|&sj| score_map(&resid, &var, h, w, sj, tile)).collect();
    let ems = find_seeds(&z, &ws, h, w, u)
        .into_iter()
        .filter(|&(i, _)| roi.is_none_or(|m| m[i]))
        .map(|(i, j)| Em { a: (resid[i] / psf::peak_factor(ws[j])).max(1.0), y: (i / w) as f64, x: (i % w) as f64, s: ws[j] })
        .collect();
    Start { ems, background: bmap, dispersion: phi, u }
}

/// Localize an offset-subtracted frame in ADU: [`start`], then the joint
/// model, then uncertainties. An ROI limits seeds and additions; everything
/// runs on its bounding box plus enough context for every filter and group.
/// Returns global coordinates and a full-frame background map (the fitted
/// node surface). An empty ROI returns no detections.
pub fn localize(d: &[f64], h: usize, w: usize, roi: Option<&[bool]>, s: &Settings) -> Output {
    assert_eq!(d.len(), h * w);
    let bb = match roi {
        None => Rect::frame(h, w),
        Some(m) => match roi_crop(m, h, w, s) {
            Some(bb) => bb,
            None => {
                return Output {
                    background: vec![BG_FLOOR; h * w],
                    dispersion: f64::NAN,
                    u: threshold(s),
                    kappa: 1.0,
                    ..Output::default()
                }
            }
        },
    };
    let (ch, cw) = (bb.r1 - bb.r0, bb.c1 - bb.c0);
    let whole = ch == h && cw == w;
    let (dsub, rsub) = if whole { (Vec::new(), None) } else { (crop(d, w, &bb), roi.map(|m| crop(m, w, &bb))) };
    let dc: &[f64] = if whole { d } else { &dsub };
    let rc: Option<&[bool]> = if whole { roi } else { rsub.as_deref() };

    let first = start(dc, ch, cw, rc, s);
    let mut md = Model::new(dc, ch, cw, &first.ems, &first.background, first.dispersion, first.u, s, rc);
    md.run();
    let bg = md.background();
    let mut out = report(&md, &bg, bb.r0, bb.c0, h, w);
    out.u = first.u;
    out.n_seeds = first.ems.len();
    // Outside the crop nothing was estimated: the map carries a fill there.
    out.background = if whole {
        bg
    } else {
        let mut full = vec![median(&bg); h * w];
        for r in 0..ch {
            full[(bb.r0 + r) * w + bb.c0..(bb.r0 + r) * w + bb.c1].copy_from_slice(&bg[r * cw..(r + 1) * cw]);
        }
        full
    };
    out
}

/// Output rows for the model's in-focus emitters, in its order, shifted by
/// `(oy, ox)` into an `h x w` frame; `bg` is [`Model::background`]. SEs
/// come from each final group's Fisher information with the background
/// nodes profiled out and every out-of-focus component free.
fn report(md: &Model, bg: &[f64], oy: usize, ox: usize, h: usize, w: usize) -> Output {
    let n = md.ems.len();
    let mut out = Output {
        dispersion: md.phi,
        fits: md.stats.fits,
        lr_fail: md.stats.lr_fail,
        adds: md.stats.adds,
        removed: md.stats.removed,
        outer: md.stats.outer,
        kappa: md.stats.kappa,
        ..Output::default()
    };
    let (mut se4, mut ff4) = (vec![f64::NAN; 4 * n], vec![f64::NAN; 4 * n]);
    for (g, fisher) in md.group_information() {
        let p = fisher.len().isqrt();
        let mut var = vec![f64::NAN; p];
        let mut chol = Chol::new(p);
        if chol.factor(&fisher, p) {
            chol.inv_diag(&mut var, &mut Vec::new());
        }
        let idx: Vec<u32> = g.iter().map(|&i| i as u32).collect();
        store_uncertainties(&fisher, &var, &idx, &mut se4, &mut ff4);
    }
    let scale = md.phi.sqrt();
    for (i, em) in md.ems.iter().enumerate() {
        let e = em.e;
        if !md.in_focus(&e) {
            out.out_of_focus += 1;
            continue;
        }
        let (gy, gx) = (e.y + oy as f64, e.x + ox as f64);
        out.pos.extend_from_slice(&[gy, gx]);
        out.amp.push(e.a);
        out.sig.push(e.s);
        out.se.extend((0..3).map(|c| se4[4 * i + c] * scale));
        out.se_sig.push(se4[4 * i + 3] * scale);
        out.fisher_fraction.extend_from_slice(&ff4[4 * i..4 * i + 4]);
        let mut flag = em.flags;
        if edge_truncated(gy, gx, e.s, h, w) {
            flag |= FLAG_EDGE;
        }
        if se4[4 * i..4 * i + 4].iter().any(|v| !v.is_finite() || *v <= 0.0) {
            flag |= FLAG_COVARIANCE;
        }
        out.flags.push(flag);
        let py = (e.y.round().max(0.0) as usize).min(md.h - 1);
        let px = (e.x.round().max(0.0) as usize).min(md.w - 1);
        out.fitted_background.push(bg[py * md.w + px]);
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
            let q = 4 * j + c;
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

/// Localize one raw frame using `d = raw - offset` in ADU.
pub fn localize_raw(raw: &[f64], h: usize, w: usize, offset: f64, roi: Option<&[bool]>, s: &Settings) -> Output {
    let d: Vec<f64> = raw.iter().map(|&r| r - offset).collect();
    localize(&d, h, w, roi, s)
}

/// Localize every frame of a stack on `n_threads` workers, in frame order.
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
    crate::frames::map(n, n_threads, || (), |t, _| localize_raw(&raw[t * h * w..(t + 1) * h * w], h, w, offset, roi, s))
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
    fn the_threshold_solves_the_expected_rate() {
        for (sigma, fp) in [(1.0, 2.0), (1.45, 16.0), (2.0, 100.0)] {
            let s = Settings { sigma, fp_per_mpx: fp, slack: SLACK };
            let u = threshold(&s);
            let l = statistics::lkc(SLACK.0 * sigma, widest(&s), Some(model::tile(&s)));
            let rate = 1e6 * statistics::expected_ec(u, &l);
            assert!((rate / fp - 1.0).abs() < 1e-9, "sigma {sigma}: rate {rate}");
        }
    }

    #[test]
    fn the_width_bank_spans_the_bounds_in_steps_no_wider_than_allowed() {
        let s = settings(1.2);
        let ws = widths(&s);
        assert!((ws[0] - SLACK.0 * 1.2).abs() < 1e-12 && (ws[ws.len() - 1] - widest(&s)).abs() < 1e-12);
        assert!(ws.windows(2).all(|p| p[1] / p[0] <= WIDTH_STEP + 1e-12));
        // Out-of-focus widths run to half the node spacing.
        assert!(widest(&s) >= 4.0 * SLACK.1 * 1.2);
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
        let p = 8;
        let rho = 0.99;
        for order in [[0u32, 1], [1, 0]] {
            for scale in [[1.0; 8], [1e-3, 2.0, 3.0, 4.0, 1e3, 5.0, 6.0, 7.0]] {
                let mut f = vec![0.0; p * p];
                for q in 0..p { f[q * p + q] = scale[q] * scale[q]; }
                f[4] = rho * scale[0] * scale[4];
                f[4 * p] = f[4];
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
        let o = localize(&d, h, w, None, &settings(sigma));
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
