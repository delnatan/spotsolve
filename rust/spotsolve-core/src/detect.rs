//! Spot detection in ADU above the camera offset, after u-track's
//! `pointSourceDetection` (Aguet et al. 2013, *Dev. Cell* 26:279).
//!
//! 1. Dispersion `phi` from the frame ([`prefilter::dispersion`]); the
//!    threshold `u` from `fp_per_mpx` ([`prefilter::threshold`]).
//! 2. Seeds: LoG maxima where the Poisson score of one emitter reaches the
//!    least a real emitter at `u` can show on the pixel grid
//!    ([`seed_factor`]), with the mask's labels ([`prefilter::screen`]).
//! 3. Each seed is fitted alone on its window ([`fit`]): a constant level
//!    and one emitter of the in-focus width, its centre held within
//!    [`CONFINE`] widths of the seed. Pixels of other mask components are
//!    left out.
//! 4. An emitter is kept if its likelihood ratio against the level alone
//!    reaches the threshold, `2 (I0 - I1) / phi >= u^2`, and its centre is
//!    not held at the confinement bound. The test is made at the in-focus
//!    width, where `u` is calibrated; with `free_sigma` a kept emitter is
//!    then refitted with its width free, and that fit is reported.
//! 5. Fits of one emitter from several seeds are merged ([`DUPLICATE`]).
//! 6. Standard errors from the reported fit's Fisher information, scaled by
//!    `phi`.

use crate::fit::{Fitter, Layout, Window};
use crate::prefilter::{self, Screen};
use crate::psf;

/// Default expected false emitters per 10^6 noise pixels.
pub const FP_PER_MPX: f64 = 16.0;
/// Widths: the fit window's half-side, `ceil(WINDOW * sigma)` px.
pub const WINDOW: f64 = 4.0;
/// Widths: how far an emitter's centre may move from its seed. Farther, the
/// fit is explaining light another seed was proposed for.
pub const CONFINE: f64 = 2.0;
/// px: kept emitters closer than this are fits of one emitter from two
/// seeds; the one with the larger likelihood ratio stays.
pub const DUPLICATE: f64 = 0.25;
/// Nats: a fit stops once no parameter can gain more by moving one
/// standard error, far below the `u^2 / 2` the decisions turn on.
pub const FIT_TOL: f64 = 1e-6;
pub const FIT_MAX_ITER: usize = 100;
/// Widths: free-width bounds are `[WIDTH_LO * sigma, R / 2]`, `R` the
/// window's half-side: wider, the window holds too little of the emitter to
/// tell it from the level. The bounds keep the fit finite; an emitter on
/// one is flagged.
pub const WIDTH_LO: f64 = 0.5;

pub const FLAG_EDGE: u8 = 1;
pub const FLAG_NOT_CONVERGED: u8 = 2;
pub const FLAG_STALLED: u8 = 4;
pub const FLAG_COVARIANCE: u8 = 8;
pub const FLAG_BOUND: u8 = 16;

/// Widths from a frame edge within which an emitter is flagged
/// [`FLAG_EDGE`]: part of its light falls outside the frame.
pub const EDGE: f64 = 3.0;

/// What a caller chooses per frame.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// In-focus PSF width, px.
    pub sigma: f64,
    /// Expected false emitters per 10^6 noise pixels; sets `u`.
    pub fp_per_mpx: f64,
    /// Report each emitter's fitted width; otherwise it is `sigma`.
    pub free_sigma: bool,
}

impl Settings {
    pub fn new(sigma: f64) -> Self {
        Self { sigma, fp_per_mpx: FP_PER_MPX, free_sigma: false }
    }

    fn radius(&self) -> usize {
        (WINDOW * self.sigma).ceil() as usize
    }
}

/// All emitters with diagnostics. Positions are global `(y, x)`.
#[derive(Clone, Debug, Default)]
pub struct Output {
    /// `2N`, row-major `(y, x)`.
    pub pos: Vec<f64>,
    /// `N`: flux above the level, ADU.
    pub amp: Vec<f64>,
    pub sig: Vec<f64>,
    /// `3N`: SE of `(flux, y, x)`; NaN without a covariance.
    pub se: Vec<f64>,
    /// `N`: SE of the fitted width; NaN when the width is fixed.
    pub se_sig: Vec<f64>,
    /// `N`: bitwise combination of `FLAG_*` diagnostics.
    pub flags: Vec<u8>,
    /// `N`: the signed root of each emitter's likelihood ratio at the
    /// in-focus width, `sqrt(2 (I0 - I1) / phi)`; at least `u`.
    pub z: Vec<f64>,
    /// `N`: the fitted level of each emitter's window.
    pub fitted_background: Vec<f64>,
    /// Scalar dispersion: pixel variance per unit of signal.
    pub dispersion: f64,
    /// `H*W`: the screening level (the window regression's constant); NaN
    /// outside the processed crop.
    pub background: Vec<f64>,
    pub u: f64,
    pub n_seeds: usize,
    /// Seeds whose emitter fell short of `u`, or left its confinement.
    pub weak: usize,
    pub unconfined: usize,
    /// Emitters merged into a stronger fit of the same light.
    pub duplicates: usize,
}

/// The least fraction of an emitter's peak score the pixel grid can lose:
/// the score field's correlation half a pixel off in y and in x. Its
/// covariance is the PSF correlated with itself, a Gaussian of variance
/// `2 (sigma^2 + 1/12)` per axis, so the correlation at offset `r` is
/// `exp(-r^2 / (4 (sigma^2 + 1/12)))`. Seeds at `u` times this are the
/// sampled maxima of the continuous field above `u`, the count
/// [`prefilter::threshold`] calibrates. A lower bar admits more maxima of
/// the refitted likelihood ratio than the field has, and with them false
/// emitters; it gains no power that a higher `fp_per_mpx` does not.
pub fn seed_factor(sigma: f64) -> f64 {
    (-0.5 / (4.0 * (sigma * sigma + 1.0 / 12.0))).exp()
}

/// One emitter as fitted, in crop coordinates.
#[derive(Clone, Copy, Debug)]
struct Spot {
    y: f64,
    x: f64,
    a: f64,
    s: f64,
    se: [f64; 3],
    se_s: f64,
    c: f64,
    z: f64,
    flags: u8,
}

enum Fate {
    Kept(Spot),
    Weak,
    Unconfined,
}

/// `I(d, c)` of the used pixels for the level alone, at its maximum
/// likelihood `c = mean(d)`.
fn level_divergence(win: &Window) -> f64 {
    let (n, sum) = win.d.iter().zip(&win.used).filter(|p| *p.1).fold((0.0, 0.0), |(n, s), (d, _)| (n + 1.0, s + d));
    let c = (sum / n).max(prefilter::LEVEL_FLOOR);
    win.d
        .iter()
        .zip(&win.used)
        .filter(|p| *p.1)
        .map(|(&d, _)| if d > 0.0 { d * (d / c).ln() } else { 0.0 } - (d - c))
        .sum()
}

#[allow(clippy::too_many_arguments)]
fn fit_seed(d: &[f64], h: usize, w: usize, sc: &Screen, seed: usize, s: &Settings, phi: f64, u: f64, fitter: &mut Fitter) -> Fate {
    let (sy, sx) = (seed / w, seed % w);
    let r = s.radius();
    let (y0, y1) = (sy.saturating_sub(r), (sy + r + 1).min(h));
    let (x0, x1) = (sx.saturating_sub(r), (sx + r + 1).min(w));
    let (rows, cols) = (y1 - y0, x1 - x0);
    let own = sc.labels[seed];
    let (mut wd, mut used) = (Vec::with_capacity(rows * cols), Vec::with_capacity(rows * cols));
    for y in y0..y1 {
        for x in x0..x1 {
            let i = y * w + x;
            wd.push(d[i]);
            used.push(sc.labels[i] == 0 || sc.labels[i] == own);
        }
    }
    let win = Window::new(rows, cols, wd, used);
    let dmax = win.d.iter().zip(&win.used).filter(|p| *p.1).fold(1.0f64, |m, (&v, _)| m.max(v));
    let (cy, cx) = ((sy - y0) as f64, (sx - x0) as f64);
    let reach = CONFINE * s.sigma;
    let fixed = Layout { k: 1, sigma: Some(s.sigma) };
    let lo = [prefilter::LEVEL_FLOOR, 0.0, cy - reach, cx - reach];
    let hi = [2.0 * dmax, 10.0 * dmax / psf::peak_factor(s.sigma), cy + reach, cx + reach];
    let start = [sc.background[seed].clamp(lo[0], hi[0]), sc.amplitude[seed].clamp(1.0, hi[1]), cy, cx];
    let tol = FIT_TOL * phi;
    let one = fitter.fit(&win, &start, fixed, &lo, &hi, FIT_MAX_ITER, tol);
    let stat = 2.0 * (level_divergence(&win) - one.divergence) / phi;
    if !(stat >= u * u) {
        return Fate::Weak;
    }
    if one.at_bound[2] || one.at_bound[3] {
        return Fate::Unconfined;
    }
    let (lay, rep) = if s.free_sigma {
        let free = Layout { k: 1, sigma: None };
        let lo = [lo[0], lo[1], lo[2], lo[3], WIDTH_LO * s.sigma];
        let hi = [hi[0], hi[1], hi[2], hi[3], 0.5 * r as f64];
        let mut start = one.theta.clone();
        start.push(s.sigma);
        (free, fitter.fit(&win, &start, free, &lo, &hi, FIT_MAX_ITER, tol))
    } else {
        (fixed, one)
    };
    let mut flags = 0;
    if !rep.converged {
        flags |= FLAG_NOT_CONVERGED;
    }
    if rep.stalled {
        flags |= FLAG_STALLED;
    }
    if rep.at_bound.iter().any(|&b| b) {
        flags |= FLAG_BOUND;
    }
    let n = lay.n();
    let mut se = [f64::NAN; 3];
    let mut se_s = f64::NAN;
    match fitter.covariance(&win, &rep.theta, lay, phi) {
        Some(cov) if (0..n).all(|q| cov[q * n + q] > 0.0) => {
            for (k, q) in [1, 2, 3].into_iter().enumerate() {
                se[k] = cov[q * n + q].sqrt();
            }
            if lay.sigma.is_none() {
                se_s = cov[n * n - 1].sqrt();
            }
        }
        _ => flags |= FLAG_COVARIANCE,
    }
    let t = &rep.theta;
    Fate::Kept(Spot {
        y: t[2] + y0 as f64,
        x: t[3] + x0 as f64,
        a: t[1],
        s: lay.sigma(t),
        se,
        se_s,
        c: t[0],
        z: stat.sqrt(),
        flags,
    })
}

/// Keep the strongest of each set of spots closer than [`DUPLICATE`].
fn merge(mut spots: Vec<Spot>) -> (Vec<Spot>, usize) {
    spots.sort_by(|a, b| b.z.total_cmp(&a.z));
    let mut kept: Vec<Spot> = Vec::with_capacity(spots.len());
    for p in &spots {
        if kept.iter().all(|q| (p.y - q.y).hypot(p.x - q.x) >= DUPLICATE) {
            kept.push(*p);
        }
    }
    let dropped = spots.len() - kept.len();
    kept.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));
    (kept, dropped)
}

/// The ROI's bounding box grown by the context its seeds need: a fit
/// window and the screening window around each of its pixels, and the LoG
/// and its maximum filter. `None` when the ROI selects no pixel.
fn roi_crop(roi: &[bool], h: usize, w: usize, s: &Settings) -> Option<[usize; 4]> {
    let (mut y0, mut y1, mut x0, mut x1) = (h, 0, w, 0);
    for (i, _) in roi.iter().enumerate().filter(|p| *p.1) {
        y0 = y0.min(i / w);
        y1 = y1.max(i / w + 1);
        x0 = x0.min(i % w);
        x1 = x1.max(i % w + 1);
    }
    if y0 >= y1 {
        return None;
    }
    let m = (2 * s.radius()).max(crate::filters::kernel_radius(s.sigma) + s.sigma.ceil() as usize);
    Some([y0.saturating_sub(m), (y1 + m).min(h), x0.saturating_sub(m), (x1 + m).min(w)])
}

/// Localize an offset-subtracted frame in ADU. An ROI limits seeds; the
/// work runs on its bounding box plus the context [`roi_crop`] gives.
pub fn localize(d: &[f64], h: usize, w: usize, roi: Option<&[bool]>, s: &Settings) -> Output {
    localize_with(d, h, w, roi, s, &mut Fitter::default())
}

fn localize_with(d: &[f64], h: usize, w: usize, roi: Option<&[bool]>, s: &Settings, fitter: &mut Fitter) -> Output {
    assert_eq!(d.len(), h * w);
    let u = prefilter::threshold(s.fp_per_mpx, s.sigma);
    let [r0, r1, c0, c1] = match roi {
        None => [0, h, 0, w],
        Some(m) => match roi_crop(m, h, w, s) {
            Some(b) => b,
            None => return Output { background: vec![f64::NAN; h * w], dispersion: f64::NAN, u, ..Output::default() },
        },
    };
    let (ch, cw) = (r1 - r0, c1 - c0);
    let cut = |v: &[f64]| -> Vec<f64> { (r0..r1).flat_map(|r| v[r * w + c0..r * w + c1].iter().copied()).collect() };
    let dc = cut(d);
    let rc: Option<Vec<bool>> = roi.map(|m| (r0..r1).flat_map(|r| m[r * w + c0..r * w + c1].iter().copied()).collect());
    let phi = prefilter::dispersion(&dc, ch, cw);
    let sc = prefilter::screen(&dc, ch, cw, s.sigma, phi, u * seed_factor(s.sigma), rc.as_deref());
    let mut out = Output { dispersion: phi, u, n_seeds: sc.seeds.len(), ..Output::default() };
    let mut spots = Vec::new();
    for &seed in &sc.seeds {
        match fit_seed(&dc, ch, cw, &sc, seed, s, phi, u, fitter) {
            Fate::Kept(p) => spots.push(p),
            Fate::Weak => out.weak += 1,
            Fate::Unconfined => out.unconfined += 1,
        }
    }
    let (spots, dropped) = merge(spots);
    out.duplicates = dropped;
    for p in spots {
        let (gy, gx) = (p.y + r0 as f64, p.x + c0 as f64);
        let border = (gy + 0.5).min(h as f64 - 0.5 - gy).min(gx + 0.5).min(w as f64 - 0.5 - gx);
        out.pos.extend([gy, gx]);
        out.amp.push(p.a);
        out.sig.push(p.s);
        out.se.extend(p.se);
        out.se_sig.push(p.se_s);
        out.flags.push(p.flags | if border < EDGE * p.s { FLAG_EDGE } else { 0 });
        out.z.push(p.z);
        out.fitted_background.push(p.c);
    }
    out.background = vec![f64::NAN; h * w];
    for r in 0..ch {
        out.background[(r0 + r) * w + c0..(r0 + r) * w + c1].copy_from_slice(&sc.background[r * cw..(r + 1) * cw]);
    }
    out
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
    crate::frames::map(n, n_threads, Fitter::default, |t, fitter| {
        let d: Vec<f64> = raw[t * h * w..(t + 1) * h * w].iter().map(|&r| r - offset).collect();
        localize_with(&d, h, w, roi, s, fitter)
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Deterministic standard normals: an LCG through Box-Muller.
    pub(crate) fn normals(n: usize, mut state: u64) -> Vec<f64> {
        let mut uni = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n).map(|_| (-2.0 * uni().ln()).sqrt() * (2.0 * std::f64::consts::PI * uni()).cos()).collect()
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
        let o = localize(&d, h, w, None, &Settings { free_sigma: true, ..Settings::new(sigma) });
        assert_eq!(o.amp.len(), 1, "found {:?} amp {:?}", o.pos, o.amp);
        assert_eq!(o.flags[0], 0);
        assert!(o.se.iter().chain(&o.se_sig).all(|v| v.is_finite() && *v > 0.0));
        assert!((o.dispersion - 1.0).abs() < 0.3, "dispersion {}", o.dispersion);
        assert!((o.pos[0] - 19.3).abs() < 3.0 * o.se[1], "y {}", o.pos[0]);
        assert!((o.pos[1] - 21.6).abs() < 3.0 * o.se[2], "x {}", o.pos[1]);
        assert!((o.amp[0] - 1500.0).abs() < 3.0 * o.se[0], "A {}", o.amp[0]);
        assert!((o.sig[0] - 1.3).abs() < 3.0 * o.se_sig[0], "sigma {}", o.sig[0]);
        assert!(o.z[0] >= o.u);
    }
}
