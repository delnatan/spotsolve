//! Spot detection in ADU above the camera offset, after u-track's
//! `pointSourceDetection` (Aguet et al. 2013, *Dev. Cell* 26:279).
//!
//! 1. Dispersion `phi` from the frame ([`prefilter::dispersion`]); the
//!    threshold `u` from `fp_per_mpx` ([`prefilter::threshold`]).
//! 2. Seeds: LoG maxima where the Poisson score of one emitter, at some
//!    width of a small bank, reaches the least a real emitter at `u` can
//!    show on that grid of positions and widths ([`bank`]), with the mask's
//!    labels ([`prefilter::screen`]).
//! 3. Each seed's window ([`crate::fit`]), `ceil(4 sigma)` px around it, holds a
//!    constant level and pixel-integrated Gaussian components; pixels of
//!    other mask components are left out. A component is kept if adding it
//!    gains `u^2 / 2` nats, `2 (I_k - I_k+1) / phi >= u^2`, where `u` is
//!    calibrated.
//!    - Alone (u-track's default), one component starts at the seed and its
//!      centre is held within [`CONFINE`] widths of it.
//!    - With mixtures (u-track's `FitMixtures`), components are added where
//!      the efficient score peaks ([`propose`]) while each refit gains as
//!      much, then the weakest is removed while removing it, the rest
//!      refitted, costs less. The seed only centres the window: every
//!      source of light in it takes a component, so a neighbour is a
//!      nuisance parameter rather than a bias. u-track instead confines
//!      components to 2 sigma, and the light of a neighbour farther off
//!      then drags the fit.
//! 4. Each component is reported by the fit of the seed nearest to it
//!    ([`owned`]), so fits from neighbouring seeds report no emitter twice
//!    (u-track merges copies within a fixed radius instead), and only those
//!    are tested for removal. A component held at a position bound is not
//!    reported.
//! 5. Every component has its own width, so a wider emitter is one
//!    component rather than two. Emitters are reported in `width * sigma`
//!    (by default [`WIDTH`]); the tests search width as well as
//!    position, and `u` is set for that search ([`prefilter::false_rate`]).
//!    A component may widen past the band to the window's half-side as
//!    out-of-focus light (defocused emitters, haze), which narrower
//!    components would otherwise split up; it is counted, not reported.
//!    Equal bounds fix the width, as u-track does.
//! 6. Standard errors from the reported fit's Fisher information, scaled by
//!    `phi`.

use crate::fit::{Fitter, Layout, Window};
use crate::prefilter::{self, Screen};
use crate::psf;

/// Ratio of neighbouring widths in the seed [`bank`]: a maximum midway
/// between two keeps 98% of its score.
pub const BANK_STEP: f64 = 1.5;
/// Default expected false emitters per 10^6 noise pixels.
pub const FP_PER_MPX: f64 = 16.0;
/// Default reported widths, multiples of `sigma`. An in-focus emitter fits
/// near `sigma`; defocus widens it, and by `sqrt(2) sigma` its peak has
/// halved, the edge of the PSF's axial FWHM. The bound sits a little above
/// that so a `sigma` set slightly narrow keeps in-focus emitters. A window
/// that sees only part of wider light (haze, a defocused blob) fits it with
/// components of `1.5-2 sigma`; above the bound they are out-of-focus light.
pub const WIDTH: (f64, f64) = (1.0, 1.5);
/// Widths: the fit window's half-side, `ceil(WINDOW * sigma)` px.
pub const WINDOW: f64 = 4.0;
/// Widths: how far an emitter's centre may move from its seed. Farther, the
/// fit is explaining light another seed was proposed for.
pub const CONFINE: f64 = 2.0;
/// Components per window at most, with mixtures: a bound on work, not a
/// statistical choice; components stop when the test stops them. A window
/// is about `(8 sigma)^2` px and emitters much closer than `sigma` are not
/// resolved, so about `(8 sigma)^2 / (pi sigma^2) = 20` can be told apart
/// in one.
pub const MAX_MIXTURES: usize = 20;
/// Nats: a fit stops once no parameter can lower the objective by more
/// alone. That leaves each within `sqrt(2 FIT_TOL) = 0.045` of its standard
/// error of the optimum, and each likelihood ratio far closer than the
/// `u^2 / 2` the decisions turn on.
pub const FIT_TOL: f64 = 1e-3;
pub const FIT_MAX_ITER: usize = 100;
/// px: the widest an emitter is reported, half the window's half-side `R`.
/// One centred in its window then keeps at least `erf(sqrt 2)^2 = 91%` of
/// its light inside. A component may widen to `R` itself: wider than this
/// it is out-of-focus light, fitted so that narrower components need not
/// explain it, and returned as background.
pub fn widest(sigma: f64) -> f64 {
    0.5 * (WINDOW * sigma).ceil()
}

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
    /// Widths a component may take, as multiples of `sigma`. Equal bounds
    /// fix every width, as u-track does; the upper one is capped at
    /// [`widest`].
    pub width: (f64, f64),
    /// Fit several components per window, after u-track's `FitMixtures`.
    pub fit_mixtures: bool,
    pub max_mixtures: usize,
}

impl Settings {
    pub fn new(sigma: f64) -> Self {
        Self { sigma, fp_per_mpx: FP_PER_MPX, width: WIDTH, fit_mixtures: false, max_mixtures: MAX_MIXTURES }
    }

    fn radius(&self) -> usize {
        (WINDOW * self.sigma).ceil() as usize
    }

    /// The width bounds in px.
    pub fn widths(&self) -> (f64, f64) {
        let hi = (self.width.1 * self.sigma).min(widest(self.sigma));
        ((self.width.0 * self.sigma).min(hi), hi)
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
    /// Components fitted from a seed other than the one nearest to them, and
    /// so reported by that one's fit instead.
    pub duplicates: usize,
    /// Mixture components added beyond each window's first, and removed again.
    pub added: usize,
    pub removed: usize,
    /// `N`: the window each emitter was fitted in, when it held several
    /// components (from 1); 0 for an emitter fitted alone.
    pub mixture: Vec<u32>,
    /// `2N`: the seed each emitter was reported from, global `(y, x)`.
    pub seed: Vec<f64>,
    /// `2S`: every seed, global `(y, x)`.
    pub seeds: Vec<f64>,
    /// Components wider than the reported widths, fitted as out-of-focus
    /// light and returned as background.
    pub out_of_focus: usize,
}

/// Correlation of the scores of two unit pixel-integrated profiles of
/// widths `a` and `b` at one centre: `2 sqrt(va vb) / (va + vb)`, `v = s^2 +
/// 1/12` their variances.
fn width_correlation(a: f64, b: f64) -> f64 {
    let (va, vb) = (a * a + 1.0 / 12.0, b * b + 1.0 / 12.0);
    2.0 * (va * vb).sqrt() / (va + vb)
}

/// The least fraction of an emitter's peak score the pixel grid can lose:
/// the score field's correlation half a pixel off in y and in x. Its
/// covariance is the profile correlated with itself, a Gaussian of
/// variance `2 (s^2 + 1/12)` per axis, so the correlation at offset `r` is
/// `exp(-r^2 / (4 (s^2 + 1/12)))`.
pub fn seed_factor(s: f64) -> f64 {
    (-0.5 / (4.0 * (s * s + 1.0 / 12.0))).exp()
}

/// Widths `[lo, hi]` sampled geometrically, neighbours at most [`BANK_STEP`]
/// apart, each with its bar: `u` less what the pixel grid and the gap to
/// the next width can lose of a maximum ([`seed_factor`],
/// [`width_correlation`] at the gap's geometric middle). The mask's seeds
/// are then the sampled maxima above `u` of the field the test searches,
/// the count [`prefilter::false_rate`] calibrates. A lower bar admits more
/// maxima of the refitted likelihood ratio than the field has, and with
/// them false emitters; it gains no power that a higher `fp_per_mpx` does
/// not.
pub fn bank(lo: f64, hi: f64, u: f64) -> Vec<(f64, f64)> {
    let n = ((hi / lo).ln() / BANK_STEP.ln()).ceil().max(0.0) as i32;
    let step = if n > 0 { (hi / lo).powf(1.0 / n as f64) } else { 1.0 };
    let gap = width_correlation(1.0, step.sqrt()).min(width_correlation(lo, lo * step.sqrt()));
    (0..=n).map(|i| lo * step.powi(i)).map(|s| (s, u * seed_factor(s) * gap)).collect()
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
    seed: usize,
    /// The window it was fitted in, from 1, when that window holds more
    /// than one component; 0 alone.
    mixture: u32,
}

/// What became of one seed's window.
#[derive(Default)]
struct Fate {
    spots: Vec<Spot>,
    weak: bool,
    unconfined: bool,
    /// Components added beyond the first, and removed again.
    added: usize,
    removed: usize,
    /// Components left to the fits of the seeds nearest them.
    others: usize,
    /// Owned components wider than the reported widths: background.
    out_of_focus: usize,
}

/// `I(d, c)` of the used pixels for a constant level alone, at its maximum
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

/// Parameter bounds of components in one window: the level, each
/// component's flux, its centre in the box `(y0, y1, x0, x1)`, and its
/// width when free.
struct Bounds {
    level: (f64, f64),
    flux: (f64, f64),
    centre: (f64, f64, f64, f64),
    width: (f64, f64),
}

impl Bounds {
    fn of(&self, lay: Layout) -> (Vec<f64>, Vec<f64>) {
        let (mut lo, mut hi) = (vec![self.level.0], vec![self.level.1]);
        let (y0, y1, x0, x1) = self.centre;
        for _ in 0..lay.k {
            lo.extend([self.flux.0, y0, x0]);
            hi.extend([self.flux.1, y1, x1]);
            if lay.sigma.is_none() {
                lo.push(self.width.0);
                hi.push(self.width.1);
            }
        }
        (lo, hi)
    }
}

/// Where one more component would gain most, and a joint start for the
/// refit: the used pixel in the box `(y0, y1, x0, x1)` maximizing the
/// efficient score of a new flux there (Neyman's C(alpha)), with the
/// components already fitted (`theta`) projected out,
///
/// ```text
/// z = (U + b^T F^-1 g) / sqrt(I - b^T F^-1 b)
/// ```
///
/// `U`, `I` the new flux's score and information at the template `k1`, `b`
/// its cross-information with `theta`, `F` and `g` theta's information and
/// objective gradient. A component that widened over an unfound neighbour
/// leaves little of it in the residual (`U` small), but the projection
/// leaves as little of the neighbour's information (`I_eff`), and `z` finds
/// it. The joint Newton step from there gives the new flux and moves the
/// rest. Returns that start, `theta` extended; `None` if no score is
/// positive.
fn propose(win: &Window, fitter: &mut Fitter, theta: &[f64], lay: Layout, k1: &[f64], (y0, y1, x0, x1): (f64, f64, f64, f64)) -> Option<Vec<f64>> {
    let (rows, cols, n) = (win.rows, win.cols, lay.n());
    let p = rows * cols;
    let lin = fitter.linearize(win, theta, lay);
    // With `F = L L^T`, `b^T F^-1 v = (L^-1 b) . (L^-1 v)`. Without a
    // factorization nothing is projected out: the plain score.
    let mut chol = crate::linalg::Chol::new(n);
    let projected = chol.factor(&lin.f, n);
    let mut lg = lin.g.clone();
    if projected {
        chol.forward_in_place(&mut lg);
    }
    let weighted = |v: &dyn Fn(usize) -> f64| -> Vec<f64> { (0..p).map(|q| if win.used[q] { v(q) / lin.m[q] } else { 0.0 }).collect() };
    let k2: Vec<f64> = k1.iter().map(|v| v * v).collect();
    let u_map = prefilter::correlate(&weighted(&|q| win.d[q] - lin.m[q]), rows, cols, k1);
    let i_map = prefilter::correlate(&weighted(&|_| 1.0), rows, cols, &k2);
    let b_maps: Vec<Vec<f64>> = (0..n).map(|a| prefilter::correlate(&weighted(&|q| lin.jac[a * p + q]), rows, cols, k1)).collect();
    let (ry0, ry1) = (y0.ceil().max(0.0) as usize, (y1.floor().max(0.0) as usize).min(rows - 1));
    let (rx0, rx1) = (x0.ceil().max(0.0) as usize, (x1.floor().max(0.0) as usize).min(cols - 1));
    // The best pixel: its score, the new flux's Newton estimate, and L^-1 b.
    let mut best: Option<(f64, usize, f64, Vec<f64>)> = None;
    let mut lb = vec![0.0; n];
    for py in ry0..=ry1 {
        for px in rx0..=rx1 {
            let q = py * cols + px;
            if !win.used[q] {
                continue;
            }
            let (mut u_eff, mut i_eff) = (u_map[q], i_map[q]);
            if projected {
                for (v, bm) in lb.iter_mut().zip(&b_maps) {
                    *v = bm[q];
                }
                chol.forward_in_place(&mut lb);
                i_eff -= lb.iter().map(|v| v * v).sum::<f64>();
                u_eff += lb.iter().zip(&lg).map(|(u, v)| u * v).sum::<f64>();
            }
            if !(i_eff > 1e-9 * i_map[q]) {
                continue;
            }
            let z = u_eff / i_eff.sqrt();
            if z > best.as_ref().map_or(0.0, |b| b.0) {
                best = Some((z, q, u_eff / i_eff, lb.clone()));
            }
        }
    }
    let (_, q, a, lb) = best?;
    // The joint Newton step: the new flux `a`, the rest moved by
    // `-F^-1 (g + b a)`.
    let mut start = theta.to_vec();
    if projected {
        let mut step: Vec<f64> = lg.iter().zip(&lb).map(|(g, b)| g + b * a).collect();
        chol.back_in_place(&mut step);
        for (t, d) in start.iter_mut().zip(&step) {
            *t -= d;
        }
    }
    start.extend([a, (q / cols) as f64, (q % cols) as f64]);
    Some(start)
}

/// Fit one seed's window: components are added while each refit gains
/// `u^2 / 2` nats, then the weakest is removed while removing it, the rest
/// refitted, costs less than that. Without mixtures, at most one.
#[allow(clippy::too_many_arguments)]
fn fit_seed(d: &[f64], h: usize, w: usize, sc: &Screen, seed: usize, at: &[bool], s: &Settings, phi: f64, u: f64, k1: &[f64], fitter: &mut Fitter) -> Fate {
    let mut fate = Fate::default();
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
    // Alone, a component is held near its seed, as u-track holds it. In a
    // mixture every source of light in the window takes a component, the
    // seed's neighbours as nuisance parameters, so the box is the window.
    let reach = CONFINE * s.sigma;
    // Reported widths are `[w_lo, w_hi]`; a component may widen to the
    // window's half-side, and is out-of-focus light beyond `w_hi`.
    let (w_lo, w_hi) = s.widths();
    let w_fit = if w_lo >= w_hi { (w_lo, w_lo) } else { (w_lo, r as f64) };
    let b = Bounds {
        level: (prefilter::LEVEL_FLOOR, 2.0 * dmax),
        flux: (0.0, 10.0 * dmax / psf::peak_factor(s.sigma)),
        centre: if s.fit_mixtures {
            (-0.5, rows as f64 - 0.5, -0.5, cols as f64 - 0.5)
        } else {
            (cy - reach, cy + reach, cx - reach, cx + reach)
        },
        width: (w_fit.0, w_fit.1),
    };
    let tol = FIT_TOL * phi;
    let bar = u * u * phi / 2.0;
    let max_k = if s.fit_mixtures { s.max_mixtures.max(1) } else { 1 };
    let lay = Layout { k: 0, sigma: (w_lo >= w_hi).then_some(w_lo) };
    let held = |theta: &[f64], at: &[bool]| {
        let lay = lay.with((theta.len() - 1) / lay.stride());
        (0..lay.k).any(|j| at[lay.at(j) + 1] || at[lay.at(j) + 2])
    };

    let mut theta = vec![sc.background[seed].clamp(b.level.0, b.level.1)];
    // The level alone: the null the first component is tested against.
    let null = level_divergence(&win);
    let mut div = null;

    // Forward: each new component where the efficient score peaks.
    let mut gains: Vec<f64> = Vec::new();
    let mut fit = None;
    while gains.len() < max_k {
        let k = gains.len();
        // Alone, the component starts at its seed. In a mixture the seed
        // only centres the window: every component, the first too, starts
        // where the efficient score peaks, and ownership decides which are
        // the seed's to report.
        let mut start = if k == 0 && !s.fit_mixtures {
            let mut t = theta.clone();
            t.extend([sc.amplitude[seed], cy, cx]);
            t
        } else {
            match propose(&win, fitter, &theta, lay.with(k), k1, b.centre) {
                Some(t) => t,
                None => break,
            }
        };
        let q = start.len() - 3;
        start[q] = start[q].clamp(1.0, b.flux.1);
        if lay.sigma.is_none() {
            start.push(s.sigma.clamp(w_lo, w_hi));
        }
        let (lo, hi) = b.of(lay.with(k + 1));
        let f = fitter.fit(&win, &start, lay.with(k + 1), &lo, &hi, FIT_MAX_ITER, tol);
        let gain = div - f.divergence;
        if !(gain >= bar) {
            break;
        }
        // In a mixture a component held at the window's edge stands for
        // light from outside it: a nuisance, not reported.
        if !s.fit_mixtures && held(&f.theta, &f.at_bound) {
            fate.unconfined = k == 0;
            break;
        }
        gains.push(gain);
        div = f.divergence;
        theta = f.theta.clone();
        fit = Some(f);
    }
    let Some(mut fit) = fit else {
        fate.weak = !fate.unconfined;
        return fate;
    };
    fate.added = gains.len() - 1;

    // Backward: the cost of removing each component this seed reports,
    // the rest refitted. The others are its neighbours' to test.
    let k_of = |theta: &[f64]| (theta.len() - 1) / lay.stride();
    let mine = |theta: &[f64], j: usize| {
        let q = lay.at(j);
        owned(seed, theta[q + 1] + y0 as f64, theta[q + 2] + x0 as f64, w, h, at)
    };
    let removal = |fitter: &mut Fitter, theta: &[f64], j: usize, div: f64| {
        let k = k_of(theta);
        if k == 1 {
            return (null - div, None);
        }
        let (lo, hi) = b.of(lay.with(k - 1));
        let mut start = theta.to_vec();
        start.drain(lay.at(j)..lay.at(j) + lay.stride());
        let f = fitter.fit(&win, &start, lay.with(k - 1), &lo, &hi, FIT_MAX_ITER, tol);
        (f.divergence - div, Some(f))
    };
    let mut cost: Vec<Option<f64>> = vec![None; k_of(&theta)];
    if cost.len() == 1 {
        cost[0] = Some(gains[0]);
    }
    loop {
        let k = k_of(&theta);
        if k == 1 && cost[0].is_some() {
            break;
        }
        cost = vec![None; k];
        let mut weakest: Option<(f64, Option<crate::fit::Fit>)> = None;
        for j in (0..k).filter(|&j| mine(&theta, j)) {
            let (c, f) = removal(fitter, &theta, j, div);
            cost[j] = Some(c);
            if weakest.as_ref().is_none_or(|w| c < w.0) {
                weakest = Some((c, f));
            }
        }
        match weakest {
            Some((c, Some(f))) if c < bar => {
                fate.removed += 1;
                div = f.divergence;
                theta = f.theta.clone();
                fit = f;
            }
            Some((c, None)) if c < bar => return fate,
            _ => break,
        }
    }

    let lay = lay.with(k_of(&theta));
    let mut common = 0;
    if !fit.converged {
        common |= FLAG_NOT_CONVERGED;
    }
    if fit.stalled {
        common |= FLAG_STALLED;
    }
    if fit.at_bound[0] {
        common |= FLAG_BOUND;
    }
    let var = fitter.variances(&win, &theta, lay, phi).filter(|v| v.iter().all(|&v| v > 0.0));
    for j in 0..lay.k {
        let q = lay.at(j);
        if fit.at_bound[q + 1] || fit.at_bound[q + 2] {
            continue;
        }
        if !mine(&theta, j) {
            fate.others += 1;
            continue;
        }
        if lay.sigma(&theta, j) > w_hi {
            fate.out_of_focus += 1;
            continue;
        }
        let mut flags = common;
        if (q..q + lay.stride()).any(|p| fit.at_bound[p]) {
            flags |= FLAG_BOUND;
        }
        let (mut se, mut se_s) = ([f64::NAN; 3], f64::NAN);
        match &var {
            Some(v) => {
                for (i, p) in (q..q + 3).enumerate() {
                    se[i] = v[p].sqrt();
                }
                if lay.sigma.is_none() {
                    se_s = v[q + 3].sqrt();
                }
            }
            None => flags |= FLAG_COVARIANCE,
        }
        fate.spots.push(Spot {
            y: theta[q + 1] + y0 as f64,
            x: theta[q + 2] + x0 as f64,
            a: theta[q],
            s: lay.sigma(&theta, j),
            se,
            se_s,
            c: theta[0],
            z: (2.0 * cost[j].expect("an owned component's cost") / phi).sqrt(),
            flags,
            seed,
            mixture: if lay.k > 1 { seed as u32 + 1 } else { 0 },
        });
    }
    fate
}

/// Whether `seed` is the seed nearest to `(y, x)` (ties to the lower
/// index): each emitter is reported by the fit centred closest to it.
fn owned(seed: usize, y: f64, x: f64, w: usize, h: usize, at: &[bool]) -> bool {
    let (sy, sx) = ((seed / w) as f64, (seed % w) as f64);
    let d_own = (y - sy).powi(2) + (x - sx).powi(2);
    let reach = d_own.sqrt().ceil() as isize;
    let (yc, xc) = (y.round() as isize, x.round() as isize);
    for yy in (yc - reach).max(0)..(yc + reach + 1).min(h as isize) {
        for xx in (xc - reach).max(0)..(xc + reach + 1).min(w as isize) {
            let i = yy as usize * w + xx as usize;
            if i == seed || !at[i] {
                continue;
            }
            let d = (y - yy as f64).powi(2) + (x - xx as f64).powi(2);
            if d < d_own || (d == d_own && i < seed) {
                return false;
            }
        }
    }
    true
}

impl Output {
    fn tally(&mut self, fate: &Fate) {
        self.weak += fate.weak as usize;
        self.unconfined += fate.unconfined as usize;
        self.added += fate.added;
        self.removed += fate.removed;
        self.duplicates += fate.others;
        self.out_of_focus += fate.out_of_focus;
    }
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
    let (w_lo, w_hi) = s.widths();
    let u = prefilter::threshold(s.fp_per_mpx, w_lo, w_hi);
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
    let sc = prefilter::screen(&dc, ch, cw, s.sigma, phi, &bank(w_lo, w_hi, u), rc.as_deref());
    let mut out = Output { dispersion: phi, u, n_seeds: sc.seeds.len(), ..Output::default() };
    out.seeds = sc.seeds.iter().flat_map(|&i| [(i / cw + r0) as f64, (i % cw + c0) as f64]).collect();
    let k1 = prefilter::psf_kernel1d(s.sigma);
    let mut at = vec![false; ch * cw];
    for &i in &sc.seeds {
        at[i] = true;
    }
    let mut spots = Vec::new();
    for &seed in &sc.seeds {
        let fate = fit_seed(&dc, ch, cw, &sc, seed, &at, s, phi, u, &k1, fitter);
        out.tally(&fate);
        spots.extend(fate.spots);
    }
    spots.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));
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
        out.mixture.push(p.mixture);
        out.seed.extend([(p.seed / cw + r0) as f64, (p.seed % cw + c0) as f64]);
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
        let o = localize(&d, h, w, None, &Settings::new(sigma));
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
