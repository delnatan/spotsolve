//! Spot detection in ADU above the camera offset, after u-track's
//! `pointSourceDetection` (Aguet et al. 2013, *Dev. Cell* 26:279).
//!
//! The movie sets the bar `u` on every decision ([`localize_stack`]): each
//! frame's seeds are fitted alone at [`posterior::CANDIDATE`], and `u` is
//! the least root likelihood ratio from which an emitter is real, not
//! noise, with probability `p_real` ([`Posterior`]). Each frame is then:
//!
//! 1. Dispersion `phi` from the frame ([`prefilter::dispersion`]).
//! 2. Seeds: LoG maxima where the Poisson score of one emitter reaches
//!    [`posterior::CANDIDATE`], with the mask's labels
//!    ([`prefilter::screen`]).
//! 3. Each seed's window ([`crate::fit`]), `ceil(4 sigma)` px around it, holds a
//!    constant level and pixel-integrated Gaussian components; pixels of
//!    other mask components are left out. A component is kept if adding it
//!    gains `u^2 / 2` nats, `2 (I_k - I_k+1) / phi >= u^2`.
//!    - Alone, one component starts at the seed and its centre is held
//!      within [`CONFINE`] widths of it.
//!    - With mixtures, components are added where the efficient score
//!      peaks ([`propose`]) while each refit gains as much, then the
//!      weakest is removed while removing it, the rest refitted, costs
//!      less. Every source of light in the window takes a component, so a
//!      neighbour is a nuisance parameter rather than a bias.
//! 4. Each component is reported by the fit of the seed nearest to it
//!    ([`owned`]), and only those are tested for removal; copies of one
//!    emitter from two fits are merged ([`merge`]). A component held at a
//!    position bound is not reported.
//! 5. Every component has its own width. Emitters are reported in `width *
//!    sigma` (by default [`WIDTH`]); the null of `u` is that of a search over
//!    position and width ([`prefilter::false_rate`]). A component may widen
//!    past that to the window's half-side as out-of-focus light, which
//!    narrower components would otherwise split; it is counted, not
//!    reported, once it is [`WIDER`] standard errors wider than the bound,
//!    and is otherwise an emitter held at the bound. Equal bounds fix the
//!    width.
//! 6. Standard errors from the reported fit's Fisher information, scaled by
//!    `phi`.

use crate::fit::{Fitter, Layout, Window};
use crate::posterior::{self, Posterior};
use crate::prefilter::{self, Screen};
use crate::psf;

/// Default least probability that a reported emitter is real, not noise.
pub const P_REAL: f64 = 0.9;
/// Default reported widths, multiples of `sigma`: from the in-focus width
/// to a little past `sqrt(2) sigma`, where defocus has halved the peak (the
/// edge of the PSF's axial FWHM).
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
/// its light inside. A component may widen to `R` itself: [`WIDER`] than
/// the reported widths it is out-of-focus light, fitted so that narrower
/// components need not explain it, and returned as background.
pub fn widest(sigma: f64) -> f64 {
    0.5 * (WINDOW * sigma).ceil()
}

/// Standard errors by which a component's width must exceed the reported
/// widths for it to be out-of-focus light rather than an emitter at the
/// upper bound: a one-sided test, so an emitter on the bound is taken for
/// out-of-focus light in `Phi(-2) = 2.3%` of frames, and one inside it
/// fewer. A point estimate alone would drop half of those on the bound.
pub const WIDER: f64 = 2.0;

/// Widths: two reports from different windows closer than this, each the
/// other window's nearest component, are one emitter fitted twice
/// ([`merge`]). Two fits of one emitter differ by how they share its
/// neighbours' light, within a width; a pair one window resolved holds two
/// components there and is never merged.
pub const MERGE: f64 = 1.0;

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
    /// Least probability that a reported emitter is real, not noise; sets
    /// `u`.
    pub p_real: f64,
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
        Self { sigma, p_real: P_REAL, width: WIDTH, fit_mixtures: false, max_mixtures: MAX_MIXTURES }
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
    /// `N`: the root of each emitter's likelihood ratio, the cost of
    /// removing it with the rest of its window refitted, `sqrt(2 (I_without
    /// - I) / phi)`; at least `u`.
    pub z: Vec<f64>,
    /// `N`: the probability each emitter is real, not noise, from its `z`
    /// ([`Posterior`]); at least `p_real`.
    pub p_real: Vec<f64>,
    /// `N`: the fitted level of each emitter's window.
    pub fitted_background: Vec<f64>,
    /// Scalar dispersion: pixel variance per unit of signal.
    pub dispersion: f64,
    /// `H*W`: the screening level (the window regression's constant); NaN
    /// outside the processed crop.
    pub background: Vec<f64>,
    /// The movie's bar on `z`, where `P(real | z)` reaches `p_real`.
    pub u: f64,
    pub n_seeds: usize,
    /// Seeds whose emitter fell short of `u`, or left its confinement.
    pub weak: usize,
    pub unconfined: usize,
    /// Components fitted from a seed other than the one nearest to them, and
    /// so reported by that one's fit instead, and reports merged as copies
    /// of another ([`merge`]).
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
    /// Components [`WIDER`] than the reported widths, fitted as
    /// out-of-focus light and returned as background.
    pub out_of_focus: usize,
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
    /// Index of its window's fit in the frame's [`Fate`]s.
    window: usize,
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
    /// Every component of the final fit, `(y, x)` in crop coordinates.
    components: Vec<(f64, f64)>,
}

/// `I(d, c)` of the used pixels for a constant level alone, at its minimum
/// `c = mean(max(d, 0))`: `I` reads a pixel below the offset as 0 counts,
/// as in the fitted models.
fn level_divergence(win: &Window) -> f64 {
    let (n, sum) = win.d.iter().zip(&win.used).filter(|p| *p.1).fold((0.0, 0.0), |(n, s), (d, _)| (n + 1.0, s + d.max(0.0)));
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
/// objective gradient. The projection finds a neighbour that a fitted
/// component has partly absorbed. The joint Newton step from there gives
/// the new flux and moves the rest. Returns that start, `theta` extended;
/// `None` if no score is positive.
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
    // Alone, a component is held near its seed; in a mixture anywhere in
    // the window.
    let reach = CONFINE * s.sigma;
    // Reported widths are `[w_lo, w_hi]`; a component may widen to the
    // window's half-side, and is out-of-focus light [`WIDER`] than `w_hi`.
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
    let mut fit = None;
    let mut k = 0;
    while k < max_k {
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
        k += 1;
        div = f.divergence;
        theta = f.theta.clone();
        fit = Some(f);
    }
    let Some(mut fit) = fit else {
        fate.weak = !fate.unconfined;
        return fate;
    };
    fate.added = k - 1;

    // Backward: the cost of removing each component this seed reports,
    // the rest refitted. The others are its neighbours' to test.
    let k_of = |theta: &[f64]| (theta.len() - 1) / lay.stride();
    let mine = |theta: &[f64], j: usize| {
        let q = lay.at(j);
        owned(seed, theta[q + 1] + y0 as f64, theta[q + 2] + x0 as f64, w, h, at)
    };
    let removal = |fitter: &mut Fitter, theta: &[f64], j: usize, div: f64| {
        let k = k_of(theta);
        let (lo, hi) = b.of(lay.with(k - 1));
        let mut start = theta.to_vec();
        start.drain(lay.at(j)..lay.at(j) + lay.stride());
        let f = fitter.fit(&win, &start, lay.with(k - 1), &lo, &hi, FIT_MAX_ITER, tol);
        (f.divergence - div, f)
    };
    // One component left costs its gain over the level alone, which every
    // forward gain reaching the bar and every removal falling short of it
    // keeps at least the bar.
    let mut cost: Vec<Option<f64>>;
    loop {
        let k = k_of(&theta);
        if k == 1 {
            cost = vec![Some(null - div)];
            break;
        }
        cost = vec![None; k];
        let mut weakest: Option<(f64, crate::fit::Fit)> = None;
        for j in (0..k).filter(|&j| mine(&theta, j)) {
            let (c, f) = removal(fitter, &theta, j, div);
            cost[j] = Some(c);
            if weakest.as_ref().is_none_or(|w| c < w.0) {
                weakest = Some((c, f));
            }
        }
        match weakest {
            Some((c, f)) if c < bar => {
                fate.removed += 1;
                div = f.divergence;
                theta = f.theta.clone();
                fit = f;
            }
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
    fate.components = (0..lay.k).map(|j| (theta[lay.at(j) + 1] + y0 as f64, theta[lay.at(j) + 2] + x0 as f64)).collect();
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
        let mut cost_j = cost[j].expect("an owned component's cost");
        // Wider than the reported widths, a component is out-of-focus light
        // only if it is [`WIDER`] standard errors wide of `w_hi`: holding
        // it there, the rest refitted, costs `WIDER^2 / 2` nats. Otherwise
        // it is an emitter at `w_hi`, if it still gains the bar.
        let held = if lay.sigma(&theta, j) > w_hi {
            let (lo, mut hi) = b.of(lay);
            hi[q + 3] = w_hi;
            let mut start = theta.clone();
            start[q + 3] = w_hi;
            let f = fitter.fit(&win, &start, lay, &lo, &hi, FIT_MAX_ITER, tol);
            let excess = f.divergence - div;
            if excess >= WIDER * WIDER * phi / 2.0 || cost_j - excess < bar || f.at_bound[q + 1] || f.at_bound[q + 2] || !mine(&f.theta, j) {
                fate.out_of_focus += 1;
                continue;
            }
            cost_j -= excess;
            let v = fitter.variances(&win, &f.theta, lay, phi).filter(|v| v.iter().all(|&v| v > 0.0));
            Some((f, v))
        } else {
            None
        };
        let (theta, at_bound, var) = match &held {
            Some((f, v)) => (&f.theta, &f.at_bound, v),
            None => (&theta, &fit.at_bound, &var),
        };
        // A reported width on its floor, the in-focus width, is where an
        // in-focus emitter belongs: not flagged. One held at `w_hi` is.
        let mut flags = common;
        if (q..q + 3).any(|p| at_bound[p]) || held.is_some() {
            flags |= FLAG_BOUND;
        }
        let (mut se, mut se_s) = ([f64::NAN; 3], f64::NAN);
        match var {
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
            s: lay.sigma(theta, j),
            se,
            se_s,
            c: theta[0],
            z: (2.0 * cost_j / phi).sqrt(),
            flags,
            seed,
            window: 0,
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

/// Drop copies of one emitter reported by two windows: reports `a` and `b`
/// from different fits, closer than `reach`, where `b`'s nearest component
/// in `a`'s fit is `a` and `a`'s nearest in `b`'s is `b`. Of the two, the
/// one whose seed is nearer their midpoint is kept, as ownership would on a
/// position both fits agreed on. Returns how many were dropped.
fn merge(spots: &mut Vec<Spot>, fates: &[Fate], w: usize, reach: f64) -> usize {
    let d2 = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2);
    let nearest_is = |fit: &[(f64, f64)], to: (f64, f64), me: (f64, f64)| fit.iter().all(|&c| d2(c, to) >= d2(me, to));
    let seed_at = |s: usize| ((s / w) as f64, (s % w) as f64);
    let mut order: Vec<usize> = (0..spots.len()).collect();
    order.sort_by(|&i, &j| spots[i].y.total_cmp(&spots[j].y));
    let mut drop = vec![false; spots.len()];
    for (n, &i) in order.iter().enumerate() {
        for &j in &order[n + 1..] {
            let (a, b) = (&spots[i], &spots[j]);
            if b.y - a.y >= reach {
                break;
            }
            let (pa, pb) = ((a.y, a.x), (b.y, b.x));
            if drop[i] || drop[j] || a.window == b.window || d2(pa, pb) >= reach * reach {
                continue;
            }
            if !nearest_is(&fates[a.window].components, pb, pa) || !nearest_is(&fates[b.window].components, pa, pb) {
                continue;
            }
            let mid = (0.5 * (a.y + b.y), 0.5 * (a.x + b.x));
            let (da, db) = (d2(seed_at(a.seed), mid), d2(seed_at(b.seed), mid));
            let a_stays = da < db || (da == db && a.seed < b.seed);
            drop[if a_stays { j } else { i }] = true;
        }
    }
    let kept: Vec<Spot> = spots.drain(..).zip(&drop).filter(|p| !*p.1).map(|p| p.0).collect();
    *spots = kept;
    drop.iter().filter(|&&d| d).count()
}

impl Output {
    /// Keep the emitters at or above `u`; those below are weak seeds.
    fn above(mut self, u: f64) -> Self {
        let keep: Vec<bool> = self.z.iter().map(|&z| z >= u).collect();
        fn take<T: Copy>(v: &[T], keep: &[bool], k: usize) -> Vec<T> {
            v.chunks(k).zip(keep).filter(|p| *p.1).flat_map(|p| p.0.iter().copied()).collect()
        }
        self.weak += keep.iter().filter(|&&k| !k).count();
        (self.pos, self.seed, self.se) = (take(&self.pos, &keep, 2), take(&self.seed, &keep, 2), take(&self.se, &keep, 3));
        (self.amp, self.sig, self.se_sig, self.z) = (take(&self.amp, &keep, 1), take(&self.sig, &keep, 1), take(&self.se_sig, &keep, 1), take(&self.z, &keep, 1));
        (self.flags, self.fitted_background, self.mixture) = (take(&self.flags, &keep, 1), take(&self.fitted_background, &keep, 1), take(&self.mixture, &keep, 1));
        self.u = u;
        self
    }

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

/// Localize an offset-subtracted frame in ADU, a movie of one frame
/// ([`localize_stack`]). An ROI limits seeds; the work runs on its bounding
/// box plus the context [`roi_crop`] gives.
pub fn localize(d: &[f64], h: usize, w: usize, roi: Option<&[bool]>, s: &Settings) -> Output {
    localize_raw(d, h, w, 0.0, roi, s)
}

/// One frame at the bar `u`.
fn localize_with(d: &[f64], h: usize, w: usize, roi: Option<&[bool]>, s: &Settings, u: f64, fitter: &mut Fitter) -> Output {
    assert_eq!(d.len(), h * w);
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
    let sc = prefilter::screen(&dc, ch, cw, s.sigma, phi, posterior::CANDIDATE, rc.as_deref());
    let mut out = Output { dispersion: phi, u, n_seeds: sc.seeds.len(), ..Output::default() };
    out.seeds = sc.seeds.iter().flat_map(|&i| [(i / cw + r0) as f64, (i % cw + c0) as f64]).collect();
    let k1 = prefilter::psf_kernel1d(s.sigma);
    let mut at = vec![false; ch * cw];
    for &i in &sc.seeds {
        at[i] = true;
    }
    let mut spots = Vec::new();
    let mut fates = Vec::with_capacity(sc.seeds.len());
    for &seed in &sc.seeds {
        let mut fate = fit_seed(&dc, ch, cw, &sc, seed, &at, s, phi, u, &k1, fitter);
        out.tally(&fate);
        spots.extend(fate.spots.drain(..).map(|p| Spot { window: fates.len(), ..p }));
        fates.push(fate);
    }
    out.duplicates += merge(&mut spots, &fates, cw, MERGE * s.sigma);
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
    localize_stack(raw, 1, h, w, offset, roi, s, 1).pop().expect("one frame")
}

/// Localize every frame of a stack, `d = raw - offset` in ADU, on
/// `n_threads` workers, in frame order. `raw` is `n*H*W`. The bar `u` is
/// the movie's: every frame's candidates, fitted alone above
/// [`posterior::CANDIDATE`], set `P(real | z)` ([`Posterior`]), and each
/// frame is then fitted at the `u` where it reaches `p_real`.
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
    let frame = |t: usize| -> Vec<f64> { raw[t * h * w..(t + 1) * h * w].iter().map(|&r| r - offset).collect() };
    let alone = Settings { fit_mixtures: false, ..*s };
    let candidates = crate::frames::map(n, n_threads, Fitter::default, |t, fitter| {
        localize_with(&frame(t), h, w, roi, &alone, posterior::CANDIDATE, fitter)
    });
    let area = n as f64 * roi.map_or((h * w) as f64, |m| m.iter().filter(|&&v| v).count() as f64);
    let (w_lo, w_hi) = s.widths();
    let heights: Vec<f64> = candidates.iter().flat_map(|o| o.z.iter().copied()).collect();
    let post = Posterior::fit(&heights, area, w_lo, w_hi);
    let u = post.cut(s.p_real);
    // Fitted alone, the frames' emitters are their candidates above `u`;
    // mixtures refit every frame at `u`.
    let mut out = if s.fit_mixtures {
        crate::frames::map(n, n_threads, Fitter::default, |t, fitter| localize_with(&frame(t), h, w, roi, s, u, fitter))
    } else {
        candidates.into_iter().map(|o| o.above(u)).collect()
    };
    for o in &mut out {
        o.p_real = o.z.iter().map(|&z| post.p_real(z)).collect();
    }
    out
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

    /// Two windows' copies of one emitter merge to the one whose seed is
    /// nearer their midpoint; a pair one window resolved stays.
    #[test]
    fn copies_merge_and_resolved_pairs_stay() {
        let w = 40;
        let spot = |y: f64, x: f64, seed: usize, window: usize| Spot {
            y, x, a: 1.0, s: 1.0, se: [0.0; 3], se_s: 0.0, c: 0.0, z: 5.0, flags: 0, seed, window, mixture: 0,
        };
        let fate = |c: Vec<(f64, f64)>| Fate { components: c, ..Fate::default() };
        // One emitter near (10, 10.3), seeds at (10, 7) and (10, 14).
        let (sa, sb) = (10 * w + 7, 10 * w + 14);
        let mut spots = vec![spot(10.0, 10.1, sa, 0), spot(10.0, 10.6, sb, 1)];
        let fates = [fate(vec![(10.0, 10.1)]), fate(vec![(10.0, 10.6)])];
        assert_eq!(merge(&mut spots, &fates, w, 1.45), 1);
        assert_eq!((spots.len(), spots[0].seed), (1, sa));
        // Window 0 resolved two emitters there: neither is a copy.
        let mut spots = vec![spot(10.0, 10.1, sa, 0), spot(10.0, 10.6, sb, 1)];
        let fates = [fate(vec![(10.0, 10.1), (10.0, 10.7)]), fate(vec![(10.0, 10.6)])];
        assert_eq!(merge(&mut spots, &fates, w, 1.45), 0);
        assert_eq!(spots.len(), 2);
    }

    #[test]
    fn an_isolated_emitter_is_found_once_where_it_is() {
        let (h, w, sigma) = (41, 43, 1.2);
        let theta = [0.0, 1500.0, 19.3, 21.6, 1.3];
        let mut d = psf::model(&theta, h, w);
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

    /// An emitter on the upper reported width is reported in all but about
    /// `Phi(-WIDER)` of frames, however its fitted width falls; a bright
    /// one well wider is out-of-focus light.
    #[test]
    fn an_emitter_on_the_width_bound_is_reported_and_wider_light_is_not() {
        let (h, w, sigma, n) = (31, 31, 1.2, 200);
        let s = Settings::new(sigma);
        let movie = |flux: f64, width: f64, seed: u64| {
            let raw: Vec<f64> = (0..n as u64)
                .flat_map(|i| {
                    let mut d = psf::model(&[0.0, flux, 15.2, 14.7, width], h, w);
                    for (v, z) in d.iter_mut().zip(normals(h * w, seed + i)) {
                        *v += 10.0;
                        *v += v.sqrt() * z;
                    }
                    d
                })
                .collect();
            localize_stack(&raw, n, h, w, 0.0, None, &s, 4)
        };
        let reported = movie(250.0, s.widths().1, 100).iter().filter(|o| o.amp.len() == 1).count();
        assert!(reported as f64 >= 0.95 * n as f64, "reported in {reported} of {n} frames");
        let wide = movie(3000.0, 2.0 * s.widths().1, 7);
        assert!(wide.iter().all(|o| o.amp.is_empty()), "wide light reported");
        assert!(wide.iter().map(|o| o.out_of_focus).sum::<usize>() >= n, "wide light not fitted");
    }
}
