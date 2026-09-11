//! Box-local localization: in each box, the best fit the data support wins.
//!
//! Ports `box.py::localize_boxes`, which stays the reference and holds the
//! measurements behind every constant here. The port is held to statistical
//! parity with it, not to its trajectory: same decisions on the referee
//! cells to within seed noise, and on real frames the same count to 1%.
//!
//! ```text
//! d_e      (raw - offset) / gain + read_noise^2             shifted Poisson
//! FIND     LoG peaks of d_e against a flat level            -> candidates
//! BMAP     smooth surface from the pixels no candidate reaches
//! BOXES    candidates within LINK_FACTOR*sigma share a box, <= k_max each
//! SWEEPS times, for each box, brightest first:
//!     fit the level alone (K = 0), BMAP's shape and the neighbours' current
//!     emitters held fixed
//!     FORWARD   place at the strongest owned residual peak that passes
//!               FIND's z-threshold; keep it iff I falls by > ADD_NATS
//!     BACKWARD  (K >= 2) drop the cheapest emitter while it costs < ADD_NATS
//! POLISH   block-Jacobi refits at fixed N until nothing moves
//! ```
//!
//! # Where Python spent the time, and why this module exists
//!
//! On frame 0 of the glycerol crop (485 emitters, 0.84 s in Python) the
//! native fits were 0.10 s. The rest was the search around them: the polish
//! (0.32 s), a scipy LoG per placement (0.21), ownership distances over every
//! candidate (0.14), gathering every box's emitters into each halo
//! (0.14, quadratic in boxes) and Python PSF renders (0.13). Here ownership
//! and the neighbour lists are computed once per frame, and a halo reads only
//! the boxes that can reach it.

use crate::filters::{self, Mode};
use crate::grid::EmitterGrid;
use crate::linalg::Chol;
use crate::lmcl::{self, Bounds, FitOpts, FitWorkspace};
use crate::patches::{self, HALO_FACTOR};
use crate::psf;
use crate::render;
use crate::statistics;

/// Nats of I-divergence an emitter must explain to exist. `box.ADD_NATS`.
pub const ADD_NATS: f64 = 10.0;
/// sigma. A box places only on pixels this near one of its own candidates,
/// and nearer to its own than to any other. `box.OWN_RADIUS`.
pub const OWN_RADIUS: f64 = 3.0;
/// Passes over every box. `box.SWEEPS`.
pub const SWEEPS: usize = 2;
/// `box.FIT_TOL_OBJ`, `box.FIT_MAX_ITER`.
pub const FIT_TOL_OBJ: f64 = 1e-6;
pub const FIT_MAX_ITER: usize = 100;
/// `core.REFINE_SWEEPS`, `REFINE_MAX_ITER`, `REFINE_TOL_OBJ`, `REFINE_TOL` (px).
pub const POLISH_SWEEPS: usize = 4;
pub const POLISH_MAX_ITER: usize = 50;
pub const POLISH_TOL_OBJ: f64 = 1e-6;
pub const POLISH_MOVE_TOL: f64 = 1e-3;
/// `core.BG_KERNEL`, `BG_FLOOR`, `BG_MASK_RADIUS`, `BG_MIN_PIXELS`.
pub const BG_KERNEL: usize = 25;
pub const BG_FLOOR: f64 = 1e-3;
pub const BG_MASK_RADIUS: f64 = 3.0;
pub const BG_MIN_PIXELS: f64 = 25.0;
/// `calibrate.SEED_ALPHA`: FIND's family-wise false-seed rate per frame.
pub const SEED_ALPHA: f64 = 0.05;
/// `moves.A_MIN` and `core.A_MIN_REL`: the amplitude floor of a fit.
pub const A_MIN: f64 = 1e-4;
pub const A_MIN_REL: f64 = 1e-6;

/// What a caller chooses per frame.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// In-focus PSF width, px.
    pub sigma: f64,
    pub k_max: usize,
    /// FIND's cut in sd of the LoG null; see [`seed_threshold`].
    pub threshold: f64,
    /// Widths a fit may take, as multiples of `sigma`.
    pub slack: (f64, f64),
    pub sweeps: usize,
    pub polish: bool,
}

/// One frame's answer. Every fitted emitter, in or out of any reporting band
/// -- the caller classifies. Positions are global `(y, x)`.
#[derive(Clone, Debug, Default)]
pub struct Output {
    /// `2N`, row-major `(y, x)`.
    pub pos: Vec<f64>,
    pub amp: Vec<f64>,
    pub sig: Vec<f64>,
    /// `3N`: SE of `(A, y, x)` from the polish's Fisher matrix; NaN without.
    pub se: Vec<f64>,
    /// `H*W`, in the units of `d_e` (shifted by `read_noise^2`).
    pub background: Vec<f64>,
    pub n_candidates: usize,
    pub n_boxes: usize,
    pub search_fits: usize,
    pub polish_fits: usize,
}

/// Reusable per-thread storage: one per worker, never shared.
pub struct Workspace {
    fit: FitWorkspace,
    f: psf::Factors,
    chol: Chol,
    scratch: Vec<f64>,
    model: Vec<f64>,
}

impl Workspace {
    pub fn new() -> Self {
        Self {
            fit: FitWorkspace::new(),
            f: psf::Factors::new(1, 1, 1),
            chol: Chol::new(1),
            scratch: Vec::new(),
            model: Vec::new(),
        }
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// FIND's seed cut, `calibrate.seed_threshold`: the Bonferroni cut at a
/// family-wise rate `alpha` over the frame's independent LoG maxima, one per
/// `(2*ceil(sigma)+1)^2` pixels.
pub fn seed_threshold(h: usize, w: usize, sigma: f64, alpha: f64) -> f64 {
    let win = (2 * sigma.ceil() as usize + 1) as f64;
    let n = (h as f64 * w as f64 / (win * win)).max(1.0);
    let p = alpha.clamp(1e-12, 0.999) / n;
    -statistics::normal_quantile(p)
}

/// `np.percentile(v, q)`, linear interpolation.
pub fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = (s.len() - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (pos - lo as f64) * (s[hi] - s[lo])
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 0 {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    } else {
        s[n / 2]
    }
}

/// FIND against a flat `level`: LoG peaks of the variance-normalized
/// residual above `threshold`, brightest first. `core.find_candidates` with
/// no committed emitters. Returns `(pos 2N, amp, strength)`.
pub fn find_candidates(
    d: &[f64],
    h: usize,
    w: usize,
    level: f64,
    sigma: f64,
    threshold: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let norm = level.max(1e-6).sqrt();
    let nr: Vec<f64> = d.iter().map(|&v| (v - level) / norm).collect();
    let l2 = filters::log_kernel_l2(sigma);
    let mut log_f = filters::gaussian_laplace(&nr, h, w, sigma, Mode::Reflect);
    for v in log_f.iter_mut() {
        *v = -*v / l2;
    }
    let win = 2 * sigma.ceil() as usize + 1;
    let mx = filters::maximum_filter(&log_f, h, w, win, Mode::Reflect);
    let pf = psf::peak_factor(sigma);
    let mut found: Vec<(f64, usize)> = (0..h * w)
        .filter(|&i| log_f[i] == mx[i] && log_f[i] > threshold)
        .map(|i| (log_f[i], i))
        .collect();
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut pos = Vec::with_capacity(2 * found.len());
    let mut amp = Vec::with_capacity(found.len());
    let mut strength = Vec::with_capacity(found.len());
    for &(s, i) in &found {
        pos.push((i / w) as f64);
        pos.push((i % w) as f64);
        amp.push((d[i] - level).max(1e-2) / pf);
        strength.push(s);
    }
    (pos, amp, strength)
}

/// `core.background_map`: a local mean over the pixels no candidate reaches,
/// taken twice with a one-sided clip between, then smoothed. Windows with
/// too few free pixels take `calibrate.robust_background`'s scalar.
pub fn background_map(d: &[f64], h: usize, w: usize, cand: &[f64], sigma: f64) -> Vec<f64> {
    let n = cand.len() / 2;
    let mut k = BG_KERNEL.min(3usize.max(h.min(w) / 3));
    if k % 2 == 0 {
        k += 1;
    }
    let free = render::emitter_free_mask(cand, n, sigma, BG_MASK_RADIUS, h, w);
    let fallback = if n == 0 {
        median(d)
    } else {
        let kept: Vec<f64> = (0..h * w).filter(|&i| free[i]).map(|i| d[i]).collect();
        if kept.len() as f64 >= 16f64.max(0.02 * (h * w) as f64) {
            median(&kept)
        } else {
            percentile(d, 10.0)
        }
    };
    let local_mean = |mask: &[bool]| -> (Vec<f64>, Vec<f64>) {
        let masked: Vec<f64> = (0..h * w).map(|i| if mask[i] { d[i] } else { 0.0 }).collect();
        let ones: Vec<f64> = mask.iter().map(|&m| if m { 1.0 } else { 0.0 }).collect();
        let num = filters::uniform_filter(&masked, h, w, k, Mode::Nearest);
        let den = filters::uniform_filter(&ones, h, w, k, Mode::Nearest);
        let b = num.iter().zip(&den).map(|(a, c)| a / c.max(1e-9)).collect();
        let cnt = den.iter().map(|c| c * (k * k) as f64).collect();
        (b, cnt)
    };
    let (b1, _) = local_mean(&free);
    let keep: Vec<bool> = (0..h * w)
        .map(|i| free[i] && d[i] <= b1[i] + 3.0 * b1[i].max(BG_FLOOR).sqrt())
        .collect();
    let (b2, cnt) = local_mean(&keep);
    let b3: Vec<f64> = (0..h * w)
        .map(|i| if cnt[i] >= BG_MIN_PIXELS { b2[i] } else { fallback })
        .collect();
    let mut out = filters::gaussian_filter(&b3, h, w, k as f64 / 6.0, Mode::Nearest);
    for v in out.iter_mut() {
        *v = v.max(BG_FLOOR);
    }
    out
}

/// An emitter: `[A, y, x, sigma]`.
type Em = [f64; 4];

/// A window's pixels and the parameter-free part of its model.
struct Window {
    y0: usize,
    x0: usize,
    h: usize,
    w: usize,
    sub: Vec<f64>,
    /// Frozen neighbours plus the background map's shape.
    halo: Vec<f64>,
    /// The background map's median here; where the free level starts.
    level: f64,
    /// Background map's shape, `bmap - level`.
    shape: Vec<f64>,
}

impl Window {
    fn new(d: &[f64], fw: usize, bmap: &[f64], bb: &patches::BBox) -> Self {
        let (h, w) = (bb.h(), bb.w());
        let mut sub = Vec::with_capacity(h * w);
        let mut bg = Vec::with_capacity(h * w);
        for r in bb.y0..bb.y1 {
            sub.extend_from_slice(&d[r * fw + bb.x0..r * fw + bb.x1]);
            bg.extend_from_slice(&bmap[r * fw + bb.x0..r * fw + bb.x1]);
        }
        let level = median(&bg);
        let shape: Vec<f64> = bg.iter().map(|v| v - level).collect();
        Self {
            y0: bb.y0,
            x0: bb.x0,
            h,
            w,
            sub,
            halo: shape.clone(),
            level,
            shape,
        }
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

/// One converged window fit: its data-only I and parameters, local coords.
struct Fitted {
    i_div: f64,
    b: f64,
    em: Vec<Em>,
}

/// `core._fit_any`'s free-width branch: bounds from the window's peak, the
/// start pulled 1e-9 inside them, one bounded ML fit.
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
        em: (0..k)
            .map(|j| [t[1 + 4 * j], t[2 + 4 * j], t[3 + 4 * j], t[4 + 4 * j]])
            .collect(),
    }
}

/// `(y, x, A0)` of the strongest owned peak of the fit's normalized residual
/// that passes FIND's own test, or `None`. `_Search.placement`: the test is
/// what makes ADD_NATS mean what it was measured to mean.
///
/// The tests here and in [`search_box`] are written `!(a > b)` on purpose: a
/// NaN must fail them, as `not a > b` does in the reference.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn placement(
    ws: &mut Workspace,
    win: &Window,
    owned: &[bool],
    state: &Fitted,
    s: &Settings,
    l2: f64,
) -> Option<(f64, f64, f64)> {
    let n = win.h * win.w;
    let mut theta = Vec::with_capacity(4 * state.em.len() + 1);
    theta.push(state.b);
    for e in &state.em {
        theta.extend_from_slice(e);
    }
    let (ay, ax) = (psf::local_axis(win.h), psf::local_axis(win.w));
    ws.f.ensure(win.h, win.w, state.em.len().max(1));
    ws.model.clear();
    ws.model.resize(n, 0.0);
    psf::model_var_sigma_ax(&theta, &ay, &ax, Some(&win.halo), &mut ws.f, &mut ws.model);
    let nr: Vec<f64> = (0..n)
        .map(|i| (win.sub[i] - ws.model[i]) / ws.model[i].max(1e-6).sqrt())
        .collect();
    let log_f = filters::gaussian_laplace(&nr, win.h, win.w, s.sigma, Mode::Nearest);
    let mut best: Option<(f64, usize)> = None;
    for i in 0..n {
        if owned[i] {
            let v = -log_f[i] / l2;
            if best.is_none_or(|(b, _)| v > b) {
                best = Some((v, i));
            }
        }
    }
    let (v, i) = best?;
    if !(v > s.threshold) {
        return None;
    }
    let resid = win.sub[i] - ws.model[i];
    Some(((i / win.w) as f64, (i % win.w) as f64, resid.max(1e-2) / psf::peak_factor(s.sigma)))
}

/// One box decided from K = 0: FORWARD placements while each pays ADD_NATS,
/// then BACKWARD removals while one costs less. `_Search.run`. Returns the
/// box's emitters (local) and the fits spent.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn search_box(
    ws: &mut Workspace,
    win: &Window,
    owned: &[bool],
    s: &Settings,
    l2: f64,
) -> (Vec<Em>, usize) {
    let mut fits = 1usize;
    let mut state = fit_window(ws, win, win.level, &[], s, FIT_MAX_ITER, FIT_TOL_OBJ);
    while state.em.len() < s.k_max {
        let Some((y, x, a0)) = placement(ws, win, owned, &state, s, l2) else {
            break;
        };
        let mut em = state.em.clone();
        em.push([a0, y, x, s.sigma]);
        let trial = fit_window(ws, win, state.b, &em, s, FIT_MAX_ITER, FIT_TOL_OBJ);
        fits += 1;
        if !(state.i_div - trial.i_div > ADD_NATS) {
            break;
        }
        state = trial;
    }
    // Not at K = 1: that removal is the K = 0 fit FORWARD already beat.
    while state.em.len() > 1 {
        let mut best: Option<Fitted> = None;
        for drop in 0..state.em.len() {
            let em: Vec<Em> = (0..state.em.len())
                .filter(|&j| j != drop)
                .map(|j| state.em[j])
                .collect();
            let reduced = fit_window(ws, win, state.b, &em, s, FIT_MAX_ITER, FIT_TOL_OBJ);
            fits += 1;
            if best.as_ref().is_none_or(|b| reduced.i_div < b.i_div) {
                best = Some(reduced);
            }
        }
        let best = best.expect("K >= 2 has removals");
        if !(best.i_div - state.i_div < ADD_NATS) {
            break;
        }
        state = best;
    }
    (state.em, fits)
}

/// Pixels a box may place on: within `OWN_RADIUS*sigma` of its own nearest
/// candidate, no farther from it than from any other candidate, and in the
/// ROI. Only candidates within that radius of the box can take a pixel from
/// it, so `grid` is asked for those alone.
#[allow(clippy::too_many_arguments)]
fn owned_mask(
    bb: &patches::BBox,
    own: &[u32],
    cand: &[f64],
    grid: &EmitterGrid,
    roi: Option<&[bool]>,
    fw: usize,
    sigma: f64,
    near: &mut Vec<u32>,
) -> Vec<bool> {
    let r = OWN_RADIUS * sigma;
    grid.query_rect(
        bb.y0 as f64,
        bb.x0 as f64,
        (bb.y1 - 1) as f64,
        (bb.x1 - 1) as f64,
        r,
        near,
    );
    near.retain(|i| !own.contains(i));
    let mut owned = Vec::with_capacity(bb.n_pixels());
    for py in bb.y0..bb.y1 {
        for px in bb.x0..bb.x1 {
            let dist = |i: u32| {
                (py as f64 - cand[2 * i as usize]).hypot(px as f64 - cand[2 * i as usize + 1])
            };
            let d_own = own.iter().map(|&i| dist(i)).fold(f64::INFINITY, f64::min);
            let d_oth = near.iter().map(|&i| dist(i)).fold(f64::INFINITY, f64::min);
            let in_roi = roi.is_none_or(|m| m[py * fw + px]);
            owned.push(d_own <= r && d_own <= d_oth && in_roi);
        }
    }
    owned
}

/// Localize one frame of `d_e`. `roi`, if given, is `H*W`: candidates
/// outside it are dropped after the background map is built from all of
/// them, and no box places outside it.
pub fn localize(
    d: &[f64],
    h: usize,
    w: usize,
    roi: Option<&[bool]>,
    s: &Settings,
    ws: &mut Workspace,
) -> Output {
    assert_eq!(d.len(), h * w);
    let b0 = percentile(d, 10.0).max(BG_FLOOR);
    let (cand_all, amp_all, str_all) = find_candidates(d, h, w, b0, s.sigma, s.threshold);
    let bmap = background_map(d, h, w, &cand_all, s.sigma);

    let (mut cand, mut camp, mut strength) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..amp_all.len() {
        let (y, x) = (cand_all[2 * j], cand_all[2 * j + 1]);
        if roi.is_none_or(|m| m[y as usize * w + x as usize]) {
            cand.extend_from_slice(&[y, x]);
            camp.push(amp_all[j]);
            strength.push(str_all[j]);
        }
    }
    let nc = camp.len();
    let mut boxes = patches::build_patches(&cand, nc, s.sigma, h, w, s.k_max);
    // Brightest first, so the strongest light is already fitted when its
    // neighbours read it through their halos. Stable, as `list.sort` is.
    let peak = |p: &patches::Patch| {
        p.indices
            .iter()
            .map(|&i| strength[i as usize])
            .fold(f64::NEG_INFINITY, f64::max)
    };
    boxes.sort_by(|a, b| peak(b).total_cmp(&peak(a)));
    let nb = boxes.len();

    // Once per frame: each box's pixels, ownership and the boxes that can
    // reach it. A held emitter stays inside its own box's fit bounds, within
    // half a pixel of the rectangle, at a width of at most slack.1 * sigma;
    // `_near_rect` then admits it only within HALO_FACTOR widths of this
    // box's pixel rectangle. Boxes farther than that can never contribute.
    let grid = EmitterGrid::build(&cand, nc, h, w, (OWN_RADIUS * s.sigma).max(1.0));
    let mut near = Vec::new();
    let mut wins: Vec<Window> = Vec::with_capacity(nb);
    let mut owned: Vec<Vec<bool>> = Vec::with_capacity(nb);
    for p in &boxes {
        wins.push(Window::new(d, w, &bmap, &p.bbox));
        owned.push(owned_mask(&p.bbox, &p.indices, &cand, &grid, roi, w, s.sigma, &mut near));
    }
    let reach = HALO_FACTOR * s.slack.1.max(1.0) * s.sigma;
    let gap = |a0: f64, a1: f64, b0: f64, b1: f64| (b0 - a1).max(a0 - b1).max(0.0);
    let neighbours: Vec<Vec<usize>> = (0..nb)
        .map(|i| {
            let bi = &boxes[i].bbox;
            (0..nb)
                .filter(|&j| j != i)
                .filter(|&j| {
                    let bj = &boxes[j].bbox;
                    let gy = gap(
                        bi.y0 as f64,
                        (bi.y1 - 1) as f64,
                        bj.y0 as f64 - 0.5,
                        bj.y1 as f64 - 0.5,
                    );
                    let gx = gap(
                        bi.x0 as f64,
                        (bi.x1 - 1) as f64,
                        bj.x0 as f64 - 0.5,
                        bj.x1 as f64 - 0.5,
                    );
                    gy.hypot(gx) <= reach
                })
                .collect()
        })
        .collect();

    // What each box holds, global coords. Until a box is first decided, its
    // candidates stand in for it as in-focus seeds.
    let mut held: Vec<Vec<Em>> = boxes
        .iter()
        .map(|p| {
            p.indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    [camp[i], cand[2 * i], cand[2 * i + 1], s.sigma]
                })
                .collect()
        })
        .collect();

    let l2 = filters::log_kernel_l2(s.sigma);
    let mut search_fits = 0usize;
    let mut ems: Vec<Em> = Vec::new();
    for _ in 0..s.sweeps {
        for i in 0..nb {
            let bb = boxes[i].bbox;
            let (ry0, ry1) = (bb.y0 as f64, (bb.y1 - 1) as f64);
            let (rx0, rx1) = (bb.x0 as f64, (bb.x1 - 1) as f64);
            ems.clear();
            for &j in &neighbours[i] {
                for e in &held[j] {
                    let dy = e[1] - e[1].clamp(ry0, ry1);
                    let dx = e[2] - e[2].clamp(rx0, rx1);
                    if dy.hypot(dx) <= HALO_FACTOR * e[3].max(s.sigma) {
                        ems.push(*e);
                    }
                }
            }
            wins[i].set_halo(&ems, &mut ws.f);
            let (local, fits) = search_box(ws, &wins[i], &owned[i], s, l2);
            search_fits += fits;
            let (y0, x0) = (bb.y0 as f64, bb.x0 as f64);
            held[i] = local.iter().map(|e| [e[0], e[1] + y0, e[2] + x0, e[3]]).collect();
        }
    }

    let all: Vec<Em> = held.into_iter().flatten().collect();
    let n = all.len();
    let mut pos: Vec<f64> = all.iter().flat_map(|e| [e[1], e[2]]).collect();
    let mut amp: Vec<f64> = all.iter().map(|e| e[0]).collect();
    let mut sig: Vec<f64> = all.iter().map(|e| e[3]).collect();
    let mut se = vec![f64::NAN; 3 * n];
    let polish_fits = if s.polish && n > 0 {
        polish(d, h, w, &bmap, &mut pos, &mut amp, &mut sig, &mut se, s, ws)
    } else {
        0
    };
    Output {
        pos,
        amp,
        sig,
        se,
        background: bmap,
        n_candidates: nc,
        n_boxes: nb,
        search_fits,
        polish_fits,
    }
}

/// Block-Jacobi refits at fixed N until nothing moves: `core.refine` with free
/// widths. Every patch in a sweep reads the sweep's INPUT state, and a patch
/// none of whose free or frozen emitters moved last sweep is skipped.
#[allow(clippy::too_many_arguments)]
fn polish(
    d: &[f64],
    h: usize,
    w: usize,
    bmap: &[f64],
    pos: &mut Vec<f64>,
    amp: &mut Vec<f64>,
    sig: &mut Vec<f64>,
    se: &mut [f64],
    s: &Settings,
    ws: &mut Workspace,
) -> usize {
    let n = amp.len();
    let mut dirty = vec![true; n];
    let mut fits = 0usize;
    let mut ems: Vec<Em> = Vec::new();
    let mut var = Vec::new();
    for _ in 0..POLISH_SWEEPS {
        let pset = patches::build_patches(pos, n, s.sigma, h, w, s.k_max);
        let (mut opos, mut oamp, mut osig) = (pos.clone(), amp.clone(), sig.clone());
        let mut moved = vec![false; n];
        let mut n_fitted = 0usize;
        for p in &pset {
            let touched = p.indices.iter().chain(&p.frozen);
            if !touched.into_iter().any(|&i| dirty[i as usize]) {
                continue;
            }
            n_fitted += 1;
            let mut win = Window::new(d, w, bmap, &p.bbox);
            ems.clear();
            ems.extend(p.frozen.iter().map(|&i| {
                let i = i as usize;
                [amp[i], pos[2 * i], pos[2 * i + 1], sig[i]]
            }));
            win.set_halo(&ems, &mut ws.f);
            let (y0, x0) = (p.bbox.y0 as f64, p.bbox.x0 as f64);
            let start: Vec<Em> = p
                .indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    [amp[i], pos[2 * i] - y0, pos[2 * i + 1] - x0, sig[i]]
                })
                .collect();
            let r = fit_window(ws, &win, win.level, &start, s, POLISH_MAX_ITER, POLISH_TOL_OBJ);
            fits += 1;
            for (j, &i) in p.indices.iter().enumerate() {
                let i = i as usize;
                let e = r.em[j];
                oamp[i] = e[0];
                opos[2 * i] = e[1] + y0;
                opos[2 * i + 1] = e[2] + x0;
                osig[i] = e[3];
                moved[i] = (opos[2 * i] - pos[2 * i]).hypot(opos[2 * i + 1] - pos[2 * i + 1])
                    > POLISH_MOVE_TOL;
            }
            // SEs from the Fisher matrix of the fit whose parameters are
            // reported. An indefinite matrix leaves the previous SEs.
            let pdim = 4 * p.indices.len() + 1;
            ws.chol.ensure(pdim);
            if ws.chol.factor(ws.fit.fisher(pdim), pdim) {
                var.resize(pdim, 0.0);
                ws.chol.inv_diag(&mut var, &mut ws.scratch);
                for (j, &i) in p.indices.iter().enumerate() {
                    for c in 0..3 {
                        let v = var[1 + 4 * j + c];
                        se[3 * i as usize + c] = if v > 0.0 { v.sqrt() } else { f64::NAN };
                    }
                }
            }
        }
        *pos = opos;
        *amp = oamp;
        *sig = osig;
        if n_fitted == 0 || !moved.iter().any(|&m| m) {
            break;
        }
        dirty = moved;
    }
    fits
}

/// Background plus every emitter at its own width, each rendered within
/// `truncate` of its own sigma. `calibrate.render_model`.
pub fn render_var(
    pos: &[f64],
    amp: &[f64],
    sig: &[f64],
    h: usize,
    w: usize,
    background: &[f64],
    truncate: f64,
) -> Vec<f64> {
    let mut m = background.to_vec();
    let mut f = psf::Factors::new(1, 1, 1);
    let mut sub = Vec::new();
    for k in 0..amp.len() {
        let (cy, cx, sk) = (pos[2 * k], pos[2 * k + 1], sig[k]);
        let rad = (truncate * sk).ceil() as i64;
        let y0 = (cy.floor() as i64 - rad).max(0) as usize;
        let y1 = ((cy.ceil() as i64 + rad + 1).max(0) as usize).min(h);
        let x0 = (cx.floor() as i64 - rad).max(0) as usize;
        let x1 = ((cx.ceil() as i64 + rad + 1).max(0) as usize).min(w);
        if y1 <= y0 || x1 <= x0 {
            continue;
        }
        let ay: Vec<f64> = (y0..y1).map(|v| v as f64).collect();
        let ax: Vec<f64> = (x0..x1).map(|v| v as f64).collect();
        sub.clear();
        sub.resize(ay.len() * ax.len(), 0.0);
        f.ensure(ay.len(), ax.len(), 1);
        psf::model_ax(&[0.0, amp[k], cy, cx], &ay, &ax, sk, None, &mut f, &mut sub);
        for r in y0..y1 {
            for c in x0..x1 {
                m[r * w + c] += sub[(r - y0) * (x1 - x0) + (c - x0)];
            }
        }
    }
    m
}

/// Localize every frame of a stack on `n_threads` workers. Frames are
/// independent, so each worker takes the next undone frame and keeps its own
/// [`Workspace`]; the output is in frame order whatever the scheduling.
///
/// `raw` is `n*H*W`; frame `t` becomes `(raw - offset) / gain[t] + shift`.
#[allow(clippy::too_many_arguments)]
pub fn localize_stack(
    raw: &[f64],
    n: usize,
    h: usize,
    w: usize,
    offset: f64,
    gain: &[f64],
    shift: f64,
    roi: Option<&[bool]>,
    s: &Settings,
    n_threads: usize,
) -> Vec<Output> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    assert_eq!(raw.len(), n * h * w);
    assert_eq!(gain.len(), n);
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Option<Output>>> = Mutex::new((0..n).map(|_| None).collect());
    let workers = n_threads.clamp(1, n.max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let mut ws = Workspace::new();
                let mut d = vec![0.0; h * w];
                loop {
                    let t = next.fetch_add(1, Ordering::Relaxed);
                    if t >= n {
                        break;
                    }
                    let frame = &raw[t * h * w..(t + 1) * h * w];
                    for (v, &r) in d.iter_mut().zip(frame) {
                        *v = (r - offset) / gain[t] + shift;
                    }
                    let o = localize(&d, h, w, roi, s, &mut ws);
                    out.lock().expect("no worker panics while holding it")[t] = Some(o);
                }
            });
        }
    });
    out.into_inner()
        .expect("workers joined")
        .into_iter()
        .map(|o| o.expect("every frame was taken"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_threshold_matches_the_reference_values() {
        // `calibrate.seed_threshold`'s docstring: 3.70 on 64^2 at sigma
        // 0.818, 4.43 on 512^2 at sigma 1.45.
        assert!((seed_threshold(64, 64, 0.818, SEED_ALPHA) - 3.70).abs() < 5e-3);
        assert!((seed_threshold(512, 512, 1.45, SEED_ALPHA) - 4.43).abs() < 5e-3);
    }

    #[test]
    fn percentile_interpolates_as_numpy_does() {
        let v = [4.0, 1.0, 3.0, 2.0, 10.0];
        // np.percentile([1, 2, 3, 4, 10], q) for q = 0, 10, 50, 90, 100.
        for (q, want) in [(0.0, 1.0), (10.0, 1.4), (50.0, 3.0), (90.0, 7.6), (100.0, 10.0)] {
            assert!((percentile(&v, q) - want).abs() < 1e-12, "q={q}");
        }
    }

    #[test]
    fn an_isolated_emitter_is_found_once_where_it_is() {
        let (h, w, sigma) = (31, 33, 1.2);
        let theta = [0.0, 1500.0, 14.3, 17.6, 1.3];
        let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
        let mut f = psf::Factors::new(h, w, 1);
        let mut d = vec![0.0; h * w];
        psf::model_var_sigma_ax(&theta, &ay, &ax, None, &mut f, &mut d);
        for v in d.iter_mut() {
            *v += 10.0;
        }
        let s = Settings {
            sigma,
            k_max: 12,
            threshold: seed_threshold(h, w, sigma, SEED_ALPHA),
            slack: (0.7, 2.2),
            sweeps: SWEEPS,
            polish: true,
        };
        let o = localize(&d, h, w, None, &s, &mut Workspace::new());
        assert_eq!(o.amp.len(), 1, "found {:?}", o.pos);
        assert!(o.se.iter().all(|v| v.is_finite() && *v > 0.0));
        // Noise-free, so what is left is the background map's own bias: the
        // mask sits on the integer candidate, and the wing beyond it leaks
        // into the 25 px mean. Measured: 0.05 SE in position, 0.2 in flux.
        let (se_a, se_y, se_x) = (o.se[0], o.se[1], o.se[2]);
        assert!((o.pos[0] - 14.3).abs() < 0.1 * se_y, "y {}", o.pos[0]);
        assert!((o.pos[1] - 17.6).abs() < 0.1 * se_x, "x {}", o.pos[1]);
        assert!((o.amp[0] - 1500.0).abs() < 0.5 * se_a, "A {}", o.amp[0]);
        assert!((o.sig[0] - 1.3).abs() < 0.01, "sigma {}", o.sig[0]);
    }
}
