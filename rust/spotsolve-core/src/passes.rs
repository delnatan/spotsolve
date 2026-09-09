//! The four passes: ADD, SPLIT, REFINE and PRUNE.
//!
//! Ports `core.py`'s `_try_add`/`_try_split`/`_split_pass`/`refine`/`_prune`
//! and the helpers underneath them. **`detect`'s round loop is not here**: it
//! stays in `core.py`, along with `find_candidates` and `background_map`'s
//! convolutions, and calls into these ~35 times per frame.
//!
//! # Where model selection happens
//!
//! In exactly three places: [`try_add`], [`try_split`] and [`prune`].
//! [`refine`] proposes nothing and is the only place the reported parameters
//! and standard errors are produced.
//!
//! # Why the loop terminates
//!
//! By construction, not by a tolerance. ADD and SPLIT both take `K -> K+1`, so
//! `N` is monotone increasing during the round loop, and every accepted emitter
//! lowers the residual that produces the candidates, so the candidate list must
//! eventually empty. PRUNE only removes, so `N` is monotone decreasing during
//! the settle loop. The two loops do not alternate, so no move can undo
//! another. There is no fixed point to chase.
//!
//! # The one guard that must not creep back in
//!
//! Only `log BF <= 0` and `COND_GUARD` may refuse an addition [P12]. A
//! significance screen on `A/SE` applied to a *proposal* fit is a detection
//! rule whatever its docstring says, and it deciding what exists on less
//! information than the joint refit that follows is what cost 68-79% of every
//! true emitter lost inside 2 sigma. Degenerate configurations are removed by
//! [`prune`], afterwards, on a joint fit. The type split below is there to keep
//! that distinction visible: a [`Validity`] may only annotate an
//! already-computed evidence; only a [`Decision`] may reject.

use crate::evidence::{COND_GUARD, Evidence, Prior};
use crate::grid::EmitterGrid;
use crate::lmcl::{self, Bounds, FitOpts, FitWorkspace};
use crate::moves;
use crate::patches::{self, BBox, HALO_FACTOR};
use crate::psf;
use crate::render;

/// `A / SE(A)` on a joint fit, below which [`prune`] removes an emitter
/// outright instead of scoring it.
///
/// **The pipeline's precision/recall dial, and the only one left.** It has
/// three roles, and the second two are why it cannot simply be dropped:
///
/// * it forces removal where the Laplace evidence cannot be computed. A
///   collapsed pair's Fisher matrix is singular along its separation direction,
///   so `var(A)` diverges and `A/SE` goes to zero on its own -- and the Bayes
///   factor there would argue to **keep** the pair, harder the more degenerate
///   it is;
/// * it keeps the Laplace approximation inside its domain of validity. The
///   Laplace form integrates the added dimensions against an *unbounded*
///   Gaussian of width `SE(A)` while the true posterior is truncated at
///   `A >= 0`, so when the mode sits within a few SE of that boundary the
///   posterior volume, hence the evidence, is overstated.
///
/// Calibrated against exact 4-D numerical integration of the same posterior,
/// the approximation is trustworthy from about 3 SE outward. Validity alone
/// would argue for 3.0; raising it that far costs localization as well as
/// recall, because removing one member of a real close pair leaves the survivor
/// absorbing both fluxes and sitting between them. 2.0 is the measured optimum
/// on both bead frames.
pub const PRUNE_TAU: f64 = 2.0;

/// Amplitude floor for a fit, as a fraction of the window's own `A_max`.
///
/// **Relative, because what it protects is a ratio.** An emitter's position
/// block of the Fisher matrix scales as `A^2`, so at bead fluxes of ~2000 e- an
/// absolute amplitude of 1e-4 puts those entries at ~5.7e-12 against a largest
/// diagonal of ~768 -- a ratio of 3e-14, about 130x f64 epsilon. At that point
/// `log|F|` is numerical noise and the Occam term of every Bayes factor built
/// on it is noise with it.
///
/// This cannot be enforced downstream in the linear-algebra layer instead: no
/// test on `F` alone distinguishes an uninformed parameter from a well-posed
/// matrix in badly scaled units, because both give a huge raw condition number
/// and a small scaled one. Here the flux scale is known, so it can [P11].
pub const A_MIN_REL: f64 = 1e-6;

pub const BG_FLOOR: f64 = 1e-3;
pub const REFINE_TOL: f64 = 1e-3;

/// A statement about whether the Laplace approximation can be *computed*.
///
/// Deliberately inert: it may annotate an already-computed evidence and nothing
/// more. It has no method that returns a rejection, because the moment such a
/// thing exists someone will call it from the search path, and then it is a
/// detection rule [P12].
#[derive(Clone, Copy, Debug)]
pub struct Validity {
    pub amplitude_resolved: bool,
}

/// A verdict that may actually refuse a move. Only these two reasons exist on
/// the add path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    Accept,
    /// The evidence did not favour the larger model.
    EvidenceAgainst,
    /// `F` is too ill-conditioned for the Occam term to mean anything, so it is
    /// not weighed against anything.
    IllConditioned,
}

/// The committed emitter set: positions `(N, 2)` row-major `(y, x)`, and
/// amplitudes as total flux in photoelectrons.
#[derive(Clone, Debug, Default)]
pub struct Emitters {
    pub pos: Vec<f64>,
    pub amp: Vec<f64>,
}

impl Emitters {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn from_parts(pos: Vec<f64>, amp: Vec<f64>) -> Self {
        assert_eq!(pos.len(), 2 * amp.len());
        Self { pos, amp }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.amp.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.amp.is_empty()
    }
    #[inline]
    pub fn y(&self, i: usize) -> f64 {
        self.pos[2 * i]
    }
    #[inline]
    pub fn x(&self, i: usize) -> f64 {
        self.pos[2 * i + 1]
    }
    pub fn push(&mut self, y: f64, x: f64, a: f64) {
        self.pos.push(y);
        self.pos.push(x);
        self.amp.push(a);
    }
}

/// Reusable storage for a whole pass.
pub struct Solver {
    /// Two fit workspaces, because every model-selection move holds an
    /// incumbent and a proposal at the same time and must not refit either.
    fit_inc: FitWorkspace,
    fit_prop: FitWorkspace,
    ev: Evidence,
    sub: Vec<f64>,
    halo: Vec<f64>,
    theta: Vec<f64>,
    theta_best: Vec<f64>,
    f_inc: Vec<f64>,
    resid: Vec<f64>,
    scratch_u32: Vec<u32>,
    scratch_f64: Vec<f64>,
    /// Pre-fit positions of the group being fitted, `(y, x)` per slot.
    pre: Vec<f64>,
    inv_diag: Vec<f64>,
    inv_scratch: Vec<f64>,
    bounds_lo: Vec<f64>,
    bounds_hi: Vec<f64>,
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver {
    pub fn new() -> Self {
        Self {
            fit_inc: FitWorkspace::new(),
            fit_prop: FitWorkspace::new(),
            ev: Evidence::new(),
            sub: Vec::new(),
            halo: Vec::new(),
            theta: Vec::new(),
            theta_best: Vec::new(),
            f_inc: Vec::new(),
            resid: Vec::new(),
            scratch_u32: Vec::new(),
            scratch_f64: Vec::new(),
            pre: Vec::new(),
            inv_diag: Vec::new(),
            inv_scratch: Vec::new(),
            bounds_lo: Vec::new(),
            bounds_hi: Vec::new(),
        }
    }
}

/// Everything a pass needs about the image it is working on.
pub struct Frame<'a> {
    /// Data in PHOTOELECTRONS, `d_e = (raw - offset) / gain`, row-major.
    ///
    /// The Poisson weighting `W = 1/m` used by the optimizer and by every
    /// Fisher matrix is only valid in these units; in ADU the variance is
    /// `gain * mean` and every standard error, Bayes factor and condition
    /// number is wrong by a factor of the gain.
    pub d_e: &'a [f64],
    /// The background SURFACE, `(H, W)`, in photoelectrons per pixel.
    pub bmap: &'a [f64],
    pub h: usize,
    pub w: usize,
    pub sigma: f64,
    pub k_max: usize,
}

impl Frame<'_> {
    fn cut(&self, b: &BBox, out: &mut Vec<f64>) {
        out.clear();
        for r in b.y0..b.y1 {
            out.extend_from_slice(&self.d_e[r * self.w + b.x0..r * self.w + b.x1]);
        }
    }
}

/// Split a window's background into a `level` and a `shape`.
///
/// The fit keeps its free scalar `b`, started at `level`; `shape` is the
/// background's variation across the window and enters as a known additive
/// term, like a frozen emitter.
///
/// **The split exists to keep `b` strictly interior.** Folding the whole
/// surface into the known term would leave `b` wanting to sit at 0, which is
/// its lower bound, and that is the one thing Coleman-Li scaling cannot
/// tolerate [P1].
fn window_bg(
    bmap: &[f64],
    w_img: usize,
    b: &BBox,
    shape: &mut Vec<f64>,
    buf: &mut Vec<f64>,
) -> f64 {
    // `shape` is filled in RASTER order and `buf` is the scratch the median
    // sorts. They must be separate: `median` reorders what it is given, and
    // `shape` is indexed by pixel.
    shape.clear();
    for r in b.y0..b.y1 {
        shape.extend_from_slice(&bmap[r * w_img + b.x0..r * w_img + b.x1]);
    }
    buf.clear();
    buf.extend_from_slice(shape);
    let level = median(buf);
    for v in shape.iter_mut() {
        *v -= level;
    }
    level
}

/// `np.median`: the middle element, or the mean of the two middle ones.
/// **Reorders the slice it is given.**
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Box constraints for one window's fit, in LOCAL coordinates.
///
/// Positions are confined to the sub-image. Letting a centre leave the frame
/// was tried -- it lets the fit put rim flux where it actually came from -- and
/// measurably lost 10 real detections elsewhere, so the bounds stay closed.
fn set_bounds(
    lo: &mut Vec<f64>,
    hi: &mut Vec<f64>,
    k: usize,
    h: usize,
    w: usize,
    b_max: f64,
    a_max: f64,
) {
    let a_min = moves::A_MIN.max(A_MIN_REL * a_max);
    lo.clear();
    hi.clear();
    lo.push(0.0);
    hi.push(b_max);
    for _ in 0..k {
        lo.extend_from_slice(&[a_min, -0.5, -0.5]);
        hi.extend_from_slice(&[a_max, h as f64 - 0.5, w as f64 - 0.5]);
    }
}

/// Fit one window, deriving its bounds from the window's own flux scale.
#[allow(clippy::too_many_arguments)]
fn fit_window(
    ws: &mut FitWorkspace,
    lo: &mut Vec<f64>,
    hi: &mut Vec<f64>,
    theta0: &mut Vec<f64>,
    sub: &[f64],
    h: usize,
    w: usize,
    sigma: f64,
    halo: Option<&[f64]>,
    max_iter: usize,
    tol_obj: f64,
) -> lmcl::FitInfo {
    let k = psf::n_emitters(theta0);
    let smax = sub.iter().fold(1.0f64, |a, &v| a.max(v));
    let b_max = (smax * 4.0).max(10.0);
    let a_max = 8.0 * smax / psf::peak_factor(sigma);
    set_bounds(lo, hi, k, h, w, b_max, a_max);
    for q in 0..theta0.len() {
        theta0[q] = theta0[q].clamp(lo[q] + 1e-9, hi[q] - 1e-9);
    }
    let bounds = Bounds::new(lo, hi);
    lmcl::fit(
        ws,
        theta0,
        h,
        w,
        sigma,
        sub,
        &bounds,
        halo,
        FitOpts {
            max_iter,
            tol_obj,
            ..Default::default()
        },
    )
}

/// Build a window's known additive term: the background's shape plus every
/// frozen neighbour's rendered PSF.
fn build_halo(
    frame: &Frame,
    em: &Emitters,
    frozen: &[u32],
    b: &BBox,
    halo: &mut Vec<f64>,
    buf: &mut Vec<f64>,
    shape: &mut Vec<f64>,
) -> f64 {
    let level = window_bg(frame.bmap, frame.w, b, shape, buf);
    render::halo_image(
        &em.pos,
        &em.amp,
        frozen,
        frame.sigma,
        b.y0,
        b.x0,
        b.h(),
        b.w(),
        halo,
    );
    for (v, &s) in halo.iter_mut().zip(shape.iter()) {
        *v += s;
    }
    level
}

fn pack_window(theta: &mut Vec<f64>, level: f64, em: &Emitters, idx: &[u32], b: &BBox) {
    theta.clear();
    theta.push(level);
    for &i in idx {
        theta.push(em.amp[i as usize]);
        theta.push(em.y(i as usize) - b.y0 as f64);
        theta.push(em.x(i as usize) - b.x0 as f64);
    }
}

fn sum_amplitudes(theta: &[f64]) -> f64 {
    (0..psf::n_emitters(theta))
        .map(|k| psf::amp(theta, k))
        .sum()
}

/// Score adding one emitter at `(cand_y, cand_x)` with initial amplitude
/// `cand_amp`.
///
/// Both models are fitted on the **same pixels with the same frozen halo** and
/// differ only by the one emitter. That is what makes their I-divergences
/// differencable into a Bayes factor; reusing a `K` fit from a different window
/// would silently break it.
///
/// On acceptance the **whole window's** refitted parameters are written back,
/// not just the new emitter's -- adding a source shifts its neighbours, and
/// keeping their stale values would leave the model worse than the fit that
/// justified the acceptance.
#[allow(clippy::too_many_arguments)]
pub fn try_add(
    s: &mut Solver,
    frame: &Frame,
    em: &mut Emitters,
    gridx: &mut EmitterGrid,
    cand_y: f64,
    cand_x: f64,
    cand_amp: f64,
    prior: Prior,
) -> Decision {
    let n = em.len();
    let (free, frozen, bbox) = patches::window(
        &em.pos,
        n,
        cand_y,
        cand_x,
        frame.sigma,
        frame.h,
        frame.w,
        frame.k_max,
        gridx,
        &mut s.scratch_u32,
    );
    frame.cut(&bbox, &mut s.sub);
    let (h, w) = (bbox.h(), bbox.w());
    let mut shape = std::mem::take(&mut s.scratch_f64);
    let level = build_halo(
        frame,
        em,
        &frozen,
        &bbox,
        &mut s.halo,
        &mut s.resid,
        &mut shape,
    );
    s.scratch_f64 = shape;

    // Incumbent, at K.
    pack_window(&mut s.theta, level, em, &free, &bbox);
    let inc = fit_window(
        &mut s.fit_inc,
        &mut s.bounds_lo,
        &mut s.bounds_hi,
        &mut s.theta,
        &s.sub,
        h,
        w,
        frame.sigma,
        Some(&s.halo),
        100,
        EVIDENCE_TOL_OBJ,
    );
    let p_inc = 3 * free.len() + 1;
    let sum_a_inc = sum_amplitudes(s.fit_inc.theta());
    s.f_inc.clear();
    s.f_inc.extend_from_slice(s.fit_inc.fisher(p_inc));

    // Proposal, at K+1: the same window, plus the candidate.
    pack_window(&mut s.theta, level, em, &free, &bbox);
    s.theta.push(cand_amp);
    s.theta.push(cand_y - bbox.y0 as f64);
    s.theta.push(cand_x - bbox.x0 as f64);
    let prop = fit_window(
        &mut s.fit_prop,
        &mut s.bounds_lo,
        &mut s.bounds_hi,
        &mut s.theta,
        &s.sub,
        h,
        w,
        frame.sigma,
        Some(&s.halo),
        100,
        EVIDENCE_TOL_OBJ,
    );
    let p_prop = p_inc + 3;

    let (log_bf, cond) = s.ev.log_bf_add(
        inc.i_div,
        prop.i_div,
        &s.f_inc,
        s.fit_prop.fisher(p_prop),
        p_inc,
        p_prop,
        sum_a_inc,
        sum_amplitudes(s.fit_prop.theta()),
        free.len(),
        prior,
        None,
    );
    if !log_bf.is_finite() || log_bf <= 0.0 {
        return Decision::EvidenceAgainst;
    }
    if cond > COND_GUARD {
        return Decision::IllConditioned;
    }

    write_back(em, gridx, s.fit_prop.theta(), &free, &bbox, true);
    Decision::Accept
}

/// Write a window's refit into the committed set, keeping the spatial index
/// current.
///
/// `theta`'s emitters correspond to `free` in order; if `append` is set, the
/// last one is a new emitter rather than an update.
fn write_back(
    em: &mut Emitters,
    gridx: &mut EmitterGrid,
    theta: &[f64],
    free: &[u32],
    b: &BBox,
    append: bool,
) {
    let k = psf::n_emitters(theta);
    debug_assert_eq!(k, free.len() + usize::from(append));
    for (slot, &i) in free.iter().enumerate() {
        let i = i as usize;
        let (oy, ox) = (em.y(i), em.x(i));
        let ny = psf::cy(theta, slot) + b.y0 as f64;
        let nx = psf::cx(theta, slot) + b.x0 as f64;
        em.pos[2 * i] = ny;
        em.pos[2 * i + 1] = nx;
        em.amp[i] = psf::amp(theta, slot);
        gridx.relocate(i as u32, oy, ox, ny, nx);
    }
    if append {
        let last = k - 1;
        let ny = psf::cy(theta, last) + b.y0 as f64;
        let nx = psf::cx(theta, last) + b.x0 as f64;
        gridx.insert(em.len() as u32, ny, nx);
        em.push(ny, nx, psf::amp(theta, last));
    }
}

/// One ADD pass over a candidate list, brightest first.
///
/// Returns the number accepted. The candidates come from `find_candidates` in
/// Python; the **proximity re-check** is here rather than there because it
/// reads the positions an earlier acceptance in *this* pass has already
/// written.
pub fn add_pass(
    s: &mut Solver,
    frame: &Frame,
    em: &mut Emitters,
    cand: &[f64],
    cand_amp: &[f64],
    prior: Prior,
) -> usize {
    let mut gridx = EmitterGrid::build(
        &em.pos,
        em.len(),
        frame.h,
        frame.w,
        HALO_FACTOR * frame.sigma,
    );
    let mut n_added = 0;
    for c in 0..cand_amp.len() {
        let (cy, cx) = (cand[2 * c], cand[2 * c + 1]);
        if em.len() > 0 {
            // An earlier acceptance in this round may have claimed this
            // candidate's flux already.
            gridx.query_disc(&em.pos, cy, cx, frame.sigma, &mut s.scratch_u32);
            if !s.scratch_u32.is_empty() {
                continue;
            }
        }
        if try_add(s, frame, em, &mut gridx, cy, cx, cand_amp[c], prior) == Decision::Accept {
            n_added += 1;
        }
    }
    n_added
}

/// Score replacing emitter `gi` with two, along the residual quadrupole axis.
///
/// The move FIND structurally cannot make: two emitters closer than about 1.5
/// sigma are fitted well by one brighter PSF, so their residual has no *peak*
/// for a LoG filter to find. Takes `K -> K+1` like [`try_add`], so the outer
/// loop stays monotone in `N`.
///
/// Both displacements in [`moves::SPLIT_DISPS`] are proposed and the higher
/// scoring one kept. The incumbent's log-determinant is computed **once** and
/// reused across them: one incumbent, several proposals, and passing it in is
/// exact rather than an approximation [P5].
pub fn try_split(
    s: &mut Solver,
    frame: &Frame,
    em: &mut Emitters,
    gridx: &mut EmitterGrid,
    gi: usize,
    prior: Prior,
) -> Decision {
    let n = em.len();
    let (free, frozen, bbox) = patches::window(
        &em.pos,
        n,
        em.y(gi),
        em.x(gi),
        frame.sigma,
        frame.h,
        frame.w,
        frame.k_max,
        gridx,
        &mut s.scratch_u32,
    );
    // The emitter being split must be free in its own window; if the k_max cap
    // pushed it out, there is nothing to propose.
    let Some(lk) = free.iter().position(|&i| i as usize == gi) else {
        return Decision::EvidenceAgainst;
    };

    frame.cut(&bbox, &mut s.sub);
    let (h, w) = (bbox.h(), bbox.w());
    let mut shape = std::mem::take(&mut s.scratch_f64);
    let level = build_halo(
        frame,
        em,
        &frozen,
        &bbox,
        &mut s.halo,
        &mut s.resid,
        &mut shape,
    );
    s.scratch_f64 = shape;

    pack_window(&mut s.theta, level, em, &free, &bbox);
    let inc = fit_window(
        &mut s.fit_inc,
        &mut s.bounds_lo,
        &mut s.bounds_hi,
        &mut s.theta,
        &s.sub,
        h,
        w,
        frame.sigma,
        Some(&s.halo),
        100,
        EVIDENCE_TOL_OBJ,
    );
    let p_inc = 3 * free.len() + 1;
    let sum_a_inc = sum_amplitudes(s.fit_inc.theta());
    s.f_inc.clear();
    s.f_inc.extend_from_slice(s.fit_inc.fisher(p_inc));
    s.theta.clear();
    s.theta.extend_from_slice(s.fit_inc.theta());
    let ld_inc = s.ev.logdet(&s.f_inc, p_inc);

    // The incumbent's residual, in the window's local frame.
    let mut model = std::mem::take(&mut s.scratch_f64);
    model.clear();
    model.resize(h * w, 0.0);
    {
        let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
        let mut f = psf::Factors::new(h, w, free.len().max(1));
        psf::model_ax(
            &s.theta,
            &ay,
            &ax,
            frame.sigma,
            Some(&s.halo),
            &mut f,
            &mut model,
        );
    }
    s.resid.clear();
    s.resid
        .extend(s.sub.iter().zip(model.iter()).map(|(&d, &m)| d - m));
    s.scratch_f64 = model;

    let (u, _) = moves::residual_axis(
        psf::cy(&s.theta, lk),
        psf::cx(&s.theta, lk),
        psf::amp(&s.theta, lk),
        0.0,
        0.0,
        h,
        w,
        frame.sigma,
        &s.resid,
    );

    let p_prop = p_inc + 3;
    let mut best: Option<f64> = None;
    for disp in moves::SPLIT_DISPS {
        let mut th = moves::split(&s.theta, lk, u, disp * frame.sigma);
        let prop = fit_window(
            &mut s.fit_prop,
            &mut s.bounds_lo,
            &mut s.bounds_hi,
            &mut th,
            &s.sub,
            h,
            w,
            frame.sigma,
            Some(&s.halo),
            100,
            EVIDENCE_TOL_OBJ,
        );
        let (log_bf, cond) = s.ev.log_bf_add(
            inc.i_div,
            prop.i_div,
            &s.f_inc,
            s.fit_prop.fisher(p_prop),
            p_inc,
            p_prop,
            sum_a_inc,
            sum_amplitudes(s.fit_prop.theta()),
            free.len(),
            prior,
            Some(ld_inc),
        );
        if log_bf.is_finite()
            && log_bf > 0.0
            && cond <= COND_GUARD
            && best.is_none_or(|b| log_bf > b)
        {
            best = Some(log_bf);
            s.theta_best.clear();
            s.theta_best.extend_from_slice(s.fit_prop.theta());
        }
    }
    if best.is_none() {
        return Decision::EvidenceAgainst;
    }

    // `moves::split` keeps the untouched emitters in order and appends the two
    // children, so the fitted vector is [free without gi] + [child0, child1].
    // The parent's slot takes child0 and child1 is appended.
    let keep: Vec<u32> = free.iter().copied().filter(|&i| i as usize != gi).collect();
    let theta = std::mem::take(&mut s.theta_best);
    let mut targets = keep.clone();
    targets.push(gi as u32);
    write_back(em, gridx, &theta, &targets, &bbox, true);
    s.theta_best = theta;
    Decision::Accept
}

/// Propose a split for every emitter, most pair-like first.
///
/// **The ranking is not an optimization detail.** A split accepted early
/// changes its neighbours, so the order decides which configuration the later
/// proposals are scored against.
///
/// `model` is the current full model image, rendered by the caller.
pub fn split_pass(
    s: &mut Solver,
    frame: &Frame,
    em: &mut Emitters,
    model: &[f64],
    prior: Prior,
) -> usize {
    let n = em.len();
    if n == 0 {
        return 0;
    }
    let pad = (patches::BBOX_PAD * frame.sigma).ceil() as i64;
    let mut strengths: Vec<(f64, u32)> = Vec::with_capacity(n);
    let mut win = Vec::new();
    for i in 0..n {
        let (cy, cx) = (em.y(i), em.x(i));
        let y0 = ((cy as i64) - pad).max(0) as usize;
        let x0 = ((cx as i64) - pad).max(0) as usize;
        let y1 = ((((cy as i64) + pad + 1).max(0)) as usize).min(frame.h);
        let x1 = ((((cx as i64) + pad + 1).max(0)) as usize).min(frame.w);
        win.clear();
        for r in y0..y1 {
            win.extend((x0..x1).map(|c| frame.d_e[r * frame.w + c] - model[r * frame.w + c]));
        }
        // Scored against a bare emitter with no background, exactly as the
        // Python does: the quadrupole is a shape statistic, and the pedestal
        // under it is common to both lobes.
        let (_, strength) = moves::residual_axis(
            cy,
            cx,
            em.amp[i],
            y0 as f64,
            x0 as f64,
            y1 - y0,
            x1 - x0,
            frame.sigma,
            &win,
        );
        strengths.push((strength, i as u32));
    }
    // Descending strength, index as an explicit tiebreak.
    strengths.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let mut gridx = EmitterGrid::build(
        &em.pos,
        em.len(),
        frame.h,
        frame.w,
        HALO_FACTOR * frame.sigma,
    );
    let mut n_split = 0;
    for &(strength, gi) in &strengths {
        if strength <= 0.0 {
            break;
        }
        if try_split(s, frame, em, &mut gridx, gi as usize, prior) == Decision::Accept {
            n_split += 1;
        }
    }
    n_split
}

/// One Gauss-Seidel pass over the patch decomposition.
///
/// Patches within a sweep are **independent**: each builds its frozen halo from
/// the sweep's input state, not from its neighbours' updates. That is what
/// makes the sweep parallelizable, and why patch order does not affect the
/// result [P14].
/// Objective tolerance for a fit whose I-divergence is differenced into a log
/// Bayes factor: ADD, SPLIT and PRUNE.
///
/// Tight on purpose. A proposal fit starts further from its optimum than the
/// incumbent it is compared against, so a loose tolerance does not add
/// symmetric noise -- it leaves the proposal's objective systematically too
/// high and biases model selection toward the smaller model [P10]. The margins
/// this must resolve are real: the closest prune decision measured on a
/// 906-emitter frame sat at `log BF = +0.022` [P19].
pub const EVIDENCE_TOL_OBJ: f64 = 1e-8;

/// Objective tolerance for REFINE.
///
/// No Bayes factor is built from a REFINE fit, so the asymmetry argument above
/// does not apply to it directly, and its own natural unit is not nats but
/// **pixels**: near the optimum `I(t) ~ I_min + 0.5 dt' F dt`, so stopping at a
/// predicted decrease of `tol` leaves a parameter about `sqrt(2*tol)` standard
/// errors short. `examples/refine_tol.rs` measures that this holds -- for the
/// bulk. It does **not** hold for the worst emitter: at 1e-3 a patch with a
/// nearly singular Fisher matrix moved 3.6 SE, because a near-singular `F` is
/// exactly where the nats-to-pixels conversion diverges [P11].
///
/// REFINE also does not stand alone. Its output is what PRUNE re-tests and what
/// the next round's candidates are scored against, so a sloppier REFINE
/// eventually moves **detections**. That is what fixes the value here
/// (256 px, N=3169, Python-vs-Rust agreement in units of the reported SE):
///
/// | `tol_obj` | frame s | N | max `|dpos|/SE` |
/// |---|---|---|---|
/// | 1e-8 | 9.78 | 3169 | 0.0010 |
/// | 1e-6 | 8.66 | 3169 | 0.0015 |
/// | 1e-5 | 7.68 | 3168 | 0.1466 |
/// | 1e-4 | 7.08 | 3161 | 0.2828 |
///
/// 1e-6 is the last value leaving `N` and the agreement band where 1e-8 puts
/// them, and it still removes ~11% of frame time and ~40% of REFINE's LM
/// iterations. Below it both move together -- the signature of a numerical
/// tolerance that has started making decisions [P12].
pub const REFINE_TOL_OBJ: f64 = 1e-6;

/// LM iterations one REFINE group fit may take. A budget, not a convergence
/// criterion. See `core.py`'s `REFINE_MAX_ITER` for the measured table; the
/// short version is that a few groups per frame descend a direction carrying
/// almost no information, they cost 10-19% of the pass, their emitters are not
/// disposable, and the truncated tail is worth ~0.002 SE of position.
///
/// The evidence fits keep 100: they start further from their optimum and their
/// objective is differenced into a Bayes factor, so truncating them biases
/// model selection [P10]. This is for the polish step only.
pub const REFINE_MAX_ITER: usize = 50;

/// Whether REFINE's sweep reads neighbours updated earlier in the same pass.
///
/// Jacobi (`false`) is order-independent and parallelizable [P14]; Gauss-Seidel
/// (`true`) converges faster and is not. See `refine_sweep`.
pub const REFINE_GAUSS_SEIDEL: bool = false;

/// What one patch fit cost, and whether it needed to run at all.
///
/// Exposed so the claims in `PORTING_NOTES` about REFINE's schedule stay
/// checkable outside a profiler -- see `examples/refine_cost.rs`. Collecting it
/// costs one branch per patch, never per pixel.
#[derive(Clone, Copy, Debug)]
pub struct PatchCost {
    /// LM iterations this fit took.
    pub n_iter: usize,
    /// The fit genuinely converged, as opposed to merely stopping.
    pub converged: bool,
    /// Largest position change this fit made, in px.
    pub moved: f64,
    /// Every value this fit reads -- its own emitters and its frozen halo --
    /// was **bit-identical** to the previous sweep, so the fit was guaranteed
    /// to reproduce its own output and skipping it would have been exact.
    /// Anything looser than bit-identity is an approximation, not a memo.
    pub reproducible: bool,
    /// Scaled condition number of this fit's Fisher matrix, or `+inf` if it
    /// could not be factorized. This is the quantity that says whether the
    /// configuration carries information in every direction; it is computed
    /// only when costs are being recorded, never in the production path.
    pub scaled_cond: f64,
    /// Closest pair inside the group, in px, or `+inf` for a single emitter.
    pub min_pair_px: f64,
    /// Emitters in the group.
    pub k: usize,
    /// Smallest distance from any fitted parameter to its nearer bound, as a
    /// fraction of that parameter's own range. A parameter converging ONTO a
    /// bound is the classic cause of a Coleman-Li crawl [P1]: the affine
    /// scaling divides by the distance to the bound the step heads toward, so
    /// as it closes, every coordinate's step collapses with it.
    pub min_bound_frac: f64,
    /// Which parameter that was: 0 background, 1 amplitude, 2 y, 3 x.
    pub min_bound_kind: u8,
    /// Emitters in this group whose fitted amplitude sits on the window's
    /// amplitude floor. Such an emitter carries no flux and therefore no
    /// information, yet still costs three free parameters in the joint fit and
    /// contributes the near-null direction that makes `F` ill-conditioned.
    pub n_at_floor: usize,
}

/// One block-Jacobi pass, optionally restricted to the groups that can have
/// changed.
///
/// `dirty` names the emitters that moved in the previous pass; a group is
/// refitted only if one of them is among its own emitters or its frozen halo.
/// `None` fits everything. `moved` reports back which emitters this pass moved
/// further than `move_eps`, which is the next pass's `dirty`. Returns the
/// number of groups actually fitted.
#[allow(clippy::too_many_arguments)]
fn refine_sweep(
    s: &mut Solver,
    frame: &Frame,
    em: &Emitters,
    max_iter: usize,
    out: &mut Emitters,
    se: &mut [f64],
    prev: Option<&Emitters>,
    mut cost: Option<&mut Vec<PatchCost>>,
    tol_obj: f64,
    dirty: Option<&[bool]>,
    mut moved_out: Option<&mut [bool]>,
    move_eps: f64,
    pset: Option<&[patches::Patch]>,
    gauss_seidel: bool,
) -> usize {
    out.pos.copy_from_slice(&em.pos);
    out.amp.copy_from_slice(&em.amp);
    if let Some(m) = moved_out.as_deref_mut() {
        m.fill(false);
    }
    let owned;
    let pset = match pset {
        Some(p) => p,
        None => {
            owned = patches::build_patches(
                &em.pos,
                em.len(),
                frame.sigma,
                frame.h,
                frame.w,
                frame.k_max,
            );
            &owned
        }
    };

    let mut n_fitted = 0usize;
    for p in pset {
        // A group's fit reads its own emitters and its frozen halo and nothing
        // else -- that locality is the whole point of the decomposition, so it
        // may as well drive the schedule too. If nothing it reads has moved,
        // refitting it would reproduce its own last answer to within the
        // optimizer's own noise, and that noise is not free: it would move the
        // group, which dirties ITS neighbours, which is how a global sweep
        // manufactures the churn that keeps it from ever converging.
        if let Some(d) = dirty {
            if !p
                .indices
                .iter()
                .chain(p.frozen.iter())
                .any(|&i| d[i as usize])
            {
                continue;
            }
        }
        n_fitted += 1;
        let b = &p.bbox;
        frame.cut(b, &mut s.sub);
        let (h, w) = (b.h(), b.w());
        let mut shape = std::mem::take(&mut s.scratch_f64);
        // Jacobi reads the sweep's INPUT state, so every group in a pass sees
        // the same frozen neighbourhood and the result does not depend on visit
        // order -- which is what makes the pass parallelizable [P14].
        // Gauss-Seidel reads `out`, picking up groups already updated in THIS
        // pass: faster convergence, at the cost of that order-independence.
        let src: &Emitters = if gauss_seidel { &*out } else { em };
        let level = build_halo(
            frame,
            src,
            &p.frozen,
            b,
            &mut s.halo,
            &mut s.resid,
            &mut shape,
        );
        s.scratch_f64 = shape;

        pack_window(&mut s.theta, level, src, &p.indices, b);
        s.pre.clear();
        for &i in &p.indices {
            s.pre.push(src.pos[2 * i as usize]);
            s.pre.push(src.pos[2 * i as usize + 1]);
        }
        let info = fit_window(
            &mut s.fit_inc,
            &mut s.bounds_lo,
            &mut s.bounds_hi,
            &mut s.theta,
            &s.sub,
            h,
            w,
            frame.sigma,
            Some(&s.halo),
            max_iter,
            tol_obj,
        );
        let theta = s.fit_inc.theta();
        let mut moved = 0.0f64;
        for (slot, &i) in p.indices.iter().enumerate() {
            let i = i as usize;
            let ny = psf::cy(theta, slot) + b.y0 as f64;
            let nx = psf::cx(theta, slot) + b.x0 as f64;
            moved = moved.max((ny - s.pre[2 * slot]).hypot(nx - s.pre[2 * slot + 1]));
            out.pos[2 * i] = ny;
            out.pos[2 * i + 1] = nx;
            out.amp[i] = psf::amp(theta, slot);
        }
        if let Some(m) = moved_out.as_deref_mut() {
            for (slot, &i) in p.indices.iter().enumerate() {
                let i = i as usize;
                let dy = out.pos[2 * i] - s.pre[2 * slot];
                let dx = out.pos[2 * i + 1] - s.pre[2 * slot + 1];
                if dy.hypot(dx) > move_eps {
                    m[i] = true;
                }
            }
        }
        if let Some(c) = cost.as_deref_mut() {
            let reproducible = prev.map_or(false, |pv| {
                pv.len() == em.len()
                    && p.indices.iter().chain(p.frozen.iter()).all(|&i| {
                        let i = i as usize;
                        em.pos[2 * i] == pv.pos[2 * i]
                            && em.pos[2 * i + 1] == pv.pos[2 * i + 1]
                            && em.amp[i] == pv.amp[i]
                    })
            });
            let pn = 3 * p.indices.len() + 1;
            let (_, cond, ok) = s.ev.logdet_cond(s.fit_inc.fisher(pn), pn);
            let mut min_pair = f64::INFINITY;
            for a in 0..p.indices.len() {
                for b in (a + 1)..p.indices.len() {
                    let (ia, ib) = (p.indices[a] as usize, p.indices[b] as usize);
                    let d = (out.pos[2 * ia] - out.pos[2 * ib])
                        .hypot(out.pos[2 * ia + 1] - out.pos[2 * ib + 1]);
                    min_pair = min_pair.min(d);
                }
            }
            let (mut mbf, mut mbk) = (f64::INFINITY, 0u8);
            let th = s.fit_inc.theta();
            for q in 0..pn {
                let (lo, hi) = (s.bounds_lo[q], s.bounds_hi[q]);
                let range = (hi - lo).max(f64::MIN_POSITIVE);
                let f = ((th[q] - lo).min(hi - th[q]) / range).max(0.0);
                if f < mbf {
                    mbf = f;
                    mbk = if q == 0 { 0 } else { (1 + (q - 1) % 3) as u8 };
                }
            }
            // "On the floor" measured against the parameter's own RANGE, not
            // against the floor value: the floor is itself 1e-6 * A_max, so a
            // ratio test against it is a different, far stricter question [P11].
            let n_at_floor = (0..p.indices.len())
                .filter(|&slot| {
                    let q = 1 + 3 * slot;
                    let range = (s.bounds_hi[q] - s.bounds_lo[q]).max(f64::MIN_POSITIVE);
                    (th[q] - s.bounds_lo[q]) / range < 1e-8
                })
                .count();
            c.push(PatchCost {
                n_iter: info.n_iter,
                converged: info.converged,
                moved,
                reproducible,
                scaled_cond: if ok { cond } else { f64::INFINITY },
                min_pair_px: min_pair,
                k: p.indices.len(),
                min_bound_frac: mbf,
                min_bound_kind: mbk,
                n_at_floor,
            });
        }

        // Standard errors come from the Fisher matrix of THIS fit, the one
        // whose parameters are reported -- never from a proposal fit.
        let pn = 3 * p.indices.len() + 1;
        s.inv_diag.clear();
        s.inv_diag.resize(pn, 0.0);
        if !s.ev.inv_diag(
            s.fit_inc.fisher(pn),
            pn,
            &mut s.inv_diag,
            &mut s.inv_scratch,
        ) {
            continue; // singular: leave NaN, as the Python does
        }
        for (slot, &i) in p.indices.iter().enumerate() {
            let i = i as usize;
            for (axis, q) in [(0usize, 1 + 3 * slot), (1, 2 + 3 * slot), (2, 3 + 3 * slot)] {
                let v = s.inv_diag[q];
                se[3 * i + axis] = if v > 0.0 { v.sqrt() } else { f64::NAN };
            }
        }
    }
    n_fitted
}

/// Joint re-fit at fixed `N` in connected groups, plus per-emitter CRLBs.
///
/// **Proposes nothing.** No move can happen here; this is the estimation half,
/// and it is where the reported parameters and standard errors come from.
///
/// # Why it is iterated
///
/// Each patch fit holds its out-of-patch neighbours frozen at whatever
/// positions it was handed, so one pass propagates any staleness in those
/// neighbours into the emitter they surround. Measured on isolated emitters at
/// density 0.055, with the target started at truth:
///
/// | start | med \|err\| | pull rsd |
/// |---|---|---|
/// | truth (reference) | 0.052 | 0.96 |
/// | target perturbed only | 0.065 | 1.26 |
/// | **neighbours perturbed only** | **0.136** | **1.98** |
/// | both | 0.146 | 2.36 |
/// | both, swept to convergence | 0.069 | 1.27 |
///
/// The optimizer reports converged 96-99% of the time in every row. This is not
/// a convergence failure, it is a *schedule* failure -- and rebuilding the patch
/// decomposition each sweep is what fixes it.
///
/// `max_sweeps = 1` reproduces the single-pass behaviour and is what the round
/// loop uses, since the round loop iterates anyway.
pub fn refine(
    s: &mut Solver,
    frame: &Frame,
    em: &mut Emitters,
    max_iter: usize,
    max_sweeps: usize,
    tol: f64,
) -> Vec<f64> {
    let n = em.len();
    let mut se = vec![f64::NAN; 3 * n];
    if n == 0 {
        return se;
    }
    let mut next = em.clone();
    // The schedule is group-wise, not global. `dirty` starts as "everything"
    // and thereafter holds only the emitters the previous pass actually moved;
    // a group is refitted only when something it reads is in that set.
    //
    // This replaces a global `max |dpos| < tol` break that could not fire: on a
    // 906-emitter frame 80% of emitters were still moving past `tol` after
    // eight sweeps, so the loop always ran to `max_sweeps` and stopped
    // mid-flight. It could not fire because it was self-defeating -- refitting
    // every group every sweep moves every group, so every group's halo is
    // always stale. Scheduling on locality lets quiet regions drop out and stay
    // out, and gives the loop a termination condition it can actually reach:
    // the queue drains.
    let mut dirty = vec![true; n];
    let mut moved = vec![false; n];
    // The grouping is rebuilt every pass, and it must be. Pinning it for the
    // whole call was tried -- it makes the map continuous, stabilizes the group
    // count and cuts 11% of the LM iterations -- and it is WRONG: over four
    // passes emitters drift across link radii, and a stale grouping puts two
    // now-adjacent emitters in separate groups, each frozen in the other's
    // halo, which fits both as if the other were a constant. Measured on a
    // 256x256 field: 8 missed emitters and a residual score peak of |z| = 129,
    // against 0 and 5.6 with the rebuild. The rebuild is not churn; it is the
    // decomposition tracking the configuration.
    for _ in 0..max_sweeps.max(1) {
        let fitted = refine_sweep(
            s,
            frame,
            em,
            max_iter,
            &mut next,
            &mut se,
            None,
            None,
            REFINE_TOL_OBJ,
            Some(&dirty),
            Some(&mut moved),
            tol,
            None,
            REFINE_GAUSS_SEIDEL,
        );
        std::mem::swap(em, &mut next);
        if fitted == 0 || !moved.iter().any(|&m| m) {
            break;
        }
        dirty.copy_from_slice(&moved);
    }
    se
}

/// One instrumented REFINE sweep, for measurement rather than production.
///
/// Same code path as [`refine`]'s sweep -- it *is* that function -- with the
/// per-patch cost recorded. `prev` is the state one sweep earlier, used only to
/// decide `PatchCost::reproducible`; pass `None` on the first sweep.
pub fn refine_sweep_costed(
    s: &mut Solver,
    frame: &Frame,
    em: &Emitters,
    max_iter: usize,
    out: &mut Emitters,
    se: &mut [f64],
    prev: Option<&Emitters>,
    cost: &mut Vec<PatchCost>,
    tol_obj: f64,
    dirty: Option<&[bool]>,
    moved_out: Option<&mut [bool]>,
    move_eps: f64,
    pset: Option<&[patches::Patch]>,
    gauss_seidel: bool,
) -> usize {
    cost.clear();
    se.fill(f64::NAN);
    refine_sweep(
        s,
        frame,
        em,
        max_iter,
        out,
        se,
        prev,
        Some(cost),
        tol_obj,
        dirty,
        moved_out,
        move_eps,
        pset,
        gauss_seidel,
    )
}

/// One pass of removal tests, faintest first.
///
/// # Why removal cannot live inside the add loop
///
/// An emitter's `A/SE` verdict depends on which of its neighbours are free in
/// that sweep. Letting removal feed back into addition made an earlier global
/// sweep oscillate with period 2 for entire runs -- the same source killed and
/// recreated indefinitely, with 2710 of 2725 accepted deaths forced by that
/// guard. Running removal once, after the loop, at monotonically decreasing
/// `N`, cannot cycle.
///
/// # Faintest first, with write-back
///
/// A spurious emitter is far likelier to be faint, and removing it may make its
/// neighbour's own removal unnecessary. When one member of a collapsed pair
/// goes, the other absorbs its flux, and the next test in the same pass must be
/// scored against **that** -- so the reduced fit is written back over the
/// survivors. An `alive` mask is used rather than deleting as we go, because
/// the visit order is computed once and deleting would shift every later index
/// onto a different emitter.
pub fn prune(s: &mut Solver, frame: &Frame, em: &mut Emitters, prior: Prior) -> usize {
    let n = em.len();
    if n == 0 {
        return 0;
    }
    let mut alive = vec![true; n];
    let mut gridx = EmitterGrid::build(&em.pos, n, frame.h, frame.w, HALO_FACTOR * frame.sigma);

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| {
        em.amp[a as usize]
            .partial_cmp(&em.amp[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    let mut n_removed = 0;
    for &gi in &order {
        let gi = gi as usize;
        if !alive[gi] {
            continue;
        }
        let (cy, cx) = (em.y(gi), em.x(gi));
        // The window, over the living emitters other than `gi`. Querying the
        // full index and skipping the dead gives the same set as the Python's
        // `_window(positions[others], ...)` with global indices already.
        let (mut free, frozen, bbox) = patches::window(
            &em.pos,
            n,
            cy,
            cx,
            frame.sigma,
            frame.h,
            frame.w,
            frame.k_max,
            &gridx,
            &mut s.scratch_u32,
        );
        free.retain(|&i| alive[i as usize] && i as usize != gi);
        free.truncate(frame.k_max.saturating_sub(1));
        let frozen: Vec<u32> = frozen
            .into_iter()
            .filter(|&i| alive[i as usize] && i as usize != gi)
            .collect();

        frame.cut(&bbox, &mut s.sub);
        let (h, w) = (bbox.h(), bbox.w());
        let mut shape = std::mem::take(&mut s.scratch_f64);
        let level = build_halo(
            frame,
            em,
            &frozen,
            &bbox,
            &mut s.halo,
            &mut s.resid,
            &mut shape,
        );
        s.scratch_f64 = shape;

        // Full model: the survivors plus `gi`, which is appended last and is
        // therefore the last emitter of the fitted vector.
        let mut keep = free.clone();
        keep.push(gi as u32);
        pack_window(&mut s.theta, level, em, &keep, &bbox);
        let full = fit_window(
            &mut s.fit_inc,
            &mut s.bounds_lo,
            &mut s.bounds_hi,
            &mut s.theta,
            &s.sub,
            h,
            w,
            frame.sigma,
            Some(&s.halo),
            100,
            EVIDENCE_TOL_OBJ,
        );
        let p_full = 3 * keep.len() + 1;
        let sum_a_full = sum_amplitudes(s.fit_inc.theta());
        s.f_inc.clear();
        s.f_inc.extend_from_slice(s.fit_inc.fisher(p_full));
        let a_gi = psf::amp(s.fit_inc.theta(), keep.len() - 1);

        // Reduced model: the survivors alone.
        pack_window(&mut s.theta, level, em, &free, &bbox);
        let reduced = fit_window(
            &mut s.fit_prop,
            &mut s.bounds_lo,
            &mut s.bounds_hi,
            &mut s.theta,
            &s.sub,
            h,
            w,
            frame.sigma,
            Some(&s.halo),
            100,
            EVIDENCE_TOL_OBJ,
        );
        let p_red = 3 * free.len() + 1;

        // Forced, not weighed, when the quantity that would do the weighing is
        // the thing that has broken.
        let k = keep.len() - 1;
        s.inv_diag.clear();
        s.inv_diag.resize(p_full, 0.0);
        let var = if s
            .ev
            .inv_diag(&s.f_inc, p_full, &mut s.inv_diag, &mut s.inv_scratch)
        {
            let v = s.inv_diag[1 + 3 * k];
            if v.is_finite() && v > 0.0 {
                Some(v)
            } else {
                None
            }
        } else {
            None
        };
        let log_bf = match var {
            None => f64::INFINITY,
            Some(v) if a_gi < PRUNE_TAU * f64::sqrt(v) => f64::INFINITY,
            Some(_) => s.ev.log_bf_remove(
                full.i_div,
                reduced.i_div,
                &s.f_inc,
                s.fit_prop.fisher(p_red),
                p_full,
                p_red,
                sum_a_full,
                sum_amplitudes(s.fit_prop.theta()),
                keep.len(),
                prior,
                None,
            ),
        };

        if log_bf > 0.0 {
            alive[gi] = false;
            n_removed += 1;
            gridx.remove(gi as u32, cy, cx);
            if !free.is_empty() {
                write_back(em, &mut gridx, s.fit_prop.theta(), &free, &bbox, false);
            }
        }
    }

    if n_removed > 0 {
        let (mut pos, mut amp) = (Vec::new(), Vec::new());
        for i in 0..n {
            if alive[i] {
                pos.push(em.y(i));
                pos.push(em.x(i));
                amp.push(em.amp[i]);
            }
        }
        em.pos = pos;
        em.amp = amp;
    }
    n_removed
}
