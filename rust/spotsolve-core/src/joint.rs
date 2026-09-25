//! Joint refinement: one likelihood for the whole frame.
//!
//! The one-pass search ([`boxsearch::one_pass`]) decides each window once,
//! against neighbours frozen at whatever they were when it ran. In crowded
//! fields that is one-sided: the first windows never see later neighbours,
//! and position errors ran 5-20x the CRLB on a toy chain even 4 sigma apart.
//! Here every emitter and the background share one Poisson model
//!
//! ```text
//! m = B(beta) + sum_k A_k g(p; y_k, x_k, w_k)
//! ```
//!
//! with `B` bilinear on a [`TILE`]-px node lattice, and it is solved by block
//! coordinate descent:
//!
//! - emitters are fitted in coupled groups by LM, everything else fixed
//!   (halo); groups are cut from the pairwise coupling so each stays small
//!   ([`KG`]) and the blocks between them are weak;
//! - the nodes are fitted by Poisson IRLS with emitters fixed.
//!
//! Counts change only after the model has converged: removal at the nominal
//! threshold u, then score-gated additions at `u * kappa`, where `kappa` is
//! the spread of the residual score far from emitters (an empirical null).
//! Then the model is re-converged, until an add round changes nothing.
//!
//! Only the last round's answer matters, so no sub-problem is solved more
//! exactly than the next round can use ([`Config`]): group refits stop at a
//! looser tolerance, removal trials after a few steps, plain rounds use
//! smaller boxes and the previous partition, and later add rounds retest only
//! groups that changed. Together 6x faster than solving each block to
//! convergence, at equal referee recall and precision.
//!
//! The design and its constants were measured on the Python prototype
//! (`scripts/jointfit_prototype.py`; flags KG 12, PRE_BG, OMEGA 1.5,
//! EMP_NULL, REMOVE at u) and `tests/fixtures/11_joint.json` pins its stages.

use crate::boxsearch::{self, Em, Fitted, Settings, Window, Workspace, BG_FLOOR, K_MAX, OWN, SUPPORT};
use crate::linalg::{self, Chol};
use crate::lmcl::{self, Bounds, FitOpts};
use crate::psf;

/// px. Background node spacing. The prototype's crowded cells (bg 20, true
/// value known) fitted nodes within -0.7..+0.3 ADU at 16 px, where the 25-px
/// median read 32 and 47 in the two densest.
pub const TILE: usize = 16;
/// Emitters per group, at most. Groups merge strongest coupling first and
/// stop at this size, so the frame never percolates into one LM problem:
/// cell 19 (density 0.055) took 49 s uncapped and 7-9 s at 12 with the same
/// accuracy; on cell 18 the strongest cut coupling was rho^2 0.12, the same
/// partition as uncapped.
pub const KG: usize = 12;
/// Pairs coupled more weakly than this (squared first canonical correlation
/// of their `(A, y, x, w)` blocks) are never joined. Gauss-Seidel contracts
/// by rho^2 per sweep, so 0.05 costs about one extra round.
pub const RHO_MIN: f64 = 0.05;
/// Emitter widths: pairs farther apart than this are not examined, and the
/// Fisher matrix of a pair is taken over its union of this-many-width supports.
pub const PAIR_REACH: f64 = 6.0;
pub const PAIR_SUPPORT: f64 = 3.0;
/// Emitter widths: group box margin, floored at the add placement support
/// `(OWN + SUPPORT) sigma`.
pub const GROUP_PAD: f64 = 3.0;
/// Emitter widths: an emitter's light is rendered within this (plus one px)
/// when the whole frame is redrawn.
pub const RENDER_REACH: f64 = 4.0;
/// Over-relaxation of each node update. Emitters and nodes contract ~0.4 per
/// round against each other; 1.5 halved the rounds (19 -> 10.5, 43 -> 20.5)
/// with no accuracy loss over 8 replicate seeds.
pub const OMEGA: f64 = 1.5;
/// IRLS steps per node update.
pub const IRLS_STEPS: usize = 3;
/// Nats per emitter: a round that improves the objective less has converged.
/// 0.01 matched 0.001 in accuracy at half the rounds.
pub const TOL: f64 = 1e-2;
pub const MAX_OUTER: usize = 80;
/// The group fit holds the level at zero (the nodes carry it) within this.
pub const LEVEL_LOCK: f64 = 1e-6;
/// sigma. Pixels this far from every emitter feed the empirical null.
pub const NULL_FAR: f64 = 3.0;
/// px. The score's reflected border is excluded from the null.
pub const NULL_BORDER: usize = 4;
/// Fewer null pixels than this leave kappa at 1.
pub const NULL_MIN_PX: usize = 100;
/// Gaussian consistency factor of the MAD.
pub const MAD_SCALE: f64 = 1.4826;

/// How exactly each block is solved. [`Config::default`] is the measured
/// fast setting; [`Config::prototype`] solves every block to convergence, as
/// the Python prototype did, and is what `tests/fixtures/11_joint.json` pins.
///
/// Measured together on 10 GEM frames (128^2), 4 bead frames (256^2) and 12
/// referee cells (64^2), s/frame, single thread (speed round 2026-09-24):
///
/// ```text
///                              GEM    beads  cells  cells recall/precision
/// prototype                    9.83   12.94  2.35   0.820 / 0.936
/// + remove_iter 3              3.83    4.85  1.12   0.823 / 0.936
/// + plain_tol 1e-3             2.42    3.64  0.85   0.822 / 0.935
/// + plain_pad, regroup         2.07    3.04  0.69   0.821 / 0.932
/// + active_tol 0.05            1.62    2.01  0.63   0.821 / 0.931
/// ```
///
/// GEM and bead counts stayed within 0.4%. Tried and rejected: removal
/// trials refitting only members near the removed one (removed too few;
/// precision 0.925), an active set in plain rounds (the groups it skipped
/// converged in one step anyway), node over-relaxation 1.0-1.95 and a
/// Schur-coupled node step (no fewer rounds), a looser outer tolerance (0.1
/// cost precision). What remains is the background drifting along a ridge
/// with wide emitters' flux and width, ~2 ADU per round on GEM.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Objective tolerance (nats) of each group refit. Iterations per refit
    /// fell 8.6 -> 3.2 from 1e-6, rounds unchanged.
    pub plain_tol: f64,
    /// LM iterations of a leave-one-out removal trial. A truncated trial
    /// overstates the cost of removal, so it can only keep an emitter the
    /// full trial would remove; removable ones show within 2-3 steps.
    /// Iterations per trial fell 25 -> 2.9. 1 and 2 were no faster overall.
    pub remove_iter: usize,
    /// Plain rounds pad group boxes by [`GROUP_PAD`] widths only; the add
    /// support `(OWN + SUPPORT) sigma` is needed only where emitters are added.
    pub plain_pad: bool,
    /// Keep the partition until the count changes.
    pub regroup_on_change: bool,
    /// After the first, an add round tests a group only if a member moved
    /// or changed width by more than this many sigma, or flux by this
    /// fraction, since the previous add round, or an emitter was added or
    /// removed inside its box. 0.02 and 0.1 gave the same counts.
    pub active_tol: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self { plain_tol: 1e-3, remove_iter: 3, plain_pad: true, regroup_on_change: true, active_tol: 0.05 }
    }
}

impl Config {
    pub fn prototype() -> Self {
        Self {
            plain_tol: boxsearch::FIT_TOL_OBJ,
            remove_iter: boxsearch::FIT_MAX_ITER,
            plain_pad: false,
            regroup_on_change: false,
            active_tol: 0.0,
        }
    }
}

/// Bilinear background on a node lattice `ny x nx`, `TILE` px apart,
/// node `(0, 0)` at pixel `(0, 0)`; the last row and column may lie beyond
/// the frame.
#[derive(Clone, Debug)]
pub struct Nodes {
    pub ny: usize,
    pub nx: usize,
    pub h: usize,
    pub w: usize,
    pub beta: Vec<f64>,
}

impl Nodes {
    pub fn new(h: usize, w: usize) -> Self {
        let ny = (h.max(1) - 1).div_ceil(TILE) + 1;
        let nx = (w.max(1) - 1).div_ceil(TILE) + 1;
        Self { ny, nx, h, w, beta: vec![0.0; ny * nx] }
    }

    /// The four `(node, weight)` pairs of pixel `(y, x)`.
    fn stencil(&self, y: usize, x: usize) -> [(usize, f64); 4] {
        let axis = |v: usize, n: usize| {
            let f = v as f64 / TILE as f64;
            let i = if n > 1 { (f.floor() as usize).min(n - 2) } else { 0 };
            (i, f - i as f64)
        };
        let (iy, ty) = axis(y, self.ny);
        let (ix, tx) = axis(x, self.nx);
        let node = |dy: usize, dx: usize| (iy + dy).min(self.ny - 1) * self.nx + (ix + dx).min(self.nx - 1);
        [
            (node(0, 0), (1.0 - ty) * (1.0 - tx)),
            (node(1, 0), ty * (1.0 - tx)),
            (node(0, 1), (1.0 - ty) * tx),
            (node(1, 1), ty * tx),
        ]
    }

    fn eval(&self, beta: &[f64], out: &mut Vec<f64>) {
        out.clear();
        for y in 0..self.h {
            for x in 0..self.w {
                out.push(self.stencil(y, x).iter().map(|&(n, t)| t * beta[n]).sum());
            }
        }
    }

    pub fn surface(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.h * self.w);
        self.eval(&self.beta, &mut out);
        out
    }

    /// Solve `sum_p wt_p s_p s_p^T step = sum_p s_p r_p` for the node step.
    fn normal_solve(&self, wt: impl Fn(usize) -> f64, r: impl Fn(usize) -> f64) -> Option<Vec<f64>> {
        let n = self.ny * self.nx;
        let kd = self.nx + 1;
        let mut a = vec![0.0; n * (kd + 1)];
        let mut b = vec![0.0; n];
        for y in 0..self.h {
            for x in 0..self.w {
                let p = y * self.w + x;
                let st = self.stencil(y, x);
                let (wp, rp) = (wt(p), r(p));
                for &(i, ti) in &st {
                    b[i] += ti * rp;
                    for &(j, tj) in &st {
                        if i >= j {
                            a[i * (kd + 1) + kd + j - i] += wp * ti * tj;
                        }
                    }
                }
            }
        }
        linalg::band_solve(&mut a, n, kd, &mut b).then_some(b)
    }

    /// Least-squares fit of the surface to `map`.
    pub fn fit_map(&mut self, map: &[f64]) {
        if let Some(b) = self.normal_solve(|_| 1.0, |p| map[p]) {
            self.beta = b;
        }
    }

    /// [`IRLS_STEPS`] Poisson scoring steps on `d` with `light` fixed, then
    /// the update scaled by `omega`. Nodes stay above [`BG_FLOOR`].
    pub fn irls(&mut self, d: &[f64], light: &[f64], omega: f64) {
        let mut b = self.beta.clone();
        let mut m = Vec::with_capacity(self.h * self.w);
        for _ in 0..IRLS_STEPS {
            self.eval(&b, &mut m);
            for (v, l) in m.iter_mut().zip(light) {
                *v = (*v + l).max(BG_FLOOR);
            }
            let Some(step) = self.normal_solve(|p| 1.0 / m[p], |p| (d[p] - m[p]) / m[p]) else { break };
            for (v, s) in b.iter_mut().zip(&step) {
                *v = (*v + s).max(BG_FLOOR);
            }
        }
        for (v, t) in self.beta.iter_mut().zip(&b) {
            *v = (*v + omega * (t - *v)).max(BG_FLOOR);
        }
    }
}

/// Light of global emitters on the box `(y0, x0, h, w)`, into `out`.
fn render(ems: &[Em], y0: usize, x0: usize, h: usize, w: usize, f: &mut psf::Factors, out: &mut Vec<f64>) {
    out.clear();
    out.resize(h * w, 0.0);
    if ems.is_empty() {
        return;
    }
    let mut theta = Vec::with_capacity(4 * ems.len() + 1);
    theta.push(0.0);
    for e in ems {
        theta.extend_from_slice(&[e[0], e[1] - y0 as f64, e[2] - x0 as f64, e[3]]);
    }
    f.ensure(h, w, ems.len());
    psf::model_var_sigma_ax(&theta, &psf::local_axis(h), &psf::local_axis(w), None, f, out);
}

/// Pixel box `[y0, y1) x [x0, x1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box2 {
    pub y0: usize,
    pub x0: usize,
    pub y1: usize,
    pub x1: usize,
}

impl Box2 {
    fn h(&self) -> usize {
        self.y1 - self.y0
    }
    fn w(&self) -> usize {
        self.x1 - self.x0
    }
}

/// `[trunc(lo) - pad, trunc(hi) + pad + 2)` clipped to `[0, n)`: the
/// prototype's `int()` box, reproduced so parity holds at the pixel level.
fn span(lo: f64, hi: f64, pad: i64, n: usize) -> (usize, usize) {
    let a = ((lo as i64) - pad).max(0) as usize;
    let b = ((hi as i64) + pad + 2).clamp(0, n as i64) as usize;
    (a, b)
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub fits: usize,
    pub adds: usize,
    pub removed: usize,
    /// Additions that passed the score but failed the LR confirmation.
    pub lr_fail: usize,
    pub outer: usize,
    pub max_group: usize,
    /// Empirical-null scale of the last add round; 1 before one.
    pub kappa: f64,
}

/// The joint model of one frame (or ROI crop): data, emitters, background.
pub struct Joint<'a> {
    d: &'a [f64],
    pub h: usize,
    pub w: usize,
    s: Settings,
    pub phi: f64,
    pub u: f64,
    /// `max(d, 1)`: the flux bound scale, frame-wide.
    smax: f64,
    roi: Option<&'a [bool]>,
    k1: Vec<f64>,
    pub ems: Vec<Em>,
    /// Fit diagnostics (`FLAG_NOT_CONVERGED`, `FLAG_STALLED`, `FLAG_BOUND`)
    /// of each emitter's latest group fit.
    pub flags: Vec<u8>,
    pub nodes: Nodes,
    /// `nodes.surface()`, kept in step with `nodes`.
    pub bg: Vec<f64>,
    /// Rendered emitter light, `h * w`.
    pub light: Vec<f64>,
    pub stats: Stats,
    pub config: Config,
    /// The partition, while `config.regroup_on_change` keeps it.
    groups: Option<Vec<Vec<usize>>>,
    /// Emitters as of the last add round, and where it added or removed.
    tested: Vec<Em>,
    changed_at: Vec<(f64, f64)>,
}

impl<'a> Joint<'a> {
    /// Start from the one-pass emitters `e0`: nodes fitted to the median map
    /// `bg0`, then one IRLS update against `e0`'s light (the prototype's
    /// PRE_BG, which halved the rounds). `roi` limits where emitters are added.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d: &'a [f64],
        h: usize,
        w: usize,
        e0: &[Em],
        bg0: &[f64],
        phi: f64,
        u: f64,
        s: &Settings,
        roi: Option<&'a [bool]>,
        ws: &mut Workspace,
    ) -> Self {
        assert_eq!(d.len(), h * w);
        let mut nodes = Nodes::new(h, w);
        nodes.fit_map(bg0);
        let smax = d.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v)).max(1.0);
        let mut j = Self {
            d,
            h,
            w,
            s: *s,
            phi,
            u,
            smax,
            roi,
            k1: boxsearch::psf_kernel1d(s.sigma),
            ems: e0.to_vec(),
            flags: vec![0; e0.len()],
            nodes,
            bg: Vec::new(),
            light: Vec::new(),
            stats: Stats { kappa: 1.0, ..Stats::default() },
            config: Config::default(),
            groups: None,
            tested: Vec::new(),
            changed_at: Vec::new(),
        };
        j.redraw(ws);
        j.nodes.irls(d, &j.light, 1.0);
        j.bg = j.nodes.surface();
        j
    }

    /// `light <- sum_k` of each emitter within [`RENDER_REACH`] widths.
    fn redraw(&mut self, ws: &mut Workspace) {
        let (h, w) = (self.h, self.w);
        self.light.clear();
        self.light.resize(h * w, 0.0);
        let mut tmp = Vec::new();
        for e in &self.ems {
            let r = (RENDER_REACH * e[3]).ceil() as i64 + 1;
            let (y0, y1) = span(e[1], e[1], r, h);
            let (x0, x1) = span(e[2], e[2], r, w);
            if y1 <= y0 || x1 <= x0 {
                continue;
            }
            render(std::slice::from_ref(e), y0, x0, y1 - y0, x1 - x0, &mut ws.f, &mut tmp);
            for (r_, y) in (y0..y1).enumerate() {
                for (c, x) in (x0..x1).enumerate() {
                    self.light[y * w + x] += tmp[r_ * (x1 - x0) + c];
                }
            }
        }
    }

    /// `max(B + light, BG_FLOOR)`.
    pub fn model(&self) -> Vec<f64> {
        self.bg.iter().zip(&self.light).map(|(b, l)| (b + l).max(BG_FLOOR)).collect()
    }

    /// Dispersion-scaled Poisson deviance of the current model, nats.
    pub fn objective(&self) -> f64 {
        let mut acc = 0.0;
        for (i, &d) in self.d.iter().enumerate() {
            let m = (self.bg[i] + self.light[i]).max(BG_FLOOR);
            acc += m - d;
            if d > 0.0 {
                acc += d * (d.max(1e-12) / m).ln();
            }
        }
        acc / self.phi
    }

    /// Candidate pairs within [`PAIR_REACH`] of the widest emitter's width,
    /// by uniform grid hash, as `(i, j)` with `i < j`.
    fn candidate_pairs(&self) -> Vec<(usize, usize)> {
        let n = self.ems.len();
        if n < 2 {
            return Vec::new();
        }
        let r = PAIR_REACH * self.ems.iter().map(|e| e[3]).fold(0.0, f64::max);
        let cell = |v: f64| (v / r).floor() as i64;
        let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = std::collections::HashMap::new();
        for (i, e) in self.ems.iter().enumerate() {
            grid.entry((cell(e[1]), cell(e[2]))).or_default().push(i);
        }
        let mut out = Vec::new();
        for (i, e) in self.ems.iter().enumerate() {
            let (cy, cx) = (cell(e[1]), cell(e[2]));
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let Some(v) = grid.get(&(cy + dy, cx + dx)) else { continue };
                    for &j in v {
                        if j > i && (e[1] - self.ems[j][1]).hypot(e[2] - self.ems[j][2]) <= r {
                            out.push((i, j));
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Squared first canonical correlation between two emitters' `(A, y, x, w)`
    /// blocks, from their Poisson Fisher matrix under `model` on the union of
    /// their [`PAIR_SUPPORT`]-width supports: the largest eigenvalue of
    /// `F11^-1 F12 F22^-1 F21`. It is the per-sweep contraction factor of
    /// fitting the two separately. 1 when a block is singular.
    pub fn pair_rho2(&self, i: usize, j: usize, model: &[f64], ws: &mut Workspace) -> f64 {
        let (a, b) = (self.ems[i], self.ems[j]);
        let r = PAIR_SUPPORT * a[3].max(b[3]);
        let y0 = ((a[1].min(b[1]) - r) as i64).max(0) as usize;
        let x0 = ((a[2].min(b[2]) - r) as i64).max(0) as usize;
        let y1 = (((a[1].max(b[1]) + r) as i64) + 2).clamp(0, self.h as i64) as usize;
        let x1 = (((a[2].max(b[2]) + r) as i64) + 2).clamp(0, self.w as i64) as usize;
        let (h, w) = (y1 - y0, x1 - x0);
        let n = h * w;
        let theta = [
            0.0, a[0], a[1] - y0 as f64, a[2] - x0 as f64, a[3],
            b[0], b[1] - y0 as f64, b[2] - x0 as f64, b[3],
        ];
        ws.f.ensure(h, w, 2);
        ws.model.clear();
        ws.model.resize(n, 0.0);
        ws.jac.clear();
        ws.jac.resize(9 * n, 0.0);
        psf::model_and_jac_var_sigma_ax(
            &theta, &psf::local_axis(h), &psf::local_axis(w), None, &mut ws.f, &mut ws.model, &mut ws.jac,
        );
        let wt: Vec<f64> = (0..n).map(|p| 1.0 / model[(y0 + p / w) * self.w + x0 + p % w]).collect();
        let mut f = [0.0; 64];
        for q in 0..8 {
            for s in 0..=q {
                let (jq, js) = (&ws.jac[(q + 1) * n..(q + 2) * n], &ws.jac[(s + 1) * n..(s + 2) * n]);
                let v: f64 = (0..n).map(|p| jq[p] * js[p] * wt[p]).sum();
                f[q * 8 + s] = v;
                f[s * 8 + q] = v;
            }
        }
        let block = |r0: usize, c0: usize| {
            let mut m = [0.0; 16];
            for r in 0..4 {
                for c in 0..4 {
                    m[r * 4 + c] = f[(r0 + r) * 8 + c0 + c];
                }
            }
            m
        };
        let (f11, f12, f22) = (block(0, 0), block(0, 4), block(4, 4));
        let (mut c1, mut c2) = (Chol::new(4), Chol::new(4));
        if !c1.factor(&f11, 4) || !c2.factor(&f22, 4) {
            return 1.0;
        }
        // A = F12 F22^-1 F21 (symmetric), then S = L1^-1 A L1^-T.
        let mut x = [0.0; 4];
        let mut a_ = [0.0; 16];
        for c in 0..4 {
            let col: [f64; 4] = std::array::from_fn(|r| f12[c * 4 + r]); // F21[:, c]
            c2.solve(&col, &mut x);
            for r in 0..4 {
                a_[r * 4 + c] = (0..4).map(|k| f12[r * 4 + k] * x[k]).sum();
            }
        }
        let mut t = [0.0; 16]; // t[:, c] = L1^-1 A[:, c]
        for c in 0..4 {
            let mut col: [f64; 4] = std::array::from_fn(|r| a_[r * 4 + c]);
            c1.solve_lower_in_place(&mut col);
            for r in 0..4 {
                t[r * 4 + c] = col[r];
            }
        }
        let mut s = [0.0; 16]; // s[:, c] = L1^-1 (t^T)[:, c] = L1^-1 A L1^-T
        for c in 0..4 {
            let mut col: [f64; 4] = std::array::from_fn(|r| t[c * 4 + r]);
            c1.solve_lower_in_place(&mut col);
            for r in 0..4 {
                s[r * 4 + c] = col[r];
            }
        }
        for r in 0..4 {
            for c in 0..r {
                let v = 0.5 * (s[r * 4 + c] + s[c * 4 + r]);
                s[r * 4 + c] = v;
                s[c * 4 + r] = v;
            }
        }
        linalg::sym_eig_max(&mut s, 4).abs()
    }

    /// Candidate pairs with their coupling, `(i, j, rho^2)`.
    pub fn pairs(&self, model: &[f64], ws: &mut Workspace) -> Vec<(usize, usize, f64)> {
        self.candidate_pairs().into_iter().map(|(i, j)| (i, j, self.pair_rho2(i, j, model, ws))).collect()
    }

    /// Partition into groups: union-find over pairs, strongest coupling
    /// first, joining two groups only while the union holds at most [`KG`]
    /// and the coupling is at least [`RHO_MIN`]. Divide and conquer, never
    /// chaining: a weak link cannot pull a big cluster into a bigger one.
    /// Groups come ordered by their root's index, members ascending.
    pub fn groups(&self, model: &[f64], ws: &mut Workspace) -> Vec<Vec<usize>> {
        let n = self.ems.len();
        let mut pairs = self.pairs(model, ws);
        pairs.sort_by(|a, b| b.2.total_cmp(&a.2).then((a.0, a.1).cmp(&(b.0, b.1))));
        let mut parent: Vec<usize> = (0..n).collect();
        let mut size = vec![1usize; n];
        fn find(parent: &mut [usize], mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        for &(i, j, rho) in &pairs {
            if rho < RHO_MIN {
                break;
            }
            let (mut a, mut b) = (find(&mut parent, i), find(&mut parent, j));
            if a == b || size[a] + size[b] > KG {
                continue;
            }
            if size[a] < size[b] {
                std::mem::swap(&mut a, &mut b);
            }
            parent[b] = a;
            size[a] += size[b];
        }
        let mut by_root: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            let r = find(&mut parent, i);
            by_root[r].push(i);
        }
        by_root.into_iter().filter(|g| !g.is_empty()).collect()
    }

    /// Empirical-null scale: the MAD spread of the residual score z at
    /// pixels farther than [`NULL_FAR`] sigma from every emitter, floored
    /// at 1. On synthetic Poisson frames it is 1; on GEM 1.46 and beads
    /// 1.24, where the model misfits (PSF wings, haze), and there testing
    /// adds at `u * kappa` kept 5 of 12 user-confirmed real additions and 4
    /// of 22 rejected (2 and 4 with no additions at all).
    pub fn kappa(&self) -> f64 {
        let (h, w) = (self.h, self.w);
        let m = self.model();
        let r: Vec<f64> = self.d.iter().zip(&m).map(|(d, m)| d - m).collect();
        let var: Vec<f64> = m.iter().map(|m| self.phi * m).collect();
        let z = boxsearch::detection_map(&r, &var, h, w, self.s.sigma);
        let reach = NULL_FAR * self.s.sigma;
        let mut near = vec![false; h * w];
        for e in &self.ems {
            let (ya, yb) = (((e[1] - reach).ceil().max(0.0)) as usize, ((e[1] + reach).floor() + 1.0).clamp(0.0, h as f64) as usize);
            let (xa, xb) = (((e[2] - reach).ceil().max(0.0)) as usize, ((e[2] + reach).floor() + 1.0).clamp(0.0, w as f64) as usize);
            for y in ya..yb {
                for x in xa..xb {
                    if (y as f64 - e[1]).hypot(x as f64 - e[2]) <= reach {
                        near[y * w + x] = true;
                    }
                }
            }
        }
        let b = NULL_BORDER;
        if h <= 2 * b || w <= 2 * b {
            return 1.0;
        }
        let zf: Vec<f64> = (b..h - b)
            .flat_map(|y| (b..w - b).map(move |x| y * w + x))
            .filter(|&p| !near[p])
            .map(|p| z[p])
            .collect();
        if zf.len() <= NULL_MIN_PX {
            return 1.0;
        }
        let med = boxsearch::median(&zf);
        let dev: Vec<f64> = zf.iter().map(|v| (v - med).abs()).collect();
        (MAD_SCALE * boxsearch::median(&dev)).max(1.0)
    }

    /// The pixel box of a group: its emitters plus the larger of
    /// [`GROUP_PAD`] widths and the add support `(OWN + SUPPORT) sigma`.
    pub fn group_box(&self, g: &[usize]) -> Box2 {
        self.group_box_for(g, true)
    }

    fn group_box_for(&self, g: &[usize], add: bool) -> Box2 {
        let e = g.iter().map(|&i| self.ems[i]);
        let wmax = e.clone().map(|e| e[3]).fold(0.0, f64::max);
        let floor = if add { (OWN + SUPPORT) * self.s.sigma } else { 0.0 };
        let pad = (GROUP_PAD * wmax).max(floor).ceil() as i64;
        let (ylo, yhi) = e.clone().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), e| (a.min(e[1]), b.max(e[1])));
        let (xlo, xhi) = e.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), e| (a.min(e[2]), b.max(e[2])));
        let (y0, y1) = span(ylo, yhi, pad, self.h);
        let (x0, x1) = span(xlo, xhi, pad, self.w);
        Box2 { y0, x0, y1, x1 }
    }

    /// A group's window: its pixels and everything else as halo. The level
    /// is locked at zero, so `level`/`shape` are unused.
    pub(crate) fn window(&self, g: &[usize], bb: &Box2, ws: &mut Workspace) -> (Window, Vec<f64>) {
        let (h, w) = (bb.h(), bb.w());
        let own_ems: Vec<Em> = g.iter().map(|&i| self.ems[i]).collect();
        let mut own = Vec::new();
        render(&own_ems, bb.y0, bb.x0, h, w, &mut ws.f, &mut own);
        let mut sub = Vec::with_capacity(h * w);
        let mut halo = Vec::with_capacity(h * w);
        for y in bb.y0..bb.y1 {
            for x in bb.x0..bb.x1 {
                let p = y * self.w + x;
                sub.push(self.d[p]);
                halo.push(self.bg[p] + self.light[p] - own[halo.len()]);
            }
        }
        let win = Window {
            y0: bb.y0,
            x0: bb.x0,
            h,
            w,
            sub,
            halo,
            level: 0.0,
            shape: Vec::new(),
            owned: Vec::new(),
            phi: self.phi,
        };
        (win, own)
    }

    /// Group fit with the level locked; `em` local. Flux bounds use the
    /// frame maximum, widths `slack * sigma`.
    fn fit(&mut self, ws: &mut Workspace, win: &Window, em: &[Em], max_iter: usize, tol_obj: f64) -> Fitted {
        self.stats.fits += 1;
        let (s, k) = (&self.s, em.len());
        let a_max = 8.0 * self.smax / psf::peak_factor(s.sigma) * s.slack.1 * s.slack.1;
        let a_min = boxsearch::A_MIN.max(boxsearch::A_MIN_REL * a_max);
        let (s_lo, s_hi) = (s.slack.0 * s.sigma, s.slack.1 * s.sigma);
        let mut lo = vec![-LEVEL_LOCK];
        let mut hi = vec![LEVEL_LOCK];
        let mut th = vec![0.0];
        for e in em {
            lo.extend_from_slice(&[a_min, -0.5, -0.5, s_lo]);
            hi.extend_from_slice(&[a_max, win.h as f64 - 0.5, win.w as f64 - 0.5, s_hi]);
            th.extend_from_slice(e);
        }
        for q in 0..th.len() {
            th[q] = th[q].clamp(lo[q] + 1e-9, hi[q] - 1e-9);
        }
        let info = lmcl::fit_var_sigma(
            &mut ws.fit,
            &th,
            win.h,
            win.w,
            &win.sub,
            &Bounds::new(&lo, &hi),
            Some(&win.halo),
            FitOpts { max_iter, tol_obj, ..Default::default() },
        );
        let t = ws.fit.theta();
        Fitted {
            i_div: info.i_div,
            b: 0.0,
            flags: (0..k)
                .map(|j| {
                    let mut flag = 0;
                    if !info.converged {
                        flag |= boxsearch::FLAG_NOT_CONVERGED;
                    }
                    if info.stalled {
                        flag |= boxsearch::FLAG_STALLED;
                    }
                    if (1 + 4 * j..5 + 4 * j).any(|q| boxsearch::at_bound(t[q], lo[q], hi[q])) {
                        flag |= boxsearch::FLAG_BOUND;
                    }
                    flag
                })
                .collect(),
            em: (0..k).map(|j| [t[1 + 4 * j], t[2 + 4 * j], t[3 + 4 * j], t[4 + 4 * j]]).collect(),
        }
    }

    /// Whether an add round must test group `g` ([`Config::active_tol`]).
    fn changed(&self, g: &[usize], bb: &Box2) -> bool {
        let tol = self.config.active_tol;
        if tol <= 0.0 || self.tested.len() != self.ems.len() {
            return true;
        }
        let sg = self.s.sigma;
        g.iter().any(|&i| {
            let (e, o) = (self.ems[i], self.tested[i]);
            (e[1] - o[1]).abs() > tol * sg
                || (e[2] - o[2]).abs() > tol * sg
                || (e[3] - o[3]).abs() > tol * sg
                || (e[0] - o[0]).abs() > tol * o[0]
        }) || self.changed_at.iter().any(|&(y, x)| {
            y >= bb.y0 as f64 && y < bb.y1 as f64 && x >= bb.x0 as f64 && x < bb.x1 as f64
        })
    }

    /// One Gauss-Seidel round: every group fitted in turn against the
    /// current light of all others, then the nodes. With `add`, each group
    /// first drops emitters whose removal costs less than `u^2/2` nats, then
    /// adds within `OWN` sigma of its members at `u * kappa`.
    /// Returns `(adds, removals, objective)`.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn round(&mut self, add: bool, ws: &mut Workspace) -> (usize, usize, f64) {
        self.stats.outer += 1;
        let ue = if add {
            self.stats.kappa = self.kappa();
            self.u * self.stats.kappa
        } else {
            self.u
        };
        let (gain_add, gain_rem) = (0.5 * ue * ue * self.phi, 0.5 * self.u * self.u * self.phi);
        let own_r = OWN * self.s.sigma;
        let groups = match self.groups.take() {
            Some(g) if self.config.regroup_on_change => g,
            _ => self.groups(&self.model(), ws),
        };
        let (plain_iter, plain_tol) = (boxsearch::FIT_MAX_ITER, self.config.plain_tol);
        let (full_iter, full_tol) = (boxsearch::FIT_MAX_ITER, boxsearch::FIT_TOL_OBJ);
        let mut changed_at = Vec::new();
        let (mut n_add, mut n_rem) = (0, 0);
        let mut appended: Vec<(Em, u8)> = Vec::new();
        let mut tmp = Vec::new();
        for g in &groups {
            self.stats.max_group = self.stats.max_group.max(g.len());
            let bb = self.group_box_for(g, add || !self.config.plain_pad);
            let (mut win, own) = self.window(g, &bb, ws);
            let (fy, fx) = (bb.y0 as f64, bb.x0 as f64);
            let loc: Vec<Em> = g.iter().map(|&i| {
                let e = self.ems[i];
                [e[0], e[1] - fy, e[2] - fx, e[3]]
            }).collect();
            let test = add && self.changed(g, &bb);
            let mut state = self.fit(ws, &win, &loc, plain_iter, plain_tol);
            // ids[j]: which group member loc[j] came from; None = added.
            let mut ids: Vec<Option<usize>> = (0..g.len()).map(Some).collect();
            if test {
                while !state.em.is_empty() {
                    let mut best: Option<(f64, usize, Fitted)> = None;
                    for k in 0..state.em.len() {
                        let mut em = state.em.clone();
                        em.remove(k);
                        let t = self.fit(ws, &win, &em, self.config.remove_iter, full_tol);
                        let marg = (t.i_div - state.i_div) - gain_rem;
                        if best.as_ref().is_none_or(|b| marg < b.0) {
                            best = Some((marg, k, t));
                        }
                    }
                    let (marg, k, t) = best.unwrap();
                    if marg >= 0.0 {
                        break;
                    }
                    state = t;
                    ids.remove(k);
                    self.stats.removed += 1;
                    n_rem += 1;
                }
                win.owned = (0..win.h * win.w)
                    .map(|p| {
                        let (py, px) = ((p / win.w) as f64, (p % win.w) as f64);
                        state.em.iter().any(|e| (py - e[1]).hypot(px - e[2]) <= own_r)
                            && self.roi.is_none_or(|m| m[(bb.y0 + p / win.w) * self.w + bb.x0 + p % win.w])
                    })
                    .collect();
                if win.owned.iter().any(|&v| v) {
                    while state.em.len() < g.len() + K_MAX {
                        let Some((z, i, a)) = boxsearch::efficient_score(ws, &win, &state, &self.k1) else { break };
                        if !(z > ue) {
                            break;
                        }
                        let mut em = state.em.clone();
                        em.push([a, (i / win.w) as f64, (i % win.w) as f64, self.s.sigma]);
                        let trial = self.fit(ws, &win, &em, full_iter, full_tol);
                        if !(state.i_div - trial.i_div > gain_add) {
                            self.stats.lr_fail += 1;
                            break;
                        }
                        self.stats.adds += 1;
                        n_add += 1;
                        state = trial;
                        ids.push(None);
                    }
                }
            }
            let new: Vec<Em> = state.em.iter().map(|e| [e[0], e[1] + fy, e[2] + fx, e[3]]).collect();
            render(&new, bb.y0, bb.x0, win.h, win.w, &mut ws.f, &mut tmp);
            for (r, y) in (bb.y0..bb.y1).enumerate() {
                for (c, x) in (bb.x0..bb.x1).enumerate() {
                    let q = r * win.w + c;
                    self.light[y * self.w + x] += tmp[q] - own[q];
                }
            }
            for (i, &gi) in g.iter().enumerate() {
                match ids.iter().position(|&q| q == Some(i)) {
                    Some(j) => {
                        self.ems[gi] = new[j];
                        self.flags[gi] = state.flags[j];
                    }
                    None => {
                        changed_at.push((self.ems[gi][1], self.ems[gi][2]));
                        self.ems[gi][0] = 0.0;
                    }
                }
            }
            for (j, q) in ids.iter().enumerate() {
                if q.is_none() {
                    changed_at.push((new[j][1], new[j][2]));
                    appended.push((new[j], state.flags[j]));
                }
            }
        }
        for (e, f) in appended {
            self.ems.push(e);
            self.flags.push(f);
        }
        let keep: Vec<bool> = self.ems.iter().map(|e| e[0] > 0.0).collect();
        let mut it = keep.iter();
        self.ems.retain(|_| *it.next().unwrap());
        let mut it = keep.iter();
        self.flags.retain(|_| *it.next().unwrap());
        if n_add == 0 && n_rem == 0 {
            self.groups = Some(groups);
        }
        if add {
            self.tested = self.ems.clone();
            self.changed_at = changed_at;
        }
        self.redraw(ws);
        self.nodes.irls(self.d, &self.light, OMEGA);
        self.bg = self.nodes.surface();
        (n_add, n_rem, self.objective())
    }

    /// Converge, then alternate add rounds with re-convergence until an add
    /// round changes nothing, within [`MAX_OUTER`] rounds.
    pub fn run(&mut self, ws: &mut Workspace) {
        let mut prev = f64::INFINITY;
        let mut add = false;
        for _ in 0..MAX_OUTER {
            let (n_add, n_rem, obj) = self.round(add, ws);
            let conv = prev - obj < TOL * self.ems.len().max(1) as f64;
            if add {
                add = false;
                if n_add == 0 && n_rem == 0 {
                    break;
                }
            } else if conv {
                add = true;
            }
            prev = obj;
        }
    }
}
