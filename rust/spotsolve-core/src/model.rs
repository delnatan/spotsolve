//! The joint model of a frame: every emitter and the background in one
//! Poisson likelihood,
//!
//! ```text
//! m = B(beta) + sum_k A_k E(r; y_k, s_k) E(c; x_k, s_k)
//! ```
//!
//! `B` is bilinear on nodes [`TILE`] px apart and `E` is the pixel-integrated
//! Gaussian profile. Each emitter is a separable [`Stamp`]: its profile and
//! derivatives along its own rows and columns, over the rectangle where it
//! carries light. Rendering, the Fisher information of a group and the score
//! for one more emitter are sums over stamps and their overlaps.
//!
//! The model is solved by block coordinate descent: emitters in coupled
//! groups by Levenberg-Marquardt with everything else fixed, the nodes by
//! Poisson IRLS with the emitters fixed. Once it has converged each group is
//! tested: an emitter stays only if removing it costs at least `u^2 / 2`
//! nats, and one is added where the residual's efficient score exceeds
//! `u * kappa` and the refit gains `(u * kappa)^2 / 2`. The model is
//! re-converged and tested until a test changes nothing.

use crate::detect::{self, Settings, BG_FLOOR, K_MAX, OWN, SUPPORT};
use crate::linalg::{self, Chol};
use crate::psf;

/// px. Background node spacing: fine enough to follow haze and crowding,
/// coarse enough that every node sees many emitter-free pixels.
pub const TILE: usize = 16;
/// Emitters per group, at most, so the frame never becomes one LM problem.
pub const KG: usize = 12;
/// Pairs coupled more weakly than this (squared first canonical correlation
/// of their parameters) are never joined: Gauss-Seidel between blocks
/// contracts by rho^2 per round, so weak links cost little.
pub const RHO_MIN: f64 = 0.05;
/// Widths: pairs farther apart than this are not examined; a pair's
/// coupling is read over stamps of `PAIR_SUPPORT` widths.
pub const PAIR_REACH: f64 = 6.0;
pub const PAIR_SUPPORT: f64 = 3.0;
/// Widths: an emitter's stamp radius. Beyond it the profile is below 1e-7
/// of its peak.
pub const STAMP: f64 = 6.0;
/// Widths: margin of a group's pixels around its members, and at least the
/// add support `(OWN + SUPPORT) sigma`.
pub const GROUP_PAD: f64 = 3.0;
/// Over-relaxation of each node update: emitters and nodes pull against
/// each other, and overshooting the nodes shortens the zig-zag.
pub const OMEGA: f64 = 1.5;
/// IRLS steps per node update.
pub const IRLS_STEPS: usize = 3;
/// Nats per emitter: a round that improves the objective less has converged.
pub const TOL: f64 = 1e-2;
pub const MAX_OUTER: usize = 80;
/// Nats: a group refit stops here, since the next round moves its halo
/// anyway; decisive fits (additions) go to [`FIT_TOL`].
pub const PLAIN_TOL: f64 = 1e-3;
pub const FIT_TOL: f64 = 1e-6;
pub const FIT_MAX_ITER: usize = 100;
/// LM steps of a leave-one-out removal trial. A truncated trial overstates
/// the cost of removal, so it can only keep an emitter a full trial would
/// remove; removable ones show within a few steps.
pub const REMOVE_ITER: usize = 3;
/// After the first test round, a group is tested again only if a member
/// moved or changed width by more than this many sigma, or flux by this
/// fraction, or the count changed inside its pixels.
pub const ACTIVE_TOL: f64 = 0.05;
/// sigma. Pixels this far from every emitter feed the empirical null.
pub const NULL_FAR: f64 = 3.0;
/// px. The score's reflected border is excluded from the null.
pub const NULL_BORDER: usize = 4;
/// Fewer null pixels than this leave kappa at 1.
pub const NULL_MIN_PX: usize = 100;
/// Gaussian consistency factor of the MAD.
pub const MAD_SCALE: f64 = 1.4826;
/// Fraction of each bound's range kept as a margin inside it.
pub const INTERIOR_FRAC: f64 = 1e-10;
/// A step that would cross a bound goes this fraction of the way there.
pub const STEP_IN: f64 = 0.995;

/// An emitter: flux `a`, centre `(y, x)` in frame pixels, width `s`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Em {
    pub a: f64,
    pub y: f64,
    pub x: f64,
    pub s: f64,
}

impl Em {
    fn from(t: &[f64]) -> Em {
        Em { a: t[0], y: t[1], x: t[2], s: t[3] }
    }

    fn params(&self) -> [f64; 4] {
        [self.a, self.y, self.x, self.s]
    }
}

/// Rows `[r0, r1)` by columns `[c0, c1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub r0: usize,
    pub r1: usize,
    pub c0: usize,
    pub c1: usize,
}

impl Rect {
    pub fn frame(h: usize, w: usize) -> Rect {
        Rect { r0: 0, r1: h, c0: 0, c1: w }
    }

    /// Pixel centres within `ry` rows and `rx` columns of `[ylo, yhi] x
    /// [xlo, xhi]`, inside `self`.
    fn around(&self, ylo: f64, yhi: f64, xlo: f64, xhi: f64, rad: f64) -> Rect {
        let span = |lo: f64, hi: f64, a: usize, b: usize| {
            let s = ((lo - rad).ceil().max(a as f64) as usize).min(b);
            let e = (((hi + rad).floor() + 1.0).max(a as f64) as usize).min(b);
            (s, e.max(s))
        };
        let (r0, r1) = span(ylo, yhi, self.r0, self.r1);
        let (c0, c1) = span(xlo, xhi, self.c0, self.c1);
        Rect { r0, r1, c0, c1 }
    }

    fn meet(&self, o: &Rect) -> Option<Rect> {
        let r = Rect { r0: self.r0.max(o.r0), r1: self.r1.min(o.r1), c0: self.c0.max(o.c0), c1: self.c1.min(o.c1) };
        (r.r0 < r.r1 && r.c0 < r.c1).then_some(r)
    }

    fn rows(&self) -> usize {
        self.r1 - self.r0
    }

    fn cols(&self) -> usize {
        self.c1 - self.c0
    }

    fn n(&self) -> usize {
        self.rows() * self.cols()
    }

    /// Index of frame pixel `(r, c)` in an array covering `self`.
    fn at(&self, r: usize, c: usize) -> usize {
        (r - self.r0) * self.cols() + c - self.c0
    }

    fn contains(&self, y: f64, x: f64) -> bool {
        y >= self.r0 as f64 && y < self.r1 as f64 && x >= self.c0 as f64 && x < self.c1 as f64
    }
}

/// An emitter's profile `E(r) E(c)` on its rectangle, with the derivatives
/// of each factor by centre (`dy`, `dx`) and width (`sy`, `sx`).
pub struct Stamp {
    pub rc: Rect,
    a: f64,
    ey: Vec<f64>,
    dy: Vec<f64>,
    sy: Vec<f64>,
    ex: Vec<f64>,
    dx: Vec<f64>,
    sx: Vec<f64>,
}

impl Stamp {
    /// Out to `rad` widths from the centre, inside `within`.
    pub fn new(e: &Em, rad: f64, within: &Rect) -> Stamp {
        let rc = within.around(e.y, e.y, e.x, e.x, rad * e.s);
        let axis = |lo: usize, hi: usize, c: f64| {
            let t: Vec<f64> = (lo..hi).map(|v| v as f64).collect();
            let (mut f, mut d, mut s) = (vec![0.0; t.len()], vec![0.0; t.len()], vec![0.0; t.len()]);
            psf::factors_axis_sigma(&t, &[c], e.s, &mut f, &mut d, &mut s);
            (f, d, s)
        };
        let (ey, dy, sy) = axis(rc.r0, rc.r1, e.y);
        let (ex, dx, sx) = axis(rc.c0, rc.c1, e.x);
        Stamp { rc, a: e.a, ey, dy, sy, ex, dx, sx }
    }

    fn value(&self, r: usize, c: usize) -> f64 {
        self.a * self.ey[r - self.rc.r0] * self.ex[c - self.rc.c0]
    }

    /// `d value / d (a, y, x, s)` at `(r, c)`.
    fn jac(&self, r: usize, c: usize) -> [f64; 4] {
        let (i, j) = (r - self.rc.r0, c - self.rc.c0);
        let (ey, ex, a) = (self.ey[i], self.ex[j], self.a);
        [ey * ex, a * self.dy[i] * ex, a * ey * self.dx[j], a * (self.sy[i] * ex + ey * self.sx[j])]
    }

    /// `img += sign * light`, where `img` covers `on`.
    fn paint(&self, img: &mut [f64], on: &Rect, sign: f64) {
        let Some(rc) = self.rc.meet(on) else { return };
        for r in rc.r0..rc.r1 {
            for c in rc.c0..rc.c1 {
                img[on.at(r, c)] += sign * self.value(r, c);
            }
        }
    }
}

/// Bilinear background on a node lattice `ny x nx`, [`TILE`] px apart,
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

    fn eval(&self, beta: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.h * self.w);
        for y in 0..self.h {
            for x in 0..self.w {
                out.push(self.stencil(y, x).iter().map(|&(n, t)| t * beta[n]).sum());
            }
        }
        out
    }

    pub fn surface(&self) -> Vec<f64> {
        self.eval(&self.beta)
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
        for _ in 0..IRLS_STEPS {
            let m: Vec<f64> = self.eval(&b).iter().zip(light).map(|(v, l)| (v + l).max(BG_FLOOR)).collect();
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

/// Poisson I-divergence `sum d log(d/m) - (d - m)`, with `0 log 0 = 0`.
fn idiv(d: &[f64], m: &[f64]) -> f64 {
    d.iter()
        .zip(m)
        .map(|(&d, &m)| if d > 0.0 { d * (d / m).ln() } else { 0.0 } - (d - m))
        .sum()
}

/// A group's pixels: the data, and everything the group does not own
/// (background and every other emitter) as `halo`.
struct Patch {
    rc: Rect,
    d: Vec<f64>,
    halo: Vec<f64>,
    /// Parameter bounds `(a, y, x, s)`, shared by every emitter here.
    lo: [f64; 4],
    hi: [f64; 4],
}

impl Patch {
    fn stamps(&self, ems: &[Em]) -> Vec<Stamp> {
        ems.iter().map(|e| Stamp::new(e, STAMP, &self.rc)).collect()
    }

    fn model(&self, st: &[Stamp]) -> Vec<f64> {
        let mut m = self.halo.clone();
        for s in st {
            s.paint(&mut m, &self.rc, 1.0);
        }
        for v in &mut m {
            *v = v.max(BG_FLOOR);
        }
        m
    }

    /// Fisher information `J^T W J` and score `J^T W (m - d+)`, `W = 1/m`,
    /// for `[level?, (a, y, x, s) per stamp]`. The level's column is 1 over
    /// the patch; an emitter's is zero outside its stamp, so blocks are
    /// sums over stamp overlaps.
    fn normal(&self, st: &[Stamp], m: &[f64], level: bool) -> (Vec<f64>, Vec<f64>) {
        let o = level as usize;
        let p = o + 4 * st.len();
        let (mut f, mut g) = (vec![0.0; p * p], vec![0.0; p]);
        if level {
            for (&m, &d) in m.iter().zip(&self.d) {
                f[0] += 1.0 / m;
                g[0] += (m - d.max(0.0)) / m;
            }
        }
        for (i, si) in st.iter().enumerate() {
            let qi = o + 4 * i;
            for r in si.rc.r0..si.rc.r1 {
                for c in si.rc.c0..si.rc.c1 {
                    let k = self.rc.at(r, c);
                    let (wt, res) = (1.0 / m[k], (m[k] - self.d[k].max(0.0)) / m[k]);
                    let j = si.jac(r, c);
                    for a in 0..4 {
                        g[qi + a] += j[a] * res;
                        if level {
                            f[qi + a] += j[a] * wt;
                        }
                        for b in a..4 {
                            f[(qi + a) * p + qi + b] += j[a] * j[b] * wt;
                        }
                    }
                }
            }
            for (l, sl) in st.iter().enumerate().skip(i + 1) {
                let Some(ov) = si.rc.meet(&sl.rc) else { continue };
                let ql = o + 4 * l;
                for r in ov.r0..ov.r1 {
                    for c in ov.c0..ov.c1 {
                        let wt = 1.0 / m[self.rc.at(r, c)];
                        let (ji, jl) = (si.jac(r, c), sl.jac(r, c));
                        for a in 0..4 {
                            for b in 0..4 {
                                f[(qi + a) * p + ql + b] += ji[a] * jl[b] * wt;
                            }
                        }
                    }
                }
            }
        }
        for a in 0..p {
            for b in 0..a {
                f[a * p + b] = f[b * p + a];
            }
        }
        (f, g)
    }
}

/// A group fit: emitters, the I-divergence on the patch, and each emitter's
/// diagnostics.
#[derive(Clone)]
struct Fit {
    ems: Vec<Em>,
    idiv: f64,
    flags: Vec<u8>,
}

/// Largest feasible, information-scaled gradient component: each parameter
/// may move one conditional standard error, or to its bound, downhill.
fn projected_score(t: &[f64], g: &[f64], f: &[f64], lo: &[f64; 4], hi: &[f64; 4]) -> f64 {
    let p = t.len();
    (0..p)
        .map(|q| {
            let room = if g[q] >= 0.0 { t[q] - lo[q % 4] } else { hi[q % 4] - t[q] };
            g[q].abs() * room.max(0.0).min(1.0 / f[q * p + q].max(1e-30).sqrt())
        })
        .fold(0.0, f64::max)
}

/// Bounded Levenberg-Marquardt (Fisher scoring) on the patch, with
/// Coleman-Li affine scaling: parameters stay strictly inside their bounds,
/// and a coordinate's step shrinks with its distance to the bound it heads
/// for, so a flux cannot collapse onto its floor in one step while the
/// positions have yet to adapt.
fn fit(p: &Patch, ems: &[Em], max_iter: usize, tol: f64) -> Fit {
    let n = 4 * ems.len();
    let (lo, hi) = (|q: usize| p.lo[q % 4], |q: usize| p.hi[q % 4]);
    let inside = |q: usize, v: f64| {
        let margin = INTERIOR_FRAC * (hi(q) - lo(q));
        v.clamp(lo(q) + margin, hi(q) - margin)
    };
    let mut t: Vec<f64> = ems.iter().flat_map(|e| e.params()).enumerate().map(|(q, v)| inside(q, v)).collect();
    let ems_of = |t: &[f64]| t.chunks(4).map(Em::from).collect::<Vec<_>>();
    let mut st = p.stamps(&ems_of(&t));
    let mut m = p.model(&st);
    let mut cur = idiv(&p.d, &m);
    let (mut lam, mut nu) = (1e-2, 2.0);
    let (mut converged, mut stalled) = (n == 0, false);
    let mut chol = Chol::new(n.max(1));
    for _ in 0..max_iter {
        if converged {
            break;
        }
        let (f, g) = p.normal(&st, &m, false);
        if projected_score(&t, &g, &f, &p.lo, &p.hi) <= (2.0 * tol).sqrt() {
            converged = true;
            break;
        }
        // v: distance to the bound the gradient heads for.
        let v: Vec<f64> =
            (0..n).map(|q| if g[q] >= 0.0 { t[q] - lo(q) } else { hi(q) - t[q] }.max(1e-12)).collect();
        let mut accepted = false;
        while lam < 1e12 {
            let mut a = f.clone();
            for q in 0..n {
                a[q * n + q] += g[q].abs() / v[q] + lam / v[q];
            }
            if !chol.factor(&a, n) {
                lam *= 10.0;
                continue;
            }
            let mut step: Vec<f64> = g.iter().map(|v| -v).collect();
            chol.solve_in_place(&mut step);
            let t2: Vec<f64> = (0..n)
                .map(|q| {
                    let bound = if step[q] > 0.0 { hi(q) } else { lo(q) };
                    let d = if (t[q] + step[q] - bound) * step[q] > 0.0 { STEP_IN * (bound - t[q]) } else { step[q] };
                    inside(q, t[q] + d)
                })
                .collect();
            let dt: Vec<f64> = t2.iter().zip(&t).map(|(a, b)| a - b).collect();
            let fd: f64 = (0..n).map(|q| dt[q] * (0..n).map(|r| f[q * n + r] * dt[r]).sum::<f64>()).sum();
            let pred = -dt.iter().zip(&g).map(|(a, b)| a * b).sum::<f64>() - 0.5 * fd;
            let st2 = p.stamps(&ems_of(&t2));
            let m2 = p.model(&st2);
            let i2 = idiv(&p.d, &m2);
            let rho = if pred > 0.0 { (cur - i2) / pred } else { -1.0 };
            if i2 < cur && rho > 1e-4 {
                (t, st, m, cur) = (t2, st2, m2, i2);
                lam = (lam * (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3))).max(1e-12);
                nu = 2.0;
                accepted = true;
                break;
            }
            lam *= nu;
            nu *= 2.0;
        }
        if !accepted {
            stalled = true;
            break;
        }
    }
    if !converged && !stalled {
        let (f, g) = p.normal(&st, &m, false);
        converged = projected_score(&t, &g, &f, &p.lo, &p.hi) <= (2.0 * tol).sqrt();
    }
    let flags = t
        .chunks(4)
        .map(|e| {
            let mut flag = 0;
            if !converged {
                flag |= detect::FLAG_NOT_CONVERGED;
            }
            if stalled {
                flag |= detect::FLAG_STALLED;
            }
            if (0..4).any(|q| detect::at_bound(e[q], p.lo[q], p.hi[q])) {
                flag |= detect::FLAG_BOUND;
            }
            flag
        })
        .collect();
    Fit { ems: ems_of(&t), idiv: cur, flags }
}

/// Separable correlation with a symmetric kernel, zero outside `rows x cols`.
fn blur(src: &[f64], rows: usize, cols: usize, k: &[f64]) -> Vec<f64> {
    let r = (k.len() / 2) as isize;
    let pass = |src: &[f64], along_rows: bool| {
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut acc = 0.0;
                for (t, &kv) in k.iter().enumerate() {
                    let o = t as isize - r;
                    let (y, x) = if along_rows { (i as isize + o, j as isize) } else { (i as isize, j as isize + o) };
                    if y >= 0 && x >= 0 && (y as usize) < rows && (x as usize) < cols {
                        acc += src[y as usize * cols + x as usize] * kv;
                    }
                }
                out[i * cols + j] = acc;
            }
        }
        out
    };
    pass(&pass(src, true), false)
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub fits: usize,
    pub adds: usize,
    pub removed: usize,
    /// Additions that passed the score but failed the LR confirmation.
    pub lr_fail: usize,
    pub outer: usize,
    pub tests: usize,
    /// Empirical-null scale of the last test round; 1 before one.
    pub kappa: f64,
}

/// An emitter of the model, its latest fit diagnostics, and where it stood
/// at the last test round.
#[derive(Clone, Copy, Debug)]
pub struct Emitter {
    pub e: Em,
    pub flags: u8,
    seen: Em,
}

/// The joint model of one frame (or ROI crop).
pub struct Model<'a> {
    d: &'a [f64],
    pub h: usize,
    pub w: usize,
    s: Settings,
    pub phi: f64,
    pub u: f64,
    roi: Option<&'a [bool]>,
    /// Flux bounds: from the frame maximum, so a fit never runs away.
    a_min: f64,
    a_max: f64,
    /// Unit-flux reference-width profile along one axis.
    k1: Vec<f64>,
    pub ems: Vec<Emitter>,
    pub nodes: Nodes,
    /// `nodes.surface()` and the emitters' light, `h * w` each.
    pub bg: Vec<f64>,
    pub light: Vec<f64>,
    pub stats: Stats,
    /// The partition, kept until the count changes.
    groups: Option<Vec<Vec<usize>>>,
    /// Where the last test round added or removed.
    changed_at: Vec<(f64, f64)>,
}

impl<'a> Model<'a> {
    /// Start from emitters `e0`: nodes fitted to the map `bg0`, then one IRLS
    /// update against `e0`'s light. `roi` limits where emitters are added.
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
    ) -> Self {
        assert_eq!(d.len(), h * w);
        let smax = d.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v)).max(1.0);
        let a_max = 8.0 * smax / psf::peak_factor(s.sigma) * s.slack.1 * s.slack.1;
        let mut nodes = Nodes::new(h, w);
        nodes.fit_map(bg0);
        let mut md = Self {
            d,
            h,
            w,
            s: *s,
            phi,
            u,
            roi,
            a_min: detect::A_MIN.max(detect::A_MIN_REL * a_max),
            a_max,
            k1: detect::psf_kernel1d(s.sigma),
            ems: e0.iter().map(|&e| Emitter { e, flags: 0, seen: e }).collect(),
            nodes,
            bg: Vec::new(),
            light: Vec::new(),
            stats: Stats { kappa: 1.0, ..Stats::default() },
            groups: None,
            changed_at: Vec::new(),
        };
        md.redraw();
        md.nodes.irls(d, &md.light, 1.0);
        md.bg = md.nodes.surface();
        md
    }

    fn frame(&self) -> Rect {
        Rect::frame(self.h, self.w)
    }

    /// `light <- sum` of every emitter's stamp.
    fn redraw(&mut self) {
        let frame = self.frame();
        self.light = vec![0.0; self.h * self.w];
        for em in &self.ems {
            Stamp::new(&em.e, STAMP, &frame).paint(&mut self.light, &frame, 1.0);
        }
    }

    /// `max(B + light, BG_FLOOR)`.
    pub fn model(&self) -> Vec<f64> {
        self.bg.iter().zip(&self.light).map(|(b, l)| (b + l).max(BG_FLOOR)).collect()
    }

    /// Dispersion-scaled I-divergence of the frame, nats.
    pub fn objective(&self) -> f64 {
        idiv(self.d, &self.model()) / self.phi
    }

    /// Squared first canonical correlation of two emitters' `(a, y, x, s)`
    /// under the Fisher information of `m`: the largest eigenvalue of
    /// `F11^-1 F12 F22^-1 F21`, the per-round contraction of fitting them
    /// apart. 1 when a block is singular.
    fn rho2(&self, a: &Em, b: &Em, m: &[f64]) -> f64 {
        let frame = self.frame();
        let (sa, sb) = (Stamp::new(a, PAIR_SUPPORT, &frame), Stamp::new(b, PAIR_SUPPORT, &frame));
        let block = |s: &Stamp, t: &Stamp, rc: Rect| {
            let mut f = [0.0; 16];
            for r in rc.r0..rc.r1 {
                for c in rc.c0..rc.c1 {
                    let wt = 1.0 / m[r * self.w + c];
                    let (js, jt) = (s.jac(r, c), t.jac(r, c));
                    for i in 0..4 {
                        for j in 0..4 {
                            f[i * 4 + j] += js[i] * jt[j] * wt;
                        }
                    }
                }
            }
            f
        };
        let (f11, f22) = (block(&sa, &sa, sa.rc), block(&sb, &sb, sb.rc));
        let f12 = sa.rc.meet(&sb.rc).map_or([0.0; 16], |ov| block(&sa, &sb, ov));
        let (mut c1, mut c2) = (Chol::new(4), Chol::new(4));
        if !c1.factor(&f11, 4) || !c2.factor(&f22, 4) {
            return 1.0;
        }
        // T = L1^-1 F12 F22^-1 F21 by columns, then S = L1^-1 T^T, which is
        // symmetric: S = L1^-1 (F12 F22^-1 F21) L1^-T.
        let mut x = [0.0; 4];
        let mut s = [0.0; 16];
        for c in 0..4 {
            let f21c: [f64; 4] = std::array::from_fn(|r| f12[c * 4 + r]);
            c2.solve(&f21c, &mut x);
            let mut v: [f64; 4] = std::array::from_fn(|r| (0..4).map(|k| f12[r * 4 + k] * x[k]).sum());
            c1.solve_lower_in_place(&mut v);
            for r in 0..4 {
                s[r * 4 + c] = v[r];
            }
        }
        for r in 0..4 {
            let mut v: [f64; 4] = std::array::from_fn(|c| s[r * 4 + c]);
            c1.solve_lower_in_place(&mut v);
            for c in 0..4 {
                s[r * 4 + c] = v[c];
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

    /// Groups: union-find over pairs within [`PAIR_REACH`] widths, strongest
    /// coupling first, joining two groups only while the union holds at
    /// most [`KG`] and the coupling is at least [`RHO_MIN`].
    pub fn partition(&self) -> Vec<Vec<usize>> {
        let n = self.ems.len();
        let mut pairs = Vec::new();
        if n > 1 {
            let reach = PAIR_REACH * self.ems.iter().map(|e| e.e.s).fold(0.0, f64::max);
            let cell = |v: f64| (v / reach).floor() as i64;
            let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> = Default::default();
            for (i, e) in self.ems.iter().enumerate() {
                grid.entry((cell(e.e.y), cell(e.e.x))).or_default().push(i);
            }
            let m = self.model();
            for (i, a) in self.ems.iter().enumerate() {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        for &j in grid.get(&(cell(a.e.y) + dy, cell(a.e.x) + dx)).into_iter().flatten() {
                            let b = &self.ems[j];
                            if j > i && (a.e.y - b.e.y).hypot(a.e.x - b.e.x) <= reach {
                                pairs.push((i, j, self.rho2(&a.e, &b.e, &m)));
                            }
                        }
                    }
                }
            }
        }
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
        for &(i, j, _) in pairs.iter().take_while(|p| p.2 >= RHO_MIN) {
            let (a, b) = (find(&mut parent, i), find(&mut parent, j));
            if a != b && size[a] + size[b] <= KG {
                let (big, small) = if size[a] >= size[b] { (a, b) } else { (b, a) };
                parent[small] = big;
                size[big] += size[small];
            }
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
    /// at 1. It is 1 where the model describes the data and grows where it
    /// misfits (PSF wings, haze), raising the bar for additions there.
    pub fn kappa(&self) -> f64 {
        let (h, w) = (self.h, self.w);
        let b = NULL_BORDER;
        if h <= 2 * b || w <= 2 * b {
            return 1.0;
        }
        let m = self.model();
        let r: Vec<f64> = self.d.iter().zip(&m).map(|(d, m)| d - m).collect();
        let var: Vec<f64> = m.iter().map(|m| self.phi * m).collect();
        let z = detect::detection_map(&r, &var, h, w, self.s.sigma);
        let reach = NULL_FAR * self.s.sigma;
        let mut near = vec![false; h * w];
        for em in &self.ems {
            let rc = self.frame().around(em.e.y, em.e.y, em.e.x, em.e.x, reach);
            for y in rc.r0..rc.r1 {
                for x in rc.c0..rc.c1 {
                    near[y * w + x] |= (y as f64 - em.e.y).hypot(x as f64 - em.e.x) <= reach;
                }
            }
        }
        let zf: Vec<f64> =
            (b..h - b).flat_map(|y| (b..w - b).map(move |x| y * w + x)).filter(|&p| !near[p]).map(|p| z[p]).collect();
        if zf.len() <= NULL_MIN_PX {
            return 1.0;
        }
        let med = detect::median(&zf);
        let dev: Vec<f64> = zf.iter().map(|v| (v - med).abs()).collect();
        (MAD_SCALE * detect::median(&dev)).max(1.0)
    }

    /// A group's pixels: its members plus the larger of [`GROUP_PAD`]
    /// widths and the add support `(OWN + SUPPORT) sigma`.
    pub fn group_rect(&self, g: &[usize]) -> Rect {
        let e = || g.iter().map(|&i| self.ems[i].e);
        let smax = e().map(|e| e.s).fold(0.0, f64::max);
        let pad = (GROUP_PAD * smax).max((OWN + SUPPORT) * self.s.sigma);
        let lohi = |v: fn(&Em) -> f64| e().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| (a.min(v(&x)), b.max(v(&x))));
        let ((ylo, yhi), (xlo, xhi)) = (lohi(|e| e.y), lohi(|e| e.x));
        self.frame().around(ylo, yhi, xlo, xhi, pad)
    }

    /// Group `g`'s patch: data on `rc`, and as halo the background plus the
    /// light of every emitter outside the group.
    fn patch(&self, g: &[usize], rc: Rect) -> Patch {
        let mut d = Vec::with_capacity(rc.n());
        let mut halo = Vec::with_capacity(rc.n());
        for r in rc.r0..rc.r1 {
            for c in rc.c0..rc.c1 {
                d.push(self.d[r * self.w + c]);
                halo.push(self.bg[r * self.w + c] + self.light[r * self.w + c]);
            }
        }
        for &i in g {
            Stamp::new(&self.ems[i].e, STAMP, &rc).paint(&mut halo, &rc, -1.0);
        }
        let (sl, sh) = (self.s.slack.0 * self.s.sigma, self.s.slack.1 * self.s.sigma);
        Patch {
            rc,
            d,
            halo,
            lo: [self.a_min, rc.r0 as f64 - 0.5, rc.c0 as f64 - 0.5, sl],
            hi: [self.a_max, rc.r1 as f64 - 0.5, rc.c1 as f64 - 0.5, sh],
        }
    }

    /// Best pixel of `owned` for one more reference-width emitter:
    /// `(z, row, col, one-step flux)`. `z = S / sqrt(I_eff)`, where `S` is
    /// the score of its flux and `I_eff` its information less the projection
    /// onto `[level, members]`, so a fitted neighbour's light cannot pass
    /// for a new emitter.
    fn score(&self, p: &Patch, ems: &[Em], owned: &[bool]) -> Option<(f64, usize, usize, f64)> {
        let (rows, cols, n) = (p.rc.rows(), p.rc.cols(), p.rc.n());
        let st = p.stamps(ems);
        let m = p.model(&st);
        let wt: Vec<f64> = m.iter().map(|m| 1.0 / (self.phi * m)).collect();
        let resid: Vec<f64> = (0..n).map(|k| (p.d[k] - m[k]) * wt[k]).collect();
        let k2: Vec<f64> = self.k1.iter().map(|v| v * v).collect();
        let s = blur(&resid, rows, cols, &self.k1);
        let igg = blur(&wt, rows, cols, &k2);
        // c[q][k] = sum_j g_k(j) J_q(j) wt(j), for the level and each stamp.
        let np = 1 + 4 * st.len();
        let mut c = Vec::with_capacity(np);
        c.push(blur(&wt, rows, cols, &self.k1));
        for sq in &st {
            let mut cols4 = vec![vec![0.0; n]; 4];
            for r in sq.rc.r0..sq.rc.r1 {
                for cc in sq.rc.c0..sq.rc.c1 {
                    let k = p.rc.at(r, cc);
                    for (a, v) in sq.jac(r, cc).iter().enumerate() {
                        cols4[a][k] = v * wt[k];
                    }
                }
            }
            c.extend(cols4.iter().map(|img| blur(img, rows, cols, &self.k1)));
        }
        let f: Vec<f64> = p.normal(&st, &m, true).0.iter().map(|v| v / self.phi).collect();
        let mut chol = Chol::new(np);
        let proj = chol.factor(&f, np);
        let (mut cq, mut x) = (vec![0.0; np], vec![0.0; np]);
        let mut best: Option<(f64, usize, usize, f64)> = None;
        for k in (0..n).filter(|&k| owned[k]) {
            let mut ieff = igg[k];
            if proj {
                for q in 0..np {
                    cq[q] = c[q][k];
                }
                chol.solve(&cq, &mut x);
                ieff -= cq.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>();
            }
            let ieff = ieff.max(1e-12);
            let z = s[k] / ieff.sqrt();
            if best.is_none_or(|b| z > b.0) {
                best = Some((z, p.rc.r0 + k / cols, p.rc.c0 + k % cols, s[k] / ieff));
            }
        }
        best
    }

    /// Whether a test round must test group `g` ([`ACTIVE_TOL`]).
    fn must_test(&self, g: &[usize], rc: &Rect) -> bool {
        let (tol, sg) = (ACTIVE_TOL, self.s.sigma);
        self.stats.tests == 0
            || g.iter().any(|&i| {
                let (e, o) = (self.ems[i].e, self.ems[i].seen);
                (e.y - o.y).abs() > tol * sg
                    || (e.x - o.x).abs() > tol * sg
                    || (e.s - o.s).abs() > tol * sg
                    || (e.a - o.a).abs() > tol * o.a
            })
            || self.changed_at.iter().any(|&(y, x)| rc.contains(y, x))
    }

    /// One Gauss-Seidel round: every group refitted in turn against the
    /// current light of all others, then the nodes. With `test`, each group
    /// first drops emitters whose removal costs less than `u^2/2` nats, then
    /// adds within `OWN` sigma of its members at `u * kappa`.
    /// Returns `(adds, removals, objective)`.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn round(&mut self, test: bool) -> (usize, usize, f64) {
        self.stats.outer += 1;
        let ut = if test {
            // The null is read off the residual, which is honest only once
            // the model holds the frame's emitters.
            self.stats.kappa = if self.stats.tests == 0 { 1.0 } else { self.kappa() };
            self.u * self.stats.kappa
        } else {
            self.u
        };
        let (gain_add, gain_rem) = (0.5 * ut * ut * self.phi, 0.5 * self.u * self.u * self.phi);
        let own_r = OWN * self.s.sigma;
        let frame = self.frame();
        let groups = self.groups.take().unwrap_or_else(|| self.partition());
        let (mut n_add, mut n_rem) = (0, 0);
        let mut changed_at = Vec::new();
        let mut dead = vec![false; self.ems.len()];
        let mut born = Vec::new();
        for g in &groups {
            let rc = self.group_rect(g);
            let p = self.patch(g, rc);
            let em0: Vec<Em> = g.iter().map(|&i| self.ems[i].e).collect();
            let mut state = fit(&p, &em0, FIT_MAX_ITER, PLAIN_TOL);
            self.stats.fits += 1;
            // origin[j]: the member state.ems[j] came from; None if added.
            let mut origin: Vec<Option<usize>> = (0..g.len()).map(Some).collect();
            if test && self.must_test(g, &rc) {
                while !state.ems.is_empty() {
                    let (cost, k, trial) = (0..state.ems.len())
                        .map(|k| {
                            let mut em = state.ems.clone();
                            em.remove(k);
                            let t = fit(&p, &em, REMOVE_ITER, FIT_TOL);
                            (t.idiv - state.idiv, k, t)
                        })
                        .min_by(|a, b| a.0.total_cmp(&b.0))
                        .unwrap();
                    self.stats.fits += state.ems.len();
                    if cost >= gain_rem {
                        break;
                    }
                    state = trial;
                    origin.remove(k);
                    n_rem += 1;
                }
                let owned: Vec<bool> = (0..rc.n())
                    .map(|k| {
                        let (r, c) = (rc.r0 + k / rc.cols(), rc.c0 + k % rc.cols());
                        state.ems.iter().any(|e| (r as f64 - e.y).hypot(c as f64 - e.x) <= own_r)
                            && self.roi.is_none_or(|m| m[r * self.w + c])
                    })
                    .collect();
                while state.ems.len() < g.len() + K_MAX {
                    let Some((z, r, c, a)) = self.score(&p, &state.ems, &owned) else { break };
                    if !(z > ut) {
                        break;
                    }
                    let mut em = state.ems.clone();
                    em.push(Em { a, y: r as f64, x: c as f64, s: self.s.sigma });
                    let trial = fit(&p, &em, FIT_MAX_ITER, FIT_TOL);
                    self.stats.fits += 1;
                    if !(state.idiv - trial.idiv > gain_add) {
                        self.stats.lr_fail += 1;
                        break;
                    }
                    state = trial;
                    origin.push(None);
                    n_add += 1;
                }
            }
            for &i in g {
                Stamp::new(&self.ems[i].e, STAMP, &frame).paint(&mut self.light, &frame, -1.0);
            }
            for (j, (e, &flags)) in state.ems.iter().zip(&state.flags).enumerate() {
                Stamp::new(e, STAMP, &frame).paint(&mut self.light, &frame, 1.0);
                match origin[j] {
                    Some(i) => {
                        self.ems[g[i]].e = *e;
                        self.ems[g[i]].flags = flags;
                    }
                    None => {
                        born.push(Emitter { e: *e, flags, seen: *e });
                        changed_at.push((e.y, e.x));
                    }
                }
            }
            for (i, &gi) in g.iter().enumerate() {
                if !origin.contains(&Some(i)) {
                    dead[gi] = true;
                    changed_at.push((self.ems[gi].e.y, self.ems[gi].e.x));
                }
            }
        }
        let mut alive = dead.iter().map(|d| !d);
        self.ems.retain(|_| alive.next().unwrap());
        self.ems.extend(born);
        if n_add == 0 && n_rem == 0 {
            self.groups = Some(groups);
        }
        if test {
            self.stats.tests += 1;
            self.stats.adds += n_add;
            self.stats.removed += n_rem;
            self.changed_at = changed_at;
            for em in &mut self.ems {
                em.seen = em.e;
            }
        }
        self.redraw();
        self.nodes.irls(self.d, &self.light, OMEGA);
        self.bg = self.nodes.surface();
        (n_add, n_rem, self.objective())
    }

    /// Converge, then alternate test rounds with re-convergence until a test
    /// round changes nothing, within [`MAX_OUTER`] rounds.
    pub fn run(&mut self) {
        let mut prev = f64::INFINITY;
        let mut test = false;
        for _ in 0..MAX_OUTER {
            let (n_add, n_rem, obj) = self.round(test);
            let converged = prev - obj < TOL * self.ems.len().max(1) as f64;
            if test {
                test = false;
                if n_add == 0 && n_rem == 0 {
                    break;
                }
            } else if converged {
                test = true;
            }
            prev = obj;
        }
    }

    /// Each final group and its Fisher information of `[level, members]`,
    /// for uncertainties: a free local level stands in for the estimated
    /// nodes.
    pub fn group_information(&self) -> Vec<(Vec<usize>, Vec<f64>)> {
        self.partition()
            .into_iter()
            .map(|g| {
                let p = self.patch(&g, self.group_rect(&g));
                let st = p.stamps(&g.iter().map(|&i| self.ems[i].e).collect::<Vec<_>>());
                let f = p.normal(&st, &p.model(&st), true).0;
                (g, f)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::{FLAG_BOUND, FLAG_NOT_CONVERGED, FP_PER_MPX, SLACK};

    fn em(a: f64, y: f64, x: f64, s: f64) -> Em {
        Em { a, y, x, s }
    }

    #[test]
    fn a_stamp_is_the_psf_model_on_its_rectangle() {
        let (h, w) = (23, 27);
        let e = em(1500.0, 11.3, 12.8, 1.4);
        let mut dense = vec![0.0; h * w];
        psf::model_var_sigma_ax(
            &[0.0, e.a, e.y, e.x, e.s], &psf::local_axis(h), &psf::local_axis(w),
            None, &mut psf::Factors::new(h, w, 1), &mut dense,
        );
        let frame = Rect::frame(h, w);
        let mut img = vec![0.0; h * w];
        Stamp::new(&e, STAMP, &frame).paint(&mut img, &frame, 1.0);
        for (a, b) in img.iter().zip(&dense) {
            assert!((a - b).abs() < 1e-6 * e.a, "{a} vs {b}");
        }
    }

    #[test]
    fn the_stamp_jacobian_matches_finite_differences() {
        let frame = Rect::frame(21, 21);
        let e = em(900.0, 10.2, 9.7, 1.3);
        let st = Stamp::new(&e, STAMP, &frame);
        for (r, c) in [(10, 10), (8, 12), (13, 7)] {
            let j = st.jac(r, c);
            for q in 0..4 {
                let h = 1e-6 * e.params()[q].abs().max(1.0);
                let (mut up, mut dn) = (e.params(), e.params());
                up[q] += h;
                dn[q] -= h;
                let v = |t: [f64; 4]| Stamp::new(&Em::from(&t), STAMP, &frame).value(r, c);
                let fd = (v(up) - v(dn)) / (2.0 * h);
                assert!((j[q] - fd).abs() <= 1e-5 * (1.0 + fd.abs()), "({r}, {c}) q {q}: {} vs {fd}", j[q]);
            }
        }
    }

    /// A 17x17 patch holding one emitter of width 2.8 on a flat 20.
    fn patch(slack_hi: f64) -> Patch {
        let rc = Rect::frame(17, 17);
        let mut d = vec![20.0; rc.n()];
        Stamp::new(&em(1500.0, 8.2, 8.4, 2.8), 20.0, &rc).paint(&mut d, &rc, 1.0);
        Patch { rc, d, halo: vec![20.0; rc.n()], lo: [1e-3, -0.5, -0.5, 0.7], hi: [1e6, 16.5, 16.5, slack_hi] }
    }

    #[test]
    fn fit_flags_distinguish_iteration_limits_and_active_bounds() {
        let start = [em(1200.0, 8.0, 8.0, 1.5)];
        let unfinished = fit(&patch(SLACK.1), &start, 0, FIT_TOL);
        assert_eq!(unfinished.flags, vec![FLAG_NOT_CONVERGED]);
        let bounded = fit(&patch(SLACK.1), &start, FIT_MAX_ITER, FIT_TOL);
        assert_eq!(bounded.flags, vec![FLAG_BOUND]);
        assert!((bounded.ems[0].s - SLACK.1).abs() < 1e-5);
        let free = fit(&patch(3.5), &start, FIT_MAX_ITER, FIT_TOL);
        assert_eq!(free.flags, vec![0]);
        assert!((free.ems[0].s - 2.8).abs() < 1e-4, "width {}", free.ems[0].s);
        assert!((free.ems[0].a - 1500.0).abs() < 0.1, "flux {}", free.ems[0].a);
    }

    #[test]
    fn coupling_falls_from_coincident_to_far_apart() {
        let (h, w) = (40, 60);
        let d = vec![20.0; h * w];
        let s = Settings { sigma: 1.2, fp_per_mpx: FP_PER_MPX, slack: SLACK };
        let md = Model::new(&d, h, w, &[], &d, 1.0, 4.0, &s, None);
        let m = vec![20.0; h * w];
        let a = em(1000.0, 20.0, 20.0, 1.2);
        let rho = |dx: f64| md.rho2(&a, &em(1000.0, 20.0, 20.0 + dx, 1.2), &m);
        let (r0, r1, r3, r8) = (rho(0.0), rho(1.0), rho(3.0), rho(8.0));
        assert!(r0 > 0.999 && r0 > r1 && r1 > r3 && r3 > r8, "{r0} {r1} {r3} {r8}");
        assert!(r8 < RHO_MIN, "{r8}");
    }
}
