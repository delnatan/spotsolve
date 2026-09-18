//! Frame-to-frame linking: the motion model, the gate, and the assignment.
//!
//! Ported from `tracksolve` (`motion`, `gate`, `score`, `linker` in
//! `mode="lap"`), whose Python is the reference this was checked against.
//! Stage 1 of Jaqaman et al. (2008, *Nat. Methods* 5:695) only: no gap
//! closing, no merges or splits, and no multiple-hypothesis deferral.
//!
//! # Why this cost and not `d^2`
//!
//! A ladder of arms measured on tracksolve's simulator (10x10 um, 30%
//! immobile and 70% at D = 0.3 um^2/s, dt = 22 ms, se ~ 29 nm with 30%
//! spread, 95% detection, some clutter, 3 seeds), as switches per 100 links
//! against the step/nearest-neighbour ratio that controls how ambiguous
//! linking is:
//!
//! | step/NN | `d^2` | one-D Gaussian | D mixture | greedy | **this** |
//! |---------|-------|----------------|-----------|--------|----------|
//! | 0.22    | 1.81  | 1.55           | 1.69      | 1.60   | **1.43** |
//! | 0.50    | 11.99 | 10.38          | 10.24     | 11.40  | **9.35** |
//! | 0.80    | 29.90 | 23.56          | 22.99     | 23.73  | **22.04**|
//! | 1.00    | 41.30 | 32.95          | 32.06     | 32.43  | **30.85**|
//! | 1.20    | 50.23 | 40.88          | 40.20     | 39.70  | **39.20**|
//!
//! Reading the columns: replacing plain squared displacement with a log
//! likelihood ratio that knows each detection's own localization error is
//! worth about 20% of the switches once crowding matters; giving every track
//! its own posterior over D (this module) takes another ~1 per 100; and
//! solving the assignment exactly rather than greedily is worth 1-2 per 100
//! on its own (`lap`).
//!
//! Deferring decisions is the one thing left, and it is not here: tracksolve
//! implements it and measured it LOSING to this, even on well-separated
//! particles, because its set-packing solver falls back to a heuristic on the
//! components that matter. With a cost that depends only on the pair being
//! linked, a whole-movie solver would add nothing either -- the movie
//! decomposes into independent frame pairs -- so per-track memory, which is
//! what the filter bank below carries, is the only thing that beats a
//! one-frame-at-a-time exact assignment.
//!
//! # The model
//!
//! Position only: Brownian position is a martingale, so the one-step
//! prediction is the current estimate and a velocity state would impose
//! directed motion that free diffusion does not have. The measurement
//! covariance is spotsolve's CRLB, which is diagonal, and the process noise
//! `2*D*dt*I` is diagonal, so the 2-D filter is two independent scalar
//! filters -- no matrix, no inverse, no Cholesky anywhere in this module.
//!
//! Filtering POSITIONS rather than scoring displacements is what makes the
//! errors come out right: `d1 = x2 - x1` and `d2 = x3 - x2` share the error
//! at `x2`, so `Cov(d1, d2) = -se^2`, and a cost built on per-step
//! displacements (u-track's, trackpy's) treats them as independent.
//!
//! D is not fitted. There is no conjugate prior for it once localization
//! error is nonzero, so it lives on a grid that includes an exact zero, and
//! every track carries a posterior over that grid. The zero is not cosmetic:
//! an immobile particle given a nonzero floor gets a gate of radius
//! `sqrt(2*D*dt + 2*se^2)` instead of `sqrt(2*se^2)`, and over-wide gates on
//! a dense immobile population are how identity switches get manufactured.
//!
//! # Units
//!
//! Pixels and frames, with `dt = 1` frame throughout, so `d_grid` is in
//! px^2/frame and `lam_birth` in births per px^2 per frame. The scores are
//! log likelihood RATIOS, which are invariant under a change of units, so
//! this produces the same links as tracksolve's um and seconds.
//!
//! # Cost, and what the port reproduces
//!
//! On `data/hyp7gem_wt_crop.tif` (49 frames, 21,438 detections after the
//! detector, ~438 per frame) this and tracksolve agree on **100% of links**
//! and on every fitted parameter to the digits printed, with each side
//! fitting its own parameters from scratch:
//!
//! | stage | tracksolve (Python) | here | speedup |
//! |-------|---------------------|------|---------|
//! | parameter fit (3 linkings) | 13.03 s | 0.13 s | 100x |
//! | one linking | 2.17 s | 0.03 s | 72x |
//!
//! The gate is what keeps that near-linear in detections: without it every
//! track would be scored against every detection in the next frame.
//!
//! # What the clutter intensity would do, and why it is absent
//!
//! tracksolve's score is `log(p_cont) + logL - log(lam_fa)` for a link,
//! `log(lam_birth) - log(lam_fa)` for a birth and `log(1 - p_cont)` for a
//! termination. In an assignment only the GAIN of linking over not linking
//! matters, and `lam_fa` cancels out of it exactly:
//!
//! ```text
//!   gain = [log p_cont + logL - log lam_fa]
//!        - [log(1 - p_cont)] - [log lam_birth - log lam_fa]
//!        =  log p_cont + logL - log(1 - p_cont) - log lam_birth
//! ```
//!
//! Its only other use there is a "stop if total evidence fell" rule in the
//! parameter fit, which fired in 0 of 27 fits across three densities and
//! three detection/clutter settings -- and cannot fire honestly anyway, since
//! in LAP mode every detection lands in some track, so the estimator that
//! feeds it sees no unlinked detections at all. Both are left out. See
//! `trackparams`.

use crate::lap;

/// Probability that the gate excludes the true successor. Split evenly
/// between the per-component chi-square level and the posterior weight
/// dropped from the union, so the union bound makes the total at most this
/// with no assumption about the mixture (tracksolve's `gate`). 1e-3 costs a
/// gate radius of about 3.9 sigma on a concentrated posterior.
pub const GATE_ALPHA: f64 = 1e-3;

/// Points in the D grid, and the decades it spans below and above the
/// resolvability floor `se^2/dt`. 15 log-spaced points plus an exact zero
/// puts the floor exactly on a grid point.
pub const D_GRID_N: usize = 16;
pub const D_GRID_DECADES: f64 = 3.0;

/// Detections of one movie, sorted by frame, with a CSR index over frames.
///
/// Frames are made dense from the FIRST frame present: a table starting at
/// frame 100 gets no hundred empty leading slices, because `lam_birth` and
/// the continuation probability are per-frame rates and leading emptiness
/// would divide both by frames that were never imaged.
pub struct Detections {
    pub n_frames: usize,
    /// `offsets[f]..offsets[f+1]` are frame `f`'s rows.
    pub offsets: Vec<usize>,
    /// `(N, 2)` row-major `(y, x)`, px.
    pub pos: Vec<f64>,
    /// `(N, 2)` row-major squared standard errors, px^2, as reported.
    pub se2: Vec<f64>,
    /// Sorted row -> the caller's row, so answers go back in input order.
    pub order: Vec<usize>,
}

impl Detections {
    /// Validates and sorts. `frame` is 0-based; `pos` and `se` are `(N, 2)`
    /// row-major `(y, x)`.
    pub fn new(frame: &[i64], pos: &[f64], se: &[f64]) -> Result<Self, String> {
        let n = frame.len();
        if pos.len() != 2 * n || se.len() != 2 * n {
            return Err(format!(
                "frame has {n} rows but positions have {} and errors {}",
                pos.len() / 2,
                se.len() / 2
            ));
        }
        if n == 0 {
            return Ok(Detections {
                n_frames: 0,
                offsets: vec![0],
                pos: vec![],
                se2: vec![],
                order: vec![],
            });
        }
        if let Some(i) = (0..2 * n).find(|&i| !pos[i].is_finite()) {
            return Err(format!("row {} has a non-finite position", i / 2));
        }
        if let Some(i) = (0..2 * n).find(|&i| !(se[i].is_finite() && se[i] > 0.0)) {
            return Err(format!(
                "row {} has a localization error that is not finite and positive ({}); \
                 spotsolve reports NaN where the Fisher information was singular, so \
                 filter those rows out before linking -- a zero error would make the \
                 gate infinitely tight",
                i / 2,
                se[i]
            ));
        }
        if let Some(i) = (0..n).find(|&i| frame[i] < 0) {
            return Err(format!("row {i} has a negative frame index"));
        }
        let lo = *frame.iter().min().unwrap();
        let hi = *frame.iter().max().unwrap();
        let n_frames = (hi - lo + 1) as usize;

        let mut counts = vec![0usize; n_frames + 1];
        for &f in frame {
            counts[(f - lo) as usize + 1] += 1;
        }
        let mut offsets = counts;
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        // Stable counting sort: rows keep their input order within a frame.
        let mut fill = offsets.clone();
        let mut order = vec![0usize; n];
        for (i, &f) in frame.iter().enumerate() {
            let k = (f - lo) as usize;
            order[fill[k]] = i;
            fill[k] += 1;
        }
        let mut p = vec![0.0; 2 * n];
        let mut s = vec![0.0; 2 * n];
        for (r, &i) in order.iter().enumerate() {
            p[2 * r] = pos[2 * i];
            p[2 * r + 1] = pos[2 * i + 1];
            s[2 * r] = se[2 * i] * se[2 * i];
            s[2 * r + 1] = se[2 * i + 1] * se[2 * i + 1];
        }
        Ok(Detections {
            n_frames,
            offsets,
            pos: p,
            se2: s,
            order,
        })
    }

    pub fn n_dets(&self) -> usize {
        self.order.len()
    }

    pub fn frame_rows(&self, f: usize) -> std::ops::Range<usize> {
        self.offsets[f]..self.offsets[f + 1]
    }

    /// Area of the bounding box of every detection, px^2. Turns counts into
    /// the spatial intensity `lam_birth`. It under-estimates the field
    /// whenever detections do not reach the edges, which biases the
    /// intensity up.
    pub fn bbox_area(&self) -> f64 {
        if self.n_dets() == 0 {
            return f64::MIN_POSITIVE;
        }
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for r in 0..self.n_dets() {
            for a in 0..2 {
                lo[a] = lo[a].min(self.pos[2 * r + a]);
                hi[a] = hi[a].max(self.pos[2 * r + a]);
            }
        }
        (hi[0] - lo[0]).max(f64::MIN_POSITIVE) * (hi[1] - lo[1]).max(f64::MIN_POSITIVE)
    }
}

/// Everything the filter and the score need. `d_grid[0]` is exactly 0.
#[derive(Clone, Debug)]
pub struct Params {
    /// px^2/frame, ascending, with an exact zero first.
    pub d_grid: Vec<f64>,
    /// Log population weight per grid point, normalized.
    pub d_logprior: Vec<f64>,
    /// Per-frame P(detected AND still alive).
    pub p_cont: f64,
    /// New tracks per px^2 within one frame.
    pub lam_birth: f64,
    /// CRLB variance inflation, >= 1.
    pub se_inflate: f64,
}

impl Params {
    pub fn m(&self) -> usize {
        self.d_grid.len()
    }

    pub fn check(&self) -> Result<(), String> {
        if self.d_grid.len() < 2 || self.d_grid.len() != self.d_logprior.len() {
            return Err("d_grid and d_logprior must have the same length, at least 2".into());
        }
        if self.d_grid[0] != 0.0 || self.d_grid.windows(2).any(|w| w[1] <= w[0]) {
            return Err("d_grid must start at exactly 0 and ascend".into());
        }
        if self.d_grid.iter().any(|v| !v.is_finite())
            || self.d_logprior.iter().any(|v| !v.is_finite())
        {
            return Err("d_grid and d_logprior must be finite".into());
        }
        if !(self.p_cont > 0.0 && self.p_cont < 1.0) {
            return Err(format!("p_cont must be in (0, 1), got {}", self.p_cont));
        }
        if !(self.lam_birth.is_finite() && self.lam_birth > 0.0) {
            return Err(format!(
                "lam_birth must be positive, got {}",
                self.lam_birth
            ));
        }
        if !(self.se_inflate >= 1.0 && self.se_inflate.is_finite()) {
            return Err(format!(
                "se_inflate must be at least 1, got {}",
                self.se_inflate
            ));
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ filter

pub fn logsumexp(a: &[f64]) -> f64 {
    let mx = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !mx.is_finite() {
        return mx;
    }
    mx + a.iter().map(|&v| (v - mx).exp()).sum::<f64>().ln()
}

/// One track's belief: a mean and variance per D-grid point, per axis, and a
/// posterior over the grid. Rooted at its first detection in the
/// diffuse-prior limit -- one observation localizes the particle to exactly
/// the precision of that observation.
#[derive(Clone)]
pub struct Filter {
    /// `(M, 2)`.
    pub mu: Vec<f64>,
    /// `(M, 2)`.
    pub var: Vec<f64>,
    /// `(M,)`, normalized.
    pub logw: Vec<f64>,
}

impl Filter {
    pub fn root(z: [f64; 2], se2: [f64; 2], p: &Params) -> Self {
        let m = p.m();
        let mut mu = vec![0.0; 2 * m];
        let mut var = vec![0.0; 2 * m];
        for i in 0..m {
            mu[2 * i] = z[0];
            mu[2 * i + 1] = z[1];
            var[2 * i] = se2[0] * p.se_inflate;
            var[2 * i + 1] = se2[1] * p.se_inflate;
        }
        Filter {
            mu,
            var,
            logw: p.d_logprior.clone(),
        }
    }

    /// One frame of free diffusion: the mean does not move, the variance
    /// grows by `2*D*dt` with `dt = 1` frame.
    pub fn predict(&self, grid: &[f64], var_p: &mut [f64]) {
        for i in 0..grid.len() {
            let q = 2.0 * grid[i];
            var_p[2 * i] = self.var[2 * i] + q;
            var_p[2 * i + 1] = self.var[2 * i + 1] + q;
        }
    }

    /// Predictive log density of `z` under each grid point.
    pub fn loglik(&self, var_p: &[f64], z: [f64; 2], r: [f64; 2], out: &mut [f64]) {
        const LOG_2PI: f64 = 1.837_877_066_409_345_5;
        for i in 0..out.len() {
            let sy = var_p[2 * i] + r[0];
            let sx = var_p[2 * i + 1] + r[1];
            let dy = z[0] - self.mu[2 * i];
            let dx = z[1] - self.mu[2 * i + 1];
            out[i] =
                -0.5 * ((LOG_2PI + sy.ln() + dy * dy / sy) + (LOG_2PI + sx.ln() + dx * dx / sx));
        }
    }

    /// Folds `loglik` into the D posterior. Returns `logL`, the predictive
    /// log density with D integrated out. Sequential marginalization like
    /// this is exact, not an approximation: the product telescopes to
    /// `logsumexp(logprior + sum of logliks)`.
    pub fn marginalize(&mut self, ll: &[f64]) -> f64 {
        for (w, &l) in self.logw.iter_mut().zip(ll) {
            *w += l;
        }
        let tot = logsumexp(&self.logw);
        for w in self.logw.iter_mut() {
            *w -= tot;
        }
        tot
    }

    /// Conditions the position on `z`. Call after `marginalize`.
    pub fn update(&mut self, var_p: &[f64], z: [f64; 2], r: [f64; 2]) {
        for i in 0..self.logw.len() {
            for a in 0..2 {
                let vp = var_p[2 * i + a];
                let k = vp / (vp + r[a]);
                self.mu[2 * i + a] += k * (z[a] - self.mu[2 * i + a]);
                self.var[2 * i + a] = (1.0 - k) * vp;
            }
        }
    }
}

// -------------------------------------------------------------------- gate

/// Squared-Mahalanobis cutoff with 2 degrees of freedom, in closed form: the
/// chi-square survival function with 2 df is `exp(-x/2)`.
pub fn chi2_cutoff(alpha: f64) -> f64 {
    -2.0 * alpha.ln()
}

/// The fewest grid components carrying at least `1 - alpha_w` of the
/// posterior, heaviest first. This is what lets a gate TIGHTEN as a track
/// accumulates evidence: without it the union is effectively the largest-D
/// gate no matter how implausible that D has become.
pub fn kept(
    logw: &[f64],
    alpha_w: f64,
    w: &mut Vec<f64>,
    order: &mut Vec<usize>,
    mask: &mut [bool],
) {
    let m = logw.len();
    let mx = logw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    w.clear();
    w.extend(logw.iter().map(|&v| (v - mx).exp()));
    let tot: f64 = w.iter().sum();
    for v in w.iter_mut() {
        *v /= tot;
    }
    order.clear();
    order.extend(0..m);
    order.sort_by(|&a, &b| w[b].total_cmp(&w[a]).then(a.cmp(&b)));
    mask.iter_mut().for_each(|v| *v = false);
    let target = 1.0 - alpha_w;
    let mut cum = 0.0;
    for (rank, &i) in order.iter().enumerate() {
        mask[i] = true;
        cum += w[i];
        if cum >= target || rank + 1 == m {
            break;
        }
    }
}

/// Centre and radius of one circle containing the whole union gate.
///
/// The kept components do not share a mean -- each has its own Kalman gain,
/// so a component that believes in a large D tracks further toward the data
/// -- so the radius carries each kept mean's offset from the centre. A
/// candidate lost here is lost for good, so this must be a superset.
fn bound(
    f: &Filter,
    var_p: &[f64],
    mask: &[bool],
    w: &[f64],
    se2_max: f64,
    c: f64,
) -> ([f64; 2], f64) {
    let tot: f64 = (0..mask.len()).filter(|&i| mask[i]).map(|i| w[i]).sum();
    let mut centre = [0.0; 2];
    for i in 0..mask.len() {
        if mask[i] {
            centre[0] += w[i] / tot * f.mu[2 * i];
            centre[1] += w[i] / tot * f.mu[2 * i + 1];
        }
    }
    let mut radius: f64 = 0.0;
    for i in 0..mask.len() {
        if !mask[i] {
            continue;
        }
        let s = (var_p[2 * i] + se2_max).max(var_p[2 * i + 1] + se2_max);
        let dy = f.mu[2 * i] - centre[0];
        let dx = f.mu[2 * i + 1] - centre[1];
        radius = radius.max(dy.hypot(dx) + (c * s).sqrt());
    }
    (centre, radius)
}

/// The exact union gate: is `z` inside any kept component's chi-square gate?
fn accept(f: &Filter, var_p: &[f64], mask: &[bool], z: [f64; 2], r: [f64; 2], c: f64) -> bool {
    (0..mask.len()).any(|i| {
        if !mask[i] {
            return false;
        }
        let sy = var_p[2 * i] + r[0];
        let sx = var_p[2 * i + 1] + r[1];
        let dy = z[0] - f.mu[2 * i];
        let dx = z[1] - f.mu[2 * i + 1];
        dy * dy / sy + dx * dx / sx < c
    })
}

// --------------------------------------------------------------- cell index

/// A uniform grid over the whole movie's bounding box, used to answer the
/// gate's radius queries and the parameter estimator's nearest-neighbour
/// queries without an O(N^2) scan. One cell holds about one detection per
/// frame on average, so a query touches a small fixed neighbourhood.
pub struct CellIndex {
    pub y0: f64,
    pub x0: f64,
    pub cell: f64,
    pub ny: usize,
    pub nx: usize,
}

/// One frame's detections bucketed into the cells, CSR.
#[derive(Default)]
pub struct Buckets {
    pub starts: Vec<usize>,
    /// Row indices into `Detections`, not indices within the frame.
    pub items: Vec<usize>,
}

impl CellIndex {
    pub fn new(d: &Detections) -> Self {
        let n = d.n_dets();
        let (mut y0, mut x0) = (0.0, 0.0);
        let (mut y1, mut x1) = (1.0, 1.0);
        if n > 0 {
            y0 = f64::INFINITY;
            x0 = f64::INFINITY;
            y1 = f64::NEG_INFINITY;
            x1 = f64::NEG_INFINITY;
            for r in 0..n {
                y0 = y0.min(d.pos[2 * r]);
                y1 = y1.max(d.pos[2 * r]);
                x0 = x0.min(d.pos[2 * r + 1]);
                x1 = x1.max(d.pos[2 * r + 1]);
            }
        }
        let live = (0..d.n_frames)
            .filter(|&f| !d.frame_rows(f).is_empty())
            .count();
        let per_frame = if live > 0 {
            n as f64 / live as f64
        } else {
            1.0
        };
        let area = ((y1 - y0) * (x1 - x0)).max(f64::MIN_POSITIVE);
        // A line of detections has zero bounding-box area but nonzero span.
        // Bound cells along its long axis too, rather than allocating billions
        // of empty buckets from the 1e-9 cell floor. Square fields retain the
        // original spacing; this only coarsens highly elongated fields.
        let cell = (area / per_frame.max(1.0))
            .sqrt()
            .max((y1 - y0).max(x1 - x0) / per_frame.max(1.0))
            .max(1e-9);
        let ny = (((y1 - y0) / cell).floor() as usize) + 1;
        let nx = (((x1 - x0) / cell).floor() as usize) + 1;
        CellIndex {
            y0,
            x0,
            cell,
            ny,
            nx,
        }
    }

    fn cell_of(&self, v: f64, lo: f64, n: usize) -> usize {
        let k = ((v - lo) / self.cell).floor();
        if k < 0.0 { 0 } else { (k as usize).min(n - 1) }
    }

    /// Buckets one frame's rows, reusing `out`'s allocations.
    pub fn bucket(&self, d: &Detections, f: usize, out: &mut Buckets) {
        let rows = d.frame_rows(f);
        out.starts.clear();
        out.starts.resize(self.ny * self.nx + 1, 0);
        out.items.clear();
        out.items.resize(rows.len(), 0);
        for r in rows.clone() {
            let c = self.cell_of(d.pos[2 * r], self.y0, self.ny) * self.nx
                + self.cell_of(d.pos[2 * r + 1], self.x0, self.nx);
            out.starts[c + 1] += 1;
        }
        for i in 1..out.starts.len() {
            out.starts[i] += out.starts[i - 1];
        }
        let mut fill = out.starts.clone();
        for r in rows {
            let c = self.cell_of(d.pos[2 * r], self.y0, self.ny) * self.nx
                + self.cell_of(d.pos[2 * r + 1], self.x0, self.nx);
            out.items[fill[c]] = r;
            fill[c] += 1;
        }
    }

    /// Calls `f` with every row in the cells the circle's bounding square
    /// touches -- a superset of the circle, which is itself a superset of the
    /// union gate.
    pub fn near(&self, b: &Buckets, centre: [f64; 2], radius: f64, mut f: impl FnMut(usize)) {
        let iy0 = self.cell_of(centre[0] - radius, self.y0, self.ny);
        let iy1 = self.cell_of(centre[0] + radius, self.y0, self.ny);
        let ix0 = self.cell_of(centre[1] - radius, self.x0, self.nx);
        let ix1 = self.cell_of(centre[1] + radius, self.x0, self.nx);
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                let c = iy * self.nx + ix;
                for &r in &b.items[b.starts[c]..b.starts[c + 1]] {
                    f(r);
                }
            }
        }
    }

    /// Distance from `q` to the nearest row in `b`, or `None` if it is
    /// empty. Rings are scanned outward and the search stops as soon as no
    /// unscanned cell can hold anything closer.
    pub fn nearest(
        &self,
        d: &Detections,
        b: &Buckets,
        q: [f64; 2],
        skip: Option<usize>,
    ) -> Option<f64> {
        if b.items.is_empty() {
            return None;
        }
        let cy = self.cell_of(q[0], self.y0, self.ny) as isize;
        let cx = self.cell_of(q[1], self.x0, self.nx) as isize;
        let mut best = f64::INFINITY;
        let reach = self.ny.max(self.nx) as isize;
        for ring in 0..=reach {
            if best.is_finite() && best <= (ring - 1).max(0) as f64 * self.cell {
                break;
            }
            let mut any = false;
            for iy in (cy - ring).max(0)..=(cy + ring).min(self.ny as isize - 1) {
                for ix in (cx - ring).max(0)..=(cx + ring).min(self.nx as isize - 1) {
                    // Ring, not disc: skip what the inner rings covered.
                    if (iy - cy).abs() != ring && (ix - cx).abs() != ring {
                        continue;
                    }
                    any = true;
                    let c = iy as usize * self.nx + ix as usize;
                    for &r in &b.items[b.starts[c]..b.starts[c + 1]] {
                        if Some(r) == skip {
                            continue;
                        }
                        let dy = q[0] - d.pos[2 * r];
                        let dx = q[1] - d.pos[2 * r + 1];
                        best = best.min(dy.hypot(dx));
                    }
                }
            }
            if !any && ring > 0 && best.is_finite() {
                break;
            }
        }
        if best.is_finite() { Some(best) } else { None }
    }
}

// ------------------------------------------------------------------ linker

/// One linking: a track id per detection, in `Detections`' sorted order.
pub struct Linking {
    pub track: Vec<u32>,
    pub n_tracks: u32,
}

/// Per-detection diagnostics for the proposed incoming assignment, in sorted
/// order. A rejected proposal still has a margin; a birth without a proposal
/// has None. These describe the original frame optimum, before abstention.
pub struct LinkDiagnostics {
    pub margin: Vec<Option<f64>>,
    pub rejected: Vec<bool>,
}

impl Linking {
    /// Rows of each track, ascending by track id, each in frame order.
    pub fn tracks(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.n_tracks as usize];
        for (r, &t) in self.track.iter().enumerate() {
            out[t as usize].push(r);
        }
        out
    }
}

struct Live {
    id: u32,
    f: Filter,
    /// Log-flux level: mean and variance. Unused without a [`FluxModel`].
    fm: f64,
    fv: f64,
}

/// Brightness as a second link cue, opt-in ([`link_with_flux`]).
///
/// Each track carries its log-flux level as a random walk: per frame the
/// level's variance grows by `q`, and a detection reads it with its own
/// variance `lf_var` plus `tau2`, the detection-to-detection scatter photon
/// noise does not explain (blinking, defocus, blur). A link's gain gains
/// `log N(lf; level, v + q + lf_var + tau2) - log pop(lf)`: the same ratio
/// of "this track" against "a new track", whose flux is drawn from the
/// movie's population density `pop`. Built from the data by
/// [`crate::trackparams::flux_model`]; nothing is set by hand.
///
/// # What it buys, measured 2026-09-14
///
/// Its job is the case positions cannot settle: a bright particle among
/// dimmer, faster ones keeps its identity. Spiked into the real GEM frames,
/// 36 clusters of one bright spot (2500 e-) with three dim fast spots (300
/// e-, D = 2 px^2/frame) moving within 5 px of it, every true consecutive
/// pair classified against truth:
///
/// ```text
/// bright spot   cue          linked   wrong link   bright -> dim   purity
/// D 0.05        positions     .956      .025            21          .98
/// D 0.05        + flux        .954      .026            20          .98
/// D 0.43        positions     .865      .078            56          .92
/// D 0.43        + flux        .909      .044            30          .96
/// ```
///
/// An immobile spot's gate is already tight, so it gains nothing; a mobile
/// one is stolen half as often. The dim fast spots are unchanged: 78-80% of
/// their pairs miss a detection, which no linker cue recovers. The noise
/// estimates there were `tau2` 0.027-0.032 and `q` 0.001-0.014, and the
/// result does not hinge on them: fixed at (0.037, 0.086), (0.10, 0.05) or
/// (0.15, 0.03) it was the same. On spike-ins whose fluxes overlap the
/// population's it does nearly nothing (wrong links at 600 e-, D = 2: .23 ->
/// .19).
///
/// Why it is not the default: real GEM flux varies by a factor of 1.7-2.2
/// from one frame to the next (sd of the log step .53-.80, of which the
/// reported flux errors explain .28-.48). On `hyp7gem_wt_01` the estimates
/// are `tau2` 0.037 and no detectable drift (`q` at its floor), and it
/// fragments slightly: single-detection tracks 47.0% -> 48.9%, tracks of 8+
/// frames 309 -> 285. Without truth on that movie, whether those breaks are
/// wrong links cut or right ones lost is open.
#[derive(Clone, Debug)]
pub struct FluxModel {
    /// `(N,)` in the caller's row order: log flux, and its variance.
    pub lf: Vec<f64>,
    pub lf_var: Vec<f64>,
    pub tau2: f64,
    pub q: f64,
    /// Population log density of log flux on uniform bins from `lo`.
    pub lo: f64,
    pub step: f64,
    pub logdens: Vec<f64>,
}

impl FluxModel {
    fn logpop(&self, v: f64) -> f64 {
        let n = self.logdens.len();
        let i = ((v - self.lo) / self.step).floor();
        let i = if i < 0.0 { 0 } else { (i as usize).min(n - 1) };
        self.logdens[i]
    }
}

/// Links every frame to the next. Every detection ends up in some track; one
/// that never links is a track of length 1.
///
/// A missed detection ENDS a track -- there are no gap hypotheses here.
/// Fragmenting a trajectory is a safe failure and switching its identity is
/// not, and closing gaps belongs in a second assignment over segments
/// (Jaqaman's stage two), where it costs one more assignment instead of
/// multiplying the hypothesis space at every frame.
pub fn link(d: &Detections, p: &Params) -> Linking {
    link_with_flux(d, p, None)
}

/// [`link`], with brightness as a second cue when `flux` is given.
pub fn link_with_flux(d: &Detections, p: &Params, flux: Option<&FluxModel>) -> Linking {
    link_scored(d, p, flux, 0.0, false).0
}

/// Conservative linking. Keep only original optimal links whose exclusion
/// margin is >= `min_margin` (finite and nonnegative). Rejected proposals end
/// their source tracks and start new tracks at their destinations. Do not
/// re-solve after rejection: removing competitors must not turn an ambiguous
/// second choice into an apparently certain association. Filtering precedes
/// updates to position, diffusion, and brightness state.
pub fn link_scored(
    d: &Detections,
    p: &Params,
    flux: Option<&FluxModel>,
    min_margin: f64,
    diagnostics: bool,
) -> (Linking, Option<LinkDiagnostics>) {
    assert!(min_margin.is_finite() && min_margin >= 0.0);
    const LOG_2PI: f64 = 1.837_877_066_409_345_5;
    // Row (sorted) -> log flux and its variance.
    let (lfs, lfv): (Vec<f64>, Vec<f64>) = match flux {
        Some(fx) => (
            d.order.iter().map(|&i| fx.lf[i]).collect(),
            d.order.iter().map(|&i| fx.lf_var[i]).collect(),
        ),
        None => (Vec::new(), Vec::new()),
    };
    let m = p.m();
    let mut track = vec![0u32; d.n_dets()];
    let mut diag = diagnostics.then(|| LinkDiagnostics {
        margin: vec![None; d.n_dets()],
        rejected: vec![false; d.n_dets()],
    });
    let mut next_id: u32 = 0;
    if d.n_dets() == 0 {
        return (Linking { track, n_tracks: 0 }, diag);
    }

    let index = CellIndex::new(d);
    let mut buckets = Buckets::default();
    let mut live: Vec<Live> = Vec::new();
    let mut next_live: Vec<Live> = Vec::new();

    let c = chi2_cutoff(GATE_ALPHA / 2.0);
    let term = (1.0 - p.p_cont).ln();
    let birth = p.lam_birth.ln();
    let log_p_cont = p.p_cont.ln();

    let mut var_p = vec![0.0; 2 * m];
    let mut ll = vec![0.0; m];
    let mut mask = vec![false; m];
    let mut wbuf: Vec<f64> = Vec::with_capacity(m);
    let mut obuf: Vec<usize> = Vec::with_capacity(m);

    for f in 0..d.n_frames {
        let rows = d.frame_rows(f);
        let nd = rows.len();
        let mut matched: Vec<Option<usize>> = vec![None; live.len()];

        if !live.is_empty() && nd > 0 {
            index.bucket(d, f, &mut buckets);
            let se2_max = rows
                .clone()
                .flat_map(|r| [d.se2[2 * r], d.se2[2 * r + 1]])
                .fold(0.0f64, f64::max)
                * p.se_inflate;

            let mut row_ptr = vec![0usize; live.len() + 1];
            let mut cols: Vec<usize> = Vec::new();
            let mut gains: Vec<f64> = Vec::new();
            for (t, lv) in live.iter().enumerate() {
                lv.f.predict(&p.d_grid, &mut var_p);
                kept(
                    &lv.f.logw,
                    GATE_ALPHA / 2.0,
                    &mut wbuf,
                    &mut obuf,
                    &mut mask,
                );
                let (centre, radius) = bound(&lv.f, &var_p, &mask, &wbuf, se2_max, c);
                let mut edges: Vec<(usize, f64)> = Vec::new();
                index.near(&buckets, centre, radius, |r| {
                    let z = [d.pos[2 * r], d.pos[2 * r + 1]];
                    let rv = [d.se2[2 * r] * p.se_inflate, d.se2[2 * r + 1] * p.se_inflate];
                    if !accept(&lv.f, &var_p, &mask, z, rv, c) {
                        return;
                    }
                    lv.f.loglik(&var_p, z, rv, &mut ll);
                    let mut a = ll.clone();
                    for (v, &w) in a.iter_mut().zip(&lv.f.logw) {
                        *v += w;
                    }
                    let log_l = logsumexp(&a);
                    // Gain of linking over ending this track and starting a
                    // new one at r; see the module docs for the cancellation
                    // that removes the clutter intensity from it.
                    let mut gain = log_p_cont + log_l - term - birth;
                    if let Some(fx) = flux {
                        let s = lv.fv + fx.q + lfv[r] + fx.tau2;
                        let e = lfs[r] - lv.fm;
                        gain += -0.5 * (LOG_2PI + s.ln() + e * e / s) - fx.logpop(lfs[r]);
                    }
                    if gain > 0.0 {
                        edges.push((r - rows.start, gain));
                    }
                });
                edges.sort_by_key(|&(c, _)| c);
                for (c, g) in edges {
                    cols.push(c);
                    gains.push(g);
                }
                row_ptr[t + 1] = cols.len();
            }
            if min_margin > 0.0 || diagnostics {
                let (proposals, margins) =
                    lap::max_gain_matching_with_margins(nd, &row_ptr, &cols, &gains);
                matched = proposals;
                for (t, proposal) in matched.iter_mut().enumerate() {
                    if let Some(c) = *proposal {
                        let margin = margins[t].expect("matched row has a margin");
                        let rejected = margin < min_margin;
                        if let Some(diag) = diag.as_mut() {
                            diag.margin[rows.start + c] = Some(margin);
                            diag.rejected[rows.start + c] = rejected;
                        }
                        if rejected {
                            *proposal = None;
                        }
                    }
                }
            } else {
                matched = lap::max_gain_matching(nd, &row_ptr, &cols, &gains);
            }
        }

        next_live.clear();
        let mut taken = vec![false; nd];
        for (t, lv) in live.iter_mut().enumerate() {
            let Some(c) = matched[t] else { continue };
            let r = rows.start + c;
            let z = [d.pos[2 * r], d.pos[2 * r + 1]];
            let rv = [d.se2[2 * r] * p.se_inflate, d.se2[2 * r + 1] * p.se_inflate];
            lv.f.predict(&p.d_grid, &mut var_p);
            lv.f.loglik(&var_p, z, rv, &mut ll);
            lv.f.marginalize(&ll);
            lv.f.update(&var_p, z, rv);
            let (mut fm, mut fv) = (lv.fm, lv.fv);
            if let Some(fx) = flux {
                let vp = fv + fx.q;
                let k = vp / (vp + lfv[r] + fx.tau2);
                fm += k * (lfs[r] - fm);
                fv = (1.0 - k) * vp;
            }
            taken[c] = true;
            track[r] = lv.id;
            next_live.push(Live {
                id: lv.id,
                f: lv.f.clone(),
                fm,
                fv,
            });
        }
        for (c, &t) in taken.iter().enumerate() {
            if t {
                continue;
            }
            let r = rows.start + c;
            let z = [d.pos[2 * r], d.pos[2 * r + 1]];
            let se2 = [d.se2[2 * r], d.se2[2 * r + 1]];
            track[r] = next_id;
            let (fm, fv) = match flux {
                Some(fx) => (lfs[r], lfv[r] + fx.tau2),
                None => (0.0, 0.0),
            };
            next_live.push(Live {
                id: next_id,
                f: Filter::root(z, se2, p),
                fm,
                fv,
            });
            next_id += 1;
        }
        std::mem::swap(&mut live, &mut next_live);
    }

    (
        Linking {
            track,
            n_tracks: next_id,
        },
        diag,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collinear_detections_use_bounded_buckets() {
        let d = Detections::new(&[0, 0, 1], &[0., 0., 0., 100., 0., 0.], &[0.2; 6]).unwrap();
        let index = CellIndex::new(&d);
        assert!(index.ny * index.nx <= 4);
        let mut b = Buckets::default();
        index.bucket(&d, 0, &mut b);
        assert_eq!(index.nearest(&d, &b, [0., 99.], None), Some(1.));
    }

    pub(crate) struct Rng(pub u64);
    impl Rng {
        pub fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        pub fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    fn params(m_extra: f64) -> Params {
        let grid = vec![0.0, 0.05, 0.2, 0.8 + m_extra];
        let n = grid.len() as f64;
        Params {
            d_grid: grid,
            d_logprior: vec![(1.0 / n).ln(); 4],
            p_cont: 0.9,
            lam_birth: 1e-3,
            se_inflate: 1.0,
        }
    }

    /// Sequential marginalization must telescope: the running sum of the
    /// per-step `logL` equals `logsumexp(logprior + sum of logliks)`, the log
    /// marginal likelihood of the whole track with D integrated out. This is
    /// the identity that makes the score exact rather than approximate.
    #[test]
    fn marginalization_telescopes_to_the_track_marginal_likelihood() {
        let p = params(0.0);
        let m = p.m();
        let pos = [[10.0, 10.0], [10.3, 9.8], [10.1, 10.4], [10.6, 10.2]];
        let se2 = [0.04, 0.03];
        let mut f = Filter::root(pos[0], se2, &p);
        let mut var_p = vec![0.0; 2 * m];
        let mut ll = vec![0.0; m];
        let mut running = 0.0;
        let mut acc = vec![0.0; m];
        for z in &pos[1..] {
            f.predict(&p.d_grid, &mut var_p);
            f.loglik(&var_p, *z, se2, &mut ll);
            for (a, &l) in acc.iter_mut().zip(&ll) {
                *a += l;
            }
            running += f.marginalize(&ll);
            f.update(&var_p, *z, se2);
        }
        let direct: Vec<f64> = (0..m).map(|i| p.d_logprior[i] + acc[i]).collect();
        assert!((running - logsumexp(&direct)).abs() < 1e-12);
    }

    /// The cell index is an accelerator and nothing more: what it returns
    /// after the exact test must equal a brute-force scan of the frame.
    #[test]
    fn gate_candidates_equal_a_brute_force_scan() {
        let mut rng = Rng(12345);
        let n = 400;
        let mut frame = vec![0i64; n];
        let mut pos = vec![0.0; 2 * n];
        let mut se = vec![0.0; 2 * n];
        for i in 0..n {
            frame[i] = (i % 4) as i64;
            pos[2 * i] = rng.uniform() * 60.0;
            pos[2 * i + 1] = rng.uniform() * 60.0;
            se[2 * i] = 0.1 + 0.4 * rng.uniform();
            se[2 * i + 1] = 0.1 + 0.4 * rng.uniform();
        }
        let d = Detections::new(&frame, &pos, &se).unwrap();
        let p = params(3.0);
        let m = p.m();
        let index = CellIndex::new(&d);
        let mut b = Buckets::default();
        index.bucket(&d, 1, &mut b);
        let rows = d.frame_rows(1);
        let se2_max = rows
            .clone()
            .flat_map(|r| [d.se2[2 * r], d.se2[2 * r + 1]])
            .fold(0.0f64, f64::max);
        let c = chi2_cutoff(GATE_ALPHA / 2.0);
        let mut var_p = vec![0.0; 2 * m];
        let (mut w, mut o, mut mask) = (vec![], vec![], vec![false; m]);
        let mut n_hit = 0;
        for r0 in d.frame_rows(0) {
            let f = Filter::root([d.pos[2 * r0], d.pos[2 * r0 + 1]], [0.09, 0.09], &p);
            f.predict(&p.d_grid, &mut var_p);
            kept(&f.logw, GATE_ALPHA / 2.0, &mut w, &mut o, &mut mask);
            let (centre, radius) = bound(&f, &var_p, &mask, &w, se2_max, c);
            let mut fast: Vec<usize> = Vec::new();
            index.near(&b, centre, radius, |r| {
                let z = [d.pos[2 * r], d.pos[2 * r + 1]];
                let rv = [d.se2[2 * r], d.se2[2 * r + 1]];
                if accept(&f, &var_p, &mask, z, rv, c) {
                    fast.push(r);
                }
            });
            fast.sort();
            let slow: Vec<usize> = rows
                .clone()
                .filter(|&r| {
                    accept(
                        &f,
                        &var_p,
                        &mask,
                        [d.pos[2 * r], d.pos[2 * r + 1]],
                        [d.se2[2 * r], d.se2[2 * r + 1]],
                        c,
                    )
                })
                .collect();
            assert_eq!(fast, slow);
            n_hit += slow.len();
        }
        assert!(n_hit > 50, "gate never accepted anything: {n_hit}");
    }

    /// `nearest` must equal a brute-force nearest-neighbour scan, including
    /// for query points outside the frame's own extent.
    #[test]
    fn nearest_equals_brute_force() {
        let mut rng = Rng(999);
        let n = 300;
        let mut frame = vec![0i64; n];
        let mut pos = vec![0.0; 2 * n];
        let se = vec![0.2; 2 * n];
        for i in 0..n {
            frame[i] = (i % 3) as i64;
            pos[2 * i] = rng.uniform() * 40.0;
            pos[2 * i + 1] = rng.uniform() * 25.0;
        }
        let d = Detections::new(&frame, &pos, &se).unwrap();
        let index = CellIndex::new(&d);
        let mut b = Buckets::default();
        index.bucket(&d, 2, &mut b);
        for r0 in d.frame_rows(0) {
            let q = [d.pos[2 * r0], d.pos[2 * r0 + 1]];
            let got = index.nearest(&d, &b, q, None).unwrap();
            let want = d
                .frame_rows(2)
                .map(|r| (q[0] - d.pos[2 * r]).hypot(q[1] - d.pos[2 * r + 1]))
                .fold(f64::INFINITY, f64::min);
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    /// Brownian particles far enough apart that the association is never
    /// ambiguous, shuffled into a different row order every frame: the linker
    /// must recover identity from geometry, with no spurious births.
    #[test]
    fn recovers_well_separated_trajectories() {
        let mut rng = Rng(2024);
        let (n_p, n_f) = (12usize, 15usize);
        let mut truth: Vec<[f64; 2]> = (0..n_p)
            .map(|i| [10.0 + 20.0 * (i / 4) as f64, 10.0 + 20.0 * (i % 4) as f64])
            .collect();
        let (mut frame, mut pos, mut se, mut who) = (vec![], vec![], vec![], vec![]);
        for f in 0..n_f {
            if f > 0 {
                for t in truth.iter_mut() {
                    t[0] += 0.5 * rng.normal();
                    t[1] += 0.5 * rng.normal();
                }
            }
            let mut order: Vec<usize> = (0..n_p).collect();
            if f % 2 == 1 {
                order.reverse();
            }
            for &i in &order {
                frame.push(f as i64);
                pos.push(truth[i][0] + 0.05 * rng.normal());
                pos.push(truth[i][1] + 0.05 * rng.normal());
                se.push(0.05);
                se.push(0.05);
                who.push(i);
            }
        }
        let d = Detections::new(&frame, &pos, &se).unwrap();
        let p = Params {
            d_grid: vec![0.0, 0.02, 0.1, 0.25, 0.6],
            d_logprior: vec![(0.2f64).ln(); 5],
            p_cont: 0.95,
            lam_birth: 1e-4,
            se_inflate: 1.0,
        };
        let l = link(&d, &p);
        assert_eq!(l.n_tracks, n_p as u32, "expected one track per particle");
        let mut id_of = vec![u32::MAX; n_p];
        for (r, &t) in l.track.iter().enumerate() {
            let particle = who[d.order[r]];
            if id_of[particle] == u32::MAX {
                id_of[particle] = t;
            }
            assert_eq!(id_of[particle], t, "particle {particle} changed identity");
        }
    }
}
