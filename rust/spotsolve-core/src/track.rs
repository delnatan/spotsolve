//! Frame-to-frame linking of particles whose mobilities differ.
//!
//! # Model
//!
//! Detection `a` in frame `f` is linked to `b` in `f + 1` when that is more
//! likely than `a` going unseen and `b` being new. The gain of the link is
//!
//! ```text
//!     log p_a(d) - log lambda
//! ```
//!
//! `p_a` is the step density `a`'s track predicts: 2-D Gaussians of per-axis
//! variance `2 D_k + se_a^2 + se_b^2` over a log grid of diffusion
//! coefficients `D_k`, weighted by the track's posterior over `D_k` from its
//! own linked steps. Each frame pair's links are the matching of greatest
//! total gain, solved exactly by [`lap`]. With one shared `D` this is Crocker
//! & Grier's least squared displacement. With several, an immobile track does
//! not take a fast particle's step, and a fast particle passing a slow one is
//! not swapped with it: under one `D`, least squares swaps them whenever the
//! slow one lies inside the circle whose diameter is the fast step.
//!
//! `lambda = (1 - q)^2 rho / q` prices a track end: `rho` is the density of
//! other detections around a detection and `q` the chance that a detection
//! continues into the next frame, so `(1 - q) rho` is the density of new
//! detections and `(1 - q) / q` the odds that a track goes unseen.
//!
//! # Read from the movie, without links
//!
//! - `g`, the detector's gap: same-frame detections closer than about `g` are
//!   reported as one. It is the radius at which the same-frame pair density
//!   first reaches half its mean over `[R/2, R]`.
//! - `r0 = g / 2`: a next-frame detection within `r0` is the same particle,
//!   because anything else had to move at least `g - r0`. `r0` is never below
//!   three standard deviations of an immobile particle's step, so a table
//!   without a gap still counts immobile continuations.
//! - `q`: the share of detections with a detection within `r0` in the next
//!   frame, averaged with the previous frame so that reversing time changes
//!   nothing.
//! - `rho`: the same-frame pair density within `R`.
//!
//! # A new track
//!
//! starts from the prior over `D_k` that its spot's occupancy gives: whether
//! frames `f - WINDOW..f + WINDOW` hold a detection within `r0` (`f + 1`, the
//! link being decided, excluded). An immobile particle leaves a column of
//! detections at one spot, across its own missed frames; a mobile one does
//! not. The occupancy only informs `D`; it never makes a link.
//!
//! # Both directions of time
//!
//! A track's history makes each pass causal: after a wrong link the track
//! carries the other particle's history. The linker runs forward and on the
//! reversed movie and keeps the links both make; a link they disagree on
//! becomes a break, not a switch.
//!
//! `R = max_step` is the largest step considered. It is the search radius,
//! and three times the rms 2-D step at the top of the `D` grid. A frame
//! without a particle's detection ends its track: there is no gap closing.
//! Units are the caller's (pixels).

use crate::lap;

/// Diffusion coefficients on the grid, log-spaced from an eighth of the
/// localization variance (immobile, for every practical purpose) to the `D`
/// whose rms 2-D step is `R / 3`.
const N_D: usize = 8;

/// Share of a track's posterior over `D` returned to the uniform prior at
/// each step, so a track that took another particle's step does not keep
/// that particle's history.
const FORGET: f64 = 0.05;

/// Frames on each side of a new track's first detection read for its
/// occupancy prior.
const WINDOW: i64 = 10;

/// Detection probabilities averaged over, uniformly on `(0, 1)`, in the
/// occupancy prior.
const N_P: usize = 16;

const NONE: usize = usize::MAX;

/// Track id per detection, in input order, numbered by first appearance.
pub struct Linking {
    pub track: Vec<u32>,
    pub n_tracks: u32,
}

/// One frame's detections bucketed into square cells of side `side`, sorted
/// by `(cell row, cell column)`. Every detection within `side` of a point
/// lies in the 3 x 3 cells around it, three contiguous runs in this order.
struct Cells {
    side: f64,
    keys: Vec<(i64, i64, usize)>,
}

impl Cells {
    /// `keys` hold each detection's index within `rows`.
    fn new(rows: &[usize], pos: &[f64], side: f64) -> Self {
        let mut keys: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(k, &i)| ((pos[2 * i] / side).floor() as i64, (pos[2 * i + 1] / side).floor() as i64, k))
            .collect();
        keys.sort_unstable();
        Cells { side, keys }
    }

    /// Visits every detection within `side` of `q` (and some farther ones).
    fn near(&self, q: [f64; 2], mut f: impl FnMut(usize)) {
        let (cy, cx) = ((q[0] / self.side).floor() as i64, (q[1] / self.side).floor() as i64);
        for iy in [cy.saturating_sub(1), cy, cy.saturating_add(1)] {
            let lo = self.keys.partition_point(|k| (k.0, k.1) < (iy, cx.saturating_sub(1)));
            let hi = self.keys.partition_point(|k| (k.0, k.1) <= (iy, cx.saturating_add(1)));
            for k in &self.keys[lo..hi] {
                f(k.2);
            }
        }
    }

    /// Whether any detection lies within `r <= side` of `q`.
    fn any_within(&self, q: [f64; 2], r: f64, rows: &[usize], pos: &[f64]) -> bool {
        let mut hit = false;
        self.near(q, |k| {
            let b = rows[k];
            hit |= (pos[2 * b] - q[0]).powi(2) + (pos[2 * b + 1] - q[1]).powi(2) < r * r;
        });
        hit
    }
}

struct Frame {
    t: i64,
    rows: Vec<usize>,
    cells: Cells,
}

/// What the linker reads from the movie before linking.
struct Stats {
    r0: f64,
    rho: f64,
    log_lambda: f64,
}

fn d2(pos: &[f64], a: usize, b: usize) -> f64 {
    (pos[2 * a] - pos[2 * b]).powi(2) + (pos[2 * a + 1] - pos[2 * b + 1]).powi(2)
}

fn frame_at(frames: &[Frame], t: i64) -> Option<&Frame> {
    frames.binary_search_by_key(&t, |f| f.t).ok().map(|i| &frames[i])
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn log_sum_exp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// log density of a 2-D Gaussian step with squared length `d2` and per-axis
/// variance `v`.
fn log_step(d2: f64, v: f64) -> f64 {
    -(2.0 * std::f64::consts::PI * v).ln() - d2 / (2.0 * v)
}

/// The detector gap, from same-frame pair distances binned at `bin`. -> `g`.
fn gap(frames: &[Frame], pos: &[f64], r: f64, bin: f64, n: usize) -> f64 {
    let nb = (r / bin).floor() as usize;
    if nb < 2 {
        return r;
    }
    let mut count = vec![0.0f64; nb];
    for fr in frames {
        for (k, &a) in fr.rows.iter().enumerate() {
            fr.cells.near([pos[2 * a], pos[2 * a + 1]], |k2| {
                if k2 > k {
                    let d = d2(pos, a, fr.rows[k2]).sqrt();
                    let i = (d / bin) as usize;
                    if i < nb {
                        count[i] += 2.0;
                    }
                }
            });
        }
    }
    let area = |i: usize| std::f64::consts::PI * bin * bin * ((i + 1).pow(2) - i.pow(2)) as f64;
    let dens: Vec<f64> = (0..nb).map(|i| count[i] / (n as f64 * area(i))).collect();
    let mid = |i: usize| (i as f64 + 0.5) * bin;
    let far: Vec<usize> = (0..nb).filter(|&i| mid(i) >= r / 2.0).collect();
    let plateau = far.iter().map(|&i| count[i]).sum::<f64>() / (n as f64 * far.iter().map(|&i| area(i)).sum::<f64>());
    if plateau.is_nan() || plateau <= 0.0 {
        return r;
    }
    let smooth = |i: usize| {
        let at = |j: isize| if j < 0 || j as usize >= nb { 0.0 } else { dens[j as usize] };
        (at(i as isize - 1) + at(i as isize) + at(i as isize + 1)) / 3.0
    };
    let half = plateau / 2.0;
    match (0..nb).find(|&i| smooth(i) >= half) {
        None => r,
        Some(0) => mid(0),
        Some(i) => {
            let (s0, s1) = (smooth(i - 1), smooth(i));
            mid(i - 1) + (half - s0) / (s1 - s0) * bin
        }
    }
}

fn movie_stats(frames: &[Frame], pos: &[f64], r: f64, se2_med: f64, n: usize) -> Stats {
    // Positions are not known finer than their error: the histogram bin.
    let bin = se2_med.sqrt();
    let r0 = (gap(frames, pos, r, bin, n) / 2.0).max(3.0 * (2.0 * se2_med).sqrt());
    let mut pairs = 0.0;
    for fr in frames {
        for (k, &a) in fr.rows.iter().enumerate() {
            fr.cells.near([pos[2 * a], pos[2 * a + 1]], |k2| {
                if k2 > k && d2(pos, a, fr.rows[k2]) < r * r {
                    pairs += 2.0;
                }
            });
        }
    }
    // One pseudo-pair keeps a movie with no neighbours finite.
    let rho = (pairs + 1.0) / (n as f64 * std::f64::consts::PI * r * r);
    // Continuation both ways in time, with the rule of succession.
    let (t_lo, t_hi) = (frames[0].t, frames[frames.len() - 1].t);
    let (mut cont, mut tried) = (0.0f64, 0.0f64);
    for fr in frames {
        for step in [1, -1] {
            if (step == 1 && fr.t == t_hi) || (step == -1 && fr.t == t_lo) {
                continue; // the movie ends there
            }
            let next = frame_at(frames, fr.t + step);
            for &a in &fr.rows {
                tried += 1.0;
                if let Some(nf) = next {
                    if nf.cells.any_within([pos[2 * a], pos[2 * a + 1]], r0, &nf.rows, pos) {
                        cont += 1.0;
                    }
                }
            }
        }
    }
    let q = (cont + 1.0) / (tried + 2.0);
    Stats { r0, rho, log_lambda: 2.0 * (1.0 - q).ln() + rho.ln() - q.ln() }
}

/// Which frames around each detection hold a detection within `r0`: bit
/// `tau + WINDOW` (tau in -WINDOW..=WINDOW) of `seen` marks a frame inside
/// the movie, of `occupied` one holding such a detection.
fn occupancy(frames: &[Frame], pos: &[f64], r0: f64, n: usize) -> (Vec<u32>, Vec<u32>) {
    let (t_lo, t_hi) = (frames[0].t, frames[frames.len() - 1].t);
    let (mut seen, mut occupied) = (vec![0u32; n], vec![0u32; n]);
    for fr in frames {
        for tau in -WINDOW..=WINDOW {
            let t = fr.t + tau;
            if tau == 0 || t < t_lo || t > t_hi {
                continue;
            }
            let bit = 1u32 << (tau + WINDOW);
            let other = frame_at(frames, t);
            for &a in &fr.rows {
                seen[a] |= bit;
                if let Some(o) = other {
                    if o.cells.any_within([pos[2 * a], pos[2 * a + 1]], r0, &o.rows, pos) {
                        occupied[a] |= bit;
                    }
                }
            }
        }
    }
    (seen, occupied)
}

/// The grid and everything a pass needs.
struct Model<'a> {
    pos: &'a [f64],
    se2: Vec<f64>,
    se2_med: f64,
    dk: [f64; N_D],
    r: f64,
    stats: Stats,
    seen: Vec<u32>,
    occupied: Vec<u32>,
}

impl Model<'_> {
    /// log of the step density track state `w` predicts for `a -> b`.
    fn log_pred(&self, w: &[f64; N_D], a: usize, b: usize, dd: f64) -> f64 {
        let s2 = self.se2[a] + self.se2[b];
        let t: [f64; N_D] = std::array::from_fn(|k| w[k] + log_step(dd, 2.0 * self.dk[k] + s2));
        log_sum_exp(&t)
    }

    /// The posterior after the step `a -> b`, partly returned to the prior.
    fn update(&self, w: &[f64; N_D], a: usize, b: usize, dd: f64) -> [f64; N_D] {
        let s2 = self.se2[a] + self.se2[b];
        let mut t: [f64; N_D] = std::array::from_fn(|k| w[k] + log_step(dd, 2.0 * self.dk[k] + s2));
        let z = log_sum_exp(&t);
        for x in &mut t {
            *x = ((1.0 - FORGET) * (*x - z).exp() + FORGET / N_D as f64).ln();
        }
        t
    }

    /// Log prior over the grid for a track starting at `a`, for a pass whose
    /// next frame is `a`'s frame + `dir`.
    fn prior(&self, a: usize, dir: i64) -> [f64; N_D] {
        let (r0, rho) = (self.stats.r0, self.stats.rho);
        let c = 1.0 - (-rho * std::f64::consts::PI * r0 * r0).exp();
        let v0 = self.se2[a] + self.se2_med;
        let mut lp = [0.0; N_D];
        for (k, out) in lp.iter_mut().enumerate() {
            let mut per_p = [0.0f64; N_P];
            for (j, acc) in per_p.iter_mut().enumerate() {
                let p = (j as f64 + 0.5) / N_P as f64;
                for tau in -WINDOW..=WINDOW {
                    let bit = 1u32 << (tau + WINDOW);
                    if tau == 0 || tau == dir || self.seen[a] & bit == 0 {
                        continue;
                    }
                    let v = 2.0 * self.dk[k] * tau.unsigned_abs() as f64 + v0;
                    let still = 1.0 - (-r0 * r0 / (2.0 * v)).exp();
                    let pi = 1.0 - (1.0 - p * still) * (1.0 - c);
                    *acc += if self.occupied[a] & bit != 0 { pi.ln() } else { (1.0 - pi).ln() };
                }
            }
            *out = log_sum_exp(&per_p);
        }
        let z = log_sum_exp(&lp);
        lp.map(|x| x - z)
    }

    /// One causal pass over `frames` in the order given; `dir` is +1 forward
    /// in time, -1 backward. -> each detection's partner in the pass's next
    /// frame, or `NONE`.
    fn pass(&self, frames: &[Frame], order: &[usize], dir: i64) -> Vec<usize> {
        let n = self.se2.len();
        let mut next = vec![NONE; n];
        let mut w = vec![[0.0f64; N_D]; n];
        let mut matched = vec![false; n];
        let r2 = self.r * self.r;
        let mut prev: Option<&Frame> = None;
        for &fi in order {
            let cur = &frames[fi];
            if let Some(pf) = prev.filter(|pf| pf.t.checked_add(dir) == Some(cur.t)) {
                let (mut row_ptr, mut cols, mut gain) = (vec![0usize], Vec::new(), Vec::new());
                let mut edges: Vec<(usize, f64)> = Vec::new();
                for &a in &pf.rows {
                    edges.clear();
                    cur.cells.near([self.pos[2 * a], self.pos[2 * a + 1]], |k| {
                        let b = cur.rows[k];
                        let dd = d2(self.pos, a, b);
                        if dd < r2 {
                            let g = self.log_pred(&w[a], a, b, dd) - self.stats.log_lambda;
                            if g > 0.0 {
                                edges.push((k, g));
                            }
                        }
                    });
                    edges.sort_unstable_by_key(|e| e.0);
                    for &(c, g) in &edges {
                        cols.push(c);
                        gain.push(g);
                    }
                    row_ptr.push(cols.len());
                }
                for (&a, m) in pf.rows.iter().zip(lap::max_gain_matching(cur.rows.len(), &row_ptr, &cols, &gain)) {
                    if let Some(k) = m {
                        let b = cur.rows[k];
                        next[a] = b;
                        w[b] = self.update(&w[a], a, b, d2(self.pos, a, b));
                        matched[b] = true;
                    }
                }
            }
            for &b in &cur.rows {
                if !matched[b] {
                    w[b] = self.prior(b, dir);
                }
            }
            prev = Some(cur);
        }
        next
    }
}

/// Links `(y, x)` positions (`pos` is `(N, 2)` row-major) by frame index,
/// with per-detection localization errors `se` (`(N, 2)`, `(se_y, se_x)`).
/// Only consecutive frames are linked; `max_step` is the largest step
/// considered.
pub fn link(frame: &[i64], pos: &[f64], se: &[f64], max_step: f64) -> Result<Linking, String> {
    let n = frame.len();
    if pos.len() != 2 * n {
        return Err(format!("frame has {n} rows but positions have {}", pos.len() / 2));
    }
    if se.len() != 2 * n {
        return Err(format!("frame has {n} rows but errors have {}", se.len() / 2));
    }
    if !(max_step.is_finite() && max_step > 0.0) {
        return Err(format!("max_step must be positive and finite, got {max_step}"));
    }
    if let Some(i) = (0..2 * n).find(|&i| !pos[i].is_finite()) {
        return Err(format!("row {} has a non-finite position", i / 2));
    }
    if let Some(i) = (0..2 * n).find(|&i| !(se[i].is_finite() && se[i] > 0.0)) {
        return Err(format!("row {} has a localization error that is not positive and finite", i / 2));
    }
    if n == 0 {
        return Ok(Linking { track: Vec::new(), n_tracks: 0 });
    }
    let r = max_step;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| frame[i]);
    let mut frames: Vec<Frame> = Vec::new();
    let mut start = 0;
    while start < n {
        let t = frame[order[start]];
        let end = start + order[start..].partition_point(|&i| frame[i] == t);
        let rows = order[start..end].to_vec();
        let cells = Cells::new(&rows, pos, r);
        frames.push(Frame { t, rows, cells });
        start = end;
    }

    let se2: Vec<f64> = (0..n).map(|i| 0.5 * (se[2 * i].powi(2) + se[2 * i + 1].powi(2))).collect();
    let se2_med = median(se2.clone());
    let d_lo = se2_med / 8.0;
    let d_hi = (r * r / 36.0 - se2_med).max(2.0 * d_lo);
    let dk: [f64; N_D] = std::array::from_fn(|k| d_lo * (d_hi / d_lo).powf(k as f64 / (N_D - 1) as f64));
    let stats = movie_stats(&frames, pos, r, se2_med, n);
    let (seen, occupied) = occupancy(&frames, pos, stats.r0, n);
    let model = Model { pos, se2, se2_med, dk, r, stats, seen, occupied };

    let forward: Vec<usize> = (0..frames.len()).collect();
    let backward: Vec<usize> = forward.iter().rev().copied().collect();
    let fwd = model.pass(&frames, &forward, 1);
    let bwd = model.pass(&frames, &backward, -1);

    let mut track = vec![u32::MAX; n];
    let mut n_tracks = 0u32;
    for fr in &frames {
        for &a in &fr.rows {
            if track[a] == u32::MAX {
                track[a] = n_tracks;
                n_tracks += 1;
            }
            let b = fwd[a];
            if b != NONE && bwd[b] == a {
                track[b] = track[a];
            }
        }
    }
    Ok(Linking { track, n_tracks })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (u, v) = (self.uniform().max(1e-300), self.uniform());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    /// Particles on a `field`-px square, each with its own `D` (px^2/frame),
    /// seen with probability `p`, localization error `se`; a detection closer
    /// than `gap` to one already kept in its frame is dropped, as a detector
    /// would. -> (frame, pos, se, particle id).
    fn movie(rng: &mut Rng, d: &[f64], n_frames: i64, field: f64, p: f64, se: f64, gap: f64)
             -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<usize>) {
        let mut x: Vec<[f64; 2]> = d.iter().map(|_| [rng.uniform() * field, rng.uniform() * field]).collect();
        let (mut fr, mut pos, mut who) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..n_frames {
            if t > 0 {
                for (xi, &di) in x.iter_mut().zip(d) {
                    let s = (2.0 * di).sqrt();
                    xi[0] = (xi[0] + s * rng.normal()).rem_euclid(field);
                    xi[1] = (xi[1] + s * rng.normal()).rem_euclid(field);
                }
            }
            let mut kept: Vec<[f64; 2]> = Vec::new();
            for (i, xi) in x.iter().enumerate() {
                if rng.uniform() >= p {
                    continue;
                }
                let o = [xi[0] + se * rng.normal(), xi[1] + se * rng.normal()];
                if kept.iter().all(|k| (k[0] - o[0]).hypot(k[1] - o[1]) >= gap) {
                    kept.push(o);
                    fr.push(t);
                    pos.extend_from_slice(&o);
                    who.push(i);
                }
            }
        }
        let n = fr.len();
        (fr, pos, vec![se; 2 * n], who)
    }

    /// Linked pairs, each as (lower row, higher row).
    fn links(frame: &[i64], track: &[u32]) -> Vec<(usize, usize)> {
        let mut o: Vec<usize> = (0..frame.len()).collect();
        o.sort_by_key(|&i| (track[i], frame[i]));
        let mut out: Vec<_> = o
            .windows(2)
            .filter(|w| track[w[0]] == track[w[1]])
            .map(|w| (w[0].min(w[1]), w[0].max(w[1])))
            .collect();
        out.sort_unstable();
        out
    }

    fn model_for(frame: &[i64], pos: &[f64], se: f64, r: f64) -> (Vec<Frame>, Stats) {
        let n = frame.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| frame[i]);
        let mut frames = Vec::new();
        let mut start = 0;
        while start < n {
            let t = frame[order[start]];
            let end = start + order[start..].partition_point(|&i| frame[i] == t);
            let rows = order[start..end].to_vec();
            let cells = Cells::new(&rows, pos, r);
            frames.push(Frame { t, rows, cells });
            start = end;
        }
        let stats = movie_stats(&frames, pos, r, se * se, n);
        (frames, stats)
    }

    #[test]
    fn the_movie_gives_its_gap_continuation_and_density() {
        // Immobile particles seen 80% of the time behind a 3 px gap.
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let d = vec![0.0; 300];
        let (fr, pos, _, who) = movie(&mut rng, &d, 30, 200.0, 0.8, 0.1, 3.0);
        let (_, s) = model_for(&fr, &pos, 0.1, 15.0);
        assert!((s.r0 - 1.5).abs() < 0.15, "r0 {}", s.r0);
        // The density of other detections around a detection: the field's,
        // less the part of each R-disc outside the field and inside the gap.
        let (r, l, g) = (15.0, 200.0, 3.0);
        let inside = 1.0 - 8.0 * r / (3.0 * std::f64::consts::PI * l) + r * r / (2.0 * std::f64::consts::PI * l * l);
        let rho_true = fr.len() as f64 / 30.0 / (l * l) * inside * (1.0 - g * g / (r * r));
        assert!((s.rho / rho_true - 1.0).abs() < 0.03, "rho {} vs {rho_true}", s.rho);
        // The true continuation, both ways in time, against q from lambda.
        let seen: std::collections::HashSet<(i64, usize)> = fr.iter().copied().zip(who.iter().copied()).collect();
        let (mut cont, mut tried) = (0.0, 0.0);
        for (&t, &w) in fr.iter().zip(&who) {
            for step in [1, -1] {
                if (0..30).contains(&(t + step)) {
                    tried += 1.0;
                    cont += seen.contains(&(t + step, w)) as u8 as f64;
                }
            }
        }
        let lam = s.log_lambda.exp() / s.rho;
        let q = ((2.0 + lam) - ((2.0 + lam).powi(2) - 4.0).sqrt()) / 2.0;
        assert!((q - cont / tried).abs() < 0.02, "q {q} vs {}", cont / tried);
    }

    #[test]
    fn a_table_without_a_gap_still_counts_immobile_continuations() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let d = vec![0.0; 300];
        let (fr, pos, _, _) = movie(&mut rng, &d, 20, 200.0, 0.9, 0.2, 0.0);
        let (_, s) = model_for(&fr, &pos, 0.2, 10.0);
        assert!(s.r0 >= 3.0 * (2.0f64 * 0.04).sqrt() - 1e-12);
        let lam = s.log_lambda.exp() / s.rho;
        let q = ((2.0 + lam) - ((2.0 + lam).powi(2) - 4.0).sqrt()) / 2.0;
        assert!(q > 0.85, "q {q}");
    }

    #[test]
    fn a_track_posterior_settles_on_its_coefficient() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        let dk: [f64; N_D] = std::array::from_fn(|k| 0.01 * 2f64.powi(k as i32 * 2));
        let model = Model {
            pos: &[], se2: vec![0.01; 2], se2_med: 0.01, dk, r: 10.0,
            stats: Stats { r0: 1.0, rho: 0.01, log_lambda: -10.0 },
            seen: vec![0; 2], occupied: vec![0; 2],
        };
        for truth in [0, 3, 6] {
            let mut w = [-(N_D as f64).ln(); N_D];
            for _ in 0..40 {
                let s = (2.0 * dk[truth] + 0.02).sqrt();
                let dd = (s * rng.normal()).powi(2) + (s * rng.normal()).powi(2);
                w = model.update(&w, 0, 1, dd);
            }
            let best = (0..N_D).max_by(|&i, &j| w[i].total_cmp(&w[j])).unwrap();
            assert_eq!(best, truth);
        }
    }

    #[test]
    fn row_order_and_the_direction_of_time_do_not_change_the_links() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let d: Vec<f64> = (0..120).map(|i| [0.0, 0.6, 4.0][i % 3]).collect();
        let (fr, pos, se, _) = movie(&mut rng, &d, 25, 120.0, 0.9, 0.25, 2.5);
        let base = links(&fr, &link(&fr, &pos, &se, 8.0).unwrap().track);
        assert!(!base.is_empty());

        let rev: Vec<i64> = fr.iter().map(|&t| -t).collect();
        assert_eq!(links(&rev, &link(&rev, &pos, &se, 8.0).unwrap().track), base);

        let n = fr.len();
        let perm: Vec<usize> = (0..n).map(|i| (i * 7919) % n).collect();
        assert!(n % 7919 != 0);
        let fr2: Vec<i64> = perm.iter().map(|&i| fr[i]).collect();
        let pos2: Vec<f64> = perm.iter().flat_map(|&i| [pos[2 * i], pos[2 * i + 1]]).collect();
        let se2: Vec<f64> = perm.iter().flat_map(|&i| [se[2 * i], se[2 * i + 1]]).collect();
        let mut back: Vec<(usize, usize)> = links(&fr2, &link(&fr2, &pos2, &se2, 8.0).unwrap().track)
            .into_iter()
            .map(|(a, b)| (perm[a].min(perm[b]), perm[a].max(perm[b])))
            .collect();
        back.sort_unstable();
        assert_eq!(back, base);
    }

    #[test]
    fn a_fast_particle_passing_an_immobile_one_keeps_both_identities() {
        // The immobile particle sits inside the circle whose diameter is the
        // fast step at frames 10 -> 11, where least squares swaps them.
        let mut rng = Rng(0x0bad_5eed_0bad_5eed);
        let (mut fr, mut pos, mut who) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..20i64 {
            fr.push(t);
            pos.extend_from_slice(&[50.0 + 0.05 * rng.normal(), 50.0 + 0.05 * rng.normal()]);
            who.push(0);
            fr.push(t);
            pos.extend_from_slice(&[52.0 + 0.05 * rng.normal(), 50.0 + 6.0 * (t - 10) as f64 - 3.0]);
            who.push(1);
        }
        let se = vec![0.05; pos.len()];
        let l = link(&fr, &pos, &se, 18.0).unwrap();
        for (a, b) in links(&fr, &l.track) {
            assert_eq!(who[a], who[b], "detections {a} and {b} linked across particles");
        }
        let still: Vec<u32> = (0..fr.len()).filter(|&i| who[i] == 0).map(|i| l.track[i]).collect();
        assert!(still.iter().all(|&t| t == still[0]), "the immobile particle's track was broken");
    }

    #[test]
    fn no_step_is_longer_than_max_step_and_a_missed_frame_ends_a_track() {
        let se = [0.1; 6];
        let l = link(&[0, 1, 2], &[0.0, 0.0, 0.0, 1.9, 0.0, 4.0], &se, 2.0).unwrap();
        assert_ne!(l.track[1], l.track[2]);
        let l = link(&[0, 2], &[0.0, 0.0, 0.0, 0.1], &se[..4], 1.0).unwrap();
        assert_eq!(l.n_tracks, 2);
    }

    #[test]
    fn bad_inputs_are_refused() {
        let se = [0.1, 0.1];
        assert!(link(&[0], &[0.0], &se, 1.0).is_err());
        assert!(link(&[0], &[0.0, f64::NAN], &se, 1.0).is_err());
        assert!(link(&[0], &[0.0, 0.0], &se, 0.0).is_err());
        assert!(link(&[0], &[0.0, 0.0], &[0.1], 1.0).is_err());
        assert!(link(&[0], &[0.0, 0.0], &[0.1, 0.0], 1.0).is_err());
        assert!(link(&[0], &[0.0, 0.0], &[0.1, f64::INFINITY], 1.0).is_err());
        assert!(link(&[], &[], &[], 1.0).unwrap().n_tracks == 0);
    }
}
