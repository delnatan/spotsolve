//! Frame-to-frame linking of particles whose mobilities differ.
//!
//! # Model
//!
//! Detection `a` in frame `f` is linked to `b` in `f + 1` when that is more
//! likely, by the odds of a switch against a break, than `a` going unseen and
//! `b` being new. The gain of the link is
//!
//! ```text
//!     log p_a(d) - log lambda - SWITCH_COST
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
//! Given each detection's strength (the detector's `detect_z`), `q` and
//! `rho` are read per strength class instead, the classes equal shares of
//! the movie's detections, and a link `a -> b` is priced by
//! `lambda_ab = (1 - q_a) / q_a * (1 - q_b) rho_b`: the odds that `a`'s own
//! class goes unseen, times the density of new detections of `b`'s class.
//! A bright in-focus particle is seen again almost surely; a dim one near
//! the detection bar often is not. One `q` for both prices a bright
//! particle's longer steps as if it were flickering, and cuts them.
//! Without
//! `SWITCH_COST` each link would be the most probable call; a wrong link
//! switches two identities, which costs more than the break a missed link
//! makes.
//!
//! # Read from the movie, without links
//!
//! - `g`, the detector's gap: same-frame detections closer than about `g` are
//!   reported as one. It is the radius at which the same-frame pair density
//!   first reaches half its mean over `[R/2, R]`.
//! - `r0 = g / 2`: a detection within `r0` in another frame is at the same
//!   spot, because anything else had to be at least `g - r0` away. `r0` is
//!   never below three standard deviations of an immobile particle's step, so
//!   a table without a gap still finds immobile spots, and never above `R`.
//! - `q`: the detections within `R` in the next frame, less the count of
//!   others around a detection in that frame, per detection. What is left is
//!   each detection's own continuation, however far it moved. Averaged with
//!   the previous frame, so that reversing time changes nothing.
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

/// Log odds a link must reach: a wrong link, which switches two identities,
/// costs `exp(SWITCH_COST)` times a missed one, which only breaks a track.
const SWITCH_COST: f64 = 2.5;

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
    /// Per strength class `c` ([`classes`]): `ln((1 - q_c) / q_c)`, the odds
    /// that a detection of the class goes unseen, and `ln((1 - q_c) rho_c)`,
    /// the density of new detections of the class. Empty without strengths:
    /// `log_lambda` prices every link.
    log_unseen: Vec<f64>,
    log_new: Vec<f64>,
}

/// Most strength classes; fewer when the movie has fewer than
/// [`CLASS_ROWS`] detections per class.
const N_CLASSES: usize = 6;
const CLASS_ROWS: usize = 500;

/// Each detection's strength class: equal shares of the detections, cut
/// at the strengths of those ranks, so equal strengths share a class and the
/// classes do not depend on row order. -> (class per row, number of classes).
fn classes(strength: &[f64]) -> (Vec<usize>, usize) {
    let n = strength.len();
    let k = (n / CLASS_ROWS).clamp(1, N_CLASSES);
    let mut sorted = strength.to_vec();
    sorted.sort_by(f64::total_cmp);
    // The lowest strength of each class above the first.
    let cuts: Vec<f64> = (1..k).map(|c| sorted[c * n / k]).collect();
    let class = strength.iter().map(|z| cuts.partition_point(|cut| cut.total_cmp(z).is_le())).collect();
    (class, k)
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
/// Only occupied bins are stored, so the cost follows the pairs, not `r / bin`.
fn gap(frames: &[Frame], pos: &[f64], r: f64, bin: f64, n: usize) -> f64 {
    let nb = (r / bin).floor() as usize;
    if nb < 2 {
        return r;
    }
    let mut count: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for fr in frames {
        for (k, &a) in fr.rows.iter().enumerate() {
            fr.cells.near([pos[2 * a], pos[2 * a + 1]], |k2| {
                if k2 > k {
                    let d = d2(pos, a, fr.rows[k2]).sqrt();
                    let i = (d / bin) as usize;
                    if i < nb {
                        *count.entry(i).or_insert(0.0) += 2.0;
                    }
                }
            });
        }
    }
    // Annulus `i` has area pi bin^2 (2 i + 1); annuli `i0..nb` together
    // pi bin^2 (nb^2 - i0^2).
    let pib2 = std::f64::consts::PI * bin * bin;
    let dens = |i: usize| count.get(&i).map_or(0.0, |c| c / (n as f64 * pib2 * (2 * i + 1) as f64));
    let mid = |i: usize| (i as f64 + 0.5) * bin;
    let mut i0 = (r / (2.0 * bin) - 0.5).ceil().max(0.0) as usize;
    while i0 > 0 && mid(i0 - 1) >= r / 2.0 {
        i0 -= 1;
    }
    while mid(i0) < r / 2.0 {
        i0 += 1;
    }
    let far_count: f64 = count.range(i0..).map(|(_, c)| c).sum();
    let far_area = pib2 * ((nb as f64).powi(2) - (i0 as f64).powi(2));
    let plateau = far_count / (n as f64 * far_area);
    if plateau.is_nan() || plateau <= 0.0 {
        return r;
    }
    let smooth = |i: usize| {
        let lo = if i == 0 { 0.0 } else { dens(i - 1) };
        let hi = if i + 1 < nb { dens(i + 1) } else { 0.0 };
        (lo + dens(i) + hi) / 3.0
    };
    let half = plateau / 2.0;
    // Smoothing is zero away from occupied bins, so only their neighbours can
    // first reach `half`.
    let first = count
        .keys()
        .flat_map(|&j| [j.wrapping_sub(1), j, j + 1])
        .filter(|&i| i < nb)
        .find(|&i| smooth(i) >= half);
    match first {
        None => r,
        Some(0) => mid(0),
        Some(i) => {
            let (s0, s1) = (smooth(i - 1), smooth(i));
            mid(i - 1) + (half - s0) / (s1 - s0) * bin
        }
    }
}

/// Detections of `other` within `r` of each detection of `fr`, summed.
fn pairs_within(fr: &Frame, other: &Frame, pos: &[f64], r: f64) -> f64 {
    let mut count = 0.0;
    for &a in &fr.rows {
        other.cells.near([pos[2 * a], pos[2 * a + 1]], |k| {
            if d2(pos, a, other.rows[k]) < r * r {
                count += 1.0;
            }
        });
    }
    count
}

fn movie_stats(frames: &[Frame], pos: &[f64], r: f64, se2_med: f64, n: usize, class: Option<(&[usize], usize)>) -> Stats {
    // Positions are not known finer than their error: the histogram bin.
    let bin = se2_med.sqrt();
    // Nothing farther than R is looked at, here as everywhere.
    let r0 = (gap(frames, pos, r, bin, n) / 2.0).max(3.0 * (2.0 * se2_med).sqrt()).min(r);
    // Each frame's mean count of other same-frame detections within R.
    let others: Vec<f64> =
        frames.iter().map(|fr| pairs_within(fr, fr, pos, r) / fr.rows.len() as f64 - 1.0).collect();
    let pairs: f64 = frames.iter().zip(&others).map(|(fr, s)| s * fr.rows.len() as f64).sum();
    // One pseudo-pair keeps a movie with no neighbours finite.
    let rho = (pairs + 1.0) / (n as f64 * std::f64::consts::PI * r * r);
    // Continuation both ways in time: the detections within R in the next
    // frame, less that frame's own count of others around a detection. What
    // is left is the detection's own continuation, however far it moved.
    let (t_lo, t_hi) = (frames[0].t, frames[frames.len() - 1].t);
    let (mut cont, mut tried) = (0.0f64, 0.0f64);
    for fr in frames {
        for step in [1, -1] {
            if (step == 1 && fr.t == t_hi) || (step == -1 && fr.t == t_lo) {
                continue; // the movie ends there
            }
            tried += fr.rows.len() as f64;
            if let Ok(i) = frames.binary_search_by_key(&(fr.t + step), |f| f.t) {
                cont += pairs_within(fr, &frames[i], pos, r) - fr.rows.len() as f64 * others[i];
            }
        }
    }
    // The rule of succession, on a count that noise can take outside [0, tried].
    let q = (cont.clamp(0.0, tried) + 1.0) / (tried + 2.0);
    let (log_unseen, log_new) = match class {
        Some((class, k)) => class_stats(frames, pos, r, rho, class, k),
        None => (Vec::new(), Vec::new()),
    };
    Stats { r0, rho, log_lambda: 2.0 * (1.0 - q).ln() + rho.ln() - q.ln(), log_unseen, log_new }
}

/// Per strength class: each detection's continuation, the detections within
/// `R` of it in the next (and previous) frame less the others within `R` of
/// it in its own frame. Its own neighbours, not the frame's mean: a bright
/// particle's neighbourhood (a halo the detector splits, a neighbour it
/// crowds) is in both counts and cancels. -> (`ln((1 - q) / q)`,
/// `ln((1 - q) rho_c)`) per class, `rho_c` the class's share of `rho`.
fn class_stats(frames: &[Frame], pos: &[f64], r: f64, rho: f64, class: &[usize], k: usize) -> (Vec<f64>, Vec<f64>) {
    let within = |a: usize, other: &Frame| -> f64 {
        let mut c = 0.0;
        other.cells.near([pos[2 * a], pos[2 * a + 1]], |j| {
            if d2(pos, a, other.rows[j]) < r * r {
                c += 1.0;
            }
        });
        c
    };
    let (t_lo, t_hi) = (frames[0].t, frames[frames.len() - 1].t);
    let (mut cont, mut tried, mut count) = (vec![0.0; k], vec![0.0; k], vec![0.0; k]);
    for fr in frames {
        for &a in &fr.rows {
            let c = class[a];
            count[c] += 1.0;
            let own = within(a, fr) - 1.0;
            for step in [1, -1] {
                if (step == 1 && fr.t == t_hi) || (step == -1 && fr.t == t_lo) {
                    continue;
                }
                tried[c] += 1.0;
                if let Ok(i) = frames.binary_search_by_key(&(fr.t + step), |f| f.t) {
                    cont[c] += within(a, &frames[i]) - own;
                }
            }
        }
    }
    let total: f64 = count.iter().sum();
    (0..k)
        .map(|c| {
            let q = (cont[c].clamp(0.0, tried[c]) + 1.0) / (tried[c] + 2.0);
            let rho_c = rho * count[c] / total;
            (((1.0 - q) / q).ln(), (1.0 - q).ln() + rho_c.ln())
        })
        .unzip()
}

/// Which frames around each detection hold a detection within `r0`: bit
/// `tau + WINDOW` (tau in -WINDOW..=WINDOW) of `seen` marks a frame inside
/// the movie, of `occupied` one holding such a detection.
fn occupancy(frames: &[Frame], pos: &[f64], r0: f64, n: usize) -> (Vec<u32>, Vec<u32>) {
    let (t_lo, t_hi) = (frames[0].t, frames[frames.len() - 1].t);
    let (mut seen, mut occupied) = (vec![0u32; n], vec![0u32; n]);
    for fr in frames {
        for tau in -WINDOW..=WINDOW {
            let Some(t) = fr.t.checked_add(tau).filter(|t| tau != 0 && (t_lo..=t_hi).contains(t)) else {
                continue;
            };
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
    /// Each detection's strength class, empty without strengths.
    class: Vec<usize>,
    /// The movie's population over `D_k` (log weights): where a new track's
    /// posterior starts, before its occupancy, and where forgetting returns
    /// it. Uniform until a pilot pass measures it ([`population`]).
    base: [f64; N_D],
}

impl Model<'_> {
    /// `ln lambda` for a link `a -> b`.
    fn log_lambda(&self, a: usize, b: usize) -> f64 {
        if self.class.is_empty() {
            self.stats.log_lambda
        } else {
            self.stats.log_unseen[self.class[a]] + self.stats.log_new[self.class[b]]
        }
    }

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
        for (x, base) in t.iter_mut().zip(self.base) {
            *x = ((1.0 - FORGET) * (*x - z).exp() + FORGET * base.exp()).ln();
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
            *out = log_sum_exp(&per_p) + self.base[k];
        }
        let z = log_sum_exp(&lp);
        lp.map(|x| x - z)
    }

    /// One causal pass over `frames` in the order given; `dir` is +1 forward
    /// in time, -1 backward. -> each detection's partner in the pass's next
    /// frame, or `NONE`, and each track's posterior where it ends with the
    /// number of steps behind it.
    fn pass(&self, frames: &[Frame], order: &[usize], dir: i64) -> (Vec<usize>, Vec<([f64; N_D], usize)>) {
        let n = self.se2.len();
        let mut next = vec![NONE; n];
        let mut steps = vec![0usize; n];
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
                            let g = self.log_pred(&w[a], a, b, dd) - self.log_lambda(a, b) - SWITCH_COST;
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
                        steps[b] = steps[a] + 1;
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
        let ends = (0..n).filter(|&a| next[a] == NONE && steps[a] > 0).map(|a| (w[a], steps[a])).collect();
        (next, ends)
    }
}

/// The movie's population over `D_k` from a pass's track ends: each track's
/// posterior weighted by its steps, the tracks of one step left out (one
/// step barely tells `D_k` apart), with `FORGET` of it uniform so no `D_k` is
/// ruled out. -> log weights.
fn population(ends: &[([f64; N_D], usize)]) -> [f64; N_D] {
    let mut acc = [0.0; N_D];
    for (w, steps) in ends.iter().filter(|e| e.1 >= 2) {
        let z = log_sum_exp(w);
        for k in 0..N_D {
            acc[k] += *steps as f64 * (w[k] - z).exp();
        }
    }
    let total: f64 = acc.iter().sum();
    if !(total > 0.0) {
        return [-(N_D as f64).ln(); N_D];
    }
    acc.map(|a| ((1.0 - FORGET) * a / total + FORGET / N_D as f64).ln())
}

/// Links `(y, x)` positions (`pos` is `(N, 2)` row-major) by frame index,
/// with per-detection localization errors `se` (`(N, 2)`, `(se_y, se_x)`).
/// Only consecutive frames are linked; `max_step` is the largest step
/// considered.
pub fn link(frame: &[i64], pos: &[f64], se: &[f64], max_step: f64) -> Result<Linking, String> {
    link_with(frame, pos, se, None, max_step)
}

/// [`link`], with each detection's `strength` (the detector's `detect_z`):
/// the continuation that prices a track end is then read and applied per
/// strength class ([`classes`]).
pub fn link_with(frame: &[i64], pos: &[f64], se: &[f64], strength: Option<&[f64]>, max_step: f64) -> Result<Linking, String> {
    let n = frame.len();
    if let Some(z) = strength {
        if z.len() != n {
            return Err(format!("frame has {n} rows but strengths have {}", z.len()));
        }
        if let Some(i) = (0..n).find(|&i| !z[i].is_finite()) {
            return Err(format!("row {i} has a strength that is not finite"));
        }
    }
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
    let (class, k) = strength.map_or((Vec::new(), 0), classes);
    let stats = movie_stats(&frames, pos, r, se2_med, n, (!class.is_empty()).then_some((&class[..], k)));
    let (seen, occupied) = occupancy(&frames, pos, stats.r0, n);
    let mut model = Model { pos, se2, se2_med, dk, r, stats, seen, occupied, class, base: [-(N_D as f64).ln(); N_D] };

    let forward: Vec<usize> = (0..frames.len()).collect();
    let backward: Vec<usize> = forward.iter().rev().copied().collect();
    if strength.is_some() {
        // Pilot passes measure the movie's population over `D_k`; new tracks
        // start from it rather than from every `D_k` alike, so a track's first
        // steps are priced as the movie's particles move, not as the fastest
        // the search radius allows. Both directions, so that reversing the
        // movie's frames still gives the same links.
        let mut ends = model.pass(&frames, &forward, 1).1;
        ends.extend(model.pass(&frames, &backward, -1).1);
        model.base = population(&ends);
    }
    let fwd = model.pass(&frames, &forward, 1).0;
    let bwd = model.pass(&frames, &backward, -1).0;

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
    /// than `gap` to one already kept in its frame is merged into the nearest,
    /// as a detector would. -> (frame, pos, se, the particles each row stands
    /// for, the kept one first).
    fn movie(rng: &mut Rng, d: &[f64], n_frames: i64, field: f64, p: f64, se: f64, gap: f64)
             -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<Vec<usize>>) {
        let mut x: Vec<[f64; 2]> = d.iter().map(|_| [rng.uniform() * field, rng.uniform() * field]).collect();
        let (mut fr, mut pos, mut who): (Vec<i64>, Vec<f64>, Vec<Vec<usize>>) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..n_frames {
            if t > 0 {
                for (xi, &di) in x.iter_mut().zip(d) {
                    let s = (2.0 * di).sqrt();
                    xi[0] = (xi[0] + s * rng.normal()).rem_euclid(field);
                    xi[1] = (xi[1] + s * rng.normal()).rem_euclid(field);
                }
            }
            let first = fr.len();
            for (i, xi) in x.iter().enumerate() {
                if rng.uniform() >= p {
                    continue;
                }
                let o = [xi[0] + se * rng.normal(), xi[1] + se * rng.normal()];
                let near = (first..fr.len())
                    .map(|k| ((pos[2 * k] - o[0]).hypot(pos[2 * k + 1] - o[1]), k))
                    .min_by(|a, b| a.0.total_cmp(&b.0));
                match near {
                    Some((dist, k)) if dist < gap => who[k].push(i),
                    _ => {
                        fr.push(t);
                        pos.extend_from_slice(&o);
                        who.push(vec![i]);
                    }
                }
            }
        }
        let n = fr.len();
        (fr, pos, vec![se; 2 * n], who)
    }

    /// The share of detections whose particle, or one merged into it, is
    /// detected within `r` in the next frame, both ways in time.
    fn continuation(fr: &[i64], pos: &[f64], who: &[Vec<usize>], n_frames: i64, r: f64) -> f64 {
        let mut at: std::collections::HashMap<(i64, usize), usize> = std::collections::HashMap::new();
        for (i, (&t, ws)) in fr.iter().zip(who).enumerate() {
            for &w in ws {
                at.insert((t, w), i);
            }
        }
        let (mut cont, mut tried) = (0.0, 0.0);
        for (i, (&t, ws)) in fr.iter().zip(who).enumerate() {
            for step in [1, -1] {
                if (0..n_frames).contains(&(t + step)) {
                    tried += 1.0;
                    let hit = ws.iter().any(|&w| at.get(&(t + step, w)).is_some_and(|&j| d2(pos, i, j) < r * r));
                    cont += hit as u8 as f64;
                }
            }
        }
        cont / tried
    }

    /// The continuation `q` that the movie's `lambda` and `rho` imply.
    fn q_of(s: &Stats) -> f64 {
        let lam = s.log_lambda.exp() / s.rho;
        ((2.0 + lam) - ((2.0 + lam).powi(2) - 4.0).sqrt()) / 2.0
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
        let stats = movie_stats(&frames, pos, r, se * se, n, None);
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
        let (q, truth) = (q_of(&s), continuation(&fr, &pos, &who, 30, r));
        assert!((q - truth).abs() < 0.02, "q {q} vs {truth}");
    }

    #[test]
    fn the_continuation_does_not_depend_on_how_far_particles_move() {
        // Seen 70% of the time behind a 2.5 px gap; immobile, or with an rms
        // 2-D step of 4 px, three times r0.
        for (d, seed) in [(0.0, 0x51ed_270b_2c3a_9e11u64), (4.0, 0x7f4a_7c15_9e37_79b9)] {
            let mut rng = Rng(seed);
            let (r, n_frames) = (15.0, 40);
            let (fr, pos, _, who) = movie(&mut rng, &vec![d; 400], n_frames, 250.0, 0.7, 0.2, 2.5);
            let (_, s) = model_for(&fr, &pos, 0.2, r);
            let (q, truth) = (q_of(&s), continuation(&fr, &pos, &who, n_frames, r));
            assert!((q - truth).abs() < 0.02, "D {d}: q {q} vs {truth}");
        }
    }

    #[test]
    fn a_table_without_a_gap_still_gives_r0_and_the_continuation() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let d = vec![0.0; 300];
        let (fr, pos, _, who) = movie(&mut rng, &d, 20, 200.0, 0.9, 0.2, 0.0);
        let (_, s) = model_for(&fr, &pos, 0.2, 10.0);
        assert!(s.r0 >= 3.0 * (2.0f64 * 0.04).sqrt() - 1e-12);
        let (q, truth) = (q_of(&s), continuation(&fr, &pos, &who, 20, 10.0));
        assert!((q - truth).abs() < 0.02, "q {q} vs {truth}");
    }

    #[test]
    fn a_track_posterior_settles_on_its_coefficient() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        let dk: [f64; N_D] = std::array::from_fn(|k| 0.01 * 2f64.powi(k as i32 * 2));
        let model = Model {
            pos: &[], se2: vec![0.01; 2], se2_med: 0.01, dk, r: 10.0,
            stats: Stats { r0: 1.0, rho: 0.01, log_lambda: -10.0, log_unseen: Vec::new(), log_new: Vec::new() },
            seen: vec![0; 2], occupied: vec![0; 2], class: Vec::new(), base: [-(N_D as f64).ln(); N_D],
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
        // With strengths, ties among them included: the classes, the pilot
        // passes and the links must not depend on row order or time's direction.
        let z: Vec<f64> = (0..fr.len()).map(|i| (i % 7) as f64).collect();
        let base_z = links(&fr, &link_with(&fr, &pos, &se, Some(&z), 8.0).unwrap().track);
        assert!(!base_z.is_empty());

        let rev: Vec<i64> = fr.iter().map(|&t| -t).collect();
        assert_eq!(links(&rev, &link(&rev, &pos, &se, 8.0).unwrap().track), base);
        assert_eq!(links(&rev, &link_with(&rev, &pos, &se, Some(&z), 8.0).unwrap().track), base_z);

        let n = fr.len();
        let perm: Vec<usize> = (0..n).map(|i| (i * 7919) % n).collect();
        assert!(n % 7919 != 0);
        let fr2: Vec<i64> = perm.iter().map(|&i| fr[i]).collect();
        let pos2: Vec<f64> = perm.iter().flat_map(|&i| [pos[2 * i], pos[2 * i + 1]]).collect();
        let se2: Vec<f64> = perm.iter().flat_map(|&i| [se[2 * i], se[2 * i + 1]]).collect();
        let unperm = |l: Linking| {
            let mut v: Vec<(usize, usize)> =
                links(&fr2, &l.track).into_iter().map(|(a, b)| (perm[a].min(perm[b]), perm[a].max(perm[b]))).collect();
            v.sort_unstable();
            v
        };
        assert_eq!(unperm(link(&fr2, &pos2, &se2, 8.0).unwrap()), base);
        let z2: Vec<f64> = perm.iter().map(|&i| z[i]).collect();
        assert_eq!(unperm(link_with(&fr2, &pos2, &se2, Some(&z2), 8.0).unwrap()), base_z);
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
    fn extreme_errors_and_frame_numbers_still_link() {
        // Errors far finer than the search radius: the gap histogram follows
        // the pairs, not `R / se`.
        let mut rng = Rng(0x0dd_ba11_cafe_babe);
        let (fr, pos, _, _) = movie(&mut rng, &vec![0.0; 100], 10, 100.0, 0.9, 0.1, 2.0);
        assert!(link(&fr, &pos, &vec![1e-12; pos.len()], 10.0).unwrap().n_tracks > 0);
        // Errors too large for the search radius: `r0` stays within it.
        let (_, s) = model_for(&fr, &pos, 2.0, 3.0);
        assert!(s.r0 <= 3.0, "r0 {}", s.r0);
        // Frame numbers at the ends of their range.
        let l = link(&[i64::MAX - 1, i64::MAX, i64::MIN], &[0.0; 6], &[0.1; 6], 1.0).unwrap();
        assert_eq!(l.track[0], l.track[1]);
        assert_ne!(l.track[0], l.track[2]);
    }

    #[test]
    fn strength_keeps_bright_particles_linked_among_flickering_detections() {
        // Bright particles seen 98% of the time, among detections that never
        // continue (dim flicker): one movie-wide continuation, dragged down by
        // the flicker, cuts the bright particles' longer steps.
        let mut rng = Rng(0x5eed_1e55_c0de_d00d);
        let (n_frames, field) = (30i64, 200.0);
        let (mut fr, mut pos, mut se, who) = movie(&mut rng, &vec![0.25; 300], n_frames, field, 0.98, 0.1, 0.0);
        let bright = fr.len();
        for t in 0..n_frames {
            for _ in 0..200 {
                fr.push(t);
                pos.extend_from_slice(&[rng.uniform() * field, rng.uniform() * field]);
                se.extend_from_slice(&[0.3, 0.3]);
            }
        }
        let strength: Vec<f64> = (0..fr.len()).map(|i| if i < bright { 20.0 } else { 5.0 + rng.uniform() }).collect();
        // Consecutive detections of one bright particle, and how many of them a linking joins.
        let mut at = std::collections::HashMap::new();
        for i in 0..bright {
            at.insert((fr[i], who[i][0]), i);
        }
        let pairs: Vec<(usize, usize)> = (0..bright).filter_map(|i| at.get(&(fr[i] + 1, who[i][0])).map(|&j| (i, j))).collect();
        let score = |l: &Linking| {
            let kept = pairs.iter().filter(|&&(i, j)| l.track[i] == l.track[j]).count() as f64 / pairs.len() as f64;
            let linked = links(&fr, &l.track);
            let wrong = linked.iter().filter(|&&(a, b)| a >= bright || b >= bright || who[a][0] != who[b][0]).count() as f64;
            (kept, wrong / linked.len() as f64)
        };
        let (one_q, one_wrong) = score(&link(&fr, &pos, &se, 6.0).unwrap());
        let (by_z, z_wrong) = score(&link_with(&fr, &pos, &se, Some(&strength), 6.0).unwrap());
        // Seed as written: 0.97 kept and 1.4% wrong with strengths, 0.68 and 6.9% without.
        assert!(by_z > 0.95, "with strengths, {by_z} of the bright particles' links kept");
        assert!(by_z > one_q + 0.1, "with strengths {by_z}, one continuation {one_q}");
        assert!(z_wrong < one_wrong, "wrong links: with strengths {z_wrong}, one continuation {one_wrong}");
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
