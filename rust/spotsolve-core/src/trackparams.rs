//! Measuring the linker's parameters instead of asking for them.
//!
//! Ported from `tracksolve`'s `params`. Every number in [`Params`] is a
//! property of the dataset -- how far a particle moves between frames, how
//! often the detector misses one, how many new particles appear -- so it is
//! measured here and the linker ships with no dials.
//!
//! # The link-free initializer
//!
//! Estimating a step size by linking and looking at the steps is circular:
//! the linking used a step size, and that is what comes back. So the first
//! estimate uses a statistic that needs no association at all -- for every
//! detection, the distance to the NEAREST detection in the following frame.
//! That distance is a mixture of the true successor, when it was detected and
//! happened to be nearest, and an unrelated particle when it was not.
//!
//! The true-successor half is itself a mixture over D: for diffusion `D` the
//! per-axis displacement variance is `2*D*dt + se_a^2 + se_b^2`, so the
//! displacement magnitude is Rayleigh with that scale. Fitting the weights of
//! that mixture over the D grid therefore estimates the POPULATION
//! DISTRIBUTION of D, which is the prior the linker needs most; the immobile
//! population shows up as weight on `D = 0`, whose Rayleigh scale is pure
//! localization jitter.
//!
//! The wrong-neighbour half is taken from the data, not from a Poisson
//! model, and that is deliberate. tracksolve measured a real GEM dataset
//! whose bounding box gives 0.229 detections/um^2 -- predicting a median
//! nearest-neighbour distance of 0.98 um -- against an observed median of
//! 0.69 um, an effective density twice the nominal one, because the box spans
//! the nucleus and the space outside the cell where no particle can be.
//! Assuming homogeneity would push that error into every score. So the
//! reference is the observed distribution of WITHIN-frame nearest-neighbour
//! distances, as a kernel density on log distance ([`LogKde`]), which absorbs
//! clustering and exclusion zones automatically.
//!
//! # The refinement loop, and why it is soft
//!
//! Then three rounds of link, re-estimate, re-link. The failure mode is a
//! feedback loop with a fixed point at zero: under-link, so the estimated D
//! shrinks, so the gate tightens, so fewer links survive. Three guards, the
//! first mattering most: re-estimation uses the POSTERIOR-weighted D of each
//! track rather than its point estimate, which is what makes this a soft EM
//! rather than the hard EM that collapses; each update is damped halfway
//! toward the new value; and every iterate is clamped within `EM_CLAMP` of
//! the link-free estimate, which cannot drift because it never saw a link.
//!
//! tracksolve has a fourth guard -- stop if the total evidence falls -- which
//! is not ported. It fired in 0 of 27 fits across three densities and three
//! detection/clutter settings, and it cannot fire honestly in this mode
//! anyway: the evidence it compares moves with the clutter intensity, which
//! in a single-scan linking is estimated from unlinked detections, of which
//! there are none (every detection lands in some track). The clutter
//! intensity itself cancels out of every linking decision; see `track`.

use crate::track::{Buckets, CellIndex, Detections, Filter, Linking, Params};
use crate::track::{D_GRID_DECADES, D_GRID_N, FluxModel, link};

/// Rounds of link / re-estimate / re-link. Two is usually enough; the third
/// rarely moves anything and is kept as headroom.
pub const EM_ITERS: usize = 3;

/// Each update moves this far from the previous value toward the new one.
/// Full steps oscillate on small datasets.
pub const EM_DAMP: f64 = 0.5;

/// How far any iterate may drift from the link-free estimate, as a
/// multiplicative factor.
pub const EM_CLAMP: f64 = 4.0;

/// How far above the bulk diffusion coefficient the grid must reach, so a
/// population with a fast tail is representable rather than railed at the top
/// grid point.
pub const HEADROOM: f64 = 10.0;

/// Cap on the CRLB variance inflation. spotsolve documents pull standard
/// deviations up to about 3 on defocused emitters, so 25 in variance is
/// generous headroom; hitting the cap means something is wrong upstream.
pub const SE_INFLATE_MAX: f64 = 25.0;

/// One iteration's parameters, for the record. `fit` returns the whole
/// trajectory so a run that goes wrong says so rather than quietly returning
/// a worse answer.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub label: String,
    pub p_cont: f64,
    pub lam_birth: f64,
    pub se_inflate: f64,
    /// Population mean of D under the fitted prior, px^2/frame.
    pub d_mean: f64,
    /// Prior weight on the exact-zero grid point.
    pub d_immobile: f64,
    pub n_tracks: u32,
}

fn snapshot(label: &str, p: &Params, n_tracks: u32) -> Snapshot {
    let w: Vec<f64> = p.d_logprior.iter().map(|v| v.exp()).collect();
    Snapshot {
        label: label.to_string(),
        p_cont: p.p_cont,
        lam_birth: p.lam_birth,
        se_inflate: p.se_inflate,
        d_mean: w.iter().zip(&p.d_grid).map(|(a, b)| a * b).sum(),
        d_immobile: w[0],
        n_tracks,
    }
}

/// numpy's `median`: the mean of the two middle values on an even count.
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// Candidate diffusion coefficients: an exact zero plus a log-spaced fan.
///
/// `grid[0]` is exactly 0 and means a genuinely immobile particle whose
/// apparent motion is all localization error. The remaining points are
/// log-spaced around `d_floor`, the resolvability floor `se^2/dt`, and one
/// lands on it exactly.
///
/// The two sides deserve different reach: everything below the floor is
/// unresolvable from zero by definition, so extra points there only subdivide
/// the immobile mass, while the upper end has to reach the fastest particles
/// present. Since the floor comes from the reported CRLB, an understated CRLB
/// drags the whole grid down with it -- errors 3x too small move the floor 9x
/// -- so the caller widens `above` from the observed displacements instead.
pub fn d_grid(d_floor: f64, n: usize, below: f64, above: f64) -> Vec<f64> {
    let k = n - 1;
    let step = (below + above) / (k - 1) as f64;
    let k_below = (below / step).round_ties_even();
    let mut out = Vec::with_capacity(n);
    out.push(0.0);
    for i in 0..k {
        out.push(d_floor * 10f64.powf(step * (i as f64 - k_below)));
    }
    out
}

/// Gaussian kernel density on LOG distance, evaluated by linear binning.
///
/// Log, because nearest-neighbour distances span decades and a fixed
/// bandwidth in linear distance either oversmooths the close pairs that
/// matter or undersmooths the far tail that does not. Binned, because the
/// exact sum is O(samples x queries) and both are the number of detections:
/// a 30k-detection movie would be 10^9 exponentials, three times over.
///
/// Bandwidth is Scott's rule, `sd * n^(-1/5)`, matching
/// `scipy.stats.gaussian_kde` (with the unbiased sd it uses). Bins are
/// `h/50` wide and the kernel is truncated at `6h`, which puts the
/// approximation error near 1e-5 relative -- measured against the exact sum
/// in `kde_matches_the_exact_sum`.
pub struct LogKde {
    lo: f64,
    step: f64,
    dens: Vec<f64>,
}

impl LogKde {
    /// `samples` are distances; non-positive ones are dropped. `None` when
    /// there is nothing to estimate from, which the caller reads as a
    /// vanishing density.
    pub fn new(samples: &[f64]) -> Option<Self> {
        let logs: Vec<f64> = samples
            .iter()
            .filter(|&&s| s > 0.0)
            .map(|s| s.ln())
            .collect();
        let n = logs.len();
        if n < 8 {
            return None;
        }
        let mean = logs.iter().sum::<f64>() / n as f64;
        let var = logs.iter().map(|l| (l - mean) * (l - mean)).sum::<f64>() / (n - 1) as f64;
        let h = var.sqrt() * (n as f64).powf(-0.2);
        if !(h.is_finite() && h > 0.0) {
            return None;
        }
        let lo = logs.iter().cloned().fold(f64::INFINITY, f64::min) - 6.0 * h;
        let hi = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 6.0 * h;
        let step = h / 50.0;
        let nb = (((hi - lo) / step).ceil() as usize + 1).clamp(64, 1 << 17);
        let step = (hi - lo) / (nb - 1) as f64;

        let mut mass = vec![0.0f64; nb];
        for l in &logs {
            let t = (l - lo) / step;
            let k = (t.floor() as usize).min(nb - 2);
            let frac = t - k as f64;
            mass[k] += 1.0 - frac;
            mass[k + 1] += frac;
        }
        let half = ((6.0 * h / step).ceil() as usize).max(1);
        let norm = 1.0 / (n as f64 * h * (std::f64::consts::TAU).sqrt());
        let kern: Vec<f64> = (0..=half)
            .map(|j| {
                let u = j as f64 * step / h;
                (-0.5 * u * u).exp() * norm
            })
            .collect();
        let mut dens = vec![0.0f64; nb];
        for (k, &m) in mass.iter().enumerate() {
            if m == 0.0 {
                continue;
            }
            let j0 = k.saturating_sub(half);
            let j1 = (k + half).min(nb - 1);
            for (j, d) in dens.iter_mut().enumerate().take(j1 + 1).skip(j0) {
                *d += m * kern[j.abs_diff(k)];
            }
        }
        Some(LogKde { lo, step, dens })
    }

    /// Density of the DISTANCE at `r`, i.e. the log-density divided by `r`
    /// (change of variables). Floored rather than zero so a log is always
    /// finite.
    pub fn eval(&self, r: f64) -> f64 {
        if r.is_nan() || r <= 0.0 {
            return 1e-300;
        }
        let t = (r.ln() - self.lo) / self.step;
        if t < 0.0 || t > (self.dens.len() - 1) as f64 {
            return 1e-300;
        }
        let k = (t.floor() as usize).min(self.dens.len() - 2);
        let f = t - k as f64;
        let d = self.dens[k] * (1.0 - f) + self.dens[k + 1] * f;
        (d / r).max(1e-300)
    }
}

fn density_of(kde: &Option<LogKde>, r: f64) -> f64 {
    match kde {
        Some(k) => k.eval(r),
        None => 1e-300,
    }
}

/// For each detection, the distance to the nearest one in the adjacent frame
/// (`+1` forward, `-1` backward), with the row it came from. No association
/// is formed. The backward direction is what separates a false positive from
/// a track that merely ended: a real particle has a partner on at least one
/// side.
pub fn nn_adjacent(d: &Detections, index: &CellIndex, forward: bool) -> (Vec<f64>, Vec<usize>) {
    let (mut dist, mut src) = (Vec::new(), Vec::new());
    let mut b = Buckets::default();
    for f in 0..d.n_frames {
        let other = if forward { f + 1 } else { f.wrapping_sub(1) };
        if other >= d.n_frames {
            continue;
        }
        if d.frame_rows(f).is_empty() || d.frame_rows(other).is_empty() {
            continue;
        }
        index.bucket(d, other, &mut b);
        for r in d.frame_rows(f) {
            if let Some(q) = index.nearest(d, &b, [d.pos[2 * r], d.pos[2 * r + 1]], None) {
                dist.push(q);
                src.push(r);
            }
        }
    }
    (dist, src)
}

/// Within-frame nearest-neighbour distances: the distance from a particle to
/// its nearest UNRELATED neighbour, under whatever spatial structure the
/// sample actually has.
pub fn nn_within(d: &Detections, index: &CellIndex) -> Vec<f64> {
    let mut out = Vec::new();
    let mut b = Buckets::default();
    for f in 0..d.n_frames {
        if d.frame_rows(f).len() < 2 {
            continue;
        }
        index.bucket(d, f, &mut b);
        for r in d.frame_rows(f) {
            if let Some(q) = index.nearest(d, &b, [d.pos[2 * r], d.pos[2 * r + 1]], Some(r)) {
                out.push(q);
            }
        }
    }
    out
}

fn rayleigh(r: f64, s2: f64) -> f64 {
    r / s2 * (-0.5 * r * r / s2).exp()
}

/// Posterior, per detection, that its nearest neighbour in the adjacent
/// frame is a true link. `NaN` where the question does not arise: a
/// detection in the first frame has no previous frame, and its lack of a
/// backward partner is not evidence of anything.
fn link_responsibility(
    d: &Detections,
    index: &CellIndex,
    grid: &[f64],
    w: &[f64],
    pi: f64,
    kde: &Option<LogKde>,
    forward: bool,
) -> Vec<f64> {
    let mut out = vec![f64::NAN; d.n_dets()];
    let (r, src) = nn_adjacent(d, index, forward);
    for (i, &s) in src.iter().enumerate() {
        let noise = d.se2[2 * s] + d.se2[2 * s + 1];
        let a: f64 = grid
            .iter()
            .zip(w)
            .map(|(&g, &wm)| pi * rayleigh(r[i], 2.0 * g + noise) * wm)
            .sum();
        let b = (1.0 - pi) * density_of(kde, r[i]);
        out[s] = a / (a + b).max(1e-300);
    }
    out
}

/// Estimate every parameter without ever forming an association.
pub fn initialize(d: &Detections) -> Params {
    let n = d.n_dets();
    let index = CellIndex::new(d);
    let se2sum: Vec<f64> = (0..n).map(|r| d.se2[2 * r] + d.se2[2 * r + 1]).collect();
    let area = d.bbox_area();
    let uniform = |grid: Vec<f64>| {
        let m = grid.len();
        Params {
            d_grid: grid,
            d_logprior: vec![(1.0 / m as f64).ln(); m],
            p_cont: 0.8,
            lam_birth: 1.0 / area,
            se_inflate: 1.0,
        }
    };
    if n == 0 {
        return uniform(d_grid(
            1.0,
            D_GRID_N,
            D_GRID_DECADES / 2.0,
            D_GRID_DECADES / 2.0,
        ));
    }
    let d_floor = median(&se2sum) / 2.0;
    let (r, src) = nn_adjacent(d, &index, true);

    // Reach the fastest particles actually present, whatever the CRLB
    // claims. The MEDIAN nearest-neighbour displacement is the right
    // statistic: it is dominated by true links, while a high quantile is
    // dominated by the wrong-neighbour component and blows the grid up
    // (tracksolve measured the 99th percentile putting the top of the grid at
    // 97 um^2/s against a true 0.3).
    let mut above = D_GRID_DECADES / 2.0;
    if r.len() >= 16 {
        let s_bulk = median(&r) / 1.1774; // a Rayleigh median is 1.177 scales
        let noise = median(&se2sum);
        let d_bulk = (s_bulk * s_bulk - noise).max(0.0) / 2.0;
        if d_bulk > 0.0 {
            above = above.max((HEADROOM * d_bulk / d_floor).log10());
        }
    }
    let grid = d_grid(d_floor, D_GRID_N, D_GRID_DECADES / 2.0, above);
    let m = grid.len();
    if r.len() < 16 {
        return uniform(grid);
    }

    let kde = LogKde::new(&nn_within(d, &index));
    let nr = r.len();
    let mut link_pdf = vec![0.0f64; nr * m];
    for i in 0..nr {
        let noise = d.se2[2 * src[i]] + d.se2[2 * src[i] + 1];
        for (j, &g) in grid.iter().enumerate() {
            link_pdf[i * m + j] = rayleigh(r[i], 2.0 * g + noise);
        }
    }
    let wrong_pdf: Vec<f64> = r.iter().map(|&v| density_of(&kde, v)).collect();

    let mut w = vec![1.0 / m as f64; m];
    let mut pi = 0.7f64;
    let mut prev = f64::NEG_INFINITY;
    let mut resp = vec![0.0f64; m];
    for _ in 0..60 {
        let mut ll = 0.0;
        let mut acc = vec![0.0f64; m];
        let mut pi_sum = 0.0;
        for i in 0..nr {
            let mut tot = (1.0 - pi) * wrong_pdf[i];
            for j in 0..m {
                resp[j] = pi * link_pdf[i * m + j] * w[j];
                tot += resp[j];
            }
            ll += tot.max(1e-300).ln();
            for j in 0..m {
                let q = resp[j] / tot.max(1e-300);
                acc[j] += q;
                pi_sum += q;
            }
        }
        pi = pi_sum / nr as f64;
        let s: f64 = acc.iter().sum();
        for (wj, a) in w.iter_mut().zip(&acc) {
            *wj = a / s.max(1e-300);
        }
        if (ll - prev).abs() < 1e-8 * 1.0f64.max(ll.abs()) {
            break;
        }
        prev = ll;
    }

    // Births, softly and in both directions: a detection is a birth to the
    // extent that its forward neighbour is a true link and its backward one
    // is not. Counting detections with no plausible neighbour instead
    // conflates a new particle with one the detector merely missed, which
    // tracksolve measured returning clutter intensities on simulations
    // containing no clutter at all.
    let fwd = link_responsibility(d, &index, &grid, &w, pi, &kde, true);
    let bwd = link_responsibility(d, &index, &grid, &w, pi, &kde, false);
    let n_int = (d.n_frames.max(2) - 2).max(1) as f64;
    let interior: Vec<usize> = (0..n)
        .filter(|&i| !fwd[i].is_nan() && !bwd[i].is_nan())
        .collect();
    let lam_birth = if interior.is_empty() {
        1.0 / (area * d.n_frames.max(1) as f64)
    } else {
        let n_birth: f64 = interior.iter().map(|&i| fwd[i] * (1.0 - bwd[i])).sum();
        (n_birth / (area * n_int)).max(1.0 / (area * n_int))
    };

    Params {
        d_grid: grid,
        d_logprior: w.iter().map(|v| v.max(1e-300).ln()).collect(),
        p_cont: pi.clamp(0.05, 0.995),
        lam_birth,
        se_inflate: 1.0,
    }
}

fn mix(old: f64, new: f64, damp: f64) -> f64 {
    damp * old + (1.0 - damp) * new
}

fn bounded(old: f64, new: f64, anchor: f64, damp: f64, clamp: f64, lo: f64, hi: f64) -> f64 {
    mix(old, new, damp).clamp(lo.max(anchor / clamp), hi.min(anchor * clamp))
}

fn bounded_log(old: f64, new: f64, anchor: f64, damp: f64, clamp: f64) -> f64 {
    let v = mix(old.max(1e-300).ln(), new.max(1e-300).ln(), damp).exp();
    v.clamp(anchor / clamp, anchor * clamp)
}

/// One soft-EM update of the parameters from a linking.
pub fn refine(d: &Detections, p: &Params, l: &Linking, anchor: &Params) -> Params {
    let m = p.m();
    let n_rows = d.n_dets();
    let n_tracks = l.n_tracks.max(1);
    let mut acc = vec![0.0f64; m];
    let mut var_p = vec![0.0; 2 * m];
    let mut ll = vec![0.0; m];
    let (mut cross_sum, mut cross_n) = (0.0f64, 0usize);
    let (mut rep_sum, mut rep_n) = (0.0f64, 0usize);

    for seq in l.tracks() {
        if seq.len() < 2 {
            continue;
        }
        let z0 = [d.pos[2 * seq[0]], d.pos[2 * seq[0] + 1]];
        let mut f = Filter::root(z0, [d.se2[2 * seq[0]], d.se2[2 * seq[0] + 1]], p);
        for &r in &seq[1..] {
            let z = [d.pos[2 * r], d.pos[2 * r + 1]];
            let rv = [d.se2[2 * r] * p.se_inflate, d.se2[2 * r + 1] * p.se_inflate];
            f.predict(&p.d_grid, &mut var_p);
            f.loglik(&var_p, z, rv, &mut ll);
            f.marginalize(&ll);
            f.update(&var_p, z, rv);
        }
        // With D static the smoothed posterior equals the final filtered
        // one, which is what the filter has just accumulated.
        for (a, &w) in acc.iter_mut().zip(&f.logw) {
            *a += w.exp();
        }
        // The CRLB inflation comes from the lag-1 displacement covariance,
        // not from standardized innovations. Over one step both 2*D*dt and
        // se^2 widen the displacement, so an understated error is simply
        // re-explained by a larger D and the pulls come back at 1.00 with
        // nothing detected -- tracksolve measured exactly that on data whose
        // true variance factor was 9. What separates them is the
        // CORRELATION: localization error is independent per frame, so it
        // makes consecutive displacements anticorrelated, and diffusion does
        // not. Per axis E[step_n * step_n+1] = -se^2, with no D in it
        // (Vestergaard, Blainey & Flyvbjerg 2014, PRE 89:022726, at R = 0).
        for a in 0..2 {
            for k in 0..seq.len() - 2 {
                let s1 = d.pos[2 * seq[k + 1] + a] - d.pos[2 * seq[k] + a];
                let s2 = d.pos[2 * seq[k + 2] + a] - d.pos[2 * seq[k + 1] + a];
                cross_sum += s1 * s2;
                cross_n += 1;
            }
            for &r in &seq {
                rep_sum += d.se2[2 * r + a];
                rep_n += 1;
            }
        }
    }

    let tot: f64 = acc.iter().sum();
    let d_prior: Vec<f64> = if tot > 0.0 {
        acc.iter().map(|v| v / tot).collect()
    } else {
        p.d_logprior.iter().map(|v| v.exp()).collect()
    };

    let area = d.bbox_area();
    let frames = d.n_frames.max(1) as f64;
    let floor = 1.0 / (area * frames);
    let lam_birth_new = (n_tracks as f64 / (area * frames)).max(floor);
    let p_cont_new = (n_rows.saturating_sub(n_tracks as usize)) as f64 / n_rows.max(1) as f64;

    let mut se_inflate = p.se_inflate;
    if cross_n >= 64 {
        let se2_hat = -(cross_sum / cross_n as f64);
        let rep = rep_sum / rep_n.max(1) as f64;
        if se2_hat > 0.0 && rep > 0.0 {
            se_inflate = (se2_hat / rep).clamp(1.0, SE_INFLATE_MAX);
        }
    }

    let mixed: Vec<f64> = p
        .d_logprior
        .iter()
        .zip(&d_prior)
        .map(|(&lo, &nw)| mix(lo.exp(), nw, EM_DAMP).max(1e-300))
        .collect();
    let s: f64 = mixed.iter().sum();
    Params {
        d_grid: p.d_grid.clone(),
        d_logprior: mixed.iter().map(|v| (v / s).ln()).collect(),
        p_cont: bounded(
            p.p_cont,
            p_cont_new,
            anchor.p_cont,
            EM_DAMP,
            EM_CLAMP,
            0.05,
            0.995,
        ),
        lam_birth: bounded_log(
            p.lam_birth,
            lam_birth_new,
            anchor.lam_birth,
            EM_DAMP,
            EM_CLAMP,
        ),
        se_inflate: mix(p.se_inflate, se_inflate, EM_DAMP).clamp(1.0, SE_INFLATE_MAX),
    }
}

/// Initialize, then refine by linking and re-estimating. Returns the final
/// parameters and every iterate.
/// The brightness cue's parameters, from a positions-only linking of `d`.
///
/// `lf` and `lf_var` are each detection's log flux and its variance, in the
/// caller's row order. Within linked tracks, with `e = lf_var + tau2` per
/// detection, the one-frame change has variance `V1 = 2e + q` and the
/// two-frame change `V2 = 2e + 2q`, so `q = V2 - V1` and
/// `tau2 = (2 V1 - V2) / 2 - mean(lf_var)`. Both variances are robust --
/// `(1.4826 MAD)^2` -- because wrong links put a heavy tail on the changes.
/// Trimming that tail instead biased clean data: with a true `tau2` of 0.05,
/// dropping the largest 5% of changes read 0.024.
///
/// The population density is a 40-bin histogram over the 0.5-99.5% range,
/// add-one smoothed.
pub fn flux_model(d: &Detections, p: &Params, lf: Vec<f64>, lf_var: Vec<f64>) -> FluxModel {
    let l = link(d, p);
    let mut frame_of = vec![0usize; d.n_dets()];
    for f in 0..d.n_frames {
        for r in d.frame_rows(f) {
            frame_of[r] = f;
        }
    }
    let at = |r: usize| lf[d.order[r]];
    let (mut one, mut two, mut vsum, mut vn) = (Vec::new(), Vec::new(), 0.0, 0usize);
    for rows in l.tracks() {
        for (i, &r) in rows.iter().enumerate() {
            vsum += lf_var[d.order[r]];
            vn += 1;
            if let Some(&r1) = rows.get(i + 1) {
                if frame_of[r1] == frame_of[r] + 1 {
                    one.push(at(r1) - at(r));
                    if let Some(&r2) = rows.get(i + 2) {
                        if frame_of[r2] == frame_of[r] + 2 {
                            two.push(at(r2) - at(r));
                        }
                    }
                }
            }
        }
    }
    let robust_var = |v: &mut Vec<f64>| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(f64::total_cmp);
        let med = v[v.len() / 2];
        let mut dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
        dev.sort_by(f64::total_cmp);
        (1.4826 * dev[dev.len() / 2]).powi(2)
    };
    let (v1, v2) = (robust_var(&mut one), robust_var(&mut two));
    let mean_var = if vn > 0 { vsum / vn as f64 } else { 0.0 };
    let (tau2, q) = if v1.is_finite() && v2.is_finite() {
        (((2.0 * v1 - v2) / 2.0 - mean_var).max(0.0), (v2 - v1).max(1e-4))
    } else {
        (0.0, 1e-4)
    };

    let mut sorted = lf.clone();
    sorted.sort_by(f64::total_cmp);
    let pick = |q: f64| sorted[((sorted.len().max(1) - 1) as f64 * q) as usize];
    let (lo, hi) = if sorted.is_empty() { (0.0, 1.0) } else { (pick(0.005), pick(0.995)) };
    let bins = 40usize;
    let step = ((hi - lo) / bins as f64).max(1e-9);
    let mut counts = vec![0.0f64; bins];
    for &v in &lf {
        let i = ((v - lo) / step).floor().clamp(0.0, (bins - 1) as f64) as usize;
        counts[i] += 1.0;
    }
    let total = lf.len() as f64 + bins as f64;
    let logdens = counts.iter().map(|c| ((c + 1.0) / (total * step)).ln()).collect();
    FluxModel { lf, lf_var, tau2, q, lo, step, logdens }
}

pub fn fit(d: &Detections) -> (Params, Vec<Snapshot>) {
    let mut p = initialize(d);
    let anchor = p.clone();
    let mut traj = vec![snapshot("initialize", &p, 0)];
    for it in 0..EM_ITERS {
        let l = link(d, &p);
        p = refine(d, &p, &l, &anchor);
        traj.push(snapshot(&format!("iter{}", it + 1), &p, l.n_tracks));
    }
    (p, traj)
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
            let u1 = self.uniform().max(1e-12);
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
        }
    }

    /// `flux_model` reads the brightness noise it was shown: immobile,
    /// well-separated particles (so the linking is right), a log-flux random
    /// walk with `q = 0.02` per frame, and detections with a reported
    /// variance of 0.01 plus an unreported scatter `tau2 = 0.05`.
    #[test]
    fn flux_model_recovers_the_brightness_noise() {
        let (n, frames) = (100usize, 40usize);
        let mut rng = Rng(7);
        let mut level: Vec<f64> = (0..n).map(|_| 7.0 + rng.normal()).collect();
        let (mut frame, mut pos, mut se, mut lf, mut var) = (vec![], vec![], vec![], vec![], vec![]);
        for f in 0..frames {
            for (k, l) in level.iter_mut().enumerate() {
                if f > 0 {
                    *l += 0.02f64.sqrt() * rng.normal();
                }
                frame.push(f as i64);
                pos.extend([20.0 * (k / 10) as f64 + 0.05 * rng.normal(), 20.0 * (k % 10) as f64 + 0.05 * rng.normal()]);
                se.extend([0.05, 0.05]);
                lf.push(*l + (0.01f64 + 0.05).sqrt() * rng.normal());
                var.push(0.01);
            }
        }
        let d = Detections::new(&frame, &pos, &se).unwrap();
        let (p, _) = fit(&d);
        let m = flux_model(&d, &p, lf, var);
        assert!((m.tau2 - 0.05).abs() < 0.015, "tau2 {}", m.tau2);
        assert!((m.q - 0.02).abs() < 0.01, "q {}", m.q);
    }

    /// The resolvability floor must land exactly on a grid point, so
    /// mobile-vs-immobile can split the grid there without interpolating.
    #[test]
    fn the_floor_is_a_grid_point() {
        for above in [1.5, 2.0, 3.7] {
            let g = d_grid(0.37, D_GRID_N, 1.5, above);
            assert_eq!(g.len(), D_GRID_N);
            assert_eq!(g[0], 0.0);
            assert!(g.windows(2).all(|w| w[1] > w[0]));
            assert!(
                g.iter().any(|v| (v - 0.37).abs() < 1e-12),
                "floor missing from {g:?}"
            );
        }
    }

    /// The binned density must agree with the exact Gaussian sum it stands
    /// in for. 1e-4 relative is far below what the EM that consumes it can
    /// notice, and it costs O(bins) instead of O(samples x queries).
    #[test]
    fn kde_matches_the_exact_sum() {
        let mut rng = Rng(7);
        let s: Vec<f64> = (0..4000)
            .map(|i| {
                if i % 4 == 0 {
                    (0.3 + 0.2 * rng.uniform()).exp()
                } else {
                    (2.0 * rng.normal()).exp()
                }
            })
            .collect();
        let kde = LogKde::new(&s).unwrap();
        let logs: Vec<f64> = s.iter().map(|v| v.ln()).collect();
        let n = logs.len() as f64;
        let mean = logs.iter().sum::<f64>() / n;
        let var = logs.iter().map(|l| (l - mean) * (l - mean)).sum::<f64>() / (n - 1.0);
        let h = var.sqrt() * n.powf(-0.2);
        for i in 0..200 {
            let r = s[i * 7 % s.len()];
            let exact: f64 = logs
                .iter()
                .map(|l| {
                    let u = (r.ln() - l) / h;
                    (-0.5 * u * u).exp()
                })
                .sum::<f64>()
                / (n * h * std::f64::consts::TAU.sqrt())
                / r;
            let got = kde.eval(r);
            assert!(
                (got - exact).abs() <= 1e-4 * exact,
                "r={r}: binned {got}, exact {exact}"
            );
        }
    }

    /// A movie of Brownian particles, 40% of them immobile: the link-free
    /// initializer must find both populations without ever linking anything.
    #[test]
    fn initializer_recovers_the_population_of_d() {
        let mut rng = Rng(4242);
        let (n_p, n_f, field) = (120usize, 25usize, 60.0);
        let d_true: Vec<f64> = (0..n_p)
            .map(|i| if i % 5 < 2 { 0.0 } else { 0.8 })
            .collect();
        let mut pos: Vec<[f64; 2]> = (0..n_p)
            .map(|_| [rng.uniform() * field, rng.uniform() * field])
            .collect();
        let (mut frame, mut xy, mut se) = (vec![], vec![], vec![]);
        for f in 0..n_f {
            for i in 0..n_p {
                if f > 0 {
                    let s = (2.0 * d_true[i]).sqrt();
                    pos[i][0] += s * rng.normal();
                    pos[i][1] += s * rng.normal();
                }
                frame.push(f as i64);
                xy.push(pos[i][0] + 0.08 * rng.normal());
                xy.push(pos[i][1] + 0.08 * rng.normal());
                se.push(0.08);
                se.push(0.08);
            }
        }
        let d = Detections::new(&frame, &xy, &se).unwrap();
        let p = initialize(&d);
        let w: Vec<f64> = p.d_logprior.iter().map(|v| v.exp()).collect();
        let immobile: f64 = w
            .iter()
            .zip(&p.d_grid)
            .filter(|&(_, &g)| g < 0.05)
            .map(|(&a, _)| a)
            .sum();
        let d_mean: f64 = w.iter().zip(&p.d_grid).map(|(&a, &g)| a * g).sum();
        assert!(
            (0.25..0.55).contains(&immobile),
            "immobile fraction {immobile}, expected about 0.4"
        );
        assert!(
            (0.3..0.8).contains(&d_mean),
            "mean D {d_mean}, expected about 0.48 (0.6 * 0.8)"
        );
        assert!(
            p.p_cont > 0.8,
            "p_cont {} with no missed detections",
            p.p_cont
        );
        // And the fit must not wander away from it.
        let (q, traj) = fit(&d);
        assert_eq!(traj.len(), EM_ITERS + 1);
        assert!(q.p_cont > 0.8 && q.se_inflate < 2.0, "{q:?}");
    }
}
