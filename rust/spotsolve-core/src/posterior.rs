//! The probability that an emitter is real rather than noise, by empirical
//! Bayes over a movie (Efron's two groups).
//!
//! Candidates are the window fits seeded and kept above a low bar
//! ([`CANDIDATE`]), each with its root likelihood ratio `z`. Their heights
//! have density `f0(z) + f1(z)` per pixel: `f0` the maxima noise alone
//! gives, known from the likelihood-ratio field ([`prefilter::false_rate`]),
//! and `f1` the emitters'. An emitter of height `z` is real with
//! probability
//!
//! ```text
//! P(real | z) = f1(z) / (f0(z) + f1(z))
//! ```
//!
//! `f1` is fitted from [`Z_LO`] to the highest candidate: heights are
//! counted in bins even in `log z`, each count Poisson with mean `E0 + E1`,
//! `E0` the bin's expected noise maxima and `log E1` a polynomial in `log
//! z`. A spread of fluxes gives a density smooth in `log z`, so bright
//! emitters inform the faint end of `f1` as faint ones do. A weak prior,
//! [`PRIOR`] emitters per pixel, holds `f1` up where candidates are few. A
//! candidate is not evidence for itself: `P(real | z)` at a candidate's
//! height comes from the fit without it. Every decision is then a bar on
//! `z`: the least `u` from which `P(real | z) >= p` ([`Posterior::cut`]).

use crate::linalg::Chol;
use crate::prefilter;

/// Root likelihood ratio at which candidates are seeded and kept: a full
/// unit under [`Z_LO`], so that every maximum above `Z_LO` has a seed
/// though its seed score falls short of its `z`.
pub const CANDIDATE: f64 = 2.0;
/// Heights are counted from here, where noise maxima are still dense
/// enough to anchor the null.
pub const Z_LO: f64 = 3.0;
/// Emitters per pixel assumed before the data, spread evenly in `log z`, a
/// decade of `z` holding them all: pseudo-counts added to every bin. A
/// movie's emitters outweigh them many times over; with few candidates
/// (a small or sparse frame) they keep `f1` from vanishing between them.
pub const PRIOR: f64 = 1e-4;
/// Bins in `log z`, and the polynomial degree of `log f1`.
const BINS: usize = 60;
const DEGREE: usize = 3;

/// `log f1` per unit `log z`, a polynomial in `(log z - centre) / scale`
/// on `[log Z_LO, log z_max]`, with the null's search widths: per bin, the
/// fit without the candidate nearest to it.
#[derive(Clone, Debug)]
pub struct Posterior {
    coef: Vec<Vec<f64>>,
    centre: f64,
    scale: f64,
    z_max: f64,
    widths: (f64, f64),
}

impl Posterior {
    /// Fit `f1` to candidate heights `z` found over `area` pixels; the null
    /// is that of a search over widths `[lo, hi]`.
    ///
    /// A candidate is not evidence for itself: a height is judged on the
    /// fit without the candidate nearest to it, its own count where it is a
    /// candidate's, and otherwise the one a refit of it most likely is.
    /// Wherever noise alone expects well under one candidate (a small frame
    /// or ROI), a lone noise maximum would otherwise raise `f1` at its own
    /// height and pass.
    pub fn fit(z: &[f64], area: f64, lo: f64, hi: f64) -> Self {
        let nb = BINS;
        // Up to the highest candidate, and at least a decade.
        let z_max = z.iter().copied().fold(10.0 * Z_LO, f64::max);
        let (l0, l1) = (Z_LO.ln(), z_max.ln());
        let bin = (l1 - l0) / nb as f64;
        let edge = |i: usize| (l0 + i as f64 * bin).exp();
        let mut count = vec![0.0; nb];
        for &v in z.iter().filter(|&&v| v >= Z_LO) {
            count[Self::bin_of(v, l0, bin)] += 1.0;
        }
        let null: Vec<f64> = (0..nb).map(|i| area * (prefilter::false_rate(edge(i), lo, hi) - prefilter::false_rate(edge(i + 1), lo, hi))).collect();
        let prior = PRIOR * area * bin / std::f64::consts::LN_10;
        let (centre, scale) = (0.5 * (l0 + l1), 0.5 * (l1 - l0));
        let x: Vec<f64> = (0..nb).map(|i| (l0 + (i as f64 + 0.5) * bin - centre) / scale).collect();
        let offset = (area * bin).ln();
        // From a flat `f1` at the excess over the null.
        let excess = (count.iter().sum::<f64>() - null.iter().sum::<f64>()).max(1.0);
        let mut start = vec![0.0; DEGREE + 1];
        start[0] = (excess / nb as f64).ln() - offset;
        let fit = |count: &[f64], start: &[f64]| {
            let with_prior: Vec<f64> = count.iter().map(|c| c + prior).collect();
            fit_bins(&x, &with_prior, &null, offset, start.to_vec())
        };
        let all = fit(&count, &start);
        let without: Vec<Option<Vec<f64>>> = (0..nb)
            .map(|i| {
                (count[i] > 0.0).then(|| {
                    let mut less = count.clone();
                    less[i] -= 1.0;
                    fit(&less, &all)
                })
            })
            .collect();
        // Nearest occupied bin, ties to the lower; without candidates, all.
        let coef = (0..nb)
            .map(|i| {
                (0..nb).filter(|&j| without[j].is_some()).min_by_key(|&j| (j.abs_diff(i), j)).map_or_else(|| all.clone(), |j| without[j].clone().expect("occupied"))
            })
            .collect();
        Self { coef, centre, scale, z_max, widths: (lo, hi) }
    }

    fn bin_of(z: f64, l0: f64, bin: f64) -> usize {
        (((z.ln() - l0) / bin).max(0.0) as usize).min(BINS - 1)
    }

    /// `P(real | z)`, with `z` held in `[Z_LO, z_max]`, on the fit without
    /// the candidate nearest to `z`.
    pub fn p_real(&self, z: f64) -> f64 {
        let z = z.clamp(Z_LO, self.z_max);
        let (lo, hi) = self.widths;
        let h = 1e-4 * z;
        let l0 = self.centre - self.scale;
        let coef = &self.coef[Self::bin_of(z, l0, 2.0 * self.scale / BINS as f64)];
        // Both densities per unit `log z`.
        let f0 = z * (prefilter::false_rate(z - h, lo, hi) - prefilter::false_rate(z + h, lo, hi)) / (2.0 * h);
        let f1 = horner(coef, (z.ln() - self.centre) / self.scale).exp();
        if f0 + f1 > 0.0 { f1 / (f0 + f1) } else { 1.0 }
    }

    /// The least `u >= Z_LO` from which `P(real | z) >= p` for every higher
    /// `z`, scanned down from `z_max` in steps of 0.01.
    pub fn cut(&self, p: f64) -> f64 {
        let step = 0.01;
        let mut u = self.z_max;
        while u - step >= Z_LO && self.p_real(u - step) >= p {
            u -= step;
        }
        u
    }
}

/// `log f1`'s coefficients on bins at `x` holding `count` candidates, each
/// count Poisson with mean `null + exp(offset + log f1)`: Fisher scoring
/// from `coef`, halved until the loss falls, a ridge holding bins no
/// candidate reaches.
fn fit_bins(x: &[f64], count: &[f64], null: &[f64], offset: f64, mut coef: Vec<f64>) -> Vec<f64> {
    let n = coef.len();
    // Negative Poisson log-likelihood (up to a constant).
    let loss = |coef: &[f64]| -> f64 {
        x.iter().zip(count).zip(null).map(|((&x, &c), &e0)| {
            let mu = e0 + (offset + horner(coef, x)).exp();
            mu - c * mu.ln()
        }).sum()
    };
    let mut chol = Chol::new(n);
    let mut current = loss(&coef);
    for _ in 0..200 {
        let (mut a, mut g) = (vec![0.0; n * n], vec![0.0; n]);
        for ((&x, &c), &e0) in x.iter().zip(count).zip(null) {
            let e1 = (offset + horner(&coef, x)).exp();
            let mu = e0 + e1;
            let phi: Vec<f64> = (0..n).map(|k| x.powi(k as i32)).collect();
            for i in 0..n {
                g[i] += phi[i] * e1 * (c / mu - 1.0);
                for j in 0..n {
                    a[i * n + j] += e1 * e1 / mu * phi[i] * phi[j];
                }
            }
        }
        for i in 0..n {
            a[i * n + i] += 1e-6;
        }
        if !chol.factor(&a, n) {
            break;
        }
        chol.solve_in_place(&mut g);
        let mut t = 1.0;
        let next = loop {
            let trial: Vec<f64> = coef.iter().zip(&g).map(|(c, d)| c + t * d).collect();
            let l = loss(&trial);
            if l <= current || t < 1e-6 {
                break (trial, l);
            }
            t *= 0.5;
        };
        let done = current - next.1 < 1e-10 * (1.0 + current.abs());
        (coef, current) = next;
        if done {
            break;
        }
    }
    coef
}

fn horner(coef: &[f64], x: f64) -> f64 {
    coef.iter().rev().fold(0.0, |acc, c| acc * x + c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Candidates drawn from the null alone keep `P(real)` low and the bar
    /// high; with emitters added, it follows the truth `f1 / (f0 + f1)`,
    /// the prior's emitters counted in `f1`.
    #[test]
    fn the_posterior_separates_signal_from_the_null() {
        let (lo, hi, area) = (1.45, 2.175, 4e7);
        // Null heights by inverting the tail count: a deterministic sample.
        let tail = |z: f64| area * prefilter::false_rate(z, lo, hi);
        let n0 = tail(Z_LO).floor() as usize;
        let mut z: Vec<f64> = (0..n0)
            .map(|k| {
                let target = (k as f64 + 0.5) * tail(Z_LO) / n0 as f64;
                let (mut a, mut b) = (Z_LO, 12.0);
                for _ in 0..60 {
                    let m = 0.5 * (a + b);
                    if tail(m) > target { a = m } else { b = m }
                }
                0.5 * (a + b)
            })
            .collect();
        let null = Posterior::fit(&z, area, lo, hi);
        assert!(null.p_real(3.5) < 0.1 && null.cut(0.9) > 4.8, "{} {}", null.p_real(3.5), null.cut(0.9));
        // 2000 emitters of heights with a triangular density on [3, 11],
        // peaked at 7: smooth, as heights of a spread of fluxes are.
        let (n1, a, m, b) = (2000, 3.0, 7.0, 11.0);
        z.extend((0..n1).map(|k| {
            let q = (k as f64 + 0.5) / n1 as f64;
            if q < 0.5 { a + (q * (b - a) * (m - a)).sqrt() } else { b - ((1.0 - q) * (b - a) * (b - m)).sqrt() }
        }));
        let mixed = Posterior::fit(&z, area, lo, hi);
        let h = 1e-4;
        for zz in [3.2, 4.0, 4.5, 5.0, 6.0, 7.0] {
            let f0 = area * (prefilter::false_rate(zz - h, lo, hi) - prefilter::false_rate(zz + h, lo, hi)) / (2.0 * h);
            let f1 = n1 as f64 * if zz < m { 2.0 * (zz - a) / ((b - a) * (m - a)) } else { 2.0 * (b - zz) / ((b - a) * (b - m)) }
                + PRIOR * area / (std::f64::consts::LN_10 * zz);
            let truth = f1 / (f0 + f1);
            assert!((mixed.p_real(zz) - truth).abs() < 0.05, "z {zz}: {} vs {truth}", mixed.p_real(zz));
        }
    }

    /// Where noise expects well under one candidate, a lone candidate is
    /// judged on the prior alone: it does not raise `f1` at its own height.
    #[test]
    fn a_lone_candidate_is_not_evidence_for_itself() {
        let (lo, hi, area) = (1.45, 2.175, 400.0);
        let post = Posterior::fit(&[3.9], area, lo, hi);
        let prior = Posterior::fit(&[], area, lo, hi);
        let (z, p) = ([3.5, 3.9, 4.3, 5.0, 8.0], [post.p_real(3.9), post.cut(0.9), prior.cut(0.9)]);
        assert!(z.iter().all(|&z| (post.p_real(z) - prior.p_real(z)).abs() < 1e-3) && (p[1] - p[2]).abs() < 0.015, "{p:?}");
    }
}
