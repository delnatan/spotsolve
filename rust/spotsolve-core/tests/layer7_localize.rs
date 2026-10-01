//! `localize` against truth: simulated fields and pure noise.
//!
//! The detector's contract is statistical: recall, precision and position
//! error on Poisson fields with known emitters, and the false-positive rate
//! `fp_per_mpx` promises on noise. Floors sit a few points under what the
//! detector achieves.

use spotsolve_core::detect::{self as bs, Settings};
use spotsolve_core::psf;

/// Uniform draws from an LCG (deterministic across platforms).
struct Rng(u64);

impl Rng {
    fn uni(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Knuth's product method; fine for the means here (< 700).
    fn poisson(&mut self, lam: f64) -> f64 {
        let l = (-lam).exp();
        let (mut k, mut p) = (0.0, 1.0);
        loop {
            p *= self.uni();
            if p <= l {
                return k;
            }
            k += 1.0;
        }
    }
}

/// A Poisson frame on a flat background of 20 with emitters on a 16 px
/// grid (jittered by up to half a pixel), flux 300-3000 ADU, width `sigma`
/// times `1 +- spread`.
fn grid(rng: &mut Rng, side: usize, sigma: f64, spread: f64) -> (Vec<f64>, Vec<[f64; 2]>) {
    let mut theta = vec![20.0];
    let mut truth = Vec::new();
    let mut y = 8.0;
    while y < side as f64 - 4.0 {
        let mut x = 8.0;
        while x < side as f64 - 4.0 {
            let (yy, xx) = (y + rng.uni() - 0.5, x + rng.uni() - 0.5);
            theta.extend_from_slice(&[300.0 + 2700.0 * rng.uni(), yy, xx, sigma * (1.0 + spread * (2.0 * rng.uni() - 1.0))]);
            truth.push([yy, xx]);
            x += 16.0;
        }
        y += 16.0;
    }
    let mut m = vec![0.0; side * side];
    let ax = psf::local_axis(side);
    psf::model_var_sigma_ax(&theta, &ax, &ax, None, &mut psf::Factors::new(side, side, truth.len()), &mut m);
    (m.iter().map(|&v| rng.poisson(v)).collect(), truth)
}

/// Greedy one-to-one matching within 1 px: `(recall, precision, rms error
/// in px, median error in units of the reported SE)`. The error in SE units
/// is `sqrt((zy^2 + zx^2) / 2)`, whose median is `sqrt(ln 2)` when the SEs
/// are right.
fn score(o: &bs::Output, truth: &[[f64; 2]]) -> (f64, f64, f64, f64) {
    let n = o.amp.len();
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for (t, p) in truth.iter().enumerate() {
        for k in 0..n {
            let d = (o.pos[2 * k] - p[0]).hypot(o.pos[2 * k + 1] - p[1]);
            if d < 1.0 {
                pairs.push((d, t, k));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut used_t, mut used_k) = (vec![false; truth.len()], vec![false; n]);
    let (mut tp, mut se, mut zs) = (0usize, 0.0, Vec::new());
    for (d, t, k) in pairs {
        if !used_t[t] && !used_k[k] {
            used_t[t] = true;
            used_k[k] = true;
            tp += 1;
            se += d * d;
            let (zy, zx) = ((o.pos[2 * k] - truth[t][0]) / o.se[3 * k + 1], (o.pos[2 * k + 1] - truth[t][1]) / o.se[3 * k + 2]);
            zs.push((0.5 * (zy * zy + zx * zx)).sqrt());
        }
    }
    zs.sort_by(f64::total_cmp);
    let med = zs.get(zs.len() / 2).copied().unwrap_or(0.0);
    (tp as f64 / truth.len() as f64, tp as f64 / n.max(1) as f64, (se / tp.max(1) as f64).sqrt(), med)
}

#[test]
fn isolated_emitters_are_recovered_with_calibrated_errors() {
    let (side, sigma) = (128, 1.2);
    let mut rng = Rng(2026);
    // Fixed width at the true width, and free width over a +-20% spread.
    // The median position error is sqrt(ln 2) = 0.83 SE when the SEs are
    // right.
    for (free_sigma, spread) in [(false, 0.0), (true, 0.2)] {
        let s = Settings { free_sigma, ..Settings::new(sigma) };
        let (mut found, mut kept, mut total, mut zs) = (0.0, 0.0, 0.0, Vec::new());
        for _ in 0..6 {
            let (d, truth) = grid(&mut rng, side, sigma, spread);
            let o = bs::localize(&d, side, side, None, &s);
            let (rec, _, _, medz) = score(&o, &truth);
            found += rec * truth.len() as f64;
            kept += o.amp.len() as f64;
            total += truth.len() as f64;
            zs.push(medz);
        }
        let (rec, prec) = (found / total, found / kept);
        zs.sort_by(f64::total_cmp);
        let medz = zs[zs.len() / 2];
        println!("free_sigma {free_sigma}: recall {rec:.4} precision {prec:.4}, median {medz:.2} SE");
        assert!(rec >= 0.995 && prec >= 0.995 && (0.73..0.93).contains(&medz), "free_sigma {free_sigma}: {rec} {prec} {medz}");
    }
}

#[test]
fn pure_noise_meets_the_false_positive_target() {
    let (h, w, sigma, frames) = (256, 256, 1.2, 16);
    let s = Settings::new(sigma);
    let mut rng = Rng(7);
    let mut n = 0;
    for _ in 0..frames {
        let d: Vec<f64> = (0..h * w).map(|_| rng.poisson(20.0)).collect();
        n += bs::localize(&d, h, w, None, &s).amp.len();
    }
    let expected = bs::FP_PER_MPX * (frames * h * w) as f64 / 1e6;
    println!("noise: {n} false emitters, {expected:.1} expected");
    // `fp_per_mpx` bounds the peaks of a continuous Gaussian field. Lattice
    // sampling lowers the count for narrow PSFs; Poisson skew at low counts
    // and frame edges raise it. Hold it within a factor of 2, with 3 sd of
    // Poisson slack.
    let (lo, hi) = (expected / 2.0, 2.0 * expected);
    assert!(n as f64 >= lo - 3.0 * lo.sqrt() && n as f64 <= hi + 3.0 * hi.sqrt(), "{n} vs {expected}");
}
