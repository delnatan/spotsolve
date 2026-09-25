//! `localize` against truth: simulated fields and pure noise.
//!
//! The Rust detector is the reference (the Python prototypes' parity
//! fixtures were retired on 2026-09-24), so its contract is statistical:
//! recall, precision and position error on Poisson fields with known
//! emitters, and the false-positive rate `fp_per_mpx` promises on noise.
//! Floors sit a few points under the values measured when they were set.

use spotsolve_core::boxsearch::{self as bs, Settings, Workspace};
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

/// A Poisson frame on a flat background of 20 with `n` emitters at least
/// 4 px from the edge, amplitude 150-3000 ADU, width `sigma` +- 20%.
fn field(rng: &mut Rng, h: usize, w: usize, n: usize, sigma: f64) -> (Vec<f64>, Vec<[f64; 2]>) {
    let mut theta = vec![20.0];
    let mut truth = Vec::with_capacity(n);
    for _ in 0..n {
        let y = 4.0 + rng.uni() * (h as f64 - 9.0);
        let x = 4.0 + rng.uni() * (w as f64 - 9.0);
        theta.extend_from_slice(&[150.0 + 2850.0 * rng.uni(), y, x, sigma * (0.8 + 0.4 * rng.uni())]);
        truth.push([y, x]);
    }
    let mut m = vec![0.0; h * w];
    psf::model_var_sigma_ax(&theta, &psf::local_axis(h), &psf::local_axis(w), None, &mut psf::Factors::new(h, w, n), &mut m);
    (m.iter().map(|&v| rng.poisson(v)).collect(), truth)
}

/// Greedy one-to-one matching within 1 px: `(recall, precision, rmse)`.
fn score(o: &bs::Output, truth: &[[f64; 2]]) -> (f64, f64, f64) {
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
    let (mut tp, mut se) = (0usize, 0.0);
    for (d, t, k) in pairs {
        if !used_t[t] && !used_k[k] {
            used_t[t] = true;
            used_k[k] = true;
            tp += 1;
            se += d * d;
        }
    }
    (tp as f64 / truth.len() as f64, tp as f64 / n.max(1) as f64, (se / tp.max(1) as f64).sqrt())
}

#[test]
fn simulated_fields_are_recovered() {
    let (h, w, sigma) = (128, 128, 1.2);
    let s = Settings { sigma, fp_per_mpx: bs::FP_PER_MPX, slack: bs::SLACK };
    let mut ws = Workspace::new();
    let mut rng = Rng(2026);
    // (emitters per px, recall, precision, rmse px) floors and ceiling;
    // measured 1.000/1.000/0.116, 0.890/0.991/0.200, 0.868/0.993/0.247.
    for (density, rec_min, prec_min, rmse_max) in [(0.005, 0.97, 0.98, 0.16), (0.015, 0.86, 0.97, 0.25), (0.03, 0.84, 0.97, 0.30)] {
        let n = (density * (h * w) as f64).round() as usize;
        let (d, truth) = field(&mut rng, h, w, n, sigma);
        let o = bs::localize(&d, h, w, None, &s, &mut ws);
        let (rec, prec, rmse) = score(&o, &truth);
        println!("density {density}: N {} of {n}, recall {rec:.3} precision {prec:.3} rmse {rmse:.3}", o.amp.len());
        assert!(rec >= rec_min && prec >= prec_min && rmse <= rmse_max, "density {density}: {rec} {prec} {rmse}");
    }
}

#[test]
fn pure_noise_meets_the_false_positive_target() {
    let (h, w, sigma, frames) = (256, 256, 1.2, 16);
    let s = Settings { sigma, fp_per_mpx: bs::FP_PER_MPX, slack: bs::SLACK };
    let mut ws = Workspace::new();
    let mut rng = Rng(7);
    let mut n = 0;
    for _ in 0..frames {
        let d: Vec<f64> = (0..h * w).map(|_| rng.poisson(20.0)).collect();
        n += bs::localize(&d, h, w, None, &s, &mut ws).amp.len();
    }
    let expected = bs::FP_PER_MPX * (frames * h * w) as f64 / 1e6;
    println!("noise: {n} false emitters, {expected:.1} expected");
    // Counts per frame are Poisson; over 67 Mpx at sigma 1.2 the rate was
    // 16.4 per Mpx. Here 23: within 3 sd.
    assert!((n as f64 - expected).abs() <= 3.0 * expected.sqrt(), "{n} vs {expected}");
}
