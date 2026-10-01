//! LAYER 3: the window fit ([`spotsolve_core::fit`]).
//!
//! Derivatives against finite differences; on simulated Poisson windows,
//! position and flux errors against the reported variances; bounds, masked
//! pixels and two-component recovery, at one width and at each its own.

use spotsolve_core::fit::{Fitter, Layout, Window};

/// Deterministic LCG; Knuth's Poisson method is fine for the means here.
struct Rng(u64);

impl Rng {
    fn uni(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

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

const SIGMA: f64 = 1.45;
/// `ceil(4 sigma)`: u-track's window half-width.
const HALF: usize = 6;
const SIDE: usize = 2 * HALF + 1;

fn clean(rows: usize, cols: usize, theta: &[f64], lay: Layout) -> Vec<f64> {
    let w = Window::new(rows, cols, vec![0.0; rows * cols], vec![true; rows * cols]);
    Fitter::default().model(&w, theta, lay).to_vec()
}

fn bounds(lay: Layout, centre: f64, reach: f64) -> (Vec<f64>, Vec<f64>) {
    let (mut lo, mut hi) = (vec![1e-3], vec![1e5]);
    for _ in 0..lay.k {
        lo.extend([0.0, centre - reach, centre - reach]);
        hi.extend([1e6, centre + reach, centre + reach]);
        if lay.sigma.is_none() {
            lo.push(0.5 * SIGMA);
            hi.push(3.0 * SIGMA);
        }
    }
    (lo, hi)
}

#[test]
fn derivatives_match_finite_differences() {
    let (rows, cols) = (11, 13);
    let lay = Layout { k: 2, sigma: None };
    let theta = [17.0, 420.0, 4.3, 5.1, 1.6, 260.0, 6.2, 8.4, 1.3];
    let mut rng = Rng(7);
    let m0 = clean(rows, cols, &theta, lay);
    let d: Vec<f64> = m0.iter().map(|&m| rng.poisson(m) - 2.0).collect();
    let used: Vec<bool> = (0..rows * cols).map(|q| q % 7 != 3).collect();
    let w = Window::new(rows, cols, d, used.clone());
    let mut f = Fitter::default();
    let lin = f.linearize(&w, &theta, lay);
    let (g, info) = (&lin.g, &lin.f);
    let n = lay.n();
    let mut dm = Vec::new();
    for a in 0..n {
        let h = 1e-6 * theta[a].abs().max(1.0);
        let (mut p, mut q) = (theta, theta);
        p[a] += h;
        q[a] -= h;
        let fd = (f.divergence(&w, &p, lay) - f.divergence(&w, &q, lay)) / (2.0 * h);
        assert!((g[a] - fd).abs() <= 1e-6 * (1.0 + fd.abs()), "gradient {a}: {} vs {fd}", g[a]);
        let (mp, mq) = (clean(rows, cols, &p, lay), clean(rows, cols, &q, lay));
        dm.push(mp.iter().zip(&mq).map(|(x, y)| (x - y) / (2.0 * h)).collect::<Vec<_>>());
        for q in 0..rows * cols {
            let want = if used[q] { dm[a][q] } else { 0.0 };
            let got = lin.jac[a * rows * cols + q];
            assert!((got - want).abs() <= 1e-5 * (1.0 + want.abs()), "jacobian {a} at {q}: {got} vs {want}");
        }
    }
    for a in 0..n {
        for b in 0..n {
            let want: f64 = (0..rows * cols).filter(|&q| used[q]).map(|q| dm[a][q] * dm[b][q] / m0[q]).sum();
            let got = info[a * n + b];
            assert!((got - want).abs() <= 1e-5 * (1.0 + want.abs()), "information {a},{b}: {got} vs {want}");
        }
    }
}

/// Fits of one emitter on Poisson windows: the errors' spread in units of
/// the reported standard errors, per parameter.
fn error_spread(lay: Layout, trials: usize) -> Vec<(f64, f64)> {
    let (flux, bg, c) = (800.0, 20.0, HALF as f64);
    let mut rng = Rng(11);
    let mut f = Fitter::default();
    let n = lay.n();
    let mut z = vec![Vec::new(); n];
    for _ in 0..trials {
        let (y, x) = (c + rng.uni() - 0.5, c + rng.uni() - 0.5);
        let mut truth = vec![bg, flux, y, x];
        if lay.sigma.is_none() {
            truth.push(SIGMA);
        }
        let d: Vec<f64> = clean(SIDE, SIDE, &truth, lay).iter().map(|&m| rng.poisson(m)).collect();
        let w = Window::new(SIDE, SIDE, d, vec![true; SIDE * SIDE]);
        let mut start = vec![15.0, 500.0, c, c];
        if lay.sigma.is_none() {
            start.push(SIGMA);
        }
        let (lo, hi) = bounds(lay, c, 2.0);
        let fit = f.fit(&w, &start, lay, &lo, &hi, 100, 1e-6);
        assert!(fit.converged && !fit.stalled, "fit did not converge: {fit:?}");
        let var = f.variances(&w, &fit.theta, lay, 1.0).expect("variances");
        for q in 0..n {
            z[q].push((fit.theta[q] - truth[q]) / var[q].sqrt());
        }
    }
    z.iter()
        .map(|v| {
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (v.len() - 1) as f64;
            (mean, var)
        })
        .collect()
}

#[test]
fn errors_match_the_reported_covariance() {
    // 2000 trials: the sample variance of a unit normal has sd 0.03, its
    // mean 0.022.
    for lay in [Layout { k: 1, sigma: Some(SIGMA) }, Layout { k: 1, sigma: None }] {
        for (q, (mean, var)) in error_spread(lay, 2000).into_iter().enumerate() {
            // The width's maximum-likelihood estimate is biased low at finite
            // SNR, by a fraction of its standard error.
            assert!(mean.abs() < 0.15, "{lay:?} parameter {q}: mean z {mean}");
            assert!((0.88..1.12).contains(&var), "{lay:?} parameter {q}: var z {var}");
        }
    }
}

#[test]
fn bounds_hold_and_are_reported() {
    let lay = Layout { k: 1, sigma: Some(SIGMA) };
    let c = HALF as f64;
    let (lo, hi) = bounds(lay, c, 2.0);
    let mut f = Fitter::default();
    // An emitter outside the position box: x stops on its bound.
    let d = clean(SIDE, SIDE, &[20.0, 800.0, c, c + 3.5], lay);
    let fit = f.fit(&Window::new(SIDE, SIDE, d, vec![true; SIDE * SIDE]), &[20.0, 500.0, c, c], lay, &lo, &hi, 100, 1e-6);
    assert_eq!(fit.at_bound, [false, false, false, true]);
    assert!(fit.theta.iter().zip(lo.iter().zip(&hi)).all(|(t, (l, h))| l < t && t < h));
    // Flat background: the flux falls far below its standard error and the
    // level is found.
    let w = Window::new(SIDE, SIDE, vec![20.0; SIDE * SIDE], vec![true; SIDE * SIDE]);
    let fit = f.fit(&w, &[15.0, 500.0, c, c], lay, &lo, &hi, 100, 1e-6);
    let var = f.variances(&w, &fit.theta, lay, 1.0).unwrap();
    assert!(fit.converged && fit.theta[1] < 0.1 * var[1].sqrt(), "{fit:?}");
    assert!((fit.theta[0] - 20.0).abs() < 0.1 * var[0].sqrt(), "{fit:?}");
}

#[test]
fn unused_pixels_do_not_enter() {
    let lay = Layout { k: 1, sigma: None };
    let c = HALF as f64;
    let mut rng = Rng(3);
    let d: Vec<f64> = clean(SIDE, SIDE, &[20.0, 600.0, c + 0.3, c - 0.2, SIGMA], lay).iter().map(|&m| rng.poisson(m)).collect();
    let used: Vec<bool> = (0..SIDE * SIDE).map(|q| q / SIDE > 9 && q % SIDE > 9).map(|v| !v).collect();
    let mut other = d.clone();
    for (o, &u) in other.iter_mut().zip(&used) {
        if !u {
            *o = 5e4;
        }
    }
    let (lo, hi) = bounds(lay, c, 2.0);
    let start = [15.0, 500.0, c, c, SIGMA];
    let mut f = Fitter::default();
    let a = f.fit(&Window::new(SIDE, SIDE, d, used.clone()), &start, lay, &lo, &hi, 100, 1e-6);
    let b = f.fit(&Window::new(SIDE, SIDE, other, used), &start, lay, &lo, &hi, 100, 1e-6);
    assert_eq!(a.theta, b.theta);
    assert_eq!(a.divergence, b.divergence);
}

#[test]
fn two_components_recover_a_close_pair() {
    let c = HALF as f64;
    let sep = 2.0 * SIGMA;
    for lay in [Layout { k: 2, sigma: Some(SIGMA) }, Layout { k: 2, sigma: None }] {
    let (truth, start) = if lay.sigma.is_some() {
        (vec![20.0, 700.0, c - 0.5 * sep, c + 0.2, 500.0, c + 0.5 * sep, c - 0.1],
         vec![15.0, 400.0, c - 0.7, c + 0.6, 400.0, c + 1.5, c - 0.6])
    } else {
        (vec![20.0, 700.0, c - 0.5 * sep, c + 0.2, 1.3, 500.0, c + 0.5 * sep, c - 0.1, 1.7],
         vec![15.0, 400.0, c - 0.7, c + 0.6, SIGMA, 400.0, c + 1.5, c - 0.6, SIGMA])
    };
    let d = clean(SIDE, SIDE, &truth, lay);
    let (lo, hi) = bounds(lay, c, 3.0);
    let fit = Fitter::default().fit(&Window::new(SIDE, SIDE, d, vec![true; SIDE * SIDE]), &start, lay, &lo, &hi, 100, 1e-12);
    assert!(fit.converged, "{fit:?}");
    for (got, want) in fit.theta.iter().zip(&truth) {
        assert!((got - want).abs() < 1e-4 * want.abs().max(1.0), "{:?} vs {truth:?}", fit.theta);
    }
    }
}
