//! One elongated emitter on a window: a constant level plus a rotated
//! Gaussian, over light the window's other components put there,
//!
//! ```text
//! m = c0 + other + A N(r; mu, S),    S = R diag(a^2, b^2) R^T + I / 12
//! ```
//!
//! fitted, as in [`crate::fit`], at maximum quasi-likelihood under the
//! window's noise curve. The
//! pixel's area enters as its variance, `1/12` per axis, which is how the
//! pixel-integrated isotropic profile widens too. It is the nuisance against
//! which a close pair is tested: an emitter moving within the exposure, or
//! defocused off-axis, is one elongated spot that two round ones tile.
//!
//! `theta = [c0, A, y, x, ln a, ln b, angle]`, fitted by Levenberg-Marquardt
//! on a central-difference Jacobian: the fit runs only on close pairs.

use crate::fit::Window;
use crate::linalg::Chol;

pub const N: usize = 7;

/// A finished fit: parameters, the deviance in nats, and the variances of the
/// estimates (the diagonal of the inverse Fisher information) when the
/// information can be factored.
#[derive(Clone, Debug)]
pub struct Fit {
    pub theta: [f64; N],
    pub divergence: f64,
    pub var: Option<[f64; N]>,
}

/// The ellipse's light alone at each pixel, row-major.
pub fn light(theta: &[f64; N], rows: usize, cols: usize, out: &mut [f64]) {
    let [_, a, y, x, la, lb, th] = *theta;
    let (sa, sb) = ((2.0 * la).exp() + 1.0 / 12.0, (2.0 * lb).exp() + 1.0 / 12.0);
    let (c, s) = (th.cos(), th.sin());
    let norm = a / (2.0 * std::f64::consts::PI * (sa * sb).sqrt());
    for r in 0..rows {
        for q in 0..cols {
            let (dy, dx) = (r as f64 - y, q as f64 - x);
            let (u, w) = (c * dy + s * dx, -s * dy + c * dx);
            out[r * cols + q] = norm * (-0.5 * (u * u / sa + w * w / sb)).exp();
        }
    }
}

fn objective(win: &Window, other: &[f64], theta: &[f64; N], buf: &mut [f64]) -> f64 {
    light(theta, win.rows, win.cols, buf);
    (0..buf.len()).filter(|&i| win.used[i]).map(|i| win.curve.deviance(win.d[i], theta[0] + other[i] + buf[i])).sum()
}

/// The model's Jacobian on the used pixels, `N x used`, by central
/// differences of steps `1e-6` of each parameter's scale.
fn jacobian(win: &Window, theta: &[f64; N], buf: &mut [f64], up: &mut [f64]) -> Vec<f64> {
    let used: Vec<usize> = (0..buf.len()).filter(|&i| win.used[i]).collect();
    let mut jac = vec![0.0; N * used.len()];
    jac[..used.len()].fill(1.0);
    for p in 1..N {
        let h = 1e-6 * theta[p].abs().max(1.0);
        let (mut a, mut b) = (*theta, *theta);
        a[p] += h;
        b[p] -= h;
        light(&a, win.rows, win.cols, up);
        light(&b, win.rows, win.cols, buf);
        for (k, &i) in used.iter().enumerate() {
            jac[p * used.len() + k] = (up[i] - buf[i]) / (2.0 * h);
        }
    }
    jac
}

/// Minimize the deviance from `start` within `[lo, hi]`. The angle is
/// periodic: it is kept in `(-pi/2, pi/2]` and its bounds are not used.
pub fn fit(win: &Window, other: &[f64], start: [f64; N], lo: [f64; N], hi: [f64; N], tol: f64) -> Fit {
    let p = win.rows * win.cols;
    let (mut buf, mut up) = (vec![0.0; p], vec![0.0; p]);
    let clamp = |t: &mut [f64; N]| {
        for q in 0..N - 1 {
            t[q] = t[q].clamp(lo[q], hi[q]);
        }
        let pi = std::f64::consts::PI;
        t[N - 1] -= pi * ((t[N - 1] + 0.5 * pi) / pi).ceil() - pi;
    };
    let mut theta = start;
    clamp(&mut theta);
    let mut cur = objective(win, other, &theta, &mut buf);
    let mut chol = Chol::new(N);
    let mut lam = 1e-3;
    let info = |win: &Window, theta: &[f64; N], buf: &mut [f64], up: &mut [f64]| -> ([f64; N * N], [f64; N]) {
        let jac = jacobian(win, theta, buf, up);
        light(theta, win.rows, win.cols, buf);
        let used: Vec<usize> = (0..buf.len()).filter(|&i| win.used[i]).collect();
        let nu = used.len();
        let (mut f, mut g) = ([0.0; N * N], [0.0; N]);
        for (k, &i) in used.iter().enumerate() {
            let m = theta[0] + other[i] + buf[i];
            let w = 1.0 / win.curve.at(m);
            let res = m - win.d[i];
            for a in 0..N {
                g[a] += jac[a * nu + k] * res * w;
                for b in a..N {
                    f[a * N + b] += jac[a * nu + k] * jac[b * nu + k] * w;
                }
            }
        }
        for a in 0..N {
            for b in 0..a {
                f[a * N + b] = f[b * N + a];
            }
        }
        (f, g)
    };
    for _ in 0..100 {
        let (f, g) = info(win, &theta, &mut buf, &mut up);
        let mut accepted = false;
        while lam < 1e12 {
            let mut a = f;
            for q in 0..N {
                a[q * N + q] += lam * f[q * N + q].max(1e-300);
            }
            if !chol.factor(&a, N) {
                lam *= 10.0;
                continue;
            }
            let mut step: Vec<f64> = g.iter().map(|v| -v).collect();
            chol.solve_in_place(&mut step);
            let mut next = theta;
            for q in 0..N {
                next[q] += step[q];
            }
            clamp(&mut next);
            let i2 = objective(win, other, &next, &mut buf);
            if i2 < cur {
                let gain = cur - i2;
                (theta, cur) = (next, i2);
                lam = (lam / 3.0).max(1e-12);
                accepted = gain > tol;
                break;
            }
            lam *= 4.0;
        }
        if !accepted {
            break;
        }
    }
    let (f, _) = info(win, &theta, &mut buf, &mut up);
    let var = chol.factor(&f, N).then(|| {
        let mut out = vec![0.0; N];
        let mut scratch = Vec::new();
        chol.inv_diag(&mut out, &mut scratch);
        let mut v = [0.0; N];
        v.copy_from_slice(&out);
        v
    });
    Fit { theta, divergence: cur, var }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A noiseless streak is recovered: centre, flux and axes.
    #[test]
    fn a_streak_is_recovered() {
        let (rows, cols) = (21, 23);
        let truth = [5.0, 900.0, 10.3, 11.6, 2.2f64.ln(), 1.2f64.ln(), 0.6];
        let mut m = vec![0.0; rows * cols];
        light(&truth, rows, cols, &mut m);
        let d: Vec<f64> = m.iter().map(|v| v + 5.0).collect();
        let win = Window::new(rows, cols, d, vec![true; rows * cols], crate::prefilter::Curve::POISSON);
        let start = [4.0, 700.0, 10.0, 11.0, 1.5f64.ln(), 1.5f64.ln(), 0.0];
        let (lo, hi) = ([-1e3, 0.0, 0.0, 0.0, -3.0, -3.0, -10.0], [1e3, 1e5, 21.0, 23.0, 3.0, 3.0, 10.0]);
        let f = fit(&win, &vec![0.0; rows * cols], start, lo, hi, 1e-12);
        for (q, (a, b)) in f.theta.iter().zip(&truth).enumerate().take(4) {
            assert!((a - b).abs() < 1e-3 * b.abs().max(1.0), "{q}: {a} vs {b}");
        }
        let (a, b) = (f.theta[4].exp(), f.theta[5].exp());
        let (a, b) = if a > b { (a, b) } else { (b, a) };
        assert!((a / 2.2 - 1.0).abs() < 5e-3 && (b / 1.2 - 1.0).abs() < 5e-3, "{a} {b}");
    }
}
