//! The distributional facts the detector needs.
//!
//! [`crate::detect::dispersion`] reads the pixel variance from the median
//! of a squared fourth difference; [`lkc`] and [`expected_ec`] set the
//! count threshold from the false emitters pure noise produces.

use std::f64::consts::PI;

/// The median of chi-squared with one degree of freedom, `Phi^-1(0.75)^2`:
/// what the local median of a squared unit-variance Gaussian filter output
/// comes out at. [`crate::detect::dispersion`] divides by it.
pub const CHI2_1_MEDIAN: f64 = 0.454_936_423_119_572_8;

/// Lipschitz-Killing curvatures, per pixel, of the space one emitter's
/// likelihood ratio is maximized over: position, and log width over
/// `[lo, hi]`. [`expected_ec`] turns them into a count.
#[derive(Clone, Copy, Debug)]
pub struct Lkc {
    pub l1: f64,
    pub l2: f64,
    pub l3: f64,
}

/// Width samples of [`lkc`]'s integrals over log width.
const LKC_WIDTHS: usize = 17;
/// Centres per axis over a node cell (or a pixel) that [`lkc`] averages.
const LKC_OFFSETS: usize = 8;
/// Widths of profile, and node spacings, a 1-D window holds each side.
const LKC_REACH: (f64, f64) = (6.0, 3.0);

/// [`Lkc`] of the likelihood-ratio field of one emitter of width `s` in
/// `[lo, hi]` (the fit's width bounds) against white noise, with a bilinear
/// background on nodes `tile` px apart profiled out (`None`: a known
/// background).
///
/// The field is `<h, r> / |h|` with `h = g - P g`: the pixel-integrated
/// profile less its projection on the background. Its metric over
/// `(y, x, tau = log s)` is the covariance of the unit field's derivatives,
/// computed from 1-D sums since `g` and the tents separate. Averaged over a
/// node cell, it is `f(tau)^2 (dy^2 + dx^2) + L_tt(tau) dtau^2`, and the
/// Gaussian kinematic formula for that warped slab gives
///
/// ```text
/// L3 = int sqrt(det Lambda) dtau
/// L2 = (f(lo)^2 + f(hi)^2) / 2
/// L1 = 1/(2 pi) int (df/dtau)^2 / sqrt(L_tt) dtau
/// ```
///
/// With no background and `s` well above a pixel, `f^2 = 1/(2 s^2)` and
/// `L_tt = 1`: the slab of hyperbolic space of the continuous Gaussian scale
/// space. Profiling out the nodes shortens `|h|` more than its derivatives,
/// so the field varies faster and has more maxima.
pub fn lkc(lo: f64, hi: f64, tile: Option<usize>) -> Lkc {
    let nt = if hi > lo { LKC_WIDTHS } else { 1 };
    let taus: Vec<f64> = (0..nt).map(|j| lo.ln() + (hi / lo).ln() * j as f64 / (nt - 1).max(1) as f64).collect();
    let period = tile.map_or(1.0, |t| t as f64);
    let offs: Vec<f64> = (0..LKC_OFFSETS).map(|k| (k as f64 + 0.5) / LKC_OFFSETS as f64 * period).collect();
    let (mut vol, mut face, mut ltt) = (vec![0.0; nt], vec![0.0; nt], vec![0.0; nt]);
    for (j, &tau) in taus.iter().enumerate() {
        let s = tau.exp();
        let axes: Vec<AxisGram> = offs.iter().map(|&c| AxisGram::new(c, s, tile)).collect();
        for gy in &axes {
            for gx in &axes {
                let lam = gy.metric(gx);
                let det2 = lam[0][0] * lam[1][1] - lam[0][1] * lam[1][0];
                let det3 = lam[0][0] * (lam[1][1] * lam[2][2] - lam[1][2] * lam[2][1])
                    - lam[0][1] * (lam[1][0] * lam[2][2] - lam[1][2] * lam[2][0])
                    + lam[0][2] * (lam[1][0] * lam[2][1] - lam[1][1] * lam[2][0]);
                vol[j] += det3.max(0.0).sqrt();
                face[j] += det2.max(0.0).sqrt();
                ltt[j] += lam[2][2];
            }
        }
        let n = (axes.len() * axes.len()) as f64;
        vol[j] /= n;
        face[j] /= n;
        ltt[j] /= n;
    }
    let l2 = 0.5 * (face[0] + face[nt - 1]);
    if nt == 1 {
        return Lkc { l1: 0.0, l2, l3: 0.0 };
    }
    let dt = taus[1] - taus[0];
    let f: Vec<f64> = face.iter().map(|v| v.sqrt()).collect();
    let df = |j: usize| {
        let (a, b) = (j.saturating_sub(1), (j + 1).min(nt - 1));
        (f[b] - f[a]) / ((b - a) as f64 * dt)
    };
    let trap = |v: &dyn Fn(usize) -> f64| dt * ((0..nt).map(v).sum::<f64>() - 0.5 * (v(0) + v(nt - 1)));
    Lkc { l1: trap(&|j| df(j).powi(2) / ltt[j].sqrt()) / (2.0 * PI), l2, l3: trap(&|j| vol[j]) }
}

/// Expected Euler characteristic, per pixel, of the excursion above `u` of
/// a unit Gaussian field with curvatures `l`. At high `u` it is the
/// expected number of local maxima above `u`.
pub fn expected_ec(u: f64, l: &Lkc) -> f64 {
    let e = (-0.5 * u * u).exp();
    let rho1 = e / (2.0 * PI);
    let rho2 = u * e / (2.0 * PI).powf(1.5);
    let rho3 = (u * u - 1.0) * e / (4.0 * PI * PI);
    l.l3 * rho3 + l.l2 * rho2 + l.l1 * rho1
}

/// Along one axis, for a profile centred at `c` of width `s`: the Gram
/// matrices of `v = (E, dE/dc, s dE/ds)` plain (`g`) and through the
/// projection on the tents (`p`).
struct AxisGram {
    g: [[f64; 3]; 3],
    p: [[f64; 3]; 3],
}

impl AxisGram {
    fn new(c: f64, s: f64, tile: Option<usize>) -> Self {
        let reach = LKC_REACH.0 * s + tile.map_or(0.0, |t| LKC_REACH.1 * t as f64);
        let (t0, t1) = ((c - reach).floor() as i64, (c + reach).ceil() as i64);
        let ax: Vec<f64> = (t0..=t1).map(|v| v as f64).collect();
        let n = ax.len();
        let (mut e, mut de, mut ds) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        crate::psf::factors_axis_sigma(&ax, &[c], s, &mut e, &mut de, &mut ds);
        let et: Vec<f64> = ds.iter().map(|v| v * s).collect();
        let v = [e, de, et];
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let g = std::array::from_fn(|i| std::array::from_fn(|j| dot(&v[i], &v[j])));
        let Some(tile) = tile else { return Self { g, p: [[0.0; 3]; 3] } };
        // Tents at multiples of `tile` that reach the window.
        let tl = tile as f64;
        let (j0, j1) = ((ax[0] / tl).floor() as i64, (ax[n - 1] / tl).ceil() as i64);
        let tent = |j: i64, x: f64| (1.0 - (x - j as f64 * tl).abs() / tl).max(0.0);
        let nodes: Vec<i64> = (j0..=j1).collect();
        let k = nodes.len();
        let mut m = vec![0.0; k * k];
        for (a, &ja) in nodes.iter().enumerate() {
            for (b, &jb) in nodes.iter().enumerate() {
                m[a * k + b] = ax.iter().map(|&x| tent(ja, x) * tent(jb, x)).sum();
            }
        }
        let mut chol = crate::linalg::Chol::new(k);
        assert!(chol.factor(&m, k), "tent Gram is positive definite");
        let tv: Vec<Vec<f64>> = v.iter().map(|vi| nodes.iter().map(|&j| dot(vi, &ax.iter().map(|&x| tent(j, x)).collect::<Vec<_>>())).collect()).collect();
        let mut x = vec![0.0; k];
        let mut p = [[0.0; 3]; 3];
        for j in 0..3 {
            chol.solve(&tv[j], &mut x);
            for i in 0..3 {
                p[i][j] = dot(&tv[i], &x);
            }
        }
        Self { g, p }
    }

    /// Covariance of the unit field's derivatives by `(y, x, tau)`, with
    /// `self` the y axis and `gx` the x axis. The 2-D vectors are sums of
    /// outer products `a (x) b`, and `<a (x) b, (I - P) (c (x) d)> =
    /// g_y[a][c] g_x[b][d] - p_y[a][c] p_x[b][d]`.
    fn metric(&self, gx: &AxisGram) -> [[f64; 3]; 3] {
        let k = |(a, b): (usize, usize), (c, d): (usize, usize)| self.g[a][c] * gx.g[b][d] - self.p[a][c] * gx.p[b][d];
        // h, d/dy, d/dx, d/dtau as sums of (y factor, x factor).
        let dirs: [&[(usize, usize)]; 4] = [&[(0, 0)], &[(1, 0)], &[(0, 1)], &[(2, 0), (0, 2)]];
        let ip = |i: usize, j: usize| dirs[i].iter().map(|&a| dirs[j].iter().map(|&b| k(a, b)).sum::<f64>()).sum::<f64>();
        let nn = ip(0, 0);
        std::array::from_fn(|i| std::array::from_fn(|j| ip(i + 1, j + 1) / nn - ip(i + 1, 0) * ip(j + 1, 0) / (nn * nn)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_the_chi2_1_mass_lies_below_its_median() {
        // P(chi2_1 <= m) = erf(sqrt(m / 2)).
        assert!((libm::erf((CHI2_1_MEDIAN / 2.0).sqrt()) - 0.5).abs() < 1e-15);
    }

    /// The continuous Gaussian scale space over `[s1, s2]` (profile
    /// standard deviations): a slab of hyperbolic space.
    fn hyperbolic_ec(u: f64, s1: f64, s2: f64) -> f64 {
        let (a, b) = (1.0 / (s1 * s1), 1.0 / (s2 * s2));
        let l = Lkc { l3: 0.25 * (a - b), l2: 0.25 * (a + b), l1: (a - b) / (8.0 * PI) };
        expected_ec(u, &l)
    }

    #[test]
    fn with_a_known_background_the_curvatures_are_the_gaussian_scale_space() {
        let sd = |s: f64| (s * s + 1.0 / 12.0).sqrt();
        for (lo, hi) in [(1.5, 1.5), (1.5, 3.3), (2.0, 4.4)] {
            let l = lkc(lo, hi, None);
            for u in [3.5, 4.5] {
                let r = expected_ec(u, &l) / hyperbolic_ec(u, sd(lo), sd(hi));
                assert!((r - 1.0).abs() < 2e-3, "lo {lo} hi {hi} u {u}: ratio {r}");
            }
        }
    }

    #[test]
    fn profiling_out_the_background_adds_maxima() {
        let known = expected_ec(4.0, &lkc(2.0, 4.4, None));
        let tiles = |t: usize| expected_ec(4.0, &lkc(2.0, 4.4, Some(t)));
        // Finer nodes take more of the profile, and the field left has
        // more maxima; nodes far coarser than the profile take nothing.
        assert!(tiles(36) > known && tiles(18) > tiles(36) && tiles(400) / known < 1.01);
    }
}

/// Upper tail of the standard normal, `P(Z > z)`.
pub fn norm_sf(z: f64) -> f64 {
    0.5 * libm::erfc(z / std::f64::consts::SQRT_2)
}

/// Inverse of [`norm_sf`]: the `z` with `P(Z > z) = q`, for `0 < q < 1`.
/// Acklam's rational approximation (relative error 1.2e-9), then two Halley
/// steps on `erfc`, which leave it at the precision `erfc` has.
pub fn norm_isf(q: f64) -> f64 {
    assert!(q > 0.0 && q < 1.0, "norm_isf needs 0 < q < 1, got {q}");
    const A: [f64; 6] = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    const B: [f64; 5] = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    // Lower quantile x of p = q, by symmetry z = -x.
    let p = q;
    let tail = |p: f64| {
        let t = (-2.0 * p.ln()).sqrt();
        (((((C[0] * t + C[1]) * t + C[2]) * t + C[3]) * t + C[4]) * t + C[5]) / ((((D[0] * t + D[1]) * t + D[2]) * t + D[3]) * t + 1.0)
    };
    let mut x = if p < 0.02425 {
        tail(p)
    } else if p > 1.0 - 0.02425 {
        -tail(1.0 - p)
    } else {
        let s = p - 0.5;
        let r = s * s;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * s / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    for _ in 0..2 {
        // Halley on Phi(x) - p, Phi(x) = P(Z > -x).
        let e = norm_sf(-x) - p;
        let u = e * (2.0 * PI).sqrt() * (0.5 * x * x).exp();
        x -= u / (1.0 + 0.5 * x * u);
    }
    -x
}

#[cfg(test)]
mod quantile_tests {
    use super::*;

    #[test]
    fn normal_quantiles_match_reference_values() {
        // scipy.stats.norm.isf
        for (q, z) in [
            (0.5, 0.0),
            (0.05, 1.6448536269514722),
            (0.025, 1.959963984540054),
            (1e-3, 3.090232306167813),
            (1e-10, 6.361340902404056),
            (0.9, -1.2815515655446004),
        ] {
            assert!((norm_isf(q) - z).abs() < 1e-12, "{q}: {} vs {z}", norm_isf(q));
            assert!((norm_sf(z) - q).abs() < 1e-14 * (1.0 + 1.0 / q), "{z}");
        }
    }
}
