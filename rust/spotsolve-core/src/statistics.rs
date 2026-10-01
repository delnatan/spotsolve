//! Distributional facts: the median of chi-squared with one degree of
//! freedom, which [`crate::prefilter::dispersion`] divides by, and the
//! standard normal tail and its inverse.

use std::f64::consts::PI;

/// The median of chi-squared with one degree of freedom, `Phi^-1(0.75)^2`:
/// what the local median of a squared unit-variance Gaussian filter output
/// comes out at.
pub const CHI2_1_MEDIAN: f64 = 0.454_936_423_119_572_8;

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
mod tests {
    use super::*;

    #[test]
    fn half_the_chi2_1_mass_lies_below_its_median() {
        // P(chi2_1 <= m) = erf(sqrt(m / 2)).
        assert!((libm::erf((CHI2_1_MEDIAN / 2.0).sqrt()) - 0.5).abs() < 1e-15);
    }

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
