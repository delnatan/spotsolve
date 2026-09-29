//! The distributional facts the detector needs.
//!
//! [`crate::detect::dispersion`] reads the pixel variance from the median
//! of a squared fourth difference; [`scale_space_ec`] sets the count
//! threshold from the false emitters pure noise produces.

use std::f64::consts::PI;

/// The median of chi-squared with one degree of freedom, `Phi^-1(0.75)^2`:
/// what the local median of a squared unit-variance Gaussian filter output
/// comes out at. [`crate::detect::dispersion`] divides by it.
pub const CHI2_1_MEDIAN: f64 = 0.454_936_423_119_572_8;

/// Expected Euler characteristic, per pixel, of the excursion above `u` of
/// the matched-filter field of white noise over position and width, widths
/// `s` in `[s1, s2]` (profile standard deviations, px). At high `u` it is
/// the expected number of its local maxima above `u`.
///
/// With `tau = log s`, the field's metric is `(dx^2 + dy^2) / (2 s^2) +
/// dtau^2` (a slab of hyperbolic space between two horospheres), and the
/// Gaussian kinematic formula gives
///
/// ```text
/// EC = 1/4 (1/s1^2 - 1/s2^2) rho3 + 1/4 (1/s1^2 + 1/s2^2) rho2
///    + 1/(8 pi) (1/s1^2 - 1/s2^2) rho1
/// ```
///
/// which for `s1 = s2` is the 2-D field's `rho2 / (2 s^2)`. It counts the
/// maxima of the continuous field; sampling on the pixel lattice finds
/// fewer where `s` is near a pixel.
pub fn scale_space_ec(u: f64, s1: f64, s2: f64) -> f64 {
    let e = (-0.5 * u * u).exp();
    let rho1 = e / (2.0 * PI);
    let rho2 = u * e / (2.0 * PI).powf(1.5);
    let rho3 = (u * u - 1.0) * e / (4.0 * PI * PI);
    let (a, b) = (1.0 / (s1 * s1), 1.0 / (s2 * s2));
    0.25 * (a - b) * rho3 + 0.25 * (a + b) * rho2 + (a - b) / (8.0 * PI) * rho1
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
    fn one_width_is_the_two_dimensional_field() {
        for (u, s) in [(3.0f64, 1.0f64), (4.5, 1.7)] {
            let flat = u * (-0.5 * u * u).exp() / ((2.0 * PI).powf(1.5) * 2.0 * s * s);
            assert!((scale_space_ec(u, s, s) / flat - 1.0).abs() < 1e-12);
        }
        // Searching more widths can only find more maxima.
        assert!(scale_space_ec(4.0, 1.0, 2.0) > scale_space_ec(4.0, 1.0, 1.0));
    }
}
