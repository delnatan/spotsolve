//! The one distributional constant the detector needs.
//!
//! [`crate::boxsearch::dispersion`] reads the pixel variance from the median
//! of a squared fourth difference.

/// The median of chi-squared with one degree of freedom, `Phi^-1(0.75)^2`:
/// what the local median of a squared unit-variance Gaussian filter output
/// comes out at. [`crate::boxsearch::dispersion`] divides by it.
pub const CHI2_1_MEDIAN: f64 = 0.454_936_423_119_572_8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_the_chi2_1_mass_lies_below_its_median() {
        // P(chi2_1 <= m) = erf(sqrt(m / 2)).
        assert!((libm::erf((CHI2_1_MEDIAN / 2.0).sqrt()) - 0.5).abs() < 1e-15);
    }
}
