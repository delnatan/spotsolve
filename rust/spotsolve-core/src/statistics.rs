//! The standard-normal quantile FIND's seed threshold needs: Acklam's rational
//! approximation with one Newton refinement, which keeps a general statistics
//! stack out of the localization runtime.

use std::f64::consts::PI;

fn refine_normal_quantile(x: f64, p: f64) -> f64 {
    let cdf = 0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2));
    x - (cdf - p) / (-0.5 * x * x).exp().mul_add(1.0 / (2.0 * PI).sqrt(), 0.0)
}

/// Standard-normal quantile. Valid for `0 < p < 1`.
pub fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    if !(0.0 < p && p < 1.0) {
        return if p == 0.0 {
            f64::NEG_INFINITY
        } else if p == 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        let x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
        return refine_normal_quantile(x, p);
    }
    if p > 1.0 - 0.02425 {
        let q = (-2.0 * (-p).ln_1p()).sqrt();
        let x = -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
        return refine_normal_quantile(x, p);
    }
    let q = p - 0.5;
    let r = q * q;
    let x = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    refine_normal_quantile(x, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_values_match_reference_values() {
        assert!((normal_quantile(0.975) - 1.959_963_984_540_054).abs() < 2e-9);
        // scipy.special.ndtri(1.2e-5), the tail FIND's Bonferroni cut lives in.
        assert!((normal_quantile(1.2e-5) + 4.224_003_799_672_023).abs() < 1e-8);
    }
}
