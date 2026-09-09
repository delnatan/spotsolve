//! Small distribution functions needed by the Aguet significance test.
//!
//! Keeping these here avoids pulling a general statistics stack into the
//! localization runtime. The implementations are the standard Lanczos
//! log-gamma, continued-fraction incomplete beta, and Acklam normal quantile.

use std::f64::consts::PI;

fn refine_normal_quantile(x: f64, p: f64) -> f64 {
    let cdf = 0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2));
    x - (cdf - p) / (-0.5 * x * x).exp().mul_add(1.0 / (2.0 * PI).sqrt(), 0.0)
}

fn log_gamma(z: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        return (PI / (PI * z).sin()).ln() - log_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = C[0];
    for (i, coefficient) in C.iter().enumerate().skip(1) {
        x += coefficient / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3e-14;
    const TINY: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let m2 = 2 * m;
        let mf = m as f64;
        let m2f = m2 as f64;
        let mut aa = mf * (b - mf) * x / ((qam + m2f) * (a + m2f));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + mf) * (qab + mf) * x / ((a + m2f) * (qap + m2f));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front =
        (log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * x.ln() + b * (-x).ln_1p()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_fraction(b, a, 1.0 - x) / b
    }
}

/// Upper-tail probability for Student's t distribution.
pub fn student_t_sf(t: f64, degrees_of_freedom: f64) -> f64 {
    if !t.is_finite() || !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
        return if t == f64::INFINITY { 0.0 } else { 1.0 };
    }
    let x = degrees_of_freedom / (degrees_of_freedom + t * t);
    let half = 0.5 * regularized_beta(x, 0.5 * degrees_of_freedom, 0.5);
    if t >= 0.0 { half } else { 1.0 - half }
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
        assert!((student_t_sf(2.0, 10.0) - 0.036_694_017_385_370_2).abs() < 2e-14);
        assert!((student_t_sf(-1.5, 37.5) - 0.929_006_322_820_826_8).abs() < 2e-14);
    }
}
