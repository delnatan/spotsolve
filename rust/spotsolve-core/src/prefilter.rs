//! Where to fit: u-track's significance mask, LoG seeds and mask labels.
//!
//! 1. **Score.** At every pixel, regress the window `[-R, R]^2`, `R =
//!    ceil(4 sigma)`, on one pixel-integrated PSF `g` and a constant:
//!    `A = <g - gbar, d> / |g - gbar|^2`. The window stops at the frame's
//!    edge; every sum is over the pixels inside it. The score test of
//!    `A = 0` under Poisson noise of dispersion `phi`, the level profiled,
//!    is
//!
//!    ```text
//!    z = A |g - gbar| / sqrt(phi c0),    c0 = mean of d over the window
//!    ```
//!
//!    u-track instead takes the noise from the window's residual and tests
//!    `A > k sigma_res` with a Welch t. A neighbour inside the window
//!    inflates that residual and hides the spot in crowded areas; here the
//!    noise comes from the Poisson model and the frame's measured `phi`.
//! 2. **Mask.** Pixels with `z >= z_min`.
//! 3. **Seeds.** Local maxima of the negative Laplacian of Gaussian over a
//!    `2 ceil(sigma) + 1` square, inside the mask. u-track's RefineMaskLoG
//!    is left out: it admitted maxima the residual-based test had hidden,
//!    which the score above does not hide, and on noise it multiplies the
//!    seeds.
//! 4. **Labels.** 8-connected components of the mask. A seed's fit leaves
//!    out the pixels of other components.
//!
//! [`threshold`] sets the score a false emitter must reach from the expected
//! false emitters per 10^6 pixels of noise.

use crate::filters::{self, Mode};
use crate::psf;
use crate::statistics;
use std::f64::consts::PI;

/// The result of screening a frame, all row-major `h x w`.
#[derive(Clone, Debug)]
pub struct Screen {
    /// Score of one emitter at each pixel.
    pub z: Vec<f64>,
    /// Regression flux and level at each pixel.
    pub amplitude: Vec<f64>,
    pub background: Vec<f64>,
    pub mask: Vec<bool>,
    /// Component of each mask pixel, from 1; 0 outside the mask.
    pub labels: Vec<u32>,
    /// Seed pixels (`y * w + x`), ascending.
    pub seeds: Vec<usize>,
}

/// Lowest mean a score's variance is taken at, in ADU.
pub const LEVEL_FLOOR: f64 = 1e-3;

/// The centred pixel-integrated unit-flux PSF along one axis, radius
/// `ceil(4 sigma)`; the 2-D kernel is its outer product.
pub fn psf_kernel1d(sigma: f64) -> Vec<f64> {
    let r = (4.0 * sigma).ceil() as isize;
    let t: Vec<f64> = (-r..=r).map(|v| v as f64).collect();
    let mut k = vec![0.0; t.len()];
    psf::shape_axis(&t, &[0.0], sigma, &mut k);
    k
}

/// Correlation of `src` with the separable symmetric kernel `k (x) k`, zero
/// outside the frame.
pub(crate) fn correlate(src: &[f64], h: usize, w: usize, k: &[f64]) -> Vec<f64> {
    let r = k.len() / 2;
    let mut mid = vec![0.0; h * w];
    for y in 0..h {
        let (t0, t1) = taps(y, r, h);
        for t in t0..t1 {
            let (row, kv) = ((y + t - r) * w, k[t]);
            for (m, s) in mid[y * w..(y + 1) * w].iter_mut().zip(&src[row..row + w]) {
                *m += kv * s;
            }
        }
    }
    let mut out = vec![0.0; h * w];
    for y in 0..h {
        let row = &mid[y * w..(y + 1) * w];
        for x in 0..w {
            let (t0, t1) = taps(x, r, w);
            out[y * w + x] = (t0..t1).map(|t| k[t] * row[x + t - r]).sum();
        }
    }
    out
}

/// Taps `t` of a radius-`r` kernel whose sample `i + t - r` lies in `0..n`.
fn taps(i: usize, r: usize, n: usize) -> (usize, usize) {
    (r.saturating_sub(i), (n + r - i).min(2 * r + 1))
}

/// Along an axis of `n` pixels, for the kernel centred on each pixel and cut
/// at the edges: the sums of `k`, of `k^2`, and the count of taps inside.
fn axis_sums(k: &[f64], n: usize) -> Vec<[f64; 3]> {
    let r = k.len() / 2;
    (0..n)
        .map(|i| {
            let (t0, t1) = taps(i, r, n);
            let s = &k[t0..t1];
            [s.iter().sum(), s.iter().map(|v| v * v).sum(), (t1 - t0) as f64]
        })
        .collect()
}

/// Score, flux and level of one emitter of width `sigma` at every pixel.
pub fn score(d: &[f64], h: usize, w: usize, sigma: f64, phi: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let k = psf_kernel1d(sigma);
    let fg = correlate(d, h, w, &k);
    let fu = correlate(d, h, w, &vec![1.0; k.len()]);
    let (sy, sx) = (axis_sums(&k, h), axis_sums(&k, w));
    let mut z = vec![0.0; h * w];
    let mut amp = vec![0.0; h * w];
    let mut bg = vec![0.0; h * w];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (s1, s2, n) = (sy[y][0] * sx[x][0], sy[y][1] * sx[x][1], sy[y][2] * sx[x][2]);
            let den = s2 - s1 * s1 / n;
            let a = (fg[i] - s1 * fu[i] / n) / den;
            let c0 = (fu[i] / n).max(LEVEL_FLOOR);
            amp[i] = a;
            bg[i] = (fu[i] - a * s1) / n;
            z[i] = a * den.sqrt() / (phi * c0).sqrt();
        }
    }
    (z, amp, bg)
}

/// Local maxima of `-LoG` inside `mask` (and `roi`): pixels equal to the
/// maximum over the `size` square around them.
fn maxima(neg_log: &[f64], local_max: &[f64], mask: &[bool], roi: Option<&[bool]>) -> Vec<usize> {
    (0..neg_log.len())
        .filter(|&i| mask[i] && neg_log[i] == local_max[i] && roi.is_none_or(|r| r[i]))
        .collect()
}

/// 8-connected components of `mask`, numbered from 1 in raster order.
pub fn label(mask: &[bool], h: usize, w: usize) -> Vec<u32> {
    let mut labels = vec![0u32; h * w];
    let mut next = 0;
    let mut stack = Vec::new();
    for start in 0..h * w {
        if !mask[start] || labels[start] != 0 {
            continue;
        }
        next += 1;
        labels[start] = next;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (y, x) = (i / w, i % w);
            for ny in y.saturating_sub(1)..(y + 2).min(h) {
                for nx in x.saturating_sub(1)..(x + 2).min(w) {
                    let j = ny * w + nx;
                    if mask[j] && labels[j] == 0 {
                        labels[j] = next;
                        stack.push(j);
                    }
                }
            }
        }
    }
    labels
}

/// Screen a frame `d` (ADU above the offset) of dispersion `phi`: the mask
/// holds every pixel whose score at some width of `bank`, `(width, bar)`
/// pairs, reaches that width's bar. Score, flux and level are reported at
/// `sigma`, which also sets the LoG. Seeds are confined to `roi`.
pub fn screen(d: &[f64], h: usize, w: usize, sigma: f64, phi: f64, bank: &[(f64, f64)], roi: Option<&[bool]>) -> Screen {
    assert_eq!(d.len(), h * w);
    let (z, amplitude, background) = score(d, h, w, sigma, phi);
    let mut mask = vec![false; h * w];
    for &(width, bar) in bank {
        let zw = if width == sigma { z.clone() } else { score(d, h, w, width, phi).0 };
        for (m, v) in mask.iter_mut().zip(zw) {
            *m |= v >= bar;
        }
    }
    let neg_log: Vec<f64> = filters::gaussian_laplace(d, h, w, sigma, Mode::Reflect).into_iter().map(|v| -v).collect();
    let size = 2 * sigma.ceil() as usize + 1;
    let local_max = filters::maximum_filter(&neg_log, h, w, size, Mode::Reflect);
    let seeds = maxima(&neg_log, &local_max, &mask, roi);
    let labels = label(&mask, h, w);
    Screen { z, amplitude, background, mask, labels, seeds }
}

/// Expected local maxima above `u`, per pixel, of the likelihood-ratio
/// field of one emitter on noise, its centre and width searched, that lie
/// at widths `s` in `[lo, hi]`. The search stops at `lo` but runs past
/// `hi`: wider components are out-of-focus light, not reported.
///
/// The field's signed root is a smooth unit Gaussian field over position
/// and `tau = log s`. Two emitters' scores correlate as the profiles do:
/// `exp(-r^2 / (4 v))` at offset `r`, `v = s^2 + 1/12` the pixel-integrated
/// profile's variance, and `2 sqrt(v1 v2) / (v1 + v2)` across widths. The
/// metric is therefore `(dy^2 + dx^2) / (2 v) + dtau^2`: hyperbolic space,
/// curvature -1, with widths as height (Siegmund & Worsley 1995, *Ann.
/// Stat.* 23:608). For high `u` the expected number of maxima above `u` is
/// the Euler characteristic density `L3 rho3(u) + L2 rho2(u) + L1 rho1(u)`
/// (Adler & Taylor 2007), its terms local: the volume and curvature of the
/// slab between `lo` and `hi`, and the face at `lo`, a horosphere with both
/// principal curvatures 1. With `a = 1 / v(lo)` and `b = 1 / v(hi)`, per
/// unit area,
///
/// ```text
/// L3 = (a - b) / 4,  L2 = a / 4,  L1 = a / (2 pi) - 3 (a - b) / (8 pi)
/// ```
///
/// With equal bounds the width is fixed and the field 2-D, of density
/// `u exp(-u^2 / 2) / (2 pi)^(3/2) / (2 v)`. The frame's edge adds a term
/// of relative size `1 / (sqrt(a) * side)`, left out.
pub fn false_rate(u: f64, lo: f64, hi: f64) -> f64 {
    let (a, b) = (1.0 / (lo * lo + 1.0 / 12.0), 1.0 / (hi * hi + 1.0 / 12.0));
    let e = (-0.5 * u * u).exp();
    let rho2 = u * e / (2.0 * PI).powf(1.5);
    if lo >= hi {
        return 0.5 * a * rho2;
    }
    let rho1 = e / (2.0 * PI);
    let rho3 = (u * u - 1.0) * e / (4.0 * PI * PI);
    0.25 * (a - b) * rho3 + 0.25 * a * rho2 + (a / (2.0 * PI) - 3.0 * (a - b) / (8.0 * PI)) * rho1
}

/// The `u >= 1` at which [`false_rate`] over widths `[lo, hi]` is
/// `fp_per_mpx` per 10^6 pixels, by bisection; the rate falls with `u`
/// there.
pub fn threshold(fp_per_mpx: f64, lo: f64, hi: f64) -> f64 {
    let rate = |u: f64| 1e6 * false_rate(u, lo, hi);
    let (mut lo, mut hi) = (1.0, 40.0);
    if rate(lo) <= fp_per_mpx {
        return lo;
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if rate(mid) > fp_per_mpx {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Scalar `phi = pixel variance / mean`, the median over pixels of a local
/// ratio. The separable fourth difference `b = sum k_ij d_ij`, `k = [1, -4,
/// 6, -4, 1]`, of pixels of variance `phi m_ij` has variance `phi sum k_ij^2
/// m_ij = 70^2 phi mbar`, `mbar` the mean under the weights `k_ij^2 / 70^2`.
/// So `b^2 / mbar` is `70^2 phi chi2_1`, with median `70^2 phi
/// CHI2_1_MEDIAN`, wherever the light is smooth on the scale of `k`.
///
/// Variance and mean come from the same pixels with the same weights, so a
/// background that varies across the frame, or an emitter's light, raises
/// both alike; an emitter's curvature adds to `b` alone, and can only raise
/// `phi`. A mean over a wider square would spread an emitter's light past
/// the pixels its variance reaches and lower `phi`, by 16% in a field of
/// 0.02 emitters per px^2. Noise in `mbar` raises `phi` by about `0.07 phi /
/// m` (4% at 2 photons per pixel). A median of the frame as the mean would
/// be shifted by skewed or integer counts, and taken from other pixels than
/// the variance.
///
/// Frames too small to filter are taken as Poisson.
pub fn dispersion(d: &[f64], h: usize, w: usize) -> f64 {
    if h < 5 || w < 5 {
        return 1.0;
    }
    const K: [f64; 5] = [1.0, -4.0, 6.0, -4.0, 1.0];
    const K2: [f64; 5] = [1.0 / 70.0, 16.0 / 70.0, 36.0 / 70.0, 16.0 / 70.0, 1.0 / 70.0];
    let sep = |k: &[f64]| {
        let (mut a, mut b) = (vec![0.0; h * w], vec![0.0; h * w]);
        filters::convolve1d(d, &mut a, h, w, k, 0, Mode::Reflect);
        filters::convolve1d(&a, &mut b, h, w, k, 1, Mode::Reflect);
        b
    };
    let (b, m) = (sep(&K), sep(&K2));
    let ratio: Vec<f64> = (2..h - 2)
        .flat_map(|r| (2..w - 2).map(move |c| r * w + c))
        .map(|i| b[i] * b[i] / m[i].max(LEVEL_FLOOR))
        .collect();
    median(&ratio) / (statistics::CHI2_1_MEDIAN * 70.0 * 70.0)
}

pub(crate) fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 0 {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    } else {
        s[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The edge-truncated sums against a direct sum over the window.
    #[test]
    fn score_matches_a_direct_regression() {
        let (h, w, sigma) = (9, 14, 1.3);
        let d: Vec<f64> = (0..h * w).map(|i| 20.0 + ((i * 37) % 11) as f64).collect();
        let (z, a, c) = score(&d, h, w, sigma, 1.7);
        let k = psf_kernel1d(sigma);
        let r = (k.len() / 2) as isize;
        for y in 0..h as isize {
            for x in 0..w as isize {
                let mut px = Vec::new();
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (yy, xx) = (y + dy, x + dx);
                        if yy >= 0 && yy < h as isize && xx >= 0 && xx < w as isize {
                            px.push((k[(dy + r) as usize] * k[(dx + r) as usize], d[(yy * w as isize + xx) as usize]));
                        }
                    }
                }
                let n = px.len() as f64;
                let gbar = px.iter().map(|p| p.0).sum::<f64>() / n;
                let dbar = px.iter().map(|p| p.1).sum::<f64>() / n;
                let sgg: f64 = px.iter().map(|p| (p.0 - gbar).powi(2)).sum();
                let sgd: f64 = px.iter().map(|p| (p.0 - gbar) * p.1).sum();
                let i = (y * w as isize + x) as usize;
                assert!((a[i] - sgd / sgg).abs() < 1e-9 * (1.0 + a[i].abs()));
                assert!((c[i] - (dbar - sgd / sgg * gbar)).abs() < 1e-9 * (1.0 + c[i].abs()));
                assert!((z[i] - sgd / (1.7 * dbar * sgg).sqrt()).abs() < 1e-9 * (1.0 + z[i].abs()));
            }
        }
    }

    /// Seeds above `threshold(fp_per_mpx)` on Poisson noise number about
    /// `fp_per_mpx`: seeds are maxima of the LoG, not of the score, and the
    /// level is fitted, so the rate is an approximation; hold it within a
    /// factor 1.5 where the count is large enough to measure.
    #[test]
    fn noise_seeds_meet_the_false_rate() {
        let (side, frames, sigma, fp) = (128, 24, 1.45, 300.0);
        let u = threshold(fp, sigma, sigma);
        assert!((1e6 * false_rate(u, sigma, sigma) - fp).abs() < 1e-6 * fp);
        let mut state = 99u64;
        let mut uni = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let mut seeds = 0;
        for _ in 0..frames {
            let d: Vec<f64> = (0..side * side)
                .map(|_| {
                    let (l, mut k, mut p) = ((-20.0f64).exp(), 0.0, 1.0);
                    loop {
                        p *= uni();
                        if p <= l {
                            return k;
                        }
                        k += 1.0;
                    }
                })
                .collect();
            let phi = dispersion(&d, side, side);
            seeds += screen(&d, side, side, sigma, phi, &[(sigma, u)], None).seeds.len();
        }
        let per_mpx = seeds as f64 * 1e6 / (frames * side * side) as f64;
        assert!(per_mpx > fp / 1.5 && per_mpx < fp * 1.5, "{per_mpx} per Mpx at u = {u}");
    }

    #[test]
    fn the_dispersion_reads_white_noise_and_scales_with_the_image() {
        let (h, w) = (200, 180);
        // Mean 50, sd 3: phi = 9 / 50.
        let d: Vec<f64> = crate::detect::tests::normals(h * w, 7).iter().map(|v| 50.0 + 3.0 * v).collect();
        let phi = dispersion(&d, h, w);
        assert!((phi - 0.18).abs() < 0.01, "phi {phi}");
        let d7: Vec<f64> = d.iter().map(|v| 7.0 * v).collect();
        assert!((dispersion(&d7, h, w) - 7.0 * phi).abs() < 1e-9 * phi);
    }

    /// Variance per unit mean, not the median variance over the median
    /// level: on a level rising from 3 to 40 across the frame, with variance
    /// equal to the mean, the latter reads 0.84.
    #[test]
    fn the_dispersion_holds_on_a_varying_background() {
        let (h, w) = (200, 180);
        let z = crate::detect::tests::normals(h * w, 5);
        let d: Vec<f64> = (0..h * w)
            .map(|i| {
                let m = 3.0 + 37.0 * (i % w) as f64 / (w - 1) as f64;
                m + m.sqrt() * z[i]
            })
            .collect();
        let phi = dispersion(&d, h, w);
        assert!((phi - 1.0).abs() < 0.04, "phi {phi}");
    }

    #[test]
    fn labels_join_diagonal_neighbours_only() {
        #[rustfmt::skip]
        let mask = [
            1, 1, 0, 0, 0,
            0, 0, 1, 0, 1,
            0, 0, 0, 0, 1,
            1, 0, 0, 0, 0,
        ].map(|v| v == 1);
        let want = [1, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0];
        assert_eq!(label(&mask, 4, 5), want);
    }
}
