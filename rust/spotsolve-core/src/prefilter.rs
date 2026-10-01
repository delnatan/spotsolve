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
fn correlate(src: &[f64], h: usize, w: usize, k: &[f64]) -> Vec<f64> {
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

/// Screen a frame `d` (ADU above the offset) of dispersion `phi` for
/// emitters of width `sigma`, at the score `z_min`. Seeds are confined to
/// `roi`.
pub fn screen(d: &[f64], h: usize, w: usize, sigma: f64, phi: f64, z_min: f64, roi: Option<&[bool]>) -> Screen {
    assert_eq!(d.len(), h * w);
    let (z, amplitude, background) = score(d, h, w, sigma, phi);
    let mask: Vec<bool> = z.iter().map(|&v| v >= z_min).collect();
    let neg_log: Vec<f64> = filters::gaussian_laplace(d, h, w, sigma, Mode::Reflect).into_iter().map(|v| -v).collect();
    let size = 2 * sigma.ceil() as usize + 1;
    let local_max = filters::maximum_filter(&neg_log, h, w, size, Mode::Reflect);
    let seeds = maxima(&neg_log, &local_max, &mask, roi);
    let labels = label(&mask, h, w);
    Screen { z, amplitude, background, mask, labels, seeds }
}

/// Expected local maxima above `u`, per pixel, of the score of one emitter
/// of width `sigma` on noise. The score is a smooth unit Gaussian field
/// whose covariance is the PSF correlated with itself: a Gaussian of
/// variance `2 (sigma^2 + 1/12)` per axis, pixel integration adding the
/// `1/12`. Its second-derivative matrix is `lambda I`, `lambda = 1 / (2
/// (sigma^2 + 1/12))`, and for high `u` the expected number of maxima
/// above `u` is the Euler characteristic density
/// `lambda u exp(-u^2 / 2) / (2 pi)^(3/2)`. The frame's edge adds a term
/// of relative size `1 / (sqrt(lambda) * side)`, left out.
pub fn false_rate(u: f64, sigma: f64) -> f64 {
    let lambda = 0.5 / (sigma * sigma + 1.0 / 12.0);
    lambda * u * (-0.5 * u * u).exp() / (2.0 * PI).powf(1.5)
}

/// The `u >= 1` at which [`false_rate`] is `fp_per_mpx` per 10^6 pixels,
/// by bisection; the rate falls with `u` there.
pub fn threshold(fp_per_mpx: f64, sigma: f64) -> f64 {
    let rate = |u: f64| 1e6 * false_rate(u, sigma);
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

/// Scalar `phi = pixel variance / mean`. The separable fourth difference
/// `[1, -4, 6, -4, 1]` of white noise has variance `var * 70^2`, and its
/// square has median `var * 70^2 * CHI2_1_MEDIAN`; the median pixel is taken
/// to be background. Frames too small to filter are taken as Poisson.
pub fn dispersion(d: &[f64], h: usize, w: usize) -> f64 {
    if h < 5 || w < 5 {
        return 1.0;
    }
    const K: [f64; 5] = [1.0, -4.0, 6.0, -4.0, 1.0];
    let mut a = vec![0.0; h * w];
    let mut b = vec![0.0; h * w];
    filters::convolve1d(d, &mut a, h, w, &K, 0, Mode::Reflect);
    filters::convolve1d(&a, &mut b, h, w, &K, 1, Mode::Reflect);
    let sq: Vec<f64> = (2..h - 2)
        .flat_map(|r| (2..w - 2).map(move |c| (r, c)))
        .map(|(r, c)| b[r * w + c] * b[r * w + c])
        .collect();
    let var = median(&sq) / (statistics::CHI2_1_MEDIAN * 70.0 * 70.0);
    var / median(d).max(1e-6)
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
        let u = threshold(fp, sigma);
        assert!((1e6 * false_rate(u, sigma) - fp).abs() < 1e-6 * fp);
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
            seeds += screen(&d, side, side, sigma, phi, u, None).seeds.len();
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
