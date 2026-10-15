//! Where to fit: a significance mask, LoG seeds and mask labels, after
//! u-track.
//!
//! 1. **Score.** At every pixel, regress the window `[-R, R]^2`, `R =
//!    ceil(4 sigma)`, on one pixel-integrated PSF `g` and a constant:
//!    `A = <g - gbar, d> / |g - gbar|^2`. The window stops at the frame's
//!    edge; every sum is over the pixels inside it. The score test of
//!    `A = 0` on pixels of variance `V`, the level profiled, is
//!
//!    ```text
//!    z = A |g - gbar| / sqrt(V)
//!    ```
//!
//!    `V` is the pixel variance the camera's noise curve gives the window's
//!    level ([`Curve`]), not the window's residual, so a neighbour in the
//!    window does not hide a spot.
//! 2. **Mask.** Pixels whose score reaches a bar.
//! 3. **Seeds.** Local maxima of the negative Laplacian of Gaussian over a
//!    `2 ceil(sigma) + 1` square, inside the mask.
//! 4. **Labels.** 8-connected components of the mask. A seed's fit leaves
//!    out the pixels of other components.
//!
//! [`false_rate`] counts the maxima noise alone gives, the null of
//! [`crate::posterior`].

use crate::filters;
use crate::psf;
use std::f64::consts::PI;

/// The result of screening a frame, all row-major `h x w`.
#[derive(Clone, Debug)]
pub struct Screen {
    /// Score of one emitter at each pixel.
    pub z: Vec<f64>,
    /// Regression flux and level at each pixel.
    pub amplitude: Vec<f64>,
    pub background: Vec<f64>,
    /// Pixel variance at each pixel's level.
    pub variance: Vec<f64>,
    /// The noise curve the score used. Without one given, the frame's
    /// points of its own ([`noise_points`]) and its variance per unit level
    /// ([`dispersion`]); otherwise empty and NaN.
    pub curve: Curve,
    pub points: Vec<[f64; 3]>,
    pub phi: f64,
    pub mask: Vec<bool>,
    /// Component of each mask pixel, from 1; 0 outside the mask.
    pub labels: Vec<u32>,
    /// Seed pixels (`y * w + x`), ascending.
    pub seeds: Vec<usize>,
}

/// The median of chi-squared with one degree of freedom, `Phi^-1(0.75)^2`.
pub const CHI2_1_MEDIAN: f64 = 0.454_936_423_119_572_8;

/// Lowest mean [`dispersion`] divides by, in ADU.
pub const LEVEL_FLOOR: f64 = 1e-3;
/// px: the side of the tiles [`noise_points`] takes medians over.
pub const TILE: usize = 32;

/// The centred pixel-integrated unit-flux PSF along one axis, radius
/// `ceil(4 sigma)`; the 2-D kernel is its outer product.
pub fn psf_kernel1d(sigma: f64) -> Vec<f64> {
    let r = (4.0 * sigma).ceil() as isize;
    let t: Vec<f64> = (-r..=r).map(|v| v as f64).collect();
    let mut k = vec![0.0; t.len()];
    psf::shape_axis(&t, 0.0, sigma, &mut k);
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

/// Score times the noise's sd, `A |g - gbar|`, flux and level of one
/// emitter of width `sigma` at every pixel.
pub fn score(d: &[f64], h: usize, w: usize, sigma: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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
            amp[i] = a;
            bg[i] = (fu[i] - a * s1) / n;
            z[i] = a * den.sqrt();
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

/// The frame's points of its noise curve where no emitter is
/// ([`noise_points`]), and its variance per unit level ([`dispersion`]).
/// Emitters stay out of the points: pixels whose score (`z_sd`, the score
/// times the noise's sd, from [`score`]) reaches `bar` at that variance per
/// unit level, a first guess, are left out.
fn quiet_points(d: &[f64], h: usize, w: usize, z_sd: &[f64], level: &[f64], bar: f64) -> (Vec<[f64; 3]>, f64) {
    let phi = dispersion(d, h, w);
    let quiet: Vec<bool> = z_sd.iter().zip(level).map(|(&c, &l)| c < bar * (phi * l.max(LEVEL_FLOOR)).sqrt()).collect();
    (noise_points(d, h, w, level, &quiet), phi)
}

/// The points of a frame's noise curve over its background, emitters at
/// `bar` at width `sigma` left out ([`quiet_points`]).
pub fn background_points(d: &[f64], h: usize, w: usize, sigma: f64, bar: f64) -> Vec<[f64; 3]> {
    let (z, _, level) = score(d, h, w, sigma);
    quiet_points(d, h, w, &z, &level, bar).0
}

/// Screen a frame `d` (ADU above the offset) on the camera's noise
/// `curve`, or, without one, the frame's own ([`fit_curve`]): the mask
/// holds every pixel whose score at `sigma`, which also sets the LoG,
/// reaches `bar`. Seeds are confined to `roi`.
pub fn screen(d: &[f64], h: usize, w: usize, sigma: f64, bar: f64, roi: Option<&[bool]>, curve: Option<Curve>) -> Screen {
    assert_eq!(d.len(), h * w);
    let (mut z, amplitude, background) = score(d, h, w, sigma);
    let (phi, points, curve) = match curve {
        Some(c) => (f64::NAN, Vec::new(), c),
        None => {
            let (points, phi) = quiet_points(d, h, w, &z, &background, bar);
            let curve = fit_curve(&points);
            (phi, points, curve)
        }
    };
    let variance: Vec<f64> = background.iter().map(|&l| curve.at(l)).collect();
    for (z, v) in z.iter_mut().zip(&variance) {
        *z /= v.sqrt();
    }
    let mask: Vec<bool> = z.iter().map(|&v| v >= bar).collect();
    let neg_log: Vec<f64> = filters::gaussian_laplace(d, h, w, sigma).into_iter().map(|v| -v).collect();
    let size = 2 * sigma.ceil() as usize + 1;
    let local_max = filters::maximum_filter(&neg_log, h, w, size);
    let seeds = maxima(&neg_log, &local_max, &mask, roi);
    let labels = label(&mask, h, w);
    Screen { z, amplitude, background, variance, curve, points, phi, mask, labels, seeds }
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

/// Scalar `phi = pixel variance / mean`, the median over pixels of a local
/// ratio. The separable fourth difference `b = sum k_ij d_ij`, `k = [1, -4,
/// 6, -4, 1]`, of pixels of variance `phi m_ij` has variance `phi sum k_ij^2
/// m_ij = 70^2 phi mbar`, `mbar` the mean under the weights `k_ij^2 / 70^2`.
/// So `b^2 / mbar` is `70^2 phi chi2_1`, with median `70^2 phi
/// CHI2_1_MEDIAN`, wherever the light is smooth on the scale of `k`.
///
/// Variance and mean come from the same pixels with the same weights, so a
/// background that varies across the frame, or an emitter's light, raises
/// both alike; an emitter's curvature adds to `b` alone and can only raise
/// `phi`. Noise in `mbar` raises `phi` by about `0.07 phi / m`.
///
/// Frames too small to filter are taken as Poisson.
pub fn dispersion(d: &[f64], h: usize, w: usize) -> f64 {
    if h < 5 || w < 5 {
        return 1.0;
    }
    const K: [f64; 5] = [1.0, -4.0, 6.0, -4.0, 1.0];
    const K2: [f64; 5] = [1.0 / 70.0, 16.0 / 70.0, 36.0 / 70.0, 16.0 / 70.0, 1.0 / 70.0];
    let (b, m) = (filters::separable(d, h, w, &K, &K), filters::separable(d, h, w, &K2, &K2));
    let ratio: Vec<f64> = (2..h - 2)
        .flat_map(|r| (2..w - 2).map(move |c| r * w + c))
        .map(|i| b[i] * b[i] / m[i].max(LEVEL_FLOOR))
        .collect();
    median(&ratio) / (CHI2_1_MEDIAN * 70.0 * 70.0)
}

/// A camera's noise: pixel variance `read + gain light` at `light` ADU
/// above the offset, `read` the read noise's variance (and any error in the
/// offset), `gain` the variance per ADU of light (the gain times any
/// excess-noise factor), and no less than `floor`, the quietest background
/// measured: a model's light near or below zero is not a quieter pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Curve {
    pub read: f64,
    pub gain: f64,
    pub floor: f64,
}

impl Curve {
    /// A camera counting photons without read noise, in photons.
    pub const POISSON: Curve = Curve { read: 0.0, gain: 1.0, floor: 1e-3 };

    /// The curve of a camera of `gain` over the background `points`
    /// ([`noise_points`]): the read variance from them given the gain
    /// ([`read_given`]), no pixel quieter than the quietest of them.
    pub fn of(gain: f64, points: &[[f64; 3]]) -> Self {
        Curve { read: read_given(points, gain), gain, floor: quietest(points) }
    }

    pub fn at(&self, light: f64) -> f64 {
        (self.read + self.gain * light).max(self.floor).max(f64::MIN_POSITIVE)
    }

    /// The quasi-likelihood deviance of `d` at mean `mu`, in nats,
    ///
    /// ```text
    /// q(d, mu) = integral from mu to d of (d - t) / V(t) dt,    V = at
    /// ```
    ///
    /// the likelihood a fit reweighted to `V(mu)` maximizes. It is the
    /// Poisson deviance (halved) of `d + read / gain` at `mu + read / gain`,
    /// over `gain`, where `V` is linear, and Gaussian below the floor:
    /// shot noise's skew is in it, so its ratios have the null of
    /// [`false_rate`] at every level of light.
    pub fn deviance(&self, d: f64, mu: f64) -> f64 {
        // Below `kink` the variance is the floor; without shot noise it is
        // flat throughout.
        let (floor, kink) = if self.gain > 0.0 {
            let floor = self.floor.max(f64::MIN_POSITIVE);
            (floor, (floor - self.read) / self.gain)
        } else {
            (self.at(0.0), f64::INFINITY)
        };
        let flat = |t: f64| (d * t - 0.5 * t * t) / floor;
        let (a, g) = (self.read, self.gain);
        let linear = |t: f64| (d + a / g) / g * (a + g * t).ln() - t / g;
        let antiderivative = |t: f64| if t < kink { flat(t) } else { flat(kink) + linear(t) - linear(kink) };
        antiderivative(d) - antiderivative(mu)
    }
}

/// The quietest of `points`' variances; 0 without points.
pub fn quietest(points: &[[f64; 3]]) -> f64 {
    if points.is_empty() { 0.0 } else { points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min) }
}

/// Points of the frame's noise curve where no emitter is: per tile of
/// [`TILE`] px, over its `quiet` pixels, `[median level, median variance,
/// pixels]`. The separable fourth difference `b_ij` of pixels of variance
/// `v` has variance `70^2 v` wherever the light is smooth on the scale of
/// its kernel (see [`dispersion`]), so `b^2 / (70^2 CHI2_1_MEDIAN)` has
/// median `v`. One tile's median scatters widely, neighbouring fourth
/// differences sharing most of their pixels; a curve pools them.
pub fn noise_points(d: &[f64], h: usize, w: usize, level: &[f64], quiet: &[bool]) -> Vec<[f64; 3]> {
    const K: [f64; 5] = [1.0, -4.0, 6.0, -4.0, 1.0];
    if h < 5 || w < 5 {
        return Vec::new();
    }
    let b = filters::separable(d, h, w, &K, &K);
    let inner = |r: usize, c: usize| r >= 2 && r < h - 2 && c >= 2 && c < w - 2;
    let mut pts = Vec::new();
    for r0 in (0..h).step_by(TILE) {
        for c0 in (0..w).step_by(TILE) {
            let idx: Vec<usize> = (r0..(r0 + TILE).min(h)).flat_map(|r| (c0..(c0 + TILE).min(w)).filter(move |&c| inner(r, c)).map(move |c| r * w + c)).filter(|&i| quiet[i]).collect();
            if idx.len() >= TILE * TILE / 4 {
                let v: Vec<f64> = idx.iter().map(|&i| b[i] * b[i] / (CHI2_1_MEDIAN * 70.0 * 70.0)).collect();
                pts.push([median(&idx.iter().map(|&i| level[i]).collect::<Vec<_>>()), median(&v), idx.len() as f64]);
            }
        }
    }
    pts
}

/// The noise curve through `points` by weighted least squares, each point
/// weighted by its pixels over the fitted variance squared (weights from
/// the points' own noisy medians would favour those that read low), both
/// coefficients non-negative as read variance and gain are. Where levels
/// barely vary only the variance at the frame's level is determined.
pub fn fit_curve(points: &[[f64; 3]]) -> Curve {
    let mean = points.iter().map(|p| p[1]).sum::<f64>() / points.len().max(1) as f64;
    let (mut read, mut gain) = (mean.max(f64::MIN_POSITIVE), 0.0);
    for _ in 0..3 {
        let (s0, s1, s2, t0, t1) = points.iter().fold((0.0, 0.0, 0.0, 0.0, 0.0), |(s0, s1, s2, t0, t1), p| {
            let q = p[2] / (Curve { read, gain, floor: 0.0 }.at(p[0])).powi(2);
            (s0 + q, s1 + q * p[0], s2 + q * p[0] * p[0], t0 + q * p[1], t1 + q * p[0] * p[1])
        });
        if s0.is_nan() || s0 <= 0.0 {
            break;
        }
        let det = s0 * s2 - s1 * s1;
        (read, gain) = if det > 1e-12 * s0 * s2 { ((s2 * t0 - s1 * t1) / det, (s0 * t1 - s1 * t0) / det) } else { (t0 / s0, 0.0) };
        if gain < 0.0 {
            (read, gain) = (t0 / s0, 0.0);
        } else if read < 0.0 {
            (read, gain) = (0.0, if s2 > 0.0 { t1 / s2 } else { 0.0 });
        }
    }
    Curve { read: read.max(0.0), gain, floor: quietest(points) }
}

/// The read variance on `points` given the `gain`: the median of each
/// point's variance less its light's, at least 0.
pub fn read_given(points: &[[f64; 3]], gain: f64) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    median(&points.iter().map(|p| p[1] - gain * p[0]).collect::<Vec<_>>()).max(0.0)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    let n = s.len();
    let (lo, &mut hi, _) = s.select_nth_unstable_by(n / 2, f64::total_cmp);
    if n % 2 == 0 {
        0.5 * (lo.iter().copied().fold(f64::NEG_INFINITY, f64::max) + hi)
    } else {
        hi
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
        let (z, a, c) = score(&d, h, w, sigma);
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
                assert!((z[i] - sgd / sgg.sqrt()).abs() < 1e-9 * (1.0 + z[i].abs()));
            }
        }
    }

    /// On Gaussian noise the score has unit variance, and its maxima above
    /// `u` number about [`false_rate`] at a fixed width: on the pixel lattice
    /// and with the level fitted, the count is approximate.
    #[test]
    fn noise_maxima_meet_the_false_rate() {
        let (side, frames, sigma, u) = (256, 12, 1.45, 3.5);
        let fp = 1e6 * false_rate(u, sigma, sigma);
        let (mut maxima, mut var) = (0, 0.0);
        for f in 0..frames {
            let d: Vec<f64> = crate::detect::tests::normals(side * side, 99 + f as u64).iter().map(|v| 20.0 + 20f64.sqrt() * v).collect();
            let z = screen(&d, side, side, sigma, u, None, None).z;
            let r = (4.0 * sigma).ceil() as usize;
            let inner: Vec<f64> = (r..side - r).flat_map(|y| (r..side - r).map(move |x| y * side + x)).map(|i| z[i]).collect();
            var += inner.iter().map(|v| v * v).sum::<f64>() / (inner.len() * frames) as f64;
            for y in r..side - r {
                for x in r..side - r {
                    let v = z[y * side + x];
                    if v >= u && (y - 1..=y + 1).all(|yy| (x - 1..=x + 1).all(|xx| z[yy * side + xx] <= v)) {
                        maxima += 1;
                    }
                }
            }
        }
        let per_mpx = maxima as f64 * 1e6 / (frames * (side - 12) * (side - 12)) as f64;
        assert!((var - 1.0).abs() < 0.03, "score variance {var}");
        assert!(per_mpx > 0.7 * fp && per_mpx < 1.3 * fp, "{per_mpx} per Mpx at u = {u}, predicted {fp}");
    }

    /// The deviance is the Poisson one where the curve is Poisson, the
    /// Gaussian one where it is flat, and continuous across the floor.
    #[test]
    fn the_deviance_has_its_limits() {
        let poisson = Curve { read: 0.0, gain: 1.0, floor: 1e-12 };
        for (d, mu) in [(7.0, 3.0), (3.0, 7.0), (20.0, 20.5)] {
            let exact = d * (d / mu as f64).ln() - (d - mu);
            assert!((poisson.deviance(d, mu) - exact).abs() < 1e-9, "{d} {mu}");
        }
        let flat = Curve { read: 9.0, gain: 0.0, floor: 4.0 };
        assert!((flat.deviance(5.0, -1.0) - 36.0 / 18.0).abs() < 1e-12);
        let cam = Curve { read: 7.8, gain: 2.7, floor: 10.0 };
        let kink = (10.0 - 7.8) / 2.7;
        let (below, above) = (cam.deviance(3.0, kink - 1e-9), cam.deviance(3.0, kink + 1e-9));
        assert!((below - above).abs() < 1e-8 && cam.deviance(3.0, 3.0) == 0.0);
        assert!([-4.0, 0.0, 0.5, 2.0, 9.0].iter().all(|&mu| cam.deviance(3.0, mu) > 0.0 || mu == 3.0));
    }

    /// On a background rising across the frame, with read noise and gain,
    /// the frame's curve: within 5% wherever tiles measured it.
    #[test]
    fn the_noise_curve_follows_the_background() {
        let (h, w) = (200, 260);
        let z = crate::detect::tests::normals(h * w, 9);
        let level: Vec<f64> = (0..h * w).map(|i| 1.0 + 30.0 * (i % w) as f64 / (w - 1) as f64).collect();
        let truth = Curve { read: 8.0, gain: 2.5, floor: 0.0 };
        let d: Vec<f64> = (0..h * w).map(|i| level[i] + truth.at(level[i]).sqrt() * z[i]).collect();
        let points = noise_points(&d, h, w, &level, &vec![true; h * w]);
        let c = fit_curve(&points);
        // Below the darkest tile the floor holds: the curve is measured above it.
        let darkest = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        let worst = level.iter().filter(|&&l| l >= darkest).map(|&l| (c.at(l) / truth.at(l) - 1.0).abs()).fold(0.0, f64::max);
        assert!(worst < 0.05, "{c:?}, worst {worst}");
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
    fn half_the_chi2_1_mass_lies_below_its_median() {
        // P(chi2_1 <= m) = erf(sqrt(m / 2)).
        assert!((libm::erf((CHI2_1_MEDIAN / 2.0).sqrt()) - 0.5).abs() < 1e-15);
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
