//! Separable image filters matching `scipy.ndimage` with its default
//! `reflect` boundary, `(d c b a | a b c d | d c b a)`.
//!
//! Gaussian derivative kernels are formed from the normalized order-0
//! kernel, as scipy forms them: a truncated second derivative does not sum
//! to zero.

/// Reflect a possibly out-of-range index onto `0..n`, for any `i`.
#[inline]
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let p = 2 * n;
    let k = i.rem_euclid(p);
    (if k >= n { p - 1 - k } else { k }) as usize
}

/// scipy's truncation radius, `int(4 sigma + 0.5)`.
#[inline]
pub fn kernel_radius(sigma: f64) -> usize {
    (4.0 * sigma + 0.5) as usize
}

/// scipy's `_gaussian_kernel1d`: `2 radius + 1` taps of the `order`-th
/// (0, 1 or 2) derivative of a Gaussian, indexed `x + radius`.
pub fn gaussian_kernel1d(sigma: f64, order: usize, radius: usize) -> Vec<f64> {
    let sigma2 = sigma * sigma;
    let r = radius as isize;
    let mut phi: Vec<f64> = (-r..=r).map(|x| (-0.5 / sigma2 * (x * x) as f64).exp()).collect();
    let s: f64 = phi.iter().sum();
    phi.iter_mut().for_each(|v| *v /= s);
    if order == 0 {
        return phi;
    }
    // The derivative is q(x) phi(x); each d/dx maps q to q' - x q / sigma^2.
    let mut q = vec![0.0; order + 1];
    q[0] = 1.0;
    for _ in 0..order {
        let mut next = vec![0.0; order + 1];
        for j in 0..order {
            next[j] += q[j + 1] * (j + 1) as f64;
            next[j + 1] -= q[j] / sigma2;
        }
        q = next;
    }
    (-r..=r)
        .map(|x| {
            let poly = q.iter().rev().fold(0.0, |acc, c| acc * x as f64 + c);
            poly * phi[(x + r) as usize]
        })
        .collect()
}

/// Convolve along `axis` (0: rows, 1: columns) with `kernel`, indexed
/// `x + radius`.
fn convolve1d(src: &[f64], dst: &mut [f64], h: usize, w: usize, kernel: &[f64], axis: usize) {
    let r = (kernel.len() / 2) as isize;
    let n = if axis == 0 { h } else { w };
    for i in 0..h {
        for j in 0..w {
            let (fixed, along) = if axis == 0 { (j, i) } else { (i, j) };
            let inside = along as isize >= r && along as isize + r < n as isize;
            let mut acc = 0.0;
            for (t, &k) in kernel.iter().enumerate() {
                let i = along as isize - (t as isize - r);
                let idx = if inside { i as usize } else { reflect(i, n) };
                acc += k * src[if axis == 0 { idx * w + fixed } else { fixed * w + idx }];
            }
            dst[i * w + j] = acc;
        }
    }
}

/// The separable filter `ky (x) kx`, rows then columns.
pub(crate) fn separable(img: &[f64], h: usize, w: usize, ky: &[f64], kx: &[f64]) -> Vec<f64> {
    let (mut a, mut b) = (vec![0.0; h * w], vec![0.0; h * w]);
    convolve1d(img, &mut a, h, w, ky, 0);
    convolve1d(&a, &mut b, h, w, kx, 1);
    b
}

/// `scipy.ndimage.gaussian_laplace`: the sum of the two second derivatives,
/// each smoothed by the Gaussian along the other axis.
pub fn gaussian_laplace(img: &[f64], h: usize, w: usize, sigma: f64) -> Vec<f64> {
    let r = kernel_radius(sigma);
    let (k0, k2) = (gaussian_kernel1d(sigma, 0, r), gaussian_kernel1d(sigma, 2, r));
    let mut out = separable(img, h, w, &k2, &k0);
    for (o, d) in out.iter_mut().zip(separable(img, h, w, &k0, &k2)) {
        *o += d;
    }
    out
}

/// `scipy.ndimage.maximum_filter` over a `size x size` square, `size` odd.
pub fn maximum_filter(img: &[f64], h: usize, w: usize, size: usize) -> Vec<f64> {
    let r = (size / 2) as isize;
    let pass = |src: &[f64], axis: usize| {
        let n = if axis == 0 { h } else { w };
        let mut dst = vec![0.0; h * w];
        for i in 0..h {
            for j in 0..w {
                let (fixed, along) = if axis == 0 { (j, i) } else { (i, j) };
                dst[i * w + j] = (-r..=r)
                    .map(|d| reflect(along as isize + d, n))
                    .map(|k| src[if axis == 0 { k * w + fixed } else { fixed * w + k }])
                    .fold(f64::NEG_INFINITY, f64::max);
            }
        }
        dst
    };
    pass(&pass(img, 0), 1)
}
