//! Separable image filters, matching `scipy.ndimage` exactly.
//!
//! Four calls in the pipeline sit on top of one scaffold: a 1-D pass along
//! each axis, with a boundary rule. `find_candidates` needs
//! `gaussian_laplace` and `maximum_filter`; `background_map` needs
//! `uniform_filter` and `gaussian_filter`.
//!
//! | call | site | kernel / reducer | mode |
//! |---|---|---|---|
//! | [`gaussian_laplace`] | FIND | Gaussian order 2, summed over axes | reflect |
//! | [`maximum_filter`] | FIND | sliding max | reflect |
//! | [`uniform_filter`] | background | box | nearest |
//! | [`gaussian_filter`] | background | Gaussian order 0 | nearest |
//!
//! # The one convention that must not be "fixed"
//!
//! [`gaussian_kernel1d`] normalizes the **order-0** kernel to sum 1 and only
//! then applies the derivative recurrence. A truncated order-2 kernel
//! therefore does **not** sum to zero -- at `sigma = 0.6, radius = 2` it sums
//! to `-6.5e-2`. That looks like a bug and is not: re-normalizing it rescales
//! the entire LoG response, which is then compared against a *fixed* threshold
//! of 1.5, and the candidate list changes. `tests/fixtures/07_filters.json` pins the
//! kernels as impulse responses precisely so this cannot drift.
//!
//! # Why matching scipy bit-for-bit is not required here
//!
//! It is required of the *kernels*, and the fixture asserts them to 1e-12. It
//! is not required of the filtered image, because of what consumes it: on a
//! 512x512 frame the weakest accepted candidate scores 1.5515 against a
//! threshold of 1.5 and the strongest rejected scores 1.4948, a margin of
//! 5.7e-2, and no accepted candidate has an exact tie in its max-filter
//! window. The candidate list has about eleven orders of magnitude of slack
//! over the difference any correct summation order could produce.

/// How a filter reads outside the array. `scipy.ndimage`'s names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `(d c b a | a b c d | d c b a)` -- the edge sample is duplicated.
    /// scipy's default, and what FIND uses.
    Reflect,
    /// `(a a a a | a b c d | d d d d)` -- clamp to the edge sample.
    /// What the background surface uses, so a window hanging off the frame
    /// extends the last real estimate rather than folding the interior back.
    Nearest,
}

impl Mode {
    /// Map a possibly out-of-range index onto a real one.
    ///
    /// Total for every `i`, including `|i| > n`, which happens whenever a
    /// filter is wider than the array -- `background_map` reaches this on a
    /// small frame.
    #[inline]
    pub fn index(self, i: isize, n: usize) -> usize {
        let n_i = n as isize;
        if n_i == 1 {
            return 0;
        }
        match self {
            Mode::Nearest => i.clamp(0, n_i - 1) as usize,
            Mode::Reflect => {
                let p = 2 * n_i;
                let mut k = i % p;
                if k < 0 {
                    k += p;
                }
                if k >= n_i {
                    k = p - 1 - k;
                }
                k as usize
            }
        }
    }
}

/// scipy's truncation radius: `int(truncate * sigma + 0.5)` at `truncate=4.0`.
#[inline]
pub fn kernel_radius(sigma: f64) -> usize {
    (4.0 * sigma + 0.5) as usize
}

/// scipy's `_gaussian_kernel1d`, as a `2*radius+1` tap vector indexed by
/// `x + radius` for `x` in `-radius..=radius`.
///
/// Normalization happens on the order-0 kernel, before differentiation --
/// see the module note. `order` is 0, 1 or 2; nothing here needs more.
pub fn gaussian_kernel1d(sigma: f64, order: usize, radius: usize) -> Vec<f64> {
    let sigma2 = sigma * sigma;
    let r = radius as isize;
    let mut phi: Vec<f64> = (-r..=r)
        .map(|x| (-0.5 / sigma2 * (x * x) as f64).exp())
        .collect();
    let s: f64 = phi.iter().sum();
    for v in &mut phi {
        *v /= s;
    }
    if order == 0 {
        return phi;
    }
    // q(x) is a polynomial in x with q(x)*phi(x) = the order-th derivative.
    // Applying `d/dx` once is q' + q*p' with p' = -x/sigma^2, which on the
    // coefficient vector is the matrix D + P below. scipy builds exactly this.
    let mut q = vec![0.0; order + 1];
    q[0] = 1.0;
    for _ in 0..order {
        let mut next = vec![0.0; order + 1];
        // D: q'(x) -- coefficient j+1 contributes (j+1) to coefficient j.
        for j in 0..order {
            next[j] += q[j + 1] * (j + 1) as f64;
        }
        // P: q(x) * (-x / sigma^2) -- coefficient j moves up to j+1.
        for j in 0..order {
            next[j + 1] += -q[j] / sigma2;
        }
        q = next;
    }
    (-r..=r)
        .map(|x| {
            let xf = x as f64;
            let mut acc = 0.0;
            let mut pow = 1.0;
            for &c in q.iter().take(order + 1) {
                acc += c * pow;
                pow *= xf;
            }
            acc * phi[(x + r) as usize]
        })
        .collect()
}

/// Convolve along one axis with `kernel` (indexed `x + radius`).
///
/// scipy reverses the Gaussian kernel and *correlates*; that is a convolution,
/// which is what this is. The distinction is invisible for the symmetric
/// order-0 and order-2 kernels and flips the sign of the antisymmetric
/// order-1 one, so the fixture carries an order-1 case to hold it down.
fn convolve1d(
    src: &[f64],
    dst: &mut [f64],
    h: usize,
    w: usize,
    kernel: &[f64],
    axis: usize,
    mode: Mode,
) {
    let r = (kernel.len() / 2) as isize;
    let n = if axis == 0 { h } else { w };
    // Interior bounds: inside these no index needs the boundary rule, which is
    // the whole array for a small kernel and none of it for a wide one.
    let lo = (r as usize).min(n);
    let hi = n.saturating_sub(r as usize).max(lo);
    for i in 0..h {
        for j in 0..w {
            let (fixed, along) = if axis == 0 { (j, i) } else { (i, j) };
            let mut acc = 0.0;
            if along >= lo && along < hi {
                for (t, &k) in kernel.iter().enumerate() {
                    let idx = along as isize - (t as isize - r);
                    let p = if axis == 0 {
                        idx as usize * w + fixed
                    } else {
                        fixed * w + idx as usize
                    };
                    acc += src[p] * k;
                }
            } else {
                for (t, &k) in kernel.iter().enumerate() {
                    let idx = mode.index(along as isize - (t as isize - r), n);
                    let p = if axis == 0 {
                        idx * w + fixed
                    } else {
                        fixed * w + idx
                    };
                    acc += src[p] * k;
                }
            }
            dst[i * w + j] = acc;
        }
    }
}

/// Separable convolution using the same reflect/nearest conventions as the
/// built-in filters. This supports the unnormalized Gaussian and box kernels
/// used by the Aguet local regression.
pub fn separable_filter(
    img: &[f64],
    h: usize,
    w: usize,
    kernel_y: &[f64],
    kernel_x: &[f64],
    mode: Mode,
) -> Vec<f64> {
    let mut intermediate = vec![0.0; h * w];
    let mut output = vec![0.0; h * w];
    convolve1d(img, &mut intermediate, h, w, kernel_y, 0, mode);
    convolve1d(&intermediate, &mut output, h, w, kernel_x, 1, mode);
    output
}

/// Sliding maximum along one axis over a window of `size` (odd).
fn max1d(src: &[f64], dst: &mut [f64], h: usize, w: usize, size: usize, axis: usize, mode: Mode) {
    let r = (size / 2) as isize;
    let n = if axis == 0 { h } else { w };
    for i in 0..h {
        for j in 0..w {
            let (fixed, along) = if axis == 0 { (j, i) } else { (i, j) };
            let mut m = f64::NEG_INFINITY;
            for d in -r..=r {
                let idx = mode.index(along as isize + d, n);
                let p = if axis == 0 {
                    idx * w + fixed
                } else {
                    fixed * w + idx
                };
                if src[p] > m {
                    m = src[p];
                }
            }
            dst[i * w + j] = m;
        }
    }
}

/// `scipy.ndimage.gaussian_filter` with a scalar sigma and per-axis order.
///
/// Axis 0 first, then axis 1 -- scipy's order, and it matters only through
/// rounding.
pub fn gaussian_filter_ord(
    img: &[f64],
    h: usize,
    w: usize,
    sigma: f64,
    order_y: usize,
    order_x: usize,
    mode: Mode,
) -> Vec<f64> {
    let r = kernel_radius(sigma);
    let ky = gaussian_kernel1d(sigma, order_y, r);
    let kx = gaussian_kernel1d(sigma, order_x, r);
    let mut a = vec![0.0; h * w];
    let mut b = vec![0.0; h * w];
    convolve1d(img, &mut a, h, w, &ky, 0, mode);
    convolve1d(&a, &mut b, h, w, &kx, 1, mode);
    b
}

/// `scipy.ndimage.gaussian_filter`.
pub fn gaussian_filter(img: &[f64], h: usize, w: usize, sigma: f64, mode: Mode) -> Vec<f64> {
    gaussian_filter_ord(img, h, w, sigma, 0, 0, mode)
}

/// `scipy.ndimage.gaussian_laplace`.
///
/// The sum of the two pure second derivatives, each taken with an order-0
/// Gaussian along the other axis -- scipy's `generic_laplace` with
/// `derivative2`, not a single 2-D LoG kernel.
pub fn gaussian_laplace(img: &[f64], h: usize, w: usize, sigma: f64, mode: Mode) -> Vec<f64> {
    let mut out = gaussian_filter_ord(img, h, w, sigma, 2, 0, mode);
    let dx = gaussian_filter_ord(img, h, w, sigma, 0, 2, mode);
    for (o, d) in out.iter_mut().zip(dx) {
        *o += d;
    }
    out
}

/// `scipy.ndimage.uniform_filter`, `size` odd.
pub fn uniform_filter(img: &[f64], h: usize, w: usize, size: usize, mode: Mode) -> Vec<f64> {
    let k = vec![1.0 / size as f64; size];
    let mut a = vec![0.0; h * w];
    let mut b = vec![0.0; h * w];
    convolve1d(img, &mut a, h, w, &k, 0, mode);
    convolve1d(&a, &mut b, h, w, &k, 1, mode);
    b
}

/// `scipy.ndimage.maximum_filter` over a square window, `size` odd.
pub fn maximum_filter(img: &[f64], h: usize, w: usize, size: usize, mode: Mode) -> Vec<f64> {
    let mut a = vec![0.0; h * w];
    let mut b = vec![0.0; h * w];
    max1d(img, &mut a, h, w, size, 0, mode);
    max1d(&a, &mut b, h, w, size, 1, mode);
    b
}
