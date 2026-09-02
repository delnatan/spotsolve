//! Pixel-integrated 2-D Gaussian PSF, model and analytic Jacobian.
//!
//! Ports `psf.py`.
//!
//! # Model
//!
//! ```text
//! m[i,j] = b + sum_k A_k * ey_k[i] * ex_k[j]
//! ey_k[i] = 0.5*(erf((i - cy_k + 0.5)/(sigma*sqrt2))
//!              - erf((i - cy_k - 0.5)/(sigma*sqrt2)))
//! ```
//!
//! `A_k` is TOTAL FLUX, not peak height: the pixel-integrated Gaussian sums to
//! `A` over all pixels, so a peak-height guess must be divided by
//! [`peak_factor`] to become an `A`.
//!
//! # Layouts
//!
//! `theta` is `[b, A_0, y_0, x_0, A_1, y_1, x_1, ...]`, length `3K+1`.
//!
//! Pixels are row-major: `m[r*w + c]`.
//!
//! The Jacobian is **parameter-major**, `j[q*n + i] = d m_i / d theta_q` --
//! i.e. transposed relative to the Python's `(h, w, 3K+1)`. `J` is only ever
//! consumed as `J^T (W r)` and `J^T (W J)`, and this layout makes both of those
//! contiguous reductions and drops the transposes [P7]. The Python fills a
//! `(h, w, p)` array through strided slices (`J[:, :, 2::3] = ...`), which is
//! cache-hostile; do not reproduce that.
//!
//! The separable 1-D factors are emitter-major, `ey[k*h + r]` and
//! `ex[k*w + c]`.
//!
//! # Why the derivatives are split into two entry points
//!
//! `d(ey)/d(sigma)` is read *only* by the free-sigma diagnostic, which the
//! production path never runs, while the fixed-sigma factors are evaluated
//! ~1.4M times per frame. So it is emitted by a separate function rather than
//! computed and discarded [P8]. Do not merge them behind a runtime flag and
//! rely on dead-code elimination -- it cannot see across the write into an
//! output slice.

use libm::erf;

pub const SQRT2: f64 = std::f64::consts::SQRT_2;
/// `sqrt(2*pi)`, as `psf.py`'s `SQRT2PI`.
pub const SQRT2PI: f64 = 2.506_628_274_631_000_7;
/// `sqrt(pi)`, as `psf.py`'s `SQRTPI`.
pub const SQRTPI: f64 = 1.772_453_850_905_516;

/// Number of emitters in a `theta` of length `3K+1`.
#[inline]
pub fn n_emitters(theta: &[f64]) -> usize {
    debug_assert!(theta.len() % 3 == 1, "theta length must be 3K+1");
    (theta.len() - 1) / 3
}

/// Number of emitters in a variable-sigma `theta` of length `4K+1`.
#[inline]
pub fn n_emitters_var(theta: &[f64]) -> usize {
    debug_assert!(theta.len() % 4 == 1, "theta length must be 4K+1");
    (theta.len() - 1) / 4
}

#[inline]
pub fn background(theta: &[f64]) -> f64 {
    theta[0]
}
#[inline]
pub fn amp(theta: &[f64], k: usize) -> f64 {
    theta[1 + 3 * k]
}
#[inline]
pub fn cy(theta: &[f64], k: usize) -> f64 {
    theta[2 + 3 * k]
}
#[inline]
pub fn cx(theta: &[f64], k: usize) -> f64 {
    theta[3 + 3 * k]
}

#[inline]
pub fn amp_var(theta: &[f64], k: usize) -> f64 {
    theta[1 + 4 * k]
}
#[inline]
pub fn cy_var(theta: &[f64], k: usize) -> f64 {
    theta[2 + 4 * k]
}
#[inline]
pub fn cx_var(theta: &[f64], k: usize) -> f64 {
    theta[3 + 4 * k]
}
#[inline]
pub fn sigma_var(theta: &[f64], k: usize) -> f64 {
    theta[4 + 4 * k]
}

/// Build a flat `theta` from a background and per-emitter arrays.
pub fn pack(b: f64, a: &[f64], ys: &[f64], xs: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), ys.len());
    assert_eq!(a.len(), xs.len());
    let mut t = Vec::with_capacity(3 * a.len() + 1);
    t.push(b);
    for k in 0..a.len() {
        t.push(a[k]);
        t.push(ys[k]);
        t.push(xs[k]);
    }
    t
}

/// Build a variable-sigma theta `[b, A0, y0, x0, sigma0, ...]`.
pub fn pack_var(b: f64, a: &[f64], ys: &[f64], xs: &[f64], sigmas: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), ys.len());
    assert_eq!(a.len(), xs.len());
    assert_eq!(a.len(), sigmas.len());
    let mut t = Vec::with_capacity(4 * a.len() + 1);
    t.push(b);
    for k in 0..a.len() {
        t.push(a[k]);
        t.push(ys[k]);
        t.push(xs[k]);
        t.push(sigmas[k]);
    }
    t
}

/// Ratio of an on-pixel-centre peak height to the amplitude parameter `A`:
/// `peak = A * peak_factor(sigma)`. Roughly 0.104 at `sigma = 1.2`, so an
/// observed peak-minus-background must be multiplied by ~9.6 to become an
/// initial guess for `A`.
pub fn peak_factor(sigma: f64) -> f64 {
    let e = erf(0.5 / (sigma * SQRT2));
    e * e
}

/// Scratch for the separable 1-D factors and the unpacked per-emitter values.
///
/// The pixel grid never changes within a fit and `K` is fixed once the proposal
/// is formed, so this is allocated once and borrowed for the duration [P5, P6].
/// The Python re-derives the axes on every one of the ~160 model evaluations a
/// single fit makes; that cost is an artifact of the interpreter, but the
/// allocation it implies is not, and this removes both.
pub struct Factors {
    /// `ey[k*h + r]`
    pub ey: Vec<f64>,
    /// `d ey / d cy`, same layout
    pub dey: Vec<f64>,
    /// `ex[k*w + c]`
    pub ex: Vec<f64>,
    /// `d ex / d cx`, same layout
    pub dex: Vec<f64>,
    /// `d ey / d sigma` and `d ex / d sigma`, same layouts; only used by the
    /// variable-sigma diagnostic path.
    pub dsy: Vec<f64>,
    pub dsx: Vec<f64>,
    /// unpacked amplitudes and centres, compact over `k`
    pub a: Vec<f64>,
    pub cy: Vec<f64>,
    pub cx: Vec<f64>,
    pub sigma: Vec<f64>,
}

impl Factors {
    pub fn new(h: usize, w: usize, k_max: usize) -> Self {
        Self {
            ey: vec![0.0; h * k_max],
            dey: vec![0.0; h * k_max],
            ex: vec![0.0; w * k_max],
            dex: vec![0.0; w * k_max],
            dsy: vec![0.0; h * k_max],
            dsx: vec![0.0; w * k_max],
            a: vec![0.0; k_max],
            cy: vec![0.0; k_max],
            cx: vec![0.0; k_max],
            sigma: vec![0.0; k_max],
        }
    }

    /// Grow every buffer to hold `k` emitters on an `h x w` patch.
    ///
    /// Each buffer is checked **independently**. They have different shapes --
    /// `ey` is `h*k` while `a` is just `k` -- so one being large enough says
    /// nothing about another. An earlier version keyed the whole decision on
    /// `ey`/`ex` alone, and a big-patch/few-emitter fit followed by a
    /// small-patch/many-emitter one then left `a` too short: `ey.len()` of
    /// 20*2 is not less than 8*5, so nothing was reallocated and `unpack` ran
    /// off the end of `a`. It took a specific sequence of patch shapes to
    /// surface, which is exactly the kind of bug a per-buffer check makes
    /// unrepresentable.
    pub fn ensure(&mut self, h: usize, w: usize, k: usize) {
        let k = k.max(1);
        grow(&mut self.ey, h * k);
        grow(&mut self.dey, h * k);
        grow(&mut self.ex, w * k);
        grow(&mut self.dex, w * k);
        grow(&mut self.dsy, h * k);
        grow(&mut self.dsx, w * k);
        grow(&mut self.a, k);
        grow(&mut self.cy, k);
        grow(&mut self.cx, k);
        grow(&mut self.sigma, k);
    }

    /// Copy the per-emitter values out of `theta` into the compact arrays.
    pub fn unpack(&mut self, theta: &[f64]) {
        let k = n_emitters(theta);
        for i in 0..k {
            self.a[i] = amp(theta, i);
            self.cy[i] = cy(theta, i);
            self.cx[i] = cx(theta, i);
        }
    }

    /// Copy per-emitter values out of a variable-sigma theta.
    pub fn unpack_var(&mut self, theta: &[f64]) {
        let k = n_emitters_var(theta);
        for i in 0..k {
            self.a[i] = amp_var(theta, i);
            self.cy[i] = cy_var(theta, i);
            self.cx[i] = cx_var(theta, i);
            self.sigma[i] = sigma_var(theta, i);
        }
    }
}

fn grow(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// `E` only, for one axis. `e[k*n + i]`, with `n = ax.len()`.
///
/// What [`model_ax`] needs and no more -- the position derivative costs two
/// extra `exp`s per element and is not read on this path.
pub fn shape_axis(ax: &[f64], centers: &[f64], sigma: f64, e: &mut [f64]) {
    let n = ax.len();
    let kk = 1.0 / (sigma * SQRT2);
    for (k, &c) in centers.iter().enumerate() {
        let row = &mut e[k * n..k * n + n];
        for (i, &x) in ax.iter().enumerate() {
            // (x - c + 0.5) * kk, NOT (x - c)*kk + 0.5*kk. The two agree
            // mathematically and differ in the last ulp, and float
            // associativity is part of the contract here [P2].
            row[i] = 0.5 * (erf((x - c + 0.5) * kk) - erf((x - c - 0.5) * kk));
        }
    }
}

/// `(E, dE/dc)` for one axis; each `[k*n + i]`.
///
/// ```text
/// d(ey)/d(cy) = -(1/(sigma*sqrt(2*pi))) * (exp(-u_+^2) - exp(-u_-^2))
/// ```
pub fn factors_axis(ax: &[f64], centers: &[f64], sigma: f64, e: &mut [f64], de: &mut [f64]) {
    let n = ax.len();
    let kk = 1.0 / (sigma * SQRT2);
    let inv = 1.0 / (sigma * SQRT2PI);
    for (k, &c) in centers.iter().enumerate() {
        let (er, der) = (&mut e[k * n..k * n + n], &mut de[k * n..k * n + n]);
        for (i, &x) in ax.iter().enumerate() {
            let up = (x - c + 0.5) * kk;
            let um = (x - c - 0.5) * kk;
            er[i] = 0.5 * (erf(up) - erf(um));
            der[i] = -((-up * up).exp() - (-um * um).exp()) * inv;
        }
    }
}

/// `(E, dE/dc, dE/dsigma)` for one axis; the free-sigma diagnostic path only.
///
/// ```text
/// d(ey)/d(sigma) = (1/(sigma*sqrt(pi))) * (u_- * exp(-u_-^2) - u_+ * exp(-u_+^2))
/// ```
pub fn factors_axis_sigma(
    ax: &[f64],
    centers: &[f64],
    sigma: f64,
    e: &mut [f64],
    de: &mut [f64],
    ds: &mut [f64],
) {
    let n = ax.len();
    let kk = 1.0 / (sigma * SQRT2);
    let inv_c = 1.0 / (sigma * SQRT2PI);
    let inv_s = 1.0 / (sigma * SQRTPI);
    for (k, &c) in centers.iter().enumerate() {
        for i in 0..n {
            let up = (ax[i] - c + 0.5) * kk;
            let um = (ax[i] - c - 0.5) * kk;
            let ep = (-up * up).exp();
            let em = (-um * um).exp();
            e[k * n + i] = 0.5 * (erf(up) - erf(um));
            de[k * n + i] = -(ep - em) * inv_c;
            ds[k * n + i] = (um * em - up * ep) * inv_s;
        }
    }
}

/// Render the model into `m` (length `h*w`, row-major).
///
/// `ay` and `ax` are the 1-D pixel-centre axes; the model separates, so only
/// those are ever read. `halo` is the parameter-free additive contribution
/// (frozen emitters plus the background's shape term), or `None` for zero.
pub fn model_ax(
    theta: &[f64],
    ay: &[f64],
    ax: &[f64],
    sigma: f64,
    halo: Option<&[f64]>,
    f: &mut Factors,
    m: &mut [f64],
) {
    let (h, w) = (ay.len(), ax.len());
    let k = n_emitters(theta);
    let b = background(theta);
    debug_assert_eq!(m.len(), h * w);

    for v in m.iter_mut() {
        *v = b;
    }
    if k > 0 {
        f.unpack(theta);
        shape_axis(ay, &f.cy[..k], sigma, &mut f.ey);
        shape_axis(ax, &f.cx[..k], sigma, &mut f.ex);
        for kk in 0..k {
            let a = f.a[kk];
            let (ey, ex) = (&f.ey[kk * h..kk * h + h], &f.ex[kk * w..kk * w + w]);
            for r in 0..h {
                let e_r = ey[r];
                let row = &mut m[r * w..r * w + w];
                for c in 0..w {
                    // a * (ey*ex), NOT (a*ey) * ex. Hoisting `a*ey[r]` out of
                    // the inner loop saves one multiply and changes the last
                    // ulp, which puts this entry point a hair away from
                    // `model_and_jac_ax` -- which must compute `ey*ex` anyway
                    // for the amplitude column of the Jacobian. The two are
                    // asserted bit-equal in tests/layer1_psf.rs [P2].
                    row[c] += a * (e_r * ex[c]);
                }
            }
        }
    }
    if let Some(hl) = halo {
        debug_assert_eq!(hl.len(), h * w);
        for (v, &x) in m.iter_mut().zip(hl) {
            *v += x;
        }
    }
}

/// Model and Jacobian in one pass -- the optimizer's inner loop.
///
/// Both are always needed at the same `theta`, and the `erf`/`exp` factors are
/// the whole cost of either; computing them separately evaluates the axis
/// factors four times per LM iteration instead of two.
///
/// `j` has length `p * n` with `p = 3K+1`, `n = h*w`, laid out parameter-major
/// (`j[q*n + i]`) -- see the module docs.
pub fn model_and_jac_ax(
    theta: &[f64],
    ay: &[f64],
    ax: &[f64],
    sigma: f64,
    halo: Option<&[f64]>,
    f: &mut Factors,
    m: &mut [f64],
    j: &mut [f64],
) {
    let (h, w) = (ay.len(), ax.len());
    let n = h * w;
    let k = n_emitters(theta);
    let p = 3 * k + 1;
    let b = background(theta);
    debug_assert_eq!(m.len(), n);
    debug_assert_eq!(j.len(), p * n);

    // d m / d b == 1 everywhere.
    for v in j[..n].iter_mut() {
        *v = 1.0;
    }
    for v in m.iter_mut() {
        *v = b;
    }
    if k > 0 {
        f.unpack(theta);
        factors_axis(ay, &f.cy[..k], sigma, &mut f.ey, &mut f.dey);
        factors_axis(ax, &f.cx[..k], sigma, &mut f.ex, &mut f.dex);
        for kk in 0..k {
            let a = f.a[kk];
            let ey = &f.ey[kk * h..kk * h + h];
            let dey = &f.dey[kk * h..kk * h + h];
            let ex = &f.ex[kk * w..kk * w + w];
            let dex = &f.dex[kk * w..kk * w + w];
            // The three parameter blocks are disjoint columns of j, so they are
            // split out and written contiguously rather than strided.
            let (q_a, q_y, q_x) = ((1 + 3 * kk) * n, (2 + 3 * kk) * n, (3 + 3 * kk) * n);
            for r in 0..h {
                let (e_r, de_r) = (ey[r], dey[r]);
                let a_e = a * e_r;
                let a_de = a * de_r;
                let off = r * w;
                for c in 0..w {
                    let ex_c = ex[c];
                    let v = e_r * ex_c;
                    j[q_a + off + c] = v;
                    j[q_y + off + c] = a_de * ex_c;
                    j[q_x + off + c] = a_e * dex[c];
                    m[off + c] += a * v;
                }
            }
        }
    }
    if let Some(hl) = halo {
        debug_assert_eq!(hl.len(), n);
        for (v, &x) in m.iter_mut().zip(hl) {
            *v += x;
        }
    }
}

/// Model and Jacobian with one sigma parameter per emitter.
///
/// `theta` is `[b, A0, y0, x0, sigma0, ...]`, and `j` is parameter-major with
/// `p = 4K+1`. This is for the post-hoc out-of-focus filtering stage; the
/// fixed-sigma detector does not call it.
pub fn model_and_jac_var_sigma_ax(
    theta: &[f64],
    ay: &[f64],
    ax: &[f64],
    halo: Option<&[f64]>,
    f: &mut Factors,
    m: &mut [f64],
    j: &mut [f64],
) {
    let (h, w) = (ay.len(), ax.len());
    let n = h * w;
    let k = n_emitters_var(theta);
    let p = 4 * k + 1;
    let b = background(theta);
    debug_assert_eq!(m.len(), n);
    debug_assert_eq!(j.len(), p * n);

    for v in j[..n].iter_mut() {
        *v = 1.0;
    }
    for v in m.iter_mut() {
        *v = b;
    }
    if k > 0 {
        f.unpack_var(theta);
        for kk in 0..k {
            let a = f.a[kk];
            let sigma = f.sigma[kk];
            factors_axis_sigma(
                ay,
                &f.cy[kk..kk + 1],
                sigma,
                &mut f.ey[kk * h..kk * h + h],
                &mut f.dey[kk * h..kk * h + h],
                &mut f.dsy[kk * h..kk * h + h],
            );
            factors_axis_sigma(
                ax,
                &f.cx[kk..kk + 1],
                sigma,
                &mut f.ex[kk * w..kk * w + w],
                &mut f.dex[kk * w..kk * w + w],
                &mut f.dsx[kk * w..kk * w + w],
            );
            let ey = &f.ey[kk * h..kk * h + h];
            let dey = &f.dey[kk * h..kk * h + h];
            let dsy = &f.dsy[kk * h..kk * h + h];
            let ex = &f.ex[kk * w..kk * w + w];
            let dex = &f.dex[kk * w..kk * w + w];
            let dsx = &f.dsx[kk * w..kk * w + w];
            let (q_a, q_y, q_x, q_s) = (
                (1 + 4 * kk) * n,
                (2 + 4 * kk) * n,
                (3 + 4 * kk) * n,
                (4 + 4 * kk) * n,
            );
            for r in 0..h {
                let (e_r, de_r, ds_r) = (ey[r], dey[r], dsy[r]);
                let a_e = a * e_r;
                let a_de = a * de_r;
                let a_ds = a * ds_r;
                let off = r * w;
                for c in 0..w {
                    let ex_c = ex[c];
                    let v = e_r * ex_c;
                    j[q_a + off + c] = v;
                    j[q_y + off + c] = a_de * ex_c;
                    j[q_x + off + c] = a_e * dex[c];
                    j[q_s + off + c] = a_ds * ex_c + a_e * dsx[c];
                    m[off + c] += a * v;
                }
            }
        }
    }
    if let Some(hl) = halo {
        debug_assert_eq!(hl.len(), n);
        for (v, &x) in m.iter_mut().zip(hl) {
            *v += x;
        }
    }
}

/// 1-D pixel-centre axis `[0, 1, ..., n-1]` for a patch of extent `n`.
///
/// Patch coordinates are always LOCAL and 0-based: `grid[0,0]` is global pixel
/// `(y0, x0)`. Do not switch this to global coordinates without removing every
/// local/global offset elsewhere.
pub fn local_axis(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}
