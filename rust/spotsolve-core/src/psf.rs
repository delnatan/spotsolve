//! Pixel-integrated Gaussian PSFs with separable factors and analytic Jacobians.
//!
//! `m[i,j] = b + sum_k A_k * ey_k[i] * ex_k[j]`, where each factor integrates
//! a unit Gaussian over one pixel and `A_k` is total flux. Images are row-major.
//!
//! Free-width fits pack `[b, A, y, x, sigma, ...]`; shared-width rendering packs
//! `[b, A, y, x, ...]`. Jacobians are parameter-major (`j[q*n + i]`) for
//! contiguous weighted reductions. Factor arrays are emitter-major.
//! Render-only entry points skip derivative evaluation.

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

/// `(E, dE/dc, dE/dsigma)` for one axis in variable-width fitting.
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
                    // a * (ey*ex), NOT (a*ey) * ex: `psf.py`'s product order,
                    // which the fixture asserts to 1e-13. Hoisting `a*ey[r]`
                    // out of the loop would save a multiply and move the last
                    // ulp [P2].
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

/// Render a model with one fitted sigma per emitter, without allocating or
/// calculating a Jacobian. Used by the sparse result image after fitting.
pub fn model_var_sigma_ax(
    theta: &[f64],
    ay: &[f64],
    ax: &[f64],
    halo: Option<&[f64]>,
    f: &mut Factors,
    m: &mut [f64],
) {
    let (h, w) = (ay.len(), ax.len());
    let k = n_emitters_var(theta);
    debug_assert_eq!(m.len(), h * w);
    m.fill(background(theta));
    // Add the fixed contribution first, then reuse the same amplitude*Ey
    // factor as the Jacobian evaluator. The render-only and fitting paths
    // must describe exactly the same mean.
    if let Some(values) = halo {
        for (value, extra) in m.iter_mut().zip(values) {
            *value += extra;
        }
    }
    if k > 0 {
        f.unpack_var(theta);
        for emitter in 0..k {
            let sigma = f.sigma[emitter];
            shape_axis(
                ay,
                &f.cy[emitter..emitter + 1],
                sigma,
                &mut f.ey[emitter * h..emitter * h + h],
            );
            shape_axis(
                ax,
                &f.cx[emitter..emitter + 1],
                sigma,
                &mut f.ex[emitter * w..emitter * w + w],
            );
            let amplitude = f.a[emitter];
            let ey = &f.ey[emitter * h..emitter * h + h];
            let ex = &f.ex[emitter * w..emitter * w + w];
            for row in 0..h {
                for column in 0..w {
                    m[row * w + column] += (amplitude * ey[row]) * ex[column];
                }
            }
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
