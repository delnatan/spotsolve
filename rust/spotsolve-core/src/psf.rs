//! The pixel-integrated Gaussian PSF.
//!
//! An emitter of flux `A` at `(y, x)` and width `s` adds `A E_y[r] E_x[c]`
//! to pixel `(r, c)`, where `E` integrates a unit Gaussian over one pixel
//! along each axis. Pixel centres are at integers; images are row-major.

use libm::erf;

const SQRT2: f64 = std::f64::consts::SQRT_2;
/// `sqrt(2 pi)`.
const SQRT2PI: f64 = 2.506_628_274_631_000_7;
/// `sqrt(pi)`.
const SQRTPI: f64 = 1.772_453_850_905_516;

/// Signal of the pixel under an emitter centred on it, per unit flux.
pub fn peak_factor(sigma: f64) -> f64 {
    let e = erf(0.5 / (sigma * SQRT2));
    e * e
}

/// Pixel-centre axis `[0, 1, ..., n - 1]`.
pub fn local_axis(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64).collect()
}

/// `E` along one axis for centre `c` and width `sigma`.
pub fn shape_axis(ax: &[f64], c: f64, sigma: f64, e: &mut [f64]) {
    let k = 1.0 / (sigma * SQRT2);
    for (e, &x) in e.iter_mut().zip(ax) {
        *e = 0.5 * (erf((x - c + 0.5) * k) - erf((x - c - 0.5) * k));
    }
}

/// `E`, `dE/dc` and `dE/dsigma` along one axis.
pub fn factors_axis_sigma(ax: &[f64], c: f64, sigma: f64, e: &mut [f64], de: &mut [f64], ds: &mut [f64]) {
    let k = 1.0 / (sigma * SQRT2);
    let (inv_c, inv_s) = (1.0 / (sigma * SQRT2PI), 1.0 / (sigma * SQRTPI));
    for (i, &x) in ax.iter().enumerate() {
        let (up, um) = ((x - c + 0.5) * k, (x - c - 0.5) * k);
        let (ep, em) = ((-up * up).exp(), (-um * um).exp());
        e[i] = 0.5 * (erf(up) - erf(um));
        de[i] = -(ep - em) * inv_c;
        ds[i] = (um * em - up * ep) * inv_s;
    }
}

/// Add one emitter, flux `a` at `(y, x)` of width `s`, to `m` laid out on
/// the axes `ay` (rows) and `ax` (columns).
pub fn add_emitter(m: &mut [f64], ay: &[f64], ax: &[f64], a: f64, y: f64, x: f64, s: f64) {
    let (mut ey, mut ex) = (vec![0.0; ay.len()], vec![0.0; ax.len()]);
    shape_axis(ay, y, s, &mut ey);
    shape_axis(ax, x, s, &mut ex);
    for (row, &e) in m.chunks_exact_mut(ax.len()).zip(&ey) {
        for (v, &f) in row.iter_mut().zip(&ex) {
            *v += a * e * f;
        }
    }
}

/// The `h x w` image of `theta = [b, A, y, x, s, ...]`: a constant `b` and
/// emitters with their own widths.
pub fn model(theta: &[f64], h: usize, w: usize) -> Vec<f64> {
    assert_eq!(theta.len() % 4, 1, "theta is [b, (A, y, x, s)...]");
    let (ay, ax) = (local_axis(h), local_axis(w));
    let mut m = vec![theta[0]; h * w];
    for p in theta[1..].chunks_exact(4) {
        add_emitter(&mut m, &ay, &ax, p[0], p[1], p[2], p[3]);
    }
    m
}
