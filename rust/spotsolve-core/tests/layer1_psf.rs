//! LAYER 1: the PSF and its derivative factors, against
//! `tests/fixtures/01_psf.json`, to 1e-13 relative (`erf` differs between
//! libms in the last bits).

mod common;

use common::*;
use spotsolve_core::psf;

#[test]
fn model_and_jacobian_match_the_fixture() {
    let fx = load("01_psf");
    const TOL: f64 = 1e-13;
    for case in fx.cases() {
        let (k, h, w) = (usize_at(case, "K"), usize_at(case, "h"), usize_at(case, "w"));
        let sigma = f64_at(case, "sigma");
        let theta = vec_at(case, "theta");
        let (_, _, want_m) = mat_at(case, "model");
        // The fixture's Jacobian is (h*w, 3K+1), pixel-major.
        let (n_pix, p3, want_j) = mat_at(case, "jac_flat");
        assert_eq!((n_pix, p3), (h * w, 3 * k + 1));
        assert_rel(psf::peak_factor(sigma), f64_at(case, "peak_factor"), TOL, "peak_factor");

        let mut theta_s = vec![theta[0]];
        for e in theta[1..].chunks_exact(3) {
            theta_s.extend_from_slice(&[e[0], e[1], e[2], sigma]);
        }
        assert_all_rel(&psf::model(&theta_s, h, w), &want_m, TOL, &format!("K={k} model"));

        let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
        let factors = |t: &[f64], c: f64| {
            let (mut e, mut d, mut s) = (vec![0.0; t.len()], vec![0.0; t.len()], vec![0.0; t.len()]);
            psf::factors_axis_sigma(t, c, sigma, &mut e, &mut d, &mut s);
            (e, d)
        };
        for q in 0..p3 {
            let got: Vec<f64> = if q == 0 {
                vec![1.0; n_pix]
            } else {
                let (e, c) = ((q - 1) / 3, (q - 1) % 3);
                let a = theta[1 + 3 * e];
                let (ey, dy) = factors(&ay, theta[2 + 3 * e]);
                let (ex, dx) = factors(&ax, theta[3 + 3 * e]);
                (0..n_pix)
                    .map(|i| match c {
                        0 => ey[i / w] * ex[i % w],
                        1 => a * dy[i / w] * ex[i % w],
                        _ => a * ey[i / w] * dx[i % w],
                    })
                    .collect()
            };
            let want: Vec<f64> = (0..n_pix).map(|i| want_j[i * p3 + q]).collect();
            assert_all_rel(&got, &want, TOL, &format!("K={k} jacobian column {q}"));
        }
    }
}

/// `A` is total flux: the model sums to it over a wide grid, and the pixel
/// under a centred emitter holds `A * peak_factor`.
#[test]
fn amplitude_is_total_flux_and_peak_factor_is_its_peak() {
    let (sigma, n) = (1.2, 41);
    let m = psf::model(&[0.0, 1000.0, 20.0, 20.0, sigma], n, n);
    assert_rel(m.iter().sum(), 1000.0, 1e-9, "flux is conserved");
    assert_rel(m[20 * n + 20], 1000.0 * psf::peak_factor(sigma), 1e-13, "peak");
}

/// The width derivative against central differences.
#[test]
fn the_width_derivative_matches_finite_differences() {
    let ax = psf::local_axis(15);
    let (c, s, h) = (6.3, 1.4, 1e-6);
    let (mut e, mut d, mut ds) = (vec![0.0; 15], vec![0.0; 15], vec![0.0; 15]);
    psf::factors_axis_sigma(&ax, c, s, &mut e, &mut d, &mut ds);
    let (mut ep, mut em) = (vec![0.0; 15], vec![0.0; 15]);
    psf::shape_axis(&ax, c, s + h, &mut ep);
    psf::shape_axis(&ax, c, s - h, &mut em);
    for i in 0..15 {
        assert!((ds[i] - (ep[i] - em[i]) / (2.0 * h)).abs() < 1e-8, "pixel {i}");
    }
}
