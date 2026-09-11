//! LAYER 1: the forward model and its Jacobian, against
//! `tests/fixtures/01_psf.json`.
//!
//! Reproducible to ~1e-13 relative, NOT bit-exact: every libm's `erf` differs,
//! and that sets the accuracy floor for every layer above this one. If this
//! layer is loose, nothing above it can be tight.

mod common;

use common::*;
use spotsolve_core::psf;

/// The fit's variable-width model and Jacobian, with every width at the
/// fixture's one sigma, against the fixture's fixed-width model: the model and
/// the background, flux and position columns must agree. The width columns
/// are checked by finite differences below.
#[test]
fn model_and_jacobian_match_the_fixture() {
    let fx = load("01_psf");
    const TOL: f64 = 1e-13;

    for case in fx.cases() {
        let k = usize_at(case, "K");
        let h = usize_at(case, "h");
        let w = usize_at(case, "w");
        let sigma = f64_at(case, "sigma");
        let theta = vec_at(case, "theta");
        let (_, _, want_m) = mat_at(case, "model");
        // The fixture stores the Jacobian as (h*w, 3K+1) row-major over
        // pixels; this crate lays it out parameter-major, so the comparison
        // transposes. See psf.rs's module docs for why the layouts differ.
        let (n_pix, p3, want_j) = mat_at(case, "jac_flat");
        assert_eq!(n_pix, h * w);
        assert_eq!(p3, 3 * k + 1);

        assert_rel(
            psf::peak_factor(sigma),
            f64_at(case, "peak_factor"),
            TOL,
            &format!("K={k} peak_factor"),
        );

        let mut theta_var = vec![theta[0]];
        for e in 0..k {
            theta_var.extend_from_slice(&[theta[1 + 3 * e], theta[2 + 3 * e], theta[3 + 3 * e], sigma]);
        }
        let p = theta_var.len();
        let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
        let mut f = psf::Factors::new(h, w, k.max(1));
        let (mut m, mut j) = (vec![0.0; n_pix], vec![0.0; p * n_pix]);
        psf::model_and_jac_var_sigma_ax(&theta_var, &ay, &ax, None, &mut f, &mut m, &mut j);
        assert_all_rel(&m, &want_m, TOL, &format!("K={k} model"));

        // Column q3 of the fixed layout is column q of the free-width one.
        let col = |q3: usize| if q3 == 0 { 0 } else { 1 + 4 * ((q3 - 1) / 3) + (q3 - 1) % 3 };
        for q3 in 0..p3 {
            let got: Vec<f64> = (0..n_pix).map(|i| j[col(q3) * n_pix + i]).collect();
            let want: Vec<f64> = (0..n_pix).map(|i| want_j[i * p3 + q3]).collect();
            assert_all_rel(&got, &want, TOL, &format!("K={k} jacobian column {q3}"));
        }

        // The render-only entry point is the same sum of the same products.
        let mut m2 = vec![0.0; n_pix];
        psf::model_ax(&theta, &ay, &ax, sigma, None, &mut f, &mut m2);
        assert_all_rel(&m2, &want_m, TOL, &format!("K={k} model_ax"));
    }
}

#[test]
fn halo_is_a_plain_additive_term() {
    let fx = load("01_psf");
    let case = &fx.cases()[2];
    let (h, w) = (usize_at(case, "h"), usize_at(case, "w"));
    let sigma = f64_at(case, "sigma");
    let theta = vec_at(case, "theta");
    let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
    let mut f = psf::Factors::new(h, w, psf::n_emitters(&theta).max(1));

    let halo: Vec<f64> = (0..h * w).map(|i| 0.5 + 0.01 * i as f64).collect();
    let (mut a, mut b) = (vec![0.0; h * w], vec![0.0; h * w]);
    psf::model_ax(&theta, &ay, &ax, sigma, None, &mut f, &mut a);
    psf::model_ax(&theta, &ay, &ax, sigma, Some(&halo), &mut f, &mut b);
    for i in 0..h * w {
        assert_rel(b[i], a[i] + halo[i], 1e-15, "halo add");
    }
}

/// The pixel-integrated Gaussian sums to `A` over the whole plane, which is
/// what makes `A` a total flux rather than a peak height, and what
/// `peak_factor` converts between. Checked on a grid wide enough that the
/// truncation is negligible.
#[test]
fn amplitude_is_total_flux_and_peak_factor_is_its_peak() {
    let sigma = 1.2;
    let (h, w) = (41, 41);
    let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
    let theta = psf::pack(0.0, &[1000.0], &[20.0], &[20.0]);
    let mut f = psf::Factors::new(h, w, 1);
    let mut m = vec![0.0; h * w];
    psf::model_ax(&theta, &ay, &ax, sigma, None, &mut f, &mut m);

    let total: f64 = m.iter().sum();
    assert_rel(total, 1000.0, 1e-9, "flux is conserved");
    // The emitter sits exactly on pixel centre (20, 20), so that pixel's value
    // is A * peak_factor by definition.
    assert_rel(
        m[20 * w + 20],
        1000.0 * psf::peak_factor(sigma),
        1e-13,
        "peak",
    );
}

#[test]
fn variable_sigma_jacobian_matches_finite_difference() {
    let (h, w) = (13, 15);
    let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
    let theta = psf::pack_var(3.0, &[500.0, 800.0], &[6.2, 4.1], &[5.7, 9.3], &[1.1, 1.6]);
    let n = h * w;
    let p = theta.len();
    let mut f = psf::Factors::new(h, w, 2);
    let mut m = vec![0.0; n];
    let mut j = vec![0.0; p * n];
    psf::model_and_jac_var_sigma_ax(&theta, &ay, &ax, None, &mut f, &mut m, &mut j);

    for q in 0..p {
        let step = 1e-6 * theta[q].abs().max(1.0);
        let mut tp = theta.clone();
        let mut tm = theta.clone();
        tp[q] += step;
        tm[q] -= step;
        let (mut mp, mut mm) = (vec![0.0; n], vec![0.0; n]);
        let (mut jp, mut jm) = (vec![0.0; p * n], vec![0.0; p * n]);
        psf::model_and_jac_var_sigma_ax(&tp, &ay, &ax, None, &mut f, &mut mp, &mut jp);
        psf::model_and_jac_var_sigma_ax(&tm, &ay, &ax, None, &mut f, &mut mm, &mut jm);
        for i in 0..n {
            let fd = (mp[i] - mm[i]) / (2.0 * step);
            let got = j[q * n + i];
            let scale = fd.abs().max(got.abs()).max(1.0);
            assert!(
                (got - fd).abs() / scale < 2e-8,
                "q={q} i={i}: got {got}, fd {fd}"
            );
        }
    }
}

#[test]
fn variable_sigma_render_matches_model_and_jacobian_path() {
    let (h, w) = (13, 11);
    let ay = (0..h).map(|v| v as f64).collect::<Vec<_>>();
    let ax = (0..w).map(|v| v as f64).collect::<Vec<_>>();
    let theta = psf::pack_var(3.5, &[700.0, 400.0], &[5.2, 8.1], &[4.7, 6.3], &[1.05, 1.7]);
    let mut direct = vec![0.0; h * w];
    let mut combined = vec![0.0; h * w];
    let mut jacobian = vec![0.0; theta.len() * h * w];
    psf::model_var_sigma_ax(
        &theta,
        &ay,
        &ax,
        None,
        &mut psf::Factors::new(h, w, 2),
        &mut direct,
    );
    psf::model_and_jac_var_sigma_ax(
        &theta,
        &ay,
        &ax,
        None,
        &mut psf::Factors::new(h, w, 2),
        &mut combined,
        &mut jacobian,
    );
    assert_all_rel(&direct, &combined, 1e-14, "variable-sigma render");
}
