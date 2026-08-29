//! LAYER 3: the bounded optimizer, against `tests/fixtures/02_lmga.json`.
//!
//! Chaotic in the last digits by nature: a 1-ulp difference in the model
//! propagates through ~100 iterations and through the accept/reject branch on
//! the gain ratio. So the fixture is asserted on the converged objective, in
//! nats -- the unit every decision downstream is actually made in -- and
//! **not** on the iteration count [P16].

mod common;

use common::*;
use spotsolve_core::lmcl::{self, Bounds, FitOpts, FitWorkspace, Interior};
use spotsolve_core::psf;

#[test]
fn converged_objective_matches_the_fixture() {
    let fx = load("02_lmga");
    let mut ws = FitWorkspace::new();

    for case in fx.cases() {
        let k = usize_at(case, "K");
        let (h, w) = (usize_at(case, "h"), usize_at(case, "w"));
        let sigma = f64_at(case, "sigma");
        let (_, _, d) = mat_at(case, "data");
        let theta0 = vec_at(case, "theta0");
        let bounds = Bounds::new(&vec_at(case, "lower"), &vec_at(case, "upper"));
        let want_theta = vec_at(case, "theta");
        let want_i = f64_at(case, "I");
        let (n, _, want_f) = mat_at(case, "F");
        let p = 3 * k + 1;
        assert_eq!(n, p);

        let info = lmcl::fit(
            &mut ws,
            &theta0,
            h,
            w,
            sigma,
            &d,
            &bounds,
            None,
            FitOpts { max_iter: 100, ..Default::default() },
        );

        // The objective, in nats. This is the assertion that matters: it is
        // the scale a log Bayes factor is decided on.
        assert_abs(info.i_div, want_i, 1e-8, &format!("K={k} converged I"));

        // Positions to 1e-6 px, amplitudes to 1e-4 relative.
        let got = ws.theta();
        assert_abs(got[0], want_theta[0], 1e-6, &format!("K={k} background"));
        for e in 0..k {
            assert_rel(
                psf::amp(got, e),
                psf::amp(&want_theta, e),
                1e-4,
                &format!("K={k} A[{e}]"),
            );
            assert_abs(psf::cy(got, e), psf::cy(&want_theta, e), 1e-6, &format!("K={k} y[{e}]"));
            assert_abs(psf::cx(got, e), psf::cx(&want_theta, e), 1e-6, &format!("K={k} x[{e}]"));
        }

        assert_all_rel(ws.fisher(p), &want_f, 1e-9, &format!("K={k} Fisher"));

        // The fixture records the gradient infinity-norm at the Python's
        // solution. Ours must be at least as good -- if it is not, this
        // optimizer stopped somewhere the Python did not.
        assert!(
            info.converged || info.stalled,
            "K={k}: fit neither converged nor stalled in {} iterations",
            info.n_iter
        );
    }
}

/// The invariant [`Interior`] exists to enforce: no constructor, and no step,
/// can produce a parameter resting on a bound. Coleman-Li divides by the
/// distance to that bound, and one stuck coordinate collapses the step for
/// every coordinate [P1].
#[test]
fn iterates_are_always_strictly_interior() {
    let lo = vec![0.0, 1e-4, -0.5, -0.5];
    let hi = vec![40.0, 9000.0, 12.5, 12.5];
    let b = Bounds::new(&lo, &hi);

    // Start ON the bounds, and past them, from both sides.
    for start in [lo.clone(), hi.clone(), vec![-1e9, -1e9, -1e9, -1e9], vec![1e9; 4]] {
        let t = Interior::new(&start, &b);
        for i in 0..4 {
            assert!(
                t.as_slice()[i] > lo[i] && t.as_slice()[i] < hi[i],
                "Interior::new left index {i} on or outside its bound: {}",
                t.as_slice()[i]
            );
        }
    }

    // ... and a step that drives hard into a bound still lands inside.
    let t = Interior::new(&[5.0, 100.0, 6.0, 6.0], &b);
    let mut out = t.clone();
    out.set_step(&t, &[-1e12, -1e12, 1e12, -1e12], &b);
    for i in 0..4 {
        assert!(
            out.as_slice()[i] > lo[i] && out.as_slice()[i] < hi[i],
            "set_step left index {i} on or outside its bound: {}",
            out.as_slice()[i]
        );
    }

    // The margin is RELATIVE to each bound's own range, because a position is
    // bounded over ~13 px and an amplitude over ~9000 electrons; one absolute
    // epsilon would be a different constraint for each.
    let at_lo = Interior::new(&lo, &b);
    let m_amp = at_lo.as_slice()[1] - lo[1];
    let m_pos = at_lo.as_slice()[2] - lo[2];
    assert!(m_amp > m_pos * 100.0, "margin is not scaled to each bound's range");
}

/// The I-divergence's `0*log(0) := 0` convention, and its behaviour on the
/// `d <= 0` pixels that offset-subtracted real data actually contains.
#[test]
fn i_divergence_handles_non_positive_data() {
    let d = [0.0, -3.0, 4.0];
    let m = [2.0, 1.0, 4.0];
    // Only the third pixel contributes a log term; the first two contribute
    // just -(d - m).
    let want = -(0.0 - 2.0) + -(-3.0 - 1.0) + (4.0 * (4.0f64 / 4.0).ln() - 0.0);
    assert_abs(lmcl::i_divergence(&d, &m), want, 1e-15, "i_divergence");

    // It is zero exactly when the model reproduces the data.
    let d2 = [1.0, 5.0, 12.5];
    assert_abs(lmcl::i_divergence(&d2, &d2), 0.0, 1e-13, "I(d,d) == 0");
}

/// A fit started at the truth on noiseless data must stay there and report
/// convergence immediately -- the cheapest possible check that the gradient,
/// the Jacobian and the objective agree with each other.
#[test]
fn noiseless_fit_from_truth_is_a_fixed_point() {
    let (h, w, sigma) = (15usize, 15usize, 1.2);
    let theta = psf::pack(4.0, &[1500.0], &[7.3], &[8.1]);
    let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
    let mut f = psf::Factors::new(h, w, 1);
    let mut d = vec![0.0; h * w];
    psf::model_ax(&theta, &ay, &ax, sigma, None, &mut f, &mut d);

    let lo = vec![0.0, 1e-4, -0.5, -0.5];
    let hi = vec![40.0, 9e4, h as f64 - 0.5, w as f64 - 0.5];
    let b = Bounds::new(&lo, &hi);
    let mut ws = FitWorkspace::new();
    let info = lmcl::fit(&mut ws, &theta, h, w, sigma, &d, &b, None, FitOpts::default());

    assert!(info.converged, "did not converge from the exact optimum");
    assert_abs(info.i_div, 0.0, 1e-9, "I at the truth on noiseless data");
    for q in 0..theta.len() {
        assert_abs(ws.theta()[q], theta[q], 1e-6, &format!("theta[{q}] moved"));
    }
}

/// `F` is symmetric by construction, and this port computes only its upper
/// triangle and mirrors it -- so it must come out *exactly* symmetric, not
/// symmetric to within rounding. That is what makes the triangle question in
/// [P3] unable to arise: `Chol`'s symmetrization has nothing left to do.
#[test]
fn reported_fisher_is_exactly_symmetric() {
    let fx = load("02_lmga");
    let mut ws = FitWorkspace::new();
    for case in fx.cases() {
        let k = usize_at(case, "K");
        let (h, w) = (usize_at(case, "h"), usize_at(case, "w"));
        let (_, _, d) = mat_at(case, "data");
        let bounds = Bounds::new(&vec_at(case, "lower"), &vec_at(case, "upper"));
        let p = 3 * k + 1;
        lmcl::fit(
            &mut ws,
            &vec_at(case, "theta0"),
            h,
            w,
            f64_at(case, "sigma"),
            &d,
            &bounds,
            None,
            FitOpts::default(),
        );
        let f = ws.fisher(p);
        for i in 0..p {
            for j in 0..p {
                assert_eq!(
                    f[i * p + j],
                    f[j * p + i],
                    "K={k}: F[{i},{j}] != F[{j},{i}] -- the two triangles have diverged"
                );
            }
        }
    }
}

/// One `FitWorkspace` is reused across every patch of a frame, and patches vary
/// wildly in shape: a wide sparse one, then a small crowded one. The buffers
/// have different shapes (`ey` is `h*k`, `a` is `k`), so growing them on a
/// single combined test lets one hide another.
///
/// This is a regression test. The original `ensure` keyed on `ey`/`ex` only,
/// and a 20x20 K=2 fit followed by an 8x8 K=5 fit left `a` two long for five
/// emitters -- `ey.len()` of 40 is not less than 40, so nothing was
/// reallocated. It panicked inside `Factors::unpack`, and only on a specific
/// sequence of patch shapes, which is why no fixture caught it.
#[test]
fn one_workspace_survives_any_sequence_of_patch_shapes() {
    let sigma = 1.2;
    let mut ws = FitWorkspace::new();
    // Deliberately adversarial: shrinking pixel counts against growing K.
    for &(h, w, k) in &[(20usize, 20usize, 2usize), (8, 8, 5), (30, 4, 1), (5, 5, 12), (13, 11, 3)]
    {
        let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
        let a: Vec<f64> = (0..k).map(|i| 500.0 + 100.0 * i as f64).collect();
        let ys: Vec<f64> = (0..k).map(|i| 1.0 + (i % h.max(1)) as f64).collect();
        let xs: Vec<f64> = (0..k).map(|i| 1.0 + (i % w.max(1)) as f64).collect();
        let theta = psf::pack(4.0, &a, &ys, &xs);

        let mut f = psf::Factors::new(h, w, k);
        let mut d = vec![0.0; h * w];
        psf::model_ax(&theta, &ay, &ax, sigma, None, &mut f, &mut d);

        let mut lo = vec![0.0];
        let mut hi = vec![50.0];
        for _ in 0..k {
            lo.extend_from_slice(&[1e-4, -0.5, -0.5]);
            hi.extend_from_slice(&[9e4, h as f64 - 0.5, w as f64 - 0.5]);
        }
        let b = Bounds::new(&lo, &hi);
        let info = lmcl::fit(&mut ws, &theta, h, w, sigma, &d, &b, None, FitOpts::default());
        assert!(
            info.i_div.is_finite(),
            "{h}x{w} K={k}: fit produced a non-finite objective after a reused workspace"
        );
    }
}
