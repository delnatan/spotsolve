//! LAYER 4: the Laplace Bayes factor and its guards, against
//! `tests/fixtures/03_evidence.json`.
//!
//! Pure arithmetic on inputs supplied by the fixture, so this layer IS
//! reproducible -- the only non-trivial part is the Cholesky, and `log|F|` is a
//! sum of logs of its diagonal.

mod common;

use common::*;
use spotsolve_core::evidence::{COND_GUARD, Evidence, Prior};

#[test]
fn bayes_factors_match_the_fixture() {
    let fx = load("03_evidence");
    let mut ev = Evidence::new();

    for case in fx.cases() {
        let k = usize_at(case, "K");
        let (nb, _, fb) = mat_at(case, "F_before");
        let (na, _, fa) = mat_at(case, "F_after");
        let prior = Prior {
            lam: f64_at(case, "lam"),
            a_s: f64_at(case, "A_s"),
        };
        let (i_b, i_a) = (f64_at(case, "I_before"), f64_at(case, "I_after"));
        // Fixtures store individual emitter amplitudes; the Rust API takes their sum.
        let (sa_b, sa_a) = (
            vec_at(case, "A_before").iter().sum(),
            vec_at(case, "A_after").iter().sum(),
        );

        let (bf, cond) = ev.log_bf_add(i_b, i_a, &fb, &fa, nb, na, sa_b, sa_a, k, prior, None);
        assert_abs(
            bf,
            f64_at(case, "log_bf_add"),
            1e-10,
            &format!("K={k} log_bf_add"),
        );
        assert!(
            cond.is_finite(),
            "K={k}: cond must be finite for a well-posed F"
        );

        let rem = ev.log_bf_remove(i_a, i_b, &fa, &fb, na, nb, sa_a, sa_b, k + 1, prior, None);
        assert_abs(
            rem,
            f64_at(case, "log_bf_remove"),
            1e-10,
            &format!("K={k} log_bf_remove"),
        );
    }
}

/// Removal is the EXACT negation of addition on the same pair. That
/// antisymmetry is what stops a greedy search cycling between a move and its
/// opposite, so the fixture demands the residual be `0.0` -- not small. If it
/// is not zero, the two paths have diverged.
#[test]
fn add_and_remove_are_exactly_antisymmetric() {
    let fx = load("03_evidence");
    let mut ev = Evidence::new();

    for case in fx.cases() {
        let k = usize_at(case, "K");
        let (nb, _, fb) = mat_at(case, "F_before");
        let (na, _, fa) = mat_at(case, "F_after");
        let prior = Prior {
            lam: f64_at(case, "lam"),
            a_s: f64_at(case, "A_s"),
        };
        let (i_b, i_a) = (f64_at(case, "I_before"), f64_at(case, "I_after"));
        // Fixtures store individual emitter amplitudes; the Rust API takes their sum.
        let (sa_b, sa_a) = (
            vec_at(case, "A_before").iter().sum(),
            vec_at(case, "A_after").iter().sum(),
        );

        let (bf, _) = ev.log_bf_add(i_b, i_a, &fb, &fa, nb, na, sa_b, sa_a, k, prior, None);
        let rem = ev.log_bf_remove(i_a, i_b, &fa, &fb, na, nb, sa_a, sa_b, k + 1, prior, None);

        assert_eq!(
            bf + rem,
            0.0,
            "K={k}: antisymmetry residual is {}, not 0.0",
            bf + rem
        );
        assert_eq!(
            f64_at(case, "antisymmetry_residual"),
            0.0,
            "K={k}: the fixture itself is not antisymmetric -- regenerate it"
        );
    }
}

/// Passing a precomputed incumbent `logdet` is exact, not an approximation: it
/// is the same function of the same matrix. One incumbent is scored against
/// several proposals per search step, so this must be bit-identical or the
/// optimization changes decisions [P5].
#[test]
fn precomputed_incumbent_logdet_is_bit_identical() {
    let fx = load("03_evidence");
    let mut ev = Evidence::new();

    for case in fx.cases() {
        let k = usize_at(case, "K");
        let (nb, _, fb) = mat_at(case, "F_before");
        let (na, _, fa) = mat_at(case, "F_after");
        let prior = Prior {
            lam: f64_at(case, "lam"),
            a_s: f64_at(case, "A_s"),
        };
        let (i_b, i_a) = (f64_at(case, "I_before"), f64_at(case, "I_after"));
        // Fixtures store individual emitter amplitudes; the Rust API takes their sum.
        let (sa_b, sa_a) = (
            vec_at(case, "A_before").iter().sum(),
            vec_at(case, "A_after").iter().sum(),
        );

        let (fresh, _) = ev.log_bf_add(i_b, i_a, &fb, &fa, nb, na, sa_b, sa_a, k, prior, None);
        let pre = ev.logdet(&fb, nb);
        let (cached, _) =
            ev.log_bf_add(i_b, i_a, &fb, &fa, nb, na, sa_b, sa_a, k, prior, Some(pre));
        assert_eq!(fresh, cached, "K={k}: `before=` changed the answer");
    }
}

/// Fails closed on both sides, in the right directions.
#[test]
fn ill_posed_fisher_matrices_fail_closed() {
    let mut ev = Evidence::new();
    let prior = Prior {
        lam: 0.02,
        a_s: 950.0,
    };
    let good = [4.0, 1.0, 1.0, 4.0];
    let singular = [1.0, 1.0, 1.0, 1.0];

    // An ill-posed LARGER model must never be accepted...
    let (bf, _) = ev.log_bf_add(
        120.0, 100.0, &good, &singular, 2, 2, 900.0, 1500.0, 1, prior, None,
    );
    assert_eq!(
        bf,
        f64::NEG_INFINITY,
        "an ill-posed proposal was not refused"
    );

    // ... and an ill-posed SMALLER model is not evidence in favour of it.
    let (bf, _) = ev.log_bf_add(
        120.0, 100.0, &singular, &good, 2, 2, 900.0, 1500.0, 1, prior, None,
    );
    assert_eq!(
        bf,
        f64::NEG_INFINITY,
        "a broken incumbent was read as support for the proposal"
    );

    // On removal the directions differ: a reduced model we cannot trust means
    // keep what we have; a degenerate FULL model means removal is right.
    let r = ev.log_bf_remove(
        100.0, 120.0, &good, &singular, 2, 2, 1500.0, 900.0, 2, prior, None,
    );
    assert_eq!(
        r,
        f64::NEG_INFINITY,
        "an untrustworthy reduced model did not block removal"
    );
    let r = ev.log_bf_remove(
        100.0, 120.0, &singular, &good, 2, 2, 1500.0, 900.0, 2, prior, None,
    );
    assert_eq!(
        r,
        f64::INFINITY,
        "a degenerate full model did not force removal"
    );
}

/// The boundary divergence the guard on the removal path exists for: as an
/// emitter's amplitude shrinks, `|F| ~ A^4`, so the Laplace volume `|F|^-1/2`
/// grows as `A^-2` and the Occam term -- whose whole job is to charge for
/// complexity -- starts *paying* about 4.5 nats per decade for making the
/// emitter fainter. Any greedy search with an honest optimizer walks into this.
#[test]
fn occam_term_rewards_a_vanishing_emitter() {
    let mut ev = Evidence::new();
    // F for one well-determined emitter (b, A, y, x), plus a second whose
    // position block scales as A^2.
    let base = |a: f64| -> Vec<f64> {
        let mut f = vec![0.0; 49];
        let d = [200.0, 1.0, 800.0, 800.0, 1.0, a * a, a * a];
        for i in 0..7 {
            f[i * 7 + i] = d[i];
        }
        f
    };
    let f_small = base(1.0);
    let mut prev = f64::NEG_INFINITY;
    for a in [30.0, 3.0, 1.0, 0.1, 0.01] {
        let (_, _, ok) = ev.logdet_cond(&base(a), 7);
        assert!(ok, "F must stay positive definite as A shrinks");
        let (ld, _) = ev.logdet(&base(a), 7);
        let (ld_s, _) = ev.logdet(&f_small, 7);
        let occam = -0.5 * (ld - ld_s);
        assert!(
            occam >= prev,
            "the Occam term must grow as A shrinks, not fall"
        );
        prev = occam;
    }
    // ... and it is not a bounded nuisance: four decades of A buy ~18 nats.
    let (ld_big, _) = ev.logdet(&base(30.0), 7);
    let (ld_tiny, _) = ev.logdet(&base(0.01), 7);
    assert!(
        -0.5 * (ld_tiny - ld_big) > 15.0,
        "the divergence is milder than measured -- check the A^2 scaling"
    );
    assert!(COND_GUARD == 1e3, "COND_GUARD drifted from evidence.py");
}
