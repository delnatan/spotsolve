//! LAYER 2: the dense SPD kernels.
//!
//! Two kinds of check. Randomized property tests against naive references,
//! which is what localized the Cholesky-triangle problem in the Python in one
//! run when the end-to-end test said only "something changed" [P15]. And the
//! `logdet`/`cond` columns of `tests/fixtures/03_evidence.json`, which are
//! golden
//! values for exactly these functions.

mod common;

use common::*;
use spotsolve_core::linalg::{self, Chol};

/// A tiny deterministic PRNG, so the property tests do not need a dependency
/// and do not flap between runs. xorshift64*.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform on [-1, 1).
    fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// A random symmetric positive-definite `n x n`, built as `B B^T + n I` the
/// same way `scripts/make_fixtures.py` builds the evidence fixture's matrices.
fn spd(rng: &mut Rng, n: usize) -> Vec<f64> {
    let b: Vec<f64> = (0..n * n).map(|_| rng.unif()).collect();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += b[i * n + k] * b[j * n + k];
            }
            a[i * n + j] = s;
        }
        a[i * n + i] += n as f64;
    }
    a
}

fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
        .collect()
}

#[test]
fn solve_recovers_the_right_hand_side() {
    let mut rng = Rng::new(7);
    let mut chol = Chol::new(linalg::P_MAX);
    for n in [1usize, 2, 4, 7, 13, 25, 37] {
        let a = spd(&mut rng, n);
        let x_true: Vec<f64> = (0..n).map(|_| rng.unif() * 100.0).collect();
        let b = matvec(&a, &x_true, n);
        assert!(chol.factor(&a, n), "n={n}: SPD matrix rejected");
        let mut x = vec![0.0; n];
        chol.solve(&b, &mut x);
        for i in 0..n {
            assert_rel(x[i], x_true[i], 1e-10, &format!("n={n} solve x[{i}]"));
        }
    }
}

/// `log|A|` against a naive LU determinant. Independent route, same answer.
#[test]
fn logdet_matches_gaussian_elimination() {
    let mut rng = Rng::new(11);
    let mut chol = Chol::new(linalg::P_MAX);
    for n in [1usize, 3, 8, 20, 37] {
        let a = spd(&mut rng, n);
        assert!(chol.factor(&a, n));

        let mut m = a.clone();
        let mut ld = 0.0;
        for k in 0..n {
            let piv = m[k * n + k];
            ld += piv.abs().ln();
            for i in (k + 1)..n {
                let f = m[i * n + k] / piv;
                for j in k..n {
                    m[i * n + j] -= f * m[k * n + j];
                }
            }
        }
        assert_rel(chol.logdet(), ld, 1e-10, &format!("n={n} logdet"));
    }
}

/// `diag(A^-1)` against `n` explicit solves against unit vectors -- the route
/// the Python takes via `np.linalg.inv`, which this deliberately replaces so
/// one factorization serves every consumer [P13].
#[test]
fn inv_diag_matches_column_solves() {
    let mut rng = Rng::new(13);
    let mut chol = Chol::new(linalg::P_MAX);
    let mut scratch = Vec::new();
    for n in [1usize, 2, 5, 16, 37] {
        let a = spd(&mut rng, n);
        assert!(chol.factor(&a, n));
        let mut got = vec![0.0; n];
        chol.inv_diag(&mut got, &mut scratch);
        for i in 0..n {
            let mut e = vec![0.0; n];
            e[i] = 1.0;
            let mut col = vec![0.0; n];
            chol.solve(&e, &mut col);
            assert_rel(got[i], col[i], 1e-10, &format!("n={n} inv_diag[{i}]"));
        }
    }
}

/// `F_ij` and `F_ji` differ by an ulp in practice, so which triangle is read
/// changes the answer unless the factorization symmetrizes first [P3]. It does,
/// so feeding it a matrix or its transpose must give bit-identical results.
#[test]
fn factorization_is_transpose_invariant() {
    let mut rng = Rng::new(17);
    let n = 21;
    let mut a = spd(&mut rng, n);
    // Perturb one off-diagonal pair so the matrix is symmetric only to within
    // rounding, exactly as `J^T W J` is.
    a[3 * n + 8] += 1e-15 * a[3 * n + 8].abs();
    let at: Vec<f64> = (0..n * n).map(|i| a[(i % n) * n + i / n]).collect();

    let (mut c1, mut c2) = (Chol::new(n), Chol::new(n));
    assert!(c1.factor(&a, n) && c2.factor(&at, n));
    assert_eq!(c1.logdet(), c2.logdet(), "logdet depends on the triangle read");

    let b: Vec<f64> = (0..n).map(|_| rng.unif()).collect();
    let (mut x1, mut x2) = (vec![0.0; n], vec![0.0; n]);
    c1.solve(&b, &mut x1);
    c2.solve(&b, &mut x2);
    assert_eq!(x1, x2, "solve depends on the triangle read");
}

#[test]
fn non_positive_definite_is_rejected() {
    let mut chol = Chol::new(8);
    // Singular: a rank-1 2x2.
    assert!(!chol.factor(&[1.0, 1.0, 1.0, 1.0], 2), "singular matrix accepted");
    // Indefinite.
    assert!(!chol.factor(&[1.0, 0.0, 0.0, -1.0], 2), "indefinite matrix accepted");
    // Non-finite.
    assert!(!chol.factor(&[f64::NAN, 0.0, 0.0, 1.0], 2), "NaN accepted");

    // ... and `logdet` reports the failure rather than a number.
    let (ld, ok) = linalg::logdet(&[1.0, 1.0, 1.0, 1.0], 2, &mut chol);
    assert!(!ok && ld.is_infinite(), "logdet must fail closed on a singular F");
}

/// The `logdet` and scaled-condition columns of the evidence fixture are golden
/// values for these functions specifically. `logdet` is asserted tightly; the
/// condition number is an estimate of `kappa_1` against numpy's exact
/// `kappa_2`, so it is asserted only to be the same order and on the same side
/// of `COND_GUARD` -- which is all it is ever used for.
#[test]
fn logdet_and_cond_match_the_evidence_fixture() {
    let fx = load("03_evidence");
    let mut chol = Chol::new(linalg::P_MAX);
    let mut scratch = Vec::new();

    for case in fx.cases() {
        for (key, ld_key, cond_key) in [
            ("F_before", "logdet_before", "cond_before"),
            ("F_after", "logdet_after", "cond_after"),
        ] {
            let (n, nc, f) = mat_at(case, key);
            assert_eq!(n, nc);
            let want_ld = f64_at(case, ld_key);
            let want_cond = f64_at(case, cond_key);
            let want_ok = bool_at(case, &format!("ok_{}", &ld_key[7..]));

            let (ld, cond, ok) = linalg::logdet_cond(&f, n, &mut chol, &mut scratch);
            assert_eq!(ok, want_ok, "{key}: `ok` disagrees");
            assert_rel(ld, want_ld, 1e-12, &format!("{key} logdet"));

            // This is now numpy's own quantity -- lambda_max/lambda_min for
            // an SPD matrix -- rather than a cheaper nearby norm, so it is
            // asserted tightly. An earlier kappa_1 estimate ran 1.2-2.2x high
            // and that was enough to move a real detection decision at the
            // COND_GUARD boundary; see linalg::logdet_cond.
            assert_rel(cond, want_cond, 1e-10, &format!("{key} scaled cond"));
        }
    }
}

/// Eigenvalue extraction against a naive reference: the characteristic
/// polynomial for 2x2, and the trace/determinant identities for any size.
#[test]
fn sym_eigvals_are_correct() {
    let mut rng = Rng::new(23);
    let (mut ev, mut work) = (Vec::new(), Vec::new());

    // Exact check at 2x2, where the eigenvalues are in closed form.
    let a = [3.0, 1.0, 1.0, 2.0];
    assert!(linalg::sym_eigvals(&a, 2, &mut ev, &mut work));
    let (tr, det) = (5.0f64, 5.0f64);
    let root = (tr * tr - 4.0 * det).sqrt();
    assert_rel(ev[0], 0.5 * (tr - root), 1e-12, "2x2 lambda_min");
    assert_rel(ev[1], 0.5 * (tr + root), 1e-12, "2x2 lambda_max");

    // At larger sizes: eigenvalues are positive, sum to the trace, and their
    // logs sum to the log-determinant the Cholesky reports.
    let mut chol = Chol::new(linalg::P_MAX);
    for n in [3usize, 6, 11, 25, 37] {
        let a = spd(&mut rng, n);
        assert!(linalg::sym_eigvals(&a, n, &mut ev, &mut work), "n={n}: QL did not converge");
        assert_eq!(ev.len(), n);
        assert!(ev.windows(2).all(|w| w[0] <= w[1]), "n={n}: eigenvalues are not ascending");
        assert!(ev[0] > 0.0, "n={n}: an SPD matrix reported a non-positive eigenvalue");

        let trace: f64 = (0..n).map(|i| a[i * n + i]).sum();
        assert_rel(ev.iter().sum::<f64>(), trace, 1e-10, &format!("n={n} sum == trace"));

        assert!(chol.factor(&a, n));
        let ld: f64 = ev.iter().map(|v| v.ln()).sum();
        assert_rel(ld, chol.logdet(), 1e-10, &format!("n={n} sum of logs == logdet"));
    }
}
/// Not an assertion -- a record of how far the `kappa_1` estimate sits from
/// numpy's exact `kappa_2` on the fixture's matrices. Run with
/// `cargo test -- --nocapture cond_estimate_report` to see it.
#[test]
fn cond_estimate_report() {
    let fx = load("03_evidence");
    let mut chol = Chol::new(linalg::P_MAX);
    let mut scratch = Vec::new();
    println!("\n{:>4} {:>14} {:>14} {:>8}", "n", "kappa_2 (numpy)", "est", "ratio");
    for case in fx.cases() {
        for (key, ck) in [("F_before", "cond_before"), ("F_after", "cond_after")] {
            let (n, _, f) = mat_at(case, key);
            let want = f64_at(case, ck);
            let (_, got, _) = linalg::logdet_cond(&f, n, &mut chol, &mut scratch);
            println!("{n:>4} {want:>14.6} {got:>14.6} {:>8.4}", got / want);
        }
    }
}
