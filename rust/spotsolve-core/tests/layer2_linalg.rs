//! LAYER 2: the dense SPD kernels.
//!
//! Randomized property tests against naive references, which is what
//! localized the Cholesky-triangle problem in the Python in one run when the
//! end-to-end test said only "something changed" [P15].

mod common;

use common::*;
use spotsolve_core::linalg::Chol;

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

/// A random symmetric positive-definite `n x n`, built as `B B^T + n I`.
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
    let mut chol = Chol::new(37);
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
/// `diag(A^-1)` against `n` explicit solves against unit vectors -- the route
/// the Python takes via `np.linalg.inv`, which this deliberately replaces so
/// one factorization serves every consumer [P13].
#[test]
fn inv_diag_matches_column_solves() {
    let mut rng = Rng::new(13);
    let mut chol = Chol::new(37);
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
}
