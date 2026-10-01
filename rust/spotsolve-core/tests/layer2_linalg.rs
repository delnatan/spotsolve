//! LAYER 2: the Cholesky factorization, by randomized property tests.

mod common;

use common::*;
use spotsolve_core::linalg::Chol;

/// xorshift64*: deterministic, no dependency.
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
        let mut x = b.clone();
        chol.solve_in_place(&mut x);
        for i in 0..n {
            assert_rel(x[i], x_true[i], 1e-10, &format!("n={n} solve x[{i}]"));
        }
    }
}

/// `diag(A^-1)` against solves for the unit vectors.
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
            let mut col = vec![0.0; n];
            col[i] = 1.0;
            chol.solve_in_place(&mut col);
            assert_rel(got[i], col[i], 1e-10, &format!("n={n} inv_diag[{i}]"));
        }
    }
}

/// A matrix and its transpose, symmetric only to rounding, factor to
/// bit-identical results.
#[test]
fn factorization_is_transpose_invariant() {
    let mut rng = Rng::new(17);
    let n = 21;
    let mut a = spd(&mut rng, n);
    a[3 * n + 8] += 1e-15 * a[3 * n + 8].abs();
    let at: Vec<f64> = (0..n * n).map(|i| a[(i % n) * n + i / n]).collect();

    let (mut c1, mut c2) = (Chol::new(n), Chol::new(n));
    assert!(c1.factor(&a, n) && c2.factor(&at, n));

    let b: Vec<f64> = (0..n).map(|_| rng.unif()).collect();
    let (mut x1, mut x2) = (b.clone(), b);
    c1.solve_in_place(&mut x1);
    c2.solve_in_place(&mut x2);
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
