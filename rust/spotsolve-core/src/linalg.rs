//! Small dense symmetric-positive-definite linear algebra, f64.
//!
//! [`Chol`] reuses storage for LM steps and inverse-diagonal estimates.
//! Inputs are symmetrized as `(A + A^T)/2` before lower-triangle factorization
//! so rounding differences between triangles do not affect the choice.

/// A Cholesky factorization `A = L L^T`, with reusable storage.
///
/// Call [`Chol::ensure`] before fitting; factorization does not allocate.
pub struct Chol {
    n: usize,
    /// Lower-triangular `L`, row-major `n x n`. Entries above the diagonal are
    /// scratch and must not be read.
    l: Vec<f64>,
    ok: bool,
}

impl Chol {
    pub fn new(p_max: usize) -> Self {
        Self {
            n: 0,
            l: vec![0.0; p_max * p_max],
            ok: false,
        }
    }

    /// Factorize the symmetric part of `a` (row-major `n x n`), in place.
    ///
    /// Returns `false` for non-positive-definite or non-finite input.
    /// Solve and inverse methods require a successful factorization.
    pub fn factor(&mut self, a: &[f64], n: usize) -> bool {
        debug_assert_eq!(a.len(), n * n);
        assert!(
            n * n <= self.l.len(),
            "Chol capacity {} < {n}x{n}",
            self.l.len()
        );
        self.n = n;
        self.ok = false;

        // Symmetrize into the working copy; see the module docs.
        for i in 0..n {
            for j in 0..=i {
                let v = 0.5 * (a[i * n + j] + a[j * n + i]);
                self.l[i * n + j] = v;
            }
        }
        // Standard right-looking Cholesky on the lower triangle.
        for i in 0..n {
            for j in 0..=i {
                let mut sum = self.l[i * n + j];
                for k in 0..j {
                    sum -= self.l[i * n + k] * self.l[j * n + k];
                }
                if i == j {
                    if !(sum > 0.0) || !sum.is_finite() {
                        return false;
                    }
                    self.l[i * n + i] = sum.sqrt();
                } else {
                    self.l[i * n + j] = sum / self.l[j * n + j];
                }
            }
        }
        self.ok = true;
        true
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Largest `n*n` this instance can factorize without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.l.len()
    }

    /// Grow so that an `n x n` factorization fits. A no-op when it already does.
    ///
    /// Call before [`Chol::factor`], which asserts capacity instead of growing.
    pub fn ensure(&mut self, n: usize) {
        if self.l.len() < n * n {
            self.l = vec![0.0; n * n];
            self.n = 0;
            self.ok = false;
        }
    }

    /// Solve `A x = b` by forward then back substitution.
    pub fn solve(&self, b: &[f64], x: &mut [f64]) {
        debug_assert!(self.ok);
        let n = self.n;
        debug_assert_eq!(b.len(), n);
        debug_assert_eq!(x.len(), n);
        // L y = b
        for i in 0..n {
            let mut sum = b[i];
            for k in 0..i {
                sum -= self.l[i * n + k] * x[k];
            }
            x[i] = sum / self.l[i * n + i];
        }
        // L^T x = y
        for i in (0..n).rev() {
            let mut sum = x[i];
            for k in (i + 1)..n {
                sum -= self.l[k * n + i] * x[k];
            }
            x[i] = sum / self.l[i * n + i];
        }
    }

    /// `diag(A^-1)` into `out`.
    ///
    /// `A^-1 = L^-T L^-1`, so `(A^-1)_ii = sum_k (L^-1)_ki^2` -- the inverse of
    /// the triangular factor is enough and the full inverse is never formed.
    /// `scratch` is resized to `n*n` and used for `L^-1`.
    ///
    /// Used for final uncertainties, outside the LM inner loop.
    pub fn inv_diag(&self, out: &mut [f64], scratch: &mut Vec<f64>) {
        debug_assert!(self.ok);
        let n = self.n;
        debug_assert_eq!(out.len(), n);
        scratch.clear();
        scratch.resize(n * n, 0.0);
        // Invert the lower-triangular L by forward substitution, column by
        // column: L * Linv[:, j] = e_j, so Linv is lower triangular too.
        for j in 0..n {
            scratch[j * n + j] = 1.0 / self.l[j * n + j];
            for i in (j + 1)..n {
                let mut sum = 0.0;
                for k in j..i {
                    sum -= self.l[i * n + k] * scratch[k * n + j];
                }
                scratch[i * n + j] = sum / self.l[i * n + i];
            }
        }
        for i in 0..n {
            let mut s = 0.0;
            for k in i..n {
                let v = scratch[k * n + i];
                s += v * v;
            }
            out[i] = s;
        }
    }

    /// Solve `A x = b` with `b` overwritten in place.
    ///
    /// Reuses the LM step buffer without allocating a second right-hand side.
    pub fn solve_in_place(&self, v: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let mut sum = v[i];
            for k in 0..i {
                sum -= self.l[i * n + k] * v[k];
            }
            v[i] = sum / self.l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut sum = v[i];
            for k in (i + 1)..n {
                sum -= self.l[k * n + i] * v[k];
            }
            v[i] = sum / self.l[i * n + i];
        }
    }
}
