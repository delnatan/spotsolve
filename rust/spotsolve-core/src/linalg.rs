//! Cholesky factorization of small dense symmetric positive-definite
//! matrices, with reusable storage. Inputs are symmetrized as `(A + A^T) / 2`
//! first, so rounding differences between the triangles do not matter.

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

    /// `diag(A^-1)` into `out`.
    ///
    /// `A^-1 = L^-T L^-1`, so `(A^-1)_ii = sum_k (L^-1)_ki^2` -- the inverse of
    /// the triangular factor is enough and the full inverse is never formed.
    /// `scratch` is resized to `n*n` and used for `L^-1`.
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

    /// Solve `A x = v`, `v` overwritten by `x`.
    pub fn solve_in_place(&self, v: &mut [f64]) {
        self.forward_in_place(v);
        self.back_in_place(v);
    }

    /// `L^-1 v`, in place: then `u^T A^-1 v` is the dot product of `L^-1 u`
    /// and `L^-1 v`, with no inverse formed.
    pub fn forward_in_place(&self, v: &mut [f64]) {
        debug_assert!(self.ok);
        let n = self.n;
        for i in 0..n {
            let mut sum = v[i];
            for k in 0..i {
                sum -= self.l[i * n + k] * v[k];
            }
            v[i] = sum / self.l[i * n + i];
        }
    }

    /// `L^-T v`, in place; after [`Chol::forward_in_place`], `A^-1 v`.
    pub fn back_in_place(&self, v: &mut [f64]) {
        debug_assert!(self.ok);
        let n = self.n;
        for i in (0..n).rev() {
            let mut sum = v[i];
            for k in (i + 1)..n {
                sum -= self.l[k * n + i] * v[k];
            }
            v[i] = sum / self.l[i * n + i];
        }
    }
}
