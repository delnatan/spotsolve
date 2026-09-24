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

impl Chol {
    /// Solve `L y = v` in place (forward substitution only), so that
    /// `L^-1 A L^-T` can be formed for a symmetric generalized eigenproblem.
    pub fn solve_lower_in_place(&self, v: &mut [f64]) {
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
}

/// Solve a banded symmetric-positive-definite system in place.
///
/// `a` holds the lower band of the `n x n` matrix, row-major `n x (kd+1)`:
/// `A[i][j]` for `i - kd <= j <= i` sits at `a[i * (kd+1) + kd + j - i]`. It
/// is overwritten by the Cholesky factor, and `b` by the solution. O(n kd^2);
/// a bilinear node lattice `nx` wide couples nodes at most `nx + 1` apart.
/// Returns `false` for non-positive-definite or non-finite input.
pub fn band_solve(a: &mut [f64], n: usize, kd: usize, b: &mut [f64]) -> bool {
    let wd = kd + 1;
    debug_assert_eq!(a.len(), n * wd);
    debug_assert_eq!(b.len(), n);
    let at = |i: usize, j: usize| i * wd + kd + j - i;
    for i in 0..n {
        let j0 = i.saturating_sub(kd);
        for j in j0..=i {
            let mut sum = a[at(i, j)];
            for k in j0.max(j.saturating_sub(kd))..j {
                sum -= a[at(i, k)] * a[at(j, k)];
            }
            if i == j {
                if !(sum > 0.0) || !sum.is_finite() {
                    return false;
                }
                a[at(i, i)] = sum.sqrt();
            } else {
                a[at(i, j)] = sum / a[at(j, j)];
            }
        }
    }
    for i in 0..n {
        let mut sum = b[i];
        for k in i.saturating_sub(kd)..i {
            sum -= a[at(i, k)] * b[k];
        }
        b[i] = sum / a[at(i, i)];
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for k in i + 1..(i + wd).min(n) {
            sum -= a[at(k, i)] * b[k];
        }
        b[i] = sum / a[at(i, i)];
    }
    true
}

/// Largest eigenvalue of a small symmetric matrix (row-major `n x n`), by
/// cyclic Jacobi rotations. `s` is destroyed.
pub fn sym_eig_max(s: &mut [f64], n: usize) -> f64 {
    debug_assert_eq!(s.len(), n * n);
    for _sweep in 0..50 {
        let off: f64 = (0..n).flat_map(|i| (0..i).map(move |j| (i, j))).map(|(i, j)| s[i * n + j].powi(2)).sum();
        let diag: f64 = (0..n).map(|i| s[i * n + i].powi(2)).sum();
        if !(off > 1e-30 * diag) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = s[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (s[q * n + q] - s[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;
                for k in 0..n {
                    let (skp, skq) = (s[k * n + p], s[k * n + q]);
                    s[k * n + p] = c * skp - sn * skq;
                    s[k * n + q] = sn * skp + c * skq;
                }
                for k in 0..n {
                    let (spk, sqk) = (s[p * n + k], s[q * n + k]);
                    s[p * n + k] = c * spk - sn * sqk;
                    s[q * n + k] = sn * spk + c * sqk;
                }
            }
        }
    }
    (0..n).map(|i| s[i * n + i]).fold(f64::NEG_INFINITY, f64::max)
}
