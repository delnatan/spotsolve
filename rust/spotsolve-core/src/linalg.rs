//! Small dense symmetric-positive-definite linear algebra, f64.
//!
//! Everything in the hot path is at most `P_MAX x P_MAX`, and nothing in a fit
//! needs the heap once the workspace is built [P13].
//!
//! # One factorization, four consumers
//!
//! The Python computes `log|F|` from `cho_factor`, then `diag(F^-1)` from a
//! separate `np.linalg.inv`, then the condition number from a full
//! `np.linalg.cond` SVD -- three routes to the same matrix, microseconds apart,
//! each disagreeing with the others in the last ulp. That is a real risk at the
//! `PRUNE_TAU` boundary for no gain. Here a [`Chol`] is computed once and feeds
//! [`Chol::logdet`], [`Chol::solve`], [`Chol::inv_diag`] and
//! [`Chol::scaled_cond_est`].
//!
//! # Which triangle
//!
//! `F = J^T W J` is symmetric only to within rounding: `F_ij` and `F_ji` are
//! separate reductions with different summation orders and can differ by an
//! ulp, so *which triangle you factorize changes the answer* [P3]. Rather than
//! pick one and hope every caller agrees, [`Chol::factor`] symmetrizes on entry
//! (`F <- (F + F^T)/2`) and then factorizes the **lower** triangle. At `n <= 37`
//! that costs ~600 flops and removes a whole class of "why did this change".

/// Cap on emitters in one joint fit, matching `core.py`'s `k_max`.
pub const K_MAX: usize = 12;
/// Cap on the parameter count, `3*K_MAX + 1`.
pub const P_MAX: usize = 3 * K_MAX + 1;

/// A Cholesky factorization `A = L L^T`, with reusable storage.
///
/// Allocated once per fit and re-factorized in place: the LM inner loop
/// factorizes a ~25x25 matrix on the order of 700k times per frame, so this
/// must not allocate [P6].
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
    /// Returns `false` if `a` is not positive definite or is not finite, in
    /// which case no other method may be called. Callers must treat that as
    /// "this model is ill-posed", never as evidence for anything.
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

    /// `log|A|`, from the factor already in hand.
    pub fn logdet(&self) -> f64 {
        debug_assert!(self.ok);
        let n = self.n;
        let mut s = 0.0;
        for i in 0..n {
            s += self.l[i * n + i].ln();
        }
        2.0 * s
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
    /// This is off the LM inner loop: it is read only by `refine`'s reported
    /// standard errors and by `_prune`'s `A/SE` test, once per fit.
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
    /// The LM inner loop solves for the step ~700k times per frame; going
    /// through [`Chol::solve`] would need a separate right-hand side buffer,
    /// and materializing one per trial is the allocation [P6] exists to
    /// forbid.
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

/// Threshold on the diagonally scaled condition number, above which `F` is
/// unusable. Matches `evidence.py`'s `COND_GUARD`.
pub const COND_GUARD: f64 = 1e3;

/// `(log|F|, scaled condition estimate, ok)` from a single factorization.
///
/// # Why the condition number must be scaled
///
/// The raw `cond(F)` is useless as an absolute test because `F` mixes
/// parameters with different units -- background in counts, amplitude in total
/// flux, position in pixels. Measured at `sigma = 1.2`: a pristine isolated
/// emitter reads 8.4e6 raw but 1.6 scaled; a genuinely degenerate pair at
/// 0.5 sigma reads 1.3e11 raw and 1.8e4 scaled. No fixed raw threshold
/// separates them; the scaled form does, and is invariant to
/// reparameterization.
///
/// # Why this is `lambda_max / lambda_min` and not something cheaper
///
/// The Python calls `np.linalg.cond`, which for a symmetric positive definite
/// matrix is exactly `lambda_max / lambda_min`. An earlier version of this
/// function substituted a Hager-Higham `kappa_1` estimate off the Cholesky
/// factor, reasoning that `COND_GUARD` is measured never to fire so any nearby
/// quantity would do.
///
/// That was wrong, and the `06_passes` fixture caught it on the first crowded
/// field: a split proposal with `log BF = +0.457` had an exact `kappa_2` under
/// the guard and a `kappa_1` estimate of 1.42e3 over it, so the port refused an
/// emitter the Python accepted -- and the refusal cascaded into a different
/// configuration for the whole cluster. `COND_GUARD` can reject a proposal
/// before its evidence is weighed, which makes it a **detection rule** [P12];
/// approximating a detection rule is not a free optimization.
///
/// So this computes the eigenvalues ([`sym_eigvals`]) rather than a norm
/// estimate. It runs once per proposal, not per LM iteration.
/// [`near_cond_guard`] is kept anyway, to flag any run that comes near the
/// boundary at all.
///
/// `ok` is false when `F` is not positive definite, or when any diagonal entry
/// is non-positive or non-finite. That last test is on `F` itself, not on the
/// factor, and it is what lets a caller that skips the condition number still
/// fail closed on exactly the same conditions.
pub fn logdet_cond(
    f: &[f64],
    n: usize,
    chol: &mut Chol,
    scratch: &mut Vec<f64>,
) -> (f64, f64, bool) {
    if !diag_is_usable(f, n) {
        // Still report the determinant when the factorization itself succeeds,
        // matching the Python, which computes it before testing the diagonal.
        let ld = if chol.factor(f, n) {
            chol.logdet()
        } else {
            f64::INFINITY
        };
        return (ld, f64::INFINITY, false);
    }
    if !chol.factor(f, n) {
        return (f64::INFINITY, f64::INFINITY, false);
    }
    let ld = chol.logdet();

    // Scale to unit diagonal, then factorize the scaled matrix. Scaling is a
    // congruence by a positive diagonal, so it cannot destroy definiteness.
    scratch.clear();
    scratch.resize(n * n + n, 0.0);
    let (sf, s) = scratch.split_at_mut(n * n);
    for i in 0..n {
        s[i] = 1.0 / f[i * n + i].sqrt();
    }
    for i in 0..n {
        for j in 0..n {
            sf[i * n + j] = f[i * n + j] * s[i] * s[j];
        }
    }
    let sf = sf.to_vec();
    let (mut ev, mut work) = (Vec::new(), Vec::new());
    if !sym_eigvals(&sf, n, &mut ev, &mut work) {
        return (ld, f64::INFINITY, false);
    }
    let (lo, hi) = (ev[0], ev[n - 1]);
    if !(lo > 0.0) || !hi.is_finite() {
        return (ld, f64::INFINITY, false);
    }
    (ld, hi / lo, true)
}

/// `log|F|` alone, with no condition number.
///
/// Split from [`logdet_cond`] because the condition number is extra work and
/// most callers throw it away: `log_bf_add` reads the `after` matrix's only,
/// and `log_bf_remove` reads neither. `ok` is exactly the conjunction
/// `logdet_cond` reports, so a caller that skips the condition number still
/// fails closed on a non-positive-definite `F` and on a non-positive diagonal.
pub fn logdet(f: &[f64], n: usize, chol: &mut Chol) -> (f64, bool) {
    if !chol.factor(f, n) {
        return (f64::INFINITY, false);
    }
    let ld = chol.logdet();
    (ld, diag_is_usable(f, n))
}

fn diag_is_usable(f: &[f64], n: usize) -> bool {
    (0..n).all(|i| {
        let d = f[i * n + i];
        d > 0.0 && d.is_finite()
    })
}

/// True when a condition estimate is close enough to [`COND_GUARD`] that the
/// difference between `kappa_1` and `kappa_2` could plausibly change the
/// decision. Measured, this never fires; a caller that sees it should say so
/// loudly rather than trust the estimate.
pub fn near_cond_guard(cond: f64) -> bool {
    cond.is_finite() && cond > COND_GUARD / 10.0 && cond < COND_GUARD * 10.0
}

/// Eigenvalues of a symmetric `n x n` matrix, into `out`, ascending.
///
/// Householder tridiagonalization followed by implicit-shift QL. No
/// eigenvectors: only the extreme eigenvalues are ever read, and accumulating
/// the transformation would triple the cost.
///
/// # Why this exists rather than a norm estimate
///
/// `np.linalg.cond` is a 2-norm condition number, `lambda_max / lambda_min` for
/// a symmetric positive definite matrix. An earlier version of this file
/// estimated `kappa_1` instead, off the Cholesky factor, on the reasoning that
/// `COND_GUARD` is never approached in practice so any nearby quantity would
/// do. That is false, and it was caught by the `06_passes` fixture: on the
/// first crowded field tried, a split proposal with `log BF = +0.457` had an
/// exact `kappa_2` under the guard and a `kappa_1` estimate of 1.42e3 over it,
/// so the port refused an emitter the Python accepted. **A guard that rejects a
/// proposal is a detection rule** [P12]; approximating it is not a free
/// optimization, it is a change to what the pipeline detects.
///
/// At `n <= 37` this is ~15-50k flops, and it runs once per proposal rather
/// than per LM iteration.
///
/// Returns `false` if the iteration fails to converge, which callers must treat
/// as "unusable", never as a small condition number.
pub fn sym_eigvals(a: &[f64], n: usize, out: &mut Vec<f64>, work: &mut Vec<f64>) -> bool {
    debug_assert_eq!(a.len(), n * n);
    out.clear();
    out.resize(n, 0.0);
    if n == 0 {
        return true;
    }
    if n == 1 {
        out[0] = a[0];
        return true;
    }
    work.clear();
    work.extend_from_slice(a);
    let m = &mut work[..];
    let mut e = vec![0.0f64; n];
    let d = out;

    // --- Householder reduction to tridiagonal form ---------------------
    for i in (1..n).rev() {
        let l = i - 1;
        let mut h = 0.0;
        if l > 0 {
            let mut scale = 0.0;
            for k in 0..=l {
                scale += m[i * n + k].abs();
            }
            if scale == 0.0 {
                e[i] = m[i * n + l];
            } else {
                for k in 0..=l {
                    m[i * n + k] /= scale;
                    h += m[i * n + k] * m[i * n + k];
                }
                let f = m[i * n + l];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                m[i * n + l] = f - g;
                let mut ff = 0.0;
                for j in 0..=l {
                    let mut g = 0.0;
                    for k in 0..=j {
                        g += m[j * n + k] * m[i * n + k];
                    }
                    for k in (j + 1)..=l {
                        g += m[k * n + j] * m[i * n + k];
                    }
                    e[j] = g / h;
                    ff += e[j] * m[i * n + j];
                }
                let hh = ff / (h + h);
                for j in 0..=l {
                    let f = m[i * n + j];
                    let g = e[j] - hh * f;
                    e[j] = g;
                    for k in 0..=j {
                        m[j * n + k] -= f * e[k] + g * m[i * n + k];
                    }
                }
            }
        } else {
            e[i] = m[i * n + l];
        }
        d[i] = h;
    }
    e[0] = 0.0;
    for i in 0..n {
        d[i] = m[i * n + i];
    }

    // --- Implicit-shift QL on the tridiagonal (d, e) -------------------
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    for l in 0..n {
        for _iter in 0..50 {
            // Find a small subdiagonal element to split on.
            let mut mm = l;
            while mm + 1 < n {
                let dd = d[mm].abs() + d[mm + 1].abs();
                if e[mm].abs() <= f64::EPSILON * dd {
                    break;
                }
                mm += 1;
            }
            if mm == l {
                break;
            }
            if _iter == 49 {
                return false;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[mm] - d[l] + e[l] / (g + if g >= 0.0 { r.abs() } else { -r.abs() });
            let (mut s, mut c) = (1.0f64, 1.0f64);
            let mut p = 0.0f64;
            for i in (l..mm).rev() {
                let mut f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[mm] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                f = 0.0;
                let _ = f;
            }
            if r == 0.0 && l < mm {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[mm] = 0.0;
        }
    }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    true
}
