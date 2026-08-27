# Notes for the Rust port

A running list of things this Python implementation learned the hard way that
should be designed in from the start in Rust, rather than rediscovered. Written
during the 2026-08-27 efficiency audit; add to it as more is found.

Each item says what the lesson is, what it cost here, and what it looks like in
Rust.

---

## 1. "Strictly interior" is an invariant, not a convention

**The single most expensive bug found in this codebase.** The bounded optimizer
uses Coleman-Li affine scaling, which divides by `v_i`, the distance from
parameter `i` to the bound its step is heading toward. The code then ended each
accepted step with `theta = clip(theta + delta, lower, upper)`, which parks a
parameter *exactly* on its bound and sets `v_i = 0`.

The failure is not local to that parameter. The fraction-to-boundary rule
computes one scalar step scale from `min_i (bound_i - theta_i)/delta_i`, so a
single stuck coordinate collapses the step for **every** coordinate. Measured on
a 39x39 frame: 68.6% of LM iterations began with a parameter on a bound, 46.5%
of inner trials were scaled to the `1e-8` floor, and 73.2% of all fits burned
their entire 100-iteration budget taking micro-steps while lambda ratcheted to
~1e5. Removing the clip made the frame solve 2.5x faster *and* made the emitter
count reproducible.

**In Rust:** make it unrepresentable. A newtype whose constructor is the only
way in:

```rust
/// A parameter vector guaranteed strictly inside its box.
pub struct Interior(Box<[f64]>);

impl Interior {
    pub fn new(theta: &[f64], lo: &[f64], hi: &[f64]) -> Self { /* pull inside */ }
    pub fn step(&self, delta: &[f64], lo: &[f64], hi: &[f64]) -> Self { /* stays inside */ }
}
```

The optimizer then cannot express the broken state. Prefer this to a debug
assertion: the Python version *documented* the invariant in its module docstring
and still violated it in the body for months.

The margin must be **relative** to each bound's own range. A position is bounded
over ~20 px and an amplitude over ~1e4 e-; one absolute epsilon is a different
constraint for each.

## 2. Do not enable fast-math; float associativity is part of the contract

Every refactor here was accepted only if the whole-pipeline output was
bit-identical (SHA of positions, amplitudes, standard errors). That caught two
real mistakes that no tolerance-based test would have:

- `A.diagonal += u + lam*v` is **not** `A.diagonal += u; A.diagonal += lam*v`.
  `(F_ii + u_i) + lam*v_i` and `F_ii + (u_i + lam*v_i)` differ in the last ulp
  and the difference compounds through a 100-iteration fit.
- `(1/s)^2` is not `1/(s*s)`.

Rust's default float semantics do not reassociate, which makes this discipline
*easier* to hold than in C. Keep it that way: no `-ffast-math` equivalent, no
`fast_math` crates, and be careful with anything that auto-vectorizes reductions
(a SIMD sum reassociates by construction — if a sum is order-sensitive, say so).

**Keep the fingerprint test.** Hash the final `(positions, amplitudes, se)` and
assert equality across a refactor. It is the cheapest possible guard on a
numerical pipeline and it repeatedly earned its keep.

## 3. `J^T W J` is symmetric only to within rounding

BLAS computes `F_ij` and `F_ji` as separate dot products with different
summation orders, so they can differ by an ulp. Consequently **which triangle
you factorize changes the answer**, and a Cholesky that reads the upper triangle
disagrees with one that reads the lower.

This cost an hour here: swapping `cho_factor` for a raw `dpotrf` call broke
bit-identity purely because f2py handed the C-contiguous array to LAPACK as its
transpose, flipping which triangle was read.

**In Rust:** pick a triangle, write it down in the type or the function name,
and be consistent. If you build `F` explicitly, consider symmetrizing it once
(`F = (F + F^T)/2`) so the choice stops mattering — cheap at p~25, and it
removes a whole class of "why did this change" questions.

## 4. At these sizes the cost is per-call overhead, not FLOPs

Patches are ~200 pixels and `p = 3K+1` is ~25. A Cholesky factor-and-solve at
that size is ~1.3 microseconds of arithmetic; going through `scipy.linalg`'s
validation wrappers cost 5.2 us, i.e. **75% of the call was overhead**. Same
story for the model evaluation: the erf/exp arithmetic is tiny next to Python's
dispatch.

**Consequence for the port:** most of the micro-optimizations in the current
Python (hoisting `_axes`, in-place diagonal adds, skipping scipy wrappers) exist
to dodge interpreter and wrapper overhead that Rust simply does not have. **Do
not port them as-is and do not assume the Python profile ranks the same.**
Re-profile after the port; the ranking will change.

What *does* transfer is everything in §5-§7, which is algorithmic.

## 5. Hoist loop invariants across the call boundary, by design

These are algorithmic wins and survive the port. Each was worth measuring here:

| Hoisted | Was recomputed | Frequency |
|---|---|---|
| the two separable 1-D pixel axes | per model evaluation | ~160x per fit |
| the I-divergence's data-only terms (`d>0` mask, guarded numerator) | per objective evaluation | ~160x per fit |
| the incumbent's model + residual | per split candidate, and again for birth | K+1x per search step |
| the Coleman-Li damping diagonals | per lambda trial | ~2x per LM iteration |
| both Fisher log-determinants in a removal Bayes factor | computed 4x where 2 suffice | per death/merge proposal |

**In Rust:** give the fit a `FitWorkspace` that owns `ay`, `ax`, `d_pos`,
`d_safe`, the scratch `p x p` matrix and the scratch Jacobian. Borrow it for the
duration of the fit. This makes the hot loop allocation-free as a side effect
(§6) and makes the invariants structural rather than remembered.

## 6. The hot loop should not allocate

The LM inner loop allocated two dense `p x p` matrices per lambda trial, whose
off-diagonals were known to be zero, purely to write `F + diag(u) + lam*diag(v)`.
There are ~700k such trials per frame.

**In Rust:** pre-allocate in the workspace, `copy_from_slice` the base matrix,
and add to the diagonal in place. Aim for zero allocation inside `fit()` after
setup; it is achievable here because every shape is known once `K` is fixed.

## 7. Exploit the separable model; think about layout

The pixel-integrated Gaussian factorizes: `m[i,j] = b + sum_k A_k Ey[i,k] Ex[j,k]`.
The 1-D factors are the whole cost; the outer products are cheap. Keep that.

Two layout points for the port:

- The current code materializes a `(h, w, 3K+1)` Jacobian and fills it through
  **strided** slices (`J[:, :, 2::3] = ...`). That is cache-hostile.
- `J` is only ever consumed as `J^T (W r)` and `J^T (W J)`. Storing it
  **column-major as `(p, n)`** makes both products contiguous and drops the
  transposes. Worth doing in Rust from the start (`faer` is column-major
  natively).

## 8. Do not compute derivatives nobody reads

The one-axis factor routine returned `(E, dE/dc, dE/dsigma)` on all 1.4M calls
per frame. `dE/dsigma` is read **only** by the free-sigma diagnostic, which the
production path never runs.

**In Rust:** separate functions, or a const-generic / trait flag, so the unused
derivative is not merely unread but not emitted. Do not rely on the optimizer to
dead-code-eliminate across an array-allocating boundary — it generally cannot.

## 9. Convergence tolerances belong in decision units

The optimizer's primary stopping test is on the *predicted decrease in the
objective, in nats*, because everything downstream compares I-divergences on the
scale of a log Bayes factor. Gradient- and step-norm tests are absolute and the
natural scale of this problem is set by fluxes running to ~2000 e-, so they are
near float64 noise and cannot be the primary test.

There is a subtler trap attached, worth carrying into the port: **a proposal fit
starts further from its optimum than the incumbent it is compared against.** So
truncating iterations does not add symmetric noise — it systematically leaves
the proposal's objective too high and biases model selection toward the smaller
model. Measured here: capping at 40 iterations still left 21% of fits more than
0.1 nat above their optimum, with a p99 gap of 72 nats. **Resist the temptation
to cap `max_iter` for speed.** Make each iteration cheaper instead.

## 10. Numerical guards belong where the scale is known

The amplitude floor is relative to the patch's own flux scale
(`1e-6 * A_max`), not an absolute constant. It has to be, because what it
protects is a *ratio*: an emitter's position block of the Fisher matrix scales
as `A^2`, and an absolute floor puts those entries at ~1e-13 of the largest
diagonal, which is float64 noise.

The general principle, which the codebase states explicitly and is worth
preserving: **no test on `F` alone can distinguish an uninformed parameter from
a well-posed matrix in badly scaled units**, because both produce a huge raw
condition number and a small scaled one. So the guard cannot live in the
linear-algebra layer. It must live where the physical scale is known.

**In Rust:** this is an argument for passing scale context down (a `PatchScale`
struct) rather than writing "clever" scale-free numerical guards deep in the
stack.

## 11. Test strategy that transferred well

Three layers, all worth reproducing:

1. **Property tests on primitives.** Randomized inputs, old vs new
   implementation, assert bit-equality. This localized the Cholesky triangle
   problem in one run after the end-to-end test said only "something changed".
   `proptest` is the Rust equivalent.
2. **A whole-pipeline fingerprint.** Hash of the final arrays plus the audit
   summary. Cheap, and the only thing that catches emergent changes.
3. **A residual audit as the acceptance test** (`audit.py`). Not summary
   statistics — a per-pixel score test for "is there a PSF-shaped thing here the
   model has not explained", calibrated by measurement rather than from a normal
   table. Judging results by `N` or by residual spread hides exactly the errors
   that matter, because a missed emitter and an invented one cancel in any
   symmetric spread.

## 12. Things that are scipy-specific and simply go away

- `scipy.linalg.cho_factor` dispatches to a private batched kernel
  (`_batched_linalg`), so raw LAPACK is *not* bit-compatible with it. The
  current code depends on that private symbol with a public fallback. In Rust,
  call `faer` (or LAPACK) directly and the whole question disappears — but fix
  the triangle convention (§3) at the same time.
- `np.linalg.cond` does a full SVD. It is used per proposal for the scaled
  condition number. At p~25 an SVD is affordable, but if it ever shows up hot,
  the scaled condition number can be estimated far more cheaply from the
  Cholesky factor already computed.

