# Notes for the Rust port

Implementation practices this Python learned the hard way, which should be
designed in from the start rather than rediscovered. [`README.md`](README.md)
describes *what the algorithm does*; this file is *how to build it*.

**Port target is `gsolve`.** [`BOXSOLVE.md`](BOXSOLVE.md) documents the
superseded box-sequential solver; nothing in it about box tiling, core seams,
`_adopt_orphans` or the BIRTH/DEATH/SPLIT/MERGE move loop should be carried
over. Where a note below was originally measured on `boxsolve`, it is because
the affected layer (`psf`, `lmga`, `evidence`, `patches`) is shared and
unchanged.

Each item says what the lesson is, what it cost here, and what it looks like in
Rust.

---

## 1. "Strictly interior" is an invariant, not a convention

**The single most expensive bug found in this codebase.** `lmga` uses Coleman–Li
affine scaling, which divides by `v_i`, the distance from parameter `i` to the
bound its step is heading toward. The code ended each accepted step with
`theta = clip(theta + delta, lower, upper)`, which parks a parameter *exactly*
on its bound and sets `v_i = 0`.

The failure is not local to that parameter. The fraction-to-boundary rule
computes one scalar step scale from `min_i (bound_i - theta_i)/delta_i`, so a
single stuck coordinate collapses the step for **every** coordinate. Measured on
a 39×39 frame: 68.6% of LM iterations began with a parameter on a bound, 46.5%
of inner trials were scaled to the `1e-8` floor, and 73.2% of all fits burned
their entire 100-iteration budget taking micro-steps while λ ratcheted to ~1e5.
Removing the clip made the frame solve 2.5× faster *and* made the emitter count
reproducible.

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
over ~20 px and an amplitude over ~1e4 e⁻; one absolute epsilon is a different
constraint for each.

Related, and the reason this bug hid for so long: **expect a class of latent
bugs to surface the first time a solver in a pipeline starts genuinely
converging.** Seeds sit on pixel centres. While every fit stalled near its
integer seed, nothing downstream ever saw a position that had actually moved.

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
(`F = (F + F^T)/2`) so the choice stops mattering — cheap at `p ~ 25`, and it
removes a whole class of "why did this change" questions.

## 4. At these sizes the cost is per-call overhead, not FLOPs

Patches are ~200 pixels and `p = 3K+1` is ~25. A Cholesky factor-and-solve at
that size is ~1.3 µs of arithmetic; going through `scipy.linalg`'s validation
wrappers cost 5.2 µs, i.e. **75% of the call was overhead**. Same story for the
model evaluation: the erf/exp arithmetic is tiny next to Python's dispatch.

**Consequence for the port:** most of the micro-optimizations in the current
Python (hoisting `_axes`, in-place diagonal adds, skipping scipy wrappers) exist
to dodge interpreter and wrapper overhead that Rust simply does not have. **Do
not port them as-is and do not assume the Python profile ranks the same.**
Re-profile after the port; the ranking will change — see §10.

What *does* transfer is everything in §5–§8, which is algorithmic.

## 5. Hoist loop invariants across the call boundary, by design

| Hoisted | Was recomputed | Frequency |
|---|---|---|
| the two separable 1-D pixel axes | per model evaluation | ~160× per fit |
| the I-divergence's data-only terms (`d>0` mask, guarded numerator) | per objective evaluation | ~160× per fit |
| the incumbent's model + residual | per split candidate | K+1× per search step |
| the Coleman–Li damping diagonals | per lambda trial | ~2× per LM iteration |
| the incumbent's Fisher log-determinant | per split proposal | `len(SPLIT_DISPS)`× per emitter |

The last one is live in `gsolve._try_split` today: one incumbent is scored
against several proposals, and `evidence.log_bf_add` takes a precomputed
`before=` for exactly that reason. Passing it is exact, not an approximation —
it is the same function of the same matrix.

**In Rust:** give the fit a `FitWorkspace` that owns `ay`, `ax`, `d_pos`,
`d_safe`, the scratch `p × p` matrix and the scratch Jacobian. Borrow it for the
duration of the fit. This makes the hot loop allocation-free as a side effect
(§6) and makes the invariants structural rather than remembered.

## 6. The hot loop should not allocate

The LM inner loop allocated two dense `p × p` matrices per lambda trial, whose
off-diagonals were known to be zero, purely to write `F + diag(u) + lam*diag(v)`.
There are ~700k such trials per frame.

**In Rust:** pre-allocate in the workspace, `copy_from_slice` the base matrix,
and add to the diagonal in place. Aim for zero allocation inside `fit()` after
setup; it is achievable because every shape is known once `K` is fixed.

## 7. Exploit the separable model; think about layout

The pixel-integrated Gaussian factorizes:
`m[i,j] = b + sum_k A_k Ey[i,k] Ex[j,k]`. The 1-D factors are the whole cost;
the outer products are cheap. Keep that.

Two layout points:

- The current code materializes a `(h, w, 3K+1)` Jacobian and fills it through
  **strided** slices (`J[:, :, 2::3] = ...`). That is cache-hostile.
- `J` is only ever consumed as `J^T (W r)` and `J^T (W J)`. Storing it
  **column-major as `(p, n)`** makes both products contiguous and drops the
  transposes. Worth doing from the start (`faer` is column-major natively).

## 8. Do not compute derivatives nobody reads

The one-axis factor routine returned `(E, dE/dc, dE/dsigma)` on all 1.4M calls
per frame. `dE/dsigma` is read **only** by the free-sigma diagnostic, which the
production path never runs.

**In Rust:** separate functions, or a const-generic / trait flag, so the unused
derivative is not merely unread but not emitted. Do not rely on the optimizer to
dead-code-eliminate across an array-allocating boundary — it generally cannot.

## 9. The committed set needs a spatial index, not a `Vec`

`gsolve` is **linear in area at fixed density** — measured µs/px is flat from
62×62 to 128×128 (983 → 1037). One term is not, and it is the only one:

| size | N | `_window` share |
|---|---|---|
| 39 | 84 | 1.2% |
| 62 | 211 | 1.3% |
| 96 | 507 | 2.1% |
| 128 | 901 | 3.0% |

`_window` measures the candidate against **every** committed emitter, so it is
`O(N)` per proposal and `O(N²)` per round — a share growing linearly with area.
It is small at these sizes and it would not be at 512×512. Three other terms
have the same shape: `background_map` masks every emitter's support,
`_split_pass` visits every emitter, and `calibrate.render_model` renders all of
them.

**In Rust: bucket emitters on a uniform grid** with a cell of about
`HALO_FACTOR * sigma`, so a window query is a fixed 3×3 neighbourhood walk
instead of a scan:

```rust
struct Committed { cells: Vec<SmallVec<[Emitter; 8]>>, nx: usize, ny: usize }
```

That makes `_window`'s free/frozen selection an index computation, removes the
per-proposal rebuild of the global emitter array, and gives dirty-set tracking
somewhere natural to live. It is the single data-structure decision that most
changes how the port scales.

## 10. Convergence tolerances belong in decision units

The optimizer's primary stopping test is on the *predicted decrease in the
objective, in nats*, because everything downstream compares I-divergences on the
scale of a log Bayes factor. Gradient- and step-norm tests are absolute and the
natural scale of this problem is set by fluxes running to ~2000 e⁻, so they are
near float64 noise and cannot be the primary test.

There is a subtler trap attached: **a proposal fit starts further from its
optimum than the incumbent it is compared against.** So truncating iterations
does not add symmetric noise — it systematically leaves the proposal's objective
too high and biases model selection toward the smaller model. Measured here:
capping at 40 iterations still left 21% of fits more than 0.1 nat above their
optimum, with a p99 gap of 72 nats. **Resist the temptation to cap `max_iter`
for speed.** Make each iteration cheaper instead.

This matters more than it looks: `lmga.fit` is 83–89% of `gsolve`'s runtime, and
it is the part Rust makes 20–50× faster. Re-profile after the port before
optimizing anything else — the ranking in README §14 will not survive.

## 11. Numerical guards belong where the scale is known

The amplitude floor is relative to the patch's own flux scale (`1e-6 * A_max`),
not an absolute constant. It has to be, because what it protects is a *ratio*:
an emitter's position block of the Fisher matrix scales as `A²`, and an absolute
floor puts those entries at ~1e-13 of the largest diagonal, which is float64
noise.

The general principle, worth preserving: **no test on `F` alone can distinguish
an uninformed parameter from a well-posed matrix in badly scaled units**,
because both produce a huge raw condition number and a small scaled one. So the
guard cannot live in the linear-algebra layer. It must live where the physical
scale is known.

**In Rust:** this is an argument for passing scale context down (a `PatchScale`
struct) rather than writing "clever" scale-free numerical guards deep in the
stack.

## 12. Keep validity conditions out of the search path

**This is the lesson the last audit added, and the easiest one to undo by
accident.**

`evidence::amplitudes_resolved` (`A/SE ≥ 3`) is a statement about *whether the
Laplace approximation can be computed*. It was also being applied as a veto on
every proposal fit, where it silently became the strictest **detection rule** in
the pipeline. Instrumented: `ADD` accepted 96–99% of what `FIND` proposed and
`log BF ≤ 0` fired on **none** of them, while 68–79% of every true emitter lost
inside 2σ died at that guard inside `_try_split`. Removing it gained 1.7 recall
points and improved the pull statistic in all six test cells at no runtime cost.

Two rules for the port:

- **A guard that can reject a proposal before the evidence is evaluated is a
  detection rule, whatever its docstring says.** Type them apart if you can —
  a `Validity` that may only annotate an already-computed evidence, versus a
  `Decision` that may reject.
- **Make removal, not rejection, carry the burden.** `_prune` re-tests the same
  conditions on a *joint* fit, with more information, and can be iterated
  because N strictly decreases. That is the right place for them.

The corollary: the two principled corrections that were built (truncated-normal
`Σ log Φ(A_k/SE_k)`, prior-overflow cap on the Laplace volume) are correct, are
exactly 0 nats where the fit is well determined, and are **worth nothing**
end-to-end. Do not port them. They cost an eigendecomposition per proposal.

## 13. Fixed-size linear algebra wants to be on the stack

`p = 3K + 1` is bounded by `k_max` (12), so every matrix in the hot path is at
most 37×37 and every patch at most a few hundred pixels. Nothing in the fit
needs the heap.

Two things deliberately **not** fixed in Python, which should be done properly
from the start:

- `evidence::amplitudes_resolved` and `gsolve::_amplitude_var` do a full
  `inv(F)` to read `diag(F^-1)[1::3]`, microseconds after the same matrix was
  Cholesky-factorized. A different route to the inverse differs in the last ulp
  and can move a decision at the `PRUNE_TAU` boundary — a real risk for a 0.3%
  gain in Python, but in Rust one factorization should feed `logdet`, the
  amplitude entries of the inverse diagonal, and the scaled condition number.
- The scaled condition number costs a full SVD (`np.linalg.cond`). It exists
  only to be compared against `COND_GUARD = 1e3` and, measured, it never fires.
  Estimate it from the Cholesky factor already in hand.

Use const generics or a small fixed-capacity matrix type so `K` is known at the
type level where it can be, and a workspace (§5) where it cannot.

## 14. Parallelism

`gsolve` has no box lattice to colour, so the `boxsolve` colouring scheme does
not apply. What it has instead:

- **The round loop is inherently sequential.** ADD is Gauss–Seidel by design —
  each acceptance writes back its window's refit, and the next candidate is
  scored against that. `_split_pass`'s ordering is load-bearing for the same
  reason (README §7).
- **`refine` parallelizes cleanly.** Within one sweep, patches are disjoint in
  their free parameters and read only a frozen halo built from the *previous*
  sweep's state. Fan out over `pset`, join, then start the next sweep.
- **`_prune` does not.** It is faintest-first with write-back onto survivors,
  and that cascade is the point.

Two correctness requirements that are easy to get wrong:

- **Determinism is not automatic.** If parallel patch fits `push` into a shared
  list, the resulting array *order* depends on completion order, and downstream
  code is order-sensitive. Write each patch's output into a **pre-assigned
  slot** and concatenate in patch order after the join. Otherwise the
  fingerprint test of §2 will flap and you will not be able to tell a real
  regression from a scheduling artifact.
- Below the patch level, `lmga.fit` is a small dense inner loop that should stay
  single-threaded — at `p ~ 25–37` the join cost exceeds the arithmetic (§4).

## 15. Test strategy that transferred well

Three layers, all worth reproducing:

1. **Property tests on primitives.** Randomized inputs, old vs new
   implementation, assert bit-equality. This localized the Cholesky triangle
   problem in one run after the end-to-end test said only "something changed".
   `proptest` is the Rust equivalent.
2. **A whole-pipeline fingerprint.** Hash of the final arrays plus the audit
   summary. Cheap, and the only thing that catches emergent changes.
3. **Three acceptance tests, not one** (README §12): recall by isolation bin
   (`bench.py`), the pull statistic against an oracle arm (`crlb.py`), and the
   residual audit (`audit.py`). Judging by `N` or by residual spread hides
   exactly the errors that matter, because a missed emitter and an invented one
   cancel in any symmetric spread.

`crlb.py`'s **oracle arm** is the one to make sure survives the port: it runs
`refine` from the true positions at the true N with the true background, so any
deficit can be attributed to the estimator or to the search instead of guessed
at. Without it, "the pull spread is 1.4" is uninterpretable.

## 16. Port against the golden fixtures, bottom up

`python make_fixtures.py` writes `fixtures/*.json` — layers of inputs and
expected outputs, every float at 17 significant digits so it round-trips f64.
Port in that order and assert at each layer; each one is meaningless until the
layer below it passes, and "the emitter count is different" at the top localizes
to nothing.

| fixture | layer | how exactly it can be reproduced |
|---|---|---|
| `01_psf` | model and Jacobian | relative 1e-13. **Not** bit-exact — every libm's `erf` differs, and that sets the floor for everything above |
| `02_lmga` | the bounded optimizer | `I` to 1e-8 **absolute** (nats). Do not assert on `n_iter`: the gain-ratio accept/reject branch is sensitive to the last ulp |
| `03_evidence` | Bayes factor and guards | 1e-10. `antisymmetry_residual` must be **exactly** 0.0 — if it is not, the add and remove paths have diverged |
| `06_end_to_end` | whole pipeline, **built from `gsolve`** | deliberately **not** bit-exact; accept on `N` ±1, precision/recall ±0.03, RMSE ±0.02 px, audit counts ±1 |

`04_score` and `05_boxes` cover `score.py` and the box tiling. **Neither is on
the `gsolve` path** — `05_boxes` can be dropped outright, and `04_score` only
matters if the projected score is revived for SPLIT ranking (README §7). If it
is, the two traps it exists to catch still apply:

- **The score denominator is marginal, not conditional.** `den = g'Wg - u'F^-1u`;
  dropping the `u'F^-1u` term leaves something that still looks like a score
  map and silently destroys the screen.
- **`den` is positive in exact arithmetic and not in floating point.** Clamping
  it at a small *absolute* floor is the worst available choice: `z = num/sqrt(den)`
  then reports a colossal score exactly where the model has the **least**
  information. The floor must be **relative to `g'Wg`** and the degenerate
  answer must be `z = 0`, not `z = inf`.

## 17. Things that are scipy-specific and simply go away

- `scipy.linalg.cho_factor` dispatches to a private batched kernel
  (`_batched_linalg`), so raw LAPACK is *not* bit-compatible with it. The
  current code depends on that private symbol with a public fallback. In Rust,
  call `faer` (or LAPACK) directly and the whole question disappears — but fix
  the triangle convention (§3) at the same time.
- `np.linalg.cond` does a full SVD; see §13.
- `f32` is fine for the Occam term — measured error 3e-6 nats. No separate f64
  path is needed there, and `COND_GUARD` covers the one bad regime.
