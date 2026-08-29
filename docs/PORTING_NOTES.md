# Notes for the Rust port

Implementation practices this Python learned the hard way, which should be
designed in from the start rather than rediscovered. [`README.md`](../README.md)
describes *what the algorithm does*; this file is *how to build it*.

**Port target is `spotsolve`.** It replaced an earlier box-sequential solver,
`boxsolve`, whose code was removed from this repository; nothing about box
tiling, core seams, `_adopt_orphans` or the BIRTH/DEATH/SPLIT/MERGE move loop
should be carried over. Where a note below was originally measured on
`boxsolve`, it is because the affected layer (`psf`, `lmga`, `evidence`,
`patches`) is shared and unchanged.

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

The last one is live in `spotsolve.core._try_split` today: one incumbent is scored
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

`spotsolve` is **linear in area at fixed density** — measured µs/px is flat from
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

This matters more than it looks: `lmga.fit` is 83–89% of `spotsolve`'s runtime, and
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

- `evidence::amplitudes_resolved` and `spotsolve::_amplitude_var` do a full
  `inv(F)` to read `diag(F^-1)[1::3]`, microseconds after the same matrix was
  Cholesky-factorized. A different route to the inverse differs in the last ulp
  and can move a decision at the `PRUNE_TAU` boundary — a real risk for a 0.3%
  gain in Python, but in Rust one factorization should feed both `logdet` and
  the amplitude entries of the inverse diagonal.
- The scaled condition number costs a full SVD (`np.linalg.cond`). It exists
  only to be compared against `COND_GUARD = 1e3` and, measured, it never fires.

  **This section used to end "estimate it from the Cholesky factor already in
  hand". That advice was wrong, and the Rust port proved it.** A Hager–Higham
  `kappa_1` estimate off the factor runs 1.2× (n=4) to 2.2× (n=16) above
  numpy's exact `kappa_2`. On the first crowded field the port ran, a SPLIT
  proposal with `log BF = +0.457` had a true `kappa_2` **under** the guard and
  an estimated `kappa_1` of 1.42e3 **over** it. The port refused a split the
  Python accepted, and because the search is forward-only the refusal was never
  revisited — it cascaded into a different configuration for the whole cluster,
  4 of 86 emitters moving by up to 0.88 px.

  The reason it fires at all is §12's lesson wearing different clothes:
  **`COND_GUARD` can reject a proposal before its evidence is weighed, so it is
  a detection rule, not a numerical guard.** "Measured, it never fires" was
  measured with the exact `kappa_2`; it says nothing about a nearby quantity.
  Approximating a detection rule changes what the pipeline detects.

  **In Rust:** compute the eigenvalues. Householder tridiagonalization plus
  implicit-shift QL is ~15–50k flops at `n <= 37` and matches numpy to 1e-10.
  It runs once per proposal, not per LM iteration, so it does not show up in a
  profile. The general rule: never substitute a different norm for a quantity
  that a guard compares against a fixed threshold.

Use const generics or a small fixed-capacity matrix type so `K` is known at the
type level where it can be, and a workspace (§5) where it cannot.

## 14. Parallelism

`spotsolve` has no box lattice to colour, so the old `boxsolve` colouring scheme
does not apply. What it has instead:

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
   (`scripts/bench.py`), the pull statistic against an oracle arm (`scripts/crlb.py`), and the
   residual audit (`audit.py`). Judging by `N` or by residual spread hides
   exactly the errors that matter, because a missed emitter and an invented one
   cancel in any symmetric spread.

`scripts/crlb.py`'s **oracle arm** is the one to make sure survives the port: it runs
`refine` from the true positions at the true N with the true background, so any
deficit can be attributed to the estimator or to the search instead of guessed
at. Without it, "the pull spread is 1.4" is uninterpretable.

## 16. Port against the golden fixtures, bottom up

`python scripts/make_fixtures.py` writes `tests/fixtures/*.json` — layers of inputs and
expected outputs, every float at 17 significant digits so it round-trips f64.
Port in that order and assert at each layer; each one is meaningless until the
layer below it passes, and "the emitter count is different" at the top localizes
to nothing.

| fixture | layer | how exactly it can be reproduced |
|---|---|---|
| `01_psf` | model and Jacobian | relative 1e-13. **Not** bit-exact — every libm's `erf` differs, and that sets the floor for everything above |
| `02_lmga` | the bounded optimizer | `I` to 1e-8 **absolute** (nats). Do not assert on `n_iter`: the gain-ratio accept/reject branch is sensitive to the last ulp |
| `03_evidence` | Bayes factor and guards | 1e-10. `antisymmetry_residual` must be **exactly** 0.0 — if it is not, the add and remove paths have diverged |
| `04_end_to_end` | whole pipeline, **built from `spotsolve`** | deliberately **not** bit-exact; accept on `N` ±1, precision/recall ±0.03, RMSE ±0.02 px, audit counts ±1 |

Two earlier fixtures, `04_score` and `05_boxes`, covered the projected score and
the box tiling. Neither was on the `spotsolve` path and both were dropped with
`boxsolve`. The score only matters again if it is revived for SPLIT ranking
(README §7); if it is, the two traps it exists to catch still apply:

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

## 18. After the port, re-profile the language you did *not* port

§10 said the ranking would not survive the port. It did not, and the surprise
was not where §10 expected. With all four passes in Rust, a 512×512 frame with
12667 emitters spent **41% of its runtime in `calibrate.robust_background`** —
Python, `scipy`-free, and never once suspected, because on the 39–128 px frames
every measurement had been taken on it was under 2%.

The cause is §9's shape in the host language. `robust_background` masks every
emitter's support with a full-frame boolean sweep, `O(N·H·W)`: at 512×512 that
is 3.3 billion boolean ops per call, six calls per frame. Worse, `_update_bg`
was paying it **twice per round** — once here for the scalar fallback, once in
`background_map` for the surface — on the same positions, the same `sigma`, and
the same radius (`BG_MASK_RADIUS` and `robust_background`'s `radius_factor` are
both 3.0). Two identical masks, neither reused.

Stamping each disc over its own bounding box instead (`O(N·σ²)`) and computing
it **once** per round took `_update_bg` from 25.9 s to 0.070 s, a 370× cut, and
the frame from 64.0 s to 38.2 s. µs/px is flat again out to 512 (96–150 with no
trend from 256 up), which is the property §9 exists to protect. The Python that
stayed behind is now 1.1% of the frame, and the profile reads:

| term | share of a 512×512 frame |
|---|---|
| `refine` (Rust) | 51% |
| `split_pass` (Rust) | 38% |
| `prune` (Rust) | 8.4% |
| `add_pass` (Rust) | 0.9% |
| all remaining Python | 1.1% |

Three lessons, in order of how much they cost:

- **A port moves the bottleneck across the language boundary, not just down
  it.** Profile the host after the port, on a frame large enough for the
  `O(N²)`-shaped terms to show. At 128 px this leak was invisible.
- **The measurement that justified leaving code in Python expires when the
  rest gets 10× faster.** "Per-round, profiles at 0.1–0.3%" was true of the
  *convolutions*, and it is still true of them. It was never true of the mask
  those convolutions needed, which nobody had timed separately.
- **Look for the same quantity computed twice** before optimizing either copy.
  The duplicate here predates the port and was costing the pure-Python path
  just as much.

`calibrate.emitter_free_mask` is now the single definition of that predicate;
`backend.emitter_free_mask` is the same predicate in Rust, and
`PythonBackend` delegates to the Python one rather than keeping a third copy.
Both were checked bit-identical to the original sweep on 200 random cases.

## 19. Cross-language bit-identity is not achievable; say what *is*

§2's fingerprint rule is the acceptance test for a refactor **within** one
implementation, and it still is. Do not extend it across the Python/Rust seam:
`libm::erf` is not `scipy`'s, and a hand-written dot product does not sum in
BLAS's order. Chasing it there wastes time and, worse, invites someone to
"fix" a correct implementation until it matches a wrong-but-familiar rounding.

What to hold instead, measured end-to-end on both real frames:

- **Per fit, ~1e-6 px.** That is what the layer 3 and layer 6 fixtures already
  assert, and it is the right number to assert.
- **Per frame, ~1e-4 px worst case** after three rounds of accumulation — still
  three orders below the CRLB (~0.03 px), and every accuracy column in
  `scripts/bench.py` matches the Python to the last printed digit at all three
  densities.
- **The search itself agrees exactly.** On a 128×128 frame with 906 emitters
  the two implementations produced an identical round trajectory — same N, same
  ADD count, same SPLIT count, every round.

And then the useful part, because a disagreement *will* eventually appear:

> **When the two implementations differ by one emitter, check whether that
> emitter is the closest decision to a threshold in the whole frame. It usually
> is, and that is a pass, not a failure.**

On that 128×128 frame the final counts were 782 (Python) and 783 (Rust). The
one emitter they disagree on was decided at `log BF = +0.0219` — the smallest
`|log BF|` among **3148** weighed prune decisions in that frame, i.e. a
likelihood ratio of 1.02. The same trace found a `A/SE` guard evaluation
sitting `4.5e-6` below `PRUNE_TAU`. Two hard thresholds, two knife edges, in
one frame; a 1e-6 perturbation flips whichever one it reaches.

The trace that establishes this is worth keeping as a tool, not a one-off: wrap
`_prune`'s decision line, record `(log_bf, position)` for every emitter, and
sort by `|log BF|`. It converts "the implementations disagree" — alarming and
unactionable — into "they disagree on the single most marginal call in the
frame", which is the expected outcome and needs no fix. Reserve alarm for a
disagreement on an emitter that is *not* near a boundary; that one is a bug.

## 20. REFINE: what its schedule actually does, measured

REFINE is the pass that produces the reported answer — every position,
amplitude and standard error the pipeline returns comes out of it — and after
§18 it was the largest single cost in the port. Four things were measured, and
three of them contradicted what the code and its comments claimed.

**It is block-Jacobi, not Gauss–Seidel.** `_refine_sweep`'s docstring says
"one Gauss–Seidel pass"; it is not. The halo is built from the sweep's *input*
state and results are written to a separate output array, so every patch in a
sweep reads the same frozen neighbourhood. §14 has this right and the Rust
matches it. Within a patch the emitters are fitted jointly, so the accurate
description is **block-Jacobi with exact block solves**, blocks being the
connected patches. Fix the docstring, not the code — the Jacobi structure is
what makes the pass parallelizable.

**It does not converge, and its stopping rule cannot fire.** `refine` breaks
when `max |Δposition|` over *all* emitters falls below `REFINE_TOL = 1e-3` px.
Measured over 8 sweeps on a 906-emitter frame, 80% of emitters were still
moving more than that, so the break never fires and the loop always runs
`max_sweeps` and stops mid-flight. The reported positions are "wherever sweep 4
landed". Worse, the movement does not decay: max per-sweep movement went
3.01, 4.24, 2.79, 0.82, 0.83, 3.83, 2.58, 2.14 px — and it is not a 2-cycle
either (`|T_k - T_{k-2}|` is comparable to `|T_k - T_{k-1}|`).

The bulk *does* settle; a minority makes sudden discrete jumps (one emitter:
0.04, 0.04, 0.05, **5.65**, 0.62, 0.16 px). Two things drive them, and both are
structural rather than numerical:

- **The patch decomposition is state-dependent.** It is rebuilt every sweep
  from the current positions, and the patch count wanders with it (360, 362,
  358, 359, 363, 355 …). When an emitter drifts far enough to merge or split a
  patch, the joint fit it belongs to changes discontinuously, and so does it.
  A fixed point need not exist, because the map is not continuous.
- **Collapsed pairs.** The worst movers sit at nearest-neighbour distances of
  0.001–1.3 px. Those are what `_prune` exists to remove, which is why the
  pipeline order is settle, prune, settle again.

**More sweeps buy nothing.** End-to-end at 1, 2, 4, 8 and 16 sweeps, across two
densities and both real frames: recall, per-isolation-bin recall, false
positives, median error, pull spread and tail fraction are all flat to the
printed digit, while runtime rises 2×. `REFINE_SWEEPS = 4` is a safety margin,
not an accuracy setting — treat it as one.

**An exact dirty-set memo is worthless.** The obvious optimization is to skip a
patch whose inputs did not change. Instrumented (`PatchCost::reproducible`,
which requires **bit**-identity of the patch's own emitters *and* its frozen
halo, since anything looser is an approximation rather than a memo): **zero**
patches qualified, on any sweep, on any frame. Everything jiggles at 1e-5 px
forever. Do not build the dirty set; the negative result is worth more than
the code.

### The one lever that paid: separate the two objective tolerances

`lmga.fit`/`lmcl::fit` stop on predicted decrease in **nats**, at `tol_obj =
1e-8` for every fit in the pipeline. That number is right for ADD, SPLIT and
PRUNE — §10's asymmetry argument — and six orders tighter than REFINE needs.
Near the optimum `I(t) ≈ I_min + ½ δt' F δt`, so stopping at a predicted
decrease of `tol` leaves a parameter about **`sqrt(2·tol)` standard errors**
short. That is the missing conversion between §10's "decision units" and the
pixels REFINE actually reports, and it is measured, not assumed
(`examples/refine_tol.rs`): at `1e-4` the predicted 0.0141 SE against a p95 of
0.0100.

Two traps, both real:

- **The conversion diverges where `F` does.** The *bulk* follows `sqrt(2·tol)`;
  the worst emitter does not. At `1e-3` a patch with a nearly singular Fisher
  matrix moved **3.6 SE** — a flat direction converts a tiny objective decrease
  into a large parameter move [P11].
- **"REFINE decides nothing" is true of REFINE and false of the pipeline.** Its
  output is what PRUNE re-tests and what the next round's candidates are scored
  against. Loosen it far enough and it moves detections. That is what fixes the
  value, and it is visible as a knee (256 px, N=3169, Python-vs-Rust agreement
  in units of the reported SE):

  | `tol_obj` | frame s | N | max `\|Δpos\|/SE` |
  |---|---|---|---|
  | 1e-8 | 9.78 | 3169 | 0.0010 |
  | **1e-6** | **8.66** | **3169** | **0.0015** |
  | 1e-5 | 7.68 | 3168 | 0.1466 |
  | 1e-4 | 7.08 | 3161 | 0.2828 |

  `N` and the agreement band move *together*, one step apart, which is the
  signature of a numerical tolerance that has begun making decisions [P12].
  `1e-6` is the last value that changes neither, and it still removes ~40% of
  REFINE's LM iterations: REFINE 19.5 s → 14.5 s on a 512×512 frame, with `N`
  identical at 12667 and every `scripts/bench.py` column unmoved.

**The general rule:** one tolerance for the whole pipeline is one too few. Give
each fit the tolerance its *consumer* needs, and find the loosest value by
watching for the point where a detection count starts to move — not by
reasoning about the fit in isolation.

### Making the schedule group-wise, and what that does not fix

REFINE's decomposition is already group-wise **in space**, in the DAOPHOT
NSTAR sense: connected components under a 2.5σ link radius, split at `k_max`,
with everything beyond 5σ frozen into a constant halo. Distant emitters
genuinely do not interact. What was global was the *schedule* — every group
refitted every sweep, convergence tested as one max over all emitters.

Scheduling on the same locality is the obvious fix and it is now implemented: a
group is refitted only when an emitter it reads — its own, or one in its frozen
halo — moved more than `tol` in the previous pass. It is correct, it costs
nothing, it replaces a break that could not fire with a queue that can drain,
and it is worth about 3–4%. **It does not drain on a crowded field**, and the
three things that would supposedly make it drain were each measured and each
failed:

- **The churn is not self-inflicted.** The appealing theory is that refitting a
  settled group anyway moves it ~1e-5 px, which dirties its neighbours forever.
  Wrong: at a threshold of 1e-1 px — a third of the CRLB, far above optimizer
  noise — a 42-group frame still refitted 40 groups on pass 8. Groups move
  genuinely, every pass.
- **Pinning the decomposition is actively harmful.** Rebuilding the grouping
  from the positions it is changing is what makes the map discontinuous
  (§20 above), so holding it fixed for the call is tempting: it stabilizes the
  group count and cuts 11% of the LM iterations. It also broke the frame. Over
  four passes emitters drift across link radii, and a stale grouping puts two
  now-adjacent emitters in **separate** groups, each frozen in the other's
  halo — fitting each as though the other were a constant. Measured on 256×256:
  **8 missed emitters and a residual score peak of |z| = 129**, against 0 and
  5.6 with the rebuild. The rebuild is not churn; it is the decomposition
  tracking the configuration. Caught only by the audit, which is why the audit
  and not `N` is the acceptance test.
- **Gauss–Seidel does not converge faster here.** Reading neighbours already
  updated in the same pass is the textbook 2× on a block-Jacobi iteration.
  Measured: groups fitted per pass 34, 40, 41, 42, 40, 40, 43, 42 — no better
  than Jacobi, and slightly *more* LM iterations (3827 vs 3597). It is
  available as `REFINE_GAUSS_SEIDEL`, and it is `false`, because it buys
  nothing and would cost the order-independence §14 relies on.

The reason all three fail is one fact: **REFINE's residual motion is not slow
convergence of a well-posed iteration.** It is a handful of degenerate groups —
collapsed pairs at nearest-neighbour distances of 0.001 px — that have no
stable answer at all, and whose dirtiness percolates through overlapping halos.
No relaxation scheme fixes a configuration that is not identifiable. `_prune`
is the fix, which is exactly why the pipeline order is settle, prune, settle.

**The transferable lesson:** before optimizing an iteration's schedule,
establish that what it is doing is converging. Here three standard accelerations
were tried against something that was never a convergence problem, and the
measurement that would have said so first — "does the queue drain at a
threshold far above the noise floor?" — takes ten minutes.

### Still open

- `it max` reaches the 200-iteration cap with `converged = false` on roughly
  one patch per sweep. These are the collapsed pairs; they are removed later by
  PRUNE, but they are also the single most expensive fits in the pass, and they
  are the source of the percolation above. Capping *their* budget, or detecting
  non-identifiability and handing them straight to PRUNE, is the one schedule
  change the measurements actually point at.
- REFINE is now ~43% of a 512×512 frame, roughly tied with SPLIT. It is the one
  pass that parallelizes cleanly (§14), and that is the next lever — the Jacobi
  structure above is exactly the property that makes it safe.

## 21. Non-identifiability: four detectors that do not work, and what does

§20 left one item open: a few REFINE groups run the full iteration cap without
converging and cost 10–19% of the pass. They are *not* stalled — they keep
taking accepted steps to the very last iteration — so they are descending a
direction the data carries almost no information about. The right instinct is
to detect that and stop: zero information means nothing to gain, and paying
200 iterations for it is waste.

**The safety fact that shapes the whole design, measured first:** the emitters
in those groups are **not** disposable. 25 of 28 (128×256 px) and 36 of 38
(256×256) survive to the final output. They are real localizations that happen
to sit in a crowded group whose *joint* fit has one flat direction; the other
3K parameters are well determined and settle early. So "give up" cannot mean
abandoning them, and any scheme that reports garbage for them is wrong.

Four candidate detectors were measured against the groups that actually stall.
All four fail:

| detector | why it fails |
|---|---|
| scaled condition number of `F` | Does not separate. A fit at `cond = 2.55e14` converged in **10** iterations; one at `cond = 1.62e5` burned 200. Every multi-emitter REFINE group runs `cond` of 1e9–1e14, so `COND_GUARD = 1e3` would reject essentially all of them. |
| closest pair in the group | Does not separate. 0.764 px (0.64σ) failed to converge; 0.017 px converged in 10 iterations. |
| a parameter on its bound (§1's Coleman–Li crawl) | Does not separate. Converged fits sit at a relative bound distance of 1e-10 routinely, and one non-converged fit had nothing near a bound. |
| emitter pinned at the amplitude floor | Real, and the *right* notion of zero information — but almost absent at scale. 0 of 905 emitters at 128 px, 1 of 3677 at 256 px (and that one was correctly pruned). It looked promising only on the 39×39 fixture, which is not representative. |

The reason none of them work is that the property is **temporal, not
structural**: what these fits share is that they were still descending when the
budget ran out, and no static feature of the configuration predicts that.

### So the answer is a budget, and it is boundable

`REFINE_MAX_ITER = 50`, down from 200, in both languages and checked by
`backend.py`. Two things make this a principled stop rather than a shrug:

- **What the truncated tail is worth can be bounded in the units that matter.**
  An iteration continues only while it predicts a decrease above
  `REFINE_TOL_OBJ`, and §20's conversion says a decrease of `t` nats moves a
  parameter about `sqrt(2·t)` standard errors. The whole abandoned tail is
  therefore worth a small fraction of one SE — measured at 0.002 SE.
- **The evidence fits keep 100.** They start further from their optimum and
  their objective *is* differenced into a Bayes factor, so truncating them
  biases model selection toward the smaller model [P10]. This budget is for the
  polish step only. That distinction is the same one §20 drew for `tol_obj`,
  and it is the second time the answer has been "one knob for the whole
  pipeline is one too few".

Measured end-to-end at 256×256, against the same run at 400 iterations:

| `max_iter` | frame s | N | audit | max `\|Δpos\|/SE` |
|---|---|---|---|---|
| 25 | 7.24 | 3166 | clean | 0.0820 |
| **50** | **7.80** | **3166** | **clean** | **0.0023** |
| 100 | 8.19 | 3168 | clean | 0.0006 |
| 200 | 8.73 | 3169 | clean | 0.0077 |
| 400 | 9.00 | 3168 | clean | 0.0006 |

Note there is **no knee here**, unlike `tol_obj` in §20 — `N` wanders by ±3 at
every budget, 400 included, and the audit is clean throughout. That is the
marginal-decision churn of [P19], not a degradation, and it is why the decision
rests on the agreement band and the bound above rather than on `N`.

Result: REFINE 14.5 s → 11.4 s on a 512×512 frame (19.5 s → **11.4 s**, 42%,
across §20 and this section together), the Python-Rust agreement band
*tightened* to 0.002 SE, and `scripts/bench.py` moved slightly the right way
(median error 0.1409 → 0.1396, pull spread 1.44 → 1.42, false positives
1.83 → 1.67).

**The transferable lesson:** when something is expensive and you believe it is
expensive *because* it is degenerate, measure whether the degenerate thing is
disposable before designing around it. Here the natural design — detect
non-identifiable emitters and drop them — was ruled out in one measurement by
the fact that 90% of them are real detections that survive to the output. Every
detector tried afterwards was a search for a structural signature of a
temporal property, and the instrumentation that proved it (`PatchCost`'s
`scaled_cond`, `min_pair_px`, `min_bound_frac`, `n_at_floor`, all recorded only
when measuring) is worth keeping precisely because the results were negative.
