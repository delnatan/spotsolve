# Full-frame Gaussian detection: Rust fitting and local decisions

This describes `spotsolve.detect`, separately from the calibrated local
`spotsolve.inference` API and its active development plan.

## Rust implementation boundary

```python
from spotsolve import detect

result = detect(image, sigma=1.2, gain=1.0, impl="rs")
result.fit_sigma  # fitted width per reported emitter, in pixels
```

With the default `slack=(0.70, 2.2)`, every window fit in ADD, SPLIT, REFINE
and PRUNE now uses Rust. Previously, `impl="rs"` was ignored whenever `slack`
was enabled. The Rust variable-width maximum-likelihood optimizer already
existed; it lacked the MAP width penalty and dispatch from these passes.

The port retains the continuous Cauchy width density of `FocusMixtureWidth`,
including its normalizer, an analytic gradient, and its nonnegative
curvature. Class count probabilities are still evaluated by
model selection. Returned `I` excludes the prior; returned `F` includes its
curvature. Otherwise evidence would charge the prior twice or use the wrong
Laplace volume. Uniform width priors add no penalty, including no constant
offset to the optimizer's objective.

Evidence fits retain `tol_obj=1e-8` and 100 iterations; refinement retains
`tol_obj=1e-6` and its 50-iteration budget. Rust now assesses the quadratic
improvement of the actual feasible step after bound scaling, rather than
comparing its improvement against an unscaled step that was never taken.
Variable-width convergence requires a feasible, information-scaled gradient
check at the returned parameters. A tiny step caused by heavy damping is no
longer sufficient to declare convergence. This is a stationarity check, not a
claim of identifiability or a global optimum. Frozen neighbors and the background
shape remain additive contributions to the original observed pixels. No
Python optimizer or prior callback runs inside a native fit.

Variable-width grouping, rendering, proposal construction, evidence, and
pass scheduling remain Python. `slack=None` retains the existing complete
Rust passes. `impl="py"` remains the reference, and is still the API default.
Custom non-flat width prior classes are supported only by the Python fitter;
the native adapter rejects them explicitly.

### The native group engine, opt-in in `detect`

A second native entry point exists alongside the per-fit dispatch above, and
`detect(..., impl="rs", search="groups")` runs it as the frame's only search:
Python supplies FIND's candidates, the background surface and the empirical
priors per epoch, and the engine makes every local decision. It replaces Python-directed window fitting and move
acceptance with one operation that optimizes a whole local group:

```python
from spotsolve import backend, prior

wprior = prior.FocusMixtureWidth(lam_focus, lam_wide, lo, mid, hi, sigma)
engine = backend.get("rs").group_engine(d_e, bmap, sigma, slack=(0.70, 2.2),
                                        k_max=12, wprior=wprior, flux_prior=A_s)
outcome = engine.search_group(positions, amplitudes, sigmas, ids, focus)
```

Rust builds the neighbourhood, generates births, splits and removals from one
incumbent, jointly refits and scores each under ONE configuration score, and
commits the winner atomically. No forced pruning rule exists on either side of
that comparison. `outcome["status"]` is one of `no_improving_proposal`,
`budget_exhausted`, `unresolved_comparison` or `context_rebuild_required`; the
last is a request, and answering it with a rebuilt context is the caller's half
of the transaction. See [the plan](RUST_GROUP_SEARCH_PLAN.md) for the score,
the validity policy, the measured controls, the frame-level comparison and
why the default is still the pass search (runtime, ~100x).

## Why the decisions are local but the schedule uses frame-wide rounds

Each ADD or SPLIT compares K and K+1 on the same local pixels, jointly refitting
the free neighbors and holding the same surrounding sources fixed. PRUNE
compares K and K-1 in the same way. On acceptance, the neighbors' refitted
positions, fluxes and widths are written back. These are already local,
conditional statistical assessments, rather than a count comparison over the
entire frame. The fitted group is bounded by `k_max`, so it can be a subset of
a larger connected crowd.

The outer rounds serve a different purpose. An accepted source changes the
residual and can expose another candidate; a joint refit can make an unresolved
pair easier to split. Background and empirical population priors also change
between growth rounds. After growth, refinement can bring two sources together;
pruning one refits its survivors and changes their next removal assessment.
That is why one growth pass or one pruning pass need not suffice.

Separating growth from removal is a termination safeguard for this particular
search. PRUNE has a forced `A/SE(A) < PRUNE_TAU` removal rule in addition to
the Bayes factor; ADD does not have that rule. Its removal decision therefore
is not always the reverse of the birth decision. Interleaving them can remove
and recreate a source indefinitely. Changed free/frozen neighborhoods and
empirical priors further prevent successive moves from being comparisons under
one fixed objective. Monotone count within each phase prevents this cycle,
while the explicit round budgets bound work. It does not establish local
optimality or guarantee that growth reaches a no-change round.

A group-based search that compares add, split, remove and the current model
together is a sensible redesign. It has since been built --
`rust/spotsolve-core/src/dense_group.rs`, reachable as
`backend.get("rs").group_engine(...)` -- and `detect` runs it with
`search="groups"`, not yet by default; see [the plan](RUST_GROUP_SEARCH_PLAN.md).
It needed:

- One fixed pixel region, halo and set of priors while comparing alternatives,
  with neighbors jointly refitted in each hypothesis.
- Consistent treatment of boundary and degenerate sources in both growth and
  removal, rather than a removal-only override of the evidence.
- A strict improvement rule under that criterion, with affected groups
  revisited when neighboring sources, group membership, background or priors
  change. Position alone is insufficient: flux and width change the halo too.

This first migration step preserves the existing schedule and decision rules,
while developing the variable-width numerical algorithm in Rust. Changing the
schedule is an algorithm change that needs close-pair recovery and residual checks, not just
equal total counts.

Those checks now exist for the group engine and are in the plan's section 6.
The headline for this section's argument: under one score with no forced
pruning rule, a close pair entered as a single merged source recovers 85% of
its members with no false positives across 0.5-2.0 sigma, and no group in that
control ends unresolved. Getting back DOWN from an over-fitted start was the
gate's failure -- the incumbent's own mode sat on a bound -- and is fixed by
integrating the Laplace volume over the prior's support. At frame level the
group search matches the pass search's recall with 22% lower RMSE; what keeps
it opt-in is cost.

## Gradual migration to full Rust ownership

Stage 2 is implemented, has passed its gate, and is opt-in in `detect`: see
the [Rust group-search plan](RUST_GROUP_SEARCH_PLAN.md) for the engine, the
controls it was measured on, and why it has not yet replaced the production
dispatch (runtime).

The target is one production algorithm implemented in Rust. Python reference
outputs are validation evidence, not required answers; mathematical invariants, converged
objective quality, emitter recovery, localization error and residual structure
are the acceptance criteria. Python remains the public array/result interface.
This is the ownership direction for `detect`, not a second calibrated-inference
research plan.

1. **Current slice: variable-width numerical fitting.** Rust owns model/Jacobian
   evaluation, analytic width-prior derivatives, bounded steps and convergence.
   The Python numerical implementation remains an explicit reference only.
2. **Local group inference.** *Built; gate passed; opt-in in `detect`.* `dense_group.rs` owns
   width-aware neighborhoods, joint fits, competing configurations, one
   configuration score and acceptance, in one operation with a reusable
   workspace; Python directs no optimizer invocation inside it. It runs in
   `detect` as `search="groups"`; the default waits on its runtime.
3. **Frame search.** Move candidate finding, background/population updates,
   affected-group scheduling and termination into a native frame operation.
   Revisit groups when positions, fluxes, widths or background change their
   inputs; do not maintain separate Python and Rust production schedulers.
4. **Retire the old production path.** Once native scientific controls and API
   contracts cover the frame operation, make Python a thin adapter. Keep only
   the reference pieces still useful for independent validation.

No Python implementation of a new native algorithm is required. Future native
changes need not preserve the previous optimizer trajectory or source count.

## Performance review and verification

The Fisher accumulation change in `9660b99` interleaves independent sums without
changing each sum's pixel order. It is applicable to both width layouts. The
existing pass-composition fingerprint still passes unchanged after this port;
the prior optimization's measured speedup is recorded in that commit, rather
than remeasured here against a rebuilt old binary.

The new tests exercise objective quality against the frozen ML/MAP fits, the
data-only objective and posterior-curvature contract with a nonzero halo, background-only
fits, malformed native inputs, and complete detection with the Python optimizer
disabled. Both flat and mixture width priors run through all passes. A heavily
damped, nonstationary fit checks that tiny steps cannot falsely certify
convergence. The fixed-width fingerprint remains a regression guard for its
unchanged algorithm, not a parity requirement for new width-aware algorithm work.

Reproduce the native scientific controls after building the release extension:

```sh
python scripts/check_detect_width.py --repeats 3
python scripts/check_detect_width.py --sizes 64 --repeats 3 --impl py rs
```

The default runs Rust alone. `--impl py rs` adds the reference and alternates
backend order for timing. Each arm is assessed against known truth: precision,
recall, localization RMSE, recall within crowded neighborhoods, and positive
and negative residual audit peaks. Agreement between implementations is not an
acceptance criterion. `--densities` and `--spreads` broaden the controls.

The default controls use a Poisson field with density 0.034 emitters/pixel²,
flux 900–1900 electrons, background 5 electrons/pixel, sigma 1.2 pixels and
log-width spread 0.2. Truth is filtered by the same physical reporting-width
band, and matched within 1.2 pixels. These are small development controls, not
an assessment of statistical calibration or a universal throughput guarantee.

The three 64×64 controls (seeds 17–19) isolate the Rust numerical changes against
the initial native MAP port, after aligning its render/model arithmetic:

| Seed | Recall before → after | Precision before → after | RMSE, pixels before → after | Residual positive/negative peaks before → after |
|---|---|---|---|---|
| 17 | 0.900 → 0.911 | 0.942 → 0.965 | 0.330 → 0.311 | 1/0 → 1/0 |
| 18 | 0.857 → 0.878 | 0.977 → 0.989 | 0.304 → 0.285 | 1/1 → 1/0 |
| 19 | 0.830 → 0.840 | 0.963 → 0.975 | 0.224 → 0.214 | 0/0 → 0/0 |

Some crowded hypotheses still exhaust their iteration budget. The new stopping
check reports this honestly; it does not resolve non-identifiability or change
the current evidence policy for unfinished fits. Handling unresolved competing
hypotheses belongs in the native group-decision stage.

Median of three alternating runs on these 64×64 controls, using the rebuilt
release extension on the development machine:

| Seed | Python reference, seconds | Native width fitter, seconds | Speedup |
|---|---|---|---|
| 17 | 4.906 | 0.712 | 6.9× |
| 18 | 4.689 | 0.553 | 8.5× |
| 19 | 4.747 | 0.581 | 8.2× |

Final validation: 99 Python tests passed, one existing slow quadrature test
skipped; 60 Rust tests passed. A further 16 native controls covered 39×39 and
64×64 fields, densities 0.015/0.055, width spreads 0/0.4, and seeds 5/6.
Strong width variation remains difficult: mean recall in the crowded 0.055,
0.4-spread arm was 0.769, with residual structure remaining. These controls
motivate the local group-search work; the numerical port does not complete it.
