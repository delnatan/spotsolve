# Rust-owned local group search

Status: **implemented; the release gate passes; `detect` runs it opt-in.**
`rust/spotsolve-core/src/dense_group.rs` owns local neighbourhood
construction, proposal competition, fitting, scoring and acceptance, and
`spotsolve_rs.DenseGroupEngine` exposes it as one operation with no Python
numerical or decision callback inside a transaction.

The section 3 gate first failed on unscorable boundary incumbents and now
passes with a box-truncated Laplace volume, validated against importance
sampling. `detect(..., impl="rs", search="groups")` runs the engine as the
frame's only search (section 5); the default is still `search="passes"`, and
deliverable 5 -- retiring the pass dispatch -- waits on runtime, which is
~100x the pass search (section 6).

This document is the implementation checklist for stage 2 of
[DENSE_DETECT.md](DENSE_DETECT.md), and now also the record of what it
measured.

## Objective and scope

Replace Python-directed window fitting and move acceptance with one native
operation that optimizes a local group of interacting emitters. Rust constructs,
fits, assesses and accepts competing configurations containing additions,
splits and removals. Python supplies frame data and settings, and adapts results.

Rust is the algorithm implementation to develop. Python results are comparison
data, not required answers. Do not reproduce Python optimizer trajectories,
vectorization patterns, source ordering or bit patterns as an acceptance target.

This step concerns the variable-width Gaussian model used by `spotsolve.detect`.
It does not replace the calibrated PSF model in `spotsolve.inference`, claim
calibrated source-presence probabilities, or change that API's uncertainty
contract. The [focused-emitter development plan](FOCUSED_EMITTER_PROPOSAL.md)
remains the plan for that calibrated model.

The frame-level candidate map, background estimation and empirical population
updates may remain Python during this step. They must not direct individual
fits or decide individual moves on the new native path. Full Rust frame
ownership is the following stage.

## Starting point and code boundaries

| Existing code | Use in this stage |
|---|---|
| `rust/spotsolve-core/src/lmcl.rs` | Reuse variable-width fitting, analytic width penalty, feasible-step assessment and stationarity checks. |
| `rust/spotsolve-core/src/psf.rs` | Reuse Gaussian means and derivatives; render frozen neighbors at their own widths. |
| `rust/spotsolve-core/src/grid.rs`, `patches.rs` | Reuse indexing and geometry primitives; add explicitly width-aware group contexts. |
| `rust/spotsolve-core/src/moves.rs` | Reuse geometric proposal ideas; add width-aware constructors without statistical decisions. |
| `rust/spotsolve-core/src/evidence.rs`, `linalg.rs` | Reuse factorization utilities; introduce configuration-level scoring and validity rather than copying asymmetric move guards. |
| `src/spotsolve/core.py` | Retain frame preparation temporarily; retire `_try_add`, `_try_split`, `_prune` and Python-directed group fitting from the new native path. |
| `src/spotsolve/prior.py`, `evidence.py` | References for formulas, defaults and known limitations; no production callbacks from Rust. |
| `rust/spotsolve-py/src/lib.rs`, `src/spotsolve/backend.py` | Expose one group operation, replacing per-fit dispatch for the new path. |

Proposed new modules: `rust/spotsolve-core/src/dense_group.rs` for group state,
context, search and outcomes, and `rust/spotsolve-py/src/dense_group.rs` for
bindings. Keep prior/scoring additions in small dedicated sections or modules
if needed; avoid a second copy of the fixed-width `passes.rs` implementation.

## 1. Define the native transaction and result contract

- [x] Introduce stable emitter IDs, separate from array positions. A deletion
  must not change the identity of another emitter or invalidate its references.
- [x] Define `GroupContext`: frame region and origin, original observations,
  background shape, frozen emitters/halo, free emitter IDs, admissible position
  and width bounds, candidate seeds, and a fixed prior snapshot.
- [x] Define `GroupState` with positions, fluxes and widths, including modeled
  nuisance sources outside the reporting band. Filtering those sources out
  during search would put their light back into the residual.
- [x] Define a reusable `GroupWorkspace` for fits, rendering, proposals and
  matrix factorizations. Trial configurations do not allocate full frame arrays.
- [x] Return a `GroupOutcome` containing the committed configuration, changed
  and removed IDs, conditional uncertainty/status, accepted-move trace and a
  termination reason. Keep fit status separate from search status.
- [x] Search statuses distinguish `no_improving_proposal`, `budget_exhausted`,
  `unresolved_comparison` and `context_rebuild_required`. Do not call a capped
  search converged or a proposal-local result globally optimal.

Expose diagnostics useful for development: objective/prior/volume contributions,
stationarity, validity reason, fit/restart counts and work spent per move type.
Normal result adaptation should not require Python to interpret these to decide
whether a move is accepted.

## 2. Fix the spatial and statistical comparison context

- [x] Build groups and halos using width-aware support, including the maximum
  widths and positions allowed during the transaction. A source that widens
  must not escape the region on which its alternatives are being compared.
- [x] Hold the pixel region, observations, halo, background parameterization,
  parameter support and prior snapshot fixed while comparing configurations.
  Never compare objectives evaluated on different cropped regions.
- [x] Include nearby candidates in context construction before searching. If a
  new proposal needs a larger region or another free neighbor, request a rebuilt
  context and refit/re-score all hypotheses there.
- [x] Give the incumbent and every alternative the same free neighbors. Sources
  outside the transaction remain frozen in every hypothesis.
- [x] Treat `k_max` as a compute limit, not a declaration of statistical
  independence. Leave space for K+1 alternatives. Oversized or strongly coupled
  neighborhoods must produce an explicit status or a documented overlapping
  block solve; do not silently discard the emitter being tested.
- [x] Audit linear algebra capacity for `4*K+1`, including the temporary K+1
  model. `linalg::P_MAX` currently describes `3*K_MAX+1`; check actual buffer
  growth and every fixed-capacity consumer before reusing them.

The first implementation can use conservative support geometry. Approximating
weak couplings for speed comes after residual and edge controls establish that
the omitted light is negligible under the chosen model.

## 3. Establish one configuration score and a symmetric validity policy

This is a prerequisite for replacing global growth/prune scheduling, not an
optional cleanup after porting the loops.

- [x] Implement native serializable prior settings for the existing exponential
  flux prior, uniform width prior and focus/wide width model. Compute their
  normalization and derivatives in Rust. Reject unsupported custom priors
  explicitly instead of calling Python or silently changing the prior.
- [x] Define one score for a fitted configuration in a fixed context. On regular
  hypotheses, start from the existing approximate Laplace expression:

  ```text
  score = -data_I + log_configuration_prior
          + parameter_count/2 * log(2*pi) - logdet(curvature)/2
  gain(candidate, incumbent) = score(candidate) - score(incumbent)
  ```

  Include count, position, flux and width terms exactly once. Derive count and
  labeling factors for the chosen unordered-emitter representation; preserve
  the area cancellation only where its assumptions hold. Score the entire
  free configuration because neighbors can change width class during refitting.
- [x] Make continuous fitting and scoring use the same continuous priors. The
  current flux prior is charged in evidence but omitted from the fit: an
  exponential MAP fit needs the amplitude gradient `1/A_s`, even though its
  curvature is zero. Validate this deliberate algorithm change separately.
- [x] Handle focus/wide class changes as discrete alternatives. Class count
  terms can jump at the width boundary; they cannot be optimized by pretending
  the objective is globally smooth. Refit affected class alternatives on their
  admissible width intervals and assess boundary cases explicitly.
- [x] Specify which curvature supports the score. The current fitter returns
  expected information plus clipped prior curvature, not an exact observed
  posterior Hessian. Do not call this exact evidence. Keep optimizer damping
  out of statistical curvature and uncertainty.
- [x] Apply the same score-validity rules to incumbents and proposals. Boundary,
  singular and nonstationary fits return a reasoned unsupported status; neither
  determinant failure nor large uncertainty is automatically evidence for a
  smaller or larger model.
- [x] Do not copy `PRUNE_TAU` as a removal-only override, and do not simply move
  it to a birth veto. Its earlier recall failures are documented in the code.
  Similarly, do not blindly reuse the fixed-width condition threshold as a new
  width-aware detection threshold.

Initial unsupported-case behavior is explicit: try the prescribed continuation
or restart budget, then return `unresolved_comparison` without inventing odds.
An unsupported incumbent does not get an arbitrary finite score or automatic
replacement. Exactly redundant representations can be canonicalized only with
an explicit mean/prior contract; near-coincident real sources cannot be deleted
as a numerical cleanup.

**Release gate:** measure how often this conservative policy leaves groups
unresolved, especially collapsed pairs and faint neighbors. If regular Laplace
scoring cannot handle a material fraction of the controls, the next required
substep is a validated boundary-aware comparison method. Reinstating asymmetric
forced pruning does not satisfy this gate. The local engine can be developed
and tested before this gate passes, but cannot replace production group search.

### Gate result: PASSED with a box-truncated Laplace volume

The first measurement failed the gate (below). The substep it required -- a
validated boundary-aware comparison -- is `dense_group::log_box_mass`: the
same quadratic model of the log posterior, integrated over the prior's
SUPPORT instead of over all of `R^p`. Each coordinate is truncated in its
marginal to its admissible interval (widths to their class interval,
positions to the transaction's box, amplitudes to `A > 0`), with the fit's
KKT multiplier at an active bound:

```text
score = -I + log p(config) + p/2 log 2pi - logdet(F)/2 + sum_q log P_q
P_q   = exp(s^2 g^2 / 2) * (Phi(u2) - Phi(u1)),  s = sqrt((F^-1)_qq)
```

Exact for any number of interior coordinates plus one active bound; the
correlation between several active bounds is ignored. It is one rule on both
sides of every comparison, and `BoundaryMode` no longer exists as an
unsupported status.

The diagnosis that led there, from the failing run: of the unresolved
incumbents, the redundant emitter sat on `sigma_lo` in 3 of 8 overfit cases,
a ~30 e- phantom sat on the position box in the other 5, and 6 of the 9
degenerate cases were a real source narrower than the model space allows.
The same defect had a second face the gate could not see: an emitter at the
amplitude floor has position/width curvature `~A^2`, so `-logdet/2` credited
it with a Gaussian many box-widths wide -- two overfit trials ACCEPTED splits
to 0.05 e- at +12 and +15 nats. A diagonally scaled condition number cannot
see a block that shrinks as a whole.

**Validated on its own referee**, `scripts/measure_box_laplace.py`:
importance sampling of the exact Poisson posterior times the exact
configuration prior over the same support (8 trials, error in nats against
the referee, rows with effective sample size >= 200):

| configuration | regular Laplace | box-truncated |
|---|---|---|
| isolated in-focus source | 0.024 | 0.024 |
| source narrower than `sigma_lo` (mode on a bound) | **3.05** | **0.05** |
| faint-emitter removal gain | +1.6 | +2.0 |

The last row is a limitation of ANY single-mode Laplace, not of the box: a
weak emitter's posterior is a ridge -- it can share its neighbour's light --
and the referee's weight sits 1-2 px from the neighbour at 40-90 e-, not at
the local mode. Both expressions under-count the larger model and so
overstate removal, by 1.6-2.0 nats. That is a bias against faint neighbours,
recorded rather than corrected.

**Controls after the change** (`check_group_search.py --trials 20`):
unresolved is 0.00 on twelve of thirteen controls. Overfit exact count
0.45 -> 0.95, degenerate 0.85 -> 1.00, underfit 0.80 -> 1.00, empty region
0.90 -> 1.00. The oversized crowd went 0.65 -> 0.10 unresolved but its recall
FELL, 0.79 -> 0.70: before, the search refused to run; now it runs, and it
removes ring members spaced 1.2 sigma apart at gains of +4 to +9 nats, of
which the box term contributes 0.02-1.5. That is the `Exp(1/A_s)` prior's
preference for one bright emitter over two, accepted as the operating point.

### The first measurement: NOT PASSED

Measured by `scripts/check_group_search.py --trials 20`; the full table is in
section 6. The named worry -- collapsed pairs -- is fine: **close pairs entered
merged leave 0% of groups unresolved**, and eight of the thirteen controls are
at 0%. Three are not:

| control | unresolved | of which the INCUMBENT was unsupported |
|---|---|---|
| initial overfit | 0.40 | 0.40 |
| degenerate (zero flux / coincident / below the width bound) | 0.45 | 0.45 |
| oversized crowd (14 coupled sources, `k_max` 12) | 0.65 | 0.65 |

Every unresolved case is an unsupported **incumbent**, and every one of those
is `boundary_mode`. Diagnosed directly on the overfit control: 8/20
transactions end unresolved, all `boundary_mode`, and the active bound is
`sigma_lo` -- three emitters stacked on one true source squeeze the redundant
one to the narrowest admissible width. That mode is genuinely on the edge of
the admissible set, so the Laplace integral around it is one-sided and the
regular expression in section 3 does not apply to it. Refusing to score it is
correct; being unable to score it 40% of the time on an ordinary control is a
material fraction.

So the gate's own conclusion applies: **the next required substep is a
validated boundary-aware comparison method** -- a Laplace expansion that
accounts for an active bound rather than one that declines to expand. What it
must NOT be, per this section, is `PRUNE_TAU` moved to the other side of the
comparison: the incumbent here is exactly the configuration a forced removal
rule would delete, and deleting it because its Laplace volume is hard to
compute is the inconsistency this stage exists to remove.

Two things the measurement changed along the way, both recorded because they
were not obvious:

* **Restarts have to move.** Ten starts 1% apart on a crowded K=3 group reached
  ten different converged optima spread 5.0 nats, and re-running each at 3000
  iterations with `tol_obj = 1e-14` moved none of them. An escalation that only
  iterates harder re-certifies the basin it already had. `perturbed_start`
  steps each parameter by one conditional standard error instead, and the
  incumbent -- the baseline every gain is measured against -- always gets it.
  On the degenerate control that moved unresolved from 0.70 to 0.45 and exact
  counts from 0.60 to 0.85.
* **A rebuild request is half the contract.** A group that reports
  `context_rebuild_required` has not failed; it has said the region it was
  comparing in is the wrong one. Honoring it moved the underfit control's
  recall from 0.45 to 0.82 and its RMSE from 0.33 px to 0.10 px, with no
  change to the engine at all.

## 4. Implement local proposal competition in Rust

- [x] Refit the incumbent at the transaction's prescribed accuracy. Retain the
  best valid evaluated state; a failed restart must not overwrite it.
- [x] Generate all move types from that same incumbent: residual births,
  width-aware quadrupole splits at multiple initial separations, and individual
  removals. A merged-pair seed may initialize a removal hypothesis; it must not
  acquire a separate acceptance rule.
- [x] Include alternative narrow and broad starts where widening can hide a
  neighbor. Seeds help explore the same model; they do not define its score.
- [x] Jointly refit every candidate configuration, including surviving neighbors.
  Reuse the incumbent fit and its factorization when valid for this context.
- [x] Use symmetric fit budgets and a common escalation policy for comparisons
  that remain unsettled. Non-convergence is not evidence against the alternative
  that happened to receive a worse initialization.
- [x] Select the best valid improvement across move types, then atomically
  commit all affected parameters. Keep the incumbent on a numerical tie.
- [x] Regenerate proposals after acceptance; do not reuse a stale residual or
  stale neighbor fit. Permit removal followed by splitting, or the reverse,
  when each is supported by the same comparison rule.
- [x] End when no generated proposal improves the score, comparison validity
  is unresolved, context needs rebuilding, or an explicit work budget expires.

Use an improvement tolerance tied to measured numerical score resolution,
distinct from a scientific detection threshold. Strict score increases prevent
an exact reversal within one fixed context; they do not prove finite convergence
over a continuous state space. Retain explicit move/fit budgets and trace reasons.

## 5. Add one native operation and temporary frame integration

- [x] Add an internal prepared group engine with owned observation buffers and
  reusable storage. Conceptually:

  ```text
  outcome = engine.search_group(context, incumbent, settings)
  ```

  The names are proposed API names, not currently available functions.
- [x] Validate arrays/settings at the binding, copy inputs once per group or
  prepared frame, release the GIL for search, and return owned result arrays.
  Separate prepared instances must not share mutable solver state.
- [x] Expose the operation first to direct scientific controls. Do not build a
  second Python implementation of this new search to test it.
- [x] After the statistical gate passes, replace the variable-width native
  branch's separate add/split/prune calls with group transactions. Python may
  supply candidate maps and prior/background snapshots between frame epochs;
  Rust owns local neighborhood construction, fitting and acceptance.
- [x] Invalidate affected group contexts after changes to source position,
  amplitude, width, membership, background or priors. Use explicit context
  versions; never compare cached scores across different contexts.
- [x] Keep frame-level no-change checks and work budgets explicit during the
  transition. Overlapping group scores are conditional scores; their sum is
  not a frame evidence, and improvement of one does not prove global monotonicity.

The temporary Python layer invokes groups, not individual moves. Do not retain
the old forced-pruning pass after native group search: that would reintroduce
the inconsistency this stage is intended to remove.

### The frame schedule, as built

`core._group_epoch`, called from `detect`'s epoch loop when
`search="groups"`. Its principle is a division of labour: **Python decides
where to look; Rust decides what exists.** There is no add pass, no split
pass, no refine and no prune on this path.

* An epoch holds one snapshot -- background surface, `A_s`, class rates --
  and the engine is re-snapshotted (version bumped) before each.
* Every FIND candidate is a focus, then every emitter. An emitter is settled
  once it was in the free set of a transaction this epoch; a commit
  unsettles emitters within the link radius of anything it changed.
* Light a transaction reports it could not reach becomes a focus of its own.
* A rebuild is answered only for what the transaction HOLDS: a source on its
  position box. Two triggers were removed because a rebuilt context could
  not satisfy them: a frozen neighbour within the link radius of a free
  emitter (the free set is chosen by distance to the focus, so the rebuild
  freezes it again -- it ended 71 of 80 transactions on a 39x39 frame), and
  unreachable light (which the reported peaks now answer by looking there).
* The frame is settled by an epoch that commits nothing under its snapshot,
  so every reported emitter's standard errors come from a transaction in the
  final snapshot.

`tests/test_dense_group.py` runs a whole `detect(search="groups")` with every
Python fitter, evidence, move, refine and prune routine replaced by one that
raises.

## 6. Validation and rollout

Freeze controls and comparison criteria before tuning search settings. Use the
current Rust fitter path as the development baseline and hold out additional
simulation seeds. Exact Python counts, ordering and trajectories are not gates.

| Control | Required assessment |
|---|---|
| Empty region; isolated in-focus source | False additions, count selection, objective quality and localization error. |
| Equal/unequal close pairs across separations and fluxes | Neighbor recovery, count errors, localization RMSE and failed/ambiguous decisions. |
| Broad source; narrow source beside broad source | Splitting broad light into false emitters, swallowed neighbors and class changes. |
| Zero/near-zero flux; coincident sources; active width/position bounds | Explicit unsupported statuses and absence of spurious infinite evidence. |
| Sloped background, frozen bright neighbor, frame edge | Halo correctness, no double counting and residual structure. |
| Oversized connected crowd; emitter crossing a group boundary | Capacity behavior, context rebuilding and stable source identities. |
| Initial overfit, underfit and alternative seed orderings | Recovery from either direction, budget usage and search sensitivity. |

- [x] Unit tests: prior derivatives/normalization; configuration score components;
  finite score antisymmetry; coordinate conversion; proposal flux bookkeeping;
  atomic commit and context invalidation.
- [x] Numerical tests: returned data-only objective, statistical curvature
  contract, stationarity, and fair handling of exhausted fit budgets.
- [x] Search tests: every committed move improves the same context's score;
  no immediate scored inverse is accepted; K=0 works; context changes discard
  incompatible cached scores; capped searches report their status.
- [x] Integration tests: disable Python fitting, evidence and move constructors
  while exercising the native group path. Python callbacks must not be needed.
- [x] Scientific report: precision/recall, close-neighbor recall, localization
  error, broad-source over-splitting, residual peaks, unresolved-group rate and
  uncertainty coverage where uncertainty is supported. Report results by case,
  not only averages or detection counts.
- [x] Runtime report: work and time per group, fit/restart counts, tail latency,
  peak memory and unresolved cases. Optimize after scientific checks pass.

Keep the existing test suite green where its contracts remain applicable.
Replace trajectory or fingerprint requirements only for deliberately changed
algorithms, with scientific controls; preserve unrelated fixed-width and
calibrated-inference behavior.

### Results

`python scripts/check_group_search.py --trials 20`, on a release build. Each
control fixes its own frame, truth and ENTRY configuration; a transaction that
reports `context_rebuild_required` is answered with a rebuilt context, up to
six times, because that is the caller's half of the contract. Matching radius
1.2 px.

| control | recall | precision | RMSE px | exact count | unresolved | rounds | fits | median ms |
|---|---|---|---|---|---|---|---|---|
| empty region | -- | 0.00 | -- | 0.90 | 0.00 | 1 | 12 | 0.7 |
| isolated in-focus source | 1.00 | 1.00 | 0.040 | 1.00 | 0.00 | 1 | 26 | 4.8 |
| equal close pair, entered merged | 0.85 | 1.00 | 0.219 | 0.70 | 0.00 | 1 | 54 | 13.0 |
| unequal close pair, entered merged | 0.85 | 1.00 | 0.306 | 0.70 | 0.00 | 1 | 54 | 9.9 |
| broad source | 1.00 | 1.00 | 0.077 | 1.00 | 0.00 | 1 | 26 | 3.8 |
| narrow source beside broad | 1.00 | 1.00 | 0.103 | 1.00 | 0.00 | 2 | 90 | 24.0 |
| degenerate (zero flux / coincident / sub-bound width) | 1.00 | 0.87 | 0.059 | 0.85 | 0.45 | 1 | 56 | 25.1 |
| sloped background | 1.00 | 1.00 | 0.065 | 1.00 | 0.00 | 1 | 26 | 4.0 |
| frozen bright neighbour | 1.00 | 1.00 | 0.038 | 1.00 | 0.00 | 1 | 26 | 3.9 |
| frame edge | 1.00 | 1.00 | 0.052 | 1.00 | 0.00 | 1 | 24 | 2.5 |
| oversized crowd, 14 vs `k_max` 12 | 0.79 | 0.90 | 0.611 | 0.60 | 0.65 | 1 | 4 | 227.6 |
| initial overfit (3 entered, 1 true) | 1.00 | 0.45 | 0.263 | 0.45 | 0.40 | 1 | 98 | 77.8 |
| initial underfit (0 entered, 3 true) | 0.82 | 0.92 | 0.103 | 0.80 | 0.00 | 3 | 150 | 95.5 |

Read by case, which is the point of reporting it this way:

* **Nothing invented an infinite score anywhere**, on any control.
* **Broad sources widen rather than tile.** 20/20 stayed one source, and the
  narrow neighbour beside a broad one was recovered 20/20 -- the case the free
  width exists for, and the one the pass-based search had to reach through a
  quadrupole that a rotationally symmetric residual does not have.
* **Close pairs entered merged recover 85% of members** with no false
  positives, across 0.5-2.0 sigma separations and unequal fluxes. For scale,
  `prior.py` records the pass-based split accepting 0%, 0% and 8% at 0.4, 0.5
  and 0.6 sigma under the same flux prior.
* **The halo is right.** The in-group source's flux beside a frozen 6000 e-
  neighbour is recovered to a median 1.7% -- the number that moves if the halo
  double counts or omits, and which a position match would not show.
* **The empty region invented an emitter twice in 20.** Both under a prior
  asserting `lam = 0.02/px^2` over a ~580 px region, i.e. ~11 sources expected
  a priori. `detect` estimates `lam` from the frame and would not assert that
  on an empty one; the control is a stress test, not the operating point.
* **`overfit` is the gate's failure** -- see section 3. Recall is 1.00 and the
  true source is always recovered; what fails 55% of the time is getting back
  DOWN to one source, and 40% of that is the unsupported-incumbent refusal.

### Runtime

Same runs. Work is dominated by the fits, and the fits by `K`.

| control | median ms | p95 ms | median fits |
|---|---|---|---|
| isolated | 4.8 | 13.9 | 26 |
| close pair | 13.0 | 19.6 | 54 |
| underfit (3 rounds) | 95.5 | 177.5 | 150 |
| overfit | 77.8 | 309.8 | 98 |
| oversized crowd | 227.6 | **12365** | 4 |

The crowd's p95 is the one number that needs work, and its cause is structural
rather than mysterious: at `K = 12` a round generates ~12 removals, ~48 splits
and the births, each a `p = 49` fit over a region sized by
`BBOX_PAD * sigma_hi`, and each scored through `linalg::logdet_cond`, which
computes a full symmetric eigendecomposition for the condition number. That is
`O(K)` proposals times `O(p^3)`, repeated over the rebuild rounds. Per this
section's own rule the optimization comes after the scientific checks, and the
scientific check on this control is the gate failure above, not the latency.
The obvious first move when it is time is to stop paying for an eigenvalue
condition estimate on every hypothesis when the Cholesky is already in hand.

### Frame level: `detect(search="groups")` against the pass search

`python scripts/check_detect_width.py --search passes groups --sizes 39 64`,
seeds 17-19, density 0.034/px^2, log-width spread 0.2, one repeat. Mean of
the six frames:

| | passes | groups |
|---|---|---|
| recall | 0.865 | 0.863 |
| precision | 0.969 | 0.959 |
| RMSE | 0.316 px | **0.247 px** |
| residual positive peaks, total | 3 | **0** |
| seconds per frame | 0.4 | **35** |

Same recall, better localization, a cleaner residual -- and two orders of
magnitude slower. Per frame the schedule variants tried while building it
moved recall on single frames by up to 0.17 (39x39 seed 17: 0.76-0.93). That
is not evidence about schedules, and the audit below says why.

**Every audited miss is the score's, none is the search's.**
`scripts/audit_group_search.py` finds each cluster where the found
configuration and the truth disagree and scores both in one context under
the final epoch's snapshot, with the search's own scoring code:

| verdict | clusters | in-focus truth missed |
|---|---|---|
| model failure: found outscores truth by > 1 nat | 27 | 30 (79%) |
| search failure: truth outscores found by > 1 nat | **0** | 0 |
| tie, within 1 nat | 7 | 6 (16%) |
| unscorable (truth side nonstationary) | 2 | 2 (5%) |

The model failures are merged pairs, not widened blobs: median closest true
pair 0.75 sigma, 13 of 26 below 0.75 sigma and none at 2 sigma or more; the
found side is wider than 1.3 sigma0 in 2 of 27. The data favour the pair by a
median +2.6 nats, the extra emitter's prior costs 12.4 and its volume returns
5.2. Below ~0.75 sigma that is the identifiability limit -- a frame's count
AUC there is ~0.5 whatever the score -- and between 1 and 1.5 sigma it is the
`Exp(1/A_s)` operating point. The schedule variant that scored best on seed 17
had stopped in configurations the score ranks LOWER.

Where the time goes, on the 64x64 seed-17 frame: 267 transactions, 8.8k fits
at 4.4 ms; splits are 52% of the fits and ~0.6% of split fits are accepted;
groups of 5-7 free emitters are two thirds of the time. The region is padded
for `sigma_hi = 2.2 sigma0` on every side.

Peak memory is bounded by the region and `p`: every buffer in
`GroupWorkspace` is region- or `p`-sized and reused, and a trial configuration
allocates no frame-sized array. The largest is `light_outside_box`'s `h*h` and
`w*w` PSF factor tables -- ~13 KB for a 40x40 region.

## Deliverable sequence and definition of done

1. **Context and native types:** width-aware geometry, capacity audit, stable IDs,
   statuses and reusable workspace, with spatial tests. **Done** --
   `dense_group.rs`, `linalg::P_MAX_VAR`, `layer8_dense_group.rs`.
2. **Scoring contract:** native priors, consistent fitting/scoring, symmetric
   validity and regular-hypothesis comparisons. Publish the unsupported-case
   control results and settle the release gate. **Done; the gate does not
   pass** -- see section 3.
3. **Local search:** proposal competition, joint refits, atomic commits and
   bounded iteration; run direct native scientific controls. **Done** --
   `search_group`, `scripts/check_group_search.py`.
4. **Binding and frame integration:** replace Python per-fit/move dispatch on
   the variable-width Rust path with native group transactions. **Done,
   opt-in**: `detect(..., impl="rs", search="groups")` (section 5). The
   default stays `search="passes"`.
5. **Retirement and report:** remove the superseded production dispatch from
   that path, retain useful reference helpers explicitly, and document results.
   **Report done** (sections 3 and 6); **retirement waits on runtime**.

This stage is done when a native group call owns neighborhood construction,
proposals, fits, scoring and local acceptance; unsupported comparisons are
handled according to the validated contract; no Python numerical or decision
callback is required; and scientific controls support enabling it. Python may
still own candidate/background epochs and the outer frame schedule.

All four hold for the science: the gate passes, and at frame level the group
search matches the pass search's recall with 22% lower RMSE and no residual
peaks, with every audited miss traced to the score rather than the search.
What does not hold is cost -- ~35 s against 0.4 s a frame -- so the default
is unchanged and the pass dispatch is not retired.

### What the next step is

Runtime, now that the audit says the search is not what limits recall. The
measured costs are in section 6: split proposals (half the fits, ~0.6%
accepted), the region padding for the widest admissible source, and
per-hypothesis work in 5-7 emitter groups. Each change there must leave
`check_group_search.py`, `check_detect_width.py --search groups` and
`audit_group_search.py` where they are, or say what moved.

The recall that remains is the score's: pairs below ~0.75 sigma that no
frame-level score can count, and 1-1.5 sigma pairs priced by `Exp(1/A_s)`.
The flux prior is the lever for the second, and the simulated `U(900, 1900)`
fluxes are its worst case -- a merged pair's 1800 e- is inside that
population -- so that change needs a monodisperse control before it is judged.

Do not switch `detect`'s public default or delete the remaining Python frame
implementation merely because the group engine exists. Full native frame
ownership and retirement of that adapter are the next stage, using the group
operation established here.
