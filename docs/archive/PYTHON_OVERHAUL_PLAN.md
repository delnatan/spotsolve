# Python model-selection overhaul

Status: active prototype implementation on `codex/python-model-selection-overhaul`.

2026-09-07 development direction: prioritize concrete model/algorithm proposals
over extending the benchmark campaign. Keep spotsolve as the starting point;
localization uncertainty is the primary objective and frame-to-frame statistical
consistency is next. The intended output feeds tracking, and the settled
numerical algorithm will be implemented in Rust through PyO3/maturin. See
[Focused-emitter proposal](FOCUSED_EMITTER_PROPOSAL.md) for a proposed shared
local solver, calibrated defocus representation, profiled nuisance fitting,
and uncertainty calculation. These are proposals, not promoted defaults.

Haze/pair comparison: calibration now accepts freshly generated correlated
haze nulls with zero or one focused source. A paired experiment compares
the existing sparse support statistic against an independently calibrated
unpenalized likelihood statistic, initially retaining the same nuisance model.
A separately calibrated 3x3 smooth-plus-wide background ablation tests whether
improved haze modeling restores power. All retain continuous focus coordinates.
New haze conditions and equal/unequal pairs test transfer and geometry separately
from count. Compact defocus and weak unequal pairs remain unresolved development
problems; no procedure is promoted to production. See
[Haze support comparison](HAZE_SUPPORT_COMPARISON.md).

Support-calibration checkpoint: the shared sparse model now has an opt-in
finite-simulation existence/multiplicity test, followed by unpenalized fitting
of accepted support. It includes separately evaluated broad-light controls,
matched focused injections, and conditional observed-Hessian position
uncertainty. The significance level is per fixed original patch, not per
frame, and nuisance conditions outside the finite calibration library remain
transfer tests. See [Support calibration](SUPPORT_CALIBRATION.md).

Visual prototype checkpoint: `fit_sparse_patch` now exposes the full fit,
continuous candidate positions, nuisance/focus renders and residuals through
one opt-in interface. `demo_sparse_prototype.py` produces reproducible simulation
and glycerol panels, including the false candidates on defocus/haze controls.
This is a two-source whole-patch prototype, not a production/full-frame
detector. See [Runnable demo](SPARSE_PROTOTYPE_DEMO.md).

2026-09-07 checkpoint: following the user's sparsity-prior direction, the
shared component fitter now supports an exponential penalty on continuous
focused-source amplitudes and an unpenalized refit of surviving slots. Focus
positions remain continuous and the acquisition-level focused PSF stays fixed;
broad nuisance width remains separately fitted. A paired 240-fit synthetic,
144-fit real-patch and 10-fit denser-start audit is complete. Shrinkage can
suppress compact defocus and refitting removes much direct flux bias, but
irregular haze, lost weak sources, coincident slots and numerical instability
still prevent interpreting active slots as reliable emitter counts. No default
prior rate or production change is introduced. The next step is a stable
continuous support/count decision with shared nuisance, using this audit as
the baseline rather than another shape veto. See
[Off-grid sparsity checkpoint](OFFGRID_SPARSITY_CHECKPOINT.md) for implementation,
equations, results, width-precision caveats and remaining work.

2026-09-05 checkpoint: the cheap transverse-shape screen is implemented and
audited, but **not promoted**. It retained all 2,105 native proposals across
four glycerol frames and therefore saved no component fits. It preserved
579/579 injection cases with proposal support, but that does not establish
useful discrimination. Bright diffuse light can also mask injected tight
structure in this statistic. The next experiment returns to fixed whole-patch
K=0/1/2 comparisons with shared nuisance light, including mixed-light injections
and acquisition-level PSF sensitivity. See the new screen results in
[Component model reassessment](COMPONENT_MODEL_REASSESSMENT.md#focus-screen-experiment-2026-09-05).

Completed initial experiment: a cheap, conservative focus-shape screen ahead of
joint fitting. Primary real-data challenge is `data/beads_80pct-glycerol_crop.tif`:
the user identifies 198-nm beads diffusing in 80% glycerol, with both focused
beads and many genuine out-of-focus blobs. Success means avoiding expensive
fits on diffuse structures while retaining tight singles and overlapping
focused sources, not obtaining fewer detections indiscriminately. Use the
effective image of an in-focus bead as the width reference: the finite bead
diameter is not zero and is not itself a PSF sigma. Do not interpret every
native bead as an in-focus ground-truth localization. The whole-patch model
comparison remains the fallback for ambiguous candidates, not the first
operation on every intensity patch.

**Current scope: targeted anti-tiling, not a general detector rewrite.**
Following the user's real-data clarification, the immediate task is the local
decision that distinguishes tight source structure from a blob. Preserve the
existing accurate localization and candidate machinery where they work. Do not
optimize background reconstruction as an end in itself or make the older
validation gates a prerequisite for every local experiment. The proposed
shrinkage-prior experiment is deferred until real-patch comparisons show it is
needed. See the real-data anti-tiling checkpoint in
[Component model reassessment](COMPONENT_MODEL_REASSESSMENT.md) for the current
experiment, user-supplied camera settings, and the narrowed next step. Historical
stage plans and results below remain available for provenance.

Stage 4 revised checkpoint, 2026-09-04: the unfinished reversible-sweep and
patch-expansion implementation has been superseded by direct local K=0/1/2
fits with a shared broad nuisance model. It is a likelihood audit, not a count
selector. A small conditional numerical Bayes-factor reference now checks
overlap integration with known nuisance/flux/centroid. Historical Stage 2/3
numbers below remain baseline diagnostics, and their `exact_count` metric is
not sufficient to establish geometric pair resolution. The current findings,
Bayesian assumptions, and ordered validation gates are in
[Component model reassessment](COMPONENT_MODEL_REASSESSMENT.md).

Stage 4 background checkpoint (completed after laptop pause): a paired
19-cell/12-draw audit compares plane-plus-wide against fixed positive smooth
backgrounds, with and without the shared broad component. Smooth-only fails
compact defocus. A 4x4 smooth-plus-wide model reduces irregular-haze prediction
error but fits more blank-field noise, costs 41% more at median local runtime,
and has material numerical instability on some blank/faint/haze cases. A
28-row enlarged-search replay confirms that optimizer success is not enough.
Neither model is promoted; background adequacy and numerical stability remain
open. The proposed background shrinkage experiment was subsequently deferred
by the targeted anti-tiling scope correction above. Full results and scope are in the
reassessment document above.

Implementation checkpoint, 2026-09-04:

- The first isolated slice now exists under `spotsolve.prototype`: H0, Hsmooth,
  H1, H2, and Hwide mean models, analytic Jacobians, the centroid/separation pair
  parameterization, multistart likelihood fitting, and fit-cost diagnostics.
- `tests/scientific` now contains deterministic semantic scenarios and matching
  metrics. A localization on nuisance light remains a false focused emitter.
- `scripts/bench_prototype.py` runs the oracle-ROI comparison and optionally
  the current detector on identical photon draws. It records raw objectives,
  fitted-pair diagnostics, counts, timing, seed, configuration, and git
  revision as JSON.
- The current `spotsolve.detect` path has not changed.

Stage 2 checkpoint, 2026-09-04:

- Selection is sequential: first focused light versus the background,
  log-quadratic smooth-field, and wide-source nuisance classes; then H2 versus
  H0/Hsmooth/H1/Hwide. Both likelihood gains use finite-sample, plus-one
  bootstrap tail probabilities rather than chi-square cutoffs.
- Calibration draws remain separate by semantic null and parameter cell. The
  runtime p-value is the worst cell-specific p-value, so pooling cannot hide a
  weak nuisance condition. Serialized calibrations fingerprint the fit setup,
  null generators, random seed, and empirical tail arrays.
- Smooth haze is calibrated by generating a new correlated field on every
  draw, rather than by adding Poisson noise repeatedly to one fitted smooth
  image. The latter was demonstrably anti-conservative under held-out haze.
- H2 now has a true radial separation bound of 2.5 sigma. A boundary-railed or
  collapsed H2 fit is diagnostic evidence of model inadequacy, not a resolved
  pair, and is ineligible for pair selection.
- A 72-ROI multistart comparison justified reducing the default grid to three
  separation starts and two flux-ratio starts. The worst H2 objective loss was
  below 6e-10 nats, while median local runtime fell from about 174 ms to 86 ms.

The independent 1% sentinel run used 99 bootstrap draws in each of 12 fitted
or generative null cells, two calibration subpixel phases, three nuisance
widths, three haze correlation lengths, random held-out phases/orientations,
100 held-out draws per null class, and 48 per pair cell:

```text
python scripts/bench_calibration.py --calibration-draws 99 \
  --null-evaluation-draws 100 --pair-evaluation-draws 48 \
  --alpha-focus 0.01 --alpha-pair 0.01 \
  --calibration-output /tmp/spotsolve-stage2-calibration-generative-alpha01.json \
  --output /tmp/spotsolve-stage2-eval-generative-alpha01.json
```

It produced zero focused false calls in 100 blank, 100 smooth-haze, and 300
wide-source draws, and zero false splits in 100 focused-single draws. A 0/100
rate has a 95% Wilson upper bound of 3.70%, so this supports compatibility with
the 1% operating point but does not estimate it precisely. At 900 e- for the
bright source and 4 e-/pixel background, exact pair selection plus localization
for equal pairs was 4.2%, 37.5%, 97.9%, 100%, and 100% at separations 0.5,
0.75, 1.0, 1.25, and 1.5 sigma. The corresponding 4:1 rates were 2.1%, 2.1%,
25.0%, 70.8%, and 97.9%. Median/p95 oracle-ROI times were 85.1/108.9 ms;
offline calibration took 109.3 seconds.

This is a checkpoint, not completion of Stage 2. Calibration still needs a
compact background/flux/phase table with conservative interpolation, larger
held-out samples for tight error bars, and measured PSF/noise mismatch. Stage
3 must include proposal generation and its multiple-search bias before any
frame-level false-positive claim is valid.

Stage 3 checkpoint, 2026-09-04:

- `prototype.proposals` implements the standardized Poisson score for adding a
  fixed in-focus PSF, using separable correlations for both score and Fisher
  normalization. It returns ranked subpixel maxima, one-step flux estimates,
  the trace-free Hessian pair axis, explicit proposal-budget status, and
  connected components.
- The proposal-only background is an iteratively upper-clipped Gaussian field.
  It is intentionally cheap and is not reused as the final scientific
  background fit. The default score cut is permissive at 2.0 standard-score
  units; it controls computation, not false-positive error.
- `prototype.frame` extracts fixed 13x13 local ROIs and fits one ranked
  representative per connected component. This is the minimal Stage 3 bridge;
  fitting multiple candidates jointly and reversibly remains Stage 4 work.
- `prototype.proposal_calibration` bootstraps the maximum focus and pair gains
  after background estimation, local-maximum search, deduplication, component
  grouping, fit budgeting, and local multistart fitting. Its fingerprint covers
  the complete proposal and fit pipeline. Applying these tails to every tested
  component controls the familywise search effect at the calibrated frame size.

The 33x33 proposal-only sweep used 100 draws in every combination of background
1/4/20/100 e-/pixel, bright-source photons 150/300/900, flux ratio 1/4, and
separation 0.5/0.75/1/1.25/1.5 sigma. All 126 source cells outside the
150-photon, background-100 corner had 100/100 component recall. A separate
200-null oracle diagnostic found only 13.5% H1 power and less than 50% focused
or pair power throughout that excluded corner, so it does not trigger the
Stage 3 proposal-recall gate. On 200 held-out nuisance frames per class, the
proposal counts were bounded: blank mean/p95/max 1.43/4/5 and smooth haze
2.25/4/7. Proposal-map median/p95 time was 0.165/0.204 ms.

The end-to-end 1% sentinel command was:

```text
python scripts/bench_proposal_calibration.py --calibration-draws 99 \
  --null-evaluation-draws 100 --pair-evaluation-draws 48 \
  --alpha-focus 0.01 --alpha-pair 0.01 \
  --calibration-output /tmp/spotsolve-stage3-edge-e2e-alpha01-calibration.json \
  --output /tmp/spotsolve-stage3-edge-e2e-alpha01.json
```

It produced zero focused calls in each of 100 blank, 100 smooth-haze, and three
100-frame wide-source classes. Two of 100 focused singles were falsely split;
the 95% Wilson interval is 0.55%-7.00% and includes the declared 1% point, but
more draws are required to distinguish calibration error from sampling noise.
At 900 bright-source photons and background 4, exact end-to-end pair recovery
was 87.5% for equal pairs at 1.0 sigma and 68.8% for 4:1 pairs at 1.25 sigma.
No tested pair frame was lost at proposal generation. Median/p95 end-to-end
frame time was 181/385 ms; offline proposal calibration took 280 seconds. The
edge-inclusive held-out mean fit counts were 1.57 for blank, 2.15 for haze, and
1.95-2.04 across the three wide-source classes.

This remains a checkpoint rather than a production detector. Fixed-size edge
ROIs are shifted inward rather than discarded, and the full-frame bootstrap
therefore includes the same boundary search. Dedicated spatial calibration
cells may still be needed if measured edge behavior differs. The current
single-representative component fit can return at most two emitters per
component. Stage 4 must replace it
with reversible joint component moves, and the calibration grid still needs
the remaining background/flux and measured-mismatch axes.

The first 24-trial diagnostic at sigma 1.2, 900 e- for the bright source, and
background 4 e-/pixel used this command:

```text
python scripts/bench_prototype.py --trials 24 --output /tmp/spotsolve-prototype-stage1-24.json
```

At an exploratory pooled-null 99th-percentile objective gain of 4.699, equal
pairs were selected and correctly localized in 95.8% of 1.0-sigma cases versus
8.3% exact-count recovery by the current detector. For a 4:1 pair the
corresponding rates were 50.0% versus 0%. No tested wide or smooth-haze null
selected H2; one of 24 focused-single nulls did. These are diagnostic numbers,
not a calibrated operating point: the threshold was estimated and evaluated
on the same small sample, and candidate generation was bypassed. Median local
hypothesis time was 125.3 ms (1,630 model/gradient evaluations), versus 23.2 ms
for the current detector on the entire 13x13 frame. The default start grid was
also compared against a grid three times larger on 24 difficult pair, single,
and wide ROIs; its worst objective loss was under 4e-9 nats. Stage 2 must
establish the real conditional error rates on independent bootstrap draws.

This plan replaces the current close-pair and defocus decision machinery in
small, falsifiable stages. The existing detector remains available as the
baseline until the new path clears the scientific and runtime gates below.
Nothing in this plan requires a Rust implementation; the point of the Python
prototype is to settle the model and its validation before optimizing a second
implementation.

## 1. Scientific contract

The detector should answer two separate questions:

1. Is there evidence for one or more *in-focus point emitters* in this region?
2. Can the data distinguish their number and positions from background,
   diffuse fluorescence, or an out-of-focus object?

It should report an emitter only when the answer to both questions supports
that interpretation. A broad nuisance component may be fitted to explain
photons, but it is not a localization. When the data support “structured light”
but do not distinguish one emitter from two, the output should say unresolved
rather than return an overconfident midpoint.

There is no single information-limit separation. Pair recoverability is a
function of separation, total photons, flux ratio, background, sampling phase,
PSF mismatch, and camera noise. The primary target is therefore a *power
surface at a fixed false-focused-emitter rate*, not a minimum distance quoted
without its imaging conditions.

## 2. Design constraints

- Keep the in-focus PSF fixed from calibration, or obtain it from a smooth
  field-dependent calibration map. Do not give every candidate an independent
  focus width.
- Represent defocus and diffuse haze as explicit nuisance hypotheses, separate
  from the reported emitter class.
- Fit all interacting sources in a local component jointly.
- Treat one-versus-two emitters as a nonregular model-selection problem. Do not
  use a single-mode Laplace approximation at a collapsed pair.
- Keep the statistical objective fixed during a complete search/refit sweep.
  Hyperparameters learned from the current data may change only in a documented
  outer iteration.
- Calibrate decisions end to end, including candidate generation and multiple
  testing.
- Optimize only after profiling. The prototype should use NumPy and SciPy
  unless a measured bottleneck justifies another dependency.
- Preserve the current public detector as a reproducible baseline until the new
  path passes every mandatory gate.

## 3. Proposed Python layout

Build the prototype behind an opt-in namespace so experimental code does not
continue to accumulate in the production modules:

```text
src/spotsolve/prototype/
    models.py          # H0, H1, H2 and nuisance mean models/Jacobians
    parameterize.py    # bounded transforms and pair parameterization
    fit.py             # local model fitting and multistart orchestration
    proposals.py       # score map, maxima and pair-axis proposals
    select.py          # statistics and calibrated decisions
    calibration.py     # bootstrap generation and threshold tables
    proposal_calibration.py # bootstrap maxima after the full proposal search
    frame.py           # proposal components to fixed local fits
    solver.py          # component-level and frame-level orchestration
    result.py          # prototype result and ambiguity records

tests/scientific/
    scenarios.py       # deterministic truth-generating scenarios
    metrics.py         # exact-count, false-emitter and localization metrics
    test_*.py           # small mandatory scientific regression tests

scripts/
    bench_prototype.py # larger sweeps, plots and timing reports
```

The scientific tests must compute their own truth-based results. Golden outputs
generated by the implementation may be added later for porting, but do not
count as evidence that the algorithm is correct.

## 4. Common hypotheses

All hypotheses use the same pixels, calibrated noise model, in-focus PSF, and
background parameterization.

### H0: background only

Use a positive local plane by default:

```text
m(x, y) = b0 + bx (x - xc) + by (y - yc)
```

The larger-frame background map supplies the starting value and, if needed, a
weak regularizing prior. The local plane remains free so uncertainty in the
plug-in map is not mistaken for source evidence.

### H1: one in-focus emitter

```text
m = background + A p_focus(x - x0, y - y0)
```

`p_focus` is pixel-integrated and normalized to unit total flux. Its shape is
fixed within the ROI. A field-dependent PSF calibration may select the shape at
the ROI center, but the source fit does not alter it independently.

### H2: two in-focus emitters

Use parameters that remain scientifically interpretable near overlap:

```text
centroid c
total flux F > 0
separation d >= 0
orientation theta modulo pi
bright-component fraction q in [0.5, 1)]
```

The two source locations are derived from `(c, d, theta, q)` so `c` remains the
flux centroid. Constraining `q >= 0.5` removes label swapping. The model is still
nonregular at `d = 0` and `q = 1`; calibration must account for this rather than
pretend the singularity is absent.

### Hwide: nuisance light

Start with the simplest useful nuisance model:

```text
m = background + Aw p_wide(x - xw, y - yw; w)
```

`w` is restricted to widths outside the physically calibrated in-focus band.
This component explains compact defocused light but is never reported as an
emitter. A later stage may replace it with a small measured defocus template
library. Very broad, smooth fluorescence belongs in the background model, not
in an ever-wider Gaussian.

The central pair statistic is initially

```text
T_pair = I(best of H0, H1, Hwide) - I(H2)
```

where `I` is the negative log likelihood up to data-only constants. This
statistic is calibrated empirically; it is not compared to a chi-square or
Laplace cutoff.

## 5. Staged implementation

### Stage 0 — Freeze the baseline and build the scientific harness

Purpose: make every subsequent claim reproducible and separate scientific
performance from implementation agreement.

Work:

- Add deterministic generators for:
  - background-only frames;
  - smooth gradients and correlated diffuse haze;
  - one focused source;
  - one compact defocused source;
  - two focused sources over separation, flux, flux ratio, angle and subpixel
    phase;
  - small dense components with three or more sources;
  - controlled PSF mismatch and camera noise.
- Record both semantic truth class and rendered photon field. Proximity to any
  real light source does not turn a nuisance localization into a true positive.
- Add metrics:
  - focused false emitters per frame;
  - probability of selecting the exact emitter count;
  - pair power and count confusion matrix;
  - localization bias/RMSE conditional on correct count;
  - unresolved/ambiguous rate;
  - interval coverage where an uncertainty is reported;
  - wall time, number of local fits, model/Jacobian calls, and pixels fitted.
- Snapshot the current detector on exactly the same seeds.

Mandatory grid for the larger benchmark:

```text
separation / sigma: 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0
flux ratio:         1, 2, 4, 8
per-source photons: 150, 300, 900, 2000
background e-/px:   1, 4, 20, 100
subpixel phase:     center plus at least 7 random phases
haze width / sigma: 1.3, 1.5, 2, 3, 5 and smooth non-Gaussian fields
```

The full Cartesian product may be sampled rather than exhausted during normal
development. A small, fixed sentinel subset runs in tests; the full sweep is a
benchmark artifact.

Exit gate:

- One command reproduces the baseline summary and writes machine-readable
  results with configuration, git revision, seed, counts, and timing.
- Haze-only frames are scored as negatives for *focused emitters*.
- Current pair and haze failures from the audit are reproduced within binomial
  uncertainty.

### Stage 1 — Implement and verify the four local hypotheses

Purpose: test the statistical model with oracle ROIs before candidate search can
hide its defects.

Work:

- Reuse the existing pixel-integrated Gaussian factors where appropriate.
- Implement H0, H1, H2, and Hwide mean models and analytic Jacobians.
- Fit a local background plane in every hypothesis.
- Use bounded, scaled parameters; do not use a hard box as an unacknowledged
  prior in model evidence.
- For H2, use multistarts over:
  - the residual-Hessian orientation and its perpendicular;
  - several fallback angles when the Hessian is isotropic;
  - separations from roughly 0.2 to 2.5 sigma;
  - flux fractions representing 1:1, 2:1, 4:1 and 8:1 pairs.
- Detect collapsed, boundary-railed and duplicate solutions explicitly.
- Return the best objective and diagnostics for every hypothesis, not just the
  winning parameters.

Efficiency work:

- Cache ROI axes and PSF factors that are invariant across starts.
- Evaluate the mean and Jacobian together.
- Profile model evaluation, linear solves and Python orchestration separately.
- Stop inferior starts early only with a bound or rule shown not to alter the
  winner on the benchmark.

Exit gate:

- Finite-difference checks pass for every analytic Jacobian.
- Oracle H1/H2/Hwide fits recover noiseless parameters from all designated
  starts to numerical tolerance.
- On noisy oracle ROIs, the best-of-multistart objective is stable when the
  start grid is doubled.
- At equal 900 e- flux and background 4, the eventual calibrated decision must
  have a credible path to at least 70% exact-two power at 1.0 sigma while
  rejecting at least 99% of designated haze nulls. This is a go/no-go gate for
  the explicit-model proposal, not a threshold to tune prematurely.
- Median oracle-ROI cost is recorded. Initial code may be slower than the
  current local fit, but must remain within 5x while still instrumented for
  correctness.

### Stage 2 — Calibrate local model selection

Purpose: turn objective differences into decisions without regular asymptotics
that fail at a collapsed pair.

Work:

- Build a parametric-bootstrap calibration for:
  - H0/Hwide versus focused-source existence;
  - H1/Hwide versus H2 multiplicity.
- Generate null data from fitted nuisance parameters and rerun the complete
  multistart fit, including the same bounds and winner selection.
- Initially calibrate a compact table over background, total flux, subpixel
  phase, and nuisance width. Interpolate conservatively between cells.
- Measure conditional error separately for H0, H1 and Hwide nulls; a mixture
  average must not hide a high error rate in one class.
- Compare two decision products:
  - a frequentist calibrated p-value or tail probability;
  - a bootstrap estimate of selection probability/stability.
- Do not introduce a learned flux prior until the likelihood-only behavior is
  understood. If a population prior is later useful, estimate it from many
  frames or calibration data and freeze it within a run.

Efficiency work:

- Calibration is primarily offline and may be expensive.
- Use common random numbers when comparing algorithm changes.
- Cache calibration tables with a versioned fingerprint of the PSF, noise
  model, fitting bounds and start strategy.
- At runtime, selection should require only table lookup/interpolation.

Exit gate:

- Empirical type-I error is within its binomial confidence interval in every
  null class, not merely in the pooled mixture.
- Pair power curves are monotone within simulation uncertainty as photon count
  or separation increases.
- Unequal-pair power is reported explicitly; it is not averaged with equal
  pairs.
- Ambiguous cells are identified as such rather than forced to H1 or H2.

### Stage 3 — Replace candidate generation as a proposal mechanism

Purpose: retain high recall cheaply while removing any claim that the seed
threshold itself is the final statistical test.

Work:

- Use a Poisson/camera-noise score or matched-filter map for H1 proposals.
- Use the trace-free local Hessian/eigenvector response to propose close-pair
  orientations and centers.
- Make the proposal cut deliberately permissive and measure proposal recall
  separately from final selection precision.
- Group overlapping proposals into connected ROIs before model selection.
- Include proposal generation in end-to-end null calibration, because selecting
  local maxima changes the tested statistic's distribution.

Efficiency work:

- Compute full-frame score/Hessian maps with a small fixed number of
  correlations or separable filters.
- Deduplicate proposals before fitting.
- Rank proposals and impose an explicit fit budget, recording when it is hit.
- Reuse the full-frame background/noise quantities in each ROI.

Exit gate:

- Proposal recall is at least 99% for all benchmark cells in which the oracle
  model selector has at least 50% power.
- Candidate count and local-fit count remain bounded on blank and haze-only
  frames.
- End-to-end false-focused-emitter rate, including maximization over proposals,
  meets the declared operating point.

### Stage 4 — Validate a shared-nuisance local count model (revised)

The earlier work list below is retained as historical intent. It is superseded
for K<=2 by the reassessment above: enumerate all three count models directly;
do not implement synthetic structural sweeps. Validate mixed-light nuisance
adequacy, numerical stability, and local evidence before frame ownership or
arbitrary-K search.

Purpose: move from isolated H1/H2 contests to crowded regions without restoring
the current path dependence.

Work:

- Start with components containing at most two focused emitters plus nuisance
  light. Do not generalize to arbitrary K until this case passes.
- Maintain add, split, delete and nuisance-reclassification proposals.
- After a structural change, jointly refit the whole connected component.
- Hold background-map and population hyperparameters fixed for a complete
  add/delete/refit sweep.
- Continue until a full reversible sweep makes no accepted structural change;
  do not use monotone addition followed by a one-way prune as the definition of
  convergence.
- Extend to K > 2 by comparing neighboring count models in the component, while
  retaining the same calibrated local decisions and ambiguity output.
- Make patch construction depend on every component's actual nuisance support,
  not only nominal focus sigma.

Efficiency work:

- Cache component data, grids, PSF factors, current render and fitted H0/H1
  states.
- Dirty only components whose fitted support overlaps a changed component.
- Cap component K and route oversized/ambiguous components to an explicit
  unresolved result rather than silently approximating them.
- Batch independent components later if profiling shows Python dispatch is a
  material cost.

Exit gate:

- Final result is invariant to proposal visitation order on the sentinel suite.
- Running another complete sweep changes neither selected classes nor counts.
- Increasing the component/fit budget does not materially alter results on the
  benchmark.
- Dense-field false-emitter and pair-recovery metrics improve over the frozen
  baseline at matched runtime or have a documented accuracy/runtime frontier.

### Stage 5 — Add the real camera and PSF calibration

Purpose: ensure simulated gains survive contact with the instrument.

Work:

- Define a noise-model interface used consistently by proposal scoring,
  likelihood fitting, bootstrap generation and uncertainty calculation.
- Support pure Poisson first, then the camera-appropriate Poisson-Gaussian or
  excess-noise model with calibrated offset, gain and variance maps.
- Replace a Gaussian PSF with an experimentally measured, pixel-integrated
  in-focus PSF when residual tests show material mismatch.
- Build the nuisance library from measured or optics-generated defocus data.
- Derive the in-focus class boundary from the desired axial tolerance and the
  measured PSF-versus-depth curve—not from detection-count tuning.
- Validate calibration on held-out frames and beads, never on the same frames
  used to choose the models or operating point.

Exit gate:

- Standardized residuals and null score maxima agree with calibrated
  simulations on held-out data.
- Isolated-source localization bias and interval coverage meet their declared
  tolerances across the sensor.
- Haze/defocus controls do not generate focused localizations above the chosen
  false-emitter operating point.

### Stage 6 — Report model uncertainty honestly

Purpose: prevent a confidently wrong midpoint from looking like a precise
localization.

Work:

- Return, per component:
  - selected class/count;
  - calibrated support or p-values for competing counts;
  - an unresolved-multiple flag;
  - conditional parameter covariance when the selected mode is regular;
  - fit-boundary, collapsed-solution and calibration-range diagnostics.
- Compute the observed likelihood/posterior Hessian for regular selected modes.
- Label that inverse as conditional local covariance, not an unconditional
  CRLB.
- Where model uncertainty is appreciable, expose it rather than folding it into
  one position standard error.

Exit gate:

- Conditional intervals have measured coverage on the simulation grid.
- Components with unstable count selection are flagged at the intended rate.
- Downstream consumers can distinguish focused localizations, nuisance light
  and unresolved components without interpreting fitted width heuristically.

### Stage 7 — Optimize and integrate the Python path

Purpose: reach a useful accuracy/runtime point before considering a port.

Work, in profiling order:

1. remove redundant hypothesis fits and repeated renders;
2. cache separable PSF factors and derivative workspaces;
3. reduce multistarts using measured dominance rules;
4. batch small independent calculations where NumPy benefits;
5. specialize the small dense linear algebra only if it is a demonstrated
   bottleneck;
6. consider compiled acceleration only after the algorithm and interfaces are
   stable.

Expose the result as an opt-in `detect_prototype` or equivalent. Keep the old
`detect` behavior unchanged during comparison. Once the prototype wins the
held-out benchmarks, decide separately whether to replace the default API and
whether a Rust port is worthwhile.

Exit gate:

- Accuracy results are unchanged by optimization within predeclared numerical
  tolerances.
- Runtime is reported as a distribution over blank, sparse, dense and haze
  frames—not one favorable average.
- The final comparison gives an accuracy/runtime frontier against the current
  Python and fixed-width Rust paths.
- No port begins while the statistical design, threshold calibration or output
  contract is still changing.

## 6. Decision checkpoints

The stages are deliberately stoppable:

1. If oracle H1/H2/Hwide competition cannot beat the current pair/haze tradeoff,
   revise the model before building proposal search.
2. If bootstrap calibration is unstable across nuisance conditions, use direct
   posterior integration or sampling for the small local model rather than
   adding heuristic thresholds.
3. If proposal recall is the bottleneck, improve score/Hessian proposals without
   touching model selection.
4. If component search is the bottleneck, improve caching and scheduling without
   weakening local decisions.
5. If measured data disagree with calibrated null residuals, fix the camera,
   PSF or background model before tuning detection thresholds.

## 7. First implementation slice

The first code change after this plan should contain only:

1. the Stage 0 scenario/metric harness;
2. H0, H1, H2 and Hwide local models;
3. an oracle-ROI benchmark using known source centers;
4. profiling counters and timing;
5. no change to `spotsolve.detect`.

That slice directly tests the proposal's central claim: an explicit fixed-focus
pair hypothesis can recover close pairs while a separate nuisance hypothesis
rejects haze. Candidate generation, frame iteration, priors, production output
and Rust are intentionally out of scope until that claim passes.
