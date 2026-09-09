# Stage 4 reassessment: shared nuisance and count evidence

2026-09-04, `codex/python-model-selection-overhaul`.

## Decision

**Scope correction after the real-data discussion:** the immediate objective
is anti-tiling, not a general background reconstruction or a replacement for
accurate isolated-source localization. The real-data checkpoint below is now
the next-step guide. The older ordered gates at the end are deferred checks
for broader statistical claims, not prerequisites for trying a targeted fix.
In particular, the shrinkage-prior experiment is no longer the automatic next
implementation step.

Keep the pixel-integrated Poisson forward model and direct enumeration of
K=0,1,2. Replace the unfinished Stage 4 orchestration with a local likelihood
audit. A fixed patch and explicit focus-center domain are inputs. The solver
returns three joint fits and numerical diagnostics; it does not choose a count.

The discarded Stage 4 path added transition enums, repeated selections called
"sweeps", a p-value ambiguity band, and data-triggered patch growth. Its full
sentinel reported five over-counted single frames out of 100; replay identified
two independent H1 fits claiming the same emitter in all five. Enlarging a
noise proposal's patch had admitted a neighboring real emitter. The planned
merge/refit loop would have added another mechanism before ownership was defined.
That path is superseded, not promoted into the Stage 3 API.

The committed Stage 3 fitter, frame bridge, calibration code, and experiment
script have been restored to their earlier implementation. Old calibration
files are again readable by that baseline. This preserves comparison, not an
endorsement of its known limitations. Production `spotsolve.detect` is unchanged.

## One generative model under every count

For the observed pixels P and allowed focused-center region D:

```
Y_i ~ Poisson(mu_i)
mu_i = exp(b0 + by*u_i + bx*v_i)
       + A_w * p_wide(i; c_w, s_w)
       + sum_{j=1..K} A_j * p_focus(i; c_j, sigma)
K in {0,1,2}; c_j in D; A_w >= 0; A_j >= 0
```

The background, broad amplitude, broad center and broad width are refitted
jointly in every count model. A broad component is allowed to be exactly absent.
The same is true of an added focus component, so count models nest exactly.
Focused emitters have Cartesian positions and exchangeable labels; there is no
radial separation rail or minimum separation imposed on their fit.

`solver.py` implements this with existing NumPy/SciPy and analytic derivatives.
One coarse-to-refined multistart fit per count replaces all structural sweeps.
It retains feasible nested starting solutions and reports optimizer success,
projected gradient, active bounds, and evaluation counts. A successful optimizer
return is not a certificate that all likelihood modes were found.

The 1000-photon amplitude unit is numerical scaling, not a prior. Likewise,
data-scaled optimizer bounds are not prior distributions. The nuisance width
range [1.25,4] times focus sigma is an explicit provisional model assumption,
carried over from the earlier prototype. It needs physical calibration and
width-mismatch tests before making a detection claim.

P includes contextual pixels while D limits what this local model may call a
focused center. Wider context does not enlarge D. The finite-patch likelihood
uses the PSF's actual pixel integrals without renormalizing its missing tails.
Gaussian tails need not all fit within P for this likelihood to be valid;
insufficient context instead affects identifiability and model adequacy.

This is a local interface, not a complete ownership solution. Future frame
assembly must give interacting sources one owner and represent neighboring
light consistently. In particular, fitting separate patches and concatenating
their locations is still insufficient. No new post-fit distance suppression or
merge loop has been added.

## Bayes factors: retain the goal, correct the calculation

For a fixed domain and a fully specified generative model:

```
Z_K = integral p(Y | theta_K, K) p(theta_K | K) d theta_K
BF_21 = Z_2 / Z_1
posterior odds(2:1) = BF_21 * p(K=2) / p(K=1)
```

Likelihood gains, Bayes factors, and posterior odds are different quantities.
Some legacy functions named `log_bf` also include count prior odds. The new
likelihood audit intentionally does not inherit that naming convention.

The nuisance prior must be the same under each count; nuisance *parameters*
still require integration and do not cancel just because their prior densities
are shared. Proper background, nuisance, flux and position priors must be
specified independently of the tested patch, or be part of an explicit
hierarchical model. Learning brightness only from accepted detections in that
same patch is not an independent calibration. A uniform prior on a fitted
parameter interval is also an informative model-selection assumption.

For a spatial Poisson count prior and iid uniform positions, account for area
and labeling once. If positions are represented as labeled iid draws, their
joint prior is normalized on that labeled domain. If using an ordered domain,
use the matching normalization instead; do not add a second arbitrary factorial
penalty or multiplicity bonus. Conditioning the count model on K<=2 must be
explicit, and larger components must be handled outside this local problem.

At coincident positions or zero amplitude, some coordinates are unidentifiable.
The regular quadratic approximation behind ordinary Laplace/BIC does not have
a blanket justification there. Clipping a Hessian determinant or refusing a
nearly singular fit does not evaluate the missing posterior mass. This is a
reason to validate the integration, not to abandon Bayesian model comparison.
See [Drton and Plummer, singular model selection](https://arxiv.org/abs/1309.0911).
Prior sensitivity is part of the scientific question; diffuse priors are not
automatically neutral for marginal likelihoods. See
[Llorente et al., prior densities in model selection](https://arxiv.org/abs/2206.05210).

There is a second approximation to audit in the legacy evidence code: for a
nonlinear Poisson mean, the observed likelihood Hessian is not generally equal
to expected Fisher information. With J_i = d mu_i / d theta,

```
H_observed = sum_i [Y_i/mu_i^2 * J_i J_i^T
                   + (1 - Y_i/mu_i) * d2 mu_i/d theta2]
F_expected = sum_i J_i J_i^T / mu_i
```

A posterior Hessian also includes negative log-prior curvature in the chosen
coordinates. Any future Laplace approximation must state these approximations
and coordinate/prior normalizations explicitly. An expected-Fisher determinant
alone does not supply a validated evidence calculation.

### A small numerical reference, with explicit conditioning

`evidence_reference.py` computes a conditional BF for one versus two focused
emitters. Nuisance mean, total focus flux and flux centroid are treated as known
under both models. Only pair displacement and bright fraction are integrated.

The reference prior is uniform displacement in a disk, hence p(d)=2d/d_max^2;
orientation is uniform over 2*pi and bright fraction is uniform on [.5,1].
This is normalized, but it is **not** the conditional prior automatically
induced by arbitrary iid source-position and brightness distributions. It is
a stated reference experiment, and it carries no count prior odds.

Gauss-Legendre quadrature and a periodic angle grid integrate this three-variable
problem directly. Collapsed separation and zero total flux give BF=1. Increasing
quadrature order supplies a convergence check without a Hessian approximation.
The first coarse grid changed log BF by 0.254 nats on a strong pair; refinement
reduced the next change below 0.0001 nats, which demonstrates why a numerical
check must accompany an evidence estimate.

Plugging a fitted nuisance field, flux and centroid into this routine would not
turn it into full evidence. Its purpose is to provide a small reference for
overlap behavior before introducing a numerical integration method for all
unknowns. It makes no detector power or false-positive claim.

## Experiments and interpretation

`bench_components.py` uses independent Poisson draws of the **sum** of focused
and nuisance means. It tests broad-only, single-plus-broad, overlapping pairs,
unequal pairs, and freshly generated irregular haze with and without focused
emitters. It records source hashes as well as the git revision so uncommitted
prototype changes cannot silently share a benchmark identity.

The shared model recovers all nine initial noiseless cases (K=0/1/2 crossed
with broad flux 0/900/3000) to numerical precision. This fixes the demonstrated
representational failure of the either-focused-or-wide model. It does not
establish calibrated count selection under noise.

Count correctness and localization are separate from resolution. The existing
legacy `exact_count` measure accepts two identical midpoint locations as a
successful pair at d=sigma. New pair diagnostics report position RMSE,
separation error, relative separation error, and separation-vector error
separately. Duplicated midpoint estimates have 100% relative separation error.
Geometry computed from the K=2 fit when truth contains two sources is labeled
**conditional on the true count**, not end-to-end recovery.

The 12-draw-per-cell run contains 204 noisy patches in 17 cells. All 14
noiseless Gaussian cells have true-count objective below 3e-12 nats. The three
irregular-haze means leave residual objectives of 14.5, 29.8 and 65.9 nats for
true counts zero, one and two, respectively. These are model-discrepancy
diagnostics, not p-values.

For equal pairs, median relative separation errors conditional on fitting K=2
were:

| True separation | No broad source | 3000-photon broad source |
|---|---:|---:|
| 0.75 sigma | 22.0% | 83.0% |
| 1.25 sigma | 5.0% | 11.8% |

The bright focused emitter carries 900 photons; background is 4 photons/pixel,
focus sigma is 1.2 pixels, broad width is 2.5 sigma, patch is 21x21, and the
focus domain is [7,13] on each axis. These small samples quantify the challenge;
they do not establish information-limit or Rayleigh-resolution performance.
Gaussian sigma is the simulation scale, not a calibrated Airy/Rayleigh mapping.

The denser-start audit runs on the first noisy frame in each cell. Sixteen of
17 K=2 objectives agree within 4.1e-8 nats; one broad-only frame improves by
0.456 nats. Two of 612 fitted count models report optimizer failure, retained
in the output rather than hidden. Median/p95 local-fit times are 257/334 ms.
These are 21x21 local fits, not comparable full-frame timings or a speedup claim.
The reference evidence is evaluated on the first draw of each nonzero-count
cell; the largest refinement change is 0.0018 nats. The raw audit is saved at
`/tmp/spotsolve-stage4-shared-nuisance-audit.json`.

A separate conditional prior-sensitivity check uses a noiseless 13x13 patch,
background 4, sigma 1.2, total focused flux 1800, known centroid (6,6), and
quadrature order (64,128,32). For the pair, both sources have 900 photons,
separation 0.75 sigma, and angle 0.37 radians:

| Displacement-prior radius | Single-source log BF_21 | Pair log BF_21 |
|---|---:|---:|
| 1.5 sigma | -1.487 | +1.681 |
| 2.5 sigma | -2.293 | +0.733 |
| 4.0 sigma | -3.038 | -0.187 |

The largest change relative to order (48,96,24) is 0.00033 nats. Thus the sign
change for the weak pair is driven by the stated prior, not quadrature error.
This demonstrates why tuning the prior to recover known pairs would not be a
principled solution. Physical/population assumptions and sensitivity analysis
are required even with accurate integration.

One broad Gaussian is not a complete model of irregular haze. Nonzero noiseless
residuals on the held-out correlated fields expose that limitation. These
controls must remain in the suite; additional emitter likelihood gain on a
misspecified background must not be called evidence of successful resolution.

Reproduce the compact audit:

```sh
source ~/uv-workspaces/microscopy/.venv/bin/activate
python scripts/bench_components.py --draws 12 \
  --output /tmp/spotsolve-stage4-shared-nuisance-audit.json
python -m pytest tests/scientific
```

## Shared-background comparison

The next implementation slice extends the same local fitter, not its decision
logic. Three fixed nuisance representations can be compared offline:

- `wide` (unchanged default): three log-plane coefficients and four broad-source
  parameters.
- `smooth`: a positive fixed Gaussian partition-of-unity background, used as an
  ablation without a broad source.
- `smooth_wide`: that background plus the same broad-source parameters.

The 3x3 and 4x4 grids have nine and sixteen nonnegative coefficients,
respectively. Centers span the observed patch; kernel sigma is three focus
sigmas. Axis weights sum to one at every pixel, so a constant background is
represented exactly, including at patch boundaries. No basis locations or
widths are fitted, and no family is selected adaptively from the tested patch.
This is a positive smooth parameterization, **not a hard spatial band limit**;
its kernel width is not a guaranteed minimum feature width. Its flexibility
must be tested, not assumed harmless to focused signals.

All coefficients and source parameters are fitted jointly under every count.
There is no background subtraction, extra veto, or new frame-level loop.
The smooth coefficients currently have optimizer bounds but no statistical
shrinkage prior; these likelihood fits are not marginal likelihoods.

The paired audit includes blank fields, compact and wider defocus, mixed
focused/defocused light, equal and unequal overlapping pairs, newly generated
irregular haze, faint 50/150-photon singles, and a shorter haze correlation
scale. All four configurations fit the same photon draws, with rotating
evaluation order for timing. Noiseless residuals measure representation error.
For noisy fits, the additional diagnostic is

```
prediction KL = sum_i [mu_true * log(mu_true / mu_fit) - mu_true + mu_fit]
```

This is the expected excess Poisson negative log likelihood on an independent
draw of the same underlying mean. It penalizes fitting noise as well as
background mismatch, unlike residuals against the fitted photons. Evaluation
uses the simulated true count; neither this diagnostic nor recovered flux is
an end-to-end detection result. Each haze realization is regenerated, so noisy
medians and the one noiseless example per cell need not differ only by noise.

### Completed comparison and restart checkpoint

The paired background run finished successfully before the laptop pause:
19 cells, 12 draws per cell, four configurations (912 local
K=0/1/2 enumerations). Raw results are saved locally at
`reference/stage4-background-audit-20260904.json` (ignored run artifact), with
the original at `/tmp/spotsolve-stage4-background-audit.json`. Reproduce with:

```sh
source ~/uv-workspaces/microscopy/.venv/bin/activate
python scripts/bench_components.py --background-audit --draws 12 \
  --seed 2026090412 --output /tmp/spotsolve-stage4-background-audit.json
```

Review completed after resuming. The smooth-only ablation cannot explain
compact defocus: at width 1.25 sigma its zero-focus noiseless mismatch is 2387
nats and median gain from adding a focused source is 2283 nats. At true count
one with 3000 broad photons, its median focused-flux ratio is 1.637. It is not
an acceptable replacement, despite doing well on some haze examples.

Retaining the broad component fixes the demonstrated noiseless Gaussian-mixture
failure. The remaining comparison is a bias/variance and computation tradeoff:

| Diagnostic | Plane + wide | 3x3 smooth + wide | 4x4 smooth + wide |
|---|---:|---:|---:|
| Nuisance parameters | 7 | 13 | 20 |
| Blank median prediction KL (nats) | 3.09 | 6.64 | 9.74 |
| Haze, no focus: median prediction KL | 29.60 | 13.04 | 10.71 |
| Haze, one focus: median prediction KL | 36.04 | 17.89 | 13.46 |
| Haze, two focus: median prediction KL | 37.49 | 14.86 | 11.66 |
| Shorter-scale haze, no focus: median prediction KL | 44.86 | 30.42 | 24.09 |
| Median / p95 local runtime (ms) | 263 / 363 | 306 / 451 | 370 / 572 |
| Flagged optimizer failures / 684 count fits | 2 | 3 | 8 |

Times are the complete K=0/1/2 enumeration on 21x21 patches. The 4x4 model
costs about 41% more at the median than the baseline. Its improved haze fit
does not establish better source detection; neither does its roughly tripled
blank prediction loss by itself establish a false-emitter rate. The 3x3 model
is cheaper but is not uniformly adequate: its haze-only median gain for adding
a focus is 2.36 nats versus 1.52 baseline and 1.06 for 4x4. No gain threshold
has been introduced.

For bright Gaussian controls the 4x4 model's median total focused-flux ratios
are 0.953--1.020. Known-count pair geometry is broadly retained, but there is
no uniform improvement: equal pairs at separation 1.25 sigma without broad
light have median relative separation error 6.9% baseline versus 8.1% for
4x4; unequal 4:1 pairs with broad light change from 33.5% to 19.0%. There are
only twelve draws per cell, so these are descriptive comparisons, not precise
power or bias estimates.

Faint-source separation from background is still uncertain. In the 150-photon
haze cell the 4x4 median recovered flux is 87.4% of truth (5th--95th draw
quantiles 46.2%--114.1%), versus 95.0% baseline (62.3%--103.7%). Median gain
for adding a focus falls from 5.57 to 3.06 nats. That could include correction
of background-driven excess gain as well as absorption of source light; this
experiment does not identify either as the sole cause. Known-count fits do
not measure detection probability.

### Numerical replay

The replay uses every row with a flagged optimizer failure (13 rows, 14 count
fits), plus the first draw of five mixed-light controls for each of the three
shared-wide models (15 rows). It raises screening iterations from 48 to 96,
retained starts from four to eight, refinement iterations from 400 to 1600,
and uses the existing denser pair-start grid. The 28 replay records, original
objectives, source/input hashes, and configuration are saved locally at
`reference/stage4-background-stability-20260904.json`.

Most changes are small, but several are scientifically material:

- On blank seed 2026090417, the 4x4 K=0 and K=1 fits improve by 2.35 and
  5.22 nats, despite both originally reporting success. The K=2 fit improves
  by 0.18 nats. Optimizer success cannot certify likelihood differences.
- On faint-flat seed 2026220420, the 4x4 K=1 and K=2 fits improve by 1.64
  and 2.64 nats.
- On shorter-scale single-plus-haze seed 2026270412, the 3x3 K=2 fit improves
  by 3.19 nats, again from an originally successful result.
- The tested bright Gaussian mixed-light controls remain stable to numerical
  precision. Some replay objectives get slightly worse (up to 0.0522 nats):
  changing screening budgets changes which basins survive. More starts alone
  do not certify a global optimum or monotone improvement across runs.

One flagged status remains in a baseline wide-only K=1 fit whose objective is
unchanged and projected gradient is tiny. The replay is a diagnostic subset,
not a replacement for all 912 fits; the table deliberately retains the original
paired-run results. The background-adequacy and numerical-stability gates are
**not passed**.

### Background-audit recommendation (superseded by the real-data checkpoint)

Keep the default unchanged and retain both smooth-grid sizes only as offline
comparison configurations. Do not add adaptive background-family selection,
another source veto, automatic grid refinement, or frame orchestration.

The next useful experiment is **one proper shrinkage prior on the smooth
background**, with the same prior under all counts. Give both the background
level and its spatial variation proper distributions; a differences-only
penalty leaves a constant mode unnormalized and cannot by itself define
Bayesian evidence. State the variation scale independently of the tested
emitters, using separate background controls or an explicit sensitivity range.
Do not tune it to maximize recovery of known pairs. Use the existing grids
and held-out blank, compact-defocus, faint, pair, and shorter-scale-haze controls
to test whether it reduces noise fitting without concealing model mismatch.

First evaluate this as a small joint posterior-mode experiment, explicitly
**not a Bayes factor**. Recheck the problematic numerical seeds and a separate
successful-fit sample. If it fails the tradeoff, revise the physical background
assumption rather than stacking penalties. Only then spend effort on full
nuisance integration with normalized priors and an independently checked
evidence calculation. A prior may reduce ambiguity; it cannot create photon
information or identify sources against an arbitrarily PSF-like background.

The full default test suite passes (69 passed, one slow check skipped); the
slow legacy evidence check also passes when run separately with `--runslow`
(70 tests exercised in total). These tests validate the implemented numerical
contracts, not global convergence or the unimplemented full Bayesian detector.
Grid size validation now rejects non-integer/nonfinite values clearly and
accepts NumPy integers. No production/default model change has been made.

## Real-data anti-tiling checkpoint

Primary early-screen challenge (user clarification): the glycerol movie shows
198-nm beads diffusing in 80% glycerol, including real focused beads and real
out-of-focus blobs. At its recorded sampling the physical bead diameter is
about 1.90 pixels; this is not a Gaussian PSF sigma. Account for finite bead
extent by using the effective sharp-bead image as the focused reference,
rather than assuming each bead is a mathematical point. The aim is to screen
clearly diffuse candidates cheaply and send only plausible or ambiguous
focused structure to joint fitting. Test narrow transverse structure so
elongated focused pairs are not rejected by a roundness cut. Measure expensive
fits avoided and retained tight-source proposals separately. Native focused
counts remain unknown; inspection of real blobs and incremental recovery of
known injected sources serve different roles. Use unused frames/regions to
check transfer, without pretending neighboring movie frames are independent
ground-truth replicates. No per-blob tuning or tracking system is required.

### What is necessary, and what is not

Lower whole-patch prediction KL is a useful diagnostic, but it is not the
scientific objective. A model can leave some diffuse texture unexplained and
still localize focused emitters correctly; a lower residual can also be bought
by assigning false emitters. The previous background comparison was starting
to optimize reconstruction rather than the actual failure mode. We should not
keep enlarging a smooth basis simply because its residual is smaller.

The smallest intended change is to the **source-count decision**, keeping the
existing localization and cheap candidate machinery where they work:

1. Examine the original pixels of a small ambiguous group, before its light
   has been partitioned among tiles. Keep that group's context and focused
   search domain fixed for the comparison.
2. Compare a blob/background explanation with the same nuisance light plus
   zero, one, or two calibrated tight PSFs, refitting jointly. Focused sources
   and diffuse light must coexist; they are not mutually exclusive classes.
3. Accept additional sources only on support for their tight spatial structure,
   not because repeatedly adding PSFs can reduce arbitrary residual structure.
   Inadequately represented or count-unstable patches must be reportable as
   unmodelled/unresolved, not forced to become emitters. The K<=2 experiment
   is not permission to tile a larger crowded component with independent pairs.

The current shared local fitter supplies a starting point for this comparison,
not a finished selector. Preserve the Bayesian interpretation only where the
calculation actually supports it: raw gains are not Bayes factors, and a Bayes
factor between two misspecified models cannot validate either interpretation.
Do not disable splitting globally, filter each completed tile by its width,
impose a universal minimum separation, or simply increase an acceptance cutoff.

An exact full evidence integrator, a background hyperparameter hierarchy,
arbitrary-K optimization, and a new frame framework are not required to test
this local change. Numerical checks should target whether decisions/locations
change, not require global optimization certification for every background
coefficient. They remain necessary before claiming reliable evidence or
information-limit performance, but need not become product architecture.

### Camera settings and real-image audit

The user supplied approximate offset 100 ADU for all files, gain 2 ADU/electron
at 0.1043 um/pixel, and gain 2.4 at 0.065 um/pixel. The TIFF ImageJ metadata
confirms this mapping:

- All three `beads_*` files: 0.1043359375 um/pixel, gain 2.
- All three `hyp7gem_*` files: 0.065 um/pixel, gain 2.4.

NA is approximately 1.4. NA and pixel size alone do not specify a Gaussian
PSF sigma: wavelength, imaging modality, and the Gaussian approximation also
matter. The audit currently uses provisional sigma 1.2 pixels for beads and
1.45 for hyp7. These are starting settings, not newly established PSF
calibrations. It does not infer gain from the observed spot population.

`scripts/audit_blob_tiling.py` runs fixed crops of frame zero in all six files,
using the current Python detector with splitting disabled, the default six
round budget, and an extended twelve-round budget. Raw pixels, display limits,
and other settings are identical within each comparison. Its only outputs are
an inspection panel and checkpointed JSON with positions, widths, histories,
timings, crop coordinates, calibration, and hashes. It does not change `detect`.

| File / crop | No split: reported N | Six rounds | Twelve rounds |
|---|---:|---:|---:|
| beads_60x_still, whole 39x39 | 48 | 44 | 43 |
| beads_60x_still_02, whole 62x62 | 114 | 109 | 109 |
| beads_80pct-glycerol, y80 x64, 64x64 | 49 | 56 | 64 |
| hyp7gem_anc-1, y0 x96, 64x64 | 53 | 65 | 65 |
| hyp7gem_wt_04, y210 x180, 64x64 | 43 | 47 | 47 |
| hyp7gem_wt, y80 x50, 64x64 | 52 | 56 | 63 |

These counts are **not truth or accuracy scores**. Turning off splits leaves
many localizations on diffuse/weak-looking regions; some are uncertain, not
proven false. Increasing the budget can increase *or decrease* the reported
count because pruning and the reporting-width filter run afterward. On the
glycerol crop the twelve-round working count reaches 128 before pruning and
width filtering, while only 64 are reported; on the wt crop it reaches 137
with 63 reported. Both still accept splits in the last round. Runtime grows
from 9.06 to 18.13 seconds for glycerol and 9.19 to 19.64 seconds for wt.
This demonstrates search-budget sensitivity and wasted work, not that every
late emitter is false. Removing the split mechanism is not an adequate fix.

The user-calibrated outputs are in
`reference/real-blob-audit-user-calibration/`. An earlier provisional run in
`reference/real-blob-audit/` used old repository gain values and is superseded;
do not mix its counts with the table above. Both directories are ignored local
run artifacts. Reproduce the current comparison with:

```sh
source ~/uv-workspaces/microscopy/.venv/bin/activate
MPLCONFIGDIR=/tmp/spotsolve-mpl python scripts/audit_blob_tiling.py \
  --output reference/real-blob-audit-user-calibration
```

### Real PSF mismatch is part of the anti-tiling problem

An exploratory fit of bright spatially isolated maxima already shows a width
spread: the fitted width quartiles (25/50/75 percentiles, pixels) are roughly
0.84/0.95/1.22 and 0.95/1.02/1.42 for the two still-bead crops,
1.20/1.24/1.51 for glycerol, and 1.26/1.33/1.51, 1.32/1.49/1.60,
1.52/1.64/1.80 for anc-1, wt_04 and wt respectively. These use unweighted
least squares with a log-plane and one pixel-integrated free-width Gaussian
on 9x9 patches. Candidate maxima are found after 0.7-pixel smoothing, above
the 85th intensity percentile, separated by at least five pixels, excluding
image edges, and taking at most 25 in brightness order (only six in the first
still crop). They are **not verified single in-focus beads**; defocus,
unresolved neighbors and background contaminate them. These numbers therefore
do not justify an automatic lower-quantile width rule or a new sigma default.

More directly, take frame zero of anc-1, original patch y15:36, x111:132,
gain 2.4 and offset 100. All resulting pixels are nonnegative. Fit the shared
plane-plus-wide component model with domain [7,13] on each patch axis:

| Assumed sigma | K=2 minus K=1 likelihood gain | K=2 positions (patch y,x) |
|---|---:|---|
| 1.305 | 60.91 nats | (10.24,9.97), (11.01,11.28) |
| 1.450 | 35.53 nats | (10.52,10.44), (9.49,7.63) |
| 1.595 | 25.64 nats | (10.51,10.37), (9.40,7.00) |

The gain remains large while the interpretation changes from a close pair to
a weak peripheral source, ultimately at the allowed-domain boundary. We do
not know the true count of this bright object. What this does establish is
that a large gain does not make its two-source localization robust to modest
PSF changes. A more accurate background integral alone would not solve that.

The next bounded implementation should therefore address the **whole-patch
point-source versus blob comparison on these real examples**, with one common
in-focus PSF calibration per acquisition, not a separately adjustable width
for every reported source. Use a small set of real isolated, overlapping and
diffuse patches alongside matched injections of known focused sources into
real backgrounds; such injections measure incremental recovery, not the
unknown count of native emitters. Use subsequent movie frames/unused regions
for checking transfer rather than tuning every crop independently. Simulated
single/pair/blob controls remain mechanism tests, not a substitute for these
observations. Background shrinkage is an option only if this direct comparison
demonstrates its need.

## Focus-screen experiment, 2026-09-05

The first early-screen experiment is complete and **not promoted** into the
detector or calibrated Stage 3 pipeline. `prototype/focus_screen.py` compares
the strongest transverse curvature of a smoothed original image with its
two-scale contrast. A pixel-integrated Gaussian at 1.6 times the provisional
focus width supplies a fixed discrete reference, taking the most permissive
response over a 5x5 subpixel-phase grid. This is a shape boundary for an
experiment, not a physically calibrated focus/defocus boundary.

The curvature matrix permits narrow structure in any direction, preserving
elongated pair proposals rather than imposing roundness. The screen keeps
edges, weak contrast, and uncertain shape. It only screens a proposal when
the largest matrix eigenvalue plus three times a Poisson filter-noise scale
is negative. The scale is the Frobenius second moment from squared linear
filter kernels with observed photons substituted for the mean. It is **not**
a calibrated confidence interval or a source-existence test, especially after
selecting intensity maxima. No optimization, background subtraction, tile
residuals, count priors, or learned per-blob widths enter the screen.

`scripts/audit_focus_screen.py` ran on complete glycerol frames 0, 7, 13 and
19 with sigma 1.2 pixels, gain 2 ADU/electron, and offset 100 ADU. The screen
settings were fixed across the four frames. None of these converted pixels
required negative clipping. The existing prototype proposal generator uses a
4096-proposal budget for this audit; it was never exhausted. This tests the
prototype proposal population, not proposals generated by production split
iterations, and therefore cannot establish production runtime savings.

| Frame | Native proposals | Retained | Whole components skippable |
|---|---:|---:|---:|
| 0 | 530 | 530 | 0 / 528 |
| 7 | 526 | 526 | 0 / 526 |
| 13 | 524 | 524 | 0 / 523 |
| 19 | 525 | 525 | 0 / 524 |

Original component membership is preserved when calculating potential savings:
every member must be screened before a whole group can be skipped. A diffuse
bridge is not removed to manufacture multiple independent source owners. No
joint fits were run by this audit, and no saved-fit timing is inferred.

At twelve random interior sites per frame, independent Poisson source photons
were added to the same real pixels. Controls include singles, equal pairs and
4:1 pairs at one-sigma separation, bright-source flux 150/900 photons, and
injected widths 0.9/1.0/1.1 times the fixed reference. This gives 864 cases;
the real background photons were not resampled. Later movie frames provide
transfer checks, not independent ground-truth replicates.

Here support means that every injected position has a proposal within 1.5
reference sigmas; a single proposal may support both pair members. It measures
whether a local fit can be proposed, **not** resolved count, localization
accuracy, or successful incremental detection.

| Injected class | Cases | Proposal support before / after screen |
|---|---:|---:|
| Single, 150 photons | 144 | 57 / 57 |
| Single, 900 photons | 144 | 123 / 123 |
| Pair, bright source 150 photons | 288 | 136 / 136 |
| Pair, bright source 900 photons | 288 | 263 / 263 |

Of 579 supported cases, 549 lacked such support at those truth positions in
the paired native frame. All remained supported after screening. In fact no
proposal was screened anywhere in the injection runs. Thus retention is
trivial here and must not be presented as a successful sensitivity/specificity
tradeoff. Median/p95 screen time was 3.16/3.53 ms per 256x256 injected frame,
but zero candidate fits or component fits would be avoided.

Mechanism tests verify cancellation of constant/linear fields, retention of
bright subpixel singles and rotated pairs over 0.75--2.5 sigma separation,
screening of a bright broad-only Gaussian, and safe retention of weak and edge
cases. A separate noiseless mixed-light check exposes a limitation: adding a
150- or 900-photon focused single at the center of a 100,000-photon broad
Gaussian of width 2.5 sigma still gives `diffuse` (background 4, sigma 1.2,
65x65 image). These controls use the existing `focused_single` and
`wide_source` generators, summing their means and removing one copy of the
background. Bright broad light can hide tight structure in this normalized
shape comparison. This argues against installing this screen as a veto even
if it removes convincing isolated broad controls.

The screen is retained solely as a reproducible negative experiment. Do not
relax its noise allowance or tune the width boundary on these same images to
manufacture savings. Next use the already implemented fixed-patch shared
nuisance K=0/1/2 fitter on a small real-patch/injection panel, retaining the
same focus domain and acquisition-level width settings across counts. Explicitly
compare native and injected fits and their PSF sensitivity; do not turn raw
likelihood gains into evidence or declare native source counts known. A cheap
screen should only return if it can distinguish superposed tight structure
from diffuse light without this demonstrated masking problem.

The audit JSON contains camera/configuration settings, input/source hashes,
frame coordinates, injection seeds, paired support metrics and timings. The
inspection panel and JSON are local ignored artifacts at
`reference/focus-screen-20260905/`. Reproduce with:

```sh
source ~/uv-workspaces/microscopy/.venv/bin/activate
MPLCONFIGDIR=/tmp/spotsolve-mpl python scripts/audit_focus_screen.py \
  --output reference/focus-screen-20260905
python -m pytest -q
```

Validation: 73 tests passed; the pre-existing slow evidence integration test
was skipped. Production `spotsolve.detect` and calibrated selection behavior
remain unchanged.

## Deferred validation gates for broader claims

1. **Background adequacy (still open).** Judge adequacy by whether background
   errors create false focused sources or conceal real overlapping sources,
   not by minimizing background reconstruction error alone. Retain joint broad
   light, held-out mixed/empty controls, and fixed model configurations. Test
   shrinkage only if the targeted real-patch comparison demonstrates its need;
   avoid stacking unrelated nuisance alternatives and vetoes.
2. **Numerical stability.** Check that reasonable increases in start budget
   leave scientifically meaningful likelihood differences stable. Audit broad
   amplitude zero, faint second emitters, and position-domain edges separately.
   Neither repeated labels nor optimizer return codes establish this gate.
3. **Full local Bayesian reference.** Specify proper physical priors, extend
   integration to uncertain nuisance/centroid/flux, and check independent
   integration accuracy plus sensitivity to plausible prior scales. Use an
   established evidence integrator for a small offline reference if needed;
   do not build a sampler framework into the detector. A cheap approximation
   must earn its place by matching this reference across overlap regimes.
4. **Frame ownership and validation.** Only after the local model passes, add
   unique ownership for interacting sources and benchmark the entire proposal
   and reporting procedure. Include isolated singles with weak neighboring
   proposals, mixed backgrounds, edge sources, and more-than-two controls.
   Assess false emitters and geometry directly, then profile runtime.

No arbitrary-K solver, dynamic patch expansion, learned population prior,
posterior ambiguity cutoff, or large bootstrap campaign is required to answer
the next gate.
