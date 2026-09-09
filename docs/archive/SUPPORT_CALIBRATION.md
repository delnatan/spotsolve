# Shared-nuisance source-support calibration

The opt-in prototype now makes a support decision before removing amplitude
shrinkage. It calibrates the actual local search on original photon pixels;
out-of-focus light is included as real nuisance signal in the null model.
Production `spotsolve.detect` is unchanged.

## What is tested

On one fixed patch and focus-center domain, the sparse solver jointly fits
the same log-plane-plus-wide nuisance under capacities K=0,1,2. With
`Q_K = Poisson_NLL_K + lambda * focused_flux_K`, its statistics are

```
T_focus = Q_0 - min(Q_1, Q_2)
T_pair  = Q_1 - Q_2
```

These are empirically calibrated penalized-objective gains, not Bayes factors
or ordinary unpenalized likelihood-ratio statistics. The calibration repeats
all data-dependent local starts and nuisance fitting, including failures and
budget limits, for each fresh null photon draw. No chi-square approximation
is used to decide focused existence or multiplicity.

For each null cell the Monte Carlo tail probability is

```
p_cell = (1 + number of calibration scores >= observed score) / (B + 1).
```

Ties count as exceedances. Existence uses the maximum p-value across K=0
cells; the pair test uses the maximum across all K<=1 cells. Taking the
maximum keeps a difficult null class from being hidden in a pooled average.
With 99 calibration draws per cell, p-values cannot be smaller than .01.

Report zero support when existence is not significant. Otherwise report two
only when the pair test is significant, or one when it is not. If the chosen
sparse capacity lacks the required positive slots, return unresolved rather
than fabricating a position. Only then refit supported amplitudes, positions
and nuisance without the flux penalty. There is no post-fit photon cutoff or
minimum separation rule.

For a listed zero-focus null distribution, any positive support requires the
existence test to pass. For a listed one-focus null, reporting two requires
the pair test to pass. The plus-one argument relies on exchangeability of
calibration and test scores and gives control over calibration/test randomness;
it does not guarantee an exact false-positive fraction for every realized
finite table. Independent evaluation is therefore retained.

**Scope:** the error level is per original fixed patch. It does not cover
selecting many patches from a full frame, arbitrary nuisance parameters, an
unknown camera model, or nuisance distributions missing from the library.
The broad nulls below are specified known means, not uncertain estimates
plugged in from the tested patch. There is no claimed nuisance-uniform bound
between calibration cells. Intermediate defocus and freshly generated haze
are explicitly outside-library transfer tests.

## Precision output

Accepted support receives an unpenalized refit. A numerical derivative of the
analytic likelihood gradient computes the observed Hessian, including the
curvature of the nonlinear mean. Its inverse is marginalized to the joint
position coordinates, retaining coupling to other emitters and nuisance.
This is not an expected-Fisher determinant or a Bayesian evidence estimate.

Boundary, singular or non-positive-curvature solutions receive no covariance.
An exactly absent broad component's four parameters are excluded, explicitly
conditioning on its absence. These estimates condition on the selected
support/model and do not include count uncertainty or camera/PSF mismatch.
The evaluation records nominal 95% two-dimensional ellipse coverage and
worst-direction standard deviations only where covariance exists and count
matches truth. No production precision cutoff is enabled from these data.

## Interfaces and reproduction

```python
from spotsolve.prototype import ComponentModel, SupportCalibration, select_supported

model = ComponentModel((21, 21), 1.2, (7, 7, 13, 13))
calibration = SupportCalibration.load("calibration.json")
result = select_supported(
    photon_patch, model, calibration, focus_rate=.03, alpha=.05,
)
# result.n_focus is 0, 1, 2, or None (unresolved).
# result.refit contains supported local positions and photon amplitudes.
# result.position_covariance may be None; inspect precision_status/diagnostics.
```

The model, focus domain, amplitude-prior rate, optimizer options, Poisson noise
assumption and relevant source files are fingerprinted. Changing them requires
recalibration; saved settings survive JSON tuple/list conversion.

```sh
source ~/uv-workspaces/microscopy/.venv/bin/activate
MPLCONFIGDIR=/tmp/spotsolve-mpl OPENBLAS_NUM_THREADS=1 \
  python scripts/bench_support_calibration.py \
  --output reference/support-calibration-20260909-final
MPLCONFIGDIR=/tmp/spotsolve-mpl python scripts/demo_supported_prototype.py \
  --calibration reference/support-calibration-20260909-final/calibration.json \
  --output reference/support-calibration-20260909-final
python -m pytest -q
```

The second command uses the original simulation pixels produced by
`scripts/demo_sparse_prototype.py`. Both positive examples and negative
controls remain in its before/after panel.

## Evaluation design

The calibration has six cells, each with 99 independent Poisson draws:
blank background; 3000-photon broad emitters of widths 1.25 and 2.5 sigma;
a 900-photon focused single on flat background; and 900/150-photon singles
with the width-2.5 broad emitter. Background is 4 photons/pixel, focus sigma
1.2, patch 21x21, focus domain [7,13] on both axes, and amplitude rate .03
per photon. These remain illustrative physical/prior settings, not instrument
calibration. The 594 draws perform 1782 local count fits.

Separate evaluation uses 40 base photon draws in each of five environments:
blank, compact defocus, broad defocus, intermediate width 1.75 sigma, and
newly generated irregular haze. Each base draw is paired with no injection,
a 150-photon single, a 900-photon single, and an equal 900-photon/source pair
at separation 1.5 pixels. Injected photon noise is independent and the same
underlying base photons are retained. This gives 800 local evaluations; four
variants of a base frame are paired, not independent replicates.

Plots show false focused support and exact single/pair counts at local alpha
.01, .025, .05 and .1. Raw rows, settings, seeds, calibration failures and
95% Wilson intervals for each curve point are saved to JSON. A small observed
false-positive count alone does not establish a precise operating rate.

## Completed results at local alpha .05

| Environment | False focused support / 40 null patches | Exact 900-photon single count / 40 | Exact equal-pair count / 40 |
|---|---:|---:|---:|
| Blank | 2 | 40 | 7 |
| Compact defocus, 1.25 sigma | 0, plus 1 unresolved | 38 | 1 |
| Broad defocus, 2.5 sigma | 0 | 40 | 37 |
| Unseen width, 1.75 sigma | 0 | 39 | 12 |
| Irregular haze stress test | 12 | 32 | 2 |

At zero false calls in 40 draws, the 95% Wilson upper bound is 8.76%, so these
results do not establish a precise 5% operating rate. Blank's 2/40 interval
is 1.38--16.50%. Haze's 12/40 interval is 18.07--45.43%: **the broad-only
calibration does not transfer to this irregular haze distribution**. The
single haze demo patch that passes is not evidence to the contrary.

Faint 150-photon exact-single counts are 5/40 on blank, 0/40 on compact
defocus, 20/40 on broad defocus, 2/40 at the unseen width, and 6/40 on haze.
The noise base is paired across injections; no prior rate or alpha was tuned
to these evaluation results. The considerable power loss must remain visible.

Exact returned count is not geometric resolution. The lone compact-defocus
pair reported with count two has 2.82-pixel position RMSE and a boundary
diagnostic, so it is not a successful localization. For the 37/40 accepted
broad-defocus pairs, median position RMSE conditional on correct count is
.217 pixels. Many flat-background pairs instead receive one source. Failing
to reject the one-source model does not prove that a spot is physically single.

Observed-Hessian coverage is also conditional and incomplete. Among correctly
counted 900-photon singles with available covariance, nominal 95% ellipses
cover 17/19 on blank, 17/18 on compact defocus, 38/40 on broad defocus,
37/39 at the unseen width, and 18/19 on haze. On blank and compact defocus,
21 and 20 of the 40 bright-single draws lack covariance because a parameter
is at a boundary. Broad-defocus pairs cover 70/74 emitter positions; unseen-
width pairs cover only 19/24. Pair members share data and are not independent
coverage trials. These small, selected subsets do not validate a universal
precision gate, particularly when count is wrong or nuisance is misspecified.

Calibration records 12 optimizer failures among 1782 count fits; evaluation
records 27 among 2400 count fits. They were retained in the replicated
procedure and output, not dropped to improve results. Median/p95 local
enumeration times are .211/.301 seconds, excluding the post-selection refit
and numerical Hessian. This is not a full-pipeline or full-frame runtime.

The original six-panel simulation demo now rejects both defocus-only false
candidates (p_focus=.34) and the haze-only false candidate (p_focus=.45).
It retains the isolated single, equal pair plus broad light, and single plus
haze. The 4:1 pair becomes one supported source (p_pair=.65), demonstrating
the cost of the current operating point. The reused demo is descriptive;
the 800-draw independent evaluation above supplies the broader assessment.

## Decision

The calibration interface and support-before-refit order are implemented and
tested, but this is not a calibrated general haze detector. Next include
independently generated haze controls in the null library and test transfer
on new haze conditions. At the same time, compare the current penalized
support statistic against a fully refitted likelihood statistic or a weaker
prior, **recalibrating each procedure** rather than relaxing a raw cutoff on
these evaluation draws. Sparse shrinkage and an overly permissive wide-source
explanation can hide pair structure; a successful false-positive check alone
does not establish useful pair power. Neighbor ownership and camera/PSF
calibration remain required before full-frame claims.

Final artifacts are in `reference/support-calibration-20260909-final/`:
`calibration.json`, `evaluation.json`, `calibration_curves.png`,
`supported_demo.json`, and `supported_simulation.png`. The earlier directory
without `-final` is an interrupted development run and is superseded.

Validation: 88 tests passed, one existing slow evidence-integration test
skipped. The saved-calibration round-trip regression test also passes after
the compatibility fix. Production detection defaults are unchanged.
