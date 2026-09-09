# Haze calibration and pair-sensitivity comparison

This experiment addresses the two failures exposed by the previous local
calibration: false focused support on irregular haze, and loss of close-pair
power with a strong amplitude penalty. It keeps the observed patch, focused
PSF, focus domain and fitting budgets fixed. The initial comparison fixes
the nuisance model; a subsequent ablation changes only its smooth background.

## Implemented changes

`HazeSupportNull` supplies a generative calibration distribution. Every draw
contains a newly generated correlated background field, followed by fresh
Poisson photons. Optional known focused light is added to the same mean
before photon sampling. This avoids calibrating noise around one fortunate
or unfortunate fitted haze realization.

The distribution uses a reflected Gaussian-filtered white-noise field,
normalized to a specified nonnegative peak above a constant background.
Null classes remain separate by correlation length and focused brightness.
Metadata stores the generator parameters and random-seed sequence rather
than pretending the class has a single fixed mean image. Fixed-mean nulls
remain supported by the original interface.

Two complete procedures are independently calibrated:

* **Sparse support:** rate .03 per photon, as in the previous experiment.
* **Likelihood support:** rate zero, so support is based on differences in
  unpenalized joint likelihood fits. This tests removing the penalty that
  can favor reallocating focused light to an unpenalized broad component.

The second procedure is an unregularized fitting ablation, not a proper
zero-rate exponential prior. It does not use a chi-square cutoff. Its own
null scores calibrate the entire numerical search, just as for sparse support.
Both use the existing continuous positions, fixed focus sigma, joint nuisance
refits and support-before-final-refit interface. Optimizer starts follow the
same strategy, although fitted basins can differ when the objective changes;
this is a comparison of complete procedures, not a proof about exact global
likelihood optima.

For the sparse procedure, the original six-cell broad-only calibration is
also reconstructed from the same null scores. That isolates the effect of
adding haze controls without changing the fitted photons or optimization.

## Frozen experimental design

Both procedures use a 21x21 original patch, focus sigma 1.2 pixels, focus
domain [7,13] on both axes, a log-plane plus one broad Gaussian, and local
alpha .05. Optimizer budgets remain 48 screening iterations, four retained
starts and 400 refinement iterations. No background basis, width boundary,
minimum separation or acceptance cutoff is tuned to the evaluation images.

Each procedure uses 99 draws in each of twelve null cells:

* the previous six fixed-mean blank, broad and single-plus-broad controls;
* fresh haze at correlation lengths 3 and 5 pixels, each with no focused
  emitter, a 150-photon single, or a 900-photon single.

Haze background is 4 photons/pixel and its peak above background is 8.
These are explicitly specified distributions, not measurements of the camera
or the full range of real haze. The max-cell plus-one tail rule is retained.
Its exchangeability argument applies to the listed distributions; it does
not establish uniform coverage over all possible haze fields or parameters.

Evaluation uses forty independent base draws in each of six environments:
flat background, compact defocus, broad defocus, new length-5 haze, unseen
length-4 haze, and unseen shorter/stronger haze (length 2, peak 12). Each base
draw receives five paired variants: no injection, a 150-photon single, a
900-photon single, an equal pair, and a 4:1 pair. Pairs have 1.5-pixel
separation and random orientation; the bright source has 900 photons.

This gives 1,188 calibration patches and 1,200 evaluation patches per
procedure. Both procedures see byte-identical evaluation images, verified
by per-image hashes. New random seeds separate this run from the earlier
calibration/evaluation. Multiple injections of a base image are paired
experiments, not independent replicates.

The report includes false focused support, exact single/pair counts, and a
separate predeclared pair-geometry diagnostic: position RMSE <=.3 pixels and
relative separation error <=25%. This diagnostic is for evaluation only;
it is not used to fit or accept sources. Exact count alone remains
insufficient evidence of successful pair localization. Wilson intervals
describe individual proportions; paired bootstrap intervals describe the
observed difference between the two procedures and do not resample the
calibration tables themselves.

## Reproduction and interfaces

### Additional background-model ablation

After inspecting the initial two procedures, a third development experiment
uses likelihood support with the existing `smooth_wide` background: a fixed
3x3 positive smooth basis plus the broad Gaussian. The basis width is 3 times
the focused sigma; its nine coefficients are jointly refitted under every
focused count. Emitter positions remain continuous. This is a small background
basis, not fine-grid emitter reconstruction.

This procedure receives its own complete calibration and the same 1,200
evaluation images. It tests whether explaining spatial haze reduces the
burden on the support threshold. The basis choice was made after seeing the
first comparison, so the three-way result is a development comparison, not
an independent confirmatory test of a selected production model.

```sh
source ~/uv-workspaces/microscopy/.venv/bin/activate
MPLCONFIGDIR=/tmp/spotsolve-mpl OPENBLAS_NUM_THREADS=1 \
  python scripts/bench_haze_support.py --rate .03 \
  --output reference/haze-support-comparison/sparse
MPLCONFIGDIR=/tmp/spotsolve-mpl OPENBLAS_NUM_THREADS=1 \
  python scripts/bench_haze_support.py --rate 0 \
  --output reference/haze-support-comparison/likelihood
MPLCONFIGDIR=/tmp/spotsolve-mpl OPENBLAS_NUM_THREADS=1 \
  python scripts/bench_haze_support.py --rate 0 --background-kind smooth_wide \
  --output reference/haze-support-comparison/smooth_likelihood
MPLCONFIGDIR=/tmp/spotsolve-mpl python scripts/report_haze_support.py \
  --input reference/haze-support-comparison --include-smooth
python -m pytest -q
```

The first two commands can run independently; both retain their own
calibration and evaluation artifacts. Saved calibrations fingerprint all
relevant model/search settings and source code. Older calibration files are
not silently applied to a changed implementation.

```python
from spotsolve.prototype import HazeSupportNull

null = HazeSupportNull(
    label="fresh_haze_with_single", null_count=1,
    correlation_length=5.0, peak=8.0, photons=900.0,
)
# Pass alongside fixed-mean null tuples to calibrate_support(...).
```

The default production detector is unchanged. These remain local fixed-patch
experiments, not full-frame error control or validation of precision on real
camera data.

## Completed results

All three procedures completed 1,188 calibration patches and 1,200 evaluation
patches each (7,164 procedure/patch runs). The report verified all 1,200
evaluation image hashes across all three procedures.

With the original sparse fits, adding haze controls reduced false focused
support on fresh length-5 haze from **12/40 to 3/40**. Removing the amplitude
penalty reduced it to **1/40**, recovered **39/40** bright singles and selected
the correct equal-pair count in **27/40** cases, compared with **0/40** equal
pairs for the haze-calibrated sparse procedure. Only **22/40** likelihood
pairs also passed the geometry diagnostic.

The smooth-background likelihood procedure gave the following results.
Every entry has 40 trials; faint singles have 150 photons, bright singles
900 photons, and pairs contain 900+900 or 900+225 photons.

| Environment | False focused calls | Bright single exact | Faint single exact | Equal pair exact | Equal pair geometry | 4:1 pair exact | 4:1 geometry |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat | 1 | 40 | 1 | 32 | 26 | 2 | 0 |
| Compact defocus | 0 | 40 | 2 | 6 | 3 | 3 | 0 |
| Broad defocus | 0 | 40 | 17 | 36 | 19 | 2 | 1 |
| Fresh haze, length 5 | 1 | 40 | 8 | 31 | 21 | 5 | 2 |
| Unseen haze, length 4 | 1 | 40 | 12 | 28 | 22 | 6 | 3 |
| Unseen haze, length 2, stronger | 7 | 37 | 23 | 31 | 23 | 5 | 1 |

The clearest improvement over plane-background likelihood support is on flat
equal pairs: **4/40 to 32/40** exact counts and **4/40 to 26/40** geometrically
accurate pairs. The paired bootstrap 95% intervals for these improvements
are +55 to +85 and +40 to +70 percentage points, respectively. These are
development-sample intervals conditional on the realized calibration tables.

There is no universal winner. On length-5 haze, the smooth model selected more
equal pairs (31 versus 27), but passed geometry in 21 versus 22 and recovered
fewer faint singles (8 versus 18). On shorter, stronger haze it made 7/40
false calls versus 4/40 for the plane model. For 7/40 the Wilson 95% interval
is approximately **8.7%–32.0%**, so this stress condition does not support a
5% false-call claim. Compact defocus and 4:1 pair localization remain weak.

Observed median K=0/1/2 fitting times were .235 s (sparse), .292 s
(likelihood/plane), and .306 s (likelihood/smooth); respective 90th percentiles
were .303, .358 and .401 s. These are prototype timings on this machine,
exclude final refit/covariance work, and were collected with overlapping jobs;
they are not production throughput benchmarks. Evaluation optimizer
non-success flags totaled 10, 3 and 12 out of 3,600 count fits per procedure.
There were respectively 3, 0 and 2 unresolved support results among 1,200
patches. Unresolved results count as failures for recovery, not false calls.

Artifacts under `reference/haze-support-comparison/`:

* `comparison.png` and `comparison.json`: complete proportions, Wilson
  intervals and paired differences.
* Each procedure directory: `calibration.json` and `evaluation.json`, with
  source/settings fingerprints, photon-image hashes and diagnostics.
* `smooth_likelihood/supported_simulation.png` and `supported_demo.json`:
  replay of the six earlier demonstration scenes. The single, equal pair,
  defocus-only, haze-only and single-plus-haze examples receive the expected
  counts; the weak member of the 4:1 pair is still missed.

Decision: retain the new calibration interface and all three opt-in
experiments. The smooth model warrants further development for equal pairs,
but this run does not justify replacing the production detector. The next
evaluation should distinguish background/defocus model mismatch from true
pair evidence and test a calibrated localization-precision decision on fresh
data; increasing conservatism globally already showed a substantial power
cost here.

## Interpretation boundaries

The 4:1 pair is particularly demanding: its weaker source contains 225
photons at a separation of 1.5 pixels. Report both the count and geometry
endpoint rather than describing every two-source fit as resolved. Reported
covariances remain conditional on the selected model; they have not been
validated as a selection-aware precision acceptance rule.

These controls model Poisson fluctuations on spatially correlated haze.
They do not model defective camera pixels, read noise, PSF mismatch or motion
blur. A model for isolated pixel contamination requires a separate experiment.
The broad component still has a hard lower width bound at 1.25 times focus
sigma. No claim is made that this is a physically calibrated defocus boundary.

The smooth model can also change how signal is allocated to the broad
Gaussian; gains cannot be attributed solely to lower statistical cutoffs.
Compact defocus and close focused pairs can compete within this nuisance
family. These experiments compare the implemented searches, including their
numerical limitations, rather than proving a fundamental resolution limit.

Validation of the implementation: **95 tests passed, 1 skipped**. Additional
script checks include compilation, saved-calibration replay, matching all
evaluation-image hashes, and visual inspection of the output figures.
