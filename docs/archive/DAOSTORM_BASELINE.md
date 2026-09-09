# DAOSTORM baseline and detector scope

2026-09-07. This comparison tests whether an established detector already
provides a better starting point than extending the recent local count-selection
experiments. Production code is unchanged.

## Implementation

Use the external [sdt-python implementation](https://github.com/schuetzgroup/sdt-python),
version 20.1.4, with Numba 0.67.0. Its
[DAOSTORM API](https://schuetzgroup.github.io/sdt-python/loc.html)
offers `2dfixed` and variable-width isotropic `2d` fits despite the module name
`daostorm_3d`. This is an implementation of 3D-DAOSTORM operated in 2D modes,
not a claim to reproduce the exact original 2011 DAOSTORM software.
The original authors also provide
[storm-analysis](https://storm-analysis.readthedocs.io/en/latest/index.html).

The installed source uses sampled Gaussian PSFs and a Poisson-deviance fit.
It alternates residual peak finding and overlapping-neighbor fitting, lowers
the search threshold over initial iterations, removes bad and close fits,
and refits. The upstream defaults include minimum distance equal to the
initial sigma, an 8-pixel Gaussian background estimate, and 20 outer iterations.
This is an established practical baseline, not a parameter-free or calibrated
false-discovery procedure. It is also not merely a single-pass peak finder.

## Comparison design

`scripts/bench_daostorm.py` runs the current production `spotsolve.detect` and
both DAOSTORM models at peak-height thresholds 5, 10 and 20 photoelectrons.
The thresholds are a sensitivity sweep, not equivalent settings to spotsolve's
statistical thresholds. The middle value is used for the main overlays; no
threshold is chosen by optimizing the evaluation truth.

Real images use all six pre-existing crops from `audit_blob_tiling.CASES`,
at frame 0. Thus this is a visual audit of six datasets, not exhaustive movie
processing. The approximate camera gain/offset and provisional focus widths
are inherited from that audit. There are no native truth labels.

Simulations use all 20 existing frames at each of three densities in
`data/sim_out`: psfkit's vectorial confocal PSFs, axial range ±0.5 µm, read
noise, and known camera conversion. This is a reuse of the existing benchmark,
not a newly generated independent test set. Focus sigma is the existing
0.818-pixel calibration; focused truth has |z| ≤ 0.2 µm.

Both methods receive identical converted pixels floored at 1e-6 photoelectrons.
The floor avoids an upstream DAOSTORM division by the observed pixel value at
zero. Counts of floored pixels are recorded. This does not make the Poisson
likelihood an exact model of read noise or negative camera fluctuations.

All fitted widths remain available. For spotsolve this includes its returned
width-rejected nuisance objects, which remain part of its fitted image.
A common [0.75, 1.25] × focus-sigma band is applied only for reporting, never
used to delete and refit broad sources. It is a provisional diagnostic band,
not an independently calibrated focus classifier. Spotsolve's internal width
prior/search/reporting defaults still differ from DAOSTORM's.

Matching associates estimates with truth at all depths within 1 pixel,
maximizes the number of one-to-one matches, then minimizes total distance.
Only centers at least 5 pixels from the original image edge enter the reported
counts. Focused recovery, matched defocused objects and unmatched estimates
are separate. Unmatched estimates are not automatically called false spots
or tiles: positional error, overlap and matching ambiguity can also cause them.
RMSE is conditional on matching, and this suite does not establish close-pair
resolution from count recovery alone.

Times measure full input-crop/frame calls, including fitting. Numba warmup is
separate and saved. Numba DAOSTORM versus Python spotsolve is an implementation
comparison; it does not establish an intrinsic algorithmic speed ratio.
The recent 21×21 prototype timings are not directly comparable.

## Reproduction

Create an isolated environment, then install the benchmark dependencies:

```sh
uv venv --python 3.13 /tmp/spotsolve-dao-venv
uv pip install --python /tmp/spotsolve-dao-venv/bin/python \
  sdt-python==20.1.4 numba==0.67.0 matplotlib tifffile
MPLCONFIGDIR=/tmp/spotsolve-mpl NUMBA_CACHE_DIR=/tmp/spotsolve-numba \
  OPENBLAS_NUM_THREADS=1 /tmp/spotsolve-dao-venv/bin/python \
  scripts/bench_daostorm.py
```

Output lives in `reference/daostorm-baseline/`: `real.png`, `simulation.png`,
`results.json` and `summary.json`. Results retain coordinates, widths, fluxes,
per-frame metrics and timings, package versions, source/input hashes, camera
settings and simulation metadata. The data and generated artifacts are ignored
by git; the runner and this document are the reproducible source.

## Observations from the real overlays

At threshold 10, variable-width DAOSTORM gives 14 and 54 all-width fits on
the first two bead crops, versus 37 and 78 for fixed-width DAOSTORM and 77
and 148 modeled objects for spotsolve. Visual inspection shows much less
subdivision with variable widths, but also missed visible structure in the
first crop. Counts alone cannot distinguish correct consolidation from
underfitting. Fixed-width DAOSTORM also places multiple centers across some
diffuse glycerol and tissue features.

Across the six crops, variable-width DAOSTORM takes approximately 28–120 ms
at threshold 10, versus 4.4–12.3 seconds for the current Python spotsolve
implementation. These are recorded for later optimization; speed is not the
current selection criterion and does not make the quicker output more accurate.

## Completed simulation results

All 60 frames completed, with seven method/settings combinations per frame.
The table uses threshold 20 for both DAO modes; the full threshold sweep is
retained in `summary.json`. Values apply to the common provisional focus band.
Counts are totals over 20 frames per density, not per-frame rates.

| Density | Method | Focus recall | Matched defocused truth | Unmatched estimates | Matched-focus RMSE (px) |
|---|---|---:|---:|---:|---:|
| Sparse | spotsolve | 98.0% | 76 | 9 | .118 |
| Sparse | DAO fixed | 98.5% | 119 | 175 | .159 |
| Sparse | DAO variable | 93.0% | 23 | 2 | .119 |
| Moderate | spotsolve | 90.2% | 243 | 98 | .163 |
| Moderate | DAO fixed | 92.8% | 429 | 505 | .211 |
| Moderate | DAO variable | 76.7% | 71 | 12 | .160 |
| Dense | spotsolve | 77.6% | 401 | 167 | .209 |
| Dense | DAO fixed | 82.7% | 742 | 513 | .280 |
| Dense | DAO variable | 58.0% | 164 | 18 | .191 |

Spotsolve is competitive: it preserves substantially more focused sources
than variable-width DAO and yields fewer unmatched estimates and lower
conditional RMSE than fixed-width DAO. Variable-width DAO rejects more
defocused light, but sacrifices focused recovery. No method dominates every
endpoint, and these settings do not have matched false-call rates.

The width screen explains only part of variable DAO's recall loss. Before
screening, its focused recalls at threshold 20 are 96.0%, 83.4%, and 66.4%,
versus spotsolve's 98.5%, 93.0%, and 83.5%. Neither these metrics nor visual
real-data inspection demonstrate calibrated localization uncertainty.

Validation: asymmetric isolated-source recovery for both DAO models checks
coordinate order and origin; matching checks cover competing associations,
empty truth/estimates and width filtering. Both figures were visually inspected.
Production source was not modified, and no production-suite result is claimed
for this benchmark-only change.

## Scope recommendation

User clarification after the comparison: retain spotsolve as the starting
point, address focused calls on out-of-focus objects first, and defer speed
optimization. The intended consumer is a tracking pipeline. Simplicity must
preserve the physical/statistical interpretation rather than remove useful
overlap handling or substitute arbitrary shape/distance vetoes.

This agrees with the targeted anti-tiling scope in
`COMPONENT_MODEL_REASSESSMENT.md` and `PYTHON_OVERHAUL_PLAN.md`: preserve useful
candidate finding and accurate localization; change the local source-count
decision where broad light is being subdivided. Compare the same original
pixels with shared jointly refitted nuisance light, with and without an extra
calibrated tight source. Broad light and nearby focused sources must coexist.
Use the effective in-focus bead image for the 198-nm glycerol beads; bead
diameter is not the focused Gaussian sigma.

Before adding another acceptance mechanism, identify whether the broad-source
failure is representational, numerical, or a count-decision error. The current
production model's upper width is 2.2 times focus sigma, while the existing
confocal calibration reaches about 3.23 times focus sigma at |z|=.5 µm.
That is a concrete model limitation to audit, not proof that widening a bound
alone fixes tiling. The recent prototype already demonstrates that a more
flexible nuisance model can lose real focused sources or absorb close pairs.

For tracking, the user ranks **localization uncertainty first and frame-to-frame
statistical consistency second**. The target is calibrated position uncertainty,
not merely a small conditional standard error or visually smooth trajectories.
Broad nuisance fits remain in the image model but should not become trackable
positions; unresolved source configurations must be distinguishable from
well-supported localizations.

Each frame is an empirical observation under the image-formation/noise model.
Under repeated draws with a fixed latent scene, position errors and reported
uncertainty should agree: assess bias, interval/ellipse coverage and standardized
errors. Joint fitting must propagate coupling to neighboring emitters and
nuisance light. Covariance conditional on one selected count does not include
uncertainty about that count or model mismatch; a precise fit to the wrong
source configuration must not be presented as an unconditionally precise
localization. The existing conditional Hessian output is a starting point,
not proof of calibration.

Frame-to-frame stability means fluctuations consistent with those statistics,
not suppressing fluctuations with smoothing, hysteresis or an arbitrary
minimum-distance rule. Real movie frames may also differ through diffusion,
defocus, brightness changes and motion blur, so consecutive coordinates are
not iid draws from a fixed scene. Separate measurement variability from latent
motion when interpreting stability; use repeated fixed-scene simulations to
isolate the former. Neighboring real frames are not independent truth labels.

False births and missed neighbors are false positives/negatives under the
influence of other emitters: tests of the shared image model and its source
decision, rather than separate tracking heuristics. Evaluate them alongside
uncertainty calibration on broad-only, focused-plus-broad and overlapping
focused-source controls. Temporal priors and a tracking implementation are
not needed for this detector fix.
