# Continuous amplitude sparsity checkpoint — 2026-09-07

The objective remains robust, efficient multi-emitter localization in images
containing focused PSFs, defocused emitters and spatially varying background.
The previous cheap shape screen failed its utility check. This slice tests
the user's proposed sparsity principle directly in the continuous component
model, keeping the focused PSF width fixed across an acquisition.

## Implemented model and statistical interpretation

For the same fixed pixels and focus-center domain under every fit:

```
mu_i = background_i(beta) + A_w p_wide(i; c_w, s_w)
       + sum_j A_j p_focus(i; c_j, sigma_cal)
minimize Poisson_NLL(Y, mu) + lambda * sum_j A_j
A_j >= 0; c_j continuous; sigma_cal fixed
```

`fit_component(..., focus_rate=lambda)` implements this with the existing
analytic Jacobian and nonnegative linear amplitude coordinates. The rate is
in inverse photons, not inverse optimizer units. The derivative with respect
to the optimizer's A/1000 coordinate is 1000*lambda. Background, broad-light
amplitude, broad position and broad width are jointly refitted; they are not
penalized by this focused-source prior. `focus_rate=0` reproduces the previous
likelihood baseline exactly. Returned `objective` remains unpenalized NLL;
`focus_penalty` and `penalized_objective` are separately exposed. Numerical
projected-gradient diagnostics refer to the actual optimized objective.

For a fixed number of amplitude slots, the normalized prior is
`p(A_j)=lambda exp(-lambda A_j), A_j>=0`. Its parameter-dependent negative log
is the implemented penalty. The omitted `-K log(lambda)` normalization is
constant within a fixed K, but not across K. No count prior or normalized
joint nuisance/position prior is supplied. These are penalized likelihood
fits with a proper amplitude-prior interpretation, not full Bayesian count
posteriors. Rate zero denotes an unregularized baseline, not a proper prior.

At a fixed candidate position c and zero amplitude, the one-sided derivative is

```
dQ/dA|0 = lambda - sum_i p_focus(i;c) * (Y_i/mu_i - 1).
```

This gives the proposed thresholding mechanism without a reconstruction grid.
Positions still make the joint optimization nonconvex. Checking a few local
starts is not a certificate that the insertion score is below lambda everywhere
in the continuous domain.

For nonnegative amplitudes the penalty is **total focused flux**, not the
number of sources. One 900-photon emitter and two 450-photon emitters have
identical penalty. At coincident positions their rendered mean also agrees.
An exponential prior can encourage boundary zeros, but it does not by itself
resolve duplicate representations or decide emitter count. Penalizing focused
flux also favors assigning light to the unpenalized nuisance model; faint
focused light can be lost as a result.

`refit_active_component` removes exactly zero slots and refits remaining
amplitudes, continuous positions and nuisance jointly without the penalty.
It keeps the same fixed focus width, patch and search domain, and never adds
sources. This removes direct shrinkage conditional on the chosen support;
it cannot undo selection bias, recover a deleted emitter, or fix a wrong
background/PSF model. Small positive slots are retained explicitly rather than
hidden behind an undocumented photon cutoff. This is a diagnostic support
refit, not a validated source-count selector.

The staged shrinkage/refinement idea has precedent in
[FALCON](https://pmc.ncbi.nlm.nih.gov/articles/PMC3974135/), which uses sparse
deconvolution, fixed-support estimation and continuous refinement. Our small
experiment starts with continuous source coordinates directly; it is not an
implementation or performance reproduction of FALCON. Continuous sparsity
also has a measure-based formulation, where the total variation of a positive
atomic measure equals its total flux; see
[Duval and Peyré](https://arxiv.org/abs/1306.6909).

## Paired audit

`scripts/bench_sparse_components.py` records 240 synthetic local enumerations
(ten cells, six noisy draws, four rates), ten denser-start replays, and 144
real-patch enumerations. Each enumeration fits K=0,1,2; the support experiment
always inspects the two-slot fit, never selects K by comparing penalized
minima. Two positive slots exhaust this experiment's capacity and do not
establish that a patch contains exactly two emitters.

Rates 0, .001, .01 and .03 per photon are a fixed sensitivity range. Positive
rates correspond to exponential means 1000, 100 and approximately 33.3 photons.
They are not measured population priors and none is selected as a detector
default. All synthetic fits use a 21x21 patch, fixed sigma 1.2, focus domain
[7,13] on both axes, and the shared log-plane-plus-wide model. Fresh haze
draws test background misspecification; one broad Gaussian cannot represent
arbitrary haze. No basis family was tuned or enlarged for these results.

The strongest tested penalty gives the following **active-slot diagnostics**:

| Control (6 draws each) | No focused slots, rate 0 | No focused slots, rate .03 |
|---|---:|---:|
| Blank | 1/6 | 1/6 |
| Broad-only, width 2.5 sigma | 0/6 | 3/6 |
| Compact defocus, width 1.25 sigma | 0/6 | 6/6 |
| Irregular haze only | 0/6 | 1/6 |

For isolated singles, rate .03 gives one active slot in 5/6 draws versus
0/6 unregularized. For single-plus-broad the corresponding number is only
1/6. Equal pairs with broad light retain two slots in all six draws, while
one 4:1 pair loses a slot. One faint-single-plus-haze and one bright-single-
plus-haze draw lose all focused slots. These six-draw cells are mechanism
checks, not precise operating-characteristic estimates.

At rate .03, median total focused-flux ratios before/after unpenalized refit are:

| Control | Penalized flux / truth | Refit flux / truth |
|---|---:|---:|
| Isolated single | .796 | .997 |
| Single + broad | .856 | 1.016 |
| Equal pair + broad | .896 | 1.018 |
| Unequal pair + broad | .807 | .926 |
| Faint single + haze | .438 | 1.036 |

These flux summaries include incorrect counts and missed sources. A median
near one does not establish individual flux accuracy, correct support or pair
resolution. The JSON separately records position RMSE and pair separation
error when the returned positive-source count matches truth.

Median/p95 enumeration-plus-refit times are 263/353 ms at rate zero and
228/340 ms at rate .03. This is a local timing comparison with rotated rate
order on common data, not a full-frame speedup claim. Three of 720 synthetic
count fits and one of 240 support refits flag optimizer failure.

### Numerical limitations that affect the scientific interpretation

Denser starts plus larger budgets improve penalized objectives by 0.413 nats
for compact defocus, 1.691 for haze, and 8.422 for single-plus-haze at rate
.01. The latter is material, so the initial multistart result is not certified.
The replay changes blank support from two slots to one with an objective
difference of only 8.5e-9 nats. The original blank fit has coincident positions
and fluxes 5.3904 and .00023 photons; the replay represents the same light with
one positive amplitude. Thus counting exact-positive slots can be unstable
even when objective agreement is excellent. The refit cannot resolve this
representation ambiguity by itself.

These issues remain visible in the saved output rather than being hidden by
a new distance threshold, photon cutoff, or an optimizer-success flag.

### Real-patch and width transfer check

Four fixed original-pixel patches use frames 0 and 13 of the glycerol movie,
21x21 origins (84,68) and (120,104), inside the historical challenge crop.
Each is tested natively and with one or two added 900-photon emitters. Pair
separation is 1.5 pixels. Injection Poisson noise is generated independently;
the native pixels are not resampled. Gains/offsets remain 2 ADU/electron and
100 ADU. No converted patch pixels required clipping.

Each exact same image is fitted at sigma 1.08, 1.2 and 1.32 and all four
rates. Width is fixed for a complete fit, never free per focused emitter.
Native counts are unknown, and added sources can make the true count exceed
the two-slot capacity. These rows therefore expose response to injection and
PSF mismatch, not end-to-end recovery scores.

At frame 0/origin (84,68), the uninjected patch has zero positive slots across
the tested widths/rates. After pair injection, rate zero returns two slots
at sigma 1.2; rate .03 returns one. With sigma 1.08 and rate .03 it returns
zero. At frame 13/origin (120,104), the unpenalized injected pair changes from
one slot at sigma 1.08 to two at 1.2. Thus a prior does not remove calibration
sensitivity, and stronger sparsity can conceal known added focused light.
These observations need to guide support/model checks, not per-patch tuning.

## Fixed versus fitted focus width

If width is known correctly, treating it as fixed cannot worsen the regular
Fisher-information bound. With position r and nuisance parameters eta,
the available position information is the Schur complement

```
I_position = I_rr - I_r_eta inverse(I_eta_eta) I_eta_r.
```

Adding an unknown width can remove information through cross-coupling. It
does not necessarily do so: symmetry can make position and width orthogonal
for an isolated source. Width mismatch is a different problem and can bias
positions or turn shape error into spurious multiple emitters.

A small expected-information calculation using this repository's integrated
Gaussian Jacobian illustrates the distinction. On a 31x31 image, background
4, sigma 1.2, 900 photons per source, fitting amplitudes/background/positions:

* For a single source at (15,15), the x/y standard-deviation bound is .04850
  pixels with either fixed or freely estimated width.
* For equal sources at (14.25,15) and (15.75,15), allowing independent widths
  changes each source's along-pair bound from .18153 to .29899 pixels; the
  transverse bound remains .07305 pixels.

These are regular expected-information bounds for specified correct models,
not measured localization errors, observed posterior intervals, or performance
at the coincident-pair singularity. The production design should retain an
independently calibrated effective focused PSF (including finite bead extent)
and check its uncertainty across acquisition-level sensitivity fits. Do not
infer that arbitrarily fixing an inaccurate sigma improves accuracy.

## Decision and next implementation

Keep the off-grid amplitude-prior and support-refit interfaces as opt-in
experiments. The direct shrinkage/refit mechanism works, but the audit does
not yet support a robust detector: haze false structure, lost mixed-light
sources, count nonidentifiability and local optima remain. No production
`spotsolve.detect` change or default prior-rate change is made.

The next bounded solver improvement should reuse fitted basins across prior
rates and refine amplitudes/background conditionally on continuous positions,
then check a continuous insertion score on the original shared patch. This
targets stability and wasted repeated fitting. It must still expose unresolved
support when coalesced or capacity-limited; L1 alone supplies no justified
emitter-count decision. A separate count/point-process model or calibrated
structural comparison remains necessary before reporting counts. Avoid adding
an arbitrary minimum separation or treating all positive slots as emitters.

Reproduce:

```sh
source ~/uv-workspaces/microscopy/.venv/bin/activate
MPLCONFIGDIR=/tmp/spotsolve-mpl python scripts/bench_sparse_components.py \
  --real --output reference/sparse-components-20260907.json
python -m pytest -q
```

The JSON contains source/input hashes, prior rates, seeds, raw fits, separate
likelihood/penalty values, support refits, geometry, timings and optimizer
diagnostics. It is a local ignored artifact. Validation: 80 tests passed and
the existing slow evidence-integration test was skipped. New tests cover prior
units/gradients, flux-splitting invariance, zero-rate compatibility, penalized
nesting, amplitude-bias removal, exact-zero pruning and invalid rates.
