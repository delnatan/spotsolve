# Local inference numerical contract

This maps the implementation of the single active plan in
[FOCUSED_EMITTER_PROPOSAL.md](FOCUSED_EMITTER_PROPOSAL.md).
`spotsolve.inference.fit_component` and `position_uncertainty` run in Rust by
default. The Python algorithm lives separately under
`spotsolve.inference.reference` for numerical validation; it is never a fallback.
The existing full-frame `spotsolve.detect` pipeline remains separate.

## Model and outputs

One `FocusedModel` describes calibrated pixel-response PSFs, one continuous
defocused component, and a fixed bicubic background. Its 16 signed coefficients
must produce nonnegative sampled background rates. Focus uses depth zero; the
caller supplies a positive defocus interval covered by the calibration. The
numerical mean floor is 1e-4 photons/pixel.

Rust fits K=0/1/2 and returns fixed-count hypotheses and raw likelihood gains.
These are neither selected detections nor emitter-presence probabilities.
The model is the retained small-control choice, not a claim of universal
superiority under every form of haze.

Position uncertainty uses observed curvature with nuisance coupling. It is
conditional on the supplied source count and regular local support. Boundary,
nonstationarity and singularity statuses withhold covariance. Optional coordinate
profiles and profile-discovered refinement remain explicit Python reference
diagnostics; they are not dependencies of the native API.

## Implementation boundary

| Location | Responsibility |
|---|---|
| `rust/spotsolve-core/src/inference.rs` | Calibration, means, analytic derivatives, affine basis |
| `rust/spotsolve-core/src/affine.rs` | Constrained Poisson amplitude/background solve |
| `rust/spotsolve-core/src/geometry.rs` | Profiled bounded geometry search |
| `rust/spotsolve-core/src/search.rs` | Initial proposals, screening, broad restarts, nested counts |
| `rust/spotsolve-core/src/uncertainty.rs` | Observed curvature and conditional joint covariance |
| `rust/spotsolve-py/src/inference.rs` | Owned buffers and PyO3 calls |
| `src/spotsolve/inference/rust.py` | Thin input/output adapters |
| `src/spotsolve/inference/model.py`, `psf_bank.py` | Python model/calibration description and reference evaluation |
| `src/spotsolve/inference/types.py` | Budgets and result records |
| `src/spotsolve/inference/reference/` | Frozen Python fitting, profiles and numerical covariance |

Rust data structures and workspaces are separate from the numerical algorithms.
There is no backend selector, shared search callback layer, Python optimizer
callback, or automatic Python fallback. New production work belongs in Rust.
Python reference changes should be limited to maintaining useful validation.
The removed experimental prototypes are documented only in `docs/archive/`.

Only arrays and scalar settings cross the boundary. Calibration is copied once
into a prepared native model. A complete count search copies the observed image
once, releases the GIL, and reuses one numerical workspace across local fits.
Returned arrays own their storage. Separate prepared instances can run
concurrently; each call exclusively borrows its instance's mutable workspace.
Calibration integration and spline prefiltering remain Python preprocessing.

The public `FitOptions` has four settings: `max_iter`, `screen_iter`,
`keep_screened`, and `gtol`. Rust stops on stationarity or budget. SciPy's
objective-change `ftol` exists only on the reference options.

```python
from spotsolve.inference import fit_component, position_uncertainty

fits = fit_component(photons, model, candidate_centres=centres)
uncertainty = position_uncertainty(photons, model, fits[1])
```

For repeated regions with the same model, retain the native storage directly:

```python
import numpy as np
from spotsolve.inference import prepare_rust

native = prepare_rust(model)
records = native.fit_component(
    np.ascontiguousarray(photons, dtype=float),
    np.ascontiguousarray(centres, dtype=float).reshape(-1, 2),
    seed_sigma=model.seed_sigma,
)
lower, upper = native.parameter_bounds(photons, 1)
uncertainty = native.position_uncertainty(photons, records[1]["theta"], lower, upper)
```

The convenience adapter constructs native storage per call. Repeated-call users
can reuse a prepared instance without adding Python search logic. An outdated
extension raises an explicit rebuild error.

## Numerical layout

| Block | Coordinates | Dimension |
|---|---|---:|
| Background | Bicubic coefficients, in 10 photons/pixel units | 16 |
| Broad emitter | Flux/1000, y, x, depth in micrometers | 4 |
| Focused emitters | Flux/1000, y, x per emitter | 3K |
| Affine inner problem | Background and all fluxes | 17+K |
| Nonlinear outer problem | Broad geometry and focused positions | 3+2K |

Total parameter count is 20+3K, at most 26. Arrays use y/x pixel-center
coordinates. Jacobians have `(height, width, parameter)` layout; calibration
responses have `(depth, offset_y, offset_x)` layout. Calibration axes are uniform,
responses positive, extrapolation disallowed, and normalization fixed. Cropping
an ROI loses tails without changing the amplitude units. `seed_sigma` affects
proposal spacing and initial flux guesses only, never the fitted PSF.

The objective is Poisson deviance/2, with saturated-model constants removed.
Zero observations contribute their mean. For fixed geometry,
`mean = offset + basis @ coefficients`; the convex solver enforces box bounds,
nonnegative source flux, and nonnegative sampled background separately.

## Native search and efficiency

The default count search retains the established proposal policy:

1. K=0 starts from image moments, the in-focus peak and the patch midpoint,
   at three defocus depths, plus an absent broad component.
2. K=1 includes a zero-flux nested start, moment/peak/user candidates, and
   candidates with absent broad light.
3. K=2 includes a nested zero-flux neighbor, symmetric/unequal splits at four
   angles and three separations, and pairs of distinct candidate centers.
4. Each count screens all starts, refines the best few, and retains untouched
   feasible starts as fallback hypotheses. K=1/2 also receive three broad
   residual restarts, fitting the original observed pixels.

An experimental `proposal_method="aguet"` keyword on `fit_component` replaces
only the initial focused moment/peak list with native Aguet maxima. Its
`proposal_alpha` (default 0.05) is a proposal threshold, never a dense acceptance
rule. User centers, nesting, splits and broad restarts remain common. Native
records expose `proposal_centres`; the default stays `"moments"`. See
[DENSE_PROPOSAL_CONTROLS.md](DENSE_PROPOSAL_CONTROLS.md) for paired results and
remaining uncertainty limitations.

Stable objective ranking and strict final comparisons preserve tie ordering.
Raw starts cannot worsen the nested likelihood simply because optimization
loses accuracy. A winning raw start receives a freshly computed constrained
stationarity certificate and an explicit `raw_start` status. Optimizer success
is not inferred from a small objective. Local search is not a global guarantee.

Broad residual proposals correlate calibrated unit-flux responses with the
Poisson score and information. Three response tables are prepared once per
component and reused across K=1/2. Numerator and information are accumulated in
one loop; asymmetric response orientation and rectangular/even patches have
explicit tests. This direct local correlation is quadratic in pixel count;
it is intended for small fitting regions, not a full-frame filtering method.

One profile evaluation samples each PSF once at unit amplitude, solves for
brightness/background, and rescales stored geometry derivatives. It works at
zero amplitude without division. Inner Newton iterations require no PSF
interpolation and allocate no new buffers after workspace sizing. Line searches
compute objective only; changing the active face without moving coefficients
reuses derivative reductions. The solver uses active-face damped Newton and
small Cholesky solves. Nearly stationary degenerate faces use nonnegative
active-normal multipliers to certify the full constrained stationarity residual.

The outer bounded BFGS step metric contains at most 49 doubles. Depth is scaled
by the defocus span; lateral coordinates use pixels. Inner tolerance is
`min(1e-7, 0.1*gtol)`, with 160 inner iterations. An inner failure provides no
envelope gradient; failed line-search trials are rejected. Failure and budget
statuses remain explicit. Numerical ridges safeguard steps only; they never
modify likelihood or reported covariance. No new Cargo dependency is required.

Diagnostic native methods remain available: `evaluate`, `prepare_affine`,
`affine_at`, `solve_affine`, `profile_at`, `fit_geometry`, and
`observed_curvature`. They accept contiguous float64 arrays; fitting methods
require feasible seeds and compatible model bounds.

## Localization uncertainty

The native evaluator computes analytic second derivatives of the calibrated
log spline. Only nonzero source-block mean derivatives are stored: nine terms
per pixel for broad light and five per focused emitter. The first-derivative
fitting path does not calculate them. The full observed likelihood Hessian is

```
H = J.T @ diag(data / mean**2) @ J
    + sum_pixels((1 - data / mean) * mean_second_derivatives)
```

The residual term includes mixed flux/geometry derivatives even at zero flux.
Fisher curvature and the optimizer's BFGS matrix are never substituted.
Nuisance elimination uses Cholesky solves and a position Schur complement;
only the 2x2 or 4x4 position block is inverted through solves. The result retains
between-emitter correlations, in `(y0, x0, y1, x1)` order and pixel² units.
`nuisance_fixed_covariance` is a comparison diagnostic, not the reported joint
uncertainty. There is no statistical ridge or eigenvalue flooring.

Stationarity is recomputed from observations. Parameter/background boundaries,
nonpositive curvature and singular nuisance blocks withhold covariance. An
absent broad component is explicitly conditioned absent. For numerical
consistency, flux within 1e-10 of zero in flux/1000 units (1e-7 photons) is also
conditioned absent only if zero is feasible and removing it changes no pixel
mean by more than 1e-10 times `max(1, mean)`. This uses the geometry solver's
active-bound tolerance, not a statistical detectability threshold. The observed
Hessian and full constrained stationarity check are recomputed at zero flux;
all other boundary and curvature guards remain. Positive broad flux outside
these numerical guards is not discarded. Input fits are not modified.

The native record and public `PositionUncertainty.broad_conditioned_absent`
expose this conditioning, including when another validity check withholds
covariance. These local checks do not establish repeated-draw coverage or
source-presence probabilities.

## Verification and remaining scope

`tests/fixtures/08_inference.npz` is an immutable synthetic pixel-integrated
Gaussian calibration family. Its JSON records units, provenance and a content
hash. It tests numerical contracts, not a real microscope calibration. Preserve
its stored values rather than regenerating it to fit a changed implementation.

```sh
maturin develop --release -m rust/spotsolve-py/Cargo.toml
python scripts/check_inference.py --rust --output reference/rust-inference-check/results.json
python -m pytest -q
cargo test --manifest-path rust/Cargo.toml --workspace
```

Controls cover means/derivatives, affine stationarity and feasibility, noisy
pairs, signed backgrounds, active pixel/coordinate boundaries, absent light,
zero images, coincident sources, uncertainty rejection, ownership, full count
nesting and clean 0/1/2-source reference agreement. A fixed noisy-haze draw tests
native feasibility/stationarity and likelihood regression. The frozen Python
full search fails an inner solve on that draw; it is not used as an assumed
oracle. Native operation with Python numerical methods disabled and import
isolation are tested. Timings are observations, never assertions of throughput.
Validation with the rebuilt release wheel: 88 Python tests and 55 Rust tests
passed; one existing slow Python check was skipped.

Next work should extend native inference where the scientific requirements
need it, rather than mirror all Python diagnostics. Optional profile curves
remain available in the reference. Source-presence probabilities, camera
likelihoods and calibrated full-frame decisions remain unresolved. Tracking
integration is deferred; an eventual table can contain coordinates, uncertainty
and properly defined emitter-presence probabilities.
