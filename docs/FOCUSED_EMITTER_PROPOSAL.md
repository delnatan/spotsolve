# Focused-emitter development plan

The production direction is explicit: Rust owns numerical localization;
Python describes calibration, supplies arrays, and adapts results. The old
prototype implementations and executable experiment sweep have been removed.
Their findings remain in `docs/archive/` as provenance, while the independent
Python implementation under `spotsolve.inference.reference` is retained only
for focused numerical validation of the calibrated Rust solver.

## Dense regime

The calibrated dense local model contains a bicubic nonnegative sampled
background, one continuous defocused component, and zero, one, or two focused
emitters. Rust owns its proposal generation, screened multistart search,
constrained Poisson profiling, nested count hypotheses, observed curvature,
and conditional joint position covariance. See
[INFERENCE_CONTRACT.md](INFERENCE_CONTRACT.md) for the numerical contract.

This model is intended for interacting emitters and for focused light sitting
on broad out-of-focus light. Its highest-priority output is localization
uncertainty with nuisance and neighbor coupling. Fixed-count likelihood gains
remain diagnostics, not source-existence probabilities.

The current `spotsolve.detect` full-frame pipeline remains available while the
calibrated local model is connected to full-frame ownership and source-count
decisions. That integration should reuse the existing candidate and overlap
ideas without introducing Python optimizer callbacks.

## Sparse regime

`spotsolve.localize_sparse` is a deliberately smaller reference and a useful
method for sparse data. Its Rust candidate pass restores the Aguet detector:
at each pixel it fits a fixed-width Gaussian plus local constant by closed-form
linear regression, estimates noise from the local residuals, and tests whether
the peak amplitude clears that noise floor. The significance mask is
intersected with LoG local maxima. Each survivor then receives one independent
bounded Poisson fit with a scalar local background and pixel-integrated
Gaussian PSF. There is no iterative residual search, split, prune, or neighbor
refit.

The default local test size is `alpha=0.05`. Returned rows carry the test
statistic and one-sided p-value. This is the sensitive statistical detector
used by Aguet; it is not Benjamini-Hochberg correction or a guarantee on the
false-discovery fraction of a whole frame.

Width may be fixed at the supplied PSF sigma or fitted independently per
emitter within explicit ratio bounds. In fitted-width mode, sigma remains a
nuisance coordinate in the Fisher calculation, so position uncertainty
includes width uncertainty. Boundary, singular, and unconverged candidates are
not returned as localizations; `candidate_count` preserves the number of
significant maxima sent to the fitter.

The sparse method assumes candidate fitting windows do not materially overlap.
Its simplicity is the point: disagreement with the dense method as separation
decreases reveals interaction effects rather than a different optimizer or PSF
implementation. On an isolated calibrated Gaussian control, the two methods
agree on position and flux.

The Aguet pass is implemented as a standalone Rust proposal function so it can
be tested again in the dense search without changing dense acceptance. The
controlled comparison should replace only the initial proposal map while
holding the joint focused/defocused model and source-count decision fixed. A
previous trial that treated the local `alpha` decision as final dense-source
evidence tested a different algorithm and would predict false neighbors around
overlapping or broad emitters.

## Next work

1. The first paired native proposal controls are complete; see
   [results and limitations](DENSE_PROPOSAL_CONTROLS.md). Keep the current
   proposal default: both methods reach the same likelihoods on these controls.
   Numerical-zero broad-flux conditioning is now consistent and explicit in
   uncertainty results. Next validate uncertainty across repeated draws and
   address statistically weak broad light beyond numerical-zero tolerance.
2. Define and calibrate the focused-emitter source-presence output under haze
   and neighbors. Likelihood gain and LoG score must not be relabeled as
   probabilities.
3. Connect the calibrated dense fitter to full-frame interacting groups after
   its source decision is settled. Tracking integration remains deferred; the
   eventual table can carry coordinates, covariance/status, flux, width, and a
   properly calibrated presence probability.

Use small discriminating controls rather than a broad performance campaign:
isolated focused light, a focused source on broad light, equal and unequal
pairs, and a focused source on noisy haze. Runtime optimization follows only
after these numerical and statistical contracts are stable.
