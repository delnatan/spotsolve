# Changelog

## Unreleased (0.4.0)

The detector is rebuilt after u-track's `pointSourceDetection`. Breaking
API changes; see [detection](docs/DETECTION.md) for the method and its
measurements.

- `localize` / `localize_stack` screen the frame by a Poisson score test,
  seed at LoG maxima and fit each seed on its own `ceil(4 sigma)` window.
  `fit_mixtures=True` (u-track's FitMixtures) fits several emitters per
  window, with neighbours as nuisance components and efficient-score
  proposals; `max_mixtures` caps the components.
- Every decision is a likelihood ratio at one threshold `u` from
  `fp_per_mpx`, now a closed-form Euler-characteristic density over
  position and width, reported widths only (wider maxima are out-of-focus
  light). Measured false emitters on noise: 11-18 per 10^6 pixels at the
  default 16.
- The dispersion is the median of the squared fourth difference over the
  local mean under the same weights, not over the frame's median pixel. The
  old pairing read 0.84 of the true value on a background rising 3-40
  photons across the frame (3x the false emitters), and 1.1-1.3 on fields
  of emitters (lost recall).
- `slack` is replaced by `width=(lo, hi)`, the reported widths as multiples
  of `sigma` (default `(1, 1.5)`); equal bounds fix the
  width. Wider light is fitted as out-of-focus background and counted in
  `info["out_of_focus"]`.
- Removed: the joint frame model, `localize_aguet` /
  `localize_aguet_stack` (the spotfitlm port), `SLACK`,
  `info["fisher_fraction"]` and the `fisher_*` table columns, and
  `FitFlag.CONTEXT_UNSETTLED`. Added: `info["z"]` and the `z` table column
  (each emitter's `sqrt(2 * likelihood ratio)`), `info["seed"]`,
  `info["seed_positions"]`, `info["mixture"]`, `MAX_MIXTURES`, `WIDTH`.
- `FitFlag.AT_BOUND` no longer marks a width on its lower bound, the
  in-focus width, where about half of in-focus emitters fit (it marked 40%
  of isolated emitters).
- Mixtures merge copies of one emitter reported by two windows (each
  fit's nearest component to the other, closer than `sigma`): about 1% of
  reports at 0.04 / px^2, precision 0.95 to 0.96.
- `info["z"]` of a window left with one component after removals is its
  own gain, not a removed component's; the level-only null takes pixels
  below the offset as 0, as the fits do.
- `result.background` is the screening level, NaN outside the processed
  crop; each emitter's own fitted level is `info["fitted_background"]`.
- Results are invariant to the camera gain, and identical across worker
  counts.
- Speed on real 256x256 GEM frames, serial: 18 ms per frame with single
  fits, 0.24 s with mixtures (0.3.0: 1.7 s). On simulated fields mixtures
  come within a few points of the joint model's recall (0.96 at 0.005 /
  px^2, 0.85 at 0.02) and resolve pairs from 1.5 sigma; on a real bead
  image they report the same beads.
- Scripts take `--mixtures`; `scripts/benchmark_detection.py` scores any
  version on seeded scenarios.

## 0.3.0

No API changes; detections differ from 0.2.0.

- Seeds are local maxima of the score over position and width, and
  additions go in at the width the score asks for, so defocused emitters
  are proposed at their own width.
- The default `slack` is `(1.0, 2.2)`: `sigma` is the in-focus width, the
  narrowest a spot can be. Lower `slack[0]` to admit narrower fits.
- Every count decision, and the reported SEs and `fisher_fraction`, refit
  the background (to second order) instead of holding it at a fit that
  includes the emitter. Background nodes sit `ceil(8 * slack[1] * sigma)`
  px apart instead of 16, so the background leaves a wide emitter most of
  its flux information; SEs of wide emitters grow accordingly.
- `fp_per_mpx` is now derived rather than fitted: the expected Euler
  characteristic of the likelihood-ratio field over position and width. It
  is an upper bound for Gaussian noise; with the default 16, pure noise at a
  background of 20 gives 5-11 false emitters per 10^6 pixels (Gaussian) and
  11-17 (Poisson) for `sigma` 1-2. In 0.2.0 false emitters grew with
  `sigma`, to 40 per 10^6 Poisson pixels at `sigma` 2.
- Fewer fits per frame: shorter stamps, removal trials only where the Wald
  cost is below the bar, refits only where the model changed, and tests
  before full convergence. Dense frames run at about 0.2.0's speed; sparse
  frames 20-30% slower.
- `examples/characterize.rs` measures the detector against what the data
  allow: recall by oracle SNR and width, position error over the
  Cramer-Rao bound, close pairs, recall by neighbour distance, and false
  emitters on noise.

## 0.2.0

Incompatible API changes: see below.

- Multi-emitter detection is rewritten as one joint Poisson model per frame.
  It has separable emitter stamps, its own bounded Levenberg-Marquardt fitter,
  and seeds taken directly from the efficient score. The one knob,
  `fp_per_mpx`, is calibrated to within 3% on pure noise for `sigma` 1.0–1.45.
  Frames run 1.5–1.7x faster than in 0.1.0, and parallel frames give
  byte-identical results at every worker count.
- Linking is now Crocker–Grier: `link(locs, max_step)` minimizes the summed
  squared displacement between consecutive frames, and ending a track costs
  `max_step`². `max_step` (pixels) is required, and the linker reads only
  `frame`, `y` and `x`.
- Removed: `fit_link_params`, `LinkParams`, the `brightness`,
  `min_link_margin`, `min_track_length` and `diagnostics` options of `link`,
  and `K_MAX`. For a minimum track length, filter
  `pl.len().over("track_id")`.
- Native bindings are renamed `detect_localize`, `detect_localize_stack`,
  `detect_render` and `DETECT_*`, matching `aguet_*` and `track_*`.

## 0.1.0

First tagged release.

- Sparse and multi-emitter localization with fitted widths, fluxes and uncertainty diagnostics.
- Brownian-motion trajectory linking using frame-to-frame LAP assignment.
- Localization diagnostics retained for downstream filtering; width calibration and aggregate rejection removed.
- A single Python distribution bundles the Rust extension.
- Tested wheels for Linux x86-64 and ARM64, macOS Intel and Apple Silicon, and Windows x86-64.
- GitHub Actions builds release assets and tests each wheel on Python 3.10 and 3.14.
