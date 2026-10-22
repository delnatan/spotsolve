# Changelog

## Unreleased

Links are unchanged; one column is added.

- `link` adds `link_p`, the probability that the link into each row joins
  one particle's detections (null where a track starts). It comes from
  belief-propagation marginals over each frame pair's matchings, from both
  directions of time. On simulated movies its mean is within a few points of
  the share of links that are right, and errs low. `link_p=False` leaves it
  out. `spotsolve_rs.track_link(..., link_p=True)` returns `(track,
  link_p)`.

## 0.6.0 (2026-10-17)

The linker's links change wherever the table has `detect_z`.

- `link` reads the continuation per detection-strength class when the table
  has `detect_z` (`strength=`; `None` for the old behaviour), and starts new
  tracks from the movie's measured population over D. On simulated beads
  diffusing in and out of focus, D's bias goes from -7% to -1% and bright
  particles' tracks are no longer cut at their longer steps.
- The population is measured in both directions of time, and equal
  strengths share a class, so row order and reversing the frames still give
  the same links.

## 0.5.4 (2026-10-15)

One column renamed; results are unchanged.

- `z` is renamed `detect_z`, in localization tables and in
  `result.info`. Beside `y` and `x` it read as an axial coordinate; it is
  each emitter's detection strength, `sqrt(2 * likelihood ratio)` in
  standard-normal units. Tables written before keep the column `z`.

## 0.5.3 (2026-10-15)

Detection fixes. Movies are unchanged; single small frames and small ROIs
report fewer noise maxima.

- A candidate is no longer evidence for itself. `P(real | z)` is judged on
  `f1` fitted without the candidate nearest to `z`. Where noise expects
  well under one candidate (a small frame or ROI), a lone noise maximum
  raised `f1` at its own height and was reported: pure-noise frames of
  20x20 px gave about 150 false emitters per 10^6 px, 40x40 about 16; now
  about 5 and 1. A lone faint emitter in such a frame is now judged on the
  prior alone.
- The quasi-likelihood deviance is computed without cancellation. Its
  rounding error grew as `(read / gain)^2`; with read noise far above shot
  noise (a tiny `gain`) fits stalled and noise was reported.

## 0.5.2 (2026-10-15)

Linking fixes. Track ids change on most movies; the API does not.

- The linker's continuation `q`, which prices a track end, counted only
  detections that stayed within half the detector gap, so it measured how
  many particles barely move rather than how many are seen again. How a
  particle linked depended on what else shared the movie: without immobile
  particles slow recall fell from 0.87 to 0.75, and a movie of fast particles
  alone linked 5% of its steps. `q` is now the next-frame detections within
  `max_step` in excess of the same-frame neighbour count, which reads every
  continuation however far it moved.
- A link must now beat its alternative by `c = 2.5` nats, the price of an
  identity switch against a break. Before, this margin came from the biased
  `q` and varied with the movie's mobility.
- `r0`, within which a detection in another frame is at the same spot, no
  longer exceeds `max_step`. With localization errors large against
  `max_step` it could, and the occupancy prior then missed detections.
- The detector-gap histogram stores only its occupied bins, so localization
  errors far finer than `max_step` no longer cost memory in proportion to
  `max_step / se`.
- `link` refuses a `frame` column with missing, fractional, non-numeric or
  out-of-range values; they were cast silently. Frame numbers at the ends of
  the 64-bit range no longer overflow.

## 0.5.1 (2026-10-14)

The detection bar is now Bayesian and set per movie. Breaking API change.

- `fp_per_mpx` / `FP_PER_MPX` are replaced by `p_real` / `P_REAL` (default
  0.9): the least probability that a reported emitter is real, not noise.
  Every frame's spots, fitted alone at a low bar, give the density of
  likelihood-ratio heights; the noise maxima's density is known from the
  Euler-characteristic formula, and `P(real | z)` is the part it does not
  explain (empirical Bayes, two groups). `u` is where that reaches
  `p_real`, so it adapts to each movie; `localize` treats one frame as a
  movie of one. Each report carries `info["p_real"]`, also a `p_real`
  table column.
- Seeds are placed at one fixed low bar on the score at `sigma`, a full
  unit under every bar the posterior sets, so a maximum above `u` is no
  longer lost for want of a seed. The width bank of seed bars is removed.
- With single fits a movie is fitted once; with mixtures, once alone and
  once at `u`.
- On pure noise `u` rises to about 5: under 2 false emitters per 10^6
  pixels in movies (16 before).
- A component fitted wider than the reported widths is out-of-focus light
  only if its width is 2 standard errors beyond the upper bound (holding it
  there costs 2 nats); otherwise it is reported at the bound, flagged
  `AT_BOUND`. Deciding on the fitted width alone dropped half of the
  emitters on the bound, so particles near it vanished from frame to frame.

The noise model is the camera's, and the camera's gain is an input.
Breaking change: `localize` and `localize_stack` require `gain`, and the
output version is 9.

- Each pixel's variance is `read + gain * light`, the light the model puts
  on it. Every window is fitted at maximum quasi-likelihood under that
  variance, by Fisher scoring with the weights of the current model, and
  every decision, and `z`, compares its deviance: the Poisson likelihood
  for a Poisson camera, so `P(real)` holds its null on dark sCMOS and EMCCD
  frames too. The scalar dispersion `phi = variance / mean` is gone: with
  read noise it overstated the variance at spot peaks and understated `z`
  on dark backgrounds, so faint spots went unreported.
- `gain` is pixel variance per ADU of light: the conversion gain times any
  excess-noise factor (2 for an EMCCD). The gain is the quasi-likelihood's
  dispersion: it scales every likelihood ratio and standard error, and
  decides whether a bright spot's misfit takes a second component.
- `estimate_gain(stack, sigma, offset=...)` measures it from a movie, from
  the leverage-corrected residuals of bright emitters, where the camera's
  calibration is not at hand.
- The read variance is measured from the background's fourth differences
  given the gain. `dispersion` is now the gain as given;
  `info["read_variance"]` and a `read_variance` frame-table column are new.
- `localize_stack(..., progress=Progress())` reports frames fitted from
  another thread, and `progress.cancel()` stops the run (`Cancelled`). A
  stack split into several calls is several movies, each with its own read
  variance and bar; one watched call is one movie.
- `motion_blur=True`, with mixtures, tests a close pair against one
  elongated spot (a rotated Gaussian): it is two emitters only if it gains
  `u^2 / 2` nats over it, so an emitter moving within the exposure is one
  spot, flagged `FitFlag.ELONGATED`, not two. Pairs closer than about
  `2 sigma` are then reported as one elongated emitter. Off by default:
  where exposures keep emitters round, mixtures resolve the pair.

## 0.5.0 (2026-10-06)

The linker is rebuilt for movies that mix mobilities; see
[tracking](docs/TRACKING.md). Breaking API change.

- Each track predicts its next step from its own diffusion coefficient,
  a posterior over a log grid from its linked steps, and a new track starts
  from its spot's occupancy in the surrounding frames. A link's gain is the
  log step density over the cost of an end, `lambda = (1 - q)^2 rho / q`.
- The detector gap, the continuation fraction `q` and the density `rho` are
  read from the movie without links. `max_step` is the only setting.
- The linker runs forward and on the reversed movie and keeps the links
  both make, so reversing a movie gives the same links.
- `link` now also reads `se_y` and `se_x`.
- Removed: Crocker & Grier's least squared displacement as the linker; it
  remains the reference in the tests.

## 0.4.0 (2026-10-01)

The detector is rebuilt after u-track's `pointSourceDetection`; see
[detection](docs/DETECTION.md). Breaking API changes.

- `localize` / `localize_stack` screen the frame by a Poisson score test,
  seed at LoG maxima and fit each seed on its own `ceil(4 sigma)` window.
  `fit_mixtures=True` fits several emitters per window, with neighbours as
  nuisance components; `max_mixtures` caps the components. Copies of one
  emitter from two windows are merged.
- Every decision is a likelihood ratio at one threshold `u` from
  `fp_per_mpx`, a closed-form Euler-characteristic density over position
  and width.
- The dispersion pairs the squared fourth difference with the local mean
  under the same weights, so a varying background or emitter light no
  longer biases it.
- `slack` is replaced by `width=(lo, hi)`, the reported widths as multiples
  of `sigma` (default `(1, 1.5)`); equal bounds fix the width. Wider light
  is fitted as out-of-focus background and counted in
  `info["out_of_focus"]`.
- `FitFlag.AT_BOUND` no longer marks a width on its lower bound, the
  in-focus width.
- Removed: the joint frame model, `localize_aguet` /
  `localize_aguet_stack`, `SLACK`, `info["fisher_fraction"]` and the
  `fisher_*` table columns, `FitFlag.CONTEXT_UNSETTLED`, and the PSF
  Jacobians in `spotsolve.psf`. Added: `info["z"]` and the `z` table
  column, `info["seed"]`, `info["seed_positions"]`, `info["mixture"]`,
  `MAX_MIXTURES`, `WIDTH`.
- `result.background` is the screening level, NaN outside the processed
  crop; each emitter's own fitted level is `info["fitted_background"]`.
- Results are invariant to the camera gain, and identical across worker
  counts.
- Speed on real 256x256 GEM frames, serial: 18 ms per frame with single
  fits, 0.24 s with mixtures (0.3.0: 1.7 s).
- Scripts take `--mixtures`; `scripts/benchmark_detection.py` scores the
  detector on seeded scenarios.

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
