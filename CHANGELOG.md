# Changelog

## Unreleased

No API changes; detections differ from 0.3.0.

- Components wider than `slack[1] * sigma` are out of focus: they are
  fitted and tested like emitters, up to half the background node spacing,
  but returned as background (`info["out_of_focus"]` counts them). The
  default `slack` is `(1.0, 2.25)`.
- Seeds are scored on the residual orthogonal to the background nodes, so
  no background estimate enters their mean.
- Additions are made in batches: every maximum of the efficient score in a
  group becomes a candidate at once, one refit takes them all in, and each
  must still cost the bar to drop while together they gain it apiece. The
  efficient score projects out the nodes' and members' Newton steps.
- Count decisions profile the background nodes exactly on the group's
  patch instead of to second order, which stops adds and removals cycling
  on dense or defocused frames.
- On beads in 80% glycerol, precision 0.86 and recall 0.94 against hand
  labels (0.3.0: 0.65 and 0.98). Slower than 0.3.0: about 2.1 s per
  256x256 glycerol frame (0.7 s), and 1.5-4x on dense simulated frames,
  mostly in the out-of-focus components and the add tests.

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
