# Changelog

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
