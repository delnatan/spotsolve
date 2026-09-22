# spotsolve

Emitter localization for sparse to moderately crowded fluorescence images,
with Brownian-motion trajectory linking through frame-to-frame LAP assignment.
Fit overlapping emitters jointly, or use the Aguet / `spotfitlm` sparse
baseline. Both return positions, fluxes, widths and uncertainties for linking.
Detection and fitting run in Rust, with movie frames processed in parallel.

## Install

Requires Python 3.10+ and a Rust toolchain. In an activated environment:

```sh
pip install -e ".[dev]"
maturin develop --release -m rust/spotsolve-py/Cargo.toml
```

Add `.[scripts]` for movie I/O and plotting, or `.[test]` for Python tests.

## Choose a detector

| Mode | Entry points | Decision rule |
|---|---|---|
| Multi-emitter, default | `localize`, `localize_stack` | Joint fits; fixed 10-nat cost plus optional `count_penalty` |
| Multi-emitter, experimental BIC | Same functions, `selection="bic"` | Compare background-only and several fitted emitter counts |
| Sparse reference | `localize_aguet`, `localize_aguet_stack` | Aguet screening, then one independent `spotfitlm` fit per candidate |

BIC asks whether fewer emitters explain the same pixels, at the cost of more
fits. The sparse baseline follows the original `spotfitlm`; overlapping
sources can bias its independent fits. Detection is per frame, before linking.
None of these controls guarantees a frame-wide false-positive rate.

## Detect spots

```python
import spotsolve

# frame: (H, W), stack: (T, H, W), in camera units (ADU).
locs = spotsolve.localize(frame, sigma=1.45, offset=100)
movie = spotsolve.localize_stack(stack, sigma=1.45, offset=100, n_threads=5)

# More conservative count selection; penalty 2 is an example, not a calibration.
locs = spotsolve.localize(
    frame, sigma=1.45, offset=100, selection="bic", count_penalty=2,
)

# Sparse reference; mask is an optional (H, W) boolean array.
sparse = spotsolve.localize_aguet(frame, sigma=1.45, offset=100, roi=mask)
sparse_movie = spotsolve.localize_aguet_stack(
    stack, sigma=1.45, offset=100, roi=mask, n_threads=5,
)
```

All return `Localizations` (one per frame for stacks):

| Field | Meaning |
|---|---|
| `positions` | `(N, 2)` coordinates `(y, x)` in pixels; centers are integers |
| `amplitudes` | `(N,)` total Gaussian flux above the offset |
| `peak` | `(N,)` model signal above background for a pixel centered on the emitter |
| `se` | `(N, 3)` standard errors of `(flux, y, x)` |
| `fit_sigma`, `sigma_se` | Fitted width and its standard error, in pixels |
| `sigma_ratio` | `fit_sigma / sigma` |
| `flags` | `(N,)` bitmask of `FitFlag` diagnostics; rows are retained |
| `background` | Background map; for Aguet, a screening estimate with NaNs outside the crop |
| `dispersion` | Estimated variance per unit signal for the multi-emitter detector; NaN for Aguet |
| `info` | Method-specific fit counts and diagnostics; Aguet includes failed fits |
| `model_image`, `residual` | Optional diagnostic images |

Flux and background use ADU above `offset`; divide flux by camera gain for
photoelectrons. The multi-emitter PSF is pixel-integrated. Aguet uses a sampled
Gaussian and reports continuous flux `2*pi*peak*sigma_fit**2`, including
amplitude-width covariance in its flux uncertainty.

`peak` is derived from flux and fitted width. It helps compare a spot with
the image background, but varies with width and can exceed the brightest
observed pixel when the emitter lies between pixels. Localization tables
include `peak`; brightness-aware linking uses total flux.

Multi-emitter uncertainties use the final, undamped expected Fisher matrix,
scaled by local dispersion. If covariance cannot be computed, errors are NaN.
`locs.info["fisher_fraction"]` contains one row per detection and columns
`(flux, y, x, sigma)`: values near zero indicate strong coupling to other
fitted parameters; 1 means no coupling. This diagnostic is independent of
parameter units and does not change count selection. All arrays stay aligned
with the returned rows. See
[curvature and uncertainty](docs/DETECTION.md#curvature-and-uncertainty).

The detectors' fitted widths differ slightly by convention: pixel integration
adds `1/12` pixel² to the profile variance, so Aguet's sampled-Gaussian width
is typically larger. See [PSF conventions](docs/AGUET_BASELINE.md#reference-and-method).

## Detection controls

Both detectors take `sigma` in pixels, `offset=0`, and optional `roi`.
The ROI restricts searching and crops work to its bounding box plus context;
fitted positions can lie outside it. Use one mask covering the desired area,
not separate tile calls. A scattered mask may save little work. Aguet selects
integer seed centers; the multi-emitter detector also uses masked pixels to
estimate its background reference level.

| Multi-emitter setting | Default | Effect |
|---|---|---|
| `threshold` | 2.75 | LoG proposal cut in local noise units; higher is stricter and cheaper |
| `selection` | `"fixed"` | Fixed cost or experimental `"bic"` count search |
| `count_penalty` | 0 | Extra non-negative cost per emitter in either mode |
| `k_max` | 12 | Maximum emitters fitted jointly in one box |
| `slack` | `(0.7, 2.2)` | Allowed fitted widths, relative to `sigma` |

The fixed cost is `10 + count_penalty` in dispersion-scaled likelihood units.
BIC minimizes `I/phi + K*(2*log(n_pixels) + count_penalty)` over a bounded
forward/backward search, including background-only and removals after
refinement. Larger penalties favor fewer emitters. This is a BIC-inspired
score, not calibrated Bayesian evidence. See [count selection](docs/COUNT_SELECTION.md)
for measured recall/error tradeoffs, cost and limitations.

Aguet instead takes `significance=0.05` (smaller is stricter), odd
`boxsize=9`, and `itermax=50`. It applies no width-reporting band. Its Poisson
fit retains the reference noise assumptions; the multi-emitter detector
estimates local noise and dispersion from the image. Details:
[multi-emitter method](docs/DETECTION.md), [Aguet baseline](docs/AGUET_BASELINE.md).

## Masks, images and speed

Stack functions default to all machine cores; `n_threads` limits the shared
native frame workers. Results stay in frame order and agree across worker
counts. Parallelism improves movie throughput, not an individual frame's
latency. `images=False` avoids returning model/residual images; it is the
default except for single-frame `localize`.

Measured on the development Apple Silicon host:

| Workload | Serial | Five workers |
|---|---:|---:|
| BIC, first five real 128x128 GEM frames | 2.810 s | 0.755 s |
| BIC, all 49 frames of that crop | 27.865 s | 6.691 s |
| Aguet, 24 simulated sparse 128x128 frames | 26.3 ms | 7.32 ms |
| Aguet, same frames with a quarter-frame ROI | 8.81 ms | 2.68 ms |

These are different workloads, not a detector accuracy/speed comparison.
The original `spotfitlm` took 172.1 ms on the sparse full-frame benchmark:
6.5× the native serial time. See [BIC measurements](docs/COUNT_SELECTION.md#speed)
and [Aguet measurements](docs/AGUET_BASELINE.md#validation-and-speed) for conditions
and reproduction commands.

## Choose a detection width

`sigma` is an approximate search scale, not an exact width shared by every
emitter. Each spot's width is fitted; the multi-emitter fit uses the bounds
`slack * sigma`. Inspect a `fit_sigma` histogram from the first frame or a
few frames to choose a representative scale. Aguet provides a cheap initial
pass when spots are sufficiently isolated:

```python
import numpy as np

preview = spotsolve.localize_aguet_stack(stack[:3], sigma=1.2, offset=100)
widths = np.concatenate([result.fit_sigma for result in preview])
counts, bin_edges = np.histogram(widths, bins="auto")
# Plot counts against bin_edges with your preferred plotting tool.
```

Choose from the main isolated-spot population rather than automatically
averaging broad objects and overlaps. In crowded images, use an isolated ROI
or the joint detector for this inspection. Aguet's sampled-Gaussian widths
differ slightly from integrated widths; an approximate starting scale does
not need an exact conversion. If fitted widths reach the optimization bounds,
reconsider `sigma` or `slack`. There is no separate width-calibration API.

## Link detections

For a compact filtering workflow for either detector, see the
[localization-quality guide](docs/LOCALIZATION_QUALITY.md).

```python
import polars as pl
from spotsolve import loctable

parts, next_id = [], 0
for t, result in enumerate(movie):
    rows, _ = loctable.frame_tables(result, frame=t, loc_id0=next_id)
    parts.append(rows)
    next_id += len(rows)
locs = loctable.concat(parts)
# Optional post-processing; the original table retains every measurement.
usable = locs.filter(pl.col("flags") == 0)
usable = loctable.filter_quality(usable)  # require usable coordinates and errors
# Optional precision cut: max_se_pos=0.5 (pixels; calibrate for your data).

tracks = spotsolve.link(usable)
tracks.select("track_id", "frame", "y", "x")

params = spotsolve.fit_link_params(usable)  # inspect or reuse estimated parameters
tracks = spotsolve.link(usable, params)

# Optional conservative linking; 1 nat is an example, not a calibrated cutoff.
tracks = spotsolve.link(usable, params, min_link_margin=1.0,
                       min_track_length=4, diagnostics=True)
accepted = tracks.filter(pl.col("track_accepted"))
```

Linking uses the frame-to-frame LAP stage of Jaqaman-style tracking with
Brownian-motion costs. It preserves rows and columns and adds `track_id`.
Motion, continuation and error scaling are estimated from the movie.
`brightness=True` optionally
uses flux and flux uncertainty. A missed detection ends a track; there is no
gap closing or merging/splitting. See [tracking](docs/TRACKING.md) for the
motion model, parameters, benchmarks and limits.

## Development

```sh
python -m pytest -q
cargo test --release --workspace --manifest-path rust/Cargo.toml
```

Tests cover simulation recovery, singular-covariance handling, fit flags,
Fisher diagnostics, tables and Brownian-motion linking. Aguet is checked against
frozen original `spotfitlm` fits, including covariance, masks and worker counts.

[Documentation index](docs/README.md) · [Source map and validation](docs/README.md#implementation)
· [Historical experiments](docs/archive/README.md)

## Measurement-first output

Localization does not discard fitted spots for brightness or width. Use
`fit_sigma`, `sigma_se`, `flux`, `se_flux` and fit flags to choose downstream
criteria. Candidate screening and emitter-count selection still determine
which sources are fitted; this is not an exhaustive list of image maxima.

`FitFlag` reports edge support, non-convergence, stalling, unavailable
covariance, active bounds and unsettled neighboring-light context. Zero flags
does not prove a correct PSF or calibrated uncertainty. The three-sigma edge
support and numerical tolerances are explicit conventions, not object classes.
See [fit diagnostics](docs/LOCALIZATION_QUALITY.md).

API changes: `band`, `BAND`, `BAND_Z`, aggregate helpers and columns,
`REJECT_DTYPE`, `Localizations.rejects`, `loctable.reject_table`,
`calibrate_sigma`, and `SigmaCalibration` have been removed. `frame_tables`
now returns `(localizations, frame_summary)`.
The movie script saves those two tables and metadata; it no longer writes
aggregate or reject tables. Rebuild the Rust extension when updating.
