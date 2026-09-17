# spotsolve

Spot detection, localization and tracking for fluorescence microscopy.
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
| `se` | `(N, 3)` standard errors of `(flux, y, x)` |
| `fit_sigma`, `sigma_se` | Fitted width and its standard error, in pixels |
| `sigma_ratio` | `fit_sigma / sigma` |
| `rejects` | Multi-emitter fits outside the width-reporting band; empty for Aguet |
| `background` | Background map; for Aguet, a screening estimate with NaNs outside the crop |
| `dispersion` | Estimated variance per unit signal for the multi-emitter detector; NaN for Aguet |
| `info` | Method-specific fit counts and diagnostics; Aguet includes failed fits |
| `model_image`, `residual` | Optional diagnostic images |

Flux and background use ADU above `offset`; divide flux by camera gain for
photoelectrons. The multi-emitter PSF is pixel-integrated. Aguet uses a sampled
Gaussian and reports continuous flux `2*pi*peak*sigma_fit**2`, including
amplitude-width covariance in its flux uncertainty.

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
| `band` | `(0.8, 2.0)` | Reported widths, with a two-SE allowance; `None` reports all fits |

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

## Calibrate sigma

```python
cal = spotsolve.calibrate_sigma(stack, sigma_guess=1.2, offset=100)
cal.sigma, cal.ci    # fitted width and 95% bootstrap interval
cal.widths          # individual fitted widths
```

Calibration uses the multi-emitter detector with the reporting band off and
iterates the median width. A starting guess within about 25% worked in the
validation; broad or out-of-focus populations can pull the estimate upward.

## Link detections

```python
from spotsolve import loctable

parts, next_id = [], 0
for t, result in enumerate(movie):
    rows, _, _ = loctable.frame_tables(result, frame=t, loc_id0=next_id)
    parts.append(rows)
    next_id += len(rows)
locs = loctable.concat(parts)
locs = loctable.filter_aggregates(locs)  # optional: retain point emitters

tracks = spotsolve.link(locs)
tracks.select("track_id", "frame", "y", "x")

params = spotsolve.fit_link_params(locs)  # inspect or reuse estimated parameters
tracks = spotsolve.link(locs, params)
```

Linking preserves rows and columns and adds `track_id`. Motion, continuation
and error scaling are estimated from the movie. `brightness=True` optionally
uses flux and flux uncertainty. A missed detection ends a track; there is no
gap closing or merging/splitting. See [tracking](docs/TRACKING.md) for the
motion model, parameters, benchmarks and limits.

## Development

```sh
python -m pytest -q
cargo test --release --workspace --manifest-path rust/Cargo.toml
```

The 2026-09-17 verification passed **69 Python and 45 Rust tests**. Cleanup
and shared threading preserved every returned field in 134 dense-detector
regression cases. Aguet is checked against frozen original `spotfitlm` fits,
including covariance, masks and worker-count consistency.

[Documentation index](docs/README.md) · [Source map and validation](docs/README.md#implementation)
· [Historical experiments](docs/archive/README.md)
