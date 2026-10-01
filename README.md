# spotsolve

Emitter localization for fluorescence images, after u-track's
`pointSourceDetection`, with frame-to-frame trajectory linking by least
squared displacement. Each spot is fitted on its own window; with
`fit_mixtures`, overlapping spots are fitted jointly. Every decision is a
Poisson likelihood ratio at one threshold, set by an expected number of
false spots. Detection and fitting run in Rust, with movie frames processed
in parallel.

## Install

Requires Python 3.10+. Download the wheel for your platform from a GitHub
release or the **Build and test wheels** Actions artifacts, then install the
`.whl` file with `python -m pip install path/to/filename.whl`. Wheels bundle
the Rust extension; a Rust toolchain is only needed for source builds.

For development, with Rust installed and an environment activated:

```sh
python -m pip install -e ".[dev,test]"
```

After changing Rust code, run `maturin develop --release` from the repository
root. Add `.[scripts]` for movie I/O and plotting. See the
[build guide](docs/BUILDING.md) for supported platforms, release artifacts,
and migration from the old separate `spotsolve-rs` installation.

## Detect spots

```python
import spotsolve

# frame: (H, W), stack: (T, H, W), in camera units (ADU).
locs = spotsolve.localize(frame, sigma=1.45, offset=100)
movie = spotsolve.localize_stack(stack, sigma=1.45, offset=100, n_threads=5)

# Crowded spots: fit several emitters per window.
locs = spotsolve.localize(frame, sigma=1.45, offset=100, fit_mixtures=True)

# Stricter: 4 expected false emitters per 10^6 noise pixels instead of 16.
locs = spotsolve.localize(frame, sigma=1.45, offset=100, fp_per_mpx=4)
```

Both return `Localizations` (one per frame for stacks):

| Field | Meaning |
|---|---|
| `positions` | `(N, 2)` coordinates `(y, x)` in pixels; centers are integers |
| `amplitudes` | `(N,)` total flux of the pixel-integrated Gaussian, above the local level |
| `peak` | `(N,)` model signal above the level for a pixel centered on the emitter |
| `se` | `(N, 3)` standard errors of `(flux, y, x)` |
| `fit_sigma`, `sigma_se` | Fitted width and its standard error, in pixels |
| `sigma_ratio` | `fit_sigma / sigma` |
| `flags` | `(N,)` bitmask of `FitFlag` diagnostics; rows are retained |
| `background` | Screening level map, NaN outside the processed crop |
| `dispersion` | The frame's measured variance per unit signal |
| `info` | `u`, per-emitter `z`, `fitted_background`, `seed`, `mixture`, and work counts |
| `model_image`, `residual` | Optional diagnostic images |

Flux and background use ADU above `offset`; divide flux by camera gain for
photoelectrons. Uncertainties come from the expected Fisher information of
each emitter's final window fit, scaled by the frame's dispersion; NaN if it
cannot be computed. `info["z"]` is each emitter's `sqrt(2 * likelihood
ratio)`, at least the threshold `info["u"]`. `peak` varies with width and
can exceed the brightest observed pixel when the emitter lies between
pixels.

## Detection controls

| Setting | Default | Effect |
|---|---|---|
| `sigma` | required | In-focus PSF width, px |
| `offset` | 0 | Camera offset, ADU |
| `roi` | None | `(H, W)` bool mask of where seeds may be placed |
| `fp_per_mpx` | 16 | Expected false emitters per 10^6 pixels of noise; lower is stricter |
| `width` | `(1.0, 1.5)` | Reported widths, as multiples of `sigma`; wider light is fitted as out-of-focus background. The upper bound is capped near `2 sigma`. Equal bounds fix the width |
| `fit_mixtures` | False | Fit several emitters per window (u-track's FitMixtures) |
| `max_mixtures` | 20 | Components per window at most |

The frame is screened by a Poisson score test, seeds are placed at local
maxima, and each seed's window is fitted with a constant level and
pixel-integrated Gaussians. A spot is kept only if it gains `u^2/2` nats of
likelihood, `u` solved so that pure noise gives `fp_per_mpx` false spots per
10^6 pixels; measured rates are 13-17 at the default. Light wider than the
reported widths is fitted as out-of-focus background and counted in
`info["out_of_focus"]`. Use `fit_mixtures=True` wherever spots come closer
than about `4 sigma`: single fits are biased by a neighbour's light and
cannot separate pairs. See [detection](docs/DETECTION.md) for the method,
its departures from u-track and their measurements.

The ROI restricts seeding and crops work to its bounding box plus context;
fitted positions can lie outside it. Use one mask covering the desired area,
not separate tile calls.

## Masks, images and speed

Stack functions default to all machine cores; `n_threads` limits the
native frame workers. Results stay in frame order and are identical across
worker counts. `images=False` avoids returning model/residual images; it is
the default except for single-frame `localize`.

On a 10-core Apple M5, real 256x256 GEM frames (about 205 spots per frame
single, 550 with mixtures; best of three runs):

| Workload | Single fits | Mixtures | 0.3.0 joint model |
|---|---:|---:|---:|
| First five frames, serial | 0.09 s | 1.19 s | 8.26 s |
| First five frames, five workers | 0.02 s | 0.30 s | 2.15 s |
| All 49 frames, ten workers | 0.16 s | 1.90 s | 13.3 s |

## Choose a detection width

`sigma` is the in-focus PSF width: the narrowest a spot can be. Inspect a
`fit_sigma` histogram from a few frames and take `sigma` from its narrow,
in-focus mode:

```python
import numpy as np

preview = spotsolve.localize_stack(stack[:3], sigma=1.2, offset=100,
                                   fit_mixtures=True)
widths = np.concatenate([result.fit_sigma for result in preview])
counts, bin_edges = np.histogram(widths, bins="auto")
```

Many fits at the lower bound mean `sigma` is set too wide; to admit
narrower fits, lower `width[0]`, at the cost of a slightly stricter
threshold. Many at the upper bound mean it is set too narrow. To report
defocused emitters too, raise `width[1]` (up to about 2); more haze is
then reported as emitters. `width=(1, 1)` fixes every width at `sigma`, as u-track does: a
little more recall when spots really are all in focus, but wider spots are
split in two by mixtures.

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

tracks = spotsolve.link(usable, max_step=6.5)  # pixels
tracks.select("track_id", "frame", "y", "x")

# Tracks of at least four frames:
long = tracks.filter(pl.len().over("track_id") >= 4)
```

Between consecutive frames, links minimize the summed squared displacement,
and ending a track costs `max_step`². No step longer than `max_step` is
linked; about three times the rms step of the fastest particles of interest
is a good start. The result is the input table plus `track_id`, in the same
row order. A missed detection ends a track. See [tracking](docs/TRACKING.md)
for the model, how to choose `max_step`, and validation.

## Development

```sh
python -m pytest -q
cargo test --release --workspace --manifest-path rust/Cargo.toml
```

Tests cover the window fit's derivatives and error calibration, simulation
recovery, the false-positive rate on noise, fit flags, tables and linking.
`scripts/benchmark_detection.py` scores the detector on seeded scenarios.

[Documentation index](docs/README.md) · [Source map and validation](docs/README.md#implementation)

## Measurement-first output

Localization does not discard fitted spots for brightness or width. Use
`fit_sigma`, `sigma_se`, `flux`, `se_flux` and fit flags to choose downstream
criteria. Candidate screening and emitter-count selection still determine
which sources are fitted; this is not an exhaustive list of image maxima.

`FitFlag` reports edge support, non-convergence, stalling, unavailable
covariance and active bounds. Zero flags
does not prove a correct PSF or calibrated uncertainty. The three-sigma edge
support and numerical tolerances are explicit conventions, not object classes.
See [fit diagnostics](docs/LOCALIZATION_QUALITY.md).

Changes between versions, including removed and renamed API, are listed in
the [changelog](CHANGELOG.md). Rebuild the Rust extension when updating a
source install.
