# Documentation

Start with the [README](../README.md) for installation, API examples and
linking.

| Guide | Contents |
|---|---|
| [Building and distribution](BUILDING.md) | Platform wheels, CI checks, source builds and releases |
| [Detection](DETECTION.md) | The model, the algorithm and its departures from u-track, the threshold, validation and limits |
| [Localization quality](LOCALIZATION_QUALITY.md) | Flags, uncertainties, `z`, and how to choose cuts |
| [Tracking](TRACKING.md) | Motion model, assignment, parameters, benchmarks and limits |

## Implementation

Python wraps native results as `Localizations`. Stack APIs use
`rust/spotsolve-core/src/frames.rs` for independent, ordered frame processing
with reusable worker storage. There is no nested thread pool.

| Component | Python | Rust core |
|---|---|---|
| Detection | `src/spotsolve/native.py` | `detect.rs` (seeds, windows, decisions), `prefilter.rs` (screen, threshold), `fit.rs` (window fit) |
| Shared PSF, filters, algebra | Native bindings | `psf.rs`, `filters.rs`, `linalg.rs`, `statistics.rs` |
| Tables and results | `src/spotsolve/loctable.py`, `results.py` | — |
| Tracking | `src/spotsolve/tracking.py` | `track.rs`, `lap.rs` |

Core files live under `rust/spotsolve-core/src`; bindings are in
`rust/spotsolve-py/src`.

## Verification

After rebuilding the release extension:

```sh
python -m pytest -q
cargo test --release --workspace --manifest-path rust/Cargo.toml
```

`layer3_fit.rs` holds the window fit to finite differences and its errors to
the reported covariance; `layer7_localize.rs` holds the detector to recall,
precision and error calibration on simulated fields and to its
false-positive rate on noise. Tracking tests check each frame's assignment
against enumeration. Fixtures under `tests/fixtures` are frozen.

Benchmark runners:

- `scripts/benchmark_detection.py`: recall, precision, pairs and noise on
  seeded scenarios; reports from different versions compare directly.
- `scripts/benchmark_detector_speed.py`: real-stack timing and output
  fingerprints.

`scripts/localize_movie.py` exports results and accepts `--fp-per-mpx`,
`--mixtures` and `--threads`.
