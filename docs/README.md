# Documentation

Start with the [README](../README.md) for installation, detector choice,
API examples and linking.

| Guide | Contents |
|---|---|
| [Building and distribution](BUILDING.md) | Platform wheels, CI checks, source builds and releases |
| [Localization quality](LOCALIZATION_QUALITY.md) | A small common filter, on-demand diagnostics, and how to choose cuts |
| [Multi-emitter detection](DETECTION.md) | Image/noise model, the joint model, count rule, widths and uncertainties |
| [Aguet / spotfitlm baseline](AGUET_BASELINE.md) | Sparse screening/fitting, reference parity, uncertainties and speed |
| [Tracking](TRACKING.md) | Motion model, assignment, parameters, benchmarks and limits |
| [Archive](archive/README.md) | Retired designs and historical experiments |

## Implementation

Python wraps native results as `Localizations`. Both stack APIs use
`rust/spotsolve-core/src/frames.rs` for independent, ordered frame processing
with reusable worker storage. There is no nested thread pool.

| Component | Python | Rust core |
|---|---|---|
| Multi-emitter detection | `src/spotsolve/native.py` | `detect.rs`, `model.rs` |
| Aguet baseline | `src/spotsolve/aguet.py` | `aguet.rs` |
| Shared filters/algebra | Native bindings | `filters.rs`, `linalg.rs` |
| Tables and results | `src/spotsolve/loctable.py`, `results.py` | — |
| Tracking | `src/spotsolve/tracking.py` | `track.rs`, `lap.rs`, `trackparams.rs` |

Core files live under `rust/spotsolve-core/src`; bindings are in
`rust/spotsolve-py/src`. Rust retains current algorithm rules and numerical
invariants. Long benchmark commentary moved to
[detector design history](archive/DETECTOR_DESIGN_NOTES.md); halo collection
and frame scheduling are shared, while sparse and joint fitting remain
separate because their models differ.

## Verification

After rebuilding the release extension:

```sh
python -m pytest -q
cargo test --release --workspace --manifest-path rust/Cargo.toml
```

Tests cover assignment margins, minimum-length semantics, collinear indexing,
peak output, fit flags, Fisher diagnostics and invalidation of stale
uncertainties. `layer7_localize.rs` holds the detector to recall, precision
and position error on simulated fields and to its false-positive rate on
pure noise.

Fixtures under `tests/fixtures` are frozen. Most came from the retired Python
reference. `08_track.json` comes from `tracksolve`; `09_aguet.json` pins the
original `spotfitlm` revision and source hashes. Its generator,
`scripts/make_aguet_fixture.py`, remains available for audit, not routine
regeneration to accommodate a failing test. Aguet checks cover candidate
selection, 18 reference fits and full covariance, numerical derivatives,
mask context, thread determinism, output conversion and failures.

Benchmark runners:

- `scripts/benchmark_tracking.py`: exclusion-margin sweeps with nearby clutter,
  detector output, and drift; reports false links, coverage, and selection bias.
- `scripts/benchmark_detector_speed.py`: real-stack timing and output fingerprints.
- `scripts/benchmark_aguet.py`: original `spotfitlm` comparison; requires its
  sibling checkout and a C compiler, neither needed at runtime.

`scripts/localize_movie.py` exports multi-emitter results and accepts
`--fp-per-mpx` and `--threads`. Aguet is currently exposed
through its Python API, not that script.

- [Code audit and API migration](archive/AUDIT_2026-09-22.md): measurement output, fit diagnostics and numerical corrections.
