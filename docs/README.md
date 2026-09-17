# Documentation

Start with the [README](../README.md) for installation, detector choice,
API examples and linking.

| Guide | Contents |
|---|---|
| [Multi-emitter detection](DETECTION.md) | Image/noise model, fixed-cost search, widths, masks and earlier validation |
| [Experimental BIC selection](COUNT_SELECTION.md) | Count score, forward/backward search, error budgets and threading measurements |
| [Aguet / spotfitlm baseline](AGUET_BASELINE.md) | Sparse screening/fitting, reference parity, uncertainties and speed |
| [Tracking](TRACKING.md) | Motion model, assignment, parameters, benchmarks and limits |
| [Archive](archive/README.md) | Retired designs and historical experiments |

## Implementation

Python wraps native results as `Localizations`. Both stack APIs use
`rust/spotsolve-core/src/frames.rs` for independent, ordered frame processing
with reusable worker storage. There is no nested thread pool.

| Component | Python | Rust core |
|---|---|---|
| Multi-emitter detection | `src/spotsolve/native.py` | `boxsearch.rs`, `lmcl.rs`, `patches.rs` |
| Aguet baseline | `src/spotsolve/aguet.py` | `aguet.rs` |
| Shared filters/algebra | Native bindings | `filters.rs`, `linalg.rs` |
| Sigma calibration | `src/spotsolve/calibration.py` | Uses multi-emitter detection |
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

The 2026-09-17 run passed **69 Python and 45 Rust tests**. Cleanup and scheduler
sharing left all output fields unchanged in 134 regression cases: 49 real
frames in both count-selection modes, plus three seeds in six simulation
conditions in both modes.

Fixtures under `tests/fixtures` are frozen. Most came from the retired Python
reference. `08_track.json` comes from `tracksolve`; `09_aguet.json` pins the
original `spotfitlm` revision and source hashes. Its generator,
`scripts/make_aguet_fixture.py`, remains available for audit, not routine
regeneration to accommodate a failing test. Aguet checks cover candidate
selection, 18 reference fits and full covariance, numerical derivatives,
mask context, thread determinism, output conversion and failures.

Benchmark runners:

- `scripts/benchmark_count_selection.py`: simulated recall/unmatched-count curves.
- `scripts/benchmark_detector_speed.py`: real-stack timing and output fingerprints.
- `scripts/benchmark_aguet.py`: original `spotfitlm` comparison; requires its
  sibling checkout and a C compiler, neither needed at runtime.

`scripts/localize_movie.py` exports multi-emitter results and accepts
`--selection`, `--count-penalty` and `--threads`. Aguet is currently exposed
through its Python API, not that script.
