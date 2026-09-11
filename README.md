# spotsolve

Spot detection and localization for fluorescence microscopy. The detector
decides how many emitters there are and where, jointly: in each small box of
the frame, an emitter exists iff it lowers the box's Poisson deviance by a
fixed number of nats, and every emitter is fitted at its own width. It runs
entirely in Rust, and a timecourse's frames run in parallel.

## Install

Requires Python and a Rust toolchain. In an activated virtual environment:

```sh
pip install -e ".[dev]"
maturin develop --release -m rust/spotsolve-py/Cargo.toml
```

## Localize

```python
import spotsolve

# frame: a 2D array in ADU; sigma: the in-focus PSF width in pixels.
result = spotsolve.localize(frame, sigma=1.27, offset=100.0, gain=2.0,
                            read_noise=1.6)
result.positions      # (N, 2): y, x in pixels
result.amplitudes     # (N,): total photoelectrons
result.se             # (N, 3): standard errors of flux, y, x
result.fit_sigma      # (N,): each emitter's fitted width
result.width_rejects  # fits outside the reporting band: too_narrow, too_wide, edge

# A whole (T, H, W) movie, frames on all cores:
movie = spotsolve.localize_stack(stack, sigma=1.27, offset=100.0, gain=2.0,
                                 read_noise=1.6)
```

`sigma`, `offset`, `gain` and `read_noise` come from your calibration: input
is converted as `(image - offset) / gain`, and the read noise (electrons rms)
enters the likelihood so that read-noise spikes at low background are not
reported as spots. `gain=None` estimates the gain from the frame; prefer a
measured one. The median of `result.sigma_ratio` over bright spots reads 1.0
when `sigma` is right.

`roi=` (a boolean mask) confines the search. Pass one ROI covering everything
you want in a single call rather than tiling a frame: a source just outside
an ROI is fitted where it actually is, so separate tiles can report the same
source twice at their seams.

`spotsolve.loctable` turns results into `polars` tables, and
`spotsolve.flag_aggregates` flags over-bright spots after the fact.

## Sparse emitters

`localize_sparse` is the cheap alternative for isolated spots whose fitting
windows do not overlap: one Aguet significance pass, then an independent fit
per candidate, with no model selection between overlapping sources.

```python
spots = spotsolve.localize_sparse(image, sigma=1.2, offset=100.0, gain=2.4)
```

## Development

`src/spotsolve/box.py` is the detector's Python reference: every constant's
measurement lives there, and the native port (`rust/spotsolve-core/src/
boxsearch.rs`) is held to statistical parity with it by
`tests/test_localize.py`. Change the algorithm there first, measure it, then
port. See [docs/README.md](docs/README.md).
