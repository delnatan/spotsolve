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
locs = spotsolve.localize(frame, sigma=1.27, offset=100.0, gain=1.93,
                          read_noise=2.41)
locs.positions      # (N, 2): y, x in pixels
locs.amplitudes     # (N,): total photoelectrons
locs.se             # (N, 3): standard errors of flux, y, x
locs.fit_sigma      # (N,): each emitter's fitted width
locs.rejects        # fits outside the reporting band: too_narrow, too_wide, edge

# A whole (T, H, W) movie, frames on all cores:
movie = spotsolve.localize_stack(stack, sigma=1.27, offset=100.0, gain=1.93,
                                 read_noise=2.41)
```

`sigma`, `offset`, `gain` and `read_noise` come from your calibration: input
is converted as `(image - offset) / gain`, and the read noise (electrons rms)
enters the likelihood so that read-noise spikes at low background are not
reported as spots. `gain=None` estimates the gain from the frame; prefer a
measured one.

`roi=` (a boolean mask) confines the search. Pass one ROI covering everything
you want in a single call rather than tiling a frame: a source just outside
an ROI is fitted where it actually is, so separate tiles can report the same
source twice at their seams.

## Calibrate the PSF width

```python
cal = spotsolve.calibrate_sigma(stack, sigma_guess=1.2, offset=100.0,
                                gain=1.93, read_noise=2.41)
cal.sigma, cal.ci   # median fitted width, px, and its 95% bootstrap interval
```

It localizes with the reporting band off, takes the median of every fitted
width, and repeats at that value until it settles; the guess need only be
within ~25%. On a field of one width it lands within ~1-3% of the truth. A
wider sub-population -- out-of-focus beads, say -- pulls the median up;
`cal.widths` holds every fitted width for a closer look.

`spotsolve.loctable` turns results into `polars` tables, and
`spotsolve.flag_aggregates` flags over-bright spots after the fact.

## Development

The detector is `rust/spotsolve-core/src/boxsearch.rs`, bound in
`rust/spotsolve-py`. `spotsolve.deprecated` is the Python reference
implementation (`deprecated/box.py` holds the measurement behind every
constant); the native path is held to statistical parity with it by
`tests/test_localize.py`, and it will be retired once the Rust is hardened.
See [docs/README.md](docs/README.md).
