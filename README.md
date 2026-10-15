# spotsolve

Emitter localization for fluorescence movies, after u-track's
`pointSourceDetection`, and frame-to-frame linking in which each track moves
on its own diffusion scale. Detection and fitting run in Rust, frames in
parallel.

## Install

Requires Python 3.10+. Install the wheel for your platform from a GitHub
release with `python -m pip install path/to/filename.whl`; wheels bundle the
Rust extension. For development, with Rust installed:

```sh
python -m pip install -e ".[dev,test]"
maturin develop --release    # after changing Rust code
```

See the [build guide](docs/BUILDING.md).

## Detect spots

```python
import spotsolve

# stack: (T, H, W) in camera units (ADU).
gain = spotsolve.estimate_gain(stack, sigma=1.45, offset=100)   # or the camera's calibration
movie = spotsolve.localize_stack(stack, sigma=1.45, gain=gain, offset=100,
                                 fit_mixtures=True)
locs = movie[0]
locs.positions, locs.amplitudes, locs.se    # (y, x) px; flux; SEs of (flux, y, x)
```

Each pixel's variance is `read + gain * light`, the camera's noise at the
light the model puts on it. The frame is screened by a score test, seeds are
placed at local maxima, and each seed's window is fitted with a constant
level and pixel-integrated Gaussians at maximum quasi-likelihood under that
noise (the Poisson likelihood for a Poisson camera). A spot is kept only if
it gains `u^2/2` nats of quasi-likelihood, where the movie sets `u`: the bar
at which a spot is real, not noise, with probability `p_real`. A frame's
result therefore depends on the stack it is in; `localize(frame, ...)` treats
one frame as a movie of one. Standard errors come from each spot's Fisher
information. See [detection](docs/DETECTION.md) for the method.

| Setting | Default | Effect |
|---|---|---|
| `sigma` | required | In-focus PSF width, px |
| `gain` | required | Pixel variance per ADU of light: the conversion gain (ADU per photoelectron) times any excess-noise factor (2 for an EMCCD). It scales every test and standard error |
| `offset` | 0 | Camera offset, ADU |
| `p_real` | 0.9 | Least probability that a reported spot is real; higher is stricter |
| `fit_mixtures` | False | Fit several emitters per window, so neighbours are fitted rather than biasing each other. Use it wherever spots come closer than about `6 sigma` |
| `motion_blur` | False | With mixtures, a close pair is one elongated spot (flagged `ELONGATED`) unless two fit better, for exposures in which emitters streak |
| `width` | `(1.0, 1.5)` | Reported widths, multiples of `sigma`; wider light is fitted as out-of-focus background. Equal bounds fix the width |
| `roi` | None | `(H, W)` bool mask of where seeds may be placed |

**Choosing `sigma`.** Detect a few frames and take `sigma` from the narrow,
in-focus end of the `fit_sigma` histogram. A pile-up at the lower bound means
`sigma` is set too wide.

**Choosing `gain`.** From the camera's calibration, or `estimate_gain`, which
reads the residuals of a movie's bright spots. The read variance is measured
from the background given the gain.

**Long runs.** Pass `progress=spotsolve.Progress()` to `localize_stack` and
read it from another thread; `progress.cancel()` stops the run. Split a stack
into several calls and each becomes a movie of its own.

Fits are reported, not discarded: each row carries `FitFlag` diagnostics,
and cuts are left to the [localization-quality guide](docs/LOCALIZATION_QUALITY.md).

## Link detections

```python
from spotsolve import loctable

parts, next_id = [], 0
for t, result in enumerate(movie):
    rows, _ = loctable.frame_tables(result, frame=t, loc_id0=next_id)
    parts.append(rows)
    next_id += len(rows)
locs = loctable.filter_quality(loctable.concat(parts))

tracks = spotsolve.link(locs, max_step=6.5)    # pixels
```

Each track predicts its next step from its own diffusion coefficient,
inferred from its steps so far, so immobile, slow and fast particles in one
movie are each linked on their own scale. `max_step` is the one setting: the
largest step considered, about three times the rms step of the fastest
particles of interest. The linker reads each detection's `se_y` and `se_x`
and measures everything else from the movie. A missed detection ends a
track. See [tracking](docs/TRACKING.md) for the model.

## More

[Documentation index](docs/README.md) · [Changelog](CHANGELOG.md)
