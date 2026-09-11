# spotsolve

Spot detection, localization and tracking for fluorescence microscopy. It
answers two questions about an image together: how many emitters are in each
part of it, and where each one is. It decides both by fitting a model of the
image to the pixel counts. Over a movie it answers the third question, which
detections are the same particle, using each detection's own localization
error to do it. All of it runs in Rust, with a timecourse's frames in
parallel.

## Install

Requires Python and a Rust toolchain. In an activated virtual environment:

```sh
pip install -e ".[dev]"
maturin develop --release -m rust/spotsolve-py/Cargo.toml
```

## Detect spots

```python
import spotsolve

# frame: a 2D array in camera units (ADU).
locs = spotsolve.localize(frame, sigma=1.27, offset=100.0, gain=1.93,
                          read_noise=2.41)

locs.positions      # (N, 2): y, x in pixels; pixel centres are integers
locs.amplitudes     # (N,): total flux of each spot, photoelectrons
locs.se             # (N, 3): standard errors of flux, y, x
locs.fit_sigma      # (N,): each spot's own fitted width, pixels
locs.sigma_ratio    # (N,): fit_sigma / sigma
locs.rejects        # fitted objects that are not reported as spots, with a reason
locs.background     # (H, W): background, photoelectrons per pixel

# A (T, H, W) movie, frames on all cores:
movie = spotsolve.localize_stack(stack, sigma=1.27, offset=100.0, gain=1.93,
                                 read_noise=2.41)
```

The four calibration inputs:

| argument | meaning |
|---|---|
| `sigma` | the in-focus PSF width, pixels -- see [Calibrate sigma](#calibrate-sigma) |
| `offset` | camera offset, ADU |
| `gain` | ADU per photoelectron; `None` estimates it from the frame, but a measured value is better |
| `read_noise` | camera read noise, electrons rms (0 if unknown) |

`roi=` (a boolean mask) restricts where spots are searched for. Pass one mask
covering everything you want in a single call; do not tile a frame into
several calls, because a source on a tile border is fitted where its light
actually is and can be reported by both tiles.

## Spot width: fitted per spot, and reported only inside a band

Every spot's width is fitted, not held at `sigma`. `sigma` sets the scale of
the search and defines "in focus"; each spot's own width may range over
0.7-2.2 x `sigma` (`spotsolve.SLACK`), and only spots whose fitted width lies
in **0.8-2.0 x `sigma`** (`spotsolve.BAND`) are reported as detections.
Everything else is still fitted -- its light is part of the model -- and is
returned in `locs.rejects` with one of three reasons:

- **too narrow** -- fitted width below 0.8 x `sigma`. Nothing the microscope
  images is narrower than its PSF, so this is not a real spot: usually a
  noise spike, or a fit squeezed by its neighbours.
- **too wide** -- fitted width above 2.0 x `sigma`, away from the frame edge:
  an out-of-focus emitter, an extended object, or a patch of haze.
- **edge** -- outside the band and within 1 `sigma` of the frame border, where
  the border cuts the spot off and its width cannot be measured.

`band=` changes the reported range (`band=None` reports every fit), and
`slack=` the range a width may take.

Why widths are not held fixed: with one fixed width, the model can only
explain a broad blob of light as several narrow spots, so defocused emitters
and haze get "tiled" into false detections. Measured on simulations with a
spread of widths and on hazy fields, holding the width fixed invented 8-33
false spots per 64x64 frame against 0.7-7 with fitted widths; on the glycerol
and GEM data it put 136-211 detections per frame inside objects the fitted
search calls too wide. It only helps when every emitter truly has one width.

## Calibrate sigma

```python
cal = spotsolve.calibrate_sigma(stack, sigma_guess=1.2, offset=100.0,
                                gain=1.93, read_noise=2.41)
cal.sigma, cal.ci   # median fitted width, pixels, and its 95% bootstrap interval
```

This localizes with the band switched off, takes the median of every fitted
width, and repeats at that value until it stops changing; the guess only
needs to be within ~25%. On simulated images whose spots all share one
width it recovers that width to within about 3%. A wider population, such as
out-of-focus beads, pulls the median up; `cal.widths` holds every fitted
width if you want to look at the distribution. On an existing result, the median of
`locs.sigma_ratio` over bright spots reads 1.0 when `sigma` is right.

## Link spots into trajectories

Linking reads the standard localization table -- one row per detection for
the whole movie -- and returns it with a `track_id` column added:

```python
from spotsolve import loctable

parts, n = [], 0
for t, r in enumerate(movie):
    rows, _, _ = loctable.frame_tables(r, frame=t, loc_id0=n)
    parts.append(rows); n += len(rows)
locs = loctable.concat(parts)
locs = loctable.filter_aggregates(locs)     # point emitters only

tracks = spotsolve.link(locs)
tracks.select("track_id", "frame", "y", "x")    # napari's Tracks convention
```

The row order is unchanged and every other column is still there, so the
result is the detector's table with identity added. Every detection belongs
to some track; one that never links is a track of length one.

**There is nothing to tune.** How far a particle moves between frames, how
often the detector misses one, how many new particles appear and how much to
trust the reported localization errors are all properties of the movie, so
they are measured from it. To read what was estimated, or to fit once and
reuse it across movies of the same sample:

```python
params = spotsolve.fit_link_params(locs)
params.d_mean        # population mean D, px^2/frame; x pixel_size^2 / dt for um^2/s
params.d_immobile    # fitted immobile fraction
params.p_cont        # per-frame P(still there AND detected)
params.se_inflate    # factor the reported CRLB variance is scaled by
params.trajectory    # the fit, iteration by iteration
tracks = spotsolve.link(locs, params)
```

Everything is in pixels and frames, which is what the table holds.

**A missed detection ends a track.** Gaps are not closed here: fragmenting a
trajectory is a failure you can recover from downstream, and switching its
identity is not. `p_cont` tells you what that costs -- mean track length is
about `1 / (1 - p_cont)` -- so if it comes back low, the answer is a better
detection or a faster frame rate rather than a bolder linker.

## How linking works

Linking is easy when particles are far apart and hard when they are not. The
quantity that decides which case you are in is the ratio of the diffusive
step `sqrt(4*D*dt)` to the distance to the nearest neighbour: as it
approaches 1, the nearest detection in the next frame is often not the same
particle, and no search radius fixes that.

**Every track carries a small filter, not just its last position.** The state
is position only -- Brownian position is a martingale, so the best prediction
of the next position is the current one -- and the measurement noise is the
CRLB the detector already reported for that detection. Diffusion is not
fitted: `D` lives on a 16-point grid that includes an exact zero, and each
track keeps a posterior over that grid. That zero matters. An immobile
particle that is allowed a nonzero `D` gets a gate of radius
`sqrt(2*D*dt + 2*se^2)` instead of `sqrt(2*se^2)`, and over-wide gates in a
dense immobile population are how identities get swapped.

**The cost of a link is a likelihood ratio, not a distance.** Every candidate
is scored by how much more probable it makes the track than clutter does, so
a dim detection with a large error and a bright one with a small error are
compared on the right terms. Which candidates are even scored is decided by a
gate whose miss rate is bounded by construction (1e-3), not by a tuned
radius.

**Each frame's assignment is solved exactly**, as the birth/death-augmented
linear assignment problem of Jaqaman et al. (2008), over the gated pairs
only. With a cost that depends only on the pair being linked, solving the
whole movie at once would give the same answer, so this is not an
approximation to something better.

The parameters come from the data in two stages. The first forms no
association at all: it fits a mixture to the distance from each detection to
the nearest one in the next frame, which estimates the whole population
distribution of `D` -- linking to estimate a step size and then using that
step size to link would be circular. Then three rounds of link and
re-estimate, damped and clamped to that link-free anchor.

**What it is worth.** Switches per 100 links on simulated movies, as each
ingredient is added (10x10 um, 30% immobile and 70% at D = 0.3 um^2/s,
22 ms frames, 29 nm localization error, 95% detection, 3 seeds):

| step/nearest-neighbour | plain `d^2` | one D, with errors | D mixture | greedy | **this** |
|---|---|---|---|---|---|
| 0.22 | 1.81 | 1.55 | 1.69 | 1.60 | **1.43** |
| 0.50 | 11.99 | 10.38 | 10.24 | 11.40 | **9.35** |
| 0.80 | 29.90 | 23.56 | 22.99 | 23.73 | **22.04** |
| 1.00 | 41.30 | 32.95 | 32.06 | 32.43 | **30.85** |

**What it does not do.** No gap closing across missed frames, no merges or
splits, and no deferring a decision for a few frames to see how it turns out
(multiple hypothesis tracking). The last one was implemented in the reference
and measured losing to this, so it was left out rather than ported.

## How it works

**Units and noise.** The frame is converted to photoelectrons,
`d = (frame - offset) / gain`. Each pixel is treated as Poisson with mean
equal to the model. Camera read noise adds Gaussian variance on top; it is
included by the standard shifted-Poisson approximation, which adds
`read_noise^2` to both the data and the model, so that a pixel's variance is
`model + read_noise^2`.

**The image model.** Each spot is a symmetric 2D Gaussian integrated over the
pixel area, with four parameters: total flux `A`, position `(y, x)` and width
`s`. The expected count in pixel `(i, j)` is the background plus the sum of
all spots:

```text
m[i,j] = background[i,j] + sum_k A_k * E(i; y_k, s_k) * E(j; x_k, s_k)
E(i; c, s) = 1/2 * [ erf((i - c + 1/2) / (s*sqrt(2))) - erf((i - c - 1/2) / (s*sqrt(2))) ]
```

`E` is the fraction of a 1D Gaussian that falls inside pixel `i`, so `A` is
the spot's total photon count, not its peak height. The background is a
smooth surface -- a local mean, over 25-pixel windows, of the pixels away
from every candidate spot -- whose overall level is refitted in every region.

**The objective.** Fits minimize the Poisson I-divergence

```text
I(d, m) = sum over pixels [ d * log(d / m) - (d - m) ]
```

which is the negative Poisson log-likelihood up to a term that depends only
on the data. So the difference in `I` between two models of the same pixels
is exactly their log-likelihood ratio, in nats. Fits are bounded
Levenberg-Marquardt with Fisher scoring.

**The search.**

1. *Find candidates.* A Laplacian-of-Gaussian filter at `sigma` is applied to
   the noise-normalized image; local maxima above a threshold become
   candidates. The threshold allows about a 5% chance of one false candidate
   per frame; candidates only seed the search, they are not detections.
2. *Group into boxes.* Candidates within 2.5 `sigma` of each other share a
   box (at most 12 per box), padded by 3 `sigma` of pixels.
3. *Decide each box*, brightest box first. Start from the background alone.
   Add one spot at a time, at the strongest remaining residual peak that
   passes the same threshold, refitting all spots in the box together. **A
   spot is kept only if it lowers the box's `I` by more than 10 nats** -- a
   likelihood ratio above e^10 ≈ 22,000. Then, while the cheapest spot to
   remove costs less than 10 nats, remove it. Spots in neighbouring boxes are
   held fixed in the model, so a neighbour's light is not claimed again. All
   boxes are decided twice, the second time against settled neighbours.
4. *Polish.* Groups of nearby spots are refitted together, holding the number
   of spots fixed, for up to four passes, stopping once no position moves by
   more than 0.001 pixel. The reported values and standard errors come from
   these final fits; the standard errors are the Cramér-Rao bounds from their
   Fisher information.
5. *Classify* each spot by its fitted width, as above.

**What it does not model.** Haze and out-of-focus light beyond the smooth
background are absorbed as too-wide objects rather than modelled exactly, so
the residual is not pure noise near them. The Gaussian is an approximation to
the real PSF: on the glycerol bead data the residual is slightly positive in
a ring 2-4 pixels around beads (+0.3 standard deviations). Two spots closer than
about one `sigma` cannot be told apart from one brighter spot, and are
reported as one.

## Development

The detector is `rust/spotsolve-core/src/boxsearch.rs` and the linker is
`track.rs` (motion model, gate, assignment), `lap.rs` (the sparse exact
solver) and `trackparams.rs` (the parameter fit), all bound in
`rust/spotsolve-py`; the measurement behind every constant above is recorded
beside it there. `tests/test_localize.py` and `tests/test_tracking.py` hold
both to simulation truth.

The linker was ported from
[tracksolve](https://github.com/delnatan/tracksolve), which stays the
reference: `rust/spotsolve-core/tests/layer6_track.rs` reproduces its linking
detection-for-detection on the frozen fixture `tests/fixtures/08_track.json`.
On a real 49-frame movie (21,438 detections) the two agree on 100% of links
and on every fitted parameter, with the fit taking 0.13 s against 13.0 s and
the linking 0.03 s against 2.2 s. See [docs/README.md](docs/README.md).
