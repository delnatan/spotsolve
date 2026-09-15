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
locs = spotsolve.localize(frame, sigma=1.27, offset=100.0)

locs.positions      # (N, 2): y, x in pixels; pixel centres are integers
locs.amplitudes     # (N,): total flux of each spot, ADU above the offset
locs.se             # (N, 3): standard errors of flux, y, x
locs.fit_sigma      # (N,): each spot's own fitted width, pixels
locs.sigma_se       # (N,): standard error of that width
locs.sigma_ratio    # (N,): fit_sigma / sigma
locs.rejects        # fitted objects that are not reported as spots, with a reason
locs.background     # (H, W): background, ADU above the offset
locs.dispersion     # the frame's measured noise: variance per unit of signal

# A (T, H, W) movie, frames on all cores:
movie = spotsolve.localize_stack(stack, sigma=1.27, offset=100.0)
```

The two calibration inputs:

| argument | meaning |
|---|---|
| `sigma` | the in-focus PSF width, pixels -- see [Calibrate sigma](#calibrate-sigma) |
| `offset` | camera offset, ADU |

**No gain or read noise.** The noise every decision is scaled by is measured
from the frame itself, locally, so it already includes the camera's gain, its
read noise and any haze. `locs.dispersion` reports what was measured -- for a
camera, about `gain + gain^2 * read_noise^2 / background` -- and dividing a
flux by your gain converts it to photoelectrons. See
[No camera calibration](#no-camera-calibration).

`roi=` (a boolean mask) restricts where spots are searched for. Pass one mask
covering everything you want in a single call; do not tile a frame into
several calls, because a source on a tile border is fitted where its light
actually is and can be reported by both tiles.

A mask also makes the call cheaper: the search runs on the mask's bounding
box plus a margin instead of the whole frame, so a 32x32 mask on a 512x512
frame costs 3.4 ms rather than 25.4 ms. The margin is wide enough that
nothing inside the mask can tell. One thing does change with a mask: the
background level the search works against is measured over the mask's own
pixels, not the frame's -- under a cell mask, the frame's dimmest pixels are
the dark field outside the cell, which is not the background inside it.

## Two thresholds: what gets searched, and what gets tried

The search runs one statistic twice. It is a Laplacian-of-Gaussian filter
at `sigma`, in standard deviations of the frame's own local noise, and each
use has its own cut:

| argument | default | decides |
|---|---|---|
| `seed_threshold` | 3.0 (`spotsolve.SEED_Z`) | which peaks in the frame get a box searched around them |
| `birth_threshold` | 2.5 (`spotsolve.BIRTH_Z`) | how strong a leftover peak inside a box must be before a new spot is tried there |

A spot tried at either kind of peak still has to pass the 10-nat test below,
but neither cut is free. A seed or birth too strict costs recall that
nothing recovers, because light that never gets a box is never fitted. Too
loose costs false spots as well as time: a spot is tried at the strongest
peak in its box, and among enough noise peaks some pass.

The defaults were chosen for dim, fast-moving particles. Measured on a GEM
movie (128x128, 49 frames) with emitters of known brightness and diffusion
added to its real frames, and false spots counted on two simulations matched
to it; recall is at D = 0 / 0.43 / 2 px^2 per frame:

| seed / birth | false spots per frame | recall, 150 e- | recall, 300 e- | spots per frame |
|---|---|---|---|---|
| 3.8 / 3.0 | 24 | .44 .34 .27 | .72 .69 .58 | 222 |
| 3.0 / 3.0 | 30 | .48 .38 .29 | .75 .69 .60 | 241 |
| **3.0 / 2.5** | **31** | **.49 .40 .29** | **.76 .72 .60** | **249** |
| 2.5 / 2.5 | 36 | .53 .41 .34 | .78 .73 .61 | 263 |
| 2.0 / 2.0 | 45 | .57 .43 .36 | .78 .74 .64 | 282 |

(Before the width band's upper-bound rule below, which removes about 12 of
those false spots per frame.) Raise both cuts toward 3.5 / 3.0 for fewer false
spots and speed: on 256x256 GEM frames, single-threaded, the defaults take
395 ms per frame and 3.5 / 3.0 takes 304 ms.

Other arguments: `k_max` (the most spots fitted jointly in one box, default
12), `images=` (`model_image` and `residual` on the result; on by default
for `localize`, off for `localize_stack`), and `n_threads=` for
`localize_stack` (default: all cores).

## Spot width: fitted per spot, and reported only inside a band

Every spot's width is fitted, not held at `sigma`. `sigma` sets the scale of
the search and defines "in focus"; each spot's own width may range over
0.7-2.2 x `sigma` (`spotsolve.SLACK`), and spots whose fitted width lies in
**0.8-2.0 x `sigma`** (`spotsolve.BAND`) are reported as detections. So are
spots outside that band by no more than 2 of their own width standard errors
(`spotsolve.BAND_Z`): a dim spot's width is noisy, and a hard cut dropped
real dim spots -- on the GEM movie, most of the spots present in the frames
before and after but missing in between had been fitted and cut as too
narrow or too wide. Everything else is still fitted -- its light is part of
the model -- and is returned in `locs.rejects` with one of three reasons:

- **too narrow** -- fitted width significantly below 0.8 x `sigma`. Nothing
  the microscope images is narrower than its PSF, so this is not a real spot:
  usually a noise spike, or a fit squeezed by its neighbours.
- **too wide** -- fitted width significantly above 2.0 x `sigma`, or at the
  2.2 x `sigma` limit (a width there is not a measurement: the fit wanted to
  be wider), away from the frame edge: an out-of-focus emitter, an extended
  object, or a patch of haze.
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
cal = spotsolve.calibrate_sigma(stack, sigma_guess=1.2, offset=100.0)
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

**Units and noise.** The frame stays in ADU, `d = frame - offset`. Two maps
of the frame's own noise are measured before anything else, over 25-pixel
windows: each pixel's standard deviation, from the local median of a
4th-difference filter that PSF-sized structure barely reaches, and the local
dispersion `phi`, that variance divided by the local median of the data.
Scale the frame by any factor and both maps scale with it, so nothing below
depends on the camera's gain; the read noise and haze are in the maps
already.

**The image model.** Each spot is a symmetric 2D Gaussian integrated over the
pixel area, with four parameters: total flux `A`, position `(y, x)` and width
`s`. The expected count in pixel `(i, j)` is the background plus the sum of
all spots:

```text
m[i,j] = background[i,j] + sum_k A_k * E(i; y_k, s_k) * E(j; x_k, s_k)
E(i; c, s) = 1/2 * [ erf((i - c + 1/2) / (s*sqrt(2))) - erf((i - c - 1/2) / (s*sqrt(2))) ]
```

`E` is the fraction of a 1D Gaussian that falls inside pixel `i`, so `A` is
the spot's total flux, not its peak height. The background is a
smooth surface -- a local mean, over 25-pixel windows, of the pixels away
from every candidate spot -- whose overall level is refitted in every region.

**The objective.** Fits minimize the Poisson I-divergence

```text
I(d, m) = sum over pixels [ d * log(d / m) - (d - m) ]
```

which is the negative Poisson log-likelihood up to a term that depends only
on the data. In photoelectrons, the difference in `I` between two models of
the same pixels is exactly their log-likelihood ratio, in nats. In ADU it is
`phi` times that, so every comparison below divides by `phi`. The fitted
positions and widths do not depend on the scale at all. Fits are bounded
Levenberg-Marquardt with Fisher scoring.

**The search.**

1. *Find candidates.* A Laplacian-of-Gaussian filter at `sigma` is applied to
   the image and divided by the local noise; local maxima above
   `seed_threshold` become candidates. Candidates only seed the search; they
   are not detections.
2. *Group into boxes.* Candidates within 2.5 `sigma` of each other share a
   box (at most 12 per box), padded by 3 `sigma` of pixels.
3. *Decide each box*, brightest box first. Start from the background alone.
   Add one spot at a time, at the strongest peak left in the box's residual
   (the same filter, at `sigma`) that passes `birth_threshold`, refitting all
   spots in the box together. **A
   spot is kept only if it lowers the box's `I` by more than 10 nats** (times
   the box's `phi`) -- a likelihood ratio above e^10 ≈ 22,000. Then, while the
   cheapest spot to remove costs less than 10 nats, remove it. Spots in neighbouring boxes are
   held fixed in the model, so a neighbour's light is not claimed again. All
   boxes are decided twice, the second time against settled neighbours.
4. *Polish.* Groups of nearby spots are refitted together, holding the number
   of spots fixed, for up to four passes, stopping once no position moves by
   more than 0.001 pixel. The reported values and standard errors come from
   these final fits; the standard errors are the Cramér-Rao bounds from their
   Fisher information, scaled by the local `phi`.
5. *Classify* each spot by its fitted width, as above.

**What it does not model.** Haze and out-of-focus light beyond the smooth
background are absorbed as too-wide objects rather than modelled exactly, so
the residual is not pure noise near them. The Gaussian is an approximation to
the real PSF: on the glycerol bead data the residual is slightly positive in
a ring 2-4 pixels around beads (+0.3 standard deviations). Two spots closer than
about one `sigma` cannot be told apart from one brighter spot, and are
reported as one.

## No camera calibration

The detector used to take a gain and a read noise. It no longer does, and on
every test it was put through the change cost nothing that could be measured
beyond noise:

- **Truth.** On simulated 128x128 frames at three densities (gain 2.4, read
  noise 1.6 e-), recall and precision with the camera's true values were
  .850/.981, .682/.917 and .511/.842; measuring the noise instead gave
  .855/.963, .683/.913 and .510/.839.
- **Read noise.** Empty frames at 1-20 e- background and 1.6-2.5 e- read noise
  give 0-0.7 fitted spots per frame without being told the read noise; plain
  Poisson gave up to 13.
- **Real data.** The measured dispersion reads 2.42-2.61 on four GEM crops
  whose calibration implies 2.55-2.85, and 2.41 on glycerol beads against
  2.23. It is least reliable on tiny, crowded bead frames (39x39 px: 1.90
  against 2.15).
- **The gain.** Multiply a frame by any factor and the same spots come back,
  with fluxes multiplied by it. Exactly the same except inside crowded
  clusters, whose decomposition is sensitive to rounding; the count stays
  within 2.

One caveat: the noise maps are computed exactly on a 12-pixel grid and
interpolated between, so shifting an image by a non-multiple of 12 pixels
samples its noise at different points. Measured on 10 crowded 256x256 GEM
frames, comparing the interior of an image with the same image shifted:
shifted by 12 pixels (the grid lines up) the count moves by 2.4% and 84% of
spots come back within 0.5 px; by 5 pixels, 2.1% and 76%. Most of that
sensitivity is the crowded search itself, not the grid.

Compared with the detector it replaced, on the GEM movie at its defaults:
false spots per frame 13 -> 20, recall of 150 e- spots .39/.27/.24 -> .50/
.38/.29, of 300 e- spots .67/.63/.53 -> .74/.71/.60, and about twice the
time.

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
