# Linking detections

`spotsolve.link(locs, max_step)` reads a localization table (`frame`, `y`,
`x`, `se_y`, `se_x`) and adds `track_id`, preserving row order and every
other column. Every detection belongs to a track, including single-frame
ones.

```python
tracks = spotsolve.link(locs, max_step=20.0)   # pixels
```

## Model

Detection `a` in frame `f` is linked to `b` in frame `f + 1` when that is
more likely than `a` going unseen and `b` being new. The gain of the link is

```text
log p_a(d) - log lambda
```

`p_a` is the step density that `a`'s track predicts. It is a mixture of 2-D
Gaussians over a log grid of diffusion coefficients `D_k`, each with
per-axis variance `2 D_k + se_a^2 + se_b^2`, weighted by the track's
posterior over `D_k` from its own linked steps. A small share of that
posterior returns to the uniform prior at every step. Each frame pair's
links are the matching of greatest total gain, solved exactly as a sparse
assignment.

With one diffusion coefficient for every particle this is Crocker &
Grier's least squared displacement (1996). A single scale fails when
mobilities mix. Under least squares, a fast particle passing a slow one is
swapped with it whenever the slow one lies inside the circle whose diameter
is the fast step. A large `max_step` for the fast particles then also lets a
slow particle whose detection was missed take any detection in reach. With
a scale per track, an immobile track does not take a fast step and a fast
one keeps its own.

`lambda = (1 - q)^2 rho / q` prices a track end. Here `rho` is the density
of other detections around a detection, and `q` the chance that a detection
continues into the next frame. `(1 - q) rho` is then the density of new
detections and `(1 - q) / q` the odds that a track goes unseen.

### Read from the movie, without links

- `g`, the detector's gap: same-frame detections closer than about `g` are
  reported as one. It is the radius at which the same-frame pair density
  first reaches half its mean over `[max_step / 2, max_step]`.
- `r0 = g / 2`. A next-frame detection within `r0` is the same particle,
  because anything else had to move at least `g - r0`. `r0` is never below
  three standard deviations of an immobile particle's step, so a table
  without a gap still counts immobile continuations.
- `q`, the share of detections with a detection within `r0` in the next
  frame, averaged with the previous frame.
- `rho`, the same-frame pair density within `max_step`.

### A new track

A new track starts from the prior over `D_k` that its spot's occupancy
gives. The occupancy is whether the ten frames on each side hold a detection
within `r0`; `f + 1`, the link being decided, is excluded. An immobile
particle leaves a column of detections at one spot, across its own missed
frames, and a mobile one does not. The occupancy only informs `D`; it never
makes a link.

### Both directions of time

A track's history makes each pass causal: after a wrong link, the track
carries the other particle's history. The linker runs forward and on the
reversed movie and keeps the links both make. A link they disagree on
becomes a break, not a switch. Reversing a movie's frames gives the same
links.

A frame without a particle's detection ends its track. There is no gap
closing: a 2-D track that vanishes and reappears cannot be guaranteed to be
the same particle. Diffusion, localization error and motion blur are
measurements to make on the tracks afterwards.

## Choosing `max_step`

`max_step` is the largest step considered: the search radius, and three
times the rms 2-D step at the top of the `D` grid. Set it to about three
times the rms step of the fastest particles of interest. For a 2-D Brownian
step, that rms step is `sqrt(4 D dt + 4 se^2)`. A larger value costs slow
particles little, because each track is held to its own scale. Nothing else
is set by hand.

## Validation

`lap.rs` checks the assignment against enumeration of every partial
matching. `track.rs` checks these properties:

- What the linker reads from the movie (gap, continuation and density) on
  simulated movies with known values.
- That a table without a gap still counts immobile continuations.
- That a track's posterior settles on its own diffusion coefficient.
- That row order and the direction of time do not change the links.
- That a fast particle passing an immobile one, which least squares swaps,
  keeps both identities.

`tests/test_tracking.py` simulates particles confined to discs: 30% immobile,
50% at 0.3 um²/s and 20% fast. The units are 104 nm per pixel and 22 ms per
frame, with 0.28 px localization error. Each particle is seen with its own
probability, uniform on 0.4 to 0.99, and same-frame detections closer than
2.5 px are reported as one. For each class, the test requires that, at the
recall the linker reaches, its identity switches do not exceed Crocker &
Grier's at the radius best for that class.

Nine seeds, `max_step` three times the fast rms step. Each cell gives
recall, switches per 100 links, and the least switches per 100 links
Crocker & Grier reaches at that recall over all radii:

| Density, fast D | Immobile | Slow | Fast |
|---|---|---|---|
| 0.010 /px², 2.5 um²/s | 0.96 / 1.3 / 1.5 | 0.87 / 2.9 / 3.5 | 0.43 / 12.0 / 14.9 |
| 0.010 /px², 5.0 um²/s | 0.96 / 1.2 / 1.5 | 0.87 / 3.1 / 3.6 | 0.28 / 20.0 / 24.5 |
| 0.030 /px², 1.0 um²/s | 0.88 / 2.0 / 2.4 | 0.68 / 5.9 / 6.3 | 0.41 / 14.2 / 15.7 |
| 0.030 /px², 2.5 um²/s | 0.87 / 2.1 / 2.3 | 0.66 / 6.3 / 6.6 | 0.22 / 28.7 / 29.7 |

## Limits

- The linker is conservative: where identity is ambiguous it breaks a track
  rather than risk a switch, so tracks are shorter than Crocker & Grier's
  at its usual radius.
- When the fast rms step exceeds the nearest-neighbour spacing, frame-to-frame
  identity is ambiguous for any linker. Only a shorter frame interval or a
  lower density helps.
- `q` and `rho` are one number per movie. Dim detections continue less often
  than bright ones, and adding detections in one region shifts links slightly
  elsewhere.
