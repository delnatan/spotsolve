# Linking detections

`spotsolve.link(locs)` reads a localization table (`frame`, `y`,
`x`, `se_y`, `se_x`) and adds `track_id` and `link_p`, preserving row order
and every other column. Every detection belongs to a track, including
single-frame ones. `link_p` is the probability that the link into a row,
from the previous row of its track, joins one particle's detections; it is
null where a track starts (`link_p=False` leaves the column out).

```python
tracks = spotsolve.link(locs)                  # radius read from the movie
tracks = spotsolve.link(locs, search_radius=20.0)   # or given, in pixels
```

## Model

Detection `a` in frame `f` is linked to `b` in frame `f + 1` when that is
more likely, by the odds of a switch against a break, than `a` going unseen
and `b` being new. The gain of the link is

```text
log p_a(d) - log lambda - c
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
is the fast step. A large search radius for the fast particles then also lets a
slow particle whose detection was missed take any detection in reach. With
a scale per track, an immobile track does not take a fast step and a fast
one keeps its own.

`lambda = (1 - q)^2 rho / q` prices a track end. Here `rho` is the density
of other detections around a detection, and `q` the chance that a detection
continues into the next frame. `(1 - q) rho` is then the density of new
detections and `(1 - q) / q` the odds that a track goes unseen.

With `c = 0` every link would be the most probable call, which weighs a
wrong link and a missed one alike. A wrong link switches two identities and
corrupts both tracks; a missed one only breaks a track. `c = 2.5` prices a
switch at `e^2.5`, about 12 missed links. Most switches are orphans: a track
whose own particle went unseen in `f + 1` takes a detection nearby.

### Continuation by detection strength

`link(locs)` uses the table's `detect_z` column when it has one
(`strength=` names another, `None` turns it off). One continuation `q` for a
whole movie mixes bright in-focus particles, seen again almost surely, with
dim detections near the bar, which often are not. Priced at the mixture, a
bright particle's longer steps cost more than a break, so its tracks are cut
and its large steps lost: on a movie of beads diffusing in and out of focus
(simulated to the real movie's continuation of 0.76), one `q` read D 7%
low; per strength class, 1% low.

With strengths, detections fall into up to six equal-count classes by
`detect_z`, and for each the linker reads `q_c` (each detection's own
next-frame neighbours less its own same-frame neighbours, so a bright
particle's halo or a crowding neighbour cancels) and `rho_c`, its share of
the density. A link `a -> b` is then priced by

```text
lambda_ab = (1 - q_a) / q_a * (1 - q_b) rho_b
```

the odds that `a`'s class goes unseen times the density of new detections of
`b`'s class. A pilot forward pass also measures the movie's population over
`D_k` (each track's final posterior, weighted by its steps); new tracks start
from it, not from every `D_k` alike, and forgetting returns there, so a new
track's first steps are priced as the movie's particles move, not as the
fastest the search radius allows. Without strengths the linker is unchanged.

Detections the detector could not fit as one in-focus emitter (`AT_BOUND`,
`ELONGATED`: haze, defocused blobs) carry `detect_z` too; leave them out of
linking for diffraction-limited tracking, as a loose price lets them take
a bright particle's step.

### Read from the movie, without links

- `g`, the detector's gap: same-frame detections closer than about `g` are
  reported as one. It is the radius at which the same-frame pair density
  first reaches half its mean over `[R/2, R]`, `R` the search radius.
- `r0 = g / 2`. A detection within `r0` in another frame is at the same
  spot, because anything else had to be at least `g - r0` away. `r0` is
  never below three standard deviations of an immobile particle's step, so a
  table without a gap still finds immobile spots, and never above
  `R`.
- `q`, the detections within `R` in the next frame, less that
  frame's own count of other detections around a detection, per detection.
  What is left is each detection's own continuation, however far it moved,
  so `q` does not depend on how the movie's particles move. It is averaged
  with the previous frame.
- `rho`, the same-frame pair density within `R`.

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

### How sure each link is

Within a frame pair, each pass knows every candidate link's odds against
"`a` went unseen and `b` is new": `p_a(d) / lambda_ab`, the gain before
`SWITCH_COST`. Loopy belief propagation over the frame pair's one-to-one
matchings (Williams & Lau 2014, *IEEE TAES* 50:2942) turns these odds into
each link's marginal probability, summed over every way the pair's other
detections could be matched. Both passes give a marginal for each kept link:
one from the track's past, one from its future. `link_p` is their product,
because a link must survive both passes to be kept. The two are not
independent, so the product leans conservative. `link_p` annotates the
links; it never changes which are made.

Calibration on simulated movies with known identities, where links are
right when their two detections share a particle:

| Movie | Links | Mean `link_p` | Share right | Right at `link_p` < 0.5 | 0.95–0.99 | ≥ 0.999 |
|---|---|---|---|---|---|---|
| 2-D mixed mobility, ~460 /frame | 18,954 | 0.964 | 0.967 | 0.69 | 0.95 | 0.998 |
| 2-D mixed mobility, ~720 /frame | 29,205 | 0.946 | 0.953 | 0.69 | 0.95 | 0.998 |
| 3-D beads in a slab, 267 /frame | 8,968 | 0.972 | 0.964 | 0.65 | 0.94 | 0.997 |
| same, 3× density | 27,599 | 0.917 | 0.918 | 0.61 | 0.93 | 0.997 |
| confined discs, 0.030 /px² (tests) | 86,028 | 0.909 | 0.945 | 0.68 | 0.97 | – |

Each direction's marginal alone, their mean or their minimum ran 1 to 3
points overconfident on the first four movies. Use `link_p` to weight or
filter steps downstream.

## The search radius

The search radius `R` is the farthest step considered, and three times the
rms 2-D step at the top of the `D` grid. It is soft. Within it, each track's
own step size and the price of a wrong link decide what is linked, so a
larger `R` changes the links little. Only a radius too small for the fastest
particles costs links. On simulated mixtures, recall and wrong links move by
about a point or less between one and three times the rule below, and half
the rule costs 15-25 points of fast recall.

By default `R` is read from the movie, without links
(`spotsolve.search_radius(locs)`).

1. **Pair counts.** For each detection, the detections within `r` in the next
   frame are its own continuation plus everyone else. Those within `r` in its
   own frame are everyone else. The own-frame count is taken as the
   background beyond twice the detector's gap, and scaled as `r^2` below it.
2. **Fit.** The excess over that background is the continuation times the
   step distribution. It is fitted as a non-negative mixture of 2-D Gaussian
   steps over a log grid of `D`.
3. **Window.** Pairs are counted out to four mean nearest-neighbour spacings.
   The grid stops at the `D` whose three rms steps fill that window: a faster
   step is lost among the neighbours for any linker.
4. **Radius.** The fastest `D` that holds at least 3% of the fit sets
   `R = 1.5 × 6 sqrt(D + se^2)`. That is 1.5 times the radius whose grid tops
   out at that `D`. The margin covers the pair count reading the fastest `D`
   up to a quarter low.

Out of sample, on 19 movies (2-D mixtures with fast `D` 1-8 px²/frame and
single populations from immobile to fast, at 230-860 detections per frame,
and 3-D beads in a slab at 1x and 3x), the radius read from the movie came
out at 1.4-3x the rule, and up to 33x for immobile-only movies, where the
rule's radius is tiny. Its links matched the rule's: recall within 1.5
points, and wrong links within 0.6 points. On the confined-disc movies of
the tests, fast recall was up to 4 points higher.

`search_radius=` overrides it, in the table's units. A value of about three
times the rms 2-D step of the fastest particles of interest,
`sqrt(4 D dt + 4 se^2)`, or more, links the same. `max_step=` is its old
name.

## Validation

`lap.rs` checks the assignment against enumeration of every partial
matching. `track.rs` checks these properties:

- What the linker reads from the movie (gap, continuation and density) on
  simulated movies with known values, and that the continuation of fast
  particles is read as well as that of immobile ones.
- That a table without a gap still counts immobile continuations.
- That a track's posterior settles on its own diffusion coefficient.
- That row order and the direction of time do not change the links.
- That a fast particle passing an immobile one, which least squares swaps,
  keeps both identities.
- That the search radius read from the movie covers three rms steps of
  the fastest particles without running away, and that a table with no
  or one row still gives one.
- That link marginals are exact where the candidate links form a tree, that
  every link has a probability and every track start none, that their mean
  matches the share of right links, and that a link's probability is the
  same with the movie reversed.

`tests/test_tracking.py` simulates particles confined to discs: 30% immobile,
50% at 0.3 um²/s and 20% fast, or no immobile ones and 70% / 30%. The units are 104 nm per pixel and 22 ms per
frame, with 0.28 px localization error. Each particle is seen with its own
probability, uniform on 0.4 to 0.99, and same-frame detections closer than
2.5 px are reported as one. For each class, the test requires that, at the
recall the linker reaches, its identity switches do not exceed Crocker &
Grier's at the radius best for that class, and that slow and fast particles
reach the same recall with or without the immobile ones.

Nine seeds, a search radius of three times the fast rms step. Each cell gives
recall, switches per 100 links, and the least switches per 100 links
Crocker & Grier reaches at that recall over all radii:

| Density, fast D | Immobile | Slow | Fast |
|---|---|---|---|
| 0.010 /px², 2.5 um²/s | 0.96 / 1.2 / 1.5 | 0.86 / 2.9 / 3.2 | 0.42 / 12.0 / 14.9 |
| 0.010 /px², 5.0 um²/s | 0.96 / 1.2 / 1.5 | 0.87 / 3.1 / 3.4 | 0.29 / 20.2 / 24.6 |
| 0.030 /px², 1.0 um²/s | 0.88 / 2.0 / 2.5 | 0.67 / 5.9 / 6.3 | 0.40 / 14.2 / 15.7 |
| 0.030 /px², 2.5 um²/s | 0.88 / 2.2 / 2.9 | 0.68 / 6.4 / 6.9 | 0.24 / 28.6 / 29.8 |
| 0.010 /px², 2.5 um²/s, no immobile | — | 0.86 / 3.6 / 3.7 | 0.41 / 15.1 / 16.8 |

## Limits

- The linker is conservative by `c`: where identity is ambiguous it breaks
  a track rather than risk a switch, so tracks are shorter than Crocker &
  Grier's at its usual radius.
- When the fast rms step exceeds the nearest-neighbour spacing, frame-to-frame
  identity is ambiguous for any linker. Only a shorter frame interval or a
  lower density helps.
- `q` and `rho` are one number per movie. Dim detections continue less often
  than bright ones, and adding detections in one region shifts links slightly
  elsewhere.
