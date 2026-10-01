# Linking detections

`spotsolve.link(locs, max_step)` reads a localization table (`frame`, `y`,
`x`) and adds `track_id`, preserving row order and every other column. Every
detection belongs to a track, including single-frame ones.

```python
tracks = spotsolve.link(locs, max_step=6.5)   # pixels
```

## Model

Between consecutive frames the links minimize

```text
sum over links of d^2  +  max_step^2 * (tracks ended)
```

where `d` is the distance a particle moved (Crocker & Grier 1996). No step
longer than `max_step` is linked. Each frame's assignment is solved exactly
as a sparse maximum-gain matching, where a link of length `d` gains
`max_step^2 - d^2` over ending the track and starting a new one. Greedy
nearest-first matching is not equivalent.

For Brownian particles with one diffusion coefficient `D` and uniform
localization error `se`, every step has the same Gaussian variance
`2D dt + 2se^2` per axis. The summed squared displacement is then the
negative log likelihood of an assignment, up to a scale. `D` only decides
when ending a track is cheaper than a long link, and `max_step` states that
directly. A diffusion model helps when immobile and mobile particles mix:
an immobile particle is then known not to jump.

A frame with no detection of a particle ends its track. There is no gap
closing, merge/split model or motion prediction. Diffusion, localization
error and motion blur are measurements to make on the tracks afterwards.

## Choosing `max_step`

`max_step` trades broken tracks against identity switches. About three times
the rms step of the fastest particles of interest is a good start. For a
2-D Brownian step, that rms step is `sqrt(4 D dt + 4 se^2)`. Much smaller
loses true links; much larger adds switches where particles are dense.

`max_step` is required and is not estimated from the movie. An estimate from
the linked steps is circular: a wider radius admits wrong links, and those
widen the estimate further.

## Validation

`spotsolve_core::track`'s unit tests check each frame pair's assignment
against enumeration of every partial matching. `tests/test_tracking.py`
holds switch-rate and continuity floors on simulated Brownian movies (30%
immobile, 70% at D = 0.61 px²/frame, localization error 0.28 px).

Switches per 100 links, and links per detection, three seeds, `max_step` =
3 × the mobile rms step (5.0 px):

| Regime | Step/spacing | Switches / 100 links | Links / detection |
|---|---:|---:|---:|
| Sparse, all detected | 0.11 | 0.57 | 0.950 |
| GEM-like, 95% detected | 0.31 | 3.48 | 0.906 |
| Dense, 90% detected | 0.54 | 13.00 | 0.877 |
