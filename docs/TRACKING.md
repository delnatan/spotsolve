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
2-D Brownian step, that rms step is `sqrt(4 D dt + 4 se^2)`. On simulated
movies the switch rate and track continuity are flat between 2.5 and 3
times the rms step. At 2 times, true links are lost (recovery 0.955 → 0.938
in the GEM-like regime). At 4 times, dense fields gain switches (15 → 18 per
100 links).

`max_step` is required and is not estimated from the movie. Iterating
"3 × rms of the linked steps" settled on simulation and on two real movies,
but on the GEM wt movie it kept rising (5.0 → 7.0 px after eight rounds) as
each wider radius admitted wrong links. A median-based version settled
everywhere but came out too small wherever the population is mixed.

## Validation

`spotsolve_core::track`'s unit tests check each frame pair's assignment
against enumeration of every partial matching. `tests/test_tracking.py`
holds switch-rate and continuity floors on simulated Brownian movies (30%
immobile, 70% at D = 0.61 px²/frame, localization error 0.28 px).

Switches per 100 links, and links per detection, three seeds, `max_step` =
3 × the mobile rms step (5.0 px). The previous linker (a per-track mixture
over D with parameters fitted from the movie) is shown for comparison:

| Regime | Step/spacing | This linker | Previous linker |
|---|---:|---:|---:|
| Sparse, all detected | 0.11 | 0.57, 0.950 | 0.26, 0.950 |
| GEM-like, 95% detected | 0.31 | 3.48, 0.906 | 3.06, 0.906 |
| Dense, 90% detected | 0.54 | 13.00, 0.877 | 10.77, 0.866 |

The previous linker's advantage here comes from the immobile population.
With every particle mobile the two are close (GEM-like 4.26 against 4.59
switches per 100 links, dense 15.82 against 15.16), and none of the real
movies below has an immobile fraction above 0.4%.

On the real movies, with `max_step` at 3 × the rms step of the previous
linker's links:

| Movie | Detections | `max_step` | Links shared with previous | Previous linker's links longer than `max_step` | Link time |
|---|---:|---:|---:|---:|---:|
| Beads, 80% glycerol | 15,066 | 3.8 px | 96% | 100 | 3 ms |
| GEM wt | 35,407 | 6.5 px | 95% | 141 | 10 ms |
| GEM anc-1 | 28,659 | 5.4 px | 97% | 173 | 7 ms |

If D were uniform, a step beyond 3 rms steps would occur about once in 8,000
links. In the GEM movies Crocker–Grier started a new track at 67–85% of the
previous linker's long links and linked the rest to a nearer predecessor.
The previous linker needed 0.23 s to fit its parameters on GEM wt before
linking.
