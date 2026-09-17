# Linking detections

`spotsolve.link` reads a standard `loctable` localization table and adds
`track_id`, preserving row order and other columns. Every detection belongs
to a track, including singletons. Both current detectors produce compatible
tables; see the [README example](../README.md#link-detections).

## Model and assignment

Each track keeps a position filter and a posterior over 16 diffusion values,
including exact zero. Brownian motion predicts the current position as the
next mean; localization uncertainty comes from each detection. Keeping an
immobile component avoids unnecessarily wide gates for stationary particles.
For equal position SEs, the per-axis prediction/difference scale includes
`sqrt(2*D*dt + 2*se²)`; assigning motion to an immobile source widens that scale.

Candidate links pass a statistical gate with nominal miss probability 1e-3.
Costs compare the track likelihood with clutter, accounting for distance and
measurement uncertainty. A sparse exact linear assignment solves each
frame's birth/death-augmented association problem. The nominal gate rate
depends on the motion and uncertainty model; it is not an identity-error rate.

Parameters are estimated without first assigning identities: nearest-neighbor
distance distributions initialize diffusion, then three damped, clamped
rounds of linking and re-estimation refine the parameters around that initial
estimate. The units are pixels and frames.

```python
params = spotsolve.fit_link_params(locs)
params.d_mean        # mean D, px²/frame; multiply by pixel_size² / dt for µm²/s
params.d_immobile    # fitted immobile fraction
params.p_cont        # probability of remaining present AND detected next frame
params.se_inflate    # multiplier on localization variance
params.trajectory    # parameter estimates by iteration
tracks = spotsolve.link(locs, params)
```

`brightness=True` additionally uses `flux` and `se_flux`. It is off by
default because intensity may fluctuate. In the reported mixed-brightness
simulation it halved takeovers of bright tracks by dim fast particles; on
the GEM movie it increased singleton tracks by about 2%.

A missed detection ends a track. There is no gap closing, merge/split model,
multi-frame deferred assignment, mobility classification or link-posterior
output. Expected track length is roughly `1/(1-p_cont)` under a constant
continuation model. Better detection or faster imaging may help when tracks
are short; more permissive linking can also increase identity errors.

## Validation

Crowding difficulty is summarized by diffusive step `sqrt(4*D*dt)` divided
by nearest-neighbor spacing. As it approaches one, neighboring particles
become plausible continuations.

Switches per 100 links in simulated 10x10 µm movies, 30% immobile and 70% at
D = 0.3 µm²/s, 22 ms frames, 29 nm localization error, 95% detection,
three seeds:

| Step/spacing | Distance only | Single D + errors | D mixture | Greedy | Current |
|---|---:|---:|---:|---:|---:|
| 0.22 | 1.81 | 1.55 | 1.69 | 1.60 | 1.43 |
| 0.50 | 11.99 | 10.38 | 10.24 | 11.40 | 9.35 |
| 0.80 | 29.90 | 23.56 | 22.99 | 23.73 | 22.04 |
| 1.00 | 41.30 | 32.95 | 32.06 | 32.43 | 30.85 |

The implementation was ported from
[tracksolve](https://github.com/delnatan/tracksolve), `mode="lap"`. Two
intentional differences:

- Gates use inflated localization variance, matching the track filter. The
  reference gate used raw variance; the frozen fixture includes this fix.
- Omit `lam_fa`, which cancels from link comparisons, and the “stop if evidence
  fell” rule, which fired in none of 27 fits. Every detection is assigned to
  some track, limiting that evidence interpretation.

At the 2026-09-11 port validation, all links agreed on a 49-frame GEM movie
with 21,438 detections, and fitted parameters agreed to the printed digits.
Parameter fitting took 13.03 → 0.13 s; linking took 2.17 → 0.03 s.
`tests/fixtures/08_track.json` covers simulated step/spacing ratios
0.11, 0.31 and 0.53; `layer6_track.rs` checks detection-for-detection parity.
`tests/test_tracking.py` checks the Python interface and simulation accuracy.
The multi-frame hypothesis variant remains in the reference; it did not
improve the reported comparison and was not ported.
