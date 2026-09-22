# Linking detections

`spotsolve.link` reads a standard `loctable` localization table and adds
`track_id`, preserving row order and other columns. Every detection belongs
to a track, including singletons. Both current detectors produce compatible
tables; see the [README example](../README.md#link-detections).

## Model and assignment

This implements frame-to-frame LAP linking in the style of Jaqaman's first
stage, with Brownian-motion likelihood costs. It does not implement the later
gap-closing or merge/split stages.

Each track keeps a position filter and a posterior over 16 diffusion values,
including exact zero. Brownian motion predicts the current position as the
next mean; localization uncertainty comes from each detection. Keeping an
immobile component avoids unnecessarily wide gates for stationary particles.
For equal position SEs, the per-axis prediction/difference scale includes
`sqrt(2*D*dt + 2*se²)`; assigning motion to an immobile source widens that scale.

Candidate links pass a statistical gate with nominal miss probability 1e-3.
Costs compare continuation with ending the track and starting a new one,
accounting for distance and measurement uncertainty. Clutter intensity cancels
from this comparison; there is no separate false-detection hypothesis yet.
A sparse exact linear assignment solves each
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

## Conservative links and minimum length

```python
import polars as pl

tracks = spotsolve.link(
    locs, params,
    min_link_margin=1.0,  # example cutoff in nats, not a calibrated default
    min_track_length=4,  # four consecutive detections, i.e. three links
    diagnostics=True,
)
accepted = tracks.filter(pl.col("track_accepted"))
```

For each link in the optimal frame assignment, `link_margin` is the loss in
total assignment score when that link is forbidden and all competitors can
be reassigned, including termination and birth. It is computed exactly using
residual shortest paths through the existing assignment, including the flow
sink; comparing only a row's best and second-best candidates is insufficient.
A tied alternative has margin zero. Margins use natural-log score units and
are conditional on histories committed before this frame. They are not link
probabilities or whole-movie confidence estimates.

`min_link_margin` defaults to zero, preserving the existing links, including
ties. A positive cutoff rejects proposals below that margin, ending the source
track and starting a new one at the destination. Rejections happen together
against the original assignment; there is no second-choice fallback after
competitors disappear. Rejected proposals do not update the position,
diffusion, or brightness state. This option does not change parameter fitting:
fitting on only accepted tracks would introduce an additional selection bias.

`min_track_length=N` adds `track_length` and `track_accepted`, marking every row
of a completed segment according to its total consecutive-frame count. A track
can end before N or at any time after N; reaching N never makes another link
easier to accept. All detections, including short dead ends, remain in the table.
A minimum length alone cannot remove a false detection inside a long track.
When omitted, these two columns are not added.

With `diagnostics=True`, `link_margin` and `link_rejected` describe the proposed
incoming link at each destination. Rejected proposals retain their margin and
have `link_rejected=True`; births without a proposal have null margin and False.
Thus a new track can start either with a null margin or with a rejected incoming
proposal. The output preserves input row order. Re-linking removes or recomputes
these generated columns so old acceptance flags cannot silently survive new IDs.

The stricter policy trades coverage for fewer identity errors. Persistent
artifacts and convincing wrong associations can still pass, and a length cutoff
can preferentially retain slow particles. Multi-frame lookahead, explicit clutter
modeling, and shared-drift estimation remain future work.

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

### Conservative-linking experiment (2026-09-17)

Reproduce with:

```sh
python scripts/benchmark_tracking.py --frames 16 --seeds 101 102 103 --out /tmp/tracking.json
```

Three seeds, 70 emitters, D drawn from 0, 0.15, and 0.6 px²/frame, and minimum
length four. The nearby-clutter arm has 90% detection and 23 independent false
spots per frame placed near emitters. The detector arms render 64×64 Poisson
images with flux 250–700, background 20, and sigma 1.2, then use the actual
multi-emitter detector and reported localization errors. Drift is [0.08, -0.04]
px/frame and is not supplied to the linker. Detector-output truth labels use
one-to-one matching within 1 px and are imperfect for unresolved emitters.

Both percentages below concern tracks passing the four-frame minimum. Wrong
fraction counts all incorrect links, including links touching unmatched
detections. True-link recovery divides correct retained links by all true
consecutive pairs present in the localization table, not by all physical motion.

| Input | Margin (nats) | Wrong fraction | True-link recovery |
|---|---:|---:|---:|
| Nearby clutter | 0 | 38.9% | 66.9% |
| Nearby clutter | 0.5 | 28.3% | 45.1% |
| Nearby clutter | 1 | 18.9% | 31.6% |
| Nearby clutter | 2 | 12.9% | 18.4% |
| Detector output | 0 | 15.0% | 91.0% |
| Detector output | 0.5 | 14.0% | 90.1% |
| Detector output | 1 | 12.9% | 88.0% |
| Detector output | 2 | 10.4% | 83.6% |
| Detector output with drift | 0 | 14.4% | 92.4% |
| Detector output with drift | 1 | 12.7% | 89.1% |
| Detector output with drift | 2 | 10.5% | 86.1% |

These are a cutoff sweep, not held-out threshold calibration or an accuracy
guarantee. In the nearby-clutter arm, mean true D among correct accepted links
falls from 0.262 to 0.181 px²/frame between cutoffs zero and two: an explicit
warning that purity alone is insufficient. This is a truth-based selection
diagnostic, not a downstream diffusion-estimator validation. Link-plus-diagnostic
calls took roughly 2–6 ms per 16-frame movie, excluding detection and parameter
fitting, in this small experiment. A separate timing check on the saved GEM
localization table, after excluding aggregates and invalid errors, used 10,986
detections across 49 frames: parameter fitting took 58 ms, baseline linking
14 ms, and linking with diagnostics and a 1-nat cutoff 45 ms. These are single
wall-clock measurements, not accuracy evidence for the unlabelled real movie.

Unit tests compare exclusion margins against exhaustive enumeration on random
small assignments, with ties, mixed-sign scores, free columns, and termination.
Python tests cover no-gap behavior, rejected proposals, row order, diagnostics,
minimum-length semantics, and unchanged default linking. Collinear coordinates
also have a bounded spatial index rather than a zero-area allocation blow-up.
