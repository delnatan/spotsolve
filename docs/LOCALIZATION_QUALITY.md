# A simple localization-quality filter

Use **coordinate precision as the main filter**, and add a flux-significance
cut only if weak detections remain a problem. Keep the original localization
table for inspection. Neither detector needs a composite quality score or
extra columns for this workflow.

## Start here: the same filter for both detectors

```python
import numpy as np
import polars as pl
import spotsolve
from spotsolve import loctable

# frame_img is a raw 2-D image; use your own sigma and camera offset.
result = spotsolve.localize(frame_img, sigma=1.45, offset=100)
# Or: result = spotsolve.localize_aguet(frame_img, sigma=1.45, offset=100)

locs, _, _ = loctable.frame_tables(result, frame=0)
usable = loctable.filter_quality(locs, max_se_pos=0.5)
rejected = locs.join(usable.select("loc_id"), on="loc_id", how="anti")
```

`0.5` pixels is an **example to evaluate, not a calibrated default**.
`filter_quality` always removes non-finite coordinates and non-positive or
non-finite coordinate SEs. Without a cutoff it performs only that basic check.
It preserves the columns, IDs, and order of retained rows.

| Optional argument | Meaning | When to use it |
|---|---|---|
| `max_se_pos` | Maximum `hypot(se_y, se_x)` | Coordinates must be precise enough for your analysis |
| `min_flux_snr` | Minimum `flux / se_flux`, requiring valid positive flux and flux SE | Additional screening of weakly supported signal |

For example, to also try a flux-significance cut:

```python
usable = loctable.filter_quality(locs, max_se_pos=0.5, min_flux_snr=3.0)
```

`se_pos` is a model-based radial RMS uncertainty, not a 68% confidence-circle
radius. The cutoff uses the input coordinate units: **pixels** for
`frame_tables`. A requested precision of 0.05 µm therefore corresponds to
`max_se_pos=0.05 / pixel_size_um` on the standard table. Both optional cutoffs
must be positive and finite.

The helper recomputes precision and flux significance from their base columns
so stale derived columns cannot affect filtering. A small SE or `flux_snr >= 3`
does not prove that an emitter is real or imply a calibrated false-positive rate.

## Multi-emitter: keep routine criteria small

The detector already decides emitter count and applies a width-reporting band
by default. Start with the precision filter. Add a flux cut only if inspection
or simulations show it helps. Increasing `count_penalty` and re-detecting is
another way to demand stronger count support; measure the loss of real
emitters too.

For difficult overlaps, inspect the **existing** Fisher fractions on demand:

```python
fraction = np.asarray(result.info["fisher_fraction"]).reshape(-1, 4)
position_fraction = np.min(fraction[:, 1:3], axis=1)  # y and x
```

The array aligns with original `result` rows, before filtering or sorting.
Parameter order is `(flux, y, x, sigma)`. Each fraction is
`1 / (F[q,q] * inverse(F)[q,q])`: near one means little parameter coupling;
near zero means substantial variance inflation from jointly fitted parameters.
NaN means unavailable. Frozen neighbors and background shape are treated as
known in this calculation.

**Use small fractions to find cases to inspect, not as a default rejection
rule.** SE already includes the fitted coupling. A bright crowded emitter can
have a small fraction and still be precisely localized. That is why this is
not another argument to `filter_quality`.

Finite SEs do not certify convergence. Refinement convergence/stalling flags,
final per-emitter removal scores, and group goodness-of-fit values are not
exposed in the localization table. The simple filter does not claim to check
them. A shared group's residual cannot be assigned unambiguously to one emitter.

## Aguet: inspect patch fit when needed

Aguet already excludes non-converged fits, failed observed-Hessian covariance
estimates, invalid parameters, and invalid image bounds. They appear in
`result.info['failures']`, not in localization rows. Start with the same
precision/optional flux filter.

If isolated-looking detections still seem questionable, inspect the
**existing** per-spot objectives:

```python
result = spotsolve.localize_aguet(frame_img, sigma=1.45, offset=100)
deviance = 2 * np.asarray(result.info["objective"], dtype=float)
dof = result.info["boxsize"] ** 2 - 5  # x, y, width, peak, background
reduced_deviance = deviance / dof
```

This is the patch's Poisson deviance, using the fitter's data-floor convention.
Large values can indicate overlap, an unsuitable PSF, or a nonconstant
background. An approximate reference value of one applies under suitable
Poisson conditions; **greater than one is not a rejection rule**. Calibrate a
limit using good isolated spots through the complete screening/fitting pipeline.
Low counts, gain, read noise, and offset subtraction affect this reference.
Aguet has no fitted dispersion correction.

After choosing `deviance_limit`, apply it without adding result columns:

```python
locs, _, _ = loctable.frame_tables(result, frame=0)
patch_ok = np.isfinite(reduced_deviance) & (reduced_deviance <= deviance_limit)
usable = loctable.filter_quality(locs.filter(pl.Series(patch_ok)), max_se_pos=0.5)
```

Apply this mask while the table still matches `result` row order. Use the
patch objective rather than the optional whole-image residual, which uses a
diagnostic screening-background map.

Aguet has no width-reporting band. If necessary, use a calibrated interval on
the existing `sigma_ratio` column. Calibrate it for Aguet's sampled PSF; its
width differs slightly from the dense detector's integrated PSF
([conventions](AGUET_BASELINE.md#reference-and-method)). A background-only
comparison could add emitter-support evidence later, but is not required for
this first workflow and is not currently returned as a score.

## Check what was lost, then link

Inspect retained and rejected patches across brightness, background, width,
and crowding. Sweep precision first; test whether flux or Aguet deviance adds
useful separation. Validate the combined policy on different movies/seeds.
`is_aggregate` describes object type, not fit quality; filter it separately
only if your analysis requires point emitters.

Measure false detections and real detections lost. For trajectories, measure
incorrect links and recovery against the **original** available true links,
so filtering cannot improve apparent recall by shrinking its denominator.
Check whether retained emitters shift toward bright, slow, or in-focus objects.

In an exploratory three-movie dense-detector simulation, a 0.5-pixel cutoff
retained 98.9% of truth-matched detections and removed 31% of unmatched
detections, but true-link recovery fell from 91.0% to 89.8%. Truth matching
at unresolved overlaps is imperfect. This supports evaluating a loose cut;
it does not calibrate one for experimental data or Aguet.

For movies, give each frame's table the correct `frame` and unique `loc_id`
values, concatenate the tables, and filter **before linking**:

```python
# locs_movie is the concatenation of per-frame tables.
usable = loctable.filter_quality(locs_movie, max_se_pos=0.5)
tracks = spotsolve.link(usable, min_track_length=4)
accepted_tracks = tracks.filter(pl.col("track_accepted"))
```

Keep original frame numbers: removing a detection creates a missing
observation and can end a trajectory. Re-link after changing cuts; filtering
already linked rows can leave IDs spanning gaps. Minimum track length and the
optional [link-margin cutoff](TRACKING.md#conservative-links-and-minimum-length)
are separate association/trajectory criteria.

Further reading: [pointwise precision](https://www.nature.com/articles/ncomms15115),
[consistency with raw images](https://www.nature.com/articles/s41467-020-20056-9),
and [Poisson goodness of fit at low counts](https://arxiv.org/abs/1707.09202).
