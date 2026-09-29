# Multi-emitter detection

`localize` and `localize_stack` describe each frame as one Poisson model and
decide every emitter by a likelihood ratio. The separate
[Aguet baseline](AGUET_BASELINE.md) fits candidates independently.

## Model

Work in camera units: `d = frame - offset`. Each emitter has flux `A`,
position `(y, x)` and width `s`; its Gaussian is integrated over each pixel.
The background `B` is bilinear on a lattice of nodes 16 px apart:

```text
m[i,j] = B[i,j] + sum_k A_k * E(i; y_k, s_k) * E(j; x_k, s_k)
E(i; c, s) = 0.5 * [erf((i-c+0.5)/(s*sqrt(2))) - erf((i-c-0.5)/(s*sqrt(2)))]
I(d,m) = sum_pixels [d*log(d/m) - (d-m)]
```

Camera pixels are not Poisson in ADU. One scalar dispersion `phi` (variance
per unit mean, from the median squared fourth difference of the frame)
converts: `I/phi` is the log-likelihood in nats. No gain or read-noise
calibration is needed; fluxes scale with gain, geometry does not.

## Algorithm

1. **Seeds.** Against a 25-px median background, compute the efficient score
   `z` for one emitter at every pixel, on a bank of widths from
   `slack[0] * sigma` to `slack[1] * sigma`, adjacent widths at most 1.5x
   apart (an emitter between two keeps 98% of its score). Local maxima over
   position and width with `z > u` become emitters at their template's
   width. `u` is solved from `fp_per_mpx` (below).
2. **Fit.** Emitters and background are fitted together by block coordinate
   descent. Emitters are fitted in groups by bounded Levenberg-Marquardt,
   with every other emitter and the background held fixed; groups join the
   most strongly coupled pairs (the canonical correlation of their
   parameters under the Fisher information), at most 12 emitters each. The
   background nodes are fitted by Poisson IRLS with emitters held fixed.
3. **Count.** Once the model has converged, each group is tested:
   - an emitter is removed if removing it costs less than `u^2/2` nats;
   - an emitter is added at the pixel and width of highest residual score,
     over the seed widths and within `4 sigma` of a member, if that score
     exceeds `u * kappa` and the refit gains `(u * kappa)^2 / 2` nats.

   The score is efficient: its information is what remains after projecting
   out the group's parameters and level. With free widths a fitted emitter
   absorbs an unfound neighbour by widening, which leaves little of the
   neighbour in the residual; the projection is what finds it there.

   Both likelihood ratios let the patch's level float, as the seed score
   does: a wide emitter and the background trade light, and a fixed level
   would credit the emitter with the background's share.

   `kappa >= 1` is the spread of the residual score far from every emitter,
   an empirical null. It is 1 where the model describes the data, and grows
   where it does not (PSF wings, haze), raising the bar for additions there.
   The model is re-converged and tested again until nothing changes.

Widths are fitted within `slack * sigma` (default 1.0-2.2): `sigma` is the
in-focus width, the narrowest a spot can be. These are optimization bounds,
not an acceptance interval; a fit at a bound is flagged.

## The threshold

An emitter survives on pure noise when its likelihood ratio, maximized over
position and width, reaches `u^2/2`: a local maximum above `u` of the
matched-filter field over position and log width. With `tau = log s`, that
field's metric is `(dx^2 + dy^2) / (2 s^2) + dtau^2`, a slab of hyperbolic
space, and the Gaussian kinematic formula gives the expected number per
pixel (`statistics::scale_space_ec`):

```text
EC(u) = 1/4 (1/s1^2 - 1/s2^2) rho3(u) + 1/4 (1/s1^2 + 1/s2^2) rho2(u)
      + 1/(8 pi) (1/s1^2 - 1/s2^2) rho1(u)
```

with `s^2` the profile variance (`(slack * sigma)^2 + 1/12` with pixel
integration) and `rho_j` the Gaussian EC densities. `u` solves
`10^6 EC(u) = fp_per_mpx`. The narrowest widths dominate the count, which
is why the search starts at the in-focus width.

This counts the maxima of a continuous Gaussian field, so `fp_per_mpx` is a
bound, not a calibration: sampling on the pixel lattice finds fewer maxima
for PSFs near a pixel wide, while Poisson skew at low counts and fits at the
frame edge (flagged `EDGE`) add some.

## Uncertainties

Standard errors come from the undamped expected Fisher information
`F = J.T @ diag(1/m) @ J` of each final group, with a free local level,
scaled by `phi`. If `F` cannot be factored, the errors are NaN.

`result.info['fisher_fraction']` has shape `(N, 4)` in `(flux, y, x, sigma)`
order:

```text
fraction[q] = 1 / (F[q,q] * inverse(F)[q,q]) = conditional / marginal variance
```

1 means no coupling to other fitted parameters; values near zero mean strong
confounding. It is invariant to parameter units and order. Neighbours
outside the group and the background nodes are treated as known, so this is
not a full uncertainty budget.

## ROI and reference width

A boolean ROI limits where emitters are seeded and added; fitted positions
may lie outside it. The frame is cropped to the ROI's bounding box plus the
context the filters and fits need. Use one mask for the requested area, not
separate tile calls, which would duplicate sources at seams.

Choose `sigma` from a histogram of fitted widths in a few representative
frames; see the [width inspection example](../README.md#choose-a-detection-width).

## Validation

`rust/spotsolve-core/tests/layer7_localize.rs` holds the detector to recall,
precision and position error on simulated fields (flux 150-3000 ADU on a
background of 20, width `sigma` +-20%), and its false-positive rate on pure
Poisson noise to within a factor of 2 of `fp_per_mpx`. Position error is
held in units of each detection's reported SE (median about `sqrt(ln 2)`),
so recovering a hard emitter does not count against the detector.

`rust/spotsolve-core/examples/characterize.rs` measures the detector against
what the data allow: recall by oracle SNR and width, position error over the
Cramer-Rao bound, close pairs, recall by neighbour distance, and false
emitters on Poisson and Gaussian noise:

```sh
cargo run --release --manifest-path rust/Cargo.toml -p spotsolve-core --example characterize
```

## Limits

The PSF model is a Gaussian. Real PSFs have wings and defocused structure the
model cannot absorb; the residual then carries structure, `kappa` rises, and
dim sources near bright ones are harder to add. Sources closer than about one
`sigma` can be reported as one brighter source.
