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
   `z` for one emitter of width `sigma` at every pixel. Local maxima with
   `z > u` become emitters. `u` is solved from `fp_per_mpx`, the expected
   number of false emitters per 10^6 pixels of pure noise.
2. **Fit.** Emitters and background are fitted together by block coordinate
   descent. Emitters are fitted in groups by bounded Levenberg-Marquardt,
   with every other emitter and the background held fixed; groups join the
   most strongly coupled pairs (the canonical correlation of their
   parameters under the Fisher information), at most 12 emitters each. The
   background nodes are fitted by Poisson IRLS with emitters held fixed.
3. **Count.** Once the model has converged, each group is tested:
   - an emitter is removed if removing it costs less than `u^2/2` nats;
   - an emitter is added at the pixel of highest residual score if that
     score exceeds `u * kappa` and the refit gains `(u * kappa)^2 / 2` nats.

   `kappa >= 1` is the spread of the residual score far from every emitter,
   an empirical null. It is 1 where the model describes the data, and grows
   where it does not (PSF wings, haze), raising the bar for additions there.
   The model is re-converged and tested again until nothing changes.

Widths are fitted within `slack * sigma` (default 0.7-2.2). These are
optimization bounds, not an acceptance interval; a fit at a bound is flagged.

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
background of 20, width `sigma` +-20%) and to its false-positive rate on pure
Poisson noise. Measured when the tests were set (sigma 1.2, 128x128):

| Emitters / px | Recall | Precision | RMS error, px |
|---:|---:|---:|---:|
| 0.005 | 1.000 | 1.000 | 0.12 |
| 0.015 | 0.890 | 0.991 | 0.20 |
| 0.03 | 0.868 | 0.993 | 0.25 |

On noise the false-positive rate is within 3% of `fp_per_mpx` for `sigma`
1.0-1.45 at the default of 16; wide PSFs at strict targets overshoot (1.5x
at `sigma` 1.8 and `fp_per_mpx` 4).

## Limits

The PSF model is a Gaussian. Real PSFs have wings and defocused structure the
model cannot absorb; the residual then carries structure, `kappa` rises, and
dim sources near bright ones are harder to add. Sources closer than about one
`sigma` can be reported as one brighter source.
