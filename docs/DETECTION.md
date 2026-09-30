# Multi-emitter detection

`localize` and `localize_stack` describe each frame as one Poisson model and
decide every emitter by a likelihood ratio. The separate
[Aguet baseline](AGUET_BASELINE.md) fits candidates independently.

## Model

Work in camera units: `d = frame - offset`. Each component has flux `A`,
position `(y, x)` and width `s`; its Gaussian is integrated over each pixel.
The background `B` is bilinear on a lattice of nodes `ceil(8 * slack[1] *
sigma)` px apart:

```text
m[i,j] = B[i,j] + sum_k A_k * E(i; y_k, s_k) * E(j; x_k, s_k)
E(i; c, s) = 0.5 * [erf((i-c+0.5)/(s*sqrt(2))) - erf((i-c-0.5)/(s*sqrt(2)))]
I(d,m) = sum_pixels [d*log(d/m) - (d-m)]
```

Camera pixels are not Poisson in ADU. One scalar dispersion `phi` (variance
per unit mean, from the median squared fourth difference of the frame)
converts: `I/phi` is the log-likelihood in nats. No gain or read-noise
calibration is needed; fluxes scale with gain, geometry does not.

Components come in two kinds, told apart by width alone:

- **Emitters**, `slack[0] * sigma` to `slack[1] * sigma` (default 1.0-2.25):
  in focus, reported.
- **Out-of-focus light**, from `slack[1] * sigma` to half the node spacing:
  fitted and tested like emitters but returned as background. Half the
  spacing is where the nodes carry most of a blob wherever it sits (at
  least 85% of its light for the default spacing); beyond it the nodes
  suffice. Without this band, defocused emitters and haze a few pixels
  across fit neither the nodes nor the emitters' widths, and the model
  covers them with emitters at the width bound.

## Algorithm

1. **Seeds.** Compute the score `z` for one component at every pixel: the
   signed root of its likelihood ratio with the nodes profiled out,
   `<g - P g, d> / sqrt(var |g - P g|^2)` with `P` the projection on the
   nodes' tents. No background estimate enters its mean; its variance takes
   the level of the nodes fitted to a 25-px median. The bank of widths runs
   from `slack[0] * sigma` to half the node spacing, adjacent widths at
   most 1.5x apart. Local maxima over
   position and width become emitters at their template's width when `z`
   exceeds `u` times what the grid can lose of a maximum: half a pixel off
   in y and x costs `1 / (8 s^2)` of it, and midway between two widths 2%.
   No emitter a test at `u` would keep goes unproposed. `u` is solved from
   `fp_per_mpx` (below).
2. **Fit.** Emitters and background are fitted together by block coordinate
   descent. Emitters are fitted in groups by bounded Levenberg-Marquardt,
   with every other emitter and the background held fixed; groups join the
   most strongly coupled pairs (the canonical correlation of their
   parameters under the Fisher information), at most 12 emitters each. The
   background nodes are fitted by Poisson IRLS with emitters held fixed.
3. **Count.** Once the model has converged, each group is tested:
   - an emitter is removed if removing it costs less than `u^2/2` nats;
   - every local maximum of the residual score over the in-focus seed
     widths, within `4 sigma` of a member, that exceeds `u * kappa` (less
     the grid's loss, as for seeds) becomes a candidate at once, and one
     refit takes them all in. A candidate stays only if dropping it, the
     rest refitted, costs `(u * kappa)^2 / 2` nats, and the candidates
     together gain that much apiece; one the refit carries more than
     `4 sigma` from where it was placed has become another source and is
     dropped.

   Penalising each emitter by the bar, every change a test keeps raises the
   objective, so adds and removals cannot cycle. That holds only while the
   decisions' likelihood ratios are accurate, which is what the exact node
   profile below is for. A group is refitted to convergence after a
   removal, before its candidates are scored, so an addition is never
   credited with the rest of the group converging.

   Out-of-focus components are tested like emitters, at the same `u^2/2`,
   so light goes to whichever explains it; they are proposed by the seeds,
   whose frame-wide score holds at every width.

   The score is efficient (Neyman's C(alpha)): the score of the new flux
   less its projection on the group's and the nodes' scores through their
   joint Newton step, over its information less its projection on theirs.
   With free widths a fitted emitter absorbs an unfound neighbour by
   widening, which leaves little of the neighbour in the residual; the
   projection is what finds it there. The Newton step makes the score exact
   for a fit that has not fully converged, and the nodes' information is
   moved to the group's current model on its patch: a template mostly
   inside the nodes' span keeps a small efficient information that either
   shortcut would misstate.

   Every likelihood ratio, and the Wald screen that spares clear emitters a
   removal trial, profiles out the background nodes. A node's tent and a
   wide emitter trade light: with the nodes held at a fit that includes the
   emitter, the test would credit it with the evidence the nodes gave up.
   A decision changes the model on one group's patch only. Outside it the
   model moves only with the nodes, and that part of the objective is taken
   to second order at the round's model, the nodes that never touch the
   patch eliminated through the frame-wide inverse information. On the
   patch the Poisson objective is kept exact and the nodes under it are
   solved by Newton's method. A second-order profile on the patch as well
   is not enough: removing a wide component changes the model there by
   thousands of counts, and it overstated what the nodes recover by tens of
   nats, several times the bar, which let adds and removals cycle.

   `kappa >= 1` is the spread of the residual score far from every emitter,
   an empirical null. It is 1 where the model describes the data, and grows
   where it does not (PSF wings, haze), raising the bar for additions there.
   The model is re-converged and tested again until nothing changes.

`sigma` is the in-focus width, the narrowest a spot can be; `slack[1]`
marks the edge of the depth of focus. Components fitted wider are returned
as background (`info["out_of_focus"]` counts them); a fit at the lower
bound is flagged.

## The threshold

An emitter survives on pure noise when its likelihood ratio, maximized over
position and width with the nodes profiled out, reaches `u^2/2`: a local
maximum above `u` of that ratio's signed root, a Gaussian field over
position and `tau = log s`. The field is `<h, r> / |h|`, with `h = g - P g`
the pixel-integrated profile less its projection on the nodes' tents. Its
metric, the covariance of the unit field's derivatives, comes from 1-D sums
because `g` and the tents separate; averaged over a node cell it is
`f(tau)^2 (dy^2 + dx^2) + L_tt(tau) dtau^2`. The Gaussian kinematic formula
for that slab gives the expected Euler characteristic per pixel
(`statistics::lkc`, `statistics::expected_ec`):

```text
EC(u) = L3 rho3(u) + L2 rho2(u) + L1 rho1(u)
L3 = int sqrt(det Lambda) dtau
L2 = (f(tau1)^2 + f(tau2)^2) / 2
L1 = 1/(2 pi) int (df/dtau)^2 / sqrt(L_tt) dtau
```

with `rho_j` the Gaussian EC densities. `u` solves `10^6 EC(u) =
fp_per_mpx`. With a known background, `f^2 = 1/(2 s^2)` and `L_tt = 1`:
the slab of hyperbolic space of the continuous Gaussian scale space. The
narrowest widths dominate the count, which is why the search starts at the
in-focus width.

A node's tent resembles a wide emitter, so the node spacing sets how much
of an emitter's flux information the background leaves it. At `8 * slack[1]
* sigma` the widest emitter keeps at least two thirds of it wherever it sits
(0.82 averaged over a node cell); narrower emitters keep more.

This counts the maxima of a continuous Gaussian field, so `fp_per_mpx` is a
bound for Gaussian noise: the detector finds fewer maxima than the field
has, while Poisson skew at low counts adds some.

## Uncertainties

Standard errors come from the undamped expected Fisher information
`F = J.T @ diag(1/m) @ J` of each final group, with the background nodes
profiled out, scaled by `phi`. If `F` cannot be factored, the errors are NaN.

`result.info['fisher_fraction']` has shape `(N, 4)` in `(flux, y, x, sigma)`
order:

```text
fraction[q] = 1 / (F[q,q] * inverse(F)[q,q]) = conditional / marginal variance
```

1 means no coupling to other fitted parameters; values near zero mean strong
confounding. It is invariant to parameter units and order. Neighbours
outside the group are treated as known, so this is not a full uncertainty
budget.

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
