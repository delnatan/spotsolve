# Detection

`localize` and `localize_stack` follow u-track's `pointSourceDetection`
(Aguet et al. 2013, *Dev. Cell* 26:279): screen the frame for significant
signal, seed at local maxima, fit each seed on its own window, and with
`fit_mixtures` fit several emitters per window. Every decision is a Poisson
likelihood ratio at one threshold, set by an expected false-positive rate.

## Model

Work in camera units: `d = frame - offset`. A window around a seed holds a
constant level `c` and components of flux `A`, centre `(y, x)` and width `s`,
each a Gaussian integrated over every pixel:

```text
m[i,j] = c + sum_k A_k * E(i; y_k, s_k) * E(j; x_k, s_k)
E(i; y, s) = 0.5 * [erf((i-y+0.5)/(s*sqrt(2))) - erf((i-y-0.5)/(s*sqrt(2)))]
I(d,m) = sum_pixels [d*log(d/m) - (d-m)]
```

Pixels below the offset count as zero. Camera pixels are not Poisson in ADU;
one scalar dispersion `phi`, the variance per unit mean, converts: `I/phi` is
the log-likelihood in nats. No gain or read-noise calibration is needed;
fluxes scale with the gain, positions and decisions do not.

`phi` is measured on each frame. The separable fourth difference `b`
(`[1, -4, 6, -4, 1]` along each axis) has variance `70^2 phi mbar`, `mbar`
the local mean under the weights `k^2 / 70^2`, wherever the light is smooth
on its 5-pixel scale. `phi` is the median of `b^2 / mbar` over the median of
`70^2 chi2_1`. Variance and mean come from the same pixels with the same
weights, so a varying background or an emitter's light raises both alike;
an emitter's curvature adds to `b` alone and can only raise `phi`.

## Algorithm

1. **Screen.** At every pixel, regress the `ceil(4 sigma)` window (cut at
   the frame's edge) on one pixel-integrated PSF and a constant. The mask
   holds pixels whose Poisson score test of the flux,
   `z = A |g - gbar| / sqrt(phi * c0)`, reaches a bar. The noise comes from
   the Poisson model rather than the window's residual, so a neighbour does
   not hide a spot.
2. **Seeds.** Local maxima of the negative Laplacian of Gaussian inside the
   mask. The score is computed at a bank of widths spanning the reported
   range, neighbours at most 1.5x apart, each with the bar `u` less what the
   pixel grid and the gap to the next width can lose of a maximum.
3. **Window fits.** Pixels of other mask components are left out. Components
   are fitted by bounded Levenberg-Marquardt (Coleman-Li scaling), each with
   its own width. A component is kept only if adding it gains `u^2/2` nats:
   `2 (I_k - I_k+1) / phi >= u^2`.
   - **Single fits** (default): one component starting at the seed, its
     centre held within `2 sigma` of it.
   - **Mixtures** (`fit_mixtures=True`): components are added where the
     efficient score of a new flux peaks (the score with the fitted
     components projected out), each refit from the joint Newton step,
     while each gains `u^2/2`; then the weakest is removed while removing
     it, the rest refitted, costs less. Every source of light in the window
     takes a component, so a neighbour is a nuisance parameter rather than a
     bias.
4. **Ownership.** Each component is reported by the fit of the seed nearest
   to it, and only those are tested for removal. A component held on a
   position bound is not reported. Two windows may each place an emitter
   midway between their seeds on their own side; reports from different
   windows closer than `sigma`, each the other window's nearest component,
   are one emitter fitted twice, and the copy whose seed is nearer their
   midpoint is kept. A pair that one window resolved is never merged.
5. **Widths.** Emitters are reported within `width * sigma`, by default
   `sigma` to `1.5 sigma`: a little past `sqrt(2) sigma`, where defocus has
   halved the peak (the edge of the PSF's axial FWHM). The upper bound is
   capped at half the window's half-side, about `2 sigma`, where a centred
   emitter keeps 91% of its light in the window. A component may widen past
   the bound, to the window's half-side, as out-of-focus light that narrower
   components would otherwise split; it is counted in `info["out_of_focus"]`,
   not reported. `width=(1, 1)` fixes every width at `sigma`.
6. **Uncertainties.** Standard errors from the inverse expected Fisher
   information `J^T diag(1/m) J` of each emitter's final window fit, every
   component and the level free, scaled by `phi`.

`info["z"]` is each emitter's `sqrt(2 * cost / phi)`, the cost being what
removing it (the rest refitted) loses: at least `u`. `info["mixture"]`
numbers the windows that held several components; `info["seed"]` is the
seed each emitter was reported from; `info["duplicates"]` counts components
left to, or merged into, another seed's report.

## The threshold

`u` is set so that pure noise yields `fp_per_mpx` false emitters per 10^6
pixels. A false emitter is a local maximum above `u` of the signed root of
the likelihood ratio, maximized over position and width: a smooth unit
Gaussian field over `(y, x, tau = log s)`. Scores of two profiles correlate
as the profiles do, `exp(-r^2 / (4 v))` at offset `r` with `v = s^2 + 1/12`
and `2 sqrt(v1 v2) / (v1 + v2)` across widths, so the field's metric is

```text
ds^2 = (dy^2 + dx^2) / (2 v) + dtau^2
```

hyperbolic space of curvature -1, with width as height (Siegmund & Worsley
1995, *Ann. Stat.* 23:608). Widths are searched from `lo` up to the window's
half-side and only maxima at `lo` to `hi` are reported: a slab whose face at
`lo` is a horosphere (both principal curvatures 1). With `a = 1/v(lo)`,
`b = 1/v(hi)`, the expected number of reported maxima above `u` per pixel is
the Euler-characteristic density (Adler & Taylor 2007):

```text
EC(u) = L3 rho3(u) + L2 rho2(u) + L1 rho1(u)
L3 = (a - b) / 4,   L2 = a / 4,   L1 = a / (2 pi) - 3 (a - b) / (8 pi)
```

and `u` solves `10^6 EC(u) = fp_per_mpx`. With fixed widths it reduces to
`u exp(-u^2/2) / (2 pi)^(3/2) / (2 v)`.

Every decision (the first component, each addition, each removal) uses the
same `u^2/2`, so each false component costs the same budget.

## ROI

A boolean ROI limits where seeds are placed; fitted positions may lie
outside it. The frame is cropped to the ROI's bounding box plus the context
the screen and windows need. Use one mask for the requested area, not
separate tile calls, which would duplicate sources at seams.

## Performance

On pure noise at the default `fp_per_mpx = 16`, 11-18 false emitters per
10^6 pixels across 2-200 photons per pixel; noise more symmetric than
Poisson's (read noise) gives fewer.

`scripts/benchmark_detection.py` (sigma 1.45, background 20, 128x128;
serial, Apple M5):

| Scenario | Single fits | Mixtures |
|---|---|---|
| Isolated, flux 100: recall | 0.38 | 0.38 |
| Isolated, flux 800: rms error, ms/frame | 0.125 px, 2.3 | 0.125 px, 5.1 |
| Equal pairs at 1.5 / 2 / 3 sigma: both found | 0 / 0 / 0 | 0.93 / 1.00 / 1.00 |
| Fields 0.005 / px^2: recall, precision | 0.71, 0.96 | 0.96, 0.99 |
| Fields 0.02 / px^2 | 0.33, 0.89 | 0.85, 0.99 |
| Fields 0.04 / px^2 | 0.14, 0.83 | 0.66, 0.96 |
| Fields 0.02 / 0.04: ms/frame | 6 / 8 | 136 / 398 |

Mixture position errors match their standard errors wherever the nearest
neighbour is at least 1.5 sigma away.

## Limits

The PSF model is a Gaussian, and each window's level is a constant. Real PSF
wings, haze and defocused light wider than the window are misfit, and the
tests read misfit as signal: within the width bound, some haze is reported
as emitters. Single fits are biased by any neighbour whose light reaches the
window, out to about `6 sigma`, and at `3 sigma` a pair fits as one
component wider than the reported widths, so neither is reported: use
mixtures wherever spots crowd. Emitters closer than about `sigma` are
reported as one. Mixtures cost one window per seed, each refitting its
neighbours. `phi` is one number per frame: it takes pixels to be
independent, and read noise, whose variance does not grow with the mean,
makes it depend on the level.
