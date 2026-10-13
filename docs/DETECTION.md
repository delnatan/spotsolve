# Detection

`localize` and `localize_stack` follow u-track's `pointSourceDetection`
(Aguet et al. 2013, *Dev. Cell* 26:279): screen the frame for significant
signal, seed at local maxima, fit each seed on its own window, and with
`fit_mixtures` fit several emitters per window. Every decision is a Poisson
likelihood ratio at one bar, set per movie so that a reported emitter is
real, not noise, with probability at least `p_real`.

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
   `z = A |g - gbar| / sqrt(phi * c0)`, reaches 2. The noise comes from
   the Poisson model rather than the window's residual, so a neighbour does
   not hide a spot.
2. **Seeds.** Local maxima of the negative Laplacian of Gaussian inside the
   mask. The bar sits a full unit under the lowest height the posterior
   counts (3), so every maximum above that has a seed though its seed
   score, at one width and on the pixel grid, falls short of its `z`.
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
   components would otherwise split. It is out-of-focus light only if its
   width is 2 standard errors beyond the bound: holding it at the bound, the
   rest refitted, costs `2` nats (`phi`-scaled). It is then counted in
   `info["out_of_focus"]`, not reported. Otherwise it is an emitter at the
   bound, flagged `AT_BOUND`, if it still gains `u^2/2` there. A one-sided
   test, so an emitter on the bound is lost to its fitted width in 2.3% of
   frames; deciding on the fitted width alone would lose half.
   `width=(1, 1)` fixes every width at `sigma`.
6. **Uncertainties.** Standard errors from the inverse expected Fisher
   information `J^T diag(1/m) J` of each emitter's final window fit, every
   component and the level free, scaled by `phi`.

`info["z"]` is each emitter's `sqrt(2 * cost / phi)`, the cost being what
removing it (the rest refitted) loses: at least `u`. `info["mixture"]`
numbers the windows that held several components; `info["seed"]` is the
seed each emitter was reported from; `info["duplicates"]` counts components
left to, or merged into, another seed's report.

## The bar

Each movie (`localize_stack`; a single frame is a movie of one) is first
fitted alone at the candidate bar 2: one window fit per seed, each with its
root likelihood ratio `z`. Candidate heights from 3 up have density
`f0(z) + f1(z)` per pixel: `f0` from noise alone, `f1` from emitters. An
emitter of height `z` is real with probability

```text
P(real | z) = f1(z) / (f0(z) + f1(z))
```

(Efron's two groups). `u` is the least `z` from which `P(real | z) >=
p_real`; every decision (the first component, each addition, each removal)
uses `u^2/2`, and each report carries its `P(real | z)` in
`info["p_real"]`. Fitted alone, a movie's emitters are its candidates at or
above `u`; with mixtures every frame is refitted at `u`.

**The null.** A noise maximum is a local maximum of the signed root of the
likelihood ratio, maximized over position and width: a smooth unit
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
`b = 1/v(hi)`, the expected number of maxima above `z` per pixel is the
Euler-characteristic density (Adler & Taylor 2007):

```text
EC(z) = L3 rho3(z) + L2 rho2(z) + L1 rho1(z)
L3 = (a - b) / 4,   L2 = a / 4,   L1 = a / (2 pi) - 3 (a - b) / (8 pi)
```

and `f0 = -dEC/dz`. With fixed widths it reduces to `EC = z exp(-z^2/2) /
(2 pi)^(3/2) / (2 v)`. On pure Poisson noise the detector's maxima exceed
`EC` by up to 1.4x at `z` 3-4 (Poisson skew and the width search), about
0.3 nats, which shifts `u` by under 0.1.

**The emitters.** `f1` is fitted from 3 to the highest candidate: heights
are counted in 60 bins even in `log z`, each count Poisson with mean `E0 +
E1`, `E0` the bin's expected noise maxima from `EC` and `log E1` a cubic in
`log z`. A spread of fluxes gives a density smooth in `log z`, so bright
emitters inform the faint end. A weak prior, `1e-4` emitters per pixel
spread evenly over a decade of `z`, is added to every bin as pseudo-counts:
a movie's emitters outweigh it many times over, and with few candidates it
keeps `f1` from vanishing between them.

`P(real)` is the probability of not being noise. Haze and other misfit
that the Gaussian model reads as signal are emitters to it.

## ROI

A boolean ROI limits where seeds are placed; fitted positions may lie
outside it. The frame is cropped to the ROI's bounding box plus the context
the screen and windows need. Use one mask for the requested area, not
separate tile calls, which would duplicate sources at seams.

## Performance

On pure noise, at the default `p_real = 0.9`, `u` rises to about 5: under
1 false emitter per 10^6 pixels in movies, 1-4 in single 128x128 frames,
across 2-200 photons per pixel and at gain 3. On simulated movies with
truth, reports in `P(real)` bins 0.90-0.95 and 0.95-0.99 are 92% and 96%
real.

`scripts/benchmark_detection.py` (sigma 1.45, background 20, 128x128;
serial, Apple M5):

| Scenario | Single fits | Mixtures |
|---|---|---|
| Isolated, flux 100: recall, precision | 0.51, 0.81 | 0.51, 0.81 |
| Isolated, flux 800: rms error, ms/frame | 0.125 px, 2.5 | 0.125 px, 7.6 |
| Equal pairs at 1.5 / 2 / 3 sigma: both found | 0 / 0 / 0 | 0.65 / 1.00 / 1.00 |
| Fields 0.005 / px^2: recall, precision | 0.72, 0.97 | 0.95, 1.00 |
| Fields 0.02 / px^2 | 0.32, 0.87 | 0.82, 0.99 |
| Fields 0.04 / px^2 | 0.14, 0.82 | 0.63, 0.96 |
| Fields 0.02 / 0.04: ms/frame | 6 / 9 | 118 / 361 |

Each scenario frame is a movie of one, so its bar is learned from that
frame alone.

Mixture position errors match their standard errors wherever the nearest
neighbour is at least 1.5 sigma away.

## Limits

The PSF model is a Gaussian, and each window's level is a constant. Real PSF
wings, haze and defocused light wider than the window are misfit, and the
tests read misfit as signal: within 2 standard errors of the width bound,
some haze is reported as emitters at the bound. Single fits are biased by
any neighbour whose light reaches the window, out to about `6 sigma`, and at
`3 sigma` a pair fits as one component wider than the reported widths,
reported, if at all, as one emitter at the bound: use mixtures wherever
spots crowd. Emitters closer than about `sigma` are reported as one.
Mixtures cost one window per seed, each refitting its neighbours.

The bar is learned from the heights of emitters fitted alone. A
component's `z` in a mixture is lowered by the light its neighbours share,
so where every emitter is equally bright, as in the benchmark's pairs, the
second of a close pair can fall under a bar set by the absence of moderate
heights; with a spread of fluxes it does not. `phi` is one number per
frame: it takes pixels to be independent, and read noise, whose variance
does not grow with the mean, makes it depend on the level.
