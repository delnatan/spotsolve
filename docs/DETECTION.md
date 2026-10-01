# Detection

`localize` and `localize_stack` follow u-track's `pointSourceDetection`
(Aguet et al. 2013, *Dev. Cell* 26:279): screen the frame for significant
signal, seed at local maxima, fit each seed on its own window, and with
`fit_mixtures` fit several emitters per window. The structure is u-track's;
the statistics and several mechanics are not. Each departure below was made
for a measured failure, and the measurements are in the
[validation](#validation) section.

## Model

Work in camera units: `d = frame - offset`. A window around a seed holds a
constant level `c` and components of flux `A`, centre `(y, x)` and width `s`,
each a Gaussian integrated over every pixel:

```text
m[i,j] = c + sum_k A_k * E(i; y_k, s_k) * E(j; x_k, s_k)
E(i; y, s) = 0.5 * [erf((i-y+0.5)/(s*sqrt(2))) - erf((i-y-0.5)/(s*sqrt(2)))]
I(d,m) = sum_pixels [d*log(d/m) - (d-m)]
```

Camera pixels are not Poisson in ADU. One scalar dispersion `phi`, the
variance per unit mean, converts: `I/phi` is the log-likelihood in nats. No
gain or read-noise calibration is needed; fluxes scale with gain, positions
and decisions do not.

`phi` is measured on the frame itself. The fourth difference `b` of the frame
(the separable `[1, -4, 6, -4, 1]`) has variance `70^2 phi mbar`, `mbar` the
local mean under the weights `k^2 / 70^2`, wherever the light is smooth on
its 5-pixel scale; `phi` is the median of `b^2 / mbar` over the frame, over
the median of `70^2 chi2_1`. Variance and mean come from the same pixels with
the same weights, so a background that varies across the frame, or an
emitter's light, raises both alike. An emitter's curvature adds to `b` alone
and can only raise `phi`.

## Algorithm

1. **Screen.** At every pixel, regress the `ceil(4 sigma)` window on one
   pixel-integrated PSF and a constant. The window stops at the frame's
   edge (u-track mirror-pads). The mask holds pixels whose Poisson score
   test of the flux, `z = A |g - gbar| / sqrt(phi * c0)`, reaches a bar.
   u-track instead tests `A > k * sigma_res` with the window's residual as
   noise; a neighbour inflates that residual and hides spots in crowded
   areas, which its RefineMaskLoG step then patches. Neither is needed here.
2. **Seeds.** Local maxima of the negative Laplacian of Gaussian inside the
   mask. The score is computed at a small bank of widths spanning the
   reported width range, neighbours at most 1.5x apart, each with the bar
   `u` less what the pixel grid and the gap to the next width can lose of a
   maximum. Seeds are then the sampled maxima of the field the test searches.
3. **Window fits.** Pixels of other mask components are left out of a
   window, as u-track does. Components are fitted by bounded
   Levenberg-Marquardt (Coleman-Li scaling, Marquardt damping), each with its
   own width. A component is kept only if adding it gains `u^2/2` nats:
   `2 (I_k - I_k+1) / phi >= u^2`.
   - **Single fits** (default): one component starting at the seed, its
     centre held within `2 sigma` of it (u-track's confinement).
   - **Mixtures** (`fit_mixtures=True`): components are added one at a time
     where the efficient score of a new flux peaks (Neyman's C(alpha): the
     score with the components already fitted projected out), each refit
     from the joint Newton step, while each gains `u^2/2`. Then the weakest
     is removed while removing it, the rest refitted, costs less. The seed
     only centres the window: every source of light in it takes a
     component, so a neighbour is a nuisance parameter rather than a bias.
     u-track confines components to `2 sigma`, and a neighbour 2-4 sigma
     away then drags the fit.
4. **Ownership.** Each component is reported by the fit of the seed
   nearest to it, so neighbouring windows report no emitter twice (u-track
   merges copies within 0.25 px), and only those components are tested for
   removal. A component held on a position bound is not reported.
5. **Widths.** Emitters are reported within `width * sigma`, by default
   `sigma` to `1.5 sigma`: just above `sqrt(2) sigma`, where a defocused
   emitter's peak has halved (the edge of the PSF's axial FWHM), with room
   for a `sigma` set slightly narrow. The bound is capped at half the
   window's half-side, about `2 sigma`, where a centred emitter keeps 91%
   of its light in the window. A component may widen past the bound, to the
   window's half-side, as out-of-focus light: defocused emitters and haze
   that narrower components would otherwise split up. It is counted in
   `info["out_of_focus"]`, not reported.
   `width=(1, 1)` fixes every width at `sigma`, as u-track does.
6. **Uncertainties.** Standard errors from the inverse expected Fisher
   information `J^T diag(1/m) J` of each emitter's final window fit, every
   component and the level free, scaled by `phi`.

`info["z"]` is each emitter's signed root `sqrt(2 * cost / phi)`, the cost
being what removing it (the rest refitted) loses: at least `u`.
`info["mixture"]` numbers the windows that held several components;
`info["seed"]` is the seed each emitter was reported from.

## The threshold

`u` is set so that pure noise yields `fp_per_mpx` false emitters per 10^6
pixels. A false emitter is a local maximum above `u` of the signed root of
the likelihood ratio, maximized over position and width: a smooth unit
Gaussian field over `(y, x, tau = log s)`. Scores of two profiles correlate
as the profiles do, `exp(-r^2 / (4 v))` at offset `r` with `v = s^2 + 1/12`
(pixel integration adds the `1/12`) and `2 sqrt(v1 v2) / (v1 + v2)` across
widths, so the field's metric is

```text
ds^2 = (dy^2 + dx^2) / (2 v) + dtau^2
```

which is hyperbolic space of curvature -1, with width as height (Siegmund &
Worsley 1995, *Ann. Stat.* 23:608). Widths are searched from `lo` up to the window's
half-side, and only maxima at `lo` to `hi` are reported: the region is a slab
whose one face, at `lo`, is a horosphere (both principal curvatures 1); at
`hi` the search goes on. With `a = 1/v(lo)`, `b = 1/v(hi)`, the expected
number of reported maxima above `u` per pixel is the Euler-characteristic
density (Adler & Taylor 2007), its terms the slab's volume and curvature and
the face's area and curvature:

```text
EC(u) = L3 rho3(u) + L2 rho2(u) + L1 rho1(u)
L3 = (a - b) / 4,   L2 = a / 4,   L1 = a / (2 pi) - 3 (a - b) / (8 pi)
```

and `u` solves `10^6 EC(u) = fp_per_mpx`. With fixed widths it reduces to the
2-D density `u exp(-u^2/2) / (2 pi)^(3/2) / (2 v)`. A face at `hi`, as if
widths stopped there, would add 13%: with one counted, free widths gave
0.89-0.92 of the false emitters fixed widths gave at the same
`fp_per_mpx`, where this formula predicts 0.88-0.89. Counting false maxima per area rather than false pixels is the peak-based
view of Cheng & Schwartzman (2017, *Ann. Stat.* 45:529).

Every decision (the first component, each addition, each removal) uses the
same `u^2/2`, so each false component costs the same budget. A per-test
`alpha` is not the knob: at u-track's 0.05, noise gives about 7000 seeds per
10^6 pixels. (u-track's own test is an amplitude-over-noise criterion, about
`z = 5` for an isolated spot, not a false-positive rate.) The seed bar is not
lowered further: below the sampled field's maxima, the refitted likelihood
ratio admits more false emitters than the field has, and gains no power that
a higher `fp_per_mpx` does not.

## ROI

A boolean ROI limits where seeds are placed; fitted positions may lie
outside it. The frame is cropped to the ROI's bounding box plus the context
the screens and windows need. Use one mask for the requested area, not
separate tile calls, which would duplicate sources at seams.

## Validation

`rust/spotsolve-core/tests/layer3_fit.rs` holds the window fit's
derivatives to finite differences and its errors to the reported covariance
(z-variance 0.98-1.05 per parameter over 2000 Poisson windows).
`layer7_localize.rs` holds the detector to recall, precision and error
calibration on isolated and crowded fields and to its false-positive rate on
noise. `scripts/benchmark_detection.py` scores any version on seeded
scenarios (sigma 1.45, background 20, 128x128).

On 26 Mpx of Poisson noise per condition, at the default `fp_per_mpx = 16`:

| Photons/px | Gain | Single fits | Mixtures |
|---:|---:|---:|---:|
| 2 | 1 | 11.5 | 11.4 |
| 6.7 | 3 | 16.1 | 15.6 |
| 20 | 1 | 16.8 | 16.5 |
| 200 | 1 | 18.4 | 18.3 |

Another seed gave 16.6 at 20 photons and 16.8 at 200. With the true `phi`
those frames give 0.92-0.96 of the target, seeds sitting on the pixel grid;
`phi` measured per frame scatters by 3.6%, and a frame measured low admits
more than one measured high excludes. At 2 photons per pixel noise in `mbar`
raises `phi` by 3%, and fewer are admitted. Noise more symmetric than
Poisson's admits fewer still, since the signed root of the likelihood ratio
corrects for a skew that is not there: with 1.6 photons of read noise at 6.7
photons per pixel, 8 per 10^6 pixels; Gaussian noise of the same variance,
0.8 of Poisson's count. On a background rising from 3 to 40 photons across
the frame, 18 per 10^6 pixels.

`phi` against its definition: on spot fields of 0.005-0.04 / px^2 it reads
0.99-1.02 of the true value, and on a 3-40 photon gradient 1.00. On a real
GEM movie whose frame differences are spatially white, it reads 2.41
against 2.42 from the pixels' frame-to-frame variance. Taken instead as the
median squared fourth difference over the median pixel, it read 0.84 on the
gradient (49 false emitters per 10^6 pixels), 1.11-1.31 on spot fields, and
2.22 on that movie.

Against the 0.3.0 joint frame model (`benchmark_detection.py`; times are
serial, Apple M5):

| Scenario | Single fits | Mixtures | 0.3.0 joint |
|---|---|---|---|
| Isolated, flux 100: recall | 0.38 | 0.38 | 0.41 |
| Isolated, flux 800: rms error, ms/frame | 0.125 px, 2.3 | 0.125 px, 5.1 | 0.125 px, 139 |
| Equal pairs at 1.5 / 2 / 3 sigma: both found | 0 / 0 / 0 | 0.93 / 1.00 / 1.00 | 0.90 / 1.00 / 1.00 |
| Fields 0.005 / px^2: recall, precision | 0.71, 0.96 | 0.96, 0.99 | 0.97, 1.00 |
| Fields 0.02 / px^2 | 0.33, 0.89 | 0.85, 0.98 | 0.87, 0.99 |
| Fields 0.04 / px^2 | 0.14, 0.83 | 0.66, 0.95 | 0.72, 0.97 |
| Fields 0.02 / 0.04: ms/frame | 6 / 8 | 136 / 398 | 568 / 1673 |

The field scenarios have exact widths, where searching widths costs a few
points of recall; `width=(1, 1)` recovers them. With widths spread +-20%,
fixed-width mixtures split wider emitters (precision 0.88), free widths do
not (0.97-1.00). On 256x256 fields at 0.03 / px^2 with widths spread +-20%
(sigma 1.2), mixtures find 0.841 of emitters at precision 0.977 in 0.9 s;
the joint model found 0.829 at 0.985 in 5.2 s. Adding defocused blobs 3-6
sigma wide raises mixtures' false emitters from 2.8 to 4.4 per 128x128 frame
(the joint model: 1.8 to 3.0); 1.1 per frame lie on blobs, away from any
emitter, and most of the rest are close pairs reported as one emitter
between them. A window sees only part of such a blob, which fits as a
`1.5-2 sigma` component on a raised level: under the true blobs these
components have no evidence. The default width bound is what keeps them
out: at `2 sigma` 2.1 per frame lay on blobs, at equal recall. On a real
39x39 bead image mixtures and the joint model report the same 70 beads. On
real GEM and glycerol frames 9-12% of emitters fit wider than `1.5 sigma`,
70-75% of them faint (`z <= 8`) and on diffuse light; the default leaves
them out.

Measured and not kept: a second, DAOPHOT-style pass on the residual (+2
points of recall in dense fields, but it fitted PSF misfit beside bright
beads as emitters and cost 37% of the time); a window dispersion
(quasi-likelihood F test, u-track's test in Poisson form), which raised the
bar beside bright beads until 18 of 70 real beads were lost; fitting each
mask cluster jointly (DAOPHOT's NSTAR groups), whose large clusters misfit a
single level and ran 30x slower; plane and quadratic window levels (no gain
at an equal false rate); 3-sigma windows (2x faster, 2 points less
recall, 4 of 70 beads lost); out-of-focus components wider than the window
or centred outside it (blob pieces halved, but 4-6 points of recall lost in
dense fields); and the joint model's empirical-null scale, the spread of
the residual's score far from emitters. That spread was 1.0 on blob frames
for the joint model, so it was not what kept blobs out, and from window
fits it reads crowding as misfit (2.1 on a dense exact field).

## Limits

The PSF model is a Gaussian, and each window's level is a constant. Real PSF
wings, haze and defocused light wider than the window are misfit, and the
tests read misfit as signal: within the width bound, some haze is reported
as emitters. Single fits are biased by any neighbour within a window and
lose emitters closer than about `4 sigma`: at `3 sigma` a pair fits as one
component wider than the reported widths, and neither is reported. Use
mixtures wherever spots crowd. Emitters closer than about `sigma` are
reported as one. Mixtures cost one window per seed, each refitting its
neighbours: on dense frames that is most of the time. `phi` is one number
per frame, read at the frame's finest scale: it takes pixels to be
independent, and read noise, whose variance does not grow with the mean,
makes it depend on the level.
