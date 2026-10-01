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

Camera pixels are not Poisson in ADU. One scalar dispersion `phi` (variance
per unit mean, from the median squared fourth difference of the frame)
converts: `I/phi` is the log-likelihood in nats. No gain or read-noise
calibration is needed; fluxes scale with gain, positions and decisions do not.

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
5. **Widths.** Emitters are reported within `width * sigma` (by default
   `sigma` to half the window's half-side, about `2 sigma`, where a
   centred emitter keeps 91% of its light in the window). A component may
   widen past that, to the window's half-side, as out-of-focus light:
   defocused emitters and haze that narrower components would otherwise
   split up. It is counted in `info["out_of_focus"]`, not reported.
   `width=(1, 1)` fixes every width at `sigma`, as u-track does.
6. **Residual pass** (mixtures only), after DAOPHOT's subtract-and-re-find
   (Stetson 1987, *PASP* 99:191): the residual of the emitters found is
   screened again, for dim emitters a bright neighbour hid from the first
   screen's level. Each new seed's window starts with the emitters found in
   it as components, refitted but neither tested nor reported there. A new
   component within `sigma` of a known emitter reshapes it rather than
   finding a hidden one, and is not reported.
7. **Uncertainties.** Standard errors from the inverse expected Fisher
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

a slab of hyperbolic space (Siegmund & Worsley 1995, *Ann. Stat.* 23:608).
With `a = 1/v(lo)`, `b = 1/v(hi)` over the reported widths, the expected
number of maxima above `u` per pixel is the Euler-characteristic density
(Adler & Taylor 2007):

```text
EC(u) = L3 rho3(u) + L2 rho2(u) + L1 rho1(u)
L3 = (a - b) / 4,   L2 = (a + b) / 4,   L1 = (a - b) / (8 pi)
```

and `u` solves `10^6 EC(u) = fp_per_mpx`. With fixed widths it reduces to the
2-D density `u exp(-u^2/2) / (2 pi)^(3/2) / (2 v)`. Counting false maxima per
area rather than false pixels is the peak-based view of Cheng & Schwartzman
(2017, *Ann. Stat.* 45:529).

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

On 6.5 Mpx of Poisson noise per condition, at the default `fp_per_mpx = 16`:

| Photons/px | Gain | Single fits | Mixtures |
|---:|---:|---:|---:|
| 2 | 1 | 13.3 | 13.4 |
| 6.7 | 3 | 15.0 | 14.8 |
| 20 | 1 | 16.5 | 16.2 |
| 200 | 1 | 16.2 | 15.9 |

Against the 0.3.0 joint frame model (`benchmark_detection.py`; times are
serial, Apple M5):

| Scenario | Single fits | Mixtures | 0.3.0 joint |
|---|---|---|---|
| Isolated, flux 100: recall | 0.36 | 0.36 | 0.41 |
| Isolated, flux 800: rms error, ms/frame | 0.125 px, 2.6 | 0.125 px, 8.0 | 0.125 px, 139 |
| Equal pairs at 1.5 / 2 / 3 sigma: both found | 0 / 0 / 0 | 0.92 / 1.00 / 1.00 | 0.90 / 1.00 / 1.00 |
| Fields 0.005 / px^2: recall, precision | 0.71, 0.93 | 0.97, 0.99 | 0.97, 1.00 |
| Fields 0.02 / px^2 | 0.33, 0.82 | 0.85, 0.98 | 0.87, 0.99 |
| Fields 0.04 / px^2 | 0.16, 0.76 | 0.68, 0.94 | 0.72, 0.97 |
| Fields 0.02 / 0.04: ms/frame | 7 / 10 | 187 / 744 | 568 / 1673 |

The field scenarios have exact widths, where searching widths costs a few
points of recall; `width=(1, 1)` recovers them. With widths spread +-20%,
fixed-width mixtures split wider emitters (precision 0.88), free widths do
not (0.97-1.00). On 256x256 fields at 0.03 / px^2 with widths spread +-20%
(sigma 1.2), mixtures find 0.84 of emitters at precision 0.97 in 1.3 s; the
joint model found 0.83 at 0.985 in 5.2 s. Adding defocused blobs 3-6 sigma
wide raises mixtures' false emitters from 3.0 to 7.1 per 128x128 frame (the
joint model: 1.8 to 3.0).

## Limits

The PSF model is a Gaussian, and each window's level is a constant. Real PSF
wings, haze and defocused light wider than the window are misfit, and the
tests read misfit as signal: mixtures find more emitters than the joint
model did on hazy frames, some of them false. Single fits are biased by any
neighbour within a window and lose emitters closer than about `4 sigma`; use
mixtures wherever spots crowd. Emitters closer than about `sigma` are
reported as one. Mixtures cost one window per seed, each refitting its
neighbours: on dense frames that is most of the time.
