# Multi-emitter detection

`localize` and `localize_stack` fit overlapping sources jointly. The default
is `selection="fixed"`; [experimental BIC](COUNT_SELECTION.md) changes count
selection. The separate [Aguet baseline](AGUET_BASELINE.md) uses independent
sampled-Gaussian fits. This page describes the multi-emitter model and records
its earlier validation measurements.

## Model and noise

Work in camera units: `d = frame - offset`. In 25-pixel windows, estimate
pixel noise from a local median of a fourth-difference filter, then estimate
local dispersion `phi` as variance divided by local median intensity. The
noise maps are evaluated on a 12-pixel grid and interpolated.

Each emitter has flux `A`, position `(y, x)` and width `s`. Integrate its
Gaussian over each pixel:

```text
m[i,j] = background[i,j] + sum_k A_k * E(i; y_k, s_k) * E(j; x_k, s_k)
E(i; c, s) = 0.5 * [erf((i-c+0.5)/(s*sqrt(2))) - erf((i-c-0.5)/(s*sqrt(2)))]
I(d,m) = sum_pixels [d*log(d/m) - (d-m)]
```

The background shape is a 25-pixel local mean away from candidates; its level
is refitted per region. Fits minimize Poisson I-divergence with bounded
Levenberg–Marquardt/Fisher scoring. For Poisson photoelectron data, differences
in `I` are log-likelihood ratios. For camera data, comparisons use `I/phi` as
a measured-noise approximation. No gain/read-noise calibration is required.
Fluxes scale with gain; fitted geometry should remain stable, subject to
roundoff and crowded-model search decisions.

## Search and width reporting

1. Find LoG maxima exceeding `threshold` (default 2.75) in local noise units.
2. Group candidates within 2.5 `sigma`, with at most `k_max=12` emitters per
   box and 3 `sigma` of padding.
3. In fixed mode, start from background-only and add the strongest eligible
   residual peak, jointly refitting the box. Require an improvement exceeding
   `(10 + count_penalty)*phi`. Remove sources whose removal costs less than
   that amount. Process boxes brightest first, holding neighboring light
   fixed; repeat against updated neighbors.
4. Jointly refine nearby groups for up to four passes, stopping below 0.001
   pixel movement. BIC additionally rechecks removals on the original group
   pixels and refits the retained sources; see its separate search description.
5. Report positions, fluxes, widths and Fisher-based SEs scaled by local
   dispersion, then apply the width-reporting rule.

Widths fit within `slack=(0.7, 2.2)` times the supplied `sigma`. The reporting
band is `(0.8, 2.0)`, allowing an extra two width SEs (`BAND_Z`) to avoid
rejecting dim sources merely because width is uncertain. `band=None` reports
all fits. Rejected fits still contribute light to the model and appear in
`rejects`:

| Reason | Meaning |
|---|---|
| `too_narrow` | Significantly below 0.8 `sigma`; often noise or a fit distorted by neighbors |
| `too_wide` | Significantly above 2.0 `sigma`, or at the 2.2 `sigma` fitting limit, away from an edge |
| `edge` | Outside the band and within one `sigma` of the frame border |

Broad objects can be defocused sources, aggregates or haze. Fitting their
width avoids explaining them as several narrow sources. Bright aggregates
with ordinary width need a separate brightness flag (`flag_aggregates`);
width alone does not identify them.

## Curvature and uncertainty

The fixed 10-nat cost is an empirical complexity penalty. A Fisher
log-determinant alone is not Bayesian evidence: its value depends on parameter
units, and a Bayes factor also requires specified, normalized priors. When
two mixture components coincide, the model is non-identifiable and the usual
isolated quadratic-mode Laplace approximation fails. This is a general
[singular-model limitation](https://www.jmlr.org/papers/v14/watanabe13a.html),
also relevant to interpreting the experimental BIC score.

The optimizer factors a **damped** Fisher matrix to choose steps. Uncertainties
use a separate factorization of the **undamped** expected information
`F = J.T @ diag(1/m) @ J` at the returned parameters, including fitted
background. Reported variances are multiplied by local dispersion. If that
factorization fails, uncertainties become NaN; they cannot be carried over
from a previous fit. No diagonal jitter hides this failure.

`result.info['fisher_fraction']` has shape `(N, 4)`, aligned with detections,
in `(flux, y, x, sigma)` order. For each parameter `q` it reports

```text
fraction[q] = 1 / (F[q,q] * inverse(F)[q,q])
           = conditional variance / marginal variance
```

This is reciprocal variance inflation in the local quadratic model: 1 means
no coupling to other fitted parameters; values near zero mean strong
confounding. It is invariant to diagonal changes of parameter units,
parameter ordering and a common dispersion scale. It reuses the inverse
diagonal already needed for SEs. `info['reject_fisher_fraction']` aligns with
`rejects`; both arrays are NaN where covariance is unavailable, including
when native refinement is disabled. Neither array changes selection or
reporting. Read it alongside SEs: a weak isolated source can be imprecise
without strong confounding, and a bright crowded source can have both small
SEs and a small fraction. Frozen neighbors and estimated background shape
are treated as known, so this is not a full uncertainty budget.

The 2026-09-17 working note supports retaining expected Fisher information
and the fixed penalty, but its scratch simulations were not supplied and its
numerical comparisons are not reproduced here. Several qualifications matter:

- Raw condition numbers and Cholesky pivots depend on units; a large value
  alone does not establish failure of Laplace. Coincident components provide
  the structural reason. Adding a flat prior's normalization does not cure a
  singular local Gaussian approximation, and a flat prior has no interior
  curvature. A fixed penalty is not generally equivalent to a Bayes factor.
- Expected Fisher information is positive semidefinite, not guaranteed
  invertible. The observed Hessian also must be positive semidefinite at an
  exact interior minimum; an indefinite result calls for checking convergence,
  bounds and numerical differentiation. Neither curvature gives reliable
  asymptotic coverage automatically at low signal or active bounds.
- Here `I` is half the conventional Poisson deviance; count comparisons use
  changes in `I/phi`. Comparisons of noisy SE estimates do not by themselves
  establish which covariance gives better tracking decisions.

For a minimal filtering workflow using existing outputs, see the
[localization-quality guide](LOCALIZATION_QUALITY.md).

## ROI and calibration

A boolean ROI restricts the search; sources may fit outside it. Process its
bounding box plus context and estimate the reference background level from
masked pixels. Under a cell mask, that avoids using the dark field outside
the cell as its background. Separate tile calls can duplicate sources at
seams and omit neighboring light; use one mask for the requested area.

`calibrate_sigma` disables the reporting band and repeatedly takes the median
fitted width until stable. The validation recovered a common simulated width
to about 3% from guesses within about 25%. Broad populations bias the median
upward. Inspect `cal.widths`; on bright in-focus detections, the median
`sigma_ratio` should be near one.

## Earlier measurements

These measurements describe the default detector and the versions recorded
in [design history](archive/DETECTOR_DESIGN_NOTES.md). They are not new Aguet
or BIC evaluations. Use their own [BIC](COUNT_SELECTION.md) and
[Aguet](AGUET_BASELINE.md) benchmarks for those modes.

### Proposal threshold

Real GEM frames with known-brightness simulated particles added; unmatched
spots counted on two matched simulations. Recall entries correspond to
D = 0 / 0.43 / 2 px² per frame. Counts/recall use 128x128 images; serial
timing uses 256x256 frames.

| Threshold | Unmatched/frame | Recall, 150 e− | Recall, 300 e− | Spots/frame | ms/frame |
|---|---:|---|---|---:|---:|
| 2.5 | 24.9 | .51 / .41 / .33 | .78 / .73 / .60 | 241 | 327 |
| 2.75 | 22.5 | .50 / .39 / .32 | .77 / .73 / .59 | 232 | 280 |
| 3.0 | 20.0 | .50 / .36 / .31 | .77 / .71 / .59 | 224 | 237 |

The default 2.75 replaced separate 3.0 frame / 2.5 residual cuts with recall
within about one percentage point. Raising it toward 3.0 reduced unmatched
spots and runtime in this dataset. Relative to the preceding detector,
default unmatched counts rose 13 → 22.5, while recall rose .39/.27/.24 →
.50/.39/.32 at 150 e− and .67/.63/.53 → .77/.73/.59 at 300 e−, at about
1.5× runtime. These tradeoffs are why count selection is evaluated separately.

### Width, masks and camera noise

- Variable-width/hazy simulations: fixed widths gave 8–33 false spots per
  64x64 frame versus 0.7–7 with fitted widths. On glycerol/GEM data, fixing
  width put 136–211 detections/frame inside objects otherwise labeled too wide.
  Fixed width helped only when sources shared one width in those tests.
- A 32x32 ROI on a 512x512 image reduced runtime from 25.4 to 3.4 ms.
- At three simulated densities, true-camera-calibration recall/precision was
  .850/.981, .682/.917, .511/.842; estimated-noise results were .855/.963,
  .683/.913, .510/.839 (128x128, gain 2.4, read noise 1.6 e−).
- Empty frames with background 1–20 e− and read noise 1.6–2.5 e− produced
  0–0.7 spots/frame with estimated noise, versus up to 13 using plain Poisson.
- Measured dispersion was 2.42–2.61 on four GEM crops (calibration implied
  2.55–2.85), and 2.41 on glycerol beads (2.23 expected). Tiny crowded 39x39
  bead images were less reliable: 1.90 versus 2.15.
- Rescaling a frame rescaled its fluxes; crowded decompositions differed by
  up to two detections through numerical sensitivity.

### Limits

Haze and defocused structure beyond the smooth background remain model
mismatch. On glycerol beads, residuals had a positive ring 2–4 pixels from
sources, about 0.3 noise SD. Sources closer than roughly one `sigma` can be
reported as one brighter source.

Shifting the image changes the noise-grid sampling and crowded search. On
ten crowded 256x256 GEM frames, a 12-pixel shift changed counts by 2.4% and
recovered 84% of interior detections within 0.5 pixels; a five-pixel shift
changed counts by 2.1% and recovered 76%. Most sensitivity came from the
crowded search, but grid alignment also matters.

### Original native port

At the 2026-09-11 port validation:

| Frame | Python reference | Native, one thread | Native stack, ten threads |
|---|---:|---:|---:|
| Glycerol f0, 256x256, 489 emitters | 815 ms | 107 ms | 24 ms/frame |
| GEM f0, 256x256, 440 emitters | 1042 ms | 104 ms | 24 ms/frame |

Recall, precision and invented counts matched on ten flat/hazy 64x64
simulation cases; counts and search-fit counts matched on both real frames.
About 87% of native time was in fitting. These are historical port timings,
not the recent BIC or Aguet benchmarks.
