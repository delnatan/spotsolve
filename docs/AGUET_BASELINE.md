# Aguet / spotfitlm sparse baseline

```python
from spotsolve import localize_aguet, localize_aguet_stack

spots = localize_aguet(frame, sigma=1.45, offset=100, roi=mask)
movie = localize_aguet_stack(
    stack, sigma=1.45, offset=100, roi=mask, n_threads=5,
)
```

Returns `Localizations`, compatible with `loctable` and linking. Defaults:
`offset=0`, `significance=0.05`, `boxsize=9`, `itermax=50`, `images=False`.
Stack workers default to the machine's cores. Smaller significance values
make screening stricter; this is a per-location test, not frame-wide false
positive or false discovery control.

## Reference and method

The reference is the sibling `spotfitlm` checkout at revision
`e0f4036de786ac3a14c15c5a353e60231fc030a6`:
[screening](../../spotfitlm/spotfitlm/utils.py),
[Python fitting wrapper](../../spotfitlm/spotfitlm/fitters.py),
[C batch fitter](../../spotfitlm/c_src/gfit.c),
[optimizer](../../spotfitlm/c_src/glm_core.c), and
[model/derivatives](../../spotfitlm/c_src/user_funcs.c).

1. Regress a sampled Gaussian plus constant background in each local window;
   screen amplitude against residual noise.
2. Keep LoG maxima passing that screen, reference border cuts and the ROI.
3. Fit each candidate once with `A*exp(-r²/(2*sigma²)) + background`, varying
   position, width, amplitude and background. Retain converged fits with valid
   covariance, positive amplitude/background and valid image bounds.

There is no mixture/count search or additional post-fit significance test.
Amplitude/width filters remain optional downstream choices. Overlapping sources
can bias independent fits. The Poisson uncertainty retains the reference's
noise assumptions: offset subtraction does not correct gain or read noise.

Fitted widths are not directly interchangeable with the multi-emitter
detector's pixel-integrated widths. Integrating a Gaussian over unit pixels
convolves it with a unit-width box, adding `1/12` pixel² to the continuous
profile variance. A sampled-Gaussian fit therefore typically gives
`sigma_sampled ≈ sqrt(sigma_integrated² + 1/12)` (about 2% larger at sigma
1.45). This is a moment-based approximation, not an exact fit correction:
finite windows, background, subpixel position and noise can affect the fit.

For an approximate detection width, inspect the `fit_sigma` histogram from
an Aguet pass over the first frame or few frames. Prefer isolated spots or an
isolated ROI; overlap can inflate independent-fit widths. This is a small
analysis step using the existing outputs, not a separate calibration routine.
See the [example](../README.md#choose-a-detection-width).

## Screening simplification

For kernel pixel count `n`, significance `alpha`, and amplitude entry `C00`
of the inverse regression normal matrix:

```text
k = normal_quantile(1 - alpha/2)
s = sqrt(RSS/(n - 1))
a = (n - 1)/(n - 3) * C00
b = k²/(2*(n - 1))
nu = (n - 1)*(a + b)²/(a² + b²)
q = k + student_t_quantile(1 - alpha, nu) * sqrt((a + b)/n)
```

The reference variances are `a*s²` and `b*s²`. Degrees of freedom are therefore
constant, and the test reduces to **`A > q*s`** for finite positive RSS.
Python caches `q`; Rust uses separable Gaussian passes and running box sums.
Regression support is `ceil(4*sigma)`; LoG retains SciPy's own rounding rule.
Zero/nonfinite RSS fails screening. Threshold-adjacent rounding can change a
decision despite algebraic equivalence.

## Masks, fitting and guards

An ROI selects integer seed centers, with surrounding pixels retained as
context; fitted centers may leave the ROI. Process its bounding box plus the
maximum of regression radius, LoG-plus-maximum-filter radius, and patch
half-width. Border cuts use global coordinates. Empty masks return no fits;
scattered masks may save little work.

The shared native scheduler preserves frame order and reuses worker storage,
with no nested pools. The dedicated five-parameter fitter reuses patch/model/
Jacobian buffers, uses `d/model²` step curvature, and computes covariance from
the full observed Hessian. It shares linear algebra with the dense fitter,
but preserves the reference's sampled PSF and optimization rule.

Intentional corrections to invalid/degenerate reference cases:

- Validate finite data/settings, `0 < sigma <= max(frame.shape)`,
  `0 < significance < 1`, odd `boxsize >= 3`, positive iterations/workers.
  Skip seeds whose patch would leave the frame; oversized boxes yield no fits.
- Floor fit observations at `1e-7` in both objective and derivatives; the
  original applies the floor only in its objective. Reject invalid trial
  means/widths and recompute the final objective at the returned parameters.
- Handle empty stacks/results without the original wrapper's index or
  DataFrame-concatenation failures.

## Outputs

The [localization-quality guide](LOCALIZATION_QUALITY.md) shows how to filter
coordinate precision and optionally inspect patch deviance without adding
diagnostic columns to the results.

Positions are `(y, x)`. Continuous sampled-Gaussian flux is `F=2*pi*A*sigma²`:

```text
Var(F) = (2*pi*sigma²)² Var(A)
       + (4*pi*A*sigma)² Var(sigma)
       + 2*(2*pi*sigma²)*(4*pi*A*sigma) Cov(A, sigma)
```

`info` retains seed positions, peak amplitudes, fitted backgrounds,
objectives/iterations, candidate counts, processed pixels and failures.
Failures are `(seed_y, seed_x, status)`: -1 iteration limit, -2 covariance
failure, -3 invalid fit, -4 post-fit bounds. There is no width band, so
Returned rows carry geometric `FitFlag.EDGE` diagnostics; dispersion is
unavailable (NaN). Failed attempts remain in `info["failures"]`.

`background` is the screening estimate, NaN outside the crop. Optional
model/residual images use this diagnostic map and sampled PSFs truncated at
four fitted sigmas. Independent patch backgrounds do not define a unique
global background surface.

## Validation and speed

Frozen `tests/fixtures/09_aguet.json` pins source hashes and revision.
`scripts/make_aguet_fixture.py` generated five candidate frames and 18 original
C fits, including iteration-limit failures and full covariance. Tests check
reference fits, finite-difference derivatives, screening boundaries, masks,
threads, flux covariance, tables, rendering, empty inputs and failures.

```sh
python scripts/benchmark_aguet.py --out /tmp/aguet-benchmark.json
```

The runner needs the sibling reference and a C compiler. It compiles the
original C batch fitter and runs its Python detector, excluding pandas
construction but including native `Localizations` construction. The
[recorded benchmark](aguet_benchmark.json) uses seed 19037: 24 128x128 frames,
16 sparse sources/frame, sigma 1.45, Poisson background 20, peak 7–100.
Times are medians of five warm runs on the development Apple Silicon host.

| All 24 frames | spotfitlm, serial | Native, serial | Native, five threads |
|---|---:|---:|---:|
| Full frame | 172.1 ms | 26.3 ms | 7.32 ms |
| Upper-left quarter ROI | 166.2 ms | 8.81 ms | 2.68 ms |

Full-frame speedup is 6.5× serial or 23.5× with five workers. Counts match
in both mask cases; maximum reference position difference is `7.1e-15` pixels.
Thread counts give identical localization arrays. Masking reduces processed
pixels from 16,384 to 5,184.

At default significance, 373 of 384 sources matched within one pixel out of
374 detections (97.1% recall, 99.7% precision); 24 empty Poisson frames gave
zero detections. This limited simulation does not calibrate false-positive
control on real/crowded images. Comparison with the dense detector at matched
false-spot budgets remains a separate accuracy study.
