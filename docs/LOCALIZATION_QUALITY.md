# Localization measurements and diagnostics

Both detectors return measured quantities without width or brightness cuts.
The multi-emitter detector retains every source selected by its count model,
including fits with diagnostic flags. Aguet omits failed attempts, recording
seed coordinates and status in `result.info["failures"]`.

## Preserve the measurements

```python
import polars as pl
import spotsolve
from spotsolve import loctable

result = spotsolve.localize(frame_img, sigma=1.45, offset=100)
locs, summary = loctable.frame_tables(result, frame=0)

# Optional downstream selection. Keep locs for inspection and alternate cuts.
usable = locs.filter(pl.col("flags") == 0)
```

Tables preserve coordinates, total flux, fitted width, their marginal SEs,
model peak, fitted background, flags and Fisher fractions. `bg` is the fitted
background at the emitter's nearest pixel, excluding neighboring emitters.
The full-frame `result.background` and rendered residual remain diagnostic:
patches fit separate background levels, so there is no single shared fitted
background surface. Aguet's map is its screening background, with NaNs outside
the processed crop.

Flux and background use ADU above offset; divide flux by gain for
photoelectrons. Coordinates and widths use pixels. The `*_um` columns use
`pixel_size` supplied to `frame_tables`. `se_pos = hypot(se_y, se_x)` is radial
RMS uncertainty, not a 68% confidence-circle radius.

## Flags describe fit conditions

Flags can coexist. Test individual bits with `result.flags & int(spotsolve.FitFlag.EDGE)`.

| Flag | Value | Meaning |
|---|---:|---|
| `OK` | 0 | No reported issue; does not certify the model or uncertainty |
| `EDGE` | 1 | Three fitted sigmas extend beyond a physical image edge |
| `NOT_CONVERGED` | 2 | Final refinement did not meet its projected-gradient tolerance, or refinement was disabled |
| `STALLED` | 4 | Final optimizer stopped without an acceptable step; also not converged |
| `COVARIANCE_UNAVAILABLE` | 8 | A positive finite variance could not be computed for every emitter parameter |
| `AT_BOUND` | 16 | A fitted emitter parameter or shared background is within numerical tolerance of an optimization bound |
| `CONTEXT_UNSETTLED` | 32 | Frozen neighboring light changed after the last fit beyond refinement tolerance |

The edge flag uses pixel boundaries at -0.5 and size-0.5, independently of
reference width or ROI boundaries. A Gaussian has infinite tails; three
sigmas is the stated finite-support convention, not a proof of failure.
Bound tolerance is `1e-6 * (1 + abs(bound))` plus twice the optimizer's
strict-interiority margin (`1e-10` of the parameter range). A bound-limited fit is constrained
by the allowed model; it is not evidence that an object is biologically too
wide or bright. Refitting with appropriate bounds can resolve that condition.

The multi-emitter optimizer checks stationarity at returned parameters.
Non-convergence and missing covariance are separate: finite SEs do not imply
convergence. Convergence flags apply to the joint patch; boundary flags also
include its shared background. Frozen-neighbor changes above 0.001 pixels in
position/width or 0.1% in flux trigger another refinement, up to four sweeps.
At the limit, `CONTEXT_UNSETTLED` exposes the remaining inconsistency.

Aguet's returned fits already passed optimizer, observed-Hessian covariance,
parameter and patch-bound checks. They carry the same geometric edge flag.
Its failed attempts do not have reliable localization rows.

## Uncertainty and coupling

The multi-emitter SEs use the undamped expected Fisher matrix at the returned
fit, scaled by local dispersion. Aguet uses observed curvature and propagates
amplitude-width covariance into total-flux uncertainty. These are local model
approximations; bounds, low counts, overlap and model mismatch can invalidate
coverage. Frozen neighbors and estimated background shape are treated as known.

`result.info["fisher_fraction"]` and table columns `fisher_flux`, `fisher_y`,
`fisher_x`, `fisher_sigma` report conditional/marginal variance:

```text
fraction[q] = 1 / (F[q,q] * inverse(F)[q,q])
```

Small fractions mean strong coupling to jointly fitted parameters. They do
not necessarily mean poor absolute precision, and are not rejection rules.
Unavailable values are NaN, including all Fisher fractions for Aguet.
Tables carry them with row IDs, so they stay aligned after sorting/filtering.

## Apply analysis-specific cuts afterwards

`loctable.filter_quality` is an optional post-processing helper. It always
requires finite coordinates and positive finite position SEs. It does not
inspect flags. Its optional `max_se_pos` and `min_flux_snr` arguments apply
user-chosen precision and flux-significance cuts; neither has a default.
Derived quantities are recomputed from base columns, preserving IDs and order.

Width cuts can use `fit_sigma` or `sigma_ratio` directly. A sampled-Gaussian
Aguet width differs from a pixel-integrated width; see [PSF conventions](AGUET_BASELINE.md).
Brightness and width alone do not identify aggregates or failed fits.

For Aguet patch diagnostics, `2 * result.info["objective"]` is Poisson
deviance. Large residuals can reflect overlap, background or PSF mismatch;
there is no universal cutoff. The whole-image residual uses a diagnostic
background and should not replace the fitted patch objective.

Validate optional cuts against representative images and simulations. Keep
original frame numbers and filter before linking: removing a row can end a
trajectory. Re-link after changing cuts. [Track length and link margin](TRACKING.md)
are separate downstream decisions.
