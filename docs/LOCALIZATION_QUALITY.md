# Localization measurements and diagnostics

The detector returns measured quantities without width or brightness cuts.
It retains every emitter its likelihood-ratio test keeps, including fits
with diagnostic flags.

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
model peak, fitted level, flags and `z`. `bg` is the level fitted in the
emitter's window. The full-frame `result.background` is the screening level
(NaN outside the processed crop) and the rendered residual is diagnostic:
windows fit separate levels, so there is no single shared background
surface.

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
| `NOT_CONVERGED` | 2 | The emitter's window fit stopped at its iteration limit, short of its tolerance |
| `STALLED` | 4 | That fit stopped without an acceptable step; also not converged |
| `COVARIANCE_UNAVAILABLE` | 8 | A positive finite variance could not be computed for every emitter parameter |
| `AT_BOUND` | 16 | A fitted level, flux or position is within numerical tolerance of an optimization bound |

The edge flag uses pixel boundaries at -0.5 and size-0.5, independently of
the ROI. Bound tolerance is `1e-6 * (1 + abs(bound))` plus twice the
optimizer's interior margin (`1e-10` of the parameter range). A width at its
lower bound, `width[0] * sigma`, is not flagged: that is the in-focus width,
where in-focus emitters belong.

Each emitter carries the convergence flags of its window's final fit, which
apply to every component of that window. Non-convergence and missing
covariance are separate: finite SEs do not imply convergence.

## Uncertainty and significance

SEs use the expected Fisher information of the emitter's final window fit,
every component, width and the window's level free, scaled by the frame's
dispersion. They are local approximations; bounds, low counts, light from
outside the window and PSF mismatch can invalidate coverage.

`z` (table column; `result.info["z"]`) is each emitter's
`sqrt(2 * likelihood ratio)`: what removing it, the rest of its window
refitted, costs, in standard-normal units. Every reported emitter has `z`
at least the frame's threshold `result.info["u"]`; it ranks evidence, not
precision.

## Apply analysis-specific cuts afterwards

`loctable.filter_quality` is an optional post-processing helper. It always
requires finite coordinates and positive finite position SEs. It does not
inspect flags. Its optional `max_se_pos` and `min_flux_snr` arguments apply
user-chosen precision and flux-significance cuts; neither has a default.
Derived quantities are recomputed from base columns, preserving IDs and order.

Width cuts can use `fit_sigma` or `sigma_ratio` directly. Brightness and
width alone do not identify aggregates or failed fits.

Validate optional cuts against representative images and simulations. Keep
original frame numbers and filter before linking: removing a row can end a
trajectory. Re-link after changing cuts. The linker's search radius and any
minimum track length are separate downstream decisions; see [tracking](TRACKING.md).
