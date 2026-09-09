# Runnable sparse prototype and visual output

`spotsolve.prototype.fit_sparse_patch` now wraps the continuous-amplitude prior
fit and unpenalized support refit into one callable interface. It returns
candidate positions/fluxes, the fitted image, separate focused and nuisance
images, signed Poisson deviance residuals, original fits and diagnostics.

```python
from spotsolve.prototype import fit_sparse_patch

# Original camera-converted photon pixels, including surrounding context.
result = fit_sparse_patch(
    photons_21x21, sigma=1.2,
    focus_bounds=(7, 7, 13, 13),
    focus_rate=0.03,
)
positions_yx = result.candidate_positions
flux_photons = result.candidate_fluxes
```

The rate and focus domain are explicit inputs. The example rate is an
illustrative sensitivity setting, not a calibrated production default.
Positions remain continuous; focused width stays fixed. Broad light and
background are jointly fitted. The model has capacity for two focused
emitters. Positive slots are candidates, not calibrated detections. This is
a whole-patch prototype, not a full-frame ownership/selection implementation;
do not tile a crowded frame and concatenate its outputs.

Generate the complete simulation and real-image demonstration:

```sh
source ~/uv-workspaces/microscopy/.venv/bin/activate
MPLCONFIGDIR=/tmp/spotsolve-mpl python scripts/demo_sparse_prototype.py \
  --real --output reference/sparse-prototype-demo
```

Omit `--real` to run without the local glycerol TIFF. Outputs:

* `simulation.png`: six fixed seeded simulations, including negative controls.
* `glycerol.png`: four original patches from glycerol frames 0 and 13, using
  the same locations as the preceding prior audit; native counts are unknown.
* `results.json`: coordinates, photon amplitudes, simulation truth/matching,
  timings, optimizer diagnostics, settings and source/input fingerprints.
* `simulation_maps.npz` and `glycerol_maps.npz`: original photons and separate
  fitted mean, nuisance, focused-light and residual arrays for every row.

Every figure compares unregularized and sparse/refitted candidates on the
same pixels, with identical grayscale limits within a row. Cyan circles mark
known simulated focused sources; red crosses are fitted candidates. Dotted
boxes delimit the allowed focus domain. Sources outside that box are context,
not eligible focused detections in that experiment. Residual color limits
are fixed at [-4,4]; larger magnitudes saturate the scale. The fitted nuisance
panel uses the same photon display limits as the raw image.

The fixed demonstration seeds 20260908--20260913 are not selected by favorable
fit results. At the illustrative rate .03 per photon, single-frame position
errors are .0203 pixels for the isolated source, .1653 pixels RMSE for the
equal pair, .1757 pixels for the 4:1 pair, and .0932 pixels for the single in
haze. Equal/unequal pair relative separation errors are 9.9%/21.0%. These are
examples, not average precision claims.

The failures are equally important: the defocused-only example still has two
focused candidates (about 116 and 106 photons after refitting), and haze-only
has one (about 16 photons, at the focus-domain boundary). In the real panel,
three patches have no focused candidates within the fixed central domain;
one changes from two to one after sparsity. These are not native accuracy
scores. Strong structured real-image residuals show incomplete modeling of
neighboring light and prevent claiming production readiness.

Prototype diagnostic flags expose uncalibrated support, exhausted two-slot
capacity, focus-domain boundaries, optimizer failure and numerical local rank
deficiency. They are not an automatic acceptance/veto policy. Production
`spotsolve.detect` remains unchanged. The next required work is a validated
support/count decision and consistent ownership/modeling of neighboring light.
