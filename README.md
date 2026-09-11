# spotsolve

Spot detection and localization for fluorescence microscopy. Sparse localization
and calibrated dense-region fitting run in Rust, with Python inputs and results.

## Install

Requires Python and a Rust toolchain. In an activated virtual environment:

```sh
pip install -e ".[dev]"
maturin develop --release -m rust/spotsolve-py/Cargo.toml
```

## Sparse emitters

Use `localize_sparse` for isolated spots whose fitting windows do not materially
overlap. It runs the Aguet significance detector once, then independently fits
each candidate with a pixel-integrated Gaussian and local background.

```python
from spotsolve import localize_sparse

# image: a 2D array; sigma: expected PSF width in pixels.
spots = localize_sparse(image, sigma=1.2, offset=100.0, gain=2.4)
spots.positions   # (N, 2): y, x in pixels
spots.amplitudes  # (N,): total photoelectrons
spots.se          # (N, 3): standard errors for flux, y, x
```

Set `offset` and `gain` from your camera calibration: input is converted as
`(image - offset) / gain`. For photoelectron inputs, omit both. Set
`fit_sigma=True` to fit each spot's width; read it from `spots.fit_sigma`.
The default `alpha=0.05` is an Aguet local test size, not a frame-level
false-discovery guarantee. Returned spots have converged, interior fits with
valid conditional uncertainty; `spots.candidate_count` also counts candidates
that did not survive fitting.

## Dense emitters

`spotsolve.inference` fits a calibrated local region with zero, one or two
focused emitters, one defocused component and a smooth background. Rust owns
the joint search and position covariance, including neighbor and nuisance
coupling. Python supplies the PSF calibration and arrays.

```python
from spotsolve.inference import fit_component, position_uncertainty

# photons: nonnegative 2D observations; model: a calibrated FocusedModel.
fits = fit_component(photons, model)
uncertainty = position_uncertainty(photons, model, fits[1])
```

These are fixed-count hypotheses, not selected detections. Check
`uncertainty.status`; covariance is withheld for unsupported fits. Calibrated
source-presence decisions and full-frame dense integration remain in progress.
The existing `spotsolve.detect` pipeline remains available separately.

For that full-frame Gaussian detector, `detect(image, sigma=1.2, gain=1.0,
impl="rs")` now runs the default per-emitter variable-width ML/MAP fits in Rust.
Width priors, halo contributions and posterior curvature are retained. Its
variable-width proposal and evidence passes still run in Python; `slack=None`
uses the existing fully native fixed-width passes. See the
[implementation and local-search notes](docs/DENSE_DETECT.md).

See the [native API and calibration contract](docs/INFERENCE_CONTRACT.md) and
[current development plan](docs/FOCUSED_EMITTER_PROPOSAL.md).
