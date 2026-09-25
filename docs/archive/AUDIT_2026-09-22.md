# Code audit — 2026-09-22

## Changes

- Removed brightness-based aggregate classification, width-reporting bands,
  reject partitions and their table/script outputs. All sources selected by
  the multi-emitter count model retain their fitted quantities.
- Exposed final optimizer convergence/stalling, missing covariance, active
  bounds, edge support and unsettled frozen-neighbor context as combinable
  `FitFlag` bits. Diagnostics do not remove rows.
- Added fitted-width SEs and Fisher fractions to localization tables. `bg`
  now uses the background fitted with the spot, including Aguet's local
  fitted background, rather than the preprocessing map.
- Made refinement revisit changes in flux and width as well as position.
  Previously, stationary centers could conceal changing neighboring light.
- Corrected the optimizer gradient for negative offset-subtracted pixels.
  The objective contributes `m-d` at those pixels, so its derivative is one.
  Previously the gradient incorrectly retained the negative observation in
  `1-d/m`. Also zeroed model derivatives where the mean is floored.
- Removed iterative width calibration and its bootstrap result type. A short
  Aguet `fit_sigma` histogram example now covers choosing a search scale.
- Fixed invalid count controls that previously truncated values
  or failed later with unrelated exceptions. Added an extension output-version
  check to prevent interpreting an old width class as a new diagnostic flag.
- Shortened narrative module comments, distinguished finite errors from
  convergence, and corrected claims that residual scores are exact Poisson
  amplitude estimates or uniquely identify missed emitters.

## API migration

`frame_tables` returns `(localizations, frame_summary)`. `flags` replaces the
width-reject partition; `sigma_se` and `fisher_*` are included in the table.
`flux_ratio`, aggregate schemas/helpers/columns, `band`, `BAND`, `BAND_Z`,
`REJECT_DTYPE`, `Localizations.rejects`, `reject_table`, `calibrate_sigma`
and `SigmaCalibration` are removed, along with the calibration module.
Rebuild the native extension with the Python update. The movie script writes
localizations, frame summaries and metadata without prefiltering detections.
See [diagnostics and downstream filtering](LOCALIZATION_QUALITY.md).

## Validation

Before removing width calibration, 116 Python tests and 55 Rust tests passed.
After removal, all 105 remaining Python tests passed. The Aguet histogram
example also ran successfully on three simulated frames. Rust code was
unchanged by this removal.

Movie-table export and residual audit scripts also completed on a synthetic stack containing an empty frame
and a spot frame. Export retained flagged rows. `git diff --check` passed.

The Python suite covers simulation recovery, flux/width uncertainty, PSF
conventions, ROI behavior, threading, tables and tracking.
New regressions retain narrow, broad and very bright sources and verify that
edge flags do not depend on a width band. The Rust suite additionally checks
finite-difference gradients with negative data and clipped means, active-bound
reporting, iteration limits, singular covariance and fitted-background recovery.
The negative-data gradient regression failed before the correction.

## Limits

Candidate thresholds and count selection still determine which sources are
fitted. Width optimization bounds remain configurable through `slack`; a
bound-limited estimate can be biased and is explicitly flagged. Three-sigma
edge support and optimizer tolerances are numerical conventions, not universal
proofs of failure. Covariance treats frozen neighbors and background shape as
known and does not certify coverage under model mismatch.

An Aguet width histogram can be distorted by overlaps or broad populations;
use isolated spots or joint fits when choosing a reference width. Aguet records
failed attempts separately, since they do not yield reliable localization rows.

Historical benchmark tables describe their original detector versions. This
audit does not recalibrate those measurements or establish accuracy on every
experimental dataset. The untracked `WORKING.md` was left unchanged.
