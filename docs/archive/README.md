# Archived development history

The documents here are superseded plans, experiment records and the former
long-form project README. Their recommendations and performance claims belong
to their recorded versions. The current entry point is [../README.md](../README.md).

Key history:

- [DETECTOR_DESIGN_NOTES.md](DETECTOR_DESIGN_NOTES.md): detector and fitter
  benchmarks moved out of source comments; preserved without revalidation.
- `COMPONENT_MODEL_REASSESSMENT.md`: shared nuisance and anti-tiling scope.
- `OFFGRID_SPARSITY_CHECKPOINT.md`, `SUPPORT_CALIBRATION.md`,
  `HAZE_SUPPORT_COMPARISON.md`: shrinkage/support experiments and limitations.
- `DAOSTORM_BASELINE.md`: completed external comparison, now reference only.
- `RUST_PORT_PLAN.md`, `PORTING_NOTES.md`: prior port contracts and practices.
- `ALGORITHM_HISTORY.md`: original full README, preserved verbatim.

The executable experiment runners and `spotsolve.prototype` implementation were
removed after their conclusions were incorporated into the native design. Old
commands in these documents are historical records and are no longer expected
to run. Source fingerprints deliberately identify the removed files; do not
reuse their calibration tables with edited code.

`manifest.json` records original/archive paths and hashes. Existing generated
artifacts under `reference/` and input data were retained. Historical relative
links and code paths in the preserved narratives may refer to the former layout.

## Retired on 2026-09-11

The Python reference implementation (`src/spotsolve/deprecated/`: `box`,
`core`, `lmga`, `backend`, `patches`, `calibrate`, `structs`), the scripts
that generated the fixtures from it (`make_fixtures.py`) and the tests that
held the port to it. At retirement the two agreed exactly on the referee
cells, the real frames and the gain estimate; its measurement notes are retained in
[DETECTOR_DESIGN_NOTES.md](DETECTOR_DESIGN_NOTES.md). To prototype an algorithm change in Python again,
restore it from commit `ea6b17f`.

The sparse localizer (`localize_sparse`, `sparse.rs`): on sparse fields it
was no faster than the box search (3.2 vs 2.8 ms/frame at 0.002 px^-2),
recalled .57-.79 against .96-.99, and its fitted sigma read 1.27 / 1.18 /
1.10 for a true 1.30 as density rose -- undetected neighbours in its 10 px
windows. The fixed-width fitter only it used went with it.

The Bayes-factor `detect` pipeline (`core.detect`, `passes.rs`,
`dense_group.rs`, `evidence`, `prior`, `moves`) and the calibrated
PSF-bank inference stack (`spotsolve.inference`, `affine`/`inference`/
`geometry`/`uncertainty`/`search.rs`) were removed in favour of the box
search. Their code, plans (`DENSE_DETECT.md`, `RUST_GROUP_SEARCH_PLAN.md`,
`FOCUSED_EMITTER_PROPOSAL.md`, `INFERENCE_CONTRACT.md`) and results are in
the git history before that date; notes in the Rust that say "measured
under the retired `detect`" refer to it.


The current `localize_aguet` is a new port of `spotfitlm`, separate from the
retired `localize_sparse`. See the [Aguet baseline](../AGUET_BASELINE.md).

## Retired on 2026-09-24

The Rust detector became its own reference. Removed, all present in commit
`002cb04`:

- the Python score-gate prototype (`src/spotsolve/scoregate.py`), the joint
  model prototype (`scripts/jointfit_prototype.py`), their experiment
  scripts and `output/scoregate/`;
- the parity fixtures that pinned the Rust to them
  (`tests/fixtures/10_scoregate.json`, `11_joint.json`, their generators and
  `layer7_scoregate.rs` / `layer8_joint.rs`); statistical tests on simulated
  fields and pure noise replaced them;
- the windowed one-pass search that seeded the joint model: seeds now enter
  the joint model directly;
- the experimental BIC count selection (`COUNT_SELECTION.md`, its benchmark
  and `output/hyp7gem_bic_*`), already gone from the API.

`AUDIT_2026-09-22.md` is the code audit of the detector before the score gate.
