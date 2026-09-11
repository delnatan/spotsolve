# Development documentation

## The detector

One detector: the box search. `spotsolve.localize` / `localize_stack` run it
natively (`rust/spotsolve-core/src/boxsearch.rs`, bound in
`rust/spotsolve-py/src/boxsearch.rs`, wrapped by `src/spotsolve/native.py`).
`src/spotsolve/box.py` is its Python reference and the record of every
design measurement: the decision rule (ADD_NATS), ownership, sweeps, the
read-noise model, the ROI, the background surface, and the arms that were
measured and rejected (a plane per box, a wider wide class, a rank-opening
haze map, a per-pixel likelihood mask).

Parity and speed when the port landed (2026-09-11):

| | Python reference | native, 1 thread | native stack, 10 threads |
|---|---|---|---|
| glycerol f0, 256², 489 emitters | 815 ms | 107 ms | 24 ms/frame |
| GEM f0, 256², 440 emitters | 1042 ms | 104 ms | 24 ms/frame |

Identical recall, precision and inventions on all ten referee cells
(64x64 `simulate`, flat and hazy), identical counts and search-fit counts on
both real frames. About 87% of native time is inside the fits themselves.

Workflow: change the algorithm in `box.py`, measure it on the referee cells
and the real frames against the previous arm, then port the change and keep
`tests/test_localize.py` passing.

## Retired on 2026-09-11

The Bayes-factor `detect` pipeline (`core.detect`, `passes.rs`,
`dense_group.rs`, `evidence`, `prior`, `moves`) and the calibrated
PSF-bank inference stack (`spotsolve.inference`, `affine`/`inference`/
`geometry`/`uncertainty`/`search.rs`) were removed in favour of the box
search. Their code, plans (`DENSE_DETECT.md`, `RUST_GROUP_SEARCH_PLAN.md`,
`FOCUSED_EMITTER_PROPOSAL.md`, `INFERENCE_CONTRACT.md`) and results are in
the git history before that date; notes in `core.py` and `calibrate.py` that
say "measured under `detect`" refer to it.

[Historical plans and experiment results](archive/README.md) are retained as
documentation, not executable code or active instructions.
`archive/PORTING_NOTES.md` holds the implementation practices the Rust code
cites as `[Pn]`.
