# Development documentation

## The detector

One detector: the box search. `spotsolve.localize` / `localize_stack`
(`src/spotsolve/native.py`) call `rust/spotsolve-core/src/boxsearch.rs`
through `rust/spotsolve-py/src/boxsearch.rs`; the whole frame -- gain, FIND,
background, search, polish, classification -- runs in Rust, and the
defaults are the Rust constants. Results are `spotsolve.Localizations`.
`spotsolve.calibrate_sigma` measures the in-focus PSF width on top of it.

`boxsearch.rs` is also the record of every design measurement, each beside
the constant or function it set: the decision rule (`ADD_NATS`),
ownership, sweeps, the fit tolerances, the read-noise model, the background
surface, the seed rate, and the arms measured and rejected (a plane per
box, a width prior, a fixed-width mode, a wider model space). The fitter's
own measurements are in `lmcl.rs`, the patch radii's in `patches.rs`.

Tests: `cargo test --workspace` in `rust/` checks each layer against the
golden fixtures in `tests/fixtures/`, and `pytest` checks the detector
against `simulate` truth (`tests/test_localize.py`) and the fitter against
`psf`'s analytic Jacobian. The fixtures are FROZEN: the Python reference
that wrote them is retired (below).

Parity and speed when the port landed (2026-09-11):

| | Python reference | native, 1 thread | native stack, 10 threads |
|---|---|---|---|
| glycerol f0, 256², 489 emitters | 815 ms | 107 ms | 24 ms/frame |
| GEM f0, 256², 440 emitters | 1042 ms | 104 ms | 24 ms/frame |

Identical recall, precision and inventions on all ten referee cells
(64x64 `simulate`, flat and hazy), identical counts and search-fit counts on
both real frames. About 87% of native time is inside the fits themselves.

## Retired on 2026-09-11

The Python reference implementation (`src/spotsolve/deprecated/`: `box`,
`core`, `lmga`, `backend`, `patches`, `calibrate`, `structs`), the scripts
that generated the fixtures from it (`make_fixtures.py`) and the tests that
held the port to it. At retirement the two agreed exactly on the referee
cells, the real frames and the gain estimate; its measurement notes were
moved into the Rust. To prototype an algorithm change in Python again,
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

[Historical plans and experiment results](archive/README.md) are retained as
documentation, not executable code or active instructions.
`archive/PORTING_NOTES.md` holds the implementation practices the Rust code
cites as `[Pn]`.
