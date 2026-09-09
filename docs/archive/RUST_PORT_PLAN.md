# The Rust port: what is left, in order

**Read this first, then `PORTING_NOTES.md` §16 for the fixtures and §1–§21 for
the practices.** `README.md` says what the algorithm does; this file says what
is already ported, what is not, and the order to do it in.

Written 2026-09-03, after the session that added the two-class width prior and
the MAP width fit. It assumes no other context.

---

## 0. Where things stand

`rust/spotsolve-core` implements the **fixed-width** pipeline: the four passes,
the optimizer, the Bayes factor, the filters, patches, geometry. It passes
`tests/fixtures/*.json` at the layer tolerances in `PORTING_NOTES.md` §16.

The Python default is no longer fixed-width. Since 2026-09-03 every emitter
carries **its own width**, fitted by MAP under a prior, and an object whose
width leaves the reporting band is modelled but not reported. `core.detect`
therefore ignores `impl` unless `slack=None`, and says so.

**More of the port exists than the README used to claim.** Check before
writing anything:

| already in Rust | where |
|---|---|
| the `4K+1` parameter layout | `psf.rs`: `pack_var`, `n_emitters_var`, `sigma_var`, `unpack_var` |
| the variable-width model and Jacobian | `psf.rs`: `factors_axis_sigma`, `model_and_jac_var_sigma_ax` |
| a variable-width bounded fit | `lmcl.rs`: `ModelKind::PerEmitterSigma`, `fit_var_sigma` |
| the Python side of that call | `backend.py`: `RustBackend.fit_var_sigma` (currently orphaned — its caller was removed; it is what step 2 reconnects to) |

So layers 1–2 are largely done. What is missing is the **prior**, the **MAP
penalty**, the **Bayes factor's count-and-width term**, and threading
per-emitter widths through the passes.

---

## 1. What must cross, and what must not

The port target is nine modules, ~3,900 lines:

```
structs 137   psf 283   lmga 420   prior 371   evidence 289
moves 125     patches 177          calibrate 345            core 1729
```

**`core.py` carries a port boundary banner near its end.** Everything above it
runs inside the search. Everything below reads a finished `DetectResult`
(`flag_aggregates`, `aggregate_report`, `_link_groups`) and must not cross.

These stay in Python permanently and are not port targets: `__init__`,
`backend`, `audit`, `metrics`, `simulate`, `loctable`. So does `detect`'s round
loop, `find_candidates`, `background_map`'s convolutions, and the reporting
split at the end of `detect` — the Rust boundary is **the four passes**, as it
already is.

Three switches exist only so measurements can be attributed to one change at a
time (README §12, Provenance). **Do not port them.** Rust implements one
layout; these select the older ones in Python:

- `slack=None` — the fixed-width pipeline (this is what Rust implements today)
- `band=None` — free width, one class, one uniform prior
- `veto_widths=False` — the pre-2026-09-03 proximity veto

---

## 2. The order, and what each step buys

Port bottom-up against the fixtures. **Each layer is meaningless until the one
below it passes** — "the emitter count is different" at the top localizes to
nothing.

### Step 1 — `psf`, variable width (probably already done)

Fixture: `01_psf.var_sigma_cases`. Relative 1e-13.

**The trap it catches: column order.** `theta = [b, A₀, y₀, x₀, s₀, A₁, …]` —
sigma is the **last** of each emitter's four. A port that puts it first passes
`model` and fails `jac_flat`.

### Step 2 — `prior`, the width prior

New file, ~120 lines, no allocation, no dependencies beyond `ln`/`atan`.

Fixture: `03_evidence.width_prior_cases`. `logpdf`, `curvature`, `log_config`
to 1e-12 relative, for both `UniformWidth` and `FocusMixtureWidth`.

A trait with three methods, and the split between them is the design:

- `logpdf(sigma)` and `curvature(sigma)` — the **smooth per-emitter density**
  and its negative log-curvature. This is what the FIT sees.
- `log_config(sigmas)` — that, plus the per-class Poisson counts. This is what
  the EVIDENCE differences between two configurations.

The rate terms are a prior over **counts**; they are constant in a fixed-`N`
fit and jump only when an emitter changes class, which is a model-selection
event and not something the optimizer should walk across.

`curvature` is **clamped at zero**. The Cauchy core is not log-concave in its
tails, and an indefinite Hessian is not a curvature the optimizer can use.

`UniformWidth` is flat: `curvature ≡ 0`, `logpdf ≡ const`. See step 3 for why
that constant must never reach the optimizer.

### Step 3 — `lmcl`, the MAP penalty

Fixture: `02_lmga.var_sigma_cases`. The same data and the same start, fitted
twice — `ml` and `map` — so `ml` passing while `map` fails localizes the fault
to the penalty alone.

The penalty is diagonal and touches only the width slots. Add its value to the
objective, its gradient to `grad`, its curvature to the Gauss–Newton matrix.

**Two contracts that point in opposite directions, both deliberate:**

- `I` is returned **without** the penalty — the data term alone. `evidence`
  adds the prior itself, so a penalized `I` charges it twice.
- `F` is returned **with** it. The Laplace volume wants the Hessian of the log
  *posterior*, `F + Λ`. This is the width half of README §15's recorded
  omission, repaired.

**A flat prior must pass NO penalty, not a constant one.** `lmcl` compares
`I_cur - I_trial` against `tol_obj = 1e-8`; a constant added to both sides of
that subtraction costs low-order bits of it. On a 39×39 field the two agree;
on a dense frame, thousands of fits later, they do not. This cost a silently
drifting baseline in Python before it was caught — see `WidthPrior::is_flat`.

### Step 4 — `evidence`, the count-and-width term

Fixture: `03_evidence.var_sigma_cases`. `log_bf_add` / `log_bf_remove` on
`4K+1`, 1e-10 absolute. `antisymmetry_residual` must be **exactly 0.0**.

Two changes only:

1. The count-and-width prior comes from `WidthPrior::log_config` differenced
   over whole configurations, replacing `log(lam) - log(K+1)`. It must be a
   whole-configuration difference: under a mixture the incumbents do **not**
   cancel, because a neighbour that widens across the class boundary in the
   joint refit moves between two Poisson processes and changes both counts.
2. The Laplace volume is `2·log(2π)`, not `1.5·log(2π)`. **A port that keeps
   the fixed-width constant fails this fixture by exactly `0.5·log(2π)` =
   0.919 nats** — that is the signature to look for.

Everything else — the amplitude prior, the Occam factor — is the same
expression.

### Step 5 — per-emitter widths through `patches` and `render`

Signature change, not an algorithm change. Every radius scales with the width
of the emitter it is measured from: `_window`'s link (`LINK_FACTOR`) and halo
(`HALO_FACTOR`) radii, the bbox pad (`BBOX_PAD`), and the background mask
(`BG_MASK_RADIUS`). A defocused source reaches further, and grouping it at the
in-focus width leaves its flux out of both the joint fit and the frozen halo —
an unmodelled pedestal, which README §6 measured as the one thing the halo may
not do.

### Step 6 — the four passes on `4K+1`

Fixture: `06_passes` (currently fixed-width; regenerate with a free-width block
once steps 1–5 pass).

Also inside `add_pass`: the **proximity veto** is `core.veto_radius` — the
incumbent's own width, **capped at the reporting band's upper edge**. The cap
is not a detail: letting the radius run to the model's full width cost 8.3
points of in-focus recall at 5 emitters/µm², because a wide object vetoed a
disc big enough to hold several real emitters.

### Step 7 — end to end

Fixture: `04_end_to_end`. Two arms and you need both:

- `cases` — Gaussian fields at `slack=None`, the layout Rust implements today.
  Keep it passing throughout; it is the regression guard.
- `sim_cases` — the confocal simulation at the shipped default. **Gaussian
  fields cannot test any of this**: every emitter in them sits at the model's
  own sigma, so nothing exercises the width prior, the class boundary or the
  MAP penalty, and a port could ignore all three and pass.

Accept on `N` ±1, precision/recall ±0.03, RMSE ±0.02 px, audit counts ±1, and
**`n_wide` ±1**. Check `n_wide`: a port whose width prior is subtly wrong lands
the class boundary somewhere else, and `N` alone hides that, because a
mis-classified object stays in the model either way — it just gets reported
through a different field.

---

## 3. Constants, and which ones are trustworthy

`backend.RustBackend` asserts on load that `PRUNE_TAU`, `REFINE_TOL`,
`REFINE_TOL_OBJ`, `EVIDENCE_TOL_OBJ`, `REFINE_MAX_ITER` and `COND_GUARD` agree
between the two languages. Add `SIGMA_SLACK`, `FOCUS_BAND` and
`FOCUS_WIDTH_GAMMA` to that assertion when they cross.

All three decision-layer constants were **re-swept under the current
algorithm** on 2026-09-03 and all three survived (README §8b, §16):

| constant | value | status |
|---|---|---|
| `PRUNE_TAU` | 2.0 | re-swept on the confocal simulation; best pull spread, fewest tiles |
| `SIGMA_SLACK` | (0.70, 2.2) | re-swept under the MAP fit AND on real beads; both put it at 2.2. **But it is data-dependent** — a widefield arm with ±1 µm of depth wants ~5.0 (README §12). Port it as a parameter, not a constant, and do NOT assert on it across datasets |
| `FOCUS_BAND` | (0.80, 2.0) | a contract, not a fitted value |
| `FOCUS_WIDTH_GAMMA` | 0.20 | swept; anything in 0.10–0.30 is within a point of recall — the shape matters, the scale does not |

**`LOG_SEED_Z` is gone — do not port a seed constant.** It was the one number
that had never been swept, and it was wrong: 6.496 against a derived 3.70–4.43.
`calibrate.seed_threshold(shape, sigma, alpha)` computes it from a family-wise
false-seed rate over the independent local-maxima tests, so it **scales with
frame size and PSF width**, which no constant can. Python derives it and passes
a float in; Rust takes the float.

That is the general rule for this port, and it is the project's standing
principle: **nothing physical or statistical is hard-coded in Rust.** Anything
derivable from `(NA, λ, n, pixel_size, z_extent, frame_shape)` is computed in
Python — it is cheap, it is done once per frame, and it keeps the Rust side
pure numerics with no optics in it. Rust receives derived numbers.

A corollary worth stating because it caught a real defect: **when a pipeline
needs many tuned parameters, that is evidence the model is wrong, not that the
parameters need more tuning.** `FOCUS_WIDTH_GAMMA` was tuned to 0.20; the depth
of field puts it at 0.26 (confocal) / 0.43 (widefield), and the sweep found
0.10–0.30 indistinguishable. The tuned value landed where the physics does,
which means the knob should be replaced by the derivation, not kept.

Read README §12 "Provenance" before quoting any other number in that file: the
decision layer changed twice on 2026-09-03 and measurements taken before then
describe a pipeline that no longer exists. Arithmetic-layer measurements
(`psf`, optimizer numerics, `COND_GUARD`, `A_MIN_REL`) are unaffected.

---

## 4. What is deliberately absent

Do not reintroduce these. Each was measured and removed; README §13 has the
numbers.

- **the ROI solver** (`aguet.py`, `detect_local`, `solve_roi`, `H_wide`) —
  dominated on every accuracy axis by the mixture prior
- **the MERGE move** — collapses nothing at the shipped bound; identical arms
  with and without it. Bring it back **only** if the width schedule of §15
  lands, which restores tiling on purpose
- **the post-hoc width filter** (`infocus.py`) — asks the right question after
  the evidence for it has been divided among the tiles
- **the pre-search aggregate pass** (`find_aggregates`, `reject_aggregates`) —
  its masking plumbing sat inside the round loop serving a flag that was off by
  default
- **the NPMLE flux prior** and **`DefocusFlux`** — never wired in, and a wash,
  respectively

---

## 4b. One thing NOT to port yet, but to know about

README §5 records a measured design note: the LoG is the **trace** of the
Hessian, and `find_candidates` discards the two traceless components that carry
elongation. Recovering them is one extra separable convolution (2.0 ms on 512²
against a 7600 ms fit) and it separates an unresolved pair from a
matched-width single emitter at d ≥ 1.5σ with no fit at all.

It is **not implemented**. Do not port it. It is recorded here because if the
port ever tempts you to add elliptical emitters — two more parameters each, to
hold the anisotropy the current model cannot — the filter measures the same
thing for free, and that is the cheaper answer.

## 5. Known defects the port inherits

Port the behaviour as it is; do not "fix" these in Rust, or the fixtures will
disagree with Python for the wrong reason.

- **the round loop does not terminate at density.** At 5 emitters/µm² and on
  real bead frames it hits `max_rounds = 6` with `0 added, N split`. The width
  prior slows the subdivision; it does not stop it (README §11).
- **sub-σ pairs.** Recall ~0.52, and a survivor of a collapsed pair reports a
  confident SE. Untouched by everything in this session (README §15).
- **the amplitude half of `Λ`.** `evidence` omits the flux prior's curvature.
  Under the default `ExponentialFlux` that is exact — the density is
  log-linear — so it only bites if a caller supplies a curved `FluxPrior`.
  Port the omission as it stands. But know that it is half of a cancelling
  PAIR (README §15): the missing `Λ` rewards degeneracy, the exponential
  prior's mode-at-zero blocks sub-σ splits, and they oppose. Fixing either
  alone makes things worse. If that repair lands in Python later, it changes
  the Bayes factor and the port must follow both halves together.
- **`fit_sigma` is a shrunk estimate of defocus, not an unbiased one.** The
  width prior pulls it toward the PSF; past |z| = 0.35 µm the fitted width
  comes back at 1.56× where the truth is 2.00×. That is what makes close pairs
  resolvable, and it is the right trade — but do not read `fit_sigma` as a
  depth proxy.

---

## 6. How to check you have not broken anything

```
uv run --with pytest python -m pytest tests -q          # layer checks
uv run python scripts/make_fixtures.py                  # regenerates all 7
uv run python scripts/bench_sim.py --densities sparse moderate \
    --frames 6 --methods legacy mix                     # the accuracy arms
```

`legacy` and `mix` must reproduce these on `data/sim_out` (6 frames, derived
seed cut):

| density | arm | recall | tiles | med err | RMSE | ghosts |
|---|---|---|---|---|---|---|
| 1/µm² | legacy | 96.2% | 1.83 | 0.073 | 0.158 | 0.33 |
| 1/µm² | **mix** | **97.5%** | **1.17** | **0.070** | **0.150** | 0.33 |
| 5/µm² | legacy | 95.1% | 8.83 | 0.079 | 0.277 | 0.17 |
| 5/µm² | **mix** | 93.4% | **3.50** | 0.080 | **0.212** | 0.17 |
| 15/µm² | legacy | 84.2% | 14.33 | 0.122 | 0.326 | 0.00 |
| 15/µm² | **mix** | 83.1% | **9.33** | **0.115** | **0.286** | 0.00 |

And on a single sparse frame of `data/sim_out`, **pinned to the historical seed
cut**, the three arms must give exactly `N = 17 / 18 / 21` for default /
`band=None` / `slack=None`:

```python
spotsolve.detect(raw, sigma=0.818, offset=100, gain=2.0, verbose=0,
                 threshold=spotsolve.CAND_THRESHOLD, **arm)
```

Those three numbers are the fastest check that a refactor changed nothing.
`threshold=CAND_THRESHOLD` is load-bearing in that check: the default now
DERIVES the cut from the frame and the PSF, so leaving it out gives 18 / 22 / 36
— a different, also-correct answer that is useless as a regression signal
because it moves whenever the frame does.
