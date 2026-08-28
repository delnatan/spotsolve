# `gsolve` — how the algorithm works

A walkthrough of the spot detector, written so it can be read without the source
in front of you, and so the Rust port has something to work from besides the
Python.

This document describes **what the algorithm does and why**.
[`PORTING_NOTES.md`](PORTING_NOTES.md) records **implementation practices** — the
traps that cost time here and should be designed out in Rust.
[`BOXSOLVE.md`](BOXSOLVE.md) documents the superseded box-sequential solver and
is not a port target.

---

## Table of contents

1. [The problem](#1-the-problem)
2. [Units](#2-units)
3. [The pipeline](#3-the-pipeline)
4. [The statistical question](#4-the-statistical-question)
5. [FIND — where to look](#5-find--where-to-look)
6. [ADD — is there one more emitter](#6-add--is-there-one-more-emitter)
7. [SPLIT — the move FIND cannot make](#7-split--the-move-find-cannot-make)
8. [REFINE — the estimation half](#8-refine--the-estimation-half)
9. [PRUNE — the only removal](#9-prune--the-only-removal)
10. [The background surface](#10-the-background-surface)
11. [Why the search terminates](#11-why-the-search-terminates)
12. [The acceptance tests](#12-the-acceptance-tests)
13. [What was removed, and why it is not coming back](#13-what-was-removed-and-why-it-is-not-coming-back)
14. [Where the time goes](#14-where-the-time-goes)
15. [Known open problems](#15-known-open-problems)
16. [Knob reference](#16-knob-reference)

---

## 1. The problem

One image of diffraction-limited point sources on a background. Two questions,
in this order:

1. **How many emitters are there, and where?** — a model-selection problem.
2. **Given that, what are their positions and fluxes, and how well do we know
   them?** — an estimation problem.

The second is settled: `refine` reaches the Cramér–Rao bound with an honest
Fisher matrix (section 12). Essentially all remaining difficulty is in the
first, and within it, entirely in pairs separated by less than 2σ. Beyond 2σ
recall is 1.000 at every density tested.

## 2. Units

**Everything downstream of `detect`'s first three lines is in photoelectrons:**

```
d_e = (raw - offset) / gain
```

This is not cosmetic. The Poisson weighting `W = 1/m` used by the optimizer and
by every Fisher matrix is only valid in photoelectrons; in ADU the variance is
`gain × mean` and every standard error, Bayes factor and condition number is
wrong by a factor of the gain. `A` is **total flux**, not peak height — the
pixel-integrated Gaussian sums to `A`, so an observed peak must be divided by
`psf.peak_factor(sigma)` (≈0.104 at σ=1.2) to become an amplitude guess.

The gain is a dispersion parameter, and the fit is invariant to it: only the
decision threshold moves, as `log BF = dI/g + 1.5 log g + const`. Getting it
wrong does not corrupt positions; it corrupts how many emitters you accept.

## 3. The pipeline

```
repeat until a round accepts nothing (max_rounds is a backstop):
    FIND    LoG peaks on the variance-normalized residual  -> candidates
    ADD     for each candidate, fit K and K+1; accept on log BF > 0
    SPLIT   for each emitter, propose two along the residual quadrupole axis
    REFINE  one joint sweep at fixed N; re-estimate the background surface

REFINE      iterated to a fixed point

repeat until nothing is removed (max_settle is a backstop):
    PRUNE   per-emitter removal test, faintest first
    REFINE  iterated to a fixed point
```

Five moving parts. The layers underneath are shared and unchanged:

| module | role |
|---|---|
| `psf` | pixel-integrated Gaussian, model and Jacobian |
| `lmga` | bounded Poisson-MLE optimizer (Coleman–Li affine scaling) |
| `evidence` | Laplace log Bayes factor, `COND_GUARD` |
| `patches` | grouping of emitters into jointly-fittable patches |
| `moves` | proposal constructors, pure `theta -> theta` |
| `calibrate` | gain, robust background, model rendering |

Model selection happens in exactly three functions: `_try_add`, `_try_split`,
`_prune`. `refine` proposes nothing.

## 4. The statistical question

"Is there one more emitter here" is **not** a likelihood-ratio test. Under the
null the extra emitter sits on the boundary `A → 0`, and exactly there its
position is unidentifiable — the likelihood is flat in two of the three added
directions. Both regularity conditions for Wilks' theorem fail.

The practical consequence is worse than a fixed offset: the LRT statistic is a
maximum over the larger model's parameter space, so a **more thorough optimizer
finds a larger one**. Measured on synthetic single-emitter fields, the false
positive rate at the nominal 5% χ²(3) cutoff moved from 3% to 10% purely by
raising the restart count. A test whose size depends on how hard you search
cannot be fixed by choosing a different critical value.

So the accept rule is a **Bayes factor**, which integrates the added dimensions
against a proper prior instead of maximizing over them, and charges for the
search volume automatically:

```
log BF = (I_K - I_{K+1})                  data term, in nats
       + log(lam) - log(K+1)              prior odds on the count
       + d log pi_A                       amplitude prior difference
       + (3/2) log(2 pi)                  Laplace volume, 3 new parameters
       - (1/2)(log|F_{K+1}| - log|F_K|)   Occam factor
```

Poisson point process on emitters, positions uniform on the patch, amplitudes
`Exp(1/A_s)`, background `Uniform[0, b_max]`. The area factors cancel exactly
for a `K → K+1` move, and `b`'s prior cancels because it has the same range on
both sides. `lam` and `A_s` are re-estimated empirically each round.

Both priors have zero curvature, so the Hessian of the negative log posterior
**is** the Fisher information exactly.

## 5. FIND — where to look

`find_candidates` runs a Laplacian-of-Gaussian on the residual of the current
model, normalized by `sqrt(model)`.

Two details carry weight. It runs on the **residual**, not the image, so an
emitter already in the model is not proposed again. And the normalization is
what makes one threshold valid across the frame: under Poisson noise the
residual's own scale is the square root of the mean, so the ratio is on a fixed
σ scale wherever the model puts flux.

**FIND is saturated and is not a bottleneck.** Measured: 100% of true emitters
are proposed in every isolation bin, and dropping `CAND_THRESHOLD` tenfold
(1.5 → 0.15) changes recall by ≤0.6 points. Do not spend effort here — the
emitters that are missed are sub-2σ companions that leave no peak *by
construction*, which is what SPLIT exists for.

## 6. ADD — is there one more emitter

For each candidate, `_try_add` builds a window around it, fits the incumbent
(`K`) and the proposal (`K+1`) **on the same pixels with the same frozen halo**,
and accepts on `log BF > 0`.

Fitting both is not redundancy. Their I-divergences are only differencable into
a Bayes factor if the data, the pixel set and the frozen contribution are
identical; reusing a `K` fit from a different window would silently break that.

The window (`_window`) has three radii, all from `patches.py`:

- **free**, `LINK_FACTOR = 2.5σ` — existing emitters close enough that adding
  the candidate changes their estimates, capped at `k_max - 1` nearest so the
  joint Fisher matrix stays small;
- **frozen**, `HALO_FACTOR = 5σ` — near enough to contribute flux, folded in as
  a constant. It must be this wide: what it excludes has to be negligible
  against the *background*, not merely small against a bead peak. At 3σ a patch
  can be handed an unmodelled 3.1 e⁻ pedestal on a 4 e⁻ background;
- **pad**, `BBOX_PAD = 3σ` — pixel context. This captures **100.00%** of an
  isolated emitter's position Fisher information and ~90% of its amplitude
  information; widening it to 5σ changes no measured outcome and costs runtime
  linearly in window area.

On acceptance the **whole window's** refitted parameters are written back, not
just the new emitter's. Adding a source shifts its neighbours, and keeping their
stale values would leave the model worse than the fit that justified the
acceptance.

The only thing that can block the move besides `log BF ≤ 0` is `COND_GUARD` on
the scaled condition number: an ill-conditioned Fisher matrix makes the Occam
term meaningless, so it is not weighed against anything. See section 13 for what
used to be here and why it is gone.

## 7. SPLIT — the move FIND cannot make

Two emitters closer than about 1.5σ are fitted well by one brighter PSF, so
their residual has **no peak** for a LoG filter to find. It has a *quadrupole*:
negative in the middle, positive on two lobes along the pair axis.
`moves.residual_axis` recovers that axis from the PSF-weighted second moment of
the residual, and `_try_split` proposes two children along it at `SPLIT_DISPS =
(1.0, 1.6)σ`, keeping whichever scores higher.

Those two displacements bracket the only band still in question. Below ~1σ the
pair is not identifiable and the evidence refuses it; beyond ~2σ FIND already
produces a separate peak and the split is redundant.

`_split_pass` visits emitters **most pair-like first**. This is not an
optimization detail — a split accepted early changes its neighbours, so the
order decides which configuration later proposals are scored against.

**Known weakness.** The ranking statistic is the residual quadrupole, and
background curvature produces one too. On a strongly structured background the
ordering degrades and this move's advantage shrinks to near zero. This is the
one place where a better statistic (a projected score, marginal against the
incumbent's full parameter block) would plausibly help — but it must not be
reused as a *screen*, for the reason in section 13.

## 8. REFINE — the estimation half

`refine` re-fits at fixed N in connected groups and produces the reported
parameters and CRLBs. It proposes nothing; no move can happen here.

**It is iterated to a fixed point, and the patch decomposition is rebuilt each
sweep.** Each patch fit holds its out-of-patch neighbours frozen at whatever
positions and amplitudes it was handed, so one pass propagates any staleness in
those neighbours into the emitter they surround. Measured on isolated emitters
at density 0.055, starting the *neighbours* 0.5 px off while the target starts
at truth:

| start | med \|err\| | pull rsd |
|---|---|---|
| truth (reference) | 0.052 | 0.96 |
| target perturbed only | 0.065 | 1.26 |
| **neighbours perturbed only** | **0.136** | **1.98** |
| both | 0.146 | 2.36 |
| both, swept to convergence | 0.069 | 1.27 |

The optimizer reports converged 96–99% of the time in every row. This is not a
convergence failure — it is a *schedule* failure, and rebuilding the
decomposition each sweep is what fixes it.

Inside the round loop `refine` runs one sweep (`max_sweeps=1`), because the
round loop is itself the outer iteration. The two calls after it sweep to
convergence.

Standard errors come from the Fisher matrix of **this** fit, the one whose
parameters are reported — never from a proposal fit.

## 9. PRUNE — the only removal

One pass of removal tests, faintest first, alternating with `refine` until
nothing is removed.

**Faintest first, with write-back.** A spurious emitter is far likelier to be
faint, and removing it may make its neighbour's own removal unnecessary. When
one member of a collapsed pair goes, the other absorbs its flux, and the next
test in the same pass must be scored against *that* — so the reduced fit is
written back over the survivors. An `alive` mask is used rather than deleting as
we go, because the visit order is computed once and deleting would shift every
later index onto a different emitter.

Removal is **forced, not weighed**, when `_amplitude_var` cannot be trusted
(singular Fisher matrix, non-positive variance) or when `A < PRUNE_TAU · SE(A)`.
Weighing is not an option there: the quantity that would do the weighing is the
thing that has broken. As a pair's separation goes to zero its Fisher matrix
goes singular along the separation direction, `log|F|` collapses, and the Occam
term turns from a penalty into a large *bonus* — the Bayes factor would argue to
**keep** the degenerate pair, and the more degenerate it is the harder it
argues.

**Why removal cannot live inside the add loop.** An emitter's `A/SE` verdict
depends on which of its neighbours are free in that sweep. Letting removal feed
back into addition made an earlier global sweep oscillate with period 2 for
entire runs — the same source killed and recreated indefinitely, 2710 of 2725
accepted deaths forced by that guard. Running removal once, after the loop, at
monotonically decreasing N, cannot cycle.

## 10. The background surface

`background_map` estimates a local background from the pixels no emitter
reaches: a masked local mean, one one-sided Poisson σ-clip, a second mean, then
smoothing. Windows left with fewer than `BG_MIN_PIXELS` pixels fall back to the
global robust value.

**Estimated from masked data, never from the PSF-subtracted residual.** The
residual route is a feedback loop and it diverges in the direction that hurts:
emitters that have absorbed background depress the residual, the surface follows
them down, and they must absorb more. Measured on FOV1, the background median
fell 3.07 → 1.43 and the residual audit went from 0 interior findings to 3 while
total flux moved the "right" way the whole time.

The σ-clip is **one-sided** because an unmodelled source is always positive;
clipping both tails biases the estimate down.

**What it is worth, honestly.** Very little, and the ceiling is known:

- the masked estimator retains **84%** of the marginal Fisher information about
  `b` at density 0.055, so masking is not the loss;
- it is *bias*-dominated: +0.62 e⁻ on a 4 e⁻ background, which is **3.1× the
  per-window SE**, from PSF wings beyond the 3σ mask;
- but that error is nearly uncorrelated with local emitter flux (corr 0.002 at
  density 0.034, 0.177 at 0.055) — a near-uniform offset, which each window's
  **free `b`** absorbs;
- and the free `b` is cheap: it inflates `SE(A)` by 0.3–5.6% and `SE(position)`
  by **exactly 1.000** — the background is orthogonal to position by symmetry.

So a simultaneous amplitude/background fit, or a prior on `b`, has a ceiling of
~5% on `SE(A)` and **zero** on localization. What the surface does buy is a
visibly cleaner residual on real frames. Do not rebuild it chasing accuracy.

`_window_bg` splits each window's background into a `level` (the free `b` starts
here) and a `shape` (handed to the fit as a known additive term). The split
exists to keep `b` strictly interior — folding the whole surface into the known
term would leave `b` wanting to sit at 0, its lower bound, which is the one
thing `lmga`'s Coleman–Li scaling cannot tolerate.

## 11. Why the search terminates

By construction, not by a tolerance:

- ADD and SPLIT both take `K → K+1`. N is **monotone increasing** during the
  round loop.
- Every accepted emitter lowers the residual that produces the candidates, so
  the candidate list must eventually empty.
- PRUNE only removes. N is **monotone decreasing** during the settle loop.
- The two loops do not alternate, so no move can undo another.

There is no fixed point to chase. `max_rounds` and `max_settle` are backstops
that should not fire.

## 12. The acceptance tests

Three, in increasing order of authority.

**Recall by isolation bin.** A frame-level recall is dominated by the easy
majority and is blind to the only regime in question. `bench.py` breaks it out
by each *true* emitter's distance to its nearest neighbour.

**The pull statistic** (`crlb.py`). `z = (estimate − truth) / reported SE`, per
axis. An estimator at the CRLB with an honest Fisher matrix gives spread 1.00
and 0.3% beyond `|z| = 3`. The tool also runs an **oracle** arm — `refine` from
the true positions at the true N with the true background — so the gap between
arms is exactly what the search costs, and a deficit can be attributed to the
estimator or to the search rather than guessed at.

Do **not** use `err / CRLB`; that column has a built-in `sqrt(2)`.

**The residual audit** (`audit.py`), which is the real acceptance test on data
without ground truth. A per-pixel score test for "is there a PSF-shaped thing
here the model has not explained", calibrated by measurement rather than from a
normal table. Judging by `N` or by residual spread hides exactly the errors that
matter, because a missed emitter and an invented one cancel in any symmetric
spread.

Current state, density 0.055, flat, 12 seeds:

```
              N   recall   FP   <1s   1-2s   2-3s   >3s  rsd z  |z|>3   dA/A  s/fr
boxsolve    69.2  0.817  0.58  0.469 0.890  1.000 1.000  1.61  17.0%  +3.4%  2.05
gsolve      72.8  0.860  0.50  0.523 0.958  1.000 1.000  1.43  10.7%  +2.3%  1.06
```

Real frames, gain 4.23 — both interior-clean, all findings on the rim:

```
FOV1  N= 74  1 missed, 0 piled   z -4.9 .. +5.9   resid rsd 0.973   0.95 s
FOV2  N=128  0 missed, 4 piled   z -5.6 .. +3.7   resid rsd 1.047   1.73 s
```

## 13. What was removed, and why it is not coming back

Each of these was in the code and was removed on measurement. **Do not
reintroduce them in the port** — that is the main reason this section exists.

**The in-search `A/SE ≥ 3` veto** (`evidence.amplitudes_resolved`, applied to
proposal fits). `RESOLVED_TAU` is a validity condition on *whether the Laplace
approximation can be computed*. Applied as a veto on the proposal fit it became
the pipeline's strictest **detection rule**, deciding what exists on less
information than the joint refit and prune that follow — and the search is
forward-only, so the decision was never revisited. Instrumented: ADD accepted
96–99% of what FIND proposed and `log BF ≤ 0` fired on **none** of them, while
68–79% of every true emitter lost inside 2σ died at this guard inside
`_try_split`. Removing it: +1.7 recall points, the 1–2σ band +2.8, `rsd z` and
`|z|>3` better in all six (density × background) cells, at no runtime cost.

`RESOLVED_TAU` still belongs where it started — as a statement about the
approximation's validity, and on the removal path via `PRUNE_TAU`.

**`MIN_SEP` and `FORCE_SEP`.** A minimum-separation gate on accepted
configurations. Measured: `_separated` blocked **0 of 2510** calls; `FORCE_SEP`
was already 0. Inert.

**The Laplace corrections.** Two principled fixes for the boundary divergence
were built and measured: a truncated-normal factor `Σ log Φ(A_k/SE_k)` for the
`A ≥ 0` boundary, and a prior-overflow cap on the Laplace volume (a Gaussian may
not claim more prior mass than exists). Both are **exactly 0 nats** wherever the
fit is well determined, and in isolation the cap cuts phantom-pair accepts from
20/144 to 4/144. End to end they are worth **nothing** on top of removing the
veto, because `_prune` already catches degenerate survivors on a joint fit. They
cost an eigendecomposition per proposal. Not kept. Revisit only if a regime
appears where `_prune` cannot cope.

**Box tiling, core seams, `_adopt_orphans`.** All of `boxsolve`'s machinery for
stabilizing N. It existed to make a non-converging global sweep converge; a
forward-only search converges by construction (section 11) and does not need it.
See [`BOXSOLVE.md`](BOXSOLVE.md) if the reasoning is ever needed again.

**`score.py`'s projected score as a screen.** Correct as a *statistic* and worth
revisiting for SPLIT ranking (section 7), but it must not be used to seed and
must not be used to screen: the raw conditional score does not screen at all,
and was exceeded by the post-fit `A/SE` in 79% of cases.

## 14. Where the time goes

Profiled at density 0.055, growing the field at fixed density (times are under
`cProfile`, so absolutes are inflated ~2×; the shares are what matter):

| size | N true | N est | µs/px | `_window` | `background_map` | `lmga.fit` |
|---|---|---|---|---|---|---|
| 39 | 84 | 77 | 1407 | 1.2% | 0.1% | 88.9% |
| 62 | 211 | 177 | 983 | 1.3% | 0.1% | 88.9% |
| 96 | 507 | 447 | 995 | 2.1% | 0.2% | 86.0% |
| 128 | 901 | 781 | 1037 | 3.0% | 0.3% | 82.9% |

Two things the port should take from this:

**µs/px is flat.** `gsolve` is linear in area at fixed density. `boxsolve` was
not — its halo rendering was `O(area²)`.

**`_window` is the one super-linear term.** It measures the candidate against
*every* committed emitter, so it is `O(N)` per proposal and `O(N²)` per round —
1.2% → 3.0% as N goes 84 → 901, growing linearly with area. It is small now and
it would not be at 512×512. `background_map` and `_split_pass` have the same
shape. See `PORTING_NOTES.md` §9 for the fix.

**`lmga.fit` is 83–89% of everything.** Only fewer or better-started fits move
the total. This ranking will **not** survive the port — it is 83–89% because
`lmga.fit` is slow in Python, and that is exactly the part Rust makes 20–50×
faster. Re-profile before optimizing anything else.

## 15. Known open problems

**A sub-σ pair's survivor reports a confident SE.** The largest honest defect
left. When only one member of a pair closer than 1σ is detected, the survivor is
a well-determined single emitter sitting between two true ones. Measured at
density 0.055: it reports `med SE 0.059 px` where the oracle reports `0.354 px`,
with a position wrong by ~0.3 px and an amplitude pull spread of 8.3 — it has
absorbed two fluxes. Nothing downstream flags this state. Detecting it, or
widening its SE honestly, is the next real problem.

**Sub-σ recall is ~0.52 and neither solver does better.** At the identifiability
limit. Not obviously fixable.

**Amplitudes run 110–113% of frame flux on the bead data.** Tracks density
(+1.0% at 0.034, +2.2% at 0.055 on a perfectly flat background), so it points at
crowding — flux from an undetected close partner absorbed by the neighbour that
was detected. It is **not** background structure; the surface does not fix it
(section 10).

**The score map runs systematically negative on real frames** (median z ≈ −1.9
on FOV1, −2.2 on FOV2, on both solvers). The model very slightly over-explains
everywhere. Unexplained.

## 16. Knob reference

| knob | value | what it controls |
|---|---|---|
| `LINK_FACTOR` | 2.5σ | emitters fitted jointly |
| `HALO_FACTOR` | 5σ | emitters frozen into the model |
| `BBOX_PAD` | 3σ | pixel context around a group |
| `CAND_THRESHOLD` | 1.5 | FIND's LoG threshold — saturated, see §5 |
| `SPLIT_DISPS` | (1.0, 1.6)σ | where a split is proposed |
| **`PRUNE_TAU`** | **2.0** | **the precision/recall dial — the only one** |
| `REFINE_SWEEPS` | 4 | cap on refine's iteration to a fixed point |
| `REFINE_TOL` | 1e-3 px | position shift below which a sweep is a no-op |
| `BG_KERNEL` | 25 px | background surface window (`None` = one scalar) |
| `BG_MASK_RADIUS` | 3σ | emitter support excluded from the background |
| `BG_MIN_PIXELS` | 25 | pixels a window needs before it is believed |
| `k_max` | 12 | cap on emitters in one joint fit |
| `COND_GUARD` | 1e3 | scaled condition number above which F is unusable |

`PRUNE_TAU` is the one to move. Raising it costs localization as well as
recall — removing one member of a real close pair leaves the survivor absorbing
both fluxes and sitting between them. 2.0 is the measured optimum on both bead
frames; 2.5 brings back a +26.6 score-test peak on FOV1.

`lam` (emitter density) and `A_s` (amplitude prior scale) are **not** knobs —
they are re-estimated empirically each round.

---

## Reading order for someone new to the code

`structs.py` (the contracts) → `psf.py` → `lmga.py` → `evidence.py` →
`gsolve.py` → `crlb.py` and `audit.py` (how it is judged).
