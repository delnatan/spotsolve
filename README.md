# spotsolve

Multi-emitter spot detection and localization for fluorescence microscopy. The
question it answers is not "where are the spots?" but **how many are there, and
where** — the two are one estimation problem, and a detector that answers the
first with a fixed threshold cannot answer the second when emitters overlap.
`detect` decides N by a Laplace Bayes factor over a bounded Poisson MLE, so
every emitter in the returned list has paid for itself in evidence.

```bash
pip install -e .                    # library: numpy + scipy only
pip install -e ".[scripts]"         # + tifffile, matplotlib, polars for scripts/
```

```python
import spotsolve

res = spotsolve.detect(img, sigma=1.45, offset=100.0, gain=2.401)
res.positions     # (N, 2) float (y, x), pixels
res.amplitudes    # (N,)   total flux, photoelectrons
res.se            # (N, 2) reported standard errors, pixels
```

```bash
python scripts/run.py --gain 4.23   # one real frame in, detections + audit panel out
pytest                              # the four layer checks
```

An optional Rust core runs the four inner passes ~2× faster with identical
results — see [Using the Rust core](#using-the-rust-core).

---

## About this document

The rest of this file is a walkthrough of **what the algorithm does and why**,
written so it can be read without the source in front of you, and so the Rust
port has something to work from besides the Python.
[`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) records **implementation
practices** — the traps that cost time here and should be designed out in Rust.

`spotsolve` replaced an earlier box-sequential solver, `boxsolve`. That code and
its walkthrough were removed once `spotsolve` superseded them; both are
recoverable from git history at the commit tagged in section 13 if the reasoning
is ever needed again.

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
10b. [Over-bright and over-wide detections](#10b-over-bright-and-over-wide-detections-aggregates)
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
wrong by a factor of the gain.

### What `A` — “flux” — actually is

`A` is the **integral of one emitter's PSF over the whole plane, in
photoelectrons**. The model is built so that the pixel-integrated Gaussian sums
to exactly `A`:

```
sum over all pixels of  A · ey_k[i] · ex_k[j]  =  A          (checked: 1000.0000 for A = 1000)
```

so `A` reads as *the number of photoelectrons this emitter contributed to this
exposure*. Three consequences, each of which has cost time here:

- **It is not a peak height.** The brightest pixel of an isolated emitter is
  `A · psf.peak_factor(σ)` — 0.1044·A at σ = 1.2 — so an observed
  peak-minus-background must be multiplied by ~9.6 to become an `A` guess.
  `peak_factor` is `erf(0.5/(σ√2))²`, so this ratio moves with σ and a
  hard-coded 9.6 is a bug waiting for a different objective.
- **It is not an aperture sum.** It is the *analytic* integral of the fitted
  Gaussian, wings included, not a sum of pixel values in a box. For an isolated
  emitter the two nearly agree — a `BBOX_PAD = 3σ` box holds 99.96% of it —
  but unlike an aperture sum it excludes the neighbours, because they are
  separate `A_k` in the same fit rather than counts inside the same box.
- **It is per-emitter and background-free.** `b` is its own parameter; nothing
  of the background surface is inside `A`.

Because `A` is in photoelectrons it is only comparable across frames at the
same gain and exposure. Every threshold in section 10b is therefore expressed
as a **ratio to the frame's own median detection**, which is unitless and
transfers.

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

The response is divided by `log_kernel_l2(sigma)`, the L2 norm of the LoG
kernel, which is the null standard deviation of a linear filter applied to a
unit-variance field. **That makes the threshold a count of standard
deviations rather than a raw filter output.** Without it the same numeric
threshold is a 1.8σ cut at σ=0.8 and a 100σ cut at σ=3.0, because ‖w‖₂ scales
as σ⁻³ — the detector silently changed with the PSF width.

**FIND is saturated only above peak SNR ≈ 3.** That claim used to be stated
here without qualification, and it was measured on the only amplitude arm
`scripts/bench.py` then had (900–1900 e⁻, peak SNR ≈ 12). It does not survive below
it. Measured, 225 emitters on 64², sweeping the seed threshold:

| peak SNR | 6.50σ (default) | 3.5σ | 1.7σ | 0.65σ | FP |
|---|---|---|---|---|---|
| 3.0 | 96.0% | 94.7% | 95.1% | 96.4% | 1 → 3 |
| 1.8 | 64.4% | 78.2% | 82.7% | **85.8%** | 1 → 10 |
| 1.1 | 23.1% | 42.7% | 53.8% | **65.8%** | 1 → 10 |

The default is a ~6.5σ cut on pixels and ~10σ on peak heights, and at SNR 1.1
it costs **43 points of recall** — the emitters are present in the LoG map and
the threshold discards them. Five times the candidates buys that recall for
nine extra false positives, so the Bayes factor is absorbing the extra
proposals rather than rubber-stamping them. Run `bench.py --amps bright dim`
before believing any claim that a stage here "is not a bottleneck": every such
claim is a claim about an SNR regime.

Missed emitters have two distinct causes and only one is a threshold. Sub-2σ
companions leave no peak *by construction* — no linear statistic, LoG or
matched filter or score map alike, resolves two sources inside ~1.5σ into two
maxima — and those belong to SPLIT. At 3.7 emitters/PSF, FIND's coverage falls
to 65% while recall stays 100%, because SPLIT recovers them.

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

## 10b. Over-bright and over-wide detections ("aggregates")

"Aggregate" is one word for two situations that need **opposite** handling, and
on some data for a third thing that is not aggregation at all. What a single
frame can actually tell you about a detection is only this:

| the detection is | σ_fit / σ_PSF | flux / median | model can represent it? | handled |
|---|---|---|---|---|
| an ordinary point source | ≈ 1 | ≈ 1 | yes | nothing to do |
| **over-bright** | ≈ 1 | ≫ 1 | yes, as one bright emitter | **post hoc**, `flag_aggregates` |
| **over-wide** (extended) | ≫ 1 | ≫ 1 | no — it gets tiled | **pre-search**, `find_aggregates` |

Only the third row is a different *kind* of object. The second is an ordinary
PSF carrying anomalous flux, and **the image does not say why**. The cheap
diagnostic that tells the rows apart is a free-σ fit at the bright sites:
`σ_fit/σ_PSF` near 1 is row two, much greater than 1 is row three.

### The diffraction limit is why width cannot be the discriminator

**Below ~200 nm, σ carries no information about object size.** A single GFP is
~3 nm; an aggregate of thousands of them may still be ~200 nm; two beads 100 nm
apart are a 200 nm object. All three image at exactly the PSF σ. The size
information is not attenuated, it is *absent*. What differs between them is how
many fluorophores sit in the spot — and that is **flux** (section 2).

### Over-bright: two causes, and the size of the ratio is the only evidence

An over-bright detection has at least two physically distinct causes that
produce the *same* picture — a PSF-width spot carrying n times the usual flux:

- **an unresolved multiple** — n separate ordinary sources within ~1σ, fitted
  as one. The flux ratio is ≈ n, a **small integer**, whenever the population is
  near-monodisperse (beads, or one species of fluorophore).
- **a sub-diffraction aggregate** — one physical object below the diffraction
  limit holding many fluorophores. The flux ratio runs to the **tens or
  hundreds**.

There is no test in this pipeline that separates them, and there cannot be a
per-detection one: a pair closer than 1σ is not identifiable. Section 15's first
open problem is this same phenomenon seen from the position side — the survivor
of an unresolved pair sits between two true emitters, reports a confident SE,
and has absorbed both fluxes. Reading the ratio as a count of sources is only
legitimate when the sources are known to be near-identical.

### Which is why the two datasets here read differently

| frame | N | median flux | max/median | detections > 20× median |
|---|---|---|---|---|
| `beads_60x_still.tif` | 74 | 926 e⁻ | **1.34** | 0 |
| `beads_60x_still_02.tif` | 128 | 951 e⁻ | **1.58** | 0 |
| `hyp7gem_wt_crop.tif` | 685 | 381 e⁻ | **176** | 11 |

**The bead frames contain no aggregates — and no unresolved multiples either.**
Their flux distribution is unimodal and tight: the brightest detection on either
frame is 1.3–1.6× the median, nowhere near the ~2× that one extra coincident
bead would produce, and nothing at all sits above 2×. So on this data a bright
diffraction-limited spot is one bead, and if a doublet did occur it would
announce itself as a **~2× detection, not as a wide one**. `flag_aggregates` at
its 20× default correctly flags nothing on either frame, and anything it *did*
flag there should be read as an artifact rather than as an aggregate.

On `hyp7gem_wt_crop.tif` the separation is ~100–176× — no coincidence of
ordinary sources explains that — so those are genuine sub-diffraction
aggregates.

### Over-bright: `flag_aggregates` (post hoc)

Measured on `hyp7gem_wt_crop.tif` (σ = 1.45): the visible aggregates fit
σ 1.47–1.65, **1.01–1.14× the PSF width** — no width signal at all — while
their detected cores run 33000–63000 e⁻ against a median detection of 378, a
separation of ~100–170× in flux. They look wide on screen only because the
display saturates.

Because a PSF-width object *is* representable, the search fits it correctly
and it stays identifiable afterwards. Nothing needs excluding beforehand:

```python
res = spotsolve.detect(img, sigma=1.45, offset=100.0, gain=2.401)
rep = spotsolve.aggregate_report(res)          # ratio=20 × the frame median
# 8 objects from 11 flagged detections, 47.1% of all detected flux
mask, objs = spotsolve.flag_aggregates(res)    # per-detection mask + objects
```

Flagged detections are linked (within 3σ) so the count is *objects*, not
detections — one aggregate raises a bright core plus a neighbour or two. The
cut is a multiple of the frame's own median, so it is unitless and transfers.
`flux_fraction` is the number worth putting in a QC table: it was stable at
43–57% across cuts from 10× to 80× on this frame, so it does not hinge on the
threshold. Note the default cut of 20× is set to catch aggregates, not
unresolved multiples; a doublet at ~2× is far inside the ordinary flux spread
of most frames and will not be found this way.

**Do not use `reject_aggregates=True` here.** Freezing a σ≈1.5 object into
`bmap` double-counts against emitters fitted beside it: measured, the residual
audit went from z ∈ [−7.0, 10.3] to [−26.6, 10.3] and N rose 689 → 715.

### Over-wide: `find_aggregates` (pre-search)

Only when the object is wider than the PSF can represent. The model has a
fixed σ, so it tiles it — measured on three synthetic aggregates (flux
20–60k e⁻, σ 2.4–4.0): **84 detections**, flux recovered accurately (32634 e⁻
against a true 30000) and spread over ~25 emitters each.

Nothing on the final table can undo that. Amplitude points the *wrong way*
(tiles 1249 e⁻ against 1370 for ordinary emitters; 69 of 84 dimmer than the
brightest genuine emitter) and a post-hoc σ refit gives 1.22 against 1.20,
because locally a tile *is* a PSF-sized bump with its neighbours frozen into
its halo. There is also damage no downstream filter reaches: three aggregates
inflated `lam`, the density prior inside every Bayes factor, by **2.3×**
(0.00662 → 0.01552).

The test is a free-σ fit at each round-0 candidate, accepted on

    flux_fit > 10 × median(candidate flux)   and   0.9σ < σ_fit < 5σ

**Flux is the discriminator; σ is only a plausibility band** — the same
lesson as above. An earlier version required `σ_fit > 1.5σ` and found none of
the real aggregates, while flagging a dim diffuse region where the fit had run
to a bound (79 of 246 fits hit a bound on that frame). What is found is
**frozen into `bmap`**, not merely masked: masking candidates alone left 68 of
120 detections inside the mask carrying 112742 e⁻, because the search kept
modelling the aggregate one PSF at a time from the boundary inward.

**Known failure — a dense cluster reads as one wide object.** This is the same
ambiguity as above at its worst, and it is why the pass is off by default. On
`beads_60x_still_02.tif` it reports 2 aggregates and masks 24.8% of the frame,
and the masked region holds 37 real beads (median 978 e⁻ against 939 outside) —
a frame that, by the table above, contains no over-bright detection at all. One
wide Gaussian fits "many close point sources" as well as "one wide object", and
the flux test cannot separate them either, because 37 beads at ~950 e⁻ carry an
aggregate's total flux. The missing discriminator: a genuinely extended object
leaves a *smooth* residual, a cluster of point sources leaves PSF-scale peaks
that `audit.score_map` already finds.

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

**They fire.** That argument is sound about *monotonicity* and wrong about
*termination*, because the second bullet assumes the candidate list empties.
Measured on 69 of 69 frames of both movies, `n_rounds == max_rounds == 6`
every time (`docs/baseline/truncation.txt`):

| movie | N at max_rounds 6 → 12 → 20 | last round's added+split |
|---|---|---|
| `beads_80pct-glycerol` | 1167 → 1213 → **1235** | 0+26 → 0+4 → 0+0 |
| `hyp7gem_wt_04` | 4269 → 4517 → **4738** | 0+100 → 0+37 → **0+31** |

`hyp7gem` has not converged at 20 rounds. In every late round **adds are zero
and splits are everything**: the loop is not discovering emitters, it is
subdividing objects the fixed-σ model cannot represent, one PSF at a time. An
object wider than σ leaves a residual no single PSF can flatten, so the list
refills as fast as it empties. Section 12's width arm reproduces this in
synthetic data with ground truth.

## 12. The acceptance tests

Four, in increasing order of authority.

**Recall by isolation bin.** A frame-level recall is dominated by the easy
majority and is blind to the only regime in question. `scripts/bench.py` breaks it out
by each *true* emitter's distance to its nearest neighbour.

**Recall and tiling by width bin** (`bench.py --widths`). Every arm rendered
every emitter at the model's own σ until 2026-09-02, so no measurement in
`docs/baseline/bench.txt` contained a single mis-widthed source — and width
mismatch is the one thing the real movies show. It is also the most damaging
axis in the benchmark by a wide margin. At the *easiest* arm (bright, density
0.015, flat), raising the per-emitter σ spread alone:

```
spread   Nest   recall     FP   med err   rsd z   |z|>3
  0.00   22.5    0.978   0.00    0.0613    1.03    3.7%
  0.20   31.8    0.971   9.50    0.1672    2.38   26.5%
  0.40   40.0    0.935  18.50    0.2871    3.10   38.0%
```

For scale: sweeping density 0.015 → 0.055 at spread 0 takes FP from 0.00 to
1.00, and sweeping SNR from `bright` to `dim` leaves it under 1. **Recall is
nearly blind to it** — it moves 4 points while FP goes from zero to eighteen.
The damage is one-sided and monotone in each emitter's own width: below 1.05×
nothing tiles, at 1.25–1.6× about 90% of emitters collect a second detection.
Read `tiles/det`, not `dets/em` or `FP`. The other two cannot tell "the move
merged the tiles" from "the seeder never found the object", and on the first
run of this arm that distinction decided the answer: `spotsolve-roi` appears to
halve FP under width mismatch (18.50 → 10.83 at bright/0.015, spread 0.40), but
with recall divided out its genuine anti-tiling effect is **0–21%** and negative
in two cells, because its matched-filter seeder finds only 0.67–0.79 of the wide
emitters against the round loop's 0.92–0.96. The same seeder loses 34–40 points
of recall on emitters *narrower* than σ, which defocus cannot produce and which
never tile in either method — the attenuation is two-sided even though the
failure it is aimed at is not.

**The pull statistic** (`scripts/crlb.py`). `z = (estimate − truth) / reported SE`, per
axis. An estimator at the CRLB with an honest Fisher matrix gives spread 1.00
and 0.3% beyond `|z| = 3`. The tool also runs an **oracle** arm — `refine` from
the true positions at the true N with the true background — so the gap between
arms is exactly what the search costs, and a deficit can be attributed to the
estimator or to the search rather than guessed at.

Both arms report one row per **true emitter**, and `match%` says what share of
the bin each pull column rests on. Until 2026-09-02 the pipeline arm reported
one row per *matched detection*, so the `<1s` rows compared 118 unconditioned
truths against the 63 the search had already resolved. That is why the
pipeline's `med err` read *better* than the oracle's (0.286 against 0.329) while
being 4.8× overconfident, and it biased against every change that helps:
resolving more close pairs admits the harder ones. The `d_nn` column — distance
from each true emitter to the nearest estimate, over all of them — has the same
denominator in both arms and inverts that verdict: 0.366 against the oracle's
0.290.

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
spotsolve      72.8  0.860  0.50  0.523 0.958  1.000 1.000  1.43  10.7%  +2.3%  1.06
```

For reference, the superseded `boxsolve` on the same benchmark reached
`N = 69.2`, recall 0.817, `rsd z` 1.61 and 2.05 s/frame — worse on every column
and twice the time.

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

**Box tiling, core seams, `_adopt_orphans`.** All of the old `boxsolve`'s
machinery for stabilizing N. It existed to make a non-converging global sweep
converge; a forward-only search converges by construction (section 11) and does
not need it. Removed with the rest of `boxsolve`; see git history before the
cleanup commit if the reasoning is ever needed again.

**The projected score as a screen.** Correct as a *statistic* and worth
revisiting for SPLIT ranking (section 7), but it must not be used to seed and
must not be used to screen: the raw conditional score does not screen at all,
and was exceeded by the post-fit `A/SE` in 79% of cases. Its implementation
(`score.py`) was removed with `boxsolve`; PORTING_NOTES section 16 keeps the two
traps that make it hard to reimplement.

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

**µs/px is flat.** `spotsolve` is linear in area at fixed density. The old
`boxsolve` was not — its halo rendering was `O(area²)`.

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
widening its SE honestly, is the next real problem. Seen from the flux side it
is an over-bright detection at ratio ≈ 2 (section 10b), which is why that
section's 20× cut does not find it.

**Sub-σ recall is ~0.52 and neither solver does better.** ~~At the
identifiability limit. Not obviously fixable.~~ Wrong: the Fisher information
is not the limit. CRLB on pair separation at bead flux gives `d/SE(d)` of
2.7–4.2 at 0.5σ (`docs/baseline/crlb.txt`). The block is three constants in
series — the amplitude prior, `COND_GUARD` and `PRUNE_TAU` — and relaxing any
one alone changes final N by nothing. Measure final N, never stage acceptance.

**The Laplace volume term omits the prior's curvature.**
`evidence._log_bf_add_from_logdet` builds the Occam factor from `log|F|` with
`F = Jᵀ W J`, the *likelihood* Fisher, where the Laplace evidence wants the
negative Hessian of the log *posterior*, `log|F + Λ|`. Under the old
`Exp(1/A_s)` prior `log g(A)` is linear in `A`, so `Λ = 0` exactly and the
omission was correct. The NPMLE `MixturePrior` is curved and it no longer is:
including `Λ` moves `log BF` for a 0.5σ split by −4.0 nats under a
`U(900,1900)` population and −5.8 under a tight one, ~0 under a broad one. A
degenerate configuration currently collects an Occam *bonus* from its own
ill-conditioning, and `Λ` is the term that regularizes it — which makes
`COND_GUARD` a hard cutoff standing in for a term that belongs in the formula.
Fixing it makes sub-σ splits **harder**, not easier. `verify_evidence`'s 4-D
quadrature check cannot see this: it passes a scalar `A_s`, which
`_as_prior` turns back into `ExponentialFlux`.

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
| `LOG_SEED_Z` | 6.50σ | FIND's seed cut, **in σ of the LoG null** — see §5 |
| `AGG_AMP_RATIO` | 20.0 | post-hoc **over-bright** cut, vs the median detection |
| `AGG_FLUX_RATIO` | 10.0 | pre-search **over-wide** cut, vs the median candidate |
| `AGG_SIGMA_LO/HI` | 0.9 / 5.0 | pre-search σ plausibility band (not a cut) |
| `AGG_MASK_RADIUS` | 3.0 | σ_fit of excluded support per over-wide object |
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

`LOG_SEED_Z` is the other one to move, and it was invisible until 2026-08-28
because it was expressed in the LoG's raw output units. Its value is exactly
the historical `1.5 / ‖w‖₂(σ=1.2)`, chosen so the normalization reproduced the
old behaviour bit-for-bit — it is a compatibility value, not a defensible
operating point.

`lam` (emitter density) and `A_s` (amplitude prior scale) are **not** knobs —
they are re-estimated empirically each round.

---

## Reading order for someone new to the code

`structs.py` (the contracts) → `psf.py` → `lmga.py` → `evidence.py` →
`core.py` → `scripts/crlb.py` and `audit.py` (how it is judged).

---

## Repository map

```
src/spotsolve/     the library — `import spotsolve`
scripts/           command-line entry points; each has a `--help`
tests/             the layer checks, and tests/fixtures/ the golden values
rust/              spotsolve-core (the algorithm) + spotsolve-py (PyO3 wrapper)
docs/              PORTING_NOTES.md, and the Aguet paper writeup
data/              two bead crops the README's numbers come from
```

Every module, in dependency order. Nothing here is optional to the layer above
it; paths below are relative to `src/spotsolve/` unless shown otherwise.

**The algorithm** — this is the port target, and it is all of it:

| file | what it is |
|---|---|
| `structs.py` | the `theta` layout and the result records; imports nothing |
| `psf.py` | pixel-integrated Gaussian, model and Jacobian |
| `lmga.py` | bounded Poisson-MLE optimizer (Coleman–Li affine scaling) |
| `evidence.py` | Laplace log Bayes factor, `logdet` and the conditioning guard |
| `moves.py` | the two proposals: residual quadrupole axis, and the split |
| `patches.py` | grouping emitters into jointly-fittable patches, and the halo |
| `calibrate.py` | gain, robust background, model rendering |
| `core.py` | `detect` and `refine` — FIND → ADD → SPLIT → REFINE → PRUNE |

**Judging it** — the three acceptance tests of section 12, and their inputs:

| file | what it answers |
|---|---|
| `simulate.py` | synthetic Poisson fields with known truth |
| `metrics.py` | matching detections to truth |
| `audit.py` | is anything PSF-shaped left in the residual? |
| `../../scripts/bench.py` | count, isolation-resolved recall, false positives, runtime |
| `../../scripts/crlb.py` | the pull statistic, against an oracle arm that fixes N and truth |
| `../../scripts/run.py` | one real frame in, detections and audit panel out |

**Checking the layers** — `tests/verify_*.py` test one layer each against
analytic truth or numerical integration; each exits non-zero on a failure.
`pytest` runs all four; run one on its own while working on that layer:

```bash
pytest                          # ~5 s
pytest --runslow                # + verify_evidence's brute-force 4-D
                                #   quadrature of the posterior, ~2 min
python tests/verify_psf.py      # layer 1 alone, PASS/FAIL per check
```

They are ordered `verify_psf` → `verify_optimizer` → `verify_geometry` →
`verify_evidence`; a failure in an early one invalidates the later ones.

**For the port** — `python scripts/make_fixtures.py` writes
`tests/fixtures/*.json`, the golden values a second implementation asserts
against, layer by layer. Those files are committed, so `cargo test` needs no
Python. See [`docs/PORTING_NOTES.md`](docs/PORTING_NOTES.md) section 16.

### Using the Rust core

`rust/` holds `spotsolve-core` (the algorithm) and `spotsolve-py` (the PyO3 wrapper,
imported as `spotsolve_rs`). `backend.py` dispatches **four passes and two image
helpers** to it — ADD, SPLIT, REFINE, PRUNE, `render_model`,
`emitter_free_mask`. The round loop, `find_candidates` and `background_map`
stay in Python either way, which is what makes the two backends a controlled
comparison rather than two different programs.

```bash
maturin develop --release -m rust/spotsolve-py/Cargo.toml   # build + install
python -c "from spotsolve import backend; print(backend.available())"
```

Then select it per call, with everything else unchanged:

```python
res = spotsolve.detect(img, sigma=1.45, offset=100.0, gain=2.401, impl="rs")
```

```bash
python scripts/run.py --image <frame.tif> --sigma 1.45 --gain 2.401 --impl rs
python scripts/bench.py --methods spotsolve spotsolve-rs        # the two side by side
```

Constructing `RustBackend` asserts that `PRUNE_TAU`, `REFINE_TOL`,
`REFINE_TOL_OBJ`, `EVIDENCE_TOL_OBJ`, `REFINE_MAX_ITER` and `COND_GUARD` agree
across the boundary, and raises telling you to rebuild if they have drifted —
a silent difference in `PRUNE_TAU` would read as an algorithmic disagreement
rather than a stale build.

Measured on `hyp7gem_wt_crop.tif` (172², σ=1.45, gain 2.401): **17.1 s → 1.9 s**,
with `N = 685` and an identical residual audit from both backends. When they
*do* disagree by an emitter, read `PORTING_NOTES.md` section 19 before calling
it a bug — one-emitter differences land on the frame's most marginal decision,
and the way to show that is to sort the weighed decisions by `|log BF|`.
