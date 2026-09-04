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
8b. [The free width — the model space and the reporting band](#8b-the-free-width--the-model-space-and-the-reporting-band)
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

Every fit above carries **one width per emitter**, bounded to `SIGMA_SLACK`
= 0.70–2.2× the PSF sigma and fitted by MAP under `prior.FocusMixtureWidth`;
objects that land outside `FOCUS_BAND` = 0.80–2.0× are modelled to the end but
reported separately rather than as detections. Those are changes to the model
space, to the prior, and to what the fit maximizes — not to any decision rule.
Section 8b is why each was necessary and section 12 is what they bought.

Five moving parts. The layers underneath are shared:

| module | role |
|---|---|
| `psf` | pixel-integrated Gaussian, model and Jacobian |
| `lmga` | bounded Poisson MLE/MAP optimizer (Coleman–Li affine scaling) |
| `evidence` | Laplace log Bayes factor, `COND_GUARD` |
| `patches` | grouping of emitters into jointly-fittable patches |
| `moves` | proposal constructors, pure `theta -> theta` |
| `prior` | flux and width priors, estimated from the frame |
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

### What SPLIT actually does, measured 2026-09-03

Not what the paragraphs above say. Disabling it and reading in-focus recall by
nearest-neighbour distance (moderate, 6 frames, derived seed cut):

| nn distance | with SPLIT | without | Δ |
|---|---|---|---|
| <1σ | 68.2% | 68.2% | **0.0** |
| 1–2σ | 82.0% | 78.7% | +3.3 |
| 2–3σ | 96.0% | 89.3% | **+6.7** |
| >3σ | 99.0% | 99.0% | 0.0 |
| all | **93.4%** | 91.4% | +2.0 |
| tiles/frame | 3.50 | 2.17 | +1.33 |

**It contributes nothing below 1σ — the regime it was written for.** What it
actually recovers is 1–3σ pairs, and the reason is visible in the seeding
ceiling below: that is exactly where the LoG has merged two maxima into one, so
FIND cannot propose the second emitter at ANY threshold, while the residual
quadrupole is still large enough to point at it. SPLIT is functioning as a
**second-chance seeder for pairs the linear filter merged**, not as the sub-σ
resolver §7 describes.

It is also inefficient at it. Per frame it gets ~27.6 proposals accepted and
PRUNE then removes ~20 of them, for a net of 7.5 emitters and +2.0 points of
recall. A mechanism that proposes blindly along every emitter's residual
quadrupole and lets removal sort it out is not the cheapest way to reach merged
maxima; a seeder that recognises a merged maximum and proposes there directly
would be. That is the open lead on §15's "split/birth is the hard part".

### The LoG discards two thirds of the shape information — a design note

Not implemented; measured 2026-09-03 and recorded because it is cheap, it is
the natural answer to "what should this candidate become", and it settles an
architectural question the other way from where the reasoning started.

**The LoG is the TRACE of the Hessian.** `find_candidates` builds
`d_yy + d_xx` of the Gaussian-smoothed frame and keeps the sum. The other two
independent components of a symmetric 2-tensor — `(d_yy − d_xx)/2` and `d_yx` —
are discarded, and they are exactly the ones that carry SHAPE:

| component | detects |
|---|---|
| monopole (`G`) | flux excess → birth |
| **trace** (`∇²G`, the LoG) | strength, and across scales, WIDTH |
| **traceless** | ELONGATION, and its eigenvector is the axis |

Why that matters here: an equal pair at separation `d` has second moment
`σ² + d²/4`, which is **identical** to a single emitter widened to `σ_eff`. The
two are indistinguishable in the trace channel *by construction*. Verified by
score test on a fitted residual: with `σ` free, `z_trace` is **0.00 for both**
at every separation — the isotropic width absorbs the entire trace signature.
That is not a search-order accident; it is why §15's "a widened emitter hides
its own neighbour" happens at all, and no prior on an isotropic width can fix
it, because the information is in a channel the model does not have.

**Measured, from filtering alone, no fit.** `|traceless| / |trace|` at the peak,
pair against the matched-width single, 300 noise seeds:

| d/σ | pair (2000 e⁻) | wide (2000 e⁻) | separated? |
|---|---|---|---|
| 0.5 | 0.015 | 0.012 | no |
| 1.0 | 0.034 | 0.016 | marginal |
| 1.5 | **0.080** [0.061–0.099] | 0.019 [0.008–0.038] | **yes, disjoint** |
| 2.0 | 0.160 | 0.025 | yes |
| 2.5 | 0.308 | 0.034 | yes |

At 500 e⁻ it separates only from d ≈ 2.5σ. Cost on a 512² frame: **2.0 ms**
against a fit budget of 7600 ms — 0.03%.

**What this would buy.** A candidate could be triaged before any fit: trace for
strength, trace across scales for the object's own width (in-focus vs wide,
Lindeberg scale selection), traceless for pair-ness and its axis. SPLIT's two
guessed displacements (`SPLIT_DISPS`) and `moves.residual_axis_var`'s
second-moment proxy would both be replaced by one measured axis and separation.

**What it would not buy, and this is the part that matters.** Nothing sees
d < 1σ — not this, not the LoG, not a score test on the residual. Sub-σ pairs
are invisible at the DETECTION level whatever statistic is used, which is
consistent with §15 and with SPLIT's measured contribution of exactly 0.0 in
that bin. And a dim companion produces a faint quadrupole regardless, because
the amplitude scales with the pair's REDUCED flux `A₁A₂/(A₁+A₂)`, not the total.

**It also argues AGAINST elliptical fitting.** The obvious response to the
trace/traceless confusion is to give each emitter two more parameters so the
model can hold the anisotropy. This measures the same thing by convolution, for
2 ms a frame and no parameters at all. Prefer the filter.

**One trap if it is implemented.** The raw ratio is not a threshold-able
quantity: its null level rises with noise (0.012 at 2000 e⁻, 0.035 at 500 e⁻),
so a fixed cut does not transfer across brightness. Express it as a z against
its own propagated null, exactly as `calibrate.seed_threshold` does for the
trace channel. That mistake has already been made once in this codebase, in
raw filter units, and cost a detector that silently differed per instrument.

### The seeding ceiling — why FIND alone cannot be the answer

The LoG is the right filter. It is a Gaussian matched filter with the
low-frequency background suppressed, the `1/sqrt(model)` normalization is the
Poisson whitening, and **a peak does remain a peak**. What it cannot do is keep
two peaks apart once they merge.

Every bona-fide local maximum on the raw frame — threshold dropped to 1.0 sd,
matched ONE-TO-ONE so a merged pair claims only one truth:

| nn distance | n | got its OWN LoG maximum |
|---|---|---|
| <1σ | 22 | 59% |
| 1–1.5σ | 35 | 60% |
| 1.5–2σ | 26 | 69% |
| 2–3σ | 75 | 67% |
| >3σ | 191 | **97%** |
| all | 349 | **83%** |

**83% is the ceiling of any seed-only architecture on this arm**, against the
pipeline's 93.4%. Isolated emitters are at 97% and no threshold change improves
them; the crowded bins sit at 59–69% and no threshold change improves them
either, because the maxima are not there to be found. That is the whole
justification for iterating FIND on the RESIDUAL and for SPLIT, and it is why
"seed generously and let PRUNE sort it out" cannot work: the seeds for the
crowded 17% cannot be generated by thresholding a linear filter.

There is a second reason, from §9: over-seeding drives configurations toward
degeneracy, and a degenerate pair's Fisher matrix goes singular, at which point
the Occam term becomes a *bonus* and the removal test has to be forced rather
than weighed. Over-seeding moves the decision into precisely the regime where
the evidence is least trustworthy.

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

## 8b. The free width — the model space and the reporting band

Every emitter carries its own sigma. Two constants govern it and they are not
the same thing:

| | | |
|---|---|---|
| `SIGMA_SLACK` | `(0.70, 2.2)` | the **model space** — widths a fit may represent |
| `FOCUS_BAND` | `(0.80, 2.0)` | the **reporting band** — widths that count as a detection |

Until 2026-09-03 these were ONE constant, `(0.95, 2.0)`, and that conflation is
the whole of the defect this section used to describe.

**A fixed-width model answers a source it cannot represent by TILING it.** Two
narrow Gaussians genuinely do fit a broad blob better than one, so the residual
has lobes, SPLIT proposes into them, and the Bayes factor accepts — correctly.
Every stage does the right thing and the answer is wrong, because the model
space does not contain the object. No threshold repairs that; section 11's
non-termination is the same fact seen from the loop.

The first fix, a free width bounded to `(0.95, 2.0)`, was right about the cause
and wrong about the remedy, because **it clipped the model space to the
reporting band**. The model space has to cover every photon on the sensor or
the light it cannot represent is tiled. The reporting band is a downstream
contract about which fitted objects the caller is handed. They are different
questions and they need different numbers.

Measured on `data/sim_out`, a `psfkit` vectorial spinning-disk confocal
simulation with emitters uniform in ±0.5 µm of focus, where a point source
images at

| \|z\| µm | 0.00 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 |
|---|---|---|---|---|---|---|
| sigma px | 0.82 | 0.84 | 0.92 | 1.26 | 1.96 | 2.64 |
| flux | 1.00 | 0.91 | 0.71 | 0.48 | 0.31 | 0.20 |

— a 3.2× width spread that is not a knob but what defocus does.

### Separating them is not free, and the naive separation is worse

Under ONE uniform width prior over the enlarged space, every emitter pays for
the enlargement. Widening `[0.95, 2.0]` to `[0.95, 8.0]` taxes every in-focus
add `log(7.05/1.05) = 1.9` nats — the detector gets less sensitive everywhere
in order to accommodate defocus. And it does not buy what it was meant to:
under a flat prior, going from a 2.0 bound to 3.2 charges a WIDE fit only
`log(2.25/1.05) = 0.76` nats *in total*, so a wide emitter swallowing a genuine
close neighbour is charged essentially nothing for the privilege.

**That is what the old sweep was really measuring.** The hard bound at 2.0 was
a cutoff standing in for a prior term that belongs in the formula —
structurally the same defect section 15 records for `COND_GUARD` standing in
for the prior's curvature.

### The term is a mixture — `prior.WidthPrior`

The field is a **superposition of two Poisson processes**: in-focus emitters at
rate `lam_focus` with widths in the band, and defocused nuisance objects at
rate `lam_wide` with widths above it. The likelihood is identical — they are
both Gaussians, fitted by the same code. Only the width prior's support and the
rate differ. So an in-focus add pays exactly what it paid before the model
space was enlarged, and a wide one pays against its own, rarer rate and its own
much broader width prior.

Neither rate is a knob. Both are re-estimated from the frame each round,
exactly as `lam` and `A_s` already are, so "a defocused object is rarer than an
in-focus one" is read off the data rather than asserted.

`evidence._count_width_delta` scores the WHOLE configuration on each side and
differences it, rather than charging the added emitter. That is not tidiness:
under a mixture the incumbents do **not** cancel, because a neighbour that
widens across the class boundary in the joint refit moves between two processes
and changes both their counts. With `prior.UniformWidth` the whole expression
reduces algebraically to `log(lam) - log(K+1) - log(sigma_width)`, which is
what it was, so `band=None` reproduces the single-band pipeline exactly.

### The width is a MAP parameter, not an ML one

The prior above is not only scored — it is **fitted under**. `core._WidthPenalty`
hands the width prior to `lmga` as a penalty, so every free-width fit maximizes
the posterior; `evidence` then adds the prior once more at the configuration
level, which is why `lmga` returns `I` without the penalty and `F` with it.

This was not an aesthetic point about consistency. Measured on `crlb.py`, whose
fields carry **no defocus at all** — every emitter is rendered at the model
sigma, so the free width has nothing to buy and only costs — the cost is not
where you would expect it:

| bin | oracle, σ fixed | oracle, σ free (ML) |
|---|---|---|
| `>3s` isolated, med err | 0.0527 px | 0.0557 px |
| `1-2s`, med err | 0.1198 px | **0.1999 px** |

An ISOLATED emitter loses essentially nothing, because sigma is orthogonal to
position by symmetry — `d model/d sigma` is radially symmetric and
`d model/d y` is antisymmetric, so their inner product is zero. A pair at 1–2
sigma loses 67%, **in the oracle arm**, where N is fixed at truth, the starting
values are truth, and nothing is searched. That is not a search failure and not
an information cost: it is a fit-level degeneracy. Two narrow emitters at 1.5
sigma and one wide emitter plus a faint one describe nearly the same pixels,
and a flat prior gives the optimizer nothing to choose with.

The MAP fit gives it something. On the same fields, at 1–2 sigma:

| | match% | med err | rsd z | d_nn |
|---|---|---|---|---|
| ML widths | 75.3% | 0.2516 | 1.92 | 0.4123 |
| **MAP widths** | **79.8%** | **0.1972** | **1.49** | **0.3516** |

About a third of the gap to the fixed-width arm, recovered by making the fit
and the evidence agree about what a width costs. The rest of that gap is the
honest price of carrying a parameter that buys nothing on a field with no
defocus, and section 15's sub-sigma problem, which this does not touch.

**One half of section 15's `Lambda`.** Returning `F + Lambda` repairs the
recorded omission for the WIDTH block only. The amplitude block is still
`J' W J`: under the default `ExponentialFlux` that is exact, because a linear
`log g(A)` has zero curvature, and under an NPMLE `MixturePrior` it is still
the open problem section 15 describes.

### The band's two edges are different in KIND

`FOCUS_BAND[1]` is a class boundary between two populations that are both real.
It lives in the prior and is decided **during the search**, while the flux is
still undivided. `FOCUS_BAND[0]` is not a class: nothing images narrower than
the PSF, so a fit below it is a fit that has BROKEN, and there is no
information about that which the search destroys. It is a post-fit check on the
survivors, and `SIGMA_SLACK[0] = 0.70` sits below it so a broken fit can reveal
itself instead of being clipped to the bound and reported.

### What it is worth, measured

`scripts/bench_sim.py`, moderate arm (5 emitters/µm²), 6 frames, the reporting
band fixed at `(0.8, 2.0)` and the model bound swept **under the MAP fit**,
i.e. in the pipeline this constant actually lives in:

| model bound | recall | med err | RMSE | rsd z | tiles |
|---|---|---|---|---|---|
| **2.2** | **92.6%** | 0.076 | 0.220 | **1.25** | **2.33** |
| 2.6 | 91.1% | 0.078 | 0.201 | 1.31 | 2.33 |
| 3.2 | 90.8% | 0.078 | **0.183** | 1.27 | 4.00 |
| 4.0 | 90.5% | 0.078 | 0.194 | 1.32 | 2.50 |

2.2 survives, and the prediction that the MAP fit would let the bound rise to
the optics' own 3.2× was **wrong**: recall falls monotonically past 2.2 while
RMSE improves. A larger model space buys better parameters for the objects it
keeps and loses the close neighbours it swallows; the trade does not reverse,
it only gets priced honestly.

The earlier sweep, **taken with ML widths** and kept for its shape — the
tiling column falls all the way to 3.2 while the residual turns over at 2.0,
which is the wide class going unpaid for:

| model bound | in-focus recall | tiles/frame | resid rsd |
|---|---|---|---|
| single band, `(0.95, 2.0)` | 93.7% | 7.83 | 1.203 |
| **2.2** | **88.5%** | **2.00** | **1.177** |
| 2.6 | 87.1% | 0.50 | 1.249 |
| 3.2 | 87.1% | 0.83 | 1.332 |
| 8.0 | 84.1% | 0.25 | 1.578 |

Both sweeps put the default at the same place, for different reasons, which is
the strongest form this evidence takes: what the wide class needs is **ROOM
just above the band, not a large box**.

The physics would prefer 3.2×, where this sample's optics actually stop
(σ(0.5 µm) = 2.64 px). It does not get it, and the re-sweep says the obstacle
is not the one section 15 originally named: it is that a wider model space
lets an emitter swallow a close neighbour faster than the prior can charge it,
and the width prior slows that down without stopping it.

At that default, against the single-band pipeline (`legacy` → `mix`, 6 frames),
**with the derived seed cut of §5 in place** — which moved both arms, and is
why these differ from the numbers recorded before 2026-09-03:

| density | in-focus recall | tiles/frame | med err px | RMSE px | ghosts/frame |
|---|---|---|---|---|---|
| 1/µm² | 96.2% → **97.5%** | 1.83 → **1.17** | 0.073 → **0.070** | 0.158 → **0.150** | 0.33 → 0.33 |
| 5/µm² | **95.1%** → 93.4% | 8.83 → **3.50** | 0.079 → 0.080 | 0.277 → **0.212** | 0.17 → 0.17 |
| 15/µm² | **84.2%** → 83.1% | 14.33 → **9.33** | 0.122 → **0.115** | 0.326 → **0.286** | 0.00 → 0.00 |

**Fixing the seeder narrowed the width prior's case, and that should be said
plainly.** Before §5's derived cut, `mix` was level with or ahead of the single
band on recall at every density. It is not any more: the single band recovers
more from a correct seed threshold than the mixture does (+2.0 against +0.8 at
5/µm²), so it now leads by 1.1–1.7 points at 5 and 15 emitters/µm². What
survives is **35–60% less tiling and 13–24% better RMSE at every density**, and
an outright win at 1/µm².

So the mixture's remaining case is multiplicity and localization, not recall.
Read it against what your downstream costs more: a missed emitter, or a
duplicated one. And note the general lesson, which is §5's: a defect upstream
of a comparison can make the comparison say the wrong thing. The seeder was
starving ADD in both arms, and the arm that leaned on SPLIT to compensate
looked better than it was.

Ghosts appear at 0.33 and 0.17 per frame at 1 and 5 emitters/µm², in **both**
arms; at 15 they stay at 0.00, because a spurious seed on a crowded frame is
almost always near something real. They are the seed cut's doing, not the width
prior's — the old cut bought immunity to invented detections by refusing to
look. See `calibrate.SEED_ALPHA`.

`FOCUS_WIDTH_GAMMA` was swept too, and the useful result is that it barely
matters: anything in 0.10–0.30 lands within a point of recall and a few
thousandths of a pixel, and only 0.50 is clearly too loose. 0.20 is kept for
the best pull spread and the fewest tiles, but the honest statement is that
the prior's SHAPE is what does the work and its scale does not — the same
thing `prior.py`'s header found for the flux prior's bandwidth. Do not tune it
against a score.

### The flux/width coupling — built, measured, removed

The one thing the width prior does not price is that **a defocused source is
wide AND DIM**, and the two are not independent: both are functions of the same
|z|. A fitted `(A, sigma)` implies a nominal brightness `A / f(sigma)`, so

    g_defocus(A | sigma) = g(A / f(sigma)) / f(sigma)

with `f` the axial response tabulated above. An object fitted at 3× the PSF
width carrying 4000 e⁻ is asserting a 20000 e⁻ emitter, which under a
population whose median is ~1000 costs about 16 nats — the correct verdict,
because a wide and *bright* thing is off the defocus locus entirely. It is a
cluster of point sources, and the narrow hypothesis should win. At the in-focus
width `f = 1` and the term vanishes exactly.

It was worth +0.5 points of recall at a model bound of 8.0, and **nothing at
2.2**: recall 95.0 → 95.0, 88.5 → 88.8, 76.8 → 76.1 at 1, 5 and 15
emitters/µm², residual moving by ±0.04 in both directions. The reading that
fits both facts is that **the coupling and a tight `SIGMA_SLACK[1]` are two
ways of pricing the same thing** — once the box is small enough there is no
wide-and-bright fit left to refuse. That made it the term that would let the
bound rise to the optics' own 3.2×, and the re-sweep above says the bound does
not want to rise. Removed with the argument that justified it (§13). The
physics is right; there is currently no operating point at which it pays.

### Three places had to learn about widths, and then two more

`_window`'s link and halo radii scale with the width of the emitter they are
measured from; `moves.residual_axis_var` weights the second moment at the
emitter's own width; and `background_map` masks each emitter's support at its
own width. Two proximity vetoes were still at the fixed PSF width and are now
`core.veto_radius`: `find_candidates`'s and `_add_pass`'s.

That radius is **capped at the reporting band's upper edge**, which is not a
detail. The veto exists to stop one object being proposed twice, and that is a
question about RESOLVABILITY; past the band the object is not a point source
and its centre has no claim on a peak a linear detector still resolves on top
of it. Letting the radius run to the model's full width cost 8.3 points of
in-focus recall at 5 emitters/µm². On its own the fix is worth +0.6 points of
recall and −0.2 tiles at that density — a correctness repair whose measured
effect is small, not a lever.

### The fitted width is a result, not a diagnostic

`result.fit_sigma` and `result.sigma_ratio` are the widths that produced the
reported positions, amplitudes and CRLBs. Objects outside the band are returned
in `result.aggregates` (wide) and `result.width_rejects` (both edges), never
silently dropped — and they stay in `model_image` and `residual`, because they
are what produced them. A post-hoc filter used to ask the same question of
the finished detections; section 10b measured that by then the answer is gone,
and section 13 records its removal.

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
| **over-wide** (extended) | ≫ 1 | ≫ 1 | yes, up to `SIGMA_SLACK[1]` | **in-search**, the wide class (§8b) |

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

Freezing such an object into `bmap` before the search double-counts against
emitters fitted beside it: measured, the residual audit went from
z ∈ [−7.0, 10.3] to [−26.6, 10.3] and N rose 689 → 715. That was one of the
reasons the pre-search route was removed (§13).

### Over-wide: what a fixed-σ model does to a genuinely extended object

Kept because the measurement is the argument for the wide class, not because
the pre-search route that came out of it survives (§13).

The model has a fixed σ, so it tiles the object — measured on three synthetic
aggregates (flux 20–60k e⁻, σ 2.4–4.0): **84 detections**, flux recovered
accurately (32634 e⁻ against a true 30000) and spread over ~25 emitters each.

Nothing on the final table can undo that. Amplitude points the *wrong way*
(tiles 1249 e⁻ against 1370 for ordinary emitters; 69 of 84 dimmer than the
brightest genuine emitter) and a post-hoc σ refit gives 1.22 against 1.20,
because locally a tile *is* a PSF-sized bump with its neighbours frozen into
its halo. There is also damage no downstream filter reaches: three aggregates
inflated `lam`, the density prior inside every Bayes factor, by **2.3×**
(0.00662 → 0.01552).

That is the case §8b's wide class exists to answer, and it answers it inside
the search rather than by excluding regions beforehand. The one regime it
still cannot represent is an object wider than `SIGMA_SLACK[1]`: a genuinely
extended structure at 3–5σ is outside the model space by construction, and
will tile. On `data/sim_out`, whose widest source is 3.2σ, that regime is
empty — the pre-search pass masked 0–0.8% of the frame and made recall
slightly *worse* where it fired. On frames that really do contain extended
structure it is untested, and the honest answer is to look at the residual.

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

**This is what the free width (§8b) is for**, and it addresses the cause
rather than the symptom: an emitter that can widen absorbs the residual that
was refilling the list, so the second bullet's assumption holds again. The
measurements in this section predate it and were taken at `slack=None`;
re-measure both movies before quoting them.

**The mixture prior does not fix this either, and that is measured.** At 1
emitter/µm² the loop now ends on its own in 3-4 rounds on 4 of 6 frames, with
the last round accepting nothing. At 5/µm² it still hits `max_rounds = 6` on
every frame, and the last round is still `0 added, 1-8 split` — adds are zero
and splits are everything, exactly the signature above. The tiling that
survives to the end IS down, 8.0 → 1.7 per frame, so the splits it is still
making are largely being undone by PRUNE rather than surviving; but the loop is
not converging, and the second bullet's assumption still does not hold at
density.

What the free width bought here is that the objects being subdivided are no
longer the DEFOCUSED ones — at 1/µm² the tiling is gone outright. What refills
the list at 5/µm² is unidentified. Do not quote §11 as fixed.

## 12. The acceptance tests

### Provenance: which numbers in this file are still worth anything

**Read this before quoting any measurement here.** The decision layer changed
twice on 2026-09-03 — first the two-class width prior (§8b), then the MAP
width fit (§8b) — and a measured optimum is a statement about the algorithm
that produced it. Numbers taken before a change are evidence about a pipeline
that no longer exists.

The split is not arbitrary, and it runs along the layer boundary:

| layer | changed? | its measurements |
|---|---|---|
| `psf`, `lmga` numerics, `patches`, the conditioning and floor guards (`COND_GUARD`, `A_MIN_REL`, `REFINE_TOL*`) | no | **still valid.** These are statements about arithmetic and are independent of what the model contains |
| the decision layer — `PRUNE_TAU`, `LOG_SEED_Z`, `SIGMA_SLACK`, `FOCUS_BAND`, `FOCUS_WIDTH_GAMMA`, and every recall / tiling / FP number downstream of them | **twice** | **re-derive before trusting.** An optimum found under a fixed-width, single-class, ML-fitted model says nothing about where it sits now |

Sections 10b, 14 and 15, and the bead-frame numbers in 16, predate both changes
unless they say otherwise. Where a table has been re-measured it says so; where
it has not, it is kept because the SHAPE of the effect is the argument and the
levels are not. Mark new tables with the arm names that produced them
(`scripts/bench_sim.py`'s `legacy` / `mix`), so the next change can tell at a
glance what it invalidated.

The reason this is a section and not a footnote: `backend.RustBackend` asserts
on load that `PRUNE_TAU` agrees between the two languages. Freezing a stale
optimum into a second implementation is the expensive version of this mistake.

### The five tests

In increasing order of authority.

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
run of this arm that distinction decided the answer: the ROI solver appeared to
halve FP under width mismatch (18.50 → 10.83 at bright/0.015, spread 0.40), but
with recall divided out its genuine anti-tiling effect was **0–21%** and
negative in two cells, because its matched-filter seeder found only 0.67–0.79
of the wide emitters against the round loop's 0.92–0.96. That solver was later
removed (§13); the reading lesson is the one that survives it.

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

**Recall, tiling and localization against a real PSF at known depth**
(`scripts/bench_sim.py`). The three arms above all render emitters as
Gaussians. `data/sim_out` does not: it is a `psfkit` vectorial spinning-disk
confocal simulation whose truth table carries each emitter's `z_um`,
`photons_in_frame` and `peak_photons`, so every row can be conditioned on
depth, on brightness or on crowding instead of averaged over them. It is the
only arm where width mismatch is set by physics rather than by a
`sigma_spread` knob, and it is what §8b's free width was measured on.

Its extra-detection columns are split by cause against **each truth's own
sigma(z)**, which is the distinction the fix depended on: a detection 2.5 px
from a source that is 1.96 px wide is a piece of that source, and a fixed
matching radius calls it invented. `scripts/psf_sigma_scan.py` regenerates the
sigma(z) table from `psfkit` when the simulation changes.

Density 1 emitter/µm², 20 frames, in-focus = |z| ≤ 0.2 µm. `fixed` is
`slack=None`, `slack` the single-band free width — the two arms that existed
before the mixture:

```
          det/fr  tile  ghost  focus recall  med err  rsd z  |z|>3  resid rsd
fixed       27.6  7.15   0.00        96.6%    0.066   1.29    3.5%      1.129
slack       21.4  1.00   0.10        95.9%    0.064   1.24    2.5%      1.106
```

Tiling falls 86% at unchanged in-focus recall and error, and the pull spread,
the pull tail and the residual all improve. On the emitters beyond |z| = 0.35
µm that are still detected, `A/A*` goes 0.34 → 0.76 and `med err` 0.592 →
0.236: the defocused population stops being modelled as a cloud of
wrong-width fragments and starts being modelled as what it is.

The fitted width tracks the depth it came from, which no fixed-width run can
report at all:

```
  |z| µm     <.05  .05-.15  .15-.25  .25-.35   >.35
  fitted σ/σ₀  0.96    0.98     1.04     1.23   2.00
  true   σ/σ₀  1.00    1.02     1.09     1.34   2.42
```

**It used to cost recall at density; it no longer does.** The same comparison
at 5 and 10 emitters/µm² (8 frames) kept the tiling win — 43.0 → 5.5 and
79.1 → 12.0 per frame — but cost in-focus recall, 96.3% → 92.0% and
91.8% → 82.2%. That was read as a search-order problem and was not one; the
MAP width fit of §8b repaid it, and §8b's own table is the current
three-density comparison (`legacy` → `mix`, recall ahead at 1 and 15
emitters/µm² and 0.5 points behind at 5). Section 15 keeps the wrong diagnosis
and how it was caught, because the two arms above are what it was caught with.

**Widefield, bright, out of focus** — the arm that was missing until
2026-09-03, and the one that matches the bead movies. Every simulated arm
above it is CONFOCAL, and a pinhole makes a defocused emitter broader *and
dimmer*: at 2.19 AU the axial response is 1.00 at focus and **0.044** at
\|z\| = 1 µm. Widefield conserves the flux and only spreads it — **0.499** at
the same depth. So no confocal arm can produce a bright, broad object at any
depth, and the regime where a fixed-σ model does its worst was untested.

Generate it with `psfkit`'s simulator at `--instrument widefield` (a
`ConfocalOptics` with the pinhole opened to 20 AU). At 1 emitter/µm²,
±1 µm of depth, 1000–20000 photons, 6 frames, against 34.3 true emitters per
frame:

| 6 frames, 34.3 truth/frame | single band | **current** |
|---|---|---|
| detections/frame | 88.7 | **34.8** (+35.8 wide) |
| tiles/frame | 67.17 | **18.67** |
| ghosts/frame | 0.00 | 0.00 |
| in-focus recall | 100.0% | 97.8% |
| in-focus med err | 0.043 | **0.040** |
| in-focus RMSE | 0.102 | **0.078** |
| in-focus rsd z / \|z\|>3 | 1.56 / 8.7% | **1.15 / 4.4%** |
| overall RMSE | 0.680 | **0.584** |
| residual rsd | 2.058 | 2.042 |

The single band returns **2.6× more detections than there are emitters** and 67
tiles per frame. This is the failure §8b exists for, seen in the regime that
provokes it hardest; the width prior removes **72%** of it for 2.2 points of
in-focus recall — one emitter of 46 — while cutting in-focus RMSE by 24% and
bringing the pull spread from 1.56 to 1.15.

**Ghosts are 0.00 in both arms.** An earlier run of this table reported 4.17
and 0.83, and that was entirely an artifact of scoring widefield data against
the CONFOCAL σ(z): `TILE_R · σ(z)` was too small, so genuine tiles were being
called invented. Nothing was hallucinating anything. It is the cleanest
illustration of why the two optics need separate tables.

Read the residual honestly though: **both arms sit near 2.0**, so neither model
explains this data. Widefield out-of-focus light is a large, smooth, heavily
overlapping haze, and a sum of Gaussians bounded at 2.2σ is not what it is.
`A/A*` runs 0.45 overall for the same reason. The tiling is reduced; the
modelling of the haze is not.

**What the 18.67 remaining tiles are, exactly.** Not false positives — the
haze invents nothing, and the zero in the ghost row is the measurement that
says so. Every one of them sits on real signal. Attributing each to the truth
it subdivides and reading that truth's own width off the widefield table:

| \|z\| of the parent | n | true σ/σ₀ there | inside the model space? |
|---|---|---|---|
| 0.2 – 0.4 | 1 | 2.0× | yes |
| 0.4 – 0.6 | 33 | 4.2× | no |
| 0.6 – 0.8 | 34 | 6.5× | no |
| 0.8 – 1.0 | 44 | 7.3× | no |

**99% of them subdivide an object wider than `SIGMA_SLACK[1]`.** That is §8b's
original failure one level up: a model that cannot represent an object tiles
it, every decision rule behaving correctly inside a model space that does not
contain the answer. The distinction from a false positive is not pedantry — it
changes the fix. An invented detection means a threshold is too loose; this
means the model space is too small *for this data*.

### The residual is a biased arbiter on this arm — read tiles instead

Sweeping the bound on this data, where truth is available:

| bound | in-focus recall | tiles/frame | med err | residual |
|---|---|---|---|---|
| 2.2 | 97.8% | 18.67 | 0.040 | 2.042 |
| 3.0 | 97.8% | 11.17 | 0.042 | **2.002** |
| 5.0 | 97.8% | **4.33** | 0.040 | 2.219 |
| 8.0 | 95.7% | **1.83** | 0.045 | 2.762 |

**Tiling falls 90% at unchanged in-focus recall, and the residual gets worse
doing it.** The two disagree, and the residual is the one that is wrong here.
A widefield defocused PSF is not a Gaussian — it has rings — so a handful of
narrow Gaussians genuinely fits its PIXELS better than one wide Gaussian does,
while being wrong about the OBJECTS. The residual rewards exactly the
overfitting the wide class exists to stop.

That is a real limit on a statistic this document leans on elsewhere. The
residual is free of any matching radius or truth definition, which is why §12
gives it authority — but it is **not** free of model-form bias, and where the
model's shape is wrong it will prefer more emitters. Where truth exists, judge
the bound on tiles and recall. Where it does not, ask whether the extra
detections are STACKED or ISOLATED: on the bead frame, raising the bound from
2.2 to 5.0 removed 50 detections while the share sitting within 2σ₀ of another
barely moved (7% → 5%), so what it removed was mostly isolated objects — real
beads absorbed into a wide fit, not tiles de-duplicated. There, 2.2 is right.

**So the bound is a property of the data's optics and depth range, not a
constant.** Confocal simulation and the bead frame both want 2.2; widefield
with ±1 µm of depth wants ~5. Sweep it per dataset and do not carry it between
instruments.

`bench_sim.py --optics widefield` selects the right σ(z) table for this arm;
`--optics confocal` is the default and is what every other arm needs. Scoring
widefield data against the confocal table understates every defocused
emitter's width, and `TILE_R · σ(z)` then calls genuine tiles "invented".

One caveat that no table fixes: the simulator's 21 px stamp truncates a
strongly defocused widefield PSF. Past \|z\| = 0.8 µm the fitted width *falls*
(6.12 → 4.61 px out to 1.0 µm) because a Gaussian fitted to what is left of a
PSF that has spread beyond its stamp is narrower. The table is truncated at
0.80 µm for that reason and `np.interp` clamps beyond it, which is the honest
extrapolation. `photons_in_frame` is likewise what was deposited, not what was
emitted.

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

**The pre-search aggregate pass** (`find_aggregates`, `render_aggregates`,
`detect(reject_aggregates=)`, and the `agg_mask` / `agg_model` / `usable_px`
plumbing that threaded through the round loop), 2026-09-03. It fitted a free-σ
Gaussian at every round-0 candidate, called the wide bright ones aggregates,
froze them into `bmap` and masked them out of the candidate list, the
background estimate and `lam`'s area.

The wide class does the same job inside the search, where the flux is still
undivided. Measured on `data/sim_out`, the pre-search pass masked **0.0% and
0.8%** of the frame at 1 and 5 emitters/µm² — it barely fires — and where it
did fire it was *worse*: recall 0.936 → 0.926, tiles 3.00 → 3.50. The reason
is §10b's own measurement, that freezing a σ≈1.5 object into `bmap`
double-counts against emitters fitted beside it.

What went with it matters more than its own 190 lines: the masking plumbing
sat **inside the round loop**, in the path the port has to reproduce, serving
a flag that is off by default. `flag_aggregates` and `aggregate_report`, which
are post-hoc and read a finished result, stay — below the port boundary
banner in `core.py`.

The regime it genuinely covered and the wide class does not is an object
*wider* than `SIGMA_SLACK[1]`. On this data that regime is empty. On frames
with real extended structure it is untested; look at the residual.

**The NPMLE flux prior** (`MixturePrior`, `npmle`, `fit_flux_prior`),
2026-09-03. A Kiefer–Wolfowitz deconvolution of the amplitude distribution,
built to fix a real and still-unfixed defect: an exponential flux prior has
its mode at zero, so it asserts that one emitter at `2A` is ~1800× more likely
than two at `A`, which is backwards for a monodisperse population and is what
makes every sub-σ split impossible (`prior.py`'s header keeps the table).

It was **never wired in**. `detect` has always run the exponential, nothing in
the repository called `fit_flux_prior`, and no benchmark arm used it — and it
would have been *incorrect* if anything had: a curved flux prior makes `Λ`
non-zero on the amplitude block and `evidence` still omits it there (§15). 200
lines of EM in the one module the Rust port needs. The finding is kept in
`prior.py` and §15; re-adding the estimator means adding `Λ` with it.

**The flux/width coupling** (`prior.DefocusFlux`, `detect(defocus=)`), 2026-09-03.
Priced "a defocused source is wide AND DIM" by pushing the population's
brightness prior through the axial response — right physics, and it was worth
+0.5 points of recall at a model bound of 8.0. At the shipped bound of 2.2 it
is a wash at every density (recall 95.0 → 95.0, 88.5 → 88.8, 76.8 → 76.1;
residual moving by ±0.04 in both directions). Its remaining argument was that
it would let the bound rise to the optics' own 3.2×, and the re-sweep under
the MAP fit says the bound does not want to rise (§8b). Removed with the
argument that justified it. If the far-defocus width bias in §15 ever needs
fixing, this is where to start, and it is in git history.

**The post-hoc width filter** (`infocus.py`, `filter_in_focus`,
`spotsolve.INF_FOCUS_*`, `spotsolve.VAR_SIGMA_*`), 2026-09-03. It refitted the
finished detections with one sigma per emitter and rejected the ones whose
width was not physically in-focus. It asked the right question at the wrong
time: a tile with its neighbours frozen into its halo IS a PSF-sized bump —
measured, sigma 1.22 against 1.20 for an ordinary emitter — so by the time it
ran, the evidence it needed had been divided up among the tiles. §8b's class
boundary asks the same question during the search, where the flux is still
undivided, and returns the same records (`result.width_rejects`,
`result.aggregates`) from the run itself. 300 lines, one module, one test file
and five public names.

**The ROI solver** (`aguet.py`, `detect_local`, `solve_roi`, `_grow`,
`_shrink`, `_try_wide`, `_fit_wide`, `_wide_start`, `_dedupe_wide`,
`evidence.log_bf_wide`), 2026-09-03. A whole alternative architecture: a
matched-filter detector seeds ROIs, the window is built once per ROI, and the
model-selection question is asked inside it — including `H_wide`, "is this one
broad object rather than K narrow ones", which §10b named as the missing
discriminator and which this file's `_try_wide` computed properly.

The mixture width prior does that job better, and by then the ROI solver was
dominated on every accuracy axis at once (6 frames, `--impl py`):

| | recall | tiles/frame | ghosts |
|---|---|---|---|
| 1/µm², round loop / roi / roi-nowide | **96.2%** / 78.8% / 96.2% | **0.83** / 2.67 / 9.33 | **0.00** / 0.17 / 0.33 |
| 5/µm², round loop / roi / roi-nowide | **92.6%** / 80.5% / 90.5% | **2.33** / 16.17 / 30.67 | **0.00** / 0.17 / 0.00 |

It was 4× faster in Python, which was its other argument, and that is not worth
12–17 points of recall and 7× the tiling — nor worth porting twice. 850 lines,
about 14% of the library. `docs/aguet-spot-algorithm.md` and the paper stay:
the detector is still the right reference for what a linear filter can and
cannot do, and §21 of `PORTING_NOTES.md` rests on it.

**The MERGE move** (`_merge_pass`, `_try_merge_wide`, `_group_width`,
`_link_components`, and `log_bf_wide`'s free-width branch), 2026-09-03. It
proposed replacing a whole linked group with ONE object of free width — SPLIT's
exact opposite, scored by the same evidence — and it was written to collapse
tilings that had already happened. At the shipped `SIGMA_SLACK` it collapses
none, because there are none: an incumbent that can widen never gets tiled in
the first place. With and without it, at 1, 5 and 15 emitters/µm², the arms are
**identical to every digit printed** — same detections, same tiles, same
residual — while it cost 10–30% of the round loop's time. It was not a bad
move; it was a move whose job the model space had already done.

It is the one entry in this section with a condition attached, and the
condition is specific: **if the width schedule of section 15 lands, bring it
back.** Holding the widths fixed while N is decided restores the tiling on
purpose, and something then has to collapse it. Its measured value at a model
bound of 2.6 — where a little tiling survives — was +1.3 points of in-focus
recall at 15/µm², which is the shape of what it would be worth again.

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

**Sub-σ recall is ~0.52 and neither solver does better — and this is SPLIT's
job, not a limit to accept.** ~~At the identifiability limit. Not obviously
fixable.~~ Wrong: the Fisher information is not the limit. Re-derived
2026-09-03 — CRLB on the separation of an equal pair, 20 e⁻/px background,
σ = 1.2, reported as `d/SE(d)`, the significance the separation itself carries:

| d/σ | A = 500 | A = 2000 | A = 8000 |
|---|---|---|---|
| 0.25 | 0.2 | 0.6 | 1.5 |
| 0.50 | 0.8 | **2.3** | **5.7** |
| 0.75 | 1.7 | **5.0** | **12.3** |
| 1.00 | 3.0 | **8.5** | **20.8** |

There is a real floor below ~0.3σ at realistic flux. Above it the information
is plainly present, and the ~68% the pipeline reaches in that bin is the
decision rule refusing what the data supports.

Note what SPLIT measurably does INSTEAD (§7): **+0.0 points below 1σ**, +3.3 at
1–2σ, +6.7 at 2–3σ. It is doing FIND's clean-up where the LoG merged two
maxima, and not the job it exists for. Those are two separate repairs — §5's
Hessian note targets the first, this entry the second.

The block is three constants in series — the amplitude prior, `COND_GUARD` and
`PRUNE_TAU` — and relaxing any one alone changes final N by nothing. Measure
final N, never stage acceptance.

**Why relaxing one alone does nothing: two of the defects cancel.** The
exponential flux prior has its mode at zero, so it asserts one emitter at `2A`
is ~1800× more likely than two at `A` — that BLOCKS sub-σ splits, measured at
0% acceptance at 0.5σ against 83% under a prior with its mode away from zero
(`prior.py`'s header). The missing `Λ` on the amplitude block lets a degenerate
configuration collect an Occam *bonus* from its own ill-conditioning — that
PERMITS them. They push opposite ways and the blocking one wins.

So the repair is a pair and neither half is safe alone. Fix only the prior and
false splits of single emitters follow; fix only `Λ` and sub-σ splits get
*harder* — measured at −4.0 nats for a 0.5σ split. The 83% was measured WITHOUT
`Λ`, so the honest expectation for the correctly-specified pair is between 0%
and 83%, and it has not been measured. `COND_GUARD` is the third, and it should
loosen once `Λ` is doing its job — it is a hard cutoff standing in for exactly
that term.

**The Laplace volume term omits the prior's curvature — in the AMPLITUDE
block.** The width block was repaired on 2026-09-03 (§8b): `lmga` adds the
width prior's curvature to the matrix it returns, so `F` there is already
`F + Λ_σ`. What follows is about the amplitude block, which is still
`J' W J`. `evidence._log_bf_add_from_logdet` builds the Occam factor from
`log|F|` where the Laplace evidence wants the negative Hessian of the log
*posterior*, `log|F + Λ|`. Under the old
`Exp(1/A_s)` prior `log g(A)` is linear in `A`, so `Λ = 0` exactly and the
omission was correct, and the shipped pipeline runs exactly that — so the
amplitude block is currently right *because* the flux prior is the one whose
shape §13 records as wrong. The two defects cancel, and fixing either alone
un-cancels them.

Measured with a curved prior in place: including `Λ` moves `log BF` for a 0.5σ
split by −4.0 nats under a `U(900,1900)` population and −5.8 under a tight one,
~0 under a broad one. A degenerate configuration collects an Occam *bonus* from
its own ill-conditioning, and `Λ` is the term that regularizes it — which makes
`COND_GUARD` a hard cutoff standing in for a term that belongs in the formula.
Fixing it makes sub-σ splits **harder**, not easier.

The width half of `Λ` is done (§8b): `lmga` returns `F + Λ_σ`. The amplitude
half is this entry, and it is only reachable by a caller who supplies their own
curved `FluxPrior` through `detect(prior=)`.

**Amplitudes run 110–113% of frame flux on the bead data.** Tracks density
(+1.0% at 0.034, +2.2% at 0.055 on a perfectly flat background), so it points at
crowding — flux from an undetected close partner absorbed by the neighbour that
was detected. It is **not** background structure; the surface does not fix it
(section 10).

**A widened emitter can hide its own neighbour.** ~~This is now the largest
defect, and it is what caps `SIGMA_SLACK[1]`.~~ Mostly repaired on 2026-09-03
by the MAP width fit (§8b), and the history is worth keeping because the wrong
diagnosis survived two attempts to act on it.

With ML widths the free width cost 4.6 points of in-focus recall at 5/µm² and
7.0 at 15/µm², all of it in emitters with a close neighbour: broken out by
nearest-neighbour distance at 5/µm², the `>3 sigma` bin lost 2 points while the
`2-3 sigma` bin lost 17. It was read as a SEARCH-ORDER problem — FIND proposes
on the residual, `refine` widens an emitter before its neighbour has ever been
a candidate, and the forward-only loop never revisits it — and the fix was
expected to be a schedule.

**It was not the search.** `crlb.py`'s ORACLE arm, where N is fixed at truth,
the starting values ARE truth and nothing is searched at all, shows the same
collapse: 0.1198 → 0.1999 px of median error at 1-2 sigma. Two things had
already been tried against the search-order reading and neither moved it — the
flux/width coupling (§8b: under a point either way, because it prices a wide
object being *bright* and this is a wide object being merely *broad*) and a
MERGE move (§13: nothing at all). The fit was the problem, and handing it the
prior it was already being scored against recovered 4.1 points of in-focus
recall at 5/µm² and 7.2 at 15/µm².

**What is left of it.** The sub-sigma bin is untouched — that is the entry
above, and a different problem. At 1-2 sigma the MAP fit recovers about a third
of the gap to a fixed-width run on fields with NO defocus, and the rest is the
honest cost of carrying a parameter that buys nothing there. The width
schedule was never measured and is no longer obviously needed.

**The width prior biases the far-defocus tail.** The new one. Past
\|z\| = 0.35 µm the fitted width comes back at 1.56x where the truth is 2.00x,
and the flux with it: `A/A*` 0.62 against the single band's 0.88. That is the
Cauchy core doing exactly what it was asked to — 3.26 nats against a width of
2x — to objects that genuinely are that wide. It costs the REPORTED set
nothing, since those objects sit outside the reporting band's intent, but it
means `fit_sigma` is a shrunk estimate of defocus rather than an unbiased one,
and anything reading it as a depth proxy has to know that. The principled fix
is `FOCUS_WIDTH_GAMMA` giving way to the pushforward of the real sigma(z),
which `defocus=` already supplies for the flux half of the same calibration.

**The free width is Python-only.** `spotsolve_rs` implements the fixed-width
passes, so `detect` ignores `impl` whenever `slack` is on and says so. That is
a 20× penalty on the default path — 15 ms/frame against 525 ms on a 64² frame
at 1 emitter/µm². The four passes are written once for both layouts
(`core._fit_any` takes and returns unpacked parameters), so the port is a
layout change in `lmga`'s parameter vector and the Bayes factor's
count-and-width term, not a second algorithm. `evidence` is now told the layout
by a `prior.WidthPrior` or by its absence, and the free-width fit is a MAP fit
under that prior, so the port carries a diagonal penalty into `lmga` as well. `tests/fixtures/*.json` are still generated at `slack=None` and must be
regenerated with the default once it lands.

**The score map runs systematically negative on real frames** (median z ≈ −1.9
on FOV1, −2.2 on FOV2, on both solvers). The model very slightly over-explains
everywhere. Unexplained.

## 16. Knob reference

| knob | value | what it controls |
|---|---|---|
| `LINK_FACTOR` | 2.5σ | emitters fitted jointly |
| `HALO_FACTOR` | 5σ | emitters frozen into the model |
| `BBOX_PAD` | 3σ | pixel context around a group |
| `SEED_ALPHA` | 0.05 | FIND's family-wise false-seed rate; the cut is DERIVED from it, the frame and σ — see §5 |
| `AGG_AMP_RATIO` | 20.0 | post-hoc **over-bright** cut, vs the median detection |
| `AGG_MASK_RADIUS` | 3.0 | σ_fit of support recorded per wide object |
| `SPLIT_DISPS` | (1.0, 1.6)σ | where a split is proposed |
| **`SIGMA_SLACK`** | **(0.70, 2.2)σ** | **the MODEL SPACE — widths a fit may represent — §8b** |
| **`FOCUS_BAND`** | **(0.80, 2.0)σ** | **the REPORTING BAND — widths that count as a detection** |
| `FOCUS_WIDTH_GAMMA` | 0.20σ | half-width of the width prior's Cauchy core — §8b |
| **`PRUNE_TAU`** | **2.0** | **the precision/recall dial — the only one** |
| `REFINE_SWEEPS` | 4 | cap on refine's iteration to a fixed point |
| `REFINE_TOL` | 1e-3 px | position shift below which a sweep is a no-op |
| `BG_KERNEL` | 25 px | background surface window (`None` = one scalar) |
| `BG_MASK_RADIUS` | 3σ | emitter support excluded from the background |
| `BG_MIN_PIXELS` | 25 | pixels a window needs before it is believed |
| `k_max` | 12 | cap on emitters in one joint fit |
| `COND_GUARD` | 1e3 | scaled condition number above which F is unusable |

`PRUNE_TAU` is the one to move. 2.0 was the measured optimum on both bead
frames under the fixed-width, single-class, ML-fitted pipeline — all three of
which have since changed — and it has been **re-swept under the current
algorithm** on the confocal simulation (moderate, 6 frames):

| `prune_tau` | recall | med err | RMSE | rsd z | \|z\|>3 | tiles |
|---|---|---|---|---|---|---|
| 1.5 | 93.4% | 0.077 | 0.217 | 1.30 | **6.1%** | 3.33 |
| **2.0** | 92.6% | 0.076 | 0.220 | **1.25** | 7.0% | **2.33** |
| 2.5 | 92.3% | 0.076 | 0.220 | 1.29 | 8.1% | 2.83 |
| 3.0 | 89.4% | 0.079 | 0.224 | 1.39 | 10.4% | 1.67 |

2.0 survives: best pull spread, fewest tiles, within a point of 1.5's recall,
and 3.0 clearly wrong. The mechanism is unchanged — raising it costs
localization as well as recall, because removing one member of a real close
pair leaves the survivor absorbing both fluxes and sitting between them. This
one mattered beyond its own value: `backend.RustBackend` asserts on it at
load, so a stale optimum would have been frozen into a second implementation.

`LOG_SEED_Z` **is no longer a knob.** It was invisible until 2026-08-28 because
it was expressed in the LoG's raw output units, and it was a compatibility
value — exactly `1.5 / ‖w‖₂(σ=1.2)` — not a defensible operating point. It is
now derived: `calibrate.seed_threshold` sets a family-wise false-seed rate
(`SEED_ALPHA = 0.05`) over the independent local-maxima tests, which gives 3.70
on 64² at σ 0.818 and 4.43 on 512² at σ 1.45. **It is not a constant** — it
scales with the frame and the PSF, and carrying one number silently gave every
instrument a different detector.

The old value was ~2.4σ too conservative, and the cost was not where you would
look for it. FIND is a seeder, not a decision rule: a spurious candidate still
has to win a Bayes factor, so it costs runtime, while a real emitter that fails
the seed cut is lost outright — the forward-only loop never revisits it.
Lowering the cut raises recall, makes SPLIT stop doing ADD's work (split/add
0.88 → 0.60), and lets the round loop terminate on its own instead of
exhausting `max_rounds`. See `calibrate.SEED_ALPHA` for the table.

**The tell needed no ground truth:** at 6.496, ADD accepted 62.5 of 63.5
candidates — 98%. A proposal mechanism the decision rule almost never refuses
is not proposing anything marginal. Watch that ratio on any new instrument.

`lam` (emitter density) and `A_s` (amplitude prior scale) are **not** knobs —
they are re-estimated empirically each round.

`FOCUS_BAND` is a statement about the optics and about what the caller wants,
not a dial to tune against a score. `SIGMA_SLACK` is half physics and half
compromise: its lower bound is below the diffraction limit so that a broken fit
can show itself, and its upper bound *should* be where the sample's defocus
actually stops (3.2× here). The 2.2 was chosen to cap a defect the MAP fit has
since largely repaired, so it too is a stale optimum (§12, Provenance). Move it
only with the residual in front of you.

Two switches restore earlier pipelines exactly, and they are for attribution,
not for use: `slack=None` is the fixed-width pipeline — which is what
`spotsolve-rs` and `crlb.py` pass, because the Rust core implements that layout
only — and `band=None` is the single-band free width, one class and one uniform
width prior. `scripts/bench_sim.py`'s `legacy`, `slack`, `mix` and `mix-phys`
arms are those settings, and are how every number in §8b and §15 was attributed
to one change at a time.

---

## Porting it to Rust

`docs/RUST_PORT_PLAN.md` is the standalone brief: what is already in
`rust/spotsolve-core` (more than this file used to claim — the `4K+1` layout
and a variable-width fit are both there), what is left, in what order, which
fixture block catches which specific mistake, and which constants are safe to
freeze. `docs/PORTING_NOTES.md` is the practices that go with it.

---

## Reading order for someone new to the code

`structs.py` (the contracts) → `psf.py` → `lmga.py` → `prior.py` →
`evidence.py` → `core.py` → `scripts/crlb.py` and `audit.py` (how it is
judged). `prior.py` before `evidence.py`: the Bayes factor's flux and width
terms are both objects now, and its header carries the argument for the
two-class model that section 8b summarizes.

---

## Repository map

```
src/spotsolve/     the library — `import spotsolve`
scripts/           command-line entry points; each has a `--help`
tests/             the layer checks, and tests/fixtures/ the golden values
rust/              spotsolve-core (the algorithm) + spotsolve-py (PyO3 wrapper)
docs/              PORTING_NOTES.md, and the Aguet paper writeup (reference
                   only; the detector built from it was removed -- see §13)
data/              two bead crops the README's numbers come from
```

Every module, in dependency order. Nothing here is optional to the layer above
it; paths below are relative to `src/spotsolve/` unless shown otherwise.

**The algorithm** — this is the port target, and it is all of it. About 3,500
lines, in dependency order:

| file | lines | what it is |
|---|---|---|
| `structs.py` | 150 | the `theta` layout and the result records; imports nothing |
| `psf.py` | 283 | pixel-integrated Gaussian, model and Jacobian, `3K+1` and `4K+1` |
| `lmga.py` | 420 | bounded Poisson MLE/MAP optimizer (Coleman–Li affine scaling) |
| `prior.py` | 365 | the flux prior, and the width prior the fit and the evidence share |
| `evidence.py` | 289 | Laplace log Bayes factor, `logdet`, the conditioning guard |
| `moves.py` | 125 | the two proposals: residual quadrupole axis, and the split |
| `patches.py` | 177 | grouping emitters into jointly-fittable patches, and the halo |
| `calibrate.py` | 345 | gain, robust background, model rendering |
| `core.py` | 1720 | `detect` and `refine` — FIND → ADD → SPLIT → REFINE → PRUNE |

`core.py` carries a **port boundary banner** near its end. Everything above it
runs inside the search; everything below reads a finished `DetectResult` and
should not cross.

**Judging it** — the three acceptance tests of section 12, and their inputs:

| file | what it answers |
|---|---|
| `simulate.py` | synthetic Poisson fields with known truth |
| `metrics.py` | matching detections to truth |
| `audit.py` | is anything PSF-shaped left in the residual? |
| `../../scripts/bench.py` | count, isolation-resolved recall, false positives, runtime |
| `../../scripts/bench_sim.py` | the same, against a real confocal PSF at known depth |
| `../../scripts/psf_sigma_scan.py` | what width and flux the PSF presents at each depth; `--optics confocal\|widefield --emit` prints the arrays `bench_sim.py`'s `OPTICS` table wants |
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
