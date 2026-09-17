# Detector and fitter design history

Moved from source comments on 2026-09-16. These are historical measurements
and explanations, preserved without revalidation; they are not guarantees
for the current algorithm. Current behavior is documented in the source and
[COUNT_SELECTION.md](../COUNT_SELECTION.md).


## boxsearch: Overview

Box-local localization: in each box, the best fit the data support wins.

Ported from the Python reference `box.localize_boxes`, retired on
2026-09-11 (last present in commit `ea6b17f`, `src/spotsolve/deprecated/`).
At retirement the two agreed exactly -- counts, recall and precision -- on
the referee cells and on both real 256x256 frames. The measurements that
set each constant are recorded beside it; the ones that shaped the design
as a whole are below.

The outline below describes the default `Selection::Fixed` mode.
Experimental `Selection::Bic` compares count models along forward and
backward paths and revisits removals after polish. See `search_box_bic`
and docs/COUNT_SELECTION.md for its score, limitations and measurements.

```text
d        raw - offset, in ADU; no gain, no read noise
NOISE    local pixel sd (NOISE_WIN px, 4th-difference filter) and the local
         dispersion phi = sd^2 / local median of d
FIND     LoG peaks of d above PEAK_Z local sds                 -> candidates
BMAP     smooth surface from the pixels no candidate reaches
BOXES    candidates within LINK_FACTOR*sigma share a box, <= k_max each
SWEEPS times, for each box, brightest first:
    fit the level alone (K = 0), BMAP's shape and the neighbours' current
    emitters held fixed
    FORWARD   place at the strongest owned residual LoG peak above
              PEAK_Z; keep it iff I falls by > ADD_NATS * phi
    BACKWARD  (K >= 2) drop the cheapest emitter while it costs less
POLISH   block-Jacobi refits at fixed N until nothing moves
CLASSIFY by fitted width, out of band only when significantly so
```

Each emitter is decided in the box that owns it, by one comparison rule.
There is no split move, no separate removal pass, no forced-removal
threshold, no conditioning guard and no width prior: a collapsed or
redundant emitter explains almost no deviance, so it cannot pay
`ADD_NATS` on the way in and costs almost nothing on the way out.

# No camera calibration: every decision is scale-free

The detector used to take `gain` and `read_noise` and work in
photoelectrons. It now takes neither, and the frame stays in ADU above the
offset. Three facts make that possible.

* The FIT never needed the gain. The I-divergence is homogeneous,
  `I(d/g, m/g) = I(d, m) / g`, so positions and widths are invariant and
  amplitudes scale as `1/g` (measured over g = 1..50: positions to 7e-7 px).
* The DECISION `dI > c` does, but only through a scale: in ADU,
  `dI_ADU = g * dI_e`. Here `c` is multiplied by the box's own dispersion
  `phi`, the pixel variance per unit of signal, measured from the frame
  ([`noise_map`]), so `dI_ADU / phi` is in the units the old test was in.
* The LoG cut, in FIND and in every placement, is divided by the local pixel sd
  instead of `sqrt(model)`. The sd includes read noise, and haze, without
  being told either.

Scaling a frame by 0.5-7.3x leaves the seeds identical and N within 4 of
290 (the absolute fitter tolerances and floors). Read noise needs no
model: on EMPTY 64x64 frames at 1-20 e- background and 1.6-2.5 e- read
noise it gives 0-0.7 fits per frame (the shifted Poisson it replaced:
0.0; plain Poisson: 1.2-13.3).

Measured 2026-09-14 against the calibrated detector on `simulate` truth
(128x128, flux U(200, 1500) e-, bg 20 e-, sigma 1.4 with log spread 0.2,
gain 2.4, read noise 1.6; six frames, 1 px match), recall / precision:

```text
density   calibrated    gain estimated   gain-free
 0.01     .850 .981      .847 .978       .855 .963
 0.03     .682 .917      .677 .918       .683 .913
 0.06     .511 .842      .496 .848       .510 .839
```

On the real GEM movie (`hyp7gem_wt_01_crop_128x128`), with emitters spiked
into its frames, gain-free recall of 150-300 e- emitters is 2-6 points
under the calibrated detector's at the same false-positive rate. That is
close to the spike-in noise (12 tracks per class).

# Background: a smooth map, not a plane

[`background_map`], a masked 25 px local mean, is each box's known shape,
with the box's level free. It was chosen over a plane per box (level and
slopes free) and over a free constant alone. Measured 2026-09-11 on the
referee frames (64x64, flux U(900, 1900) e-, bg 20 e-, seeds 17-19),
recall / precision / invented per frame. Haze `L, H` is white noise
Gaussian-filtered at L px, scaled to span [0, H] e-:

```text
cell               constant           plane              map
flat  0.015 0.4    .702 .980  0.7     .660 .949  1.7     .695 .970  1.0
flat  0.055 0.4    .424 .855 12.3     .426 .856 12.3     .438 .873 11.0
haze  L15 H60      .701 .957  3.3     .707 .978  1.7     .729 .992  0.7
haze  L5  H60      .682 .952  3.7     .670 .960  3.0     .704 .966  2.7
haze  L15 H20      .710 .983  1.3     .698 .978  1.7     .698 .961  3.0
```

The plane's two extra parameters cost sparse fields; the map costs no fit
parameters or time. Weak haze (H20) is the map's one worse cell, by 3-6
events over three frames. Re-estimating the map from the fitted emitters
after the first sweep moved nothing beyond seed noise. A per-pixel
likelihood mask and a rank-opening haze map were also measured and
rejected.

# No width prior

Every fit is a plain ML fit over the width bounds `slack`. The retired
`detect` pipeline's MAP width penalty pulled every width toward sigma0; in
this search that leaves the wings of a broad emitter unexplained, and the
next placement lands on them as a faint satellite. Removing it, six frames
per cell, recall / precision / tiles per frame:

```text
density spread     MAP width           flat width
 0.015   0.4    .876 .893  3.3    .866 .940  1.7
 0.034   0.2    .825 .929  5.7    .813 .949  4.2
 0.034   0.4    .783 .823 11.5    .776 .858  9.0
 0.055   0.2    .745 .908 10.3    .737 .941  6.7
```

Widths always float. A fixed-width mode was measured and rejected on
2026-09-11: it tiled haze and defocused spots, inventing 8-33 spots per
64x64 frame against 0.7-7 with fitted widths.

# Where Python spent the time, and why this module exists

On frame 0 of the glycerol crop (485 emitters, 0.84 s in Python) the
native fits were 0.10 s. The rest was the search around them: the polish
(0.32 s), a scipy LoG per placement (0.21), ownership distances over every
candidate (0.14), gathering every box's emitters into each halo
(0.14, quadratic in boxes) and Python PSF renders (0.13). Here ownership
and the neighbour lists are computed once per frame, and a halo reads only
the boxes that can reach it.


## boxsearch: SLACK

Widths a fit may take, as multiples of `sigma`: the MODEL SPACE.

The model space has to cover every photon on the sensor, or the light it
cannot represent gets tiled: a fixed-sigma model meets anything out of
focus with two narrow Gaussians, which genuinely do fit a wide blob better
than one. Measured on a confocal simulation with emitters uniform in
+/- 0.5 um, at 1 emitter/um^2 the fixed-sigma search returned 7.2 extra
detections per frame against 13.4 in-focus emitters, every one within
3 sigma(z) of a real emitter.

`SLACK.0` sits below `BAND.0` so that a broken fit -- nothing images
narrower than the PSF -- reveals itself instead of being clipped to the
bound and reported. The upper edge, measured under the retired `detect`
(moderate arm, 6 frames, band fixed at (0.8, 2.0)):

```text
 hi    recall   med err    RMSE   rsd z   |z|>3   tiles
2.2     92.6%     0.076   0.220    1.25    7.0%    2.33
2.6     91.1%     0.078   0.201    1.31    8.0%    2.33
3.2     90.8%     0.078   0.183    1.27    8.7%    4.00
4.0     90.5%     0.078   0.194    1.32    8.1%    2.50
```

Recall falls monotonically past 2.2: a larger model space buys better
parameters for the objects it keeps and swallows close neighbours. The box
search re-measured it: a bound of 4-6 sigma lost 3-5 recall points under
haze and, on a GEM frame, swallowed emitters (N 440 -> 271). 2.2 stands.


## boxsearch: ADD_NATS

Nats of I-divergence an emitter must explain to exist: the whole decision
rule.

Measured 2026-09-10 by replacing every Laplace Bayes factor in the
retired `detect` with `dI - c` (its conditioning guard, prune threshold
and MAP width fit unchanged), 64x64 `simulate` fields, three seeds per
cell, recall / precision / tiles per frame:

```text
density spread     Laplace BF          c = 10            c = 14
 0.015   0.4    .888 .920  2.3    .888 .938  2.0    .898 .969  1.0
 0.034   0.4    .832 .807 13.3    .815 .868  8.3    .805 .899  6.0
 0.055   0.4    .721 .797 20.0    .704 .833 15.3    .688 .867 11.3
```

c = 10 matched the Bayes factor within seed noise in all six cells; c = 14
trades 2-3 recall points for about half the tiles. The Bayes factor's
priors, log-determinants and empirical-Bayes rates were an operating point.

Since 2026-09-14 the frame is in ADU and the test is `dI > ADD_NATS * phi`,
with `phi` the box's median dispersion ([`noise_map`]); that is these nats
in photoelectron units. Re-measured there, on the GEM spike-in referee
(false positives per frame from two matched simulations, then recall of
150 and 300 e- spike-ins at D = 0, 0.43 and 2 px^2/frame), with FIND and
placement cuts of 3.79 and 3.0:

```text
  c     FP    150 e-           300 e-
  5     33    .48 .36 .29      .80 .75 .64
  7     28    .47 .36 .27      .76 .74 .61
 10     24    .44 .34 .27      .72 .69 .58
 14     20    .41 .29 .22      .64 .64 .54
```

Kept at 10. Lowering it buys recall at 300 e-; at 150 e- recall stops near
.48 however low `c` goes, because those emitters are lost at the LoG cut
first (see [`PEAK_Z`]). Aguet et al.'s local-noise floor (keep
an emitter iff its peak exceeds k residual sds) was measured in its place
on the same referee and lies on the same curve (k = 2: FP 35, .50 .38 .31;
k = 3: FP 19, .36 .24 .15), with shorter tracks and more fits.


## boxsearch: OWN_RADIUS

sigma. A box places only on pixels this near one of its own candidates,
and nearer to its own than to any other box's. The nearest-candidate rule
stops two boxes claiming the same light; the radius stops a box reaching
across empty space. Measured on six 64x64 frames at density 0.034, spread
0.2 (one sweep, MAP widths):

```text
radius    recall   prec   tiles/frame
  2.0     .783     .890      8.2
  3.0     .823     .886      8.8
  4.0     .842     .868     10.8
  inf     .844     .863     11.2
```

At 2.0 a partner 2-3 sigma from the candidate it hides behind was outside
the box's reach -- those pairs were 31 of the misses, against 3 for
`detect`. Past 3.0 recall rises about as fast as tiles do.


## boxsearch: SWEEPS

Passes over every box. A box decides against its neighbours in the halo,
and on the first sweep an undecided neighbour is only its FIND seed. A
mismatched seed leaves light the current box claims, and the neighbour's
source ends up split across two boxes. The second sweep re-decides every
box from K = 0 against neighbours that have all been fitted. Same six
frames as `OWN_RADIUS`:

```text
sweeps    recall   prec   tiles/frame   search+polish fits
  1       .823     .886      8.8             504
  2       .821     .927      5.8             725
  3       .827     .935      5.5             942
```


## boxsearch: FIT_TOL_OBJ

nats. A search fit only has to resolve `dI` against `ADD_NATS`. Measured
2026-09-11 against 1e-8 and 1e-4 on the eight referee cells (flat and
haze, three seeds): recall and precision agreed to within one emitter per
cell at all three. On the real frames, 1e-8 moved positions by 0.006 px
(glycerol f0) and 0.036 px (GEM f0), and 1e-4 by 0.03-0.04 px.


## boxsearch: POLISH_MAX_ITER

LM iterations one polish fit may take. A BUDGET, not a convergence
criterion.

A few groups per frame never converge: they are not stalled -- they keep
taking accepted steps to the cap -- because they are descending a
direction the data carries almost no information about. At 256x256 they
are ~1% of emitters and 10-19% of the polish's LM iterations, and they are
NOT disposable: 36 of 38 of their emitters survive to the output. What
that descent is worth is boundable: a decrease of `t` nats moves a
parameter about `sqrt(2t)` standard errors, and each iteration continues
only while it predicts more than `POLISH_TOL_OBJ`. Measured under the
retired `detect` at 256x256, against the same fit run to 400 iterations:

```text
  max_iter   frame s        N   audit   max |dpos|/SE
        25      7.24     3166   clean          0.0820
        50      7.80     3166   clean          0.0023   <- here
       100      8.19     3168   clean          0.0006
       200      8.73     3169   clean          0.0077
       400      9.00     3168   clean          0.0006
```

N wanders by +-3 at EVERY budget, 400 included: the usual churn at
marginal decisions, not a signal. 50 is where the agreement band is
tightest for the least work; on a sweep with no stuck group it is
bit-identical to a budget of 4000.


## boxsearch: POLISH_TOL_OBJ

nats of predicted decrease. Near the optimum
`I(t) ~ I_min + 0.5 dt' F dt`, so stopping at a predicted decrease of
`tol` leaves a parameter about `sqrt(2 tol)` standard errors short.
Measured under the retired `detect` (256 px, N = 3169, Python-vs-Rust
band in SE):

```text
  tol_obj    frame s        N     max |dpos|/SE
    1e-8        9.78     3169            0.0010
    1e-6        8.66     3169            0.0015   <- here
    1e-5        7.68     3168            0.1466
    1e-4        7.08     3161            0.2828
```

1e-6 is the last value that leaves N and the agreement band where 1e-8
does.


## boxsearch: PEAK_Z

The one cut on the LoG statistic, in sd of its null (the local noise).
FIND uses it on the frame to decide which light gets a box; each box uses
it on its fit's residual to decide whether one more emitter is tried.
Either way a peak that clears it still has to pay `ADD_NATS`.

Neither use is free, and neither is the side a loose cut can safely err
on. Too tight costs recall that nothing recovers -- light that never gets
a box, or a placement, is never fitted. Too loose costs false positives as
well as time: a placement is the strongest peak in its box, a maximum over
positions, and among enough noise peaks some pay 10 nats.

# History: two cuts, then one

Until 2026-09-14 these were two numbers. FIND's was a Bonferroni cut at a
family-wise 0.05 false seeds per frame (3.8-4.4 by frame size), on the
argument that a seed only costs runtime; the placement cut was 3.0,
chosen on `simulate` truth across nine field types under the Poisson
normalization (2026-09-12: 4.0 clearly wrong, 2.0 falling off a cliff on
bright arms, where a mis-modelled bright emitter's wing can pay 10 nats).
With the local noise the z is calibrated rather than inflated 1.07-1.19x,
so both were re-measured on the GEM spike-in referee (false positives per
frame from two matched simulations; recall of 150 and 300 e- spike-ins at
D = 0, 0.43 and 2 px^2/frame; the real movie's N; single-threaded time):

```text
seed / placement   FP    150 e-          300 e-          N     ms
3.79 / 3.0         24    .44 .34 .27     .72 .69 .58     222   476 (128^2, Python)
3.0  / 3.0         30    .48 .38 .29     .75 .69 .60     241   506
3.0  / 2.5         31    .49 .40 .29     .76 .72 .60     249   558
2.5  / 2.5         36    .53 .41 .34     .78 .73 .61     263   572
2.0  / 2.0         45    .57 .43 .36     .78 .74 .64     282   631
```

The first row is the "seed costs only time" claim failing: each 0.5 off
the seed cut cost 4-5 false positives. 3.0 / 2.5 was adopted for the dim,
fast population the GEM data is collected for. Then, with the width
band's upper-bound rule and the per-coordinate fitter step in place,
one number for both against that pair (ms on 256^2 GEM frames):

```text
cut            FP    150 e-          300 e-          N     ms
3.0 / 2.5      21.7  .50 .38 .31     .77 .73 .61     233   297
one 2.5        24.9  .51 .41 .33     .78 .73 .60     241   327
one 2.75       22.5  .50 .39 .32     .77 .73 .59     232   280
one 3.0        20.0  .50 .36 .31     .77 .71 .59     224   237
```

One cut at 2.75 matches the pair to about a point and is slightly faster,
so the pair went. Raise it toward 3.0 for fewer false positives and 20%
less time at two points of recall.


## boxsearch: A_MIN

The amplitude floor of a fit: `max(A_MIN, A_MIN_REL * A_max)`, with
`A_max` the window's own amplitude bound. `A_MIN` is only a backstop for a
window whose `A_max` is itself tiny.

The floor has to be relative because what it protects is a RATIO. An
emitter's position block of the Fisher matrix scales as `A^2`, so at bead
fluxes of ~2000 e- an amplitude of 1e-4 puts those entries at ~5.7e-12
against a largest diagonal of ~768 -- a ratio of 3e-14, about 130x float64
epsilon. There the LM step along that direction is unbounded: traced on
such a patch, the fit predicted a 5.2e4 nat decrease, delivered 1.05e-3,
and crawled for 3000+ iterations still 364 nats above the optimum.
Smallest over largest `diag(F)` with a second emitter parked at the floor:

```text
  floor / A_max     min/max diag(F)
    0 (1e-4 abs)        3.0e-14      <- float64 noise
        1e-6            6.4e-10
        1e-4            4.8e-07      (saturates; a different parameter
        3e-3            4.8e-07       becomes the smallest)
```

1e-6 buys six orders of margin over epsilon while remaining physically
negligible -- on a bead patch, a floor of ~0.02 e- of total flux. A larger
floor would start to decide how faint an emitter may be, and that decision
belongs to the search, not to a numerical guard.


## boxsearch: NOISE_STRIDE

px. [`local_median`] is exact on a grid this far apart and bilinear
between. A per-pixel median is too slow here and no better: a 25 px window
holds about 25 independent samples of the filter output, so the full map
is itself noisy (moving the window one pixel changes it by 3% at the
median and 18% at the 99th percentile). On the GEM spike-in referee at
FIND / placement cuts 3.0 / 2.5, full / stride 5 / stride 12 gave FP 31.4 / 31.5 /
31.3, 150 e- recall .49 .40 .29 / .53 .39 .32 / .51 .39 .30, and N 249 /
249 / 248. Half the window.


## boxsearch: BAND_Z

SEs. An out-of-band width is a width flag only if it is out of the band by
more than this many of its own standard errors; otherwise it is a
detection.

A hard band dropped dim emitters whose widths are noisy rather than wrong.
On `hyp7gem_wt_01_crop_128x128`, 9.3 per frame of the emitters present
within 1.5 px at t-1 and t+1 were missing at t; at 39% of those the fit
was there and too narrow, at 31% too wide. Measured there (estimated
gain), per frame, with linking's continuation probability:

```text
rule             N     one-frame gaps    p_cont
hard band       180        8.8           .733
z = 1           217        9.0           .772
z = 2           235        7.3           .826
z = 3           238        6.8           .835
no band         240        6.3           .840
```

The added narrow fits above 150 e- persist into the neighbouring frames at
.91 against a chance rate of .28: real. The wide ones persist at .64, which
static haze pieces also would.

# Except at the upper slack bound

A width at `SLACK.1` is not a measurement: the fit wanted to be wider than
the model space allows. It is `Wide` whatever its SE says. It has to be,
because the SE cannot say it: `SLACK.1` is only 0.2 sigma above
`BAND.1`, and the joint Fisher matrix, with the level free, gives a wide
emitter a width SE of ~0.6 px (0.19 from a single-emitter Fisher). On GEM
frame 0, 28 of the 34 fits above the band sat on the bound, at z < 1.
Measured on the GEM referee (see [`ADD_NATS`]), on the port's own fits:

```text
rule                           FP    150 e-          300 e-          N
significance only              32    .51 .40 .31     .75 .71 .63     253
+ upper bound is wide          20    .50 .38 .29     .74 .71 .60     226
+ lower bound is narrow too    17    .47 .36 .28     .71 .68 .59     210
hard band                      15    .43 .33 .26     .67 .65 .56     187
```

The upper bound buys 12 false positives per frame for 1-3 recall points.
The lower one is not treated the same way: a dim emitter's width is noisy
enough to reach `SLACK.0` while the emitter is real, and flagging it costs
3-4 points of exactly the recall this band exists for.


## boxsearch: noise_map

The frame's own noise: `(sd, phi)`, both `h*w`. `sd` is each pixel's
standard deviation, ADU; `phi = sd^2 / local median of d`, the variance
per unit of signal. `(oy, ox)` is the array's origin in the frame.

`sd` comes from the separable 4th-difference product
`k (x) k`, `k = [1, -4, 6, -4, 1]`: on white noise its output is
`N(0, sd^2 * 70^2)`, and it nulls everything up to cubic along each axis,
so a PSF's curvature barely reaches it. The local median of its square
over `0.4549 * 70^2` (the median of chi-squared with one degree of
freedom) is the variance. It is taken on the VALID region only and the
last two rows and columns copied outward: any padding mode fakes
structure the filter reads as quiet, and measured on a sparse simulation
that put 23 of 61 false positives within 3 px of the border.

Measured 2026-09-14, `phi` against what the camera calibration implies at
the background (`gain + gain^2 read_noise^2 / background`):

```text
                                      expected   this    3-tap 2nd diff
pure noise, bg 1 / 5 / 20 e-          (exact)    +-1%        -16%
simulate, density .01 / .03 / .06       2.71    2.95 2.97 2.73
hyp7gem crops (4)                  2.55-2.85    2.42-2.61
beads_80pct-glycerol                    2.23    2.41
beads_60x_still / _02 (39, 62 px)  2.15 2.34    1.90 3.01     19x 14x
```

The 3-tap second difference (the high-pass the retired gain estimator
used) is biased on pure noise, because the two axes share their centre
pixel, and reads bead frames an order of magnitude high: crowded PSF
curvature. The denominator matters as much. Over the masked background
map instead of the local median of the data, `phi` climbed 3.24 -> 4.49
across the three densities, because the masked map reads low when little
is left unmasked.


## boxsearch: crop_margin

Pixels of context a crop needs outside the ROI for the answer inside it to
be the answer the whole frame would give.

Three supports reach in from the crop's edge, and the widest wins:

| stage | reach | at sigma 1.3 |
|---|---|---|
| [`find_candidates`] | `ceil(sigma)` (the max-filter window) + [`filters::kernel_radius`] | 7 |
| boxes | `ceil(BBOX_PAD * sigma) + 1` ([`patches::build_patches`]) | 5 |
| [`background_map`] | `2 * (BG_KERNEL / 2) + kernel_radius(BG_KERNEL / 6)` | 41 |
| [`noise_map`] | `2 + NOISE_WIN / 2 + NOISE_STRIDE` (filter, median, grid) | 26 |

The background chain sets it, at 41 px, and unlike the other two it does
not shrink with `sigma`: its kernel is a fixed 25 px. A candidate needs
its LoG exact over its whole max-filter window, and each of those needs
data out to the LoG's own radius, so FIND's two radii add rather than max.


## boxsearch: find_candidates

FIND against a flat `level`: LoG peaks of the residual above `threshold`
sds, brightest first. Returns `(pos 2N, amp, strength)`.

Two normalizations, doing different jobs. Dividing by the local pixel sd
`sd` ([`noise_map`]) makes one threshold valid across the FRAME, and
across cameras. Dividing by [`filters::log_kernel_l2`] makes one threshold
valid across SIGMA: the filter's null sd is its kernel's L2 norm, which
scales as sigma^-3, so without it one number is a 1.8-sigma cut at sigma
0.8 and a 100-sigma cut at sigma 3.0. `threshold` is therefore a count of
standard deviations.

The measurements below were made when the first normalization was
`sqrt(level)` in photoelectrons, before 2026-09-14.

# The LoG is a poor statistic, and the better one is worse here

Measured 2026-09-12, recorded so it is not re-attempted. The LoG is NOT
the matched filter for a Gaussian spot: `gaussian_laplace(sigma)`
correlates best with a spot of width `0.60 * sigma`, and against the
pixel-integrated PSF at its own sigma it delivers 0.69 of the z a matched
filter would on-pixel and 0.60 at the pixel corner -- at sigma 0.8 the
corner case falls to 0.43, so a sharper PSF buys the LoG nothing. A
difference of Gaussians, `g(sigma) - g(k*sigma)`, is that matched filter
with the background projected out -- the exact GLRT for unknown amplitude
on a smooth background, its null sd closed-form as `||K||_2` exactly like
the LoG's -- and reaches 0.95. As a SEEDER in isolation it is worth 1.3x
to 1.75x in flux: at a matched false-seed rate, recall at peak SNR 1.8
went 0.217 -> 0.477, and at sigma 0.8 0.467 -> 0.927.

None of that survives into the pipeline. Swapped in behind FIND's cut with
the placement cut pinned, each kernel bisected onto 15 false seeds per empty
128x128 frame so the operating points match (see the alpha note below),
six to twelve frames per cell, `band=None` so recall measures detection
and not classification:

```text
ISOLATED (density 0.002, 22 px spacing), recall
  peak SNR   1.40   2.01   2.81   4.01   6.02
  LoG        .090   .317   .740   .940   .960
  DoG k=2    .090   .323   .737   .937   .957

CROWDED (flux 200-600, matched width), recall / candidates per frame
  density   0.002      0.005      0.010      0.015      0.025      0.040
  LoG     .975  35  .925  64  .875 108  .820 144  .727 195  .615 236
  DoG k=2 .975  33  .915  60  .856 100  .805 133  .684 165  .565 188
  DoG k=3 .975  32  .911  59  .837  96  .787 126  .657 151  .531 166
```

Two things are happening. Where emitters are ISOLATED the seeder is not
what limits recall -- `ADD_NATS` is. The DoG seeds strictly more (28
candidates against 22 at peak SNR 1.4) and every extra one fails to pay
its 10 nats, so a better statistic buys exactly what a looser cut buys,
which is nothing. Where emitters are CROWDED the DoG's broader core
merges neighbours inside the max-filter window, and a box that never
forms cannot be recovered.

So the LoG's 29% efficiency loss is the price of a narrow core, and the
narrow core is worth more than the efficiency. A better FIND has to
SEPARATE better, not detect better; sensitivity is `ADD_NATS`' problem.

# What the old Bonferroni cut actually bought

Under the Poisson normalization the cut was not the achieved rate. `level`
is the 10th percentile, deliberately -- a background estimate under
emitters -- but it was also the variance normalizer, and too low a
normalizer inflates the z: on emitter-free Poisson frames the normalized
residual had sd 1.19 at a background of 20 e- and 1.07 at 100 e-, not 1.
The derived cut of 3.79 therefore passed 17.9 false seeds per empty
128x128 frame, not the 0.05 per frame it named. The local sd does not
share that bias, which is one reason [`PEAK_Z`] was re-measured rather
than carried over. A cut compared across two different
statistics must be calibrated empirically: at one nominal z the DoG passed
10.8 seeds against the LoG's 17.9, and comparing them there compares the
calibrations, not the filters.


## boxsearch: background_map

The smooth background surface, ADU: a local mean over the pixels no
candidate reaches, taken twice with a one-sided clip at 3 local sds
between, then smoothed.

Estimated from masked DATA, never from the PSF-subtracted residual: that
route is a feedback loop, because emitters that have absorbed background
depress the residual, the surface follows them down, and they must absorb
more. An undetected emitter is not masked, so the clip removes the upper
tail only; its contamination is always positive, and clipping both tails
biases the estimate down.

Windows with fewer than `BG_MIN_PIXELS` free pixels take one frame-wide
scalar: at high density that is most of them, and the surface degrades to
a scalar rather than to an average over four pixels. The scalar is the
median of the free pixels, falling back to the 10th percentile when too
few are free. Neither alone serves both regimes: the image median is no
background on a crowded field (99 ADU against a true 12-19 on
beads_60x_still), and a low quantile is none on a sparse one (9.7 against
a true 20 on an emitter-free frame).

`roi`, if given, is `h*w` and confines those two SCALARS -- and nothing
else -- to the pixels it selects. The surface itself is estimated from
every pixel, because a window straddling the ROI's edge is entitled to
the real data on both sides of it. The scalar is the one number the whole
crop can fall back to, so it has to describe the region asked about: on
`hyp7gem_wt_crop` the frame's own 10th percentile is 11.1 e-, which is the
dark field OUTSIDE the cell, against 18-20 inside it. Confining it also
makes it independent of how much margin the crop carries, which a
crop-wide statistic is not.

Returns the surface and the fallback level, which is what a caller
scattering this back into a larger array should fill the rest with.


## boxsearch: placement

`(y, x, A0)` of the strongest owned LoG peak of the fit's residual, in
local sds, that passes [`Settings::threshold`], or `None`.

The test is not optional. The deviance of the best of many placements is
a maximum over positions, and `ADD_NATS` was measured only on placements
that had already been screened; without the screen every bright emitter
collected faint satellites on its wings (130 emitters for 107 true on
seed 17, precision 0.74).

# The LoG stays, though it is the wrong statistic on paper

Inside a box the argument FIND lost (see [`find_candidates`]) looks
stronger. The level is one free constant, fitted, so the exact GLRT for
one more emitter at a grid position is the pixel-integrated PSF matched
filter with that constant projected out under the Poisson weights `1/m`.
It lost anyway, and in BOTH of this function's jobs: deciding whether to
place, and where.

Measured 2026-09-12 on the nine placement-cut arms (see [`PEAK_Z`]; same
seeds, 1 px match, `band=None`), each statistic's cut swept 5.0-1.5 on its own scale
and taken at its own minimax cut. Mean / worst-arm F1 regret is against
each arm's best over every row and cut. Then the paired F1 change against
the LoG at 3.0, per arm (bright sparse, mid, dense, very dense; faint mid
matched, spread, bright bg; sigma 0.8 faint; sigma 2.0 bright):

```text
gate     position  cut     mean   worst   per-arm dF1 vs LoG
LoG      LoG        3.0   .0083   .0149   (reference)
MF       MF         3.5   .0362   .0765   -.006 -.025 -.030 -.028 -.062 -.030 -.028 -.009 -.033
MF+proj  MF+proj    4.0   .0395   .0812   -.008 -.026 -.034 -.032 -.066 -.031 -.027 -.010 -.047
MF+proj  LoG        3.0   .0225   .0374   -.006 -.019 -.020 -.015 -.020 -.010 -.007 -.007 -.024
LoG      MF+proj    3.5   .0229   .0483   -.001 -.007 -.020 -.022 -.021 -.010 -.010 -.005 -.035
```

MF is the pixel-integrated Gaussian at `sigma`, truncated at the window
and normalized by its weighted norm; "+proj" projects the level out.
Projecting the level out did not help. No arm gains at any row's cut, and
with a free cut per arm the best any matched-filter row does is tie. The
two split rows show where the loss comes from:

* As a POSITION PICKER (LoG gate, MF peak) recall is unchanged and
  precision falls with the gate: .913 -> .879 at 3.0 on bright mid, .732
  -> .608 at sigma 2.0, the arms where widths are mis-modelled. That is
  consistent with the broad core peaking on the smooth residual a
  width-mismatched bright emitter leaves on its wings, where the start
  then pays `ADD_NATS` as a satellite; the LoG's negative surround
  rejects smooth residual. The satellites were not traced one by one.
* As a GATE (MF z, LoG peak) precision is below the LoG's at EVERY cut on
  the bright arms, even at 5.0 (.916 against .957 on bright mid), so no
  constant recovers it. The bright arms want a statistic that ignores
  smooth residual, not a higher cut on one that sums it.

Even where the residual is closest to one emitter plus noise -- faint, or
matched width -- the matched filter at best ties (.905 against .905 on
faint matched, .982 against .986 on bright sparse). Its optimality holds
for a residual the search never sees. The LoG's band pass is rejecting
model mismatch, which the GLRT does not. The
0.60-0.69 efficiency the note at [`find_candidates`] measures is the price.

The tests here and in [`search_box`] are written `!(a > b)` on purpose: a
NaN must fail them.


## boxsearch: localize

Localize one frame of `d_e`. `roi`, if given, is `H*W`: candidates
outside it are dropped after the background map is built from all of
them, and no box places outside it.

# The crop

An ROI confines the search, so the whole-frame preamble -- FIND, the
background surface, and the level they are measured against -- runs on
the ROI's bounding box plus [`crop_margin`] rather than on the frame.
Nothing else moves: positions come back in global coordinates and the
background is scattered into an `H*W` map, so boxes, the halo, the polish
and the edge classification all still see the whole frame.

It is worth doing because that preamble is a third of the frame's work and
none of it depends on N -- on a 256^2 frame, 5.1 ms of 15.3 ms
(`percentile` 0.33, FIND 1.64, background 3.16); on 512^2, 23.0 ms of
67.4 ms. `background_map` is the larger half and does not shrink with
`sigma`: its kernel is a fixed 25 px. Measured end to end on
`hyp7gem_wt_crop`, a centred square ROI:

| frame | ROI | crop | before | after |
|---|---|---|---|---|
| 256^2 | none | -- | 50.1 ms | 50.5 ms |
| 256^2 | 32^2 | 114^2 | 6.09 ms | 1.75 ms |
| 256^2 | 64^2 | 146^2 | 8.27 ms | 4.67 ms |
| 256^2 | 128^2 | 210^2 | 23.4 ms | 18.3 ms |
| 512^2 | 32^2 | 114^2 | 25.4 ms | 3.43 ms |
| 512^2 | 64^2 | 146^2 | 29.1 ms | 8.07 ms |
| 512^2 | 128^2 | 210^2 | 38.1 ms | 18.7 ms |

The win is what the ROI throws away, so it grows with the frame and
shrinks as the ROI approaches it; at `roi = None` the crop is the frame
and nothing changes, bit for bit (asserted across three images and three
sigmas, positions, amplitudes, widths and the whole background map). An
ROI whose bounding box is the frame -- scattered cells, a diagonal band --
buys nothing and costs nothing.

The margin makes the crop invisible to everything inside the ROI:
measured, the same ROI given more surrounding frame than the crop needs
returns the same N with positions agreeing to 1.3e-12 px, which is filter
summation order over a differently-shaped array. What the ROI does change
-- deliberately -- is `b0` and the background's fallback, now measured
over the ROI's own pixels rather than the frame's; see [`background_map`].
On `hyp7gem_wt_crop` that moved N by up to 5% on the larger ROIs (78 to
74 at 128^2), because the frame's 10th percentile is the dark field
outside the cell.


## boxsearch: polish

Block-Jacobi refits at fixed N until nothing moves, with free widths, and
per-emitter CRLBs from the Fisher matrix of the fit whose parameters are
reported.

Jacobi, not Gauss-Seidel: every patch in a sweep reads the sweep's INPUT
state, so the answer does not depend on visit order. Iterated, not run
once: each patch holds its neighbours frozen where it was handed them, so
one pass propagates their staleness. Measured on isolated emitters at
density 0.055, a 0.5 px error in the NEIGHBOURS alone (target started at
truth) takes the pull sd from 0.96 to 1.98; sweeping recovers it to 1.27.
The patches are rebuilt each sweep, which refreshes the halo -- so the map
is not continuous, and a fixed point need not exist (PORTING_NOTES 20).

Scheduled group-wise: a patch none of whose free or frozen emitters moved
more than `POLISH_MOVE_TOL` last sweep is skipped. It replaces a global
`max |dpos| < tol` break that could not fire (on a 906-emitter frame, 80%
of emitters still moved past it after eight sweeps). Do not expect the
queue to drain on a crowded field either: halos overlap and dirtiness
percolates from a few degenerate groups, collapsed pairs 0.001 px apart
that never converge. Neither Gauss-Seidel nor pinning the decomposition
fixes that (both measured), so `POLISH_SWEEPS` bounds it.


## lmcl: Overview

Bounded Levenberg-Marquardt with Coleman-Li affine scaling.

Ported from the retired Python reference's `lmga.py` (last present in
`ea6b17f`). **The name differs deliberately**: `lmga` meant "LM with
geodesic acceleration", and the geodesic acceleration was implemented,
measured and removed long ago -- it changed the converged objective by under
1e-13 on ordinary fits while costing 133 model/Jacobian evaluations per fit
instead of 18. What actually distinguishes this optimizer is the Coleman-Li
affine scaling, so it is named for that. The `02_lmga` fixture keeps its
name; it is the same code.

# Objective

The Poisson I-divergence

```text
I(d, m) = sum_i [ d_i*log(d_i/m_i) - (d_i - m_i) ]      (0*log(0) := 0)
```

minimized by Fisher scoring: `W = diag(1/m)` is held fixed within an
iteration, so `F = J^T W J` is the *expected* Fisher information -- exact
for Poisson's canonical link, where Fisher scoring coincides with IRLS.

# Strict interiority is a precondition, not a preference

Coleman-Li divides by `v_i`, the distance from parameter `i` to the bound
its step is heading toward. A parameter resting exactly *on* a bound sets
`v_i = 0`, and the damage is not local to it: a fraction-to-boundary rule
that computes one scalar step scale from `min_i (bound_i - theta_i)/delta_i`
lets a single stuck coordinate collapse the step for **every** coordinate
(which is why [`scale_into_box`] now limits each coordinate alone). The
step then buys ~1e-11 nats, its gain ratio reads ~2e-8 -- which measures the
clipping, not the quadratic model -- so it is rejected, lambda ratchets up,
and the fit burns its whole budget on micro-steps.

Measured on a 39x39 frame before this was fixed in the Python: 68.6% of LM
iterations began with a parameter on a bound, 46.5% of inner trials were
scaled to the 1e-8 floor, and 73.2% of all fits exhausted `max_iter`.
Removing the clip made the frame solve 2.5x faster *and* made the emitter
count reproducible.

The Python documented this invariant in its module docstring and violated it
in its body for months. Here [`Interior`] makes the broken state
unrepresentable: there is no way to obtain one that is not strictly inside
its box, and no `&mut [f64]` escapes [P1].


## lmcl: FitWorkspace

Everything one fit needs, allocated once and borrowed for the duration.

The Python's LM inner loop allocates two dense `p x p` matrices per lambda
trial -- whose off-diagonals are known to be zero -- purely to write
`F + diag(u) + lam*diag(v)`, and there are ~700k such trials per frame [P6].
It also re-derives the separable pixel axes and the I-divergence's data-only
terms on every one of the ~160 model evaluations a fit makes [P5]. All of
that lives here instead.

`ensure` grows the buffers on demand, so after the first few patches of a
run `fit` allocates nothing at all.


## lmcl: fit_var_sigma

Fit a variable-sigma theta `[b, A0, y0, x0, sigma0, ...]` on a `h x w`
patch by bounded Fisher-scoring LM: the Poisson maximum likelihood.

`d` is the patch data in photoelectrons, row-major. `halo` is the
parameter-free additive contribution (frozen neighbours plus the
background's shape term), or `None`. Results are left in the workspace:
[`FitWorkspace::theta`] and [`FitWorkspace::fisher`].

Each step is assessed on the actual feasible quadratic step, and
convergence is certified by an information-scaled projected gradient. The
trajectory intentionally need not follow the Python reference's.

# Why `max_iter` must not be cut for speed

Truncating iterations does *not* add symmetric noise. A proposal fit starts
further from its optimum than the incumbent it is compared against, so
truncation systematically leaves the proposal's objective too high and
biases model selection toward the smaller model -- it under-credits exactly
the moves that add an emitter. Measured: capping at 40 iterations still left
21% of fits more than 0.1 nat above their optimum, with a p99 gap of 72
nats. Make each iteration cheaper instead.


## lmcl: fisher

`F = J^T W J` with `W = diag(1/m)`, into `ws.f`.

Only the upper triangle is computed and then mirrored. `F` is symmetric by
construction, and computing both halves independently is not merely wasted
work: it produces `F_ij != F_ji` in the last ulp, because the two are
separate reductions over different orders. That is the whole reason [P3]
exists. Mirroring makes the matrix exactly symmetric, so
`Chol::factor`'s symmetrization becomes a no-op and the question of which
triangle gets read cannot arise at all.

This is the dominant arithmetic in the fit: `p^2 * n` per outer iteration
against `p*n` for everything else. The Python reaches it through BLAS
`dgemm`, so it is the one place the port does NOT start ahead.


## lmcl: Fisher loop

Four columns of `wj` per pass over `c1`.

**This does not reassociate anything** [P2]. Each `s*` accumulates one
output element over `i` in the same increasing order the scalar loop
used, so every individual sum is bit-identical; what changes is only how
many *different* sums are in flight. A lone dot product is bound by the
~3-cycle latency of the dependent `FADD`, not by throughput, so it
retires one add per three cycles however wide the machine is. Several
independent chains fill those slots with useful work.

Measured on frame 0 of `beads_80pct-glycerol_crop.tif` (1141 emitters,
14k window fits), median of 9 runs, interleaved builds, `positions /
amplitudes / se` SHA equal for every arm:

```text
  scalar    2.844 s  2.888 s
  width 2   2.531 s
  width 4   2.541 s  2.568 s     <- 11.2% under scalar
  width 8   2.637 s
```

2 and 4 tie; 8 regresses, because `p` is 5 at the median (`K = 1`) and
only reaches 37, so a width-8 block almost never fills and the work
falls through to the scalar tail with the wider prologue already paid.
4 is kept over 2 for the larger windows, where it has the longer runs to
amortize. This buys nothing on its own if `p` is small -- see §4: at
these sizes the ranking is not the FLOP count.


## lmcl: scale_into_box

Keep `theta + delta` inside the box, one coordinate at a time: a
coordinate whose step would cross a bound moves 0.995 of the way to it,
and every other coordinate keeps its step.

It used to shrink the WHOLE step by one scalar -- the same collapse the
module note describes for a parameter on its bound, one level milder: a
width creeping toward `SLACK.1` held every other parameter to its creep.
Measured 2026-09-14 on 10 GEM frames (256x256), 77k fits: 22% ended with
a width at a bound, averaging 53 iterations against 10 for an interior
single emitter, and they were 9030 of the 10168 fits that exhausted
`max_iter` -- about 56% of all iterations. Per coordinate, on 200 noisy
13x13 fits of each kind (identical data and starts):

```text
case                      old: maxed out   dI new - old   new iterations
wide single (3.6 px)          97/200        -0.17 nats         12.1
wide + point emitter         129/200        -0.41 nats         12.0
in-band single, narrow         0/200         0 (identical)   5.5, 7.5
```

Never a worse optimum, and in the detector 26% faster (127 -> 94 ms per
128x128 GEM frame; fits exhausting `max_iter` 10168 -> 297 on the 256x256
frames). Because a truncated fit under-credits the larger model (see
[`fit_var_sigma`]), converging them moves decisions: on the GEM spike-in
referee false positives went 20 -> 22 per frame and 300 e- recall
.74/.71/.60 -> .77/.73/.61, N 225 -> 233.

The step stays strictly interior, and the gain ratio is taken on this
feasible step, not on the Newton step it came from.


## lmcl: Damped system

Coleman-Li form:
  (F + diag(|grad|/s^2) + lam*diag(1/s^2)) delta = -grad

The scaled-space system is (D F D + diag(|grad|) + lam I) shat =
-D grad with D = diag(s), s = sqrt(v). Mapping back to the
unscaled step delta = D shat sends BOTH extra diagonals through
D^-1 (.) D^-1, so both pick up 1/s^2 -- not 1/s for one of them.
The mismatched version under-damps every parameter approaching a
bound (at v = 0.01 it applies 10|g| where the correct term is
100|g|). Over 200 randomized 1-3 emitter fits the consistent form
reached a lower converged I 6 times to 1 with 193 ties, better by
3.5 nats on average -- large on the scale a Bayes factor is
decided on -- in 10.9 iterations against 20.4.

## lmcl: Interior

A parameter vector guaranteed strictly inside its box.

The constructors are the only way in and every one of them pulls the value
inside, so the optimizer cannot express the state that broke it [P1].
Deliberately exposes no mutable slice.


## lmcl: Interior step

`self <- interior(base + delta)`. The only way to advance an iterate.

NOT a clamp onto the bounds: clamping parks a parameter exactly on one,
which is precisely the state this type exists to prevent.


## lmcl: Interior reuse

`self <- interior(theta)`, reusing this vector's allocation.

`fit` is called on the order of 700k times per frame, so even a
once-per-fit allocation is worth not making [P6].


## lmcl: Objective tolerance

Objective resolution, **in nats**: a fit has converged when its
information-scaled projected score is <= sqrt(2*tol_obj).

The only test in units the caller cares about: decisions compare
I-divergences, so "this fit cannot improve I by more than `tol_obj`"
is a statement about the decision, not the parameterization. Before
it existed, with only the absolute `tol_grad` and a step test, 34.5% of
the 9400 patch fits on a bead-matched 39x39 field exhausted
`max_iter = 100` and only 62.7% reported convergence.


## lmcl: Gain ratio

LM gain ratio: how much of the promised improvement was real.
lambda MUST be driven by this and not by the sign of the
improvement alone. Accepting any decrease and halving lambda for
it lets lambda collapse to its floor while the quadratic model is
worthless, and then nothing damps the near-null directions of F.
Traced on a real patch (K=7, one emitter at the amplitude floor
so cond(F) = 1.5e20): every iteration predicted 5.2e4 nats,
delivered 1.05e-3, halved lambda anyway, and took the identical
2.3e-5 step again -- 3000+ iterations, finishing 364 nats above
the optimum.

## filters: Original overview

Separable image filters, matching `scipy.ndimage` exactly.

Four calls in the pipeline sit on top of one scaffold: a 1-D pass along
each axis, with a boundary rule. `find_candidates` needs
`gaussian_laplace` and `maximum_filter`; `placement` needs
`gaussian_laplace`; `background_map` needs `uniform_filter` and
`gaussian_filter`.

| call | site | kernel / reducer | mode |
|---|---|---|---|
| [`gaussian_laplace`] | FIND, BIRTH | Gaussian order 2, summed over axes | reflect / nearest |
| [`maximum_filter`] | FIND | sliding max | reflect |
| [`uniform_filter`] | background | box | nearest |
| [`gaussian_filter`] | background | Gaussian order 0 | nearest |

# The one convention that must not be "fixed"

[`gaussian_kernel1d`] normalizes the **order-0** kernel to sum 1 and only
then applies the derivative recurrence. A truncated order-2 kernel
therefore does **not** sum to zero -- at `sigma = 0.6, radius = 2` it sums
to `-6.5e-2`. That looks like a bug and is not: re-normalizing it rescales
the entire LoG response, which is then compared against a *fixed* threshold
of 1.5, and the candidate list changes. `tests/fixtures/07_filters.json` pins the
kernels as impulse responses precisely so this cannot drift.

# Why matching scipy bit-for-bit is not required here

It is required of the *kernels*, and the fixture asserts them to 1e-12. It
is not required of the filtered image, because of what consumes it: on a
512x512 frame the weakest accepted candidate scores 1.5515 against a
threshold of 1.5 and the strongest rejected scores 1.4948, a margin of
5.7e-2, and no accepted candidate has an exact tie in its max-filter
window. The candidate list has about eleven orders of magnitude of slack
over the difference any correct summation order could produce.
