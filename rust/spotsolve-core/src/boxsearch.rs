//! Box-local localization: in each box, the best fit the data support wins.
//!
//! Ported from the Python reference `box.localize_boxes`, retired on
//! 2026-09-11 (last present in commit `ea6b17f`, `src/spotsolve/deprecated/`).
//! At retirement the two agreed exactly -- counts, recall and precision -- on
//! the referee cells and on both real 256x256 frames. The measurements that
//! set each constant are recorded beside it; the ones that shaped the design
//! as a whole are below.
//!
//! ```text
//! d_e      (raw - offset) / gain + read_noise^2             shifted Poisson
//! FIND     LoG peaks of d_e against a flat level            -> candidates
//! BMAP     smooth surface from the pixels no candidate reaches
//! BOXES    candidates within LINK_FACTOR*sigma share a box, <= k_max each
//! SWEEPS times, for each box, brightest first:
//!     fit the level alone (K = 0), BMAP's shape and the neighbours' current
//!     emitters held fixed
//!     FORWARD   place at the strongest owned residual peak that passes
//!               the BIRTH cut; keep it iff I falls by > ADD_NATS
//!     BACKWARD  (K >= 2) drop the cheapest emitter while it costs < ADD_NATS
//! POLISH   block-Jacobi refits at fixed N until nothing moves
//! ```
//!
//! Each emitter is decided in the box that owns it, by one comparison rule.
//! There is no split move, no separate removal pass, no forced-removal
//! threshold, no conditioning guard and no width prior: a collapsed or
//! redundant emitter explains almost no deviance, so it cannot pay
//! `ADD_NATS` on the way in and costs almost nothing on the way out.
//!
//! # The gain is a dispersion parameter, not a scale
//!
//! Everything here works in photoelectrons, `d_e = (adu - offset) / g`; the
//! Poisson weighting `W = 1/m` is only valid there. Three exact identities
//! (verified numerically to machine precision):
//!
//! * the I-divergence is homogeneous, `I(d/g, m/g) = I(d, m) / g`;
//! * so the FIT does not depend on `g`: positions are invariant and
//!   amplitudes scale as `1/g`. Measured over g = 1..50, positions agreed
//!   to 7e-7 px and amplitudes to 7e-7 relative;
//! * `log|F| = (1 - K) log g + log|F_ADU|`.
//!
//! The whole gain dependence of the decision -- keep an emitter iff `I`
//! falls by more than `ADD_NATS` -- is the `1/g` on the data term:
//! `dI_ADU / g > ADD_NATS`. The gain moves that threshold and nothing else.
//! It is a property of the CAMERA, not of the field: measure it once and
//! pass it in. [`estimate_gain`] is the fallback.
//!
//! # Read noise: the shifted-Poisson approximation
//!
//! A pixel's variance is `m + sigma_r^2`, not `m`. Adding `s = sigma_r^2`
//! (e-^2) to both data and model gives a Poisson likelihood with exactly
//! that variance and mean `m + s`. Here it is a shift of `d_e` alone
//! ([`localize_raw`]): every box carries a free constant background, which
//! absorbs `s`, and FIND's `sqrt(model)` normalization then divides by the
//! right standard deviation. The caller takes `s` back off the reported
//! background and model.
//!
//! Without it, at low background a read-noise spike reads as a significant
//! single-pixel source. Measured 2026-09-10 on 64x64 `simulate` frames with
//! Gaussian read noise added: false detections per frame on EMPTY frames (six
//! seeds), then recall / precision at density 0.015, flux U(150, 500) e-
//! (three seeds):
//!
//! ```text
//! bg e-  sigma_r   empty: Poisson  shifted    Poisson        shifted
//!  1.0     1.6             1.2       0.0     .837 .937      .830 .983
//!  1.0     2.5            13.3       0.0     .823 .811      .823 .983
//!  3.0     2.5             5.7       0.0     .837 .922      .816 .975
//! 10.0     2.5             0.5       0.0     .773 .965      .730 .963
//! ```
//!
//! With no read noise the two are the same pipeline. The recall the shifted
//! model gives up where `sigma_r^2` rivals the background is not yet traced
//! to individual emitters.
//!
//! # Background: a smooth map, not a plane
//!
//! [`background_map`], a masked 25 px local mean, is each box's known shape,
//! with the box's level free. It was chosen over a plane per box (level and
//! slopes free) and over a free constant alone. Measured 2026-09-11 on the
//! referee frames (64x64, flux U(900, 1900) e-, bg 20 e-, seeds 17-19),
//! recall / precision / invented per frame. Haze `L, H` is white noise
//! Gaussian-filtered at L px, scaled to span [0, H] e-:
//!
//! ```text
//! cell               constant           plane              map
//! flat  0.015 0.4    .702 .980  0.7     .660 .949  1.7     .695 .970  1.0
//! flat  0.055 0.4    .424 .855 12.3     .426 .856 12.3     .438 .873 11.0
//! haze  L15 H60      .701 .957  3.3     .707 .978  1.7     .729 .992  0.7
//! haze  L5  H60      .682 .952  3.7     .670 .960  3.0     .704 .966  2.7
//! haze  L15 H20      .710 .983  1.3     .698 .978  1.7     .698 .961  3.0
//! ```
//!
//! The plane's two extra parameters cost sparse fields; the map costs no fit
//! parameters or time. Weak haze (H20) is the map's one worse cell, by 3-6
//! events over three frames. Re-estimating the map from the fitted emitters
//! after the first sweep moved nothing beyond seed noise. A per-pixel
//! likelihood mask and a rank-opening haze map were also measured and
//! rejected.
//!
//! # No width prior
//!
//! Every fit is a plain ML fit over the width bounds `slack`. The retired
//! `detect` pipeline's MAP width penalty pulled every width toward sigma0; in
//! this search that leaves the wings of a broad emitter unexplained, and the
//! next placement lands on them as a faint satellite. Removing it, six frames
//! per cell, recall / precision / tiles per frame:
//!
//! ```text
//! density spread     MAP width           flat width
//!  0.015   0.4    .876 .893  3.3    .866 .940  1.7
//!  0.034   0.2    .825 .929  5.7    .813 .949  4.2
//!  0.034   0.4    .783 .823 11.5    .776 .858  9.0
//!  0.055   0.2    .745 .908 10.3    .737 .941  6.7
//! ```
//!
//! Widths always float. A fixed-width mode was measured and rejected on
//! 2026-09-11: it tiled haze and defocused spots, inventing 8-33 spots per
//! 64x64 frame against 0.7-7 with fitted widths.
//!
//! # Where Python spent the time, and why this module exists
//!
//! On frame 0 of the glycerol crop (485 emitters, 0.84 s in Python) the
//! native fits were 0.10 s. The rest was the search around them: the polish
//! (0.32 s), a scipy LoG per placement (0.21), ownership distances over every
//! candidate (0.14), gathering every box's emitters into each halo
//! (0.14, quadratic in boxes) and Python PSF renders (0.13). Here ownership
//! and the neighbour lists are computed once per frame, and a halo reads only
//! the boxes that can reach it.

use crate::filters::{self, Mode};
use crate::grid::EmitterGrid;
use crate::linalg::Chol;
use crate::lmcl::{self, Bounds, FitOpts, FitWorkspace};
use crate::patches::{self, HALO_FACTOR};
use crate::psf;
use crate::render;
use crate::statistics;

/// Widths a fit may take, as multiples of `sigma`: the MODEL SPACE.
///
/// The model space has to cover every photon on the sensor, or the light it
/// cannot represent gets tiled: a fixed-sigma model meets anything out of
/// focus with two narrow Gaussians, which genuinely do fit a wide blob better
/// than one. Measured on a confocal simulation with emitters uniform in
/// +/- 0.5 um, at 1 emitter/um^2 the fixed-sigma search returned 7.2 extra
/// detections per frame against 13.4 in-focus emitters, every one within
/// 3 sigma(z) of a real emitter.
///
/// `SLACK.0` sits below `BAND.0` so that a broken fit -- nothing images
/// narrower than the PSF -- reveals itself instead of being clipped to the
/// bound and reported. The upper edge, measured under the retired `detect`
/// (moderate arm, 6 frames, band fixed at (0.8, 2.0)):
///
/// ```text
///  hi    recall   med err    RMSE   rsd z   |z|>3   tiles
/// 2.2     92.6%     0.076   0.220    1.25    7.0%    2.33
/// 2.6     91.1%     0.078   0.201    1.31    8.0%    2.33
/// 3.2     90.8%     0.078   0.183    1.27    8.7%    4.00
/// 4.0     90.5%     0.078   0.194    1.32    8.1%    2.50
/// ```
///
/// Recall falls monotonically past 2.2: a larger model space buys better
/// parameters for the objects it keeps and swallows close neighbours. The box
/// search re-measured it: a bound of 4-6 sigma lost 3-5 recall points under
/// haze and, on a GEM frame, swallowed emitters (N 440 -> 271). 2.2 stands.
pub const SLACK: (f64, f64) = (0.70, 2.2);
/// Widths reported as detections, as multiples of `sigma`: the REPORTING
/// BAND, a downstream contract about what the caller is handed. Its upper
/// edge is how far out of focus an emitter may still be reported: a point
/// source images at 1.26x the in-focus width at |z| = 0.25 um, 1.95x at 0.35
/// and 3.2x at 0.50, so 2.0 is |z| < ~0.36 um.
pub const BAND: (f64, f64) = (0.80, 2.0);
/// Most emitters one box fits jointly. `patches::K_MAX`.
pub const K_MAX: usize = crate::patches::K_MAX;

/// Nats of I-divergence an emitter must explain to exist: the whole decision
/// rule.
///
/// Measured 2026-09-10 by replacing every Laplace Bayes factor in the
/// retired `detect` with `dI - c` (its conditioning guard, prune threshold
/// and MAP width fit unchanged), 64x64 `simulate` fields, three seeds per
/// cell, recall / precision / tiles per frame:
///
/// ```text
/// density spread     Laplace BF          c = 10            c = 14
///  0.015   0.4    .888 .920  2.3    .888 .938  2.0    .898 .969  1.0
///  0.034   0.4    .832 .807 13.3    .815 .868  8.3    .805 .899  6.0
///  0.055   0.4    .721 .797 20.0    .704 .833 15.3    .688 .867 11.3
/// ```
///
/// c = 10 matched the Bayes factor within seed noise in all six cells; c = 14
/// trades 2-3 recall points for about half the tiles. The Bayes factor's
/// priors, log-determinants and empirical-Bayes rates were an operating point.
pub const ADD_NATS: f64 = 10.0;
/// sigma. A box places only on pixels this near one of its own candidates,
/// and nearer to its own than to any other box's. The nearest-candidate rule
/// stops two boxes claiming the same light; the radius stops a box reaching
/// across empty space. Measured on six 64x64 frames at density 0.034, spread
/// 0.2 (one sweep, MAP widths):
///
/// ```text
/// radius    recall   prec   tiles/frame
///   2.0     .783     .890      8.2
///   3.0     .823     .886      8.8
///   4.0     .842     .868     10.8
///   inf     .844     .863     11.2
/// ```
///
/// At 2.0 a partner 2-3 sigma from the candidate it hides behind was outside
/// the box's reach -- those pairs were 31 of the misses, against 3 for
/// `detect`. Past 3.0 recall rises about as fast as tiles do.
pub const OWN_RADIUS: f64 = 3.0;
/// Passes over every box. A box decides against its neighbours in the halo,
/// and on the first sweep an undecided neighbour is only its FIND seed. A
/// mismatched seed leaves light the current box claims, and the neighbour's
/// source ends up split across two boxes. The second sweep re-decides every
/// box from K = 0 against neighbours that have all been fitted. Same six
/// frames as `OWN_RADIUS`:
///
/// ```text
/// sweeps    recall   prec   tiles/frame   search+polish fits
///   1       .823     .886      8.8             504
///   2       .821     .927      5.8             725
///   3       .827     .935      5.5             942
/// ```
pub const SWEEPS: usize = 2;
/// nats. A search fit only has to resolve `dI` against `ADD_NATS`. Measured
/// 2026-09-11 against 1e-8 and 1e-4 on the eight referee cells (flat and
/// haze, three seeds): recall and precision agreed to within one emitter per
/// cell at all three. On the real frames, 1e-8 moved positions by 0.006 px
/// (glycerol f0) and 0.036 px (GEM f0), and 1e-4 by 0.03-0.04 px.
pub const FIT_TOL_OBJ: f64 = 1e-6;
pub const FIT_MAX_ITER: usize = 100;
/// Most sweeps of the polish; see `polish` for why the queue does not
/// drain on its own.
pub const POLISH_SWEEPS: usize = 4;
/// LM iterations one polish fit may take. A BUDGET, not a convergence
/// criterion.
///
/// A few groups per frame never converge: they are not stalled -- they keep
/// taking accepted steps to the cap -- because they are descending a
/// direction the data carries almost no information about. At 256x256 they
/// are ~1% of emitters and 10-19% of the polish's LM iterations, and they are
/// NOT disposable: 36 of 38 of their emitters survive to the output. What
/// that descent is worth is boundable: a decrease of `t` nats moves a
/// parameter about `sqrt(2t)` standard errors, and each iteration continues
/// only while it predicts more than `POLISH_TOL_OBJ`. Measured under the
/// retired `detect` at 256x256, against the same fit run to 400 iterations:
///
/// ```text
///   max_iter   frame s        N   audit   max |dpos|/SE
///         25      7.24     3166   clean          0.0820
///         50      7.80     3166   clean          0.0023   <- here
///        100      8.19     3168   clean          0.0006
///        200      8.73     3169   clean          0.0077
///        400      9.00     3168   clean          0.0006
/// ```
///
/// N wanders by +-3 at EVERY budget, 400 included: the usual churn at
/// marginal decisions, not a signal. 50 is where the agreement band is
/// tightest for the least work; on a sweep with no stuck group it is
/// bit-identical to a budget of 4000.
pub const POLISH_MAX_ITER: usize = 50;
/// nats of predicted decrease. Near the optimum
/// `I(t) ~ I_min + 0.5 dt' F dt`, so stopping at a predicted decrease of
/// `tol` leaves a parameter about `sqrt(2 tol)` standard errors short.
/// Measured under the retired `detect` (256 px, N = 3169, Python-vs-Rust
/// band in SE):
///
/// ```text
///   tol_obj    frame s        N     max |dpos|/SE
///     1e-8        9.78     3169            0.0010
///     1e-6        8.66     3169            0.0015   <- here
///     1e-5        7.68     3168            0.1466
///     1e-4        7.08     3161            0.2828
/// ```
///
/// 1e-6 is the last value that leaves N and the agreement band where 1e-8
/// does.
pub const POLISH_TOL_OBJ: f64 = 1e-6;
/// px. An emitter that moved less than this in a polish sweep does not
/// dirty the groups that read it.
pub const POLISH_MOVE_TOL: f64 = 1e-3;
/// Side, px, of the window [`background_map`] averages over.
pub const BG_KERNEL: usize = 25;
/// e-. `W = 1/m` is singular at `m = 0`; this is far below one photoelectron.
pub const BG_FLOOR: f64 = 1e-3;
/// sigma. Emitter support excluded from the background estimate.
pub const BG_MASK_RADIUS: f64 = 3.0;
/// Unmasked pixels a window needs before its local mean is believed.
pub const BG_MIN_PIXELS: f64 = 25.0;
/// BIRTH's cut, in sd of the LoG null: how strong a residual peak must be
/// before a placement is even tried. Not derived from the frame -- the test
/// is inside a box, and a box does not get bigger when the frame does.
///
/// It used to share [`seed_threshold`]'s frame-wide value (3.8 to 4.1 on the
/// sizes here), which is a Bonferroni over ~1820 independent windows. A box
/// holds about four. That number was inherited, not derived, and it was
/// costing recall that `ADD_NATS` would have rejected anyway.
///
/// Measured 2026-09-12 with `seed` pinned at its default, nine arms on
/// `simulate` truth, twelve seeds each, `band=None` so recall measures
/// detection and not width classification, and a detection matched within
/// 1 px. F1, and below it the regret against each arm's own best:
///
/// ```text
/// arm                          4.0     3.5     3.0     2.5     2.0     1.5
/// bright sparse, matched      .983    .985    .986    .985    .983    .980
/// bright mid, spread          .912    .913    .908    .903    .890    .884
/// bright dense, spread        .820    .816    .816    .809    .798    .794
/// bright v.dense, spread      .736    .735    .732    .716    .709    .710
/// faint mid, matched          .846    .870    .891    .899    .903    .905
/// faint mid, spread           .802    .821    .834    .842    .848    .847
/// faint, bright bg            .766    .780    .791    .800    .803    .804
/// sigma 0.8 faint, spread     .924    .936    .942    .946    .945    .942
/// sigma 2.0 bright, spread    .730    .724    .719    .704    .677    .680
///
/// mean regret               .0189   .0121   .0079   .0095   .0146   .0156
/// worst-arm regret          .0593   .0352   .0149   .0261   .0523   .0496
/// ```
///
/// Score at 1 px, not 2. At 2 px a displaced extra detection beside a real
/// one still counts as a hit, which flatters the looser gates: scored that
/// way, 2.5 tied 3.0 (worst-arm regret .0174 against .0179).
///
/// The two ends want opposite things and the reason is `ADD_NATS`. On FAINT
/// fields the gate is pure loss: nothing faint enough to be a satellite can
/// pay 10 nats, so precision holds above .98 all the way down to 1.0 while
/// recall climbs 11 points. On BRIGHT fields it is load-bearing: a bright
/// emitter whose width is mis-modelled leaves a residual big enough that a
/// satellite on its wings CAN pay 10 nats, and only the gate stops the
/// tiling (see [`placement`]). 3.0 wins on both mean and worst-arm regret,
/// 4.0 is clearly wrong, and 2.0 falls off a cliff on the bright arms.
///
/// 3.0 over 2.5 is also the runtime: a lower gate means more emitters per box,
/// and a box's fits grow faster than its emitter count. On 20 GEM frames,
/// single-threaded, against detections per frame:
///
/// ```text
/// birth   4.12    3.00    2.50    2.00
/// ms/fr    104     187     249     318
/// N/fr   460.9   555.9   583.6   596.9
/// ```
///
/// 3.0 takes 21 of the 27 points of N that 2.5 does, for 1.8x the time
/// rather than 2.4x. Raise it toward 4 for speed, lower it toward 2.5 on
/// faint sparse data where the extra fits are cheap.
///
/// Confirmed without ground truth on `hyp7gem_wt_crop`, 30 frames, linked
/// with `tracking::link`: a real emitter is in the next frame and a spurious
/// one is not, so the extra detections are real only if they link. They do
/// -- N per frame 452.6 -> 545.7 -> 572.4 with the linked fraction HELD at
/// 87.0% -> 87.3% -> 87.7% and tracks of 5+ frames 799 -> 946 -> 1000. At
/// 2.0 both signals turn: the linked fraction drops back to 87.1% and mean
/// track length falls, for 2% more detections.
pub const BIRTH_Z: f64 = 3.0;
/// FIND's family-wise false-seed rate per frame, and the only free number in
/// [`seed_threshold`].
///
/// It can be this loose because FIND IS A SEEDER, NOT A DECISION RULE: every
/// candidate still has to lower its box's deviance by `ADD_NATS`, and every
/// placement inside a box passes [`Settings::birth`], so a spurious seed
/// costs runtime rather than a detection. An over-tight seed costs what
/// cannot be recovered -- a box never forms around light FIND did not seed.
///
/// This asymmetry is the whole reason `seed` and `birth` are separate
/// fields. While they were one number, loosening the seed to buy recall
/// silently loosened the gate that holds precision, so neither could move.
///
/// Measured under the retired `detect`, whose ADD step played the same role:
/// the historical cut at alpha ~ 2e-8 missed faint emitters that a looser
/// seed found (peak SNR 1.1: recall 23.1% against 65.8% at a 10x lower
/// threshold), while at alpha = 0.05 the decision rule, not the seed, did the
/// rejecting. The derived cut is a LOWER bound on the seeder's conservatism:
/// the LoG response is standard normal only where the model is right, and on
/// frames with model mismatch the effective test count is larger than the
/// pixel geometry says.
pub const SEED_ALPHA: f64 = 0.05;
/// The amplitude floor of a fit: `max(A_MIN, A_MIN_REL * A_max)`, with
/// `A_max` the window's own amplitude bound. `A_MIN` is only a backstop for a
/// window whose `A_max` is itself tiny.
///
/// The floor has to be relative because what it protects is a RATIO. An
/// emitter's position block of the Fisher matrix scales as `A^2`, so at bead
/// fluxes of ~2000 e- an amplitude of 1e-4 puts those entries at ~5.7e-12
/// against a largest diagonal of ~768 -- a ratio of 3e-14, about 130x float64
/// epsilon. There the LM step along that direction is unbounded: traced on
/// such a patch, the fit predicted a 5.2e4 nat decrease, delivered 1.05e-3,
/// and crawled for 3000+ iterations still 364 nats above the optimum.
/// Smallest over largest `diag(F)` with a second emitter parked at the floor:
///
/// ```text
///   floor / A_max     min/max diag(F)
///     0 (1e-4 abs)        3.0e-14      <- float64 noise
///         1e-6            6.4e-10
///         1e-4            4.8e-07      (saturates; a different parameter
///         3e-3            4.8e-07       becomes the smallest)
/// ```
///
/// 1e-6 buys six orders of margin over epsilon while remaining physically
/// negligible -- on a bead patch, a floor of ~0.02 e- of total flux. A larger
/// floor would start to decide how faint an emitter may be, and that decision
/// belongs to the search, not to a numerical guard.
pub const A_MIN: f64 = 1e-4;
pub const A_MIN_REL: f64 = 1e-6;
/// sigma. An out-of-band fit this near the frame border is `Edge`, not a
/// width flag. On beads_60x_still (in-focus beads on the coverslip, sigma0
/// 1.0 px) the fits the border cut off sat 0-0.5 px from it; the two narrow
/// interior fits sat 1.9 px and further in.
pub const EDGE_MARGIN: f64 = 1.0;
/// [`estimate_gain`]'s share of dimmest pixels; see its note.
pub const GAIN_FRAC: f64 = 0.2;

/// What a fitted emitter is reported as, by [`classify`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Class {
    /// Width inside the reporting band: a detection.
    Focus = 0,
    /// Narrower than the band, away from the border.
    Narrow = 1,
    /// Wider than the band, away from the border.
    Wide = 2,
    /// Out of band within `EDGE_MARGIN` sigma of the border, which cuts it:
    /// not an interior width measurement at all.
    Edge = 3,
}

/// What a caller chooses per frame.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// In-focus PSF width, px.
    pub sigma: f64,
    pub k_max: usize,
    /// SEED: FIND's frame-wide cut, in sd of the LoG null. What decides
    /// which light gets a box at all; see [`seed_threshold`].
    pub seed: f64,
    /// BIRTH: the same statistic inside a box, on the current fit's residual
    /// -- the cut a candidate placement must clear before [`ADD_NATS`] is
    /// even asked. See [`placement`].
    ///
    /// It is a separate number from `seed` because the two answer different
    /// questions. A loose `seed` costs runtime and nothing else; a loose
    /// `birth` costs precision directly, because a placement that clears it
    /// goes on to be judged by `ADD_NATS` alone. It defaults to [`BIRTH_Z`],
    /// which is calibrated for this test rather than derived from the frame.
    pub birth: f64,
    /// Widths a fit may take, as multiples of `sigma`.
    pub slack: (f64, f64),
    pub sweeps: usize,
    pub polish: bool,
    /// Widths reported as detections, as multiples of `sigma`; `None`
    /// reports every fit.
    pub band: Option<(f64, f64)>,
}

/// One frame's answer. Every fitted emitter, in or out of any reporting band
/// -- the caller classifies. Positions are global `(y, x)`.
#[derive(Clone, Debug, Default)]
pub struct Output {
    /// `2N`, row-major `(y, x)`.
    pub pos: Vec<f64>,
    pub amp: Vec<f64>,
    pub sig: Vec<f64>,
    /// `3N`: SE of `(A, y, x)` from the polish's Fisher matrix; NaN without.
    pub se: Vec<f64>,
    /// `N`: each emitter's [`Class`].
    pub class: Vec<Class>,
    /// ADU per photoelectron: the caller's, or [`estimate_gain`]'s.
    pub gain: f64,
    /// `H*W`, in the units of `d_e` (shifted by `read_noise^2`).
    pub background: Vec<f64>,
    pub n_candidates: usize,
    pub n_boxes: usize,
    pub search_fits: usize,
    pub polish_fits: usize,
}

/// Reusable per-thread storage: one per worker, never shared.
pub struct Workspace {
    fit: FitWorkspace,
    f: psf::Factors,
    chol: Chol,
    scratch: Vec<f64>,
    model: Vec<f64>,
}

impl Workspace {
    pub fn new() -> Self {
        Self {
            fit: FitWorkspace::new(),
            f: psf::Factors::new(1, 1, 1),
            chol: Chol::new(1),
            scratch: Vec::new(),
            model: Vec::new(),
        }
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// FIND's seed cut in sd of the LoG null, derived rather than carried: the
/// Bonferroni cut `Phi^-1(1 - alpha/n)` at a family-wise rate `alpha` over the
/// frame's `n` independent LoG maxima, one per `(2*ceil(sigma)+1)^2` pixels
/// (the local-maximum window of [`find_candidates`]).
///
/// So it scales with the frame and the PSF, which a constant cannot: 3.70 on
/// 64^2 at sigma 0.818, 4.43 on 512^2 at sigma 1.45.
pub fn seed_threshold(h: usize, w: usize, sigma: f64, alpha: f64) -> f64 {
    let win = (2 * sigma.ceil() as usize + 1) as f64;
    let n = (h as f64 * w as f64 / (win * win)).max(1.0);
    let p = alpha.clamp(1e-12, 0.999) / n;
    -statistics::normal_quantile(p)
}

/// `np.quantile(v, q)`, linear interpolation, `q` in `[0, 1]`.
pub fn quantile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = (s.len() - 1) as f64 * q;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (pos - lo as f64) * (s[hi] - s[lo])
}

/// `np.percentile(v, q)`, `q` in `[0, 100]`.
pub fn percentile(v: &[f64], q: f64) -> f64 {
    quantile(v, q / 100.0)
}

/// Photon-transfer gain from the dimmest pixels, ADU per photoelectron. Needs
/// no emitter model and no camera calibration. Prefer a measured gain.
///
/// On emitter-free pixels `Var = gain * (mean - offset)`, so a high-pass
/// variance over the mean is the gain. Background is chosen on a 3x3-smoothed
/// copy, because choosing the dimmest RAW pixels selects on their own noise,
/// biasing the mean down and the gain up. `GAIN_FRAC` is small: across
/// synthetic fields the dimmest 20% gives 1.00, 2.03, 4.69, 10.02 against a
/// true 1, 2, 4.7, 10, while 0.4 and 0.6 drift high as density rises because
/// PSF tails leak into the selection.
///
/// KNOWN LIMIT -- it fails at high density. The same leak reaches 0.2
/// eventually: on a bead-matched field at 0.055 emitters/px^2 this returns
/// 12.5 against a true 4.23, because even the dimmest fifth of the image sits
/// on PSF tails. Everything downstream inherits it, and a residual-based
/// correction cannot rescue it: the bias is baked into the units the residual
/// is measured in.
///
/// SECOND KNOWN LIMIT -- it does not transfer between fields. On the two
/// real bead frames it returns 4.23 and 2.94, though both were taken on the
/// same camera. The fitted bead amplitudes agree across the two at 4.23
/// (median 939 vs 933 e-), which they could not if the gain were off by 1.4x
/// on one of them.
///
/// It runs BEFORE any search: a gain read off the fitted residual arrives
/// after the search has already run with the wrong data term.
pub fn estimate_gain(raw: &[f64], h: usize, w: usize, offset: f64) -> f64 {
    if w < 5 {
        return 1.0;
    }
    let wi = w - 2;
    let mut hp = Vec::with_capacity(h * wi);
    let mut mid = Vec::with_capacity(h * wi);
    for r in 0..h {
        let row = &raw[r * w..(r + 1) * w];
        for c in 1..w - 1 {
            hp.push((row[c - 1] - 2.0 * row[c] + row[c + 1]) / 6f64.sqrt());
            mid.push(row[c]);
        }
    }
    let smooth = filters::uniform_filter(&mid, h, wi, 3, Mode::Reflect);
    let cut = quantile(&smooth, GAIN_FRAC);
    let sel: Vec<usize> = (0..h * wi).filter(|&i| smooth[i] <= cut).collect();
    if sel.len() < 32 {
        return 1.0;
    }
    let n = sel.len() as f64;
    let m = sel.iter().map(|&i| mid[i] - offset).sum::<f64>() / n;
    if m <= 1e-6 {
        return 1.0;
    }
    let hm = sel.iter().map(|&i| hp[i]).sum::<f64>() / n;
    let var = sel.iter().map(|&i| (hp[i] - hm).powi(2)).sum::<f64>() / n;
    (var / m).clamp(0.05, 200.0)
}

/// Each emitter's [`Class`]: in the band it is a detection; out of it, near
/// the border it is `Edge`, else `Narrow` or `Wide` by which side of the band
/// it fell. A source the border cuts is not an interior width measurement,
/// so a width flag always means an interior fit.
pub fn classify(pos: &[f64], sig: &[f64], h: usize, w: usize, sigma: f64, band: Option<(f64, f64)>) -> Vec<Class> {
    (0..sig.len())
        .map(|k| {
            let Some((lo, hi)) = band else {
                return Class::Focus;
            };
            let s = sig[k];
            if s >= lo * sigma && s <= hi * sigma {
                return Class::Focus;
            }
            let (y, x) = (pos[2 * k], pos[2 * k + 1]);
            let border = (y + 0.5)
                .min(h as f64 - 0.5 - y)
                .min(x + 0.5)
                .min(w as f64 - 0.5 - x);
            if border <= EDGE_MARGIN * sigma {
                Class::Edge
            } else if s < lo * sigma {
                Class::Narrow
            } else {
                Class::Wide
            }
        })
        .collect()
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 0 {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    } else {
        s[n / 2]
    }
}

/// The values `keep` selects, or `None` to mean "use the array itself".
///
/// `None` for no mask, and also for a mask that selects nothing: an empty
/// selection is no statistic at all, and the whole array is a better answer
/// than a panic.
fn masked_values(v: &[f64], keep: Option<&[bool]>) -> Option<Vec<f64>> {
    let m = keep?;
    let sel: Vec<f64> = v.iter().zip(m).filter(|&(_, &k)| k).map(|(&x, _)| x).collect();
    (!sel.is_empty()).then_some(sel)
}

/// Pixels of context a crop needs outside the ROI for the answer inside it to
/// be the answer the whole frame would give.
///
/// Three supports reach in from the crop's edge, and the widest wins:
///
/// | stage | reach | at sigma 1.3 |
/// |---|---|---|
/// | [`find_candidates`] | `ceil(sigma)` (the max-filter window) + [`filters::kernel_radius`] | 7 |
/// | boxes | `ceil(BBOX_PAD * sigma) + 1` ([`patches::build_patches`]) | 5 |
/// | [`background_map`] | `2 * (BG_KERNEL / 2) + kernel_radius(BG_KERNEL / 6)` | 41 |
///
/// The background chain sets it, at 41 px, and unlike the other two it does
/// not shrink with `sigma`: its kernel is a fixed 25 px. A candidate needs
/// its LoG exact over its whole max-filter window, and each of those needs
/// data out to the LoG's own radius, so FIND's two radii add rather than max.
fn crop_margin(sigma: f64) -> usize {
    let find = sigma.ceil() as usize + filters::kernel_radius(sigma);
    let boxes = (patches::BBOX_PAD * sigma).ceil() as usize + 1;
    let bg = 2 * (BG_KERNEL / 2) + filters::kernel_radius(BG_KERNEL as f64 / 6.0);
    find.max(boxes).max(bg)
}

/// Widen `[lo, hi)` to at least `want` pixels without leaving `[0, n)`, and
/// without ever giving up ground it already held.
fn widen(lo: usize, hi: usize, want: usize, n: usize) -> (usize, usize) {
    let want = want.min(n);
    if hi - lo >= want {
        return (lo, hi);
    }
    let mid = (lo + hi) / 2;
    let start = mid.saturating_sub(want / 2).min(n - want);
    (start.min(lo), (start + want).max(hi))
}

/// The sub-frame [`localize`] does its whole-frame work on: the ROI's
/// bounding box plus [`crop_margin`], clamped to the frame. `None` when the
/// ROI selects no pixel at all.
///
/// Also at least `3 * BG_KERNEL` px on a side where the frame allows, because
/// [`background_map`] narrows its kernel on an array too small to hold it
/// (`BG_KERNEL.min(3.max(min(h, w) / 3))`). Without the floor a thin ROI
/// would silently get a different background kernel from the one the frame
/// would have used, which is exactly the margin's promise broken. It costs
/// nothing in the ordinary case: the margin alone already gives 83 px.
fn roi_crop(roi: &[bool], h: usize, w: usize, sigma: f64) -> Option<patches::BBox> {
    let (mut y0, mut y1) = (usize::MAX, 0usize);
    let (mut x0, mut x1) = (usize::MAX, 0usize);
    for r in 0..h {
        for c in 0..w {
            if roi[r * w + c] {
                y0 = y0.min(r);
                y1 = y1.max(r + 1);
                x0 = x0.min(c);
                x1 = x1.max(c + 1);
            }
        }
    }
    if y0 == usize::MAX {
        return None;
    }
    let m = crop_margin(sigma);
    let side = 3 * BG_KERNEL;
    let (y0, y1) = widen(y0.saturating_sub(m), (y1 + m).min(h), side, h);
    let (x0, x1) = widen(x0.saturating_sub(m), (x1 + m).min(w), side, w);
    Some(patches::BBox { y0, x0, y1, x1 })
}

/// Copy `bb` out of an `h*w` array, row by row.
fn crop<T: Copy>(v: &[T], w: usize, bb: &patches::BBox) -> Vec<T> {
    let mut out = Vec::with_capacity(bb.n_pixels());
    for r in bb.y0..bb.y1 {
        out.extend_from_slice(&v[r * w + bb.x0..r * w + bb.x1]);
    }
    out
}

/// FIND against a flat `level`: LoG peaks of the variance-normalized
/// residual above `threshold`, brightest first. Returns
/// `(pos 2N, amp, strength)`.
///
/// Two normalizations, doing different jobs. Dividing by `sqrt(level)`
/// before the filter makes one threshold valid across the FRAME: under
/// Poisson noise the residual's scale is the square root of the mean.
/// Dividing by [`filters::log_kernel_l2`] after it makes one threshold valid
/// across SIGMA: the filter's null sd is its kernel's L2 norm, which scales
/// as sigma^-3, so without it one number is a 1.8-sigma cut at sigma 0.8 and
/// a 100-sigma cut at sigma 3.0. `threshold` is therefore a count of standard
/// deviations.
///
/// # The LoG is a poor statistic, and the better one is worse here
///
/// Measured 2026-09-12, recorded so it is not re-attempted. The LoG is NOT
/// the matched filter for a Gaussian spot: `gaussian_laplace(sigma)`
/// correlates best with a spot of width `0.60 * sigma`, and against the
/// pixel-integrated PSF at its own sigma it delivers 0.69 of the z a matched
/// filter would on-pixel and 0.60 at the pixel corner -- at sigma 0.8 the
/// corner case falls to 0.43, so a sharper PSF buys the LoG nothing. A
/// difference of Gaussians, `g(sigma) - g(k*sigma)`, is that matched filter
/// with the background projected out -- the exact GLRT for unknown amplitude
/// on a smooth background, its null sd closed-form as `||K||_2` exactly like
/// the LoG's -- and reaches 0.95. As a SEEDER in isolation it is worth 1.3x
/// to 1.75x in flux: at a matched false-seed rate, recall at peak SNR 1.8
/// went 0.217 -> 0.477, and at sigma 0.8 0.467 -> 0.927.
///
/// None of that survives into the pipeline. Swapped in behind `seed` with
/// `birth` pinned, each kernel bisected onto 15 false seeds per empty
/// 128x128 frame so the operating points match (see the alpha note below),
/// six to twelve frames per cell, `band=None` so recall measures detection
/// and not classification:
///
/// ```text
/// ISOLATED (density 0.002, 22 px spacing), recall
///   peak SNR   1.40   2.01   2.81   4.01   6.02
///   LoG        .090   .317   .740   .940   .960
///   DoG k=2    .090   .323   .737   .937   .957
///
/// CROWDED (flux 200-600, matched width), recall / candidates per frame
///   density   0.002      0.005      0.010      0.015      0.025      0.040
///   LoG     .975  35  .925  64  .875 108  .820 144  .727 195  .615 236
///   DoG k=2 .975  33  .915  60  .856 100  .805 133  .684 165  .565 188
///   DoG k=3 .975  32  .911  59  .837  96  .787 126  .657 151  .531 166
/// ```
///
/// Two things are happening. Where emitters are ISOLATED the seeder is not
/// what limits recall -- `ADD_NATS` is. The DoG seeds strictly more (28
/// candidates against 22 at peak SNR 1.4) and every extra one fails to pay
/// its 10 nats, so a better statistic buys exactly what a looser cut buys,
/// which is nothing. Where emitters are CROWDED the DoG's broader core
/// merges neighbours inside the max-filter window, and a box that never
/// forms cannot be recovered.
///
/// So the LoG's 29% efficiency loss is the price of a narrow core, and the
/// narrow core is worth more than the efficiency. A better FIND has to
/// SEPARATE better, not detect better; sensitivity is `ADD_NATS`' problem.
///
/// # What `SEED_ALPHA` actually buys
///
/// [`seed_threshold`]'s cut is not the achieved rate. `level` is the 10th
/// percentile, deliberately -- it is a background estimate under emitters --
/// but it is also the variance normalizer here, and too low a normalizer
/// inflates the z: on emitter-free Poisson frames the normalized residual
/// has sd 1.19 at a background of 20 e- and 1.07 at 100 e-, not 1. The
/// derived cut of 3.79 therefore passes 17.9 false seeds per empty 128x128
/// frame, not the 0.05 `SEED_ALPHA` names. For a seeder that errs in the
/// safe direction, which is why it has never mattered. It does mean a cut
/// compared across two different statistics must be calibrated empirically:
/// at one nominal z the DoG passes 10.8 seeds against the LoG's 17.9, and
/// comparing them there compares the calibrations, not the filters.
pub fn find_candidates(
    d: &[f64],
    h: usize,
    w: usize,
    level: f64,
    sigma: f64,
    threshold: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let norm = level.max(1e-6).sqrt();
    let nr: Vec<f64> = d.iter().map(|&v| (v - level) / norm).collect();
    let l2 = filters::log_kernel_l2(sigma);
    let mut log_f = filters::gaussian_laplace(&nr, h, w, sigma, Mode::Reflect);
    for v in log_f.iter_mut() {
        *v = -*v / l2;
    }
    let win = 2 * sigma.ceil() as usize + 1;
    let mx = filters::maximum_filter(&log_f, h, w, win, Mode::Reflect);
    let pf = psf::peak_factor(sigma);
    let mut found: Vec<(f64, usize)> = (0..h * w)
        .filter(|&i| log_f[i] == mx[i] && log_f[i] > threshold)
        .map(|i| (log_f[i], i))
        .collect();
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut pos = Vec::with_capacity(2 * found.len());
    let mut amp = Vec::with_capacity(found.len());
    let mut strength = Vec::with_capacity(found.len());
    for &(s, i) in &found {
        pos.push((i / w) as f64);
        pos.push((i % w) as f64);
        amp.push((d[i] - level).max(1e-2) / pf);
        strength.push(s);
    }
    (pos, amp, strength)
}

/// The smooth background surface, in photoelectrons: a local mean over the
/// pixels no candidate reaches, taken twice with a one-sided clip between,
/// then smoothed.
///
/// Estimated from masked DATA, never from the PSF-subtracted residual: that
/// route is a feedback loop, because emitters that have absorbed background
/// depress the residual, the surface follows them down, and they must absorb
/// more. An undetected emitter is not masked, so the clip removes the upper
/// tail only; its contamination is always positive, and clipping both tails
/// biases the estimate down.
///
/// Windows with fewer than `BG_MIN_PIXELS` free pixels take one frame-wide
/// scalar: at high density that is most of them, and the surface degrades to
/// a scalar rather than to an average over four pixels. The scalar is the
/// median of the free pixels, falling back to the 10th percentile when too
/// few are free. Neither alone serves both regimes: the image median is no
/// background on a crowded field (99 ADU against a true 12-19 on
/// beads_60x_still), and a low quantile is none on a sparse one (9.7 against
/// a true 20 on an emitter-free frame).
///
/// `roi`, if given, is `h*w` and confines those two SCALARS -- and nothing
/// else -- to the pixels it selects. The surface itself is estimated from
/// every pixel, because a window straddling the ROI's edge is entitled to
/// the real data on both sides of it. The scalar is the one number the whole
/// crop can fall back to, so it has to describe the region asked about: on
/// `hyp7gem_wt_crop` the frame's own 10th percentile is 11.1 e-, which is the
/// dark field OUTSIDE the cell, against 18-20 inside it. Confining it also
/// makes it independent of how much margin the crop carries, which a
/// crop-wide statistic is not.
///
/// Returns the surface and the fallback level, which is what a caller
/// scattering this back into a larger array should fill the rest with.
pub fn background_map(
    d: &[f64],
    h: usize,
    w: usize,
    cand: &[f64],
    sigma: f64,
    roi: Option<&[bool]>,
) -> (Vec<f64>, f64) {
    let n = cand.len() / 2;
    let mut k = BG_KERNEL.min(3usize.max(h.min(w) / 3));
    if k % 2 == 0 {
        k += 1;
    }
    let free = render::emitter_free_mask(cand, n, sigma, BG_MASK_RADIUS, h, w);
    let inside = masked_values(d, roi);
    let stat = inside.as_deref().unwrap_or(d);
    let fallback = if n == 0 {
        median(stat)
    } else {
        let kept: Vec<f64> = (0..h * w)
            .filter(|&i| free[i] && roi.is_none_or(|m| m[i]))
            .map(|i| d[i])
            .collect();
        if kept.len() as f64 >= 16f64.max(0.02 * stat.len() as f64) {
            median(&kept)
        } else {
            percentile(stat, 10.0)
        }
    };
    let local_mean = |mask: &[bool]| -> (Vec<f64>, Vec<f64>) {
        let masked: Vec<f64> = (0..h * w).map(|i| if mask[i] { d[i] } else { 0.0 }).collect();
        let ones: Vec<f64> = mask.iter().map(|&m| if m { 1.0 } else { 0.0 }).collect();
        let num = filters::uniform_filter(&masked, h, w, k, Mode::Nearest);
        let den = filters::uniform_filter(&ones, h, w, k, Mode::Nearest);
        let b = num.iter().zip(&den).map(|(a, c)| a / c.max(1e-9)).collect();
        let cnt = den.iter().map(|c| c * (k * k) as f64).collect();
        (b, cnt)
    };
    let (b1, _) = local_mean(&free);
    let keep: Vec<bool> = (0..h * w)
        .map(|i| free[i] && d[i] <= b1[i] + 3.0 * b1[i].max(BG_FLOOR).sqrt())
        .collect();
    let (b2, cnt) = local_mean(&keep);
    let b3: Vec<f64> = (0..h * w)
        .map(|i| if cnt[i] >= BG_MIN_PIXELS { b2[i] } else { fallback })
        .collect();
    let mut out = filters::gaussian_filter(&b3, h, w, k as f64 / 6.0, Mode::Nearest);
    for v in out.iter_mut() {
        *v = v.max(BG_FLOOR);
    }
    (out, fallback.max(BG_FLOOR))
}

/// An emitter: `[A, y, x, sigma]`.
type Em = [f64; 4];

/// A window's pixels and the parameter-free part of its model.
///
/// The background map enters split into a free `level` (its median here,
/// where the fit's `b` starts) and a known `shape`, added like a frozen
/// emitter. The split keeps `b` strictly interior: folding the whole surface
/// into the known term would leave `b` wanting to be 0, its lower bound, and
/// Coleman-Li collapses every coordinate's step when one parameter sits on a
/// bound (see [`lmcl`]).
struct Window {
    y0: usize,
    x0: usize,
    h: usize,
    w: usize,
    sub: Vec<f64>,
    /// Frozen neighbours plus the background map's shape.
    halo: Vec<f64>,
    /// The background map's median here; where the free level starts.
    level: f64,
    /// Background map's shape, `bmap - level`.
    shape: Vec<f64>,
}

impl Window {
    fn new(d: &[f64], fw: usize, bmap: &[f64], bb: &patches::BBox) -> Self {
        let (h, w) = (bb.h(), bb.w());
        let mut sub = Vec::with_capacity(h * w);
        let mut bg = Vec::with_capacity(h * w);
        for r in bb.y0..bb.y1 {
            sub.extend_from_slice(&d[r * fw + bb.x0..r * fw + bb.x1]);
            bg.extend_from_slice(&bmap[r * fw + bb.x0..r * fw + bb.x1]);
        }
        let level = median(&bg);
        let shape: Vec<f64> = bg.iter().map(|v| v - level).collect();
        Self {
            y0: bb.y0,
            x0: bb.x0,
            h,
            w,
            sub,
            halo: shape.clone(),
            level,
            shape,
        }
    }

    /// `halo <- (emitters, rendered here) + shape`. `ems` are global.
    fn set_halo(&mut self, ems: &[Em], f: &mut psf::Factors) {
        let n = self.h * self.w;
        self.halo.clear();
        self.halo.resize(n, 0.0);
        if !ems.is_empty() {
            let mut theta = Vec::with_capacity(4 * ems.len() + 1);
            theta.push(0.0);
            for e in ems {
                theta.extend_from_slice(&[e[0], e[1] - self.y0 as f64, e[2] - self.x0 as f64, e[3]]);
            }
            let (ay, ax) = (psf::local_axis(self.h), psf::local_axis(self.w));
            f.ensure(self.h, self.w, ems.len());
            psf::model_var_sigma_ax(&theta, &ay, &ax, None, f, &mut self.halo);
        }
        for (v, s) in self.halo.iter_mut().zip(&self.shape) {
            *v += s;
        }
    }
}

/// One converged window fit: its data-only I and parameters, local coords.
struct Fitted {
    i_div: f64,
    b: f64,
    em: Vec<Em>,
}

/// One bounded free-width ML fit: bounds from the window's peak, the start
/// pulled 1e-9 inside them.
///
/// `a_max` is raised by `slack.1^2`: it comes from the window's peak through
/// `peak_factor(sigma)`, and a source `n` times the PSF width carries the
/// same flux at `1/n^2` of the peak, so the in-focus bound would clip exactly
/// the defocused emitters the slack exists for. Positions stay inside the
/// window: letting a centre leave it was tried -- it lets the fit put rim
/// flux where it came from -- and measurably lost real detections elsewhere.
fn fit_window(
    ws: &mut Workspace,
    win: &Window,
    b: f64,
    em: &[Em],
    s: &Settings,
    max_iter: usize,
    tol_obj: f64,
) -> Fitted {
    let k = em.len();
    let smax = win.sub.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v)).max(1.0);
    let b_max = (4.0 * smax).max(10.0);
    let a_max = 8.0 * smax / psf::peak_factor(s.sigma) * s.slack.1 * s.slack.1;
    let a_min = A_MIN.max(A_MIN_REL * a_max);
    let (s_lo, s_hi) = (s.slack.0 * s.sigma, s.slack.1 * s.sigma);
    let mut lo = Vec::with_capacity(4 * k + 1);
    let mut hi = Vec::with_capacity(4 * k + 1);
    lo.push(0.0);
    hi.push(b_max);
    for _ in 0..k {
        lo.extend_from_slice(&[a_min, -0.5, -0.5, s_lo]);
        hi.extend_from_slice(&[a_max, win.h as f64 - 0.5, win.w as f64 - 0.5, s_hi]);
    }
    let mut th0 = Vec::with_capacity(4 * k + 1);
    th0.push(b);
    for e in em {
        th0.extend_from_slice(&[e[0], e[1], e[2], e[3].clamp(s_lo, s_hi)]);
    }
    for q in 0..th0.len() {
        th0[q] = th0[q].clamp(lo[q] + 1e-9, hi[q] - 1e-9);
    }
    let bounds = Bounds::new(&lo, &hi);
    let info = lmcl::fit_var_sigma(
        &mut ws.fit,
        &th0,
        win.h,
        win.w,
        &win.sub,
        &bounds,
        Some(&win.halo),
        FitOpts {
            max_iter,
            tol_obj,
            ..Default::default()
        },
    );
    let t = ws.fit.theta();
    Fitted {
        i_div: info.i_div,
        b: t[0],
        em: (0..k)
            .map(|j| [t[1 + 4 * j], t[2 + 4 * j], t[3 + 4 * j], t[4 + 4 * j]])
            .collect(),
    }
}

/// `(y, x, A0)` of the strongest owned peak of the fit's normalized residual
/// that passes [`Settings::birth`], or `None`.
///
/// The test is not optional. The deviance of the best of many placements is
/// a maximum over positions, and `ADD_NATS` was measured only on placements
/// that had already been screened; without the screen every bright emitter
/// collected faint satellites on its wings (130 emitters for 107 true on
/// seed 17, precision 0.74).
///
/// # The LoG stays, though it is the wrong statistic on paper
///
/// Inside a box the argument FIND lost (see [`find_candidates`]) looks
/// stronger. The level is one free constant, fitted, so the exact GLRT for
/// one more emitter at a grid position is the pixel-integrated PSF matched
/// filter with that constant projected out under the Poisson weights `1/m`.
/// It lost anyway, and in BOTH of this function's jobs: deciding whether to
/// place, and where.
///
/// Measured 2026-09-12 on the [`BIRTH_Z`] arms (same seeds, 1 px match,
/// `band=None`), each statistic with `birth` swept 5.0-1.5 on its own scale
/// and taken at its own minimax cut. Mean / worst-arm F1 regret is against
/// each arm's best over every row and cut. Then the paired F1 change against
/// the LoG at 3.0, per arm in the [`BIRTH_Z`] table's order:
///
/// ```text
/// gate     position  birth   mean   worst   per-arm dF1 vs LoG
/// LoG      LoG        3.0   .0083   .0149   (reference)
/// MF       MF         3.5   .0362   .0765   -.006 -.025 -.030 -.028 -.062 -.030 -.028 -.009 -.033
/// MF+proj  MF+proj    4.0   .0395   .0812   -.008 -.026 -.034 -.032 -.066 -.031 -.027 -.010 -.047
/// MF+proj  LoG        3.0   .0225   .0374   -.006 -.019 -.020 -.015 -.020 -.010 -.007 -.007 -.024
/// LoG      MF+proj    3.5   .0229   .0483   -.001 -.007 -.020 -.022 -.021 -.010 -.010 -.005 -.035
/// ```
///
/// MF is the pixel-integrated Gaussian at `sigma`, truncated at the window
/// and normalized by its weighted norm; "+proj" projects the level out.
/// Projecting the level out did not help. No arm gains at any row's cut, and
/// with a free cut per arm the best any matched-filter row does is tie. The
/// two split rows show where the loss comes from:
///
/// * As a POSITION PICKER (LoG gate, MF peak) recall is unchanged and
///   precision falls with the gate: .913 -> .879 at 3.0 on bright mid, .732
///   -> .608 at sigma 2.0, the arms where widths are mis-modelled. That is
///   consistent with the broad core peaking on the smooth residual a
///   width-mismatched bright emitter leaves on its wings, where the start
///   then pays `ADD_NATS` as a satellite; the LoG's negative surround
///   rejects smooth residual. The satellites were not traced one by one.
/// * As a GATE (MF z, LoG peak) precision is below the LoG's at EVERY cut on
///   the bright arms, even at 5.0 (.916 against .957 on bright mid), so no
///   constant recovers it. The bright arms want a statistic that ignores
///   smooth residual, not a higher cut on one that sums it.
///
/// Even where the residual is closest to one emitter plus noise -- faint, or
/// matched width -- the matched filter at best ties (.905 against .905 on
/// faint matched, .982 against .986 on bright sparse). Its optimality holds
/// for a residual the search never sees. The LoG's band pass is rejecting
/// model mismatch, which the GLRT does not. The
/// 0.60-0.69 efficiency the note at [`find_candidates`] measures is the price.
///
/// The tests here and in [`search_box`] are written `!(a > b)` on purpose: a
/// NaN must fail them.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn placement(
    ws: &mut Workspace,
    win: &Window,
    owned: &[bool],
    state: &Fitted,
    s: &Settings,
    l2: f64,
) -> Option<(f64, f64, f64)> {
    let n = win.h * win.w;
    let mut theta = Vec::with_capacity(4 * state.em.len() + 1);
    theta.push(state.b);
    for e in &state.em {
        theta.extend_from_slice(e);
    }
    let (ay, ax) = (psf::local_axis(win.h), psf::local_axis(win.w));
    ws.f.ensure(win.h, win.w, state.em.len().max(1));
    ws.model.clear();
    ws.model.resize(n, 0.0);
    psf::model_var_sigma_ax(&theta, &ay, &ax, Some(&win.halo), &mut ws.f, &mut ws.model);
    let nr: Vec<f64> = (0..n)
        .map(|i| (win.sub[i] - ws.model[i]) / ws.model[i].max(1e-6).sqrt())
        .collect();
    let log_f = filters::gaussian_laplace(&nr, win.h, win.w, s.sigma, Mode::Nearest);
    let mut best: Option<(f64, usize)> = None;
    for i in 0..n {
        if owned[i] {
            let v = -log_f[i] / l2;
            if best.is_none_or(|(b, _)| v > b) {
                best = Some((v, i));
            }
        }
    }
    let (v, i) = best?;
    if !(v > s.birth) {
        return None;
    }
    let resid = win.sub[i] - ws.model[i];
    Some(((i / win.w) as f64, (i % win.w) as f64, resid.max(1e-2) / psf::peak_factor(s.sigma)))
}

/// One box decided from K = 0: FORWARD placements while each pays ADD_NATS,
/// then BACKWARD removals while one costs less. Returns the box's emitters
/// (local) and the fits spent.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn search_box(
    ws: &mut Workspace,
    win: &Window,
    owned: &[bool],
    s: &Settings,
    l2: f64,
) -> (Vec<Em>, usize) {
    let mut fits = 1usize;
    let mut state = fit_window(ws, win, win.level, &[], s, FIT_MAX_ITER, FIT_TOL_OBJ);
    while state.em.len() < s.k_max {
        let Some((y, x, a0)) = placement(ws, win, owned, &state, s, l2) else {
            break;
        };
        let mut em = state.em.clone();
        em.push([a0, y, x, s.sigma]);
        let trial = fit_window(ws, win, state.b, &em, s, FIT_MAX_ITER, FIT_TOL_OBJ);
        fits += 1;
        if !(state.i_div - trial.i_div > ADD_NATS) {
            break;
        }
        state = trial;
    }
    // Not at K = 1: that removal is the K = 0 fit FORWARD already beat.
    // Measured: output bit-identical, 11-25% fewer fits.
    while state.em.len() > 1 {
        let mut best: Option<Fitted> = None;
        for drop in 0..state.em.len() {
            let em: Vec<Em> = (0..state.em.len())
                .filter(|&j| j != drop)
                .map(|j| state.em[j])
                .collect();
            let reduced = fit_window(ws, win, state.b, &em, s, FIT_MAX_ITER, FIT_TOL_OBJ);
            fits += 1;
            if best.as_ref().is_none_or(|b| reduced.i_div < b.i_div) {
                best = Some(reduced);
            }
        }
        let best = best.expect("K >= 2 has removals");
        if !(best.i_div - state.i_div < ADD_NATS) {
            break;
        }
        state = best;
    }
    (state.em, fits)
}

/// Pixels a box may place on: within `OWN_RADIUS*sigma` of its own nearest
/// candidate, no farther from it than from any other candidate, and in the
/// ROI. Only candidates within that radius of the box can take a pixel from
/// it, so `grid` is asked for those alone.
#[allow(clippy::too_many_arguments)]
fn owned_mask(
    bb: &patches::BBox,
    own: &[u32],
    cand: &[f64],
    grid: &EmitterGrid,
    roi: Option<&[bool]>,
    fw: usize,
    sigma: f64,
    near: &mut Vec<u32>,
) -> Vec<bool> {
    let r = OWN_RADIUS * sigma;
    grid.query_rect(
        bb.y0 as f64,
        bb.x0 as f64,
        (bb.y1 - 1) as f64,
        (bb.x1 - 1) as f64,
        r,
        near,
    );
    near.retain(|i| !own.contains(i));
    let mut owned = Vec::with_capacity(bb.n_pixels());
    for py in bb.y0..bb.y1 {
        for px in bb.x0..bb.x1 {
            let dist = |i: u32| {
                (py as f64 - cand[2 * i as usize]).hypot(px as f64 - cand[2 * i as usize + 1])
            };
            let d_own = own.iter().map(|&i| dist(i)).fold(f64::INFINITY, f64::min);
            let d_oth = near.iter().map(|&i| dist(i)).fold(f64::INFINITY, f64::min);
            let in_roi = roi.is_none_or(|m| m[py * fw + px]);
            owned.push(d_own <= r && d_own <= d_oth && in_roi);
        }
    }
    owned
}

/// Localize one frame of `d_e`. `roi`, if given, is `H*W`: candidates
/// outside it are dropped after the background map is built from all of
/// them, and no box places outside it.
///
/// # The crop
///
/// An ROI confines the search, so the whole-frame preamble -- FIND, the
/// background surface, and the level they are measured against -- runs on
/// the ROI's bounding box plus [`crop_margin`] rather than on the frame.
/// Nothing else moves: positions come back in global coordinates and the
/// background is scattered into an `H*W` map, so boxes, the halo, the polish
/// and the edge classification all still see the whole frame.
///
/// It is worth doing because that preamble is a third of the frame's work and
/// none of it depends on N -- on a 256^2 frame, 5.1 ms of 15.3 ms
/// (`percentile` 0.33, FIND 1.64, background 3.16); on 512^2, 23.0 ms of
/// 67.4 ms. `background_map` is the larger half and does not shrink with
/// `sigma`: its kernel is a fixed 25 px. Measured end to end on
/// `hyp7gem_wt_crop`, a centred square ROI:
///
/// | frame | ROI | crop | before | after |
/// |---|---|---|---|---|
/// | 256^2 | none | -- | 50.1 ms | 50.5 ms |
/// | 256^2 | 32^2 | 114^2 | 6.09 ms | 1.75 ms |
/// | 256^2 | 64^2 | 146^2 | 8.27 ms | 4.67 ms |
/// | 256^2 | 128^2 | 210^2 | 23.4 ms | 18.3 ms |
/// | 512^2 | 32^2 | 114^2 | 25.4 ms | 3.43 ms |
/// | 512^2 | 64^2 | 146^2 | 29.1 ms | 8.07 ms |
/// | 512^2 | 128^2 | 210^2 | 38.1 ms | 18.7 ms |
///
/// The win is what the ROI throws away, so it grows with the frame and
/// shrinks as the ROI approaches it; at `roi = None` the crop is the frame
/// and nothing changes, bit for bit (asserted across three images and three
/// sigmas, positions, amplitudes, widths and the whole background map). An
/// ROI whose bounding box is the frame -- scattered cells, a diagonal band --
/// buys nothing and costs nothing.
///
/// The margin makes the crop invisible to everything inside the ROI:
/// measured, the same ROI given more surrounding frame than the crop needs
/// returns the same N with positions agreeing to 1.3e-12 px, which is filter
/// summation order over a differently-shaped array. What the ROI does change
/// -- deliberately -- is `b0` and the background's fallback, now measured
/// over the ROI's own pixels rather than the frame's; see [`background_map`].
/// On `hyp7gem_wt_crop` that moved N by up to 5% on the larger ROIs (78 to
/// 74 at 128^2), because the frame's 10th percentile is the dark field
/// outside the cell.
pub fn localize(
    d: &[f64],
    h: usize,
    w: usize,
    roi: Option<&[bool]>,
    s: &Settings,
    ws: &mut Workspace,
) -> Output {
    assert_eq!(d.len(), h * w);
    let bb = match roi {
        None => patches::BBox { y0: 0, x0: 0, y1: h, x1: w },
        Some(m) => match roi_crop(m, h, w, s.sigma) {
            Some(bb) => bb,
            // An ROI selecting nothing asks for nothing. `gain` is NaN as
            // on every other path out of here; `localize_raw` fills it in.
            None => {
                return Output {
                    background: vec![BG_FLOOR; h * w],
                    gain: f64::NAN,
                    ..Output::default()
                }
            }
        },
    };
    let whole = bb.h() == h && bb.w() == w;
    let (cw, ch) = (bb.w(), bb.h());
    let dsub = if whole { Vec::new() } else { crop(d, w, &bb) };
    let dc: &[f64] = if whole { d } else { &dsub };
    let rsub = match roi {
        Some(m) if !whole => Some(crop(m, w, &bb)),
        _ => None,
    };
    let rc: Option<&[bool]> = match (roi, &rsub) {
        (Some(m), None) => Some(m),
        (_, Some(v)) => Some(v),
        (None, None) => None,
    };

    let level = masked_values(dc, rc);
    let b0 = percentile(level.as_deref().unwrap_or(dc), 10.0).max(BG_FLOOR);
    let (cand_all, amp_all, str_all) = find_candidates(dc, ch, cw, b0, s.sigma, s.seed);
    let (bsub, fill) = background_map(dc, ch, cw, &cand_all, s.sigma, rc);
    // Back to the frame. Outside the crop nothing was estimated, so the map
    // carries the fallback level rather than a hole: no fit reads it -- every
    // box lies inside the crop by `crop_margin` -- but callers render it.
    let bmap = if whole {
        bsub
    } else {
        let mut full = vec![fill; h * w];
        for r in 0..ch {
            full[(bb.y0 + r) * w + bb.x0..(bb.y0 + r) * w + bb.x1]
                .copy_from_slice(&bsub[r * cw..(r + 1) * cw]);
        }
        full
    };

    let (mut cand, mut camp, mut strength) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..amp_all.len() {
        let (y, x) = (cand_all[2 * j] + bb.y0 as f64, cand_all[2 * j + 1] + bb.x0 as f64);
        if roi.is_none_or(|m| m[y as usize * w + x as usize]) {
            cand.extend_from_slice(&[y, x]);
            camp.push(amp_all[j]);
            strength.push(str_all[j]);
        }
    }
    let nc = camp.len();
    let mut boxes = patches::build_patches(&cand, nc, s.sigma, h, w, s.k_max);
    // Brightest first, so the strongest light is already fitted when its
    // neighbours read it through their halos. Stable, so ties keep FIND's
    // order.
    let peak = |p: &patches::Patch| {
        p.indices
            .iter()
            .map(|&i| strength[i as usize])
            .fold(f64::NEG_INFINITY, f64::max)
    };
    boxes.sort_by(|a, b| peak(b).total_cmp(&peak(a)));
    let nb = boxes.len();

    // Once per frame: each box's pixels, ownership and the boxes that can
    // reach it. A held emitter stays inside its own box's fit bounds, within
    // half a pixel of the rectangle, at a width of at most slack.1 * sigma;
    // the halo then admits it only within HALO_FACTOR widths of this
    // box's pixel rectangle. Boxes farther than that can never contribute.
    let grid = EmitterGrid::build(&cand, nc, h, w, (OWN_RADIUS * s.sigma).max(1.0));
    let mut near = Vec::new();
    let mut wins: Vec<Window> = Vec::with_capacity(nb);
    let mut owned: Vec<Vec<bool>> = Vec::with_capacity(nb);
    for p in &boxes {
        wins.push(Window::new(d, w, &bmap, &p.bbox));
        owned.push(owned_mask(&p.bbox, &p.indices, &cand, &grid, roi, w, s.sigma, &mut near));
    }
    let reach = HALO_FACTOR * s.slack.1.max(1.0) * s.sigma;
    let gap = |a0: f64, a1: f64, b0: f64, b1: f64| (b0 - a1).max(a0 - b1).max(0.0);
    let neighbours: Vec<Vec<usize>> = (0..nb)
        .map(|i| {
            let bi = &boxes[i].bbox;
            (0..nb)
                .filter(|&j| j != i)
                .filter(|&j| {
                    let bj = &boxes[j].bbox;
                    let gy = gap(
                        bi.y0 as f64,
                        (bi.y1 - 1) as f64,
                        bj.y0 as f64 - 0.5,
                        bj.y1 as f64 - 0.5,
                    );
                    let gx = gap(
                        bi.x0 as f64,
                        (bi.x1 - 1) as f64,
                        bj.x0 as f64 - 0.5,
                        bj.x1 as f64 - 0.5,
                    );
                    gy.hypot(gx) <= reach
                })
                .collect()
        })
        .collect();

    // What each box holds, global coords. Until a box is first decided, its
    // candidates stand in for it as in-focus seeds.
    let mut held: Vec<Vec<Em>> = boxes
        .iter()
        .map(|p| {
            p.indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    [camp[i], cand[2 * i], cand[2 * i + 1], s.sigma]
                })
                .collect()
        })
        .collect();

    let l2 = filters::log_kernel_l2(s.sigma);
    let mut search_fits = 0usize;
    let mut ems: Vec<Em> = Vec::new();
    for _ in 0..s.sweeps {
        for i in 0..nb {
            let bb = boxes[i].bbox;
            let (ry0, ry1) = (bb.y0 as f64, (bb.y1 - 1) as f64);
            let (rx0, rx1) = (bb.x0 as f64, (bb.x1 - 1) as f64);
            ems.clear();
            for &j in &neighbours[i] {
                for e in &held[j] {
                    let dy = e[1] - e[1].clamp(ry0, ry1);
                    let dx = e[2] - e[2].clamp(rx0, rx1);
                    if dy.hypot(dx) <= HALO_FACTOR * e[3].max(s.sigma) {
                        ems.push(*e);
                    }
                }
            }
            wins[i].set_halo(&ems, &mut ws.f);
            let (local, fits) = search_box(ws, &wins[i], &owned[i], s, l2);
            search_fits += fits;
            let (y0, x0) = (bb.y0 as f64, bb.x0 as f64);
            held[i] = local.iter().map(|e| [e[0], e[1] + y0, e[2] + x0, e[3]]).collect();
        }
    }

    let all: Vec<Em> = held.into_iter().flatten().collect();
    let n = all.len();
    let mut pos: Vec<f64> = all.iter().flat_map(|e| [e[1], e[2]]).collect();
    let mut amp: Vec<f64> = all.iter().map(|e| e[0]).collect();
    let mut sig: Vec<f64> = all.iter().map(|e| e[3]).collect();
    let mut se = vec![f64::NAN; 3 * n];
    let polish_fits = if s.polish && n > 0 {
        polish(d, h, w, &bmap, &mut pos, &mut amp, &mut sig, &mut se, s, ws)
    } else {
        0
    };
    let class = classify(&pos, &sig, h, w, s.sigma, s.band);
    Output {
        pos,
        amp,
        sig,
        se,
        class,
        gain: f64::NAN,
        background: bmap,
        n_candidates: nc,
        n_boxes: nb,
        search_fits,
        polish_fits,
    }
}

/// Block-Jacobi refits at fixed N until nothing moves, with free widths, and
/// per-emitter CRLBs from the Fisher matrix of the fit whose parameters are
/// reported.
///
/// Jacobi, not Gauss-Seidel: every patch in a sweep reads the sweep's INPUT
/// state, so the answer does not depend on visit order. Iterated, not run
/// once: each patch holds its neighbours frozen where it was handed them, so
/// one pass propagates their staleness. Measured on isolated emitters at
/// density 0.055, a 0.5 px error in the NEIGHBOURS alone (target started at
/// truth) takes the pull sd from 0.96 to 1.98; sweeping recovers it to 1.27.
/// The patches are rebuilt each sweep, which refreshes the halo -- so the map
/// is not continuous, and a fixed point need not exist (PORTING_NOTES 20).
///
/// Scheduled group-wise: a patch none of whose free or frozen emitters moved
/// more than `POLISH_MOVE_TOL` last sweep is skipped. It replaces a global
/// `max |dpos| < tol` break that could not fire (on a 906-emitter frame, 80%
/// of emitters still moved past it after eight sweeps). Do not expect the
/// queue to drain on a crowded field either: halos overlap and dirtiness
/// percolates from a few degenerate groups, collapsed pairs 0.001 px apart
/// that never converge. Neither Gauss-Seidel nor pinning the decomposition
/// fixes that (both measured), so `POLISH_SWEEPS` bounds it.
#[allow(clippy::too_many_arguments)]
fn polish(
    d: &[f64],
    h: usize,
    w: usize,
    bmap: &[f64],
    pos: &mut Vec<f64>,
    amp: &mut Vec<f64>,
    sig: &mut Vec<f64>,
    se: &mut [f64],
    s: &Settings,
    ws: &mut Workspace,
) -> usize {
    let n = amp.len();
    let mut dirty = vec![true; n];
    let mut fits = 0usize;
    let mut ems: Vec<Em> = Vec::new();
    let mut var = Vec::new();
    for _ in 0..POLISH_SWEEPS {
        let pset = patches::build_patches(pos, n, s.sigma, h, w, s.k_max);
        let (mut opos, mut oamp, mut osig) = (pos.clone(), amp.clone(), sig.clone());
        let mut moved = vec![false; n];
        let mut n_fitted = 0usize;
        for p in &pset {
            let touched = p.indices.iter().chain(&p.frozen);
            if !touched.into_iter().any(|&i| dirty[i as usize]) {
                continue;
            }
            n_fitted += 1;
            let mut win = Window::new(d, w, bmap, &p.bbox);
            ems.clear();
            ems.extend(p.frozen.iter().map(|&i| {
                let i = i as usize;
                [amp[i], pos[2 * i], pos[2 * i + 1], sig[i]]
            }));
            win.set_halo(&ems, &mut ws.f);
            let (y0, x0) = (p.bbox.y0 as f64, p.bbox.x0 as f64);
            let start: Vec<Em> = p
                .indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    [amp[i], pos[2 * i] - y0, pos[2 * i + 1] - x0, sig[i]]
                })
                .collect();
            let r = fit_window(ws, &win, win.level, &start, s, POLISH_MAX_ITER, POLISH_TOL_OBJ);
            fits += 1;
            for (j, &i) in p.indices.iter().enumerate() {
                let i = i as usize;
                let e = r.em[j];
                oamp[i] = e[0];
                opos[2 * i] = e[1] + y0;
                opos[2 * i + 1] = e[2] + x0;
                osig[i] = e[3];
                moved[i] = (opos[2 * i] - pos[2 * i]).hypot(opos[2 * i + 1] - pos[2 * i + 1])
                    > POLISH_MOVE_TOL;
            }
            // SEs from the Fisher matrix of the fit whose parameters are
            // reported. An indefinite matrix leaves the previous SEs.
            let pdim = 4 * p.indices.len() + 1;
            ws.chol.ensure(pdim);
            if ws.chol.factor(ws.fit.fisher(pdim), pdim) {
                var.resize(pdim, 0.0);
                ws.chol.inv_diag(&mut var, &mut ws.scratch);
                for (j, &i) in p.indices.iter().enumerate() {
                    for c in 0..3 {
                        let v = var[1 + 4 * j + c];
                        se[3 * i as usize + c] = if v > 0.0 { v.sqrt() } else { f64::NAN };
                    }
                }
            }
        }
        *pos = opos;
        *amp = oamp;
        *sig = osig;
        if n_fitted == 0 || !moved.iter().any(|&m| m) {
            break;
        }
        dirty = moved;
    }
    fits
}

/// Localize one raw frame: `d_e = (raw - offset) / gain + shift`, with the
/// gain estimated from the frame when `gain` is `None`. `shift` is
/// `read_noise^2` (e-^2), the shifted-Poisson term.
#[allow(clippy::too_many_arguments)]
pub fn localize_raw(
    raw: &[f64],
    h: usize,
    w: usize,
    offset: f64,
    gain: Option<f64>,
    shift: f64,
    roi: Option<&[bool]>,
    s: &Settings,
    ws: &mut Workspace,
    d: &mut Vec<f64>,
) -> Output {
    let g = gain.unwrap_or_else(|| estimate_gain(raw, h, w, offset));
    d.clear();
    d.extend(raw.iter().map(|&r| (r - offset) / g + shift));
    let mut o = localize(d, h, w, roi, s, ws);
    o.gain = g;
    o
}

/// Localize every frame of a stack on `n_threads` workers. Frames are
/// independent, so each worker takes the next undone frame and keeps its own
/// [`Workspace`]; the output is in frame order whatever the scheduling.
///
/// `raw` is `n*H*W`; each frame goes through [`localize_raw`], with one
/// `gain` for the whole stack or, if `None`, one estimated per frame.
#[allow(clippy::too_many_arguments)]
pub fn localize_stack(
    raw: &[f64],
    n: usize,
    h: usize,
    w: usize,
    offset: f64,
    gain: Option<f64>,
    shift: f64,
    roi: Option<&[bool]>,
    s: &Settings,
    n_threads: usize,
) -> Vec<Output> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    assert_eq!(raw.len(), n * h * w);
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<Option<Output>>> = Mutex::new((0..n).map(|_| None).collect());
    let workers = n_threads.clamp(1, n.max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let mut ws = Workspace::new();
                let mut d = Vec::with_capacity(h * w);
                loop {
                    let t = next.fetch_add(1, Ordering::Relaxed);
                    if t >= n {
                        break;
                    }
                    let frame = &raw[t * h * w..(t + 1) * h * w];
                    let o = localize_raw(frame, h, w, offset, gain, shift, roi, s, &mut ws, &mut d);
                    out.lock().expect("no worker panics while holding it")[t] = Some(o);
                }
            });
        }
    });
    out.into_inner()
        .expect("workers joined")
        .into_iter()
        .map(|o| o.expect("every frame was taken"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_threshold_matches_the_documented_values() {
        // [`seed_threshold`]'s note: 3.70 on 64^2 at sigma 0.818, 4.43 on
        // 512^2 at sigma 1.45.
        assert!((seed_threshold(64, 64, 0.818, SEED_ALPHA) - 3.70).abs() < 5e-3);
        assert!((seed_threshold(512, 512, 1.45, SEED_ALPHA) - 4.43).abs() < 5e-3);
    }

    #[test]
    fn percentile_interpolates_as_numpy_does() {
        let v = [4.0, 1.0, 3.0, 2.0, 10.0];
        // np.percentile([1, 2, 3, 4, 10], q) for q = 0, 10, 50, 90, 100.
        for (q, want) in [(0.0, 1.0), (10.0, 1.4), (50.0, 3.0), (90.0, 7.6), (100.0, 10.0)] {
            assert!((percentile(&v, q) - want).abs() < 1e-12, "q={q}");
        }
    }

    #[test]
    fn an_isolated_emitter_is_found_once_where_it_is() {
        let (h, w, sigma) = (31, 33, 1.2);
        let theta = [0.0, 1500.0, 14.3, 17.6, 1.3];
        let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
        let mut f = psf::Factors::new(h, w, 1);
        let mut d = vec![0.0; h * w];
        psf::model_var_sigma_ax(&theta, &ay, &ax, None, &mut f, &mut d);
        for v in d.iter_mut() {
            *v += 10.0;
        }
        let s = Settings {
            sigma,
            k_max: 12,
            seed: seed_threshold(h, w, sigma, SEED_ALPHA),
            birth: BIRTH_Z,
            slack: (0.7, 2.2),
            sweeps: SWEEPS,
            polish: true,
            band: Some((0.8, 2.0)),
        };
        let o = localize(&d, h, w, None, &s, &mut Workspace::new());
        assert_eq!(o.class, vec![Class::Focus]);
        assert_eq!(o.amp.len(), 1, "found {:?}", o.pos);
        assert!(o.se.iter().all(|v| v.is_finite() && *v > 0.0));
        // Noise-free, so what is left is the background map's own bias: the
        // mask sits on the integer candidate, and the wing beyond it leaks
        // into the 25 px mean. Measured: 0.05 SE in position, 0.2 in flux.
        let (se_a, se_y, se_x) = (o.se[0], o.se[1], o.se[2]);
        assert!((o.pos[0] - 14.3).abs() < 0.1 * se_y, "y {}", o.pos[0]);
        assert!((o.pos[1] - 17.6).abs() < 0.1 * se_x, "x {}", o.pos[1]);
        assert!((o.amp[0] - 1500.0).abs() < 0.5 * se_a, "A {}", o.amp[0]);
        assert!((o.sig[0] - 1.3).abs() < 0.01, "sigma {}", o.sig[0]);
    }
}
