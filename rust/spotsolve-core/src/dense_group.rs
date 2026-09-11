//! Rust-owned local group search for the variable-width Gaussian model.
//!
//! One native operation optimizes a local group of interacting emitters:
//! [`search_group`] constructs nothing, but it fits, scores, compares and
//! accepts competing configurations containing births, splits and removals,
//! and commits the winner atomically. Python supplies frame data and settings
//! and adapts the result; it directs no individual fit and decides no
//! individual move.
//!
//! This is stage 2 of `docs/DENSE_DETECT.md`, specified by
//! `docs/RUST_GROUP_SEARCH_PLAN.md`. It concerns `spotsolve.detect`'s
//! variable-width model only. It is not the calibrated PSF model in
//! `spotsolve.inference`, it makes no claim about calibrated source-presence
//! probabilities, and it does not change that API's uncertainty contract.
//!
//! # What replaces the pass-based decision rules
//!
//! `passes.rs` and `core.py` decide each move with a Bayes factor between two
//! configurations, plus -- on the removal path only -- a forced `A/SE <
//! PRUNE_TAU` override. The asymmetry is deliberate there and documented as a
//! termination safeguard: with it, removal is not the reverse of birth, so
//! growth and removal must run in separate monotone phases or the search
//! cycles.
//!
//! Here there is **one score for a fitted configuration in a fixed context**,
//! and every move is the difference of two such scores:
//!
//! ```text
//! score = -I_data + log p(configuration)
//!         + p/2 * log(2*pi) - logdet(curvature)/2
//! gain(candidate, incumbent) = score(candidate) - score(incumbent)
//! ```
//!
//! Birth, split and removal are compared against each other and against the
//! incumbent by that one expression, so the search can interleave them without
//! a monotone-count safeguard: a strict score increase in a fixed context
//! cannot be obtained by a move and then by its exact inverse.
//!
//! The forced pruning rule is **not** reproduced, in either direction. Its
//! recall failures are recorded in `evidence.rs`'s header (68-79% of every true
//! emitter lost inside 2 sigma died at that guard) and moving it to a birth
//! veto would reintroduce the same asymmetry with the sign flipped. What
//! replaces it is a symmetric validity policy: a configuration whose Laplace
//! score cannot be computed is *unsupported*, on either side of any
//! comparison, and an unsupported hypothesis is not evidence for its
//! alternative.
//!
//! # What this deliberately does not claim
//!
//! * The curvature is the fit's expected Fisher information plus the width
//!   prior's clipped curvature. It is **not** an exact observed posterior
//!   Hessian, so the score is not exact evidence -- see [`Hypothesis::score`].
//! * A finished search is locally best among the proposals it generated in one
//!   context. [`SearchStatus`] never says "converged" and never says "optimal".
//! * Overlapping groups produce conditional scores. Their sum is not a frame
//!   evidence and improving one does not prove global monotonicity.

use crate::linalg::{self, Chol};
use crate::lmcl::{self, Bounds, FitInfo, FitOpts, FitWorkspace, FluxPenalty, Penalty, WidthPenalty};
use crate::moves;
use crate::patches::{BBOX_PAD, HALO_FACTOR, LINK_FACTOR};
use crate::psf;

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// A stable emitter identity, separate from any array position.
///
/// Array positions renumber whenever a removal compacts a vector, and the
/// group search removes things in the middle of its own state. An id is
/// minted once by [`IdAllocator`], is never reused within one allocator, and
/// survives every refit and every neighbour's deletion.
pub type EmitterId = u32;

/// Mints ids that are unique for the lifetime of one frame.
///
/// Held by the caller across transactions rather than by a context, because
/// two groups in the same frame must not mint the same id.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdAllocator {
    next: EmitterId,
}

impl IdAllocator {
    /// Start minting above every id already in use.
    pub fn starting_at(next: EmitterId) -> Self {
        Self { next }
    }

    pub fn mint(&mut self) -> EmitterId {
        let id = self.next;
        self.next += 1;
        id
    }

    pub fn peek(&self) -> EmitterId {
        self.next
    }

    /// Guarantee that `id` will never be minted again.
    pub fn reserve_through(&mut self, id: EmitterId) {
        self.next = self.next.max(id + 1);
    }
}

/// One emitter, in GLOBAL frame pixel coordinates.
///
/// Global rather than window-local because an emitter outlives the context it
/// was last fitted in, and a local coordinate is meaningless once the region
/// moves. [`GroupContext::to_local`] and [`GroupContext::to_global`] are the
/// only places the two frames meet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emitter {
    pub id: EmitterId,
    pub y: f64,
    pub x: f64,
    /// Total flux, in photoelectrons -- not peak height.
    pub flux: f64,
    pub sigma: f64,
}

/// A configuration: the free background scalar plus every free emitter.
///
/// The emitter list carries **every** modelled source in the group, including
/// nuisance objects whose fitted width is outside the reporting band. Dropping
/// those during the search would put their light straight back into the
/// residual, where it refills the candidate list -- which is the tiling the
/// free width exists to stop (`core.py::SIGMA_SLACK`). The reporting split
/// happens once, downstream, and never inside a search.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GroupState {
    pub background: f64,
    pub emitters: Vec<Emitter>,
}

impl GroupState {
    pub fn len(&self) -> usize {
        self.emitters.len()
    }
    pub fn is_empty(&self) -> bool {
        self.emitters.is_empty()
    }
    fn index_of(&self, id: EmitterId) -> Option<usize> {
        self.emitters.iter().position(|e| e.id == id)
    }
}

// ---------------------------------------------------------------------------
// Priors, natively
// ---------------------------------------------------------------------------

/// The flux density, evaluated in Rust.
///
/// Only the exponential is representable. A curved flux prior also contributes
/// a `Lambda` block on the amplitudes that the Laplace volume here does not
/// carry, so accepting one would silently score a posterior nobody computed --
/// `prior.py` records that this is exactly why the NPMLE estimator was removed
/// rather than wired in. The binding rejects anything else explicitly instead
/// of calling back into Python or quietly substituting this one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FluxPrior {
    /// `g(A) = (1/a_s) exp(-A/a_s)`, `A > 0`.
    Exponential { a_s: f64 },
}

impl FluxPrior {
    pub fn logpdf(&self, a: f64) -> f64 {
        match *self {
            FluxPrior::Exponential { a_s } => {
                if a > 0.0 {
                    -a_s.ln() - a / a_s
                } else {
                    f64::NEG_INFINITY
                }
            }
        }
    }

    /// `sum_k log g(A_k)` over a whole configuration.
    ///
    /// The whole vector, not "the cost of the added emitter": in a joint refit
    /// every incumbent's amplitude moves too, and under a general `g` its
    /// prior contribution moves with it. For a split the parent's flux is
    /// merely redistributed, and this form prices one bright emitter against
    /// two faint ones correctly without a special case.
    pub fn log_config(&self, amplitudes: impl Iterator<Item = f64>) -> f64 {
        amplitudes.map(|a| self.logpdf(a)).sum()
    }

    /// The same density as a continuous MAP penalty for the fit.
    pub fn penalty(&self) -> Option<FluxPenalty> {
        match *self {
            FluxPrior::Exponential { a_s } => Some(FluxPenalty { a_s }),
        }
    }
}

/// The count-and-width part of the configuration prior, evaluated in Rust.
///
/// Ports `prior.UniformWidth` and `prior.FocusMixtureWidth`, including their
/// normalizers. The split between the smooth per-emitter density and the
/// per-class count terms is the whole design and is preserved here: the
/// density is what the *fit* is penalized by, and the count terms jump at the
/// class boundary and belong to model selection. See [`WidthPrior::penalty`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WidthPrior {
    /// One class, `sigma ~ U[lo, hi]`. Flat, so the MAP fit is the ML fit.
    Uniform { lam: f64, lo: f64, hi: f64 },
    /// Two classes split at `mid`, one continuous Cauchy density over
    /// `[lo, hi]` centred at `sigma0` with half-width `scale`.
    FocusMixture {
        lam_focus: f64,
        lam_wide: f64,
        lo: f64,
        mid: f64,
        hi: f64,
        sigma0: f64,
        scale: f64,
    },
}

impl WidthPrior {
    /// Normalizer of the Cauchy core over `[lo, hi]`, in logs.
    ///
    /// Computed here rather than handed in, so the density really is a density
    /// on the model space the bounds define. `prior.FocusMixtureWidth._logZ`
    /// is the same expression; agreement is asserted by the binding's tests
    /// rather than by passing the Python value across.
    pub fn log_z(&self) -> f64 {
        match *self {
            WidthPrior::Uniform { lo, hi, .. } => (hi - lo).ln(),
            WidthPrior::FocusMixture {
                lo,
                hi,
                sigma0,
                scale,
                ..
            } => scale.ln() + (((hi - sigma0) / scale).atan() - ((lo - sigma0) / scale).atan()).ln(),
        }
    }

    pub fn logpdf(&self, sigma: f64) -> f64 {
        match *self {
            WidthPrior::Uniform { .. } => -self.log_z(),
            WidthPrior::FocusMixture { sigma0, scale, .. } => {
                let u = (sigma - sigma0) / scale;
                -(u * u).ln_1p() - self.log_z()
            }
        }
    }

    /// `log p(K, classes, widths)` for one configuration, area-free.
    ///
    /// # Where the area went, and when the cancellation is allowed
    ///
    /// Under a Poisson process of rate `lam` on a region `R`, the count in `R`
    /// is `Poisson(lam*|R|)` and positions are uniform with density `1/|R|`.
    /// Their product is
    ///
    /// ```text
    /// e^{-lam|R|} (lam|R|)^K / K! * |R|^{-K} = e^{-lam|R|} * lam^K / K!
    /// ```
    ///
    /// so `|R|` survives only in a factor that is the same for every
    /// hypothesis and drops out of any difference. That is what makes this
    /// area-free -- and it holds **only** because every hypothesis in one
    /// comparison places its emitters in the same admissible region. That
    /// invariant is [`GroupBounds`]'s job, not this function's; compare two
    /// configurations fitted on different regions and this term is wrong.
    ///
    /// `1/K!` is the labeling factor for the unordered representation used
    /// here: a `GroupState` is a *set* of emitters and any permutation of the
    /// vector describes the same configuration.
    pub fn log_config(&self, sigmas: &[f64]) -> f64 {
        if sigmas.is_empty() {
            return 0.0;
        }
        let density: f64 = sigmas.iter().map(|&s| self.logpdf(s)).sum();
        match *self {
            WidthPrior::Uniform { lam, .. } => {
                let k = sigmas.len();
                k as f64 * lam.ln() - ln_factorial(k) + density
            }
            WidthPrior::FocusMixture {
                lam_focus,
                lam_wide,
                mid,
                ..
            } => {
                let k_f = sigmas.iter().filter(|&&s| s <= mid).count();
                let k_w = sigmas.len() - k_f;
                k_f as f64 * lam_focus.ln() - ln_factorial(k_f) + k_w as f64 * lam_wide.ln()
                    - ln_factorial(k_w)
                    + density
            }
        }
    }

    /// Which class a fitted width falls in. A FUNCTION of the width, never a
    /// parameter of its own, which is why nothing downstream carries a label.
    pub fn in_focus(&self, sigma: f64) -> bool {
        match *self {
            WidthPrior::Uniform { .. } => true,
            WidthPrior::FocusMixture { mid, .. } => sigma <= mid,
        }
    }

    /// The width interval of the class `sigma` falls in, intersected with the
    /// fit's own `[lo, hi]`.
    ///
    /// This is the width support a configuration's SCORE integrates over. The
    /// score is `log p(data, K, classes)`: the class labels are part of the
    /// configuration, because the count terms in [`WidthPrior::log_config`]
    /// jump at `mid`. So the Laplace volume of a focused emitter is the mass
    /// of widths on the focused side only, and mass across `mid` belongs to a
    /// different configuration with different count terms. The fit is not
    /// bounded at `mid` -- its penalty is continuous there -- so this can
    /// truncate the volume without ever being an active constraint of the fit.
    pub fn class_interval(&self, sigma: f64, lo: f64, hi: f64) -> (f64, f64) {
        match *self {
            WidthPrior::Uniform { .. } => (lo, hi),
            WidthPrior::FocusMixture { mid, .. } => {
                if sigma <= mid {
                    (lo, mid.min(hi))
                } else {
                    (mid.max(lo), hi)
                }
            }
        }
    }

    /// The smooth density as a continuous MAP penalty for the fit, or `None`
    /// when it is flat.
    ///
    /// Flat means *no term at all*, not a constant one: the LM loop compares
    /// `i_cur - i_trial` against `tol_obj = 1e-8`, and a constant added to both
    /// sides costs low-order bits of that subtraction.
    /// `prior.WidthPrior.is_flat` records the same reasoning.
    pub fn penalty(&self) -> Option<WidthPenalty> {
        match *self {
            WidthPrior::Uniform { .. } => None,
            WidthPrior::FocusMixture { sigma0, scale, .. } => Some(WidthPenalty {
                sigma0,
                scale,
                log_z: self.log_z(),
            }),
        }
    }
}

/// `log(k!)`, summed rather than taken from `lgamma`.
///
/// `k` is at most [`linalg::K_MAX_GROUP`], so this is a loop of at most a
/// dozen `ln`s -- cheaper than `lgamma` and free of any question about which
/// platform's `lgamma` was linked.
fn ln_factorial(k: usize) -> f64 {
    (2..=k).map(|i| (i as f64).ln()).sum()
}

/// Everything the score charges, frozen for the duration of a transaction.
///
/// A snapshot, not a live reference: `detect` re-estimates `lam`, `A_s` and
/// the two class rates between frame epochs, and a comparison whose prior
/// changed halfway through is not a comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PriorSnapshot {
    pub flux: FluxPrior,
    pub width: WidthPrior,
}

impl PriorSnapshot {
    /// `log p(configuration)` for a fitted `theta`, counting each term once.
    ///
    /// Contains the count, position (via the area cancellation), flux and
    /// width terms. It does **not** contain the background's uniform prior:
    /// `b` has the same range in every hypothesis of one context, so its
    /// `-log(b_max)` is a constant of the context and cancels from every
    /// difference. Adding it would change no gain and would make the absolute
    /// score look like a quantity it is not.
    fn log_config(&self, theta: &[f64], sigmas: &mut Vec<f64>) -> f64 {
        let k = psf::n_emitters_var(theta);
        sigmas.clear();
        sigmas.extend((0..k).map(|i| psf::sigma_var(theta, i)));
        self.flux
            .log_config((0..k).map(|i| psf::amp_var(theta, i)))
            + self.width.log_config(sigmas)
    }

    /// The continuous half of the same prior, as the fit's MAP penalty.
    ///
    /// This is the point of the type: the fit and the score are charged by one
    /// object, so they cannot disagree about what a width or a flux costs. A
    /// fit and an evidence that disagree will disagree about what exists.
    pub fn penalty(&self) -> Penalty {
        Penalty {
            width: self.width.penalty(),
            flux: self.flux.penalty(),
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry and the comparison context
// ---------------------------------------------------------------------------

/// The amplitude floor as a fraction of the window's own `A_max`.
///
/// `core.py::A_MIN_REL`'s constant and its measurement: at an absolute `1e-4`
/// floor a parked emitter's position block sits ~130x float64 epsilon below
/// the largest diagonal, `log|F|` is noise, and the LM step along it is
/// unbounded. `1e-6` of `A_max` buys six orders of margin while remaining
/// physically negligible (~0.02 e- of total flux on a bead patch).
pub const A_MIN_REL: f64 = 1e-6;

/// How far, in units of the largest admissible width, a source may travel
/// inside one transaction.
///
/// The admissible position box is the entry configuration's bounding box grown
/// by this much, and the pixel region is that box grown again by
/// `BBOX_PAD * sigma_hi`. Two separate paddings, because they answer different
/// questions: this one bounds where a source may *go*, and `BBOX_PAD` bounds
/// how much context it needs once it is there.
///
/// Sized against the largest displacement a proposal can construct: a split at
/// `SPLIT_DISPS`'s upper end of 1.6 puts each child `0.8 * sigma_parent` from
/// the parent, and the joint refit then moves them further. `1.0 * sigma_hi`
/// covers that with room, and [`GroupDiagnostics::position_bound_active`]
/// reports whether the bound was ever reached so the margin is measured rather
/// than asserted.
pub const DRIFT_FACTOR: f64 = 1.0;

/// Box constraints for one context, shared by every hypothesis in it.
///
/// Positions and widths are in LOCAL window coordinates and pixels. Holding
/// these fixed is what makes two configurations comparable at all: the
/// area cancellation in [`WidthPrior::log_config`] assumes one admissible
/// region, and a Laplace volume taken over a different support is a different
/// integral.
#[derive(Clone, Copy, Debug)]
pub struct GroupBounds {
    pub b_max: f64,
    pub a_min: f64,
    pub a_max: f64,
    pub y_lo: f64,
    pub y_hi: f64,
    pub x_lo: f64,
    pub x_hi: f64,
    pub sigma_lo: f64,
    pub sigma_hi: f64,
}

impl GroupBounds {
    /// `(lo, hi)` for a `4*k + 1` parameter vector.
    pub fn arrays(&self, k: usize) -> (Vec<f64>, Vec<f64>) {
        let mut lo = Vec::with_capacity(4 * k + 1);
        let mut hi = Vec::with_capacity(4 * k + 1);
        lo.push(0.0);
        hi.push(self.b_max);
        for _ in 0..k {
            lo.extend_from_slice(&[self.a_min, self.y_lo, self.x_lo, self.sigma_lo]);
            hi.extend_from_slice(&[self.a_max, self.y_hi, self.x_hi, self.sigma_hi]);
        }
        (lo, hi)
    }
}

/// A proposal seed: a place worth trying, with no statistical standing.
///
/// Seeds help the optimizer reach a basin. They do not define the score, they
/// are not screened, and a seed that produces a worse configuration is simply
/// outscored.
#[derive(Clone, Copy, Debug)]
pub struct Seed {
    /// GLOBAL frame coordinates.
    pub y: f64,
    pub x: f64,
    /// Total-flux start. Zero or negative means "derive it from the residual".
    pub flux: f64,
}

/// The fixed spatial and statistical context a transaction compares within.
///
/// Everything here is immutable for the duration of [`search_group`]:
/// the pixel region, the observations, the halo, the background
/// parameterization, the parameter support and the prior snapshot. Nothing in
/// the search may compare objectives evaluated on different cropped regions.
pub struct GroupContext {
    /// Bumped by the caller whenever anything this context was built from
    /// changes. A cached score from a different version is not comparable and
    /// must be discarded, not adjusted.
    pub version: u64,
    /// Region origin in the frame, and its extent.
    pub y0: usize,
    pub x0: usize,
    pub h: usize,
    pub w: usize,
    /// Observations over the region, photoelectrons, row-major.
    pub obs: Vec<f64>,
    /// The parameter-free additive term: the background surface's *shape*
    /// plus every frozen emitter rendered at its OWN width.
    pub halo: Vec<f64>,
    /// The background's level over the region; the fit's free `b` starts here.
    pub bg_level: f64,
    /// Entry state of the free emitters. Every hypothesis has these same free
    /// neighbours; the only difference between hypotheses is the one emitter
    /// being added or removed.
    pub free: Vec<Emitter>,
    /// Emitters held fixed in every hypothesis, already folded into `halo`.
    /// Retained for diagnostics. Coupling to them is conditioned on, never a
    /// reason to rebuild: see `search_group`'s position-bound check.
    pub frozen: Vec<Emitter>,
    pub bounds: GroupBounds,
    pub seeds: Vec<Seed>,
    pub prior: PriorSnapshot,
    /// The PSF width the frame was acquired at.
    pub sigma0: f64,
    /// The free set was truncated to `k_max`; some coupled neighbours are
    /// frozen that the geometry says should be free.
    pub capacity_limited: bool,
    /// The region or the position box was clipped by the frame edge, so a
    /// source here has less pixel context than the geometry asked for.
    pub edge_clipped: bool,
    ay: Vec<f64>,
    ax: Vec<f64>,
}

impl GroupContext {
    #[inline]
    pub fn to_local(&self, y: f64, x: f64) -> (f64, f64) {
        (y - self.y0 as f64, x - self.x0 as f64)
    }

    #[inline]
    pub fn to_global(&self, y: f64, x: f64) -> (f64, f64) {
        (y + self.y0 as f64, x + self.x0 as f64)
    }

    #[inline]
    pub fn n_pixels(&self) -> usize {
        self.h * self.w
    }

    /// Pack a state into a fit vector, in local coordinates.
    pub fn pack_state(&self, state: &GroupState, out: &mut Vec<f64>) {
        out.clear();
        out.push(state.background);
        for e in &state.emitters {
            let (y, x) = self.to_local(e.y, e.x);
            out.extend_from_slice(&[e.flux, y, x, e.sigma]);
        }
    }

    /// Read a fit vector back into a state, keeping `ids` aligned with it.
    fn unpack(&self, theta: &[f64], ids: &[EmitterId]) -> GroupState {
        let k = psf::n_emitters_var(theta);
        debug_assert_eq!(k, ids.len());
        let emitters = (0..k)
            .map(|i| {
                let (y, x) = self.to_global(psf::cy_var(theta, i), psf::cx_var(theta, i));
                Emitter {
                    id: ids[i],
                    y,
                    x,
                    flux: psf::amp_var(theta, i),
                    sigma: psf::sigma_var(theta, i),
                }
            })
            .collect();
        GroupState {
            background: psf::background(theta),
            emitters,
        }
    }
}

/// A borrowed view of one frame's arrays.
pub struct FrameView<'a> {
    /// Observations in photoelectrons, row-major `h * w`.
    pub d: &'a [f64],
    /// The background surface, same shape.
    pub bmap: &'a [f64],
    pub h: usize,
    pub w: usize,
}

/// Everything context construction needs that is not the frame itself.
#[derive(Clone, Copy, Debug)]
pub struct ContextSpec {
    pub sigma0: f64,
    /// `(lo, hi)` multiples of `sigma0` bounding the model space.
    pub slack: (f64, f64),
    /// Compute limit on the free set. NOT a statement of statistical
    /// independence -- see [`GroupContext::capacity_limited`].
    pub k_max: usize,
    pub prior: PriorSnapshot,
}

/// Build the comparison context around a focus point.
///
/// # Width-aware support
///
/// Every radius is measured at the LARGEST width the transaction may reach,
/// not at the width an emitter happens to have now. A source that widens
/// during the search must not escape the region on which its alternatives are
/// being compared, and a neighbour that a widened source would start to
/// overlap must already be free rather than frozen. Both radii therefore use
/// `sigma_hi = slack.1 * sigma0`:
///
/// * free, `LINK_FACTOR * sigma_hi` around the focus;
/// * region, the free set's bounding box grown by `DRIFT_FACTOR * sigma_hi`
///   (the position box) and then by `BBOX_PAD * sigma_hi` (pixel context);
/// * frozen, `HALO_FACTOR * sigma_i` from the region rectangle, at each
///   candidate's own width, because a frozen source does not change width.
///
/// This is deliberately conservative geometry. Approximating weak couplings
/// for speed comes after the residual and edge controls establish that the
/// omitted light is negligible.
///
/// The focus emitter, when one is named, is always in the free set: the free
/// list is ordered by distance to the focus before it is capped, so capacity
/// pressure freezes the farthest coupled neighbour and never the source being
/// tested.
pub fn build_context(
    frame: &FrameView<'_>,
    emitters: &[Emitter],
    focus: (f64, f64),
    seeds: &[Seed],
    spec: &ContextSpec,
    version: u64,
) -> GroupContext {
    let sigma_hi = spec.slack.1 * spec.sigma0;
    let sigma_lo = spec.slack.0 * spec.sigma0;
    let link_r = LINK_FACTOR * sigma_hi;

    // Free: coupled to the focus at the widest admissible width, nearest
    // first, capped. Index is the tiebreak so the order is defined.
    let mut near: Vec<(f64, usize)> = emitters
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let (dy, dx) = (e.y - focus.0, e.x - focus.1);
            (dy * dy + dx * dx, i)
        })
        .filter(|(d2, _)| *d2 <= link_r * link_r)
        .collect();
    near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    let capacity_limited = near.len() > spec.k_max;
    near.truncate(spec.k_max);
    let free: Vec<Emitter> = near.iter().map(|&(_, i)| emitters[i]).collect();
    let free_set: Vec<usize> = near.iter().map(|&(_, i)| i).collect();

    // The position box: where a source in this group may go.
    let (mut ylo, mut yhi) = (focus.0, focus.0);
    let (mut xlo, mut xhi) = (focus.1, focus.1);
    for e in &free {
        ylo = ylo.min(e.y);
        yhi = yhi.max(e.y);
        xlo = xlo.min(e.x);
        xhi = xhi.max(e.x);
    }
    for s in seeds {
        ylo = ylo.min(s.y);
        yhi = yhi.max(s.y);
        xlo = xlo.min(s.x);
        xhi = xhi.max(s.x);
    }
    let drift = DRIFT_FACTOR * sigma_hi;
    let (pos_y0, pos_y1) = (ylo - drift, yhi + drift);
    let (pos_x0, pos_x1) = (xlo - drift, xhi + drift);

    // The pixel region: the position box plus context for the widest source.
    let pad = BBOX_PAD * sigma_hi;
    let y0 = (pos_y0 - pad).floor().max(0.0) as usize;
    let x0 = (pos_x0 - pad).floor().max(0.0) as usize;
    let y1 = (((pos_y1 + pad).ceil() as i64 + 1).max(0) as usize).min(frame.h);
    let x1 = (((pos_x1 + pad).ceil() as i64 + 1).max(0) as usize).min(frame.w);
    let (y1, x1) = (y1.max(y0 + 1), x1.max(x0 + 1));
    let (h, w) = (y1 - y0, x1 - x0);
    let edge_clipped = (pos_y0 - pad) < 0.0
        || (pos_x0 - pad) < 0.0
        || (pos_y1 + pad) >= frame.h as f64
        || (pos_x1 + pad) >= frame.w as f64;

    // Observations and the background split. The fit keeps a free scalar `b`
    // started at the level; the surface's variation across the region enters
    // as a known additive term, like a frozen emitter. Folding the whole
    // surface into the known term would leave `b` wanting to be 0, which is
    // its lower bound, and Coleman-Li scaling collapses every coordinate's
    // step when any one parameter sits on a bound.
    let mut obs = Vec::with_capacity(h * w);
    let mut bg = Vec::with_capacity(h * w);
    for r in 0..h {
        let src = (y0 + r) * frame.w + x0;
        obs.extend_from_slice(&frame.d[src..src + w]);
        bg.extend_from_slice(&frame.bmap[src..src + w]);
    }
    let bg_level = median(&mut bg.clone());
    let mut halo: Vec<f64> = bg.iter().map(|v| v - bg_level).collect();

    // Frozen: everything else near enough to contribute flux at its own width.
    let ay = psf::local_axis(h);
    let ax = psf::local_axis(w);
    let mut frozen = Vec::new();
    let mut factors = psf::Factors::new(h, w, 1);
    for (i, e) in emitters.iter().enumerate() {
        if free_set.contains(&i) {
            continue;
        }
        if dist_to_rect(e.y, e.x, y0 as f64, x0 as f64, (y1 - 1) as f64, (x1 - 1) as f64)
            > HALO_FACTOR * e.sigma.max(spec.sigma0)
        {
            continue;
        }
        frozen.push(*e);
        let (ly, lx) = (e.y - y0 as f64, e.x - x0 as f64);
        // Rendered over the whole region rather than truncated. The region is
        // small, the arithmetic is once per context, and a truncation radius
        // here would be a second, silently different definition of a frozen
        // emitter's reach from the one the halo radius above states.
        add_gaussian(&ay, &ax, e.flux, ly, lx, e.sigma, &mut factors, &mut halo);
    }

    let smax = obs.iter().fold(1.0f64, |a, &v| a.max(v));
    let a_max = 8.0 * smax / psf::peak_factor(spec.sigma0) * spec.slack.1 * spec.slack.1;
    let bounds = GroupBounds {
        b_max: (smax * 4.0).max(10.0),
        a_min: moves::A_MIN.max(A_MIN_REL * a_max),
        a_max,
        y_lo: (pos_y0 - y0 as f64).max(-0.5),
        y_hi: (pos_y1 - y0 as f64).min(h as f64 - 0.5),
        x_lo: (pos_x0 - x0 as f64).max(-0.5),
        x_hi: (pos_x1 - x0 as f64).min(w as f64 - 0.5),
        sigma_lo,
        sigma_hi,
    };

    GroupContext {
        version,
        y0,
        x0,
        h,
        w,
        obs,
        halo,
        bg_level,
        free,
        frozen,
        bounds,
        seeds: seeds.to_vec(),
        prior: spec.prior,
        sigma0: spec.sigma0,
        capacity_limited,
        edge_clipped,
        ay,
        ax,
    }
}

/// The entry configuration implied by a context's free set.
pub fn entry_state(ctx: &GroupContext) -> GroupState {
    GroupState {
        background: ctx.bg_level,
        emitters: ctx.free.clone(),
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn dist_to_rect(y: f64, x: f64, y0: f64, x0: f64, y1: f64, x1: f64) -> f64 {
    let py = y.clamp(y0, y1);
    let px = x.clamp(x0, x1);
    ((y - py).powi(2) + (x - px).powi(2)).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn add_gaussian(
    ay: &[f64],
    ax: &[f64],
    flux: f64,
    cy: f64,
    cx: f64,
    sigma: f64,
    f: &mut psf::Factors,
    out: &mut [f64],
) {
    let (h, w) = (ay.len(), ax.len());
    f.ensure(h, w, 1);
    let mut ey = vec![0.0; h];
    let mut ex = vec![0.0; w];
    psf::shape_axis(ay, &[cy], sigma, &mut ey);
    psf::shape_axis(ax, &[cx], sigma, &mut ex);
    for r in 0..h {
        let a_e = flux * ey[r];
        for c in 0..w {
            out[r * w + c] += a_e * ex[c];
        }
    }
}

// ---------------------------------------------------------------------------
// Scoring one configuration
// ---------------------------------------------------------------------------

/// Upper limit on the diagonally scaled condition number of the curvature,
/// above which its `logdet` -- and so the whole Laplace volume -- is numerical
/// noise rather than a quantity.
///
/// # Why not `COND_GUARD`
///
/// `evidence::COND_GUARD = 1e3` is applied on the fixed-width path to the
/// *proposal* only, so it functions there as an asymmetric detection rule: an
/// ill-conditioned larger model is refused and an ill-conditioned incumbent is
/// not. Reusing that number and that placement here would rebuild the
/// asymmetry this stage exists to remove, and 1e3 was never calibrated against
/// the `4*K+1` layout, whose width block is far more collinear with amplitude
/// than anything in `3*K+1`.
///
/// This constant is a **numerical validity limit**, applied identically to
/// every hypothesis including the incumbent. It is set from the precision of
/// the quantity it protects. `scripts/measure_group_score.py` factorizes
/// `p = 53` matrices whose log-determinant is known exactly by construction,
/// with the same Cholesky the score uses:
///
/// | scaled cond | max `logdet` error |
/// |---|---|
/// | 1e0 | 1.5e-14 |
/// | 1e2 | 3.6e-14 |
/// | 1e4 | 7.7e-13 |
/// | **1e6** | **3.9e-11** |
/// | 1e8 | 2.8e-9 |
/// | 1e10 | 5.4e-7 |
/// | 1e12 | 1.7e-5 |
///
/// Precision alone would permit 1e10, where the error is still 200x under
/// [`SCORE_TOL`]. 1e6 is kept because it costs nothing: on the same script's
/// detector groups the realized scaled condition number has median 61, p99
/// 2.1e3 and maximum 2.6e3, and `evidence.rs` measures a genuinely degenerate
/// pair at 0.5 sigma at 1.8e4 scaled. So the limit sits ~50x above the worst
/// thing it should ever have to accept and ~380x above anything these controls
/// produce -- which is the evidence that it is a validity guard and not a
/// detection rule wearing one's clothes.
pub const COND_LIMIT: f64 = 1e6;

/// Relative distance to a bound, below which a fitted parameter counts as
/// resting ON it.
///
/// `lmcl::Interior` keeps every parameter strictly inside by `1e-10` of its
/// range, so a parameter driven to a bound comes back at exactly that offset
/// and never at the bound itself. This threshold is four orders above that
/// margin -- far enough to catch the pinned case, small enough that an
/// interior mode a hundredth of a pixel from the edge is not mistaken for one.
pub const BOUND_TOL: f64 = 1e-6;

// ---------------------------------------------------------------------------
// The Laplace volume over the admissible box
// ---------------------------------------------------------------------------

/// `log Phi_c(u) = log P(Z > u)`, accurate in the upper tail.
///
/// `erfc` is exact to the last digits until it underflows near `u = 37`, so
/// it is used directly up to 30 and the asymptotic series takes over after;
/// the series' first omitted term at 30 is ~1e-10 relative.
pub fn log_ndtr_upper(u: f64) -> f64 {
    if u < 30.0 {
        (0.5 * libm::erfc(u / std::f64::consts::SQRT_2)).ln()
    } else {
        let u2 = u * u;
        -0.5 * u2 - u.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
            + (1.0 - 1.0 / u2 + 3.0 / (u2 * u2) - 15.0 / (u2 * u2 * u2)).ln()
    }
}

/// `log(Phi(u2) - Phi(u1))` for `u1 <= u2`, without cancellation.
///
/// Three regimes, each evaluated where it has no subtraction of nearly equal
/// numbers: both points in the upper tail, both in the lower tail (the
/// mirror), or straddling zero, where the two `erf`s have opposite signs and
/// ADD.
pub fn log_ndtr_diff(u1: f64, u2: f64) -> f64 {
    debug_assert!(u1 <= u2);
    if u1 >= 0.0 {
        let a = log_ndtr_upper(u1);
        let b = log_ndtr_upper(u2);
        a + (-(b - a).exp()).ln_1p()
    } else if u2 <= 0.0 {
        log_ndtr_diff(-u2, -u1)
    } else {
        let k = std::f64::consts::FRAC_1_SQRT_2;
        (0.5 * (libm::erf(u2 * k) - libm::erf(u1 * k))).ln()
    }
}

/// One coordinate's correction to the Laplace volume for a bounded support.
///
/// The exact integral of the one-dimensional quadratic model over the
/// coordinate's admissible interval, divided by the full-line Gaussian volume
/// `sqrt(2*pi)*s` the regular Laplace expression already charges:
///
/// ```text
/// int_{lo-theta}^{hi-theta} exp(-g t - t^2/(2 s^2)) dt / (sqrt(2 pi) s)
///   = exp(s^2 g^2 / 2) * (Phi(u2) - Phi(u1)),
/// u1 = (lo - theta)/s + s g,   u2 = (hi - theta)/s + s g
/// ```
///
/// `s` is the MARGINAL standard deviation `sqrt((F^-1)_qq)` and `g` the KKT
/// multiplier (zero off the bound). Three limits say what it does:
///
/// * a well-determined interior mode: `Phi(u2) - Phi(u1) -> 1`, correction 0;
/// * a mode resting on a bound with `g = 0`: exactly `log(1/2)`;
/// * a Gaussian far wider than the interval: `-> log(width / (sqrt(2 pi) s))`,
///   which caps the volume at the prior's own support.
pub fn log_box_factor(theta: f64, lo: f64, hi: f64, s: f64, g: f64) -> f64 {
    let sg = s * g;
    let u1 = (lo - theta) / s + sg;
    let u2 = (hi - theta) / s + sg;
    0.5 * sg * sg + log_ndtr_diff(u1.min(u2), u2)
}

/// `log` of the fraction of the Laplace Gaussian's mass that lies inside the
/// admissible box, one coordinate at a time.
///
/// # Why the regular expression is not enough
///
/// The regular Laplace volume `(2 pi)^{p/2} |F|^{-1/2}` integrates the
/// quadratic model over ALL of `R^p`. The prior has support only on the
/// admissible box -- widths in their class interval, positions in the
/// transaction's position box, amplitudes positive -- so that integral
/// includes mass the model gives zero probability. Two consequences, both
/// measured on the gate's controls before this existed:
///
/// * **A mode resting on a bound was unscorable.** The redundant emitter of an
///   over-fitted group is squeezed to `sigma_lo`, a phantom drifts to the edge
///   of the position box, a genuinely narrow source sits at the width floor.
///   The integral there is one-sided, the regular expression overstates it by
///   at least `log 2` per active bound, and the old policy refused to score
///   it -- 40% of overfit starts, 45% of the degenerate control.
/// * **A near-zero-flux emitter was scored too WELL.** Its position and width
///   curvature scales with `A^2`, so `-logdet/2` grows without bound as `A`
///   falls, and a Gaussian many box-widths wide is credited as volume. The
///   diagonally scaled condition number cannot see this -- a block that
///   shrinks as a whole is perfectly conditioned -- so two overfit trials
///   accepted a split to a 0.05 e- emitter at +12 and +15 nats.
///
/// Integrating the same quadratic model over the box fixes both with one
/// expression, applied identically to every hypothesis. It is not a removal
/// rule: an emitter on a bound whose data support it keeps a finite score and
/// loses only the mass it genuinely does not have.
///
/// # The approximation
///
/// Each coordinate is truncated in its own MARGINAL, with the others free.
/// That is exact for any number of interior coordinates plus one active
/// bound: integrating the interior coordinates over the line leaves the
/// active one with the Schur-complement curvature `1/(F^-1)_qq`, which is the
/// marginal variance used here. With several active bounds it ignores their
/// correlation; `scripts/measure_box_laplace.py` measures the resulting error
/// against importance sampling of the exact posterior over the same box.
///
/// `g` is used only where a coordinate rests on its FIT bound (within
/// [`BOUND_TOL`]) with the gradient pushing outward. Everywhere else the fit
/// is stationary, the gradient is zero to its tolerance, and including that
/// residue would only add noise.
///
/// `inv_diag` is `diag(F^-1)`; `grad` the penalized objective's gradient;
/// `(fit_lo, fit_hi)` the fit bounds and `(lo, hi)` the score's support.
#[allow(clippy::too_many_arguments)]
pub fn log_box_mass(
    theta: &[f64],
    grad: &[f64],
    inv_diag: &[f64],
    fit_lo: &[f64],
    fit_hi: &[f64],
    lo: &[f64],
    hi: &[f64],
    n_active: &mut usize,
) -> f64 {
    let mut total = 0.0;
    *n_active = 0;
    for q in 0..theta.len() {
        let s = inv_diag[q].sqrt();
        let range = (fit_hi[q] - fit_lo[q]).max(1e-12);
        let g = grad[q];
        let active = ((theta[q] - fit_lo[q]) / range < BOUND_TOL && g > 0.0)
            || ((fit_hi[q] - theta[q]) / range < BOUND_TOL && g < 0.0);
        *n_active += active as usize;
        // The support is clamped to contain the mode: the class interval can
        // end exactly at a fitted width, and a mode a rounding error outside
        // its own support would otherwise produce `log(negative)`.
        let (l, h) = (lo[q].min(theta[q]), hi[q].max(theta[q]));
        total += log_box_factor(theta[q], l, h, s, if active { g } else { 0.0 });
    }
    total
}

/// Why a configuration's score is or is not usable in a comparison.
///
/// The same rules are applied to incumbents and to proposals. None of these is
/// evidence for the alternative: an unsupported hypothesis is *unknown*, not
/// *rejected*, and neither a determinant failure nor a large uncertainty
/// argues for a smaller or a larger model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreStatus {
    /// The Laplace score is computable and the mode is interior.
    Supported,
    /// The fit did not certify stationarity at the parameters it returned, so
    /// the expansion point is not a mode.
    Nonstationary,
    /// The curvature is not positive definite.
    Singular,
    /// The curvature is positive definite but its scaled condition number
    /// exceeds [`COND_LIMIT`], so `logdet` is noise.
    IllConditioned,
    /// The objective or a parameter is not finite.
    NonFinite,
}
// There is deliberately no `BoundaryMode`. A mode resting on a bound was once
// unsupported here, because the regular Laplace integral is one-sided there;
// it is now scored by integrating over the admissible box instead -- see
// [`log_box_mass`]. Refusing to score it left 40% of over-fitted groups stuck
// with an incumbent nothing could be compared against.

impl ScoreStatus {
    pub fn is_supported(self) -> bool {
        self == ScoreStatus::Supported
    }

    /// A short reason, for diagnostics and for the binding.
    pub fn reason(self) -> &'static str {
        match self {
            ScoreStatus::Supported => "supported",
            ScoreStatus::Nonstationary => "nonstationary",
            ScoreStatus::Singular => "singular_curvature",
            ScoreStatus::IllConditioned => "ill_conditioned_curvature",
            ScoreStatus::NonFinite => "non_finite",
        }
    }
}

/// One fitted configuration and everything the comparison reads off it.
#[derive(Clone, Debug)]
pub struct Hypothesis {
    /// Fitted parameters, `4*K + 1`, LOCAL coordinates.
    pub theta: Vec<f64>,
    /// The curvature this score was built on, `p x p` row-major.
    pub curvature: Vec<f64>,
    pub p: usize,
    /// Data-only Poisson I-divergence, in nats. The prior is added by the
    /// score and must not be paid twice.
    pub i_div: f64,
    /// `log p(configuration)` -- count, position, flux and width, once each.
    pub log_prior: f64,
    pub logdet: f64,
    /// Diagonally scaled condition number of the curvature.
    pub cond: f64,
    /// `log` of the Laplace Gaussian's mass inside the admissible box, `<= 0`
    /// up to rounding. See [`log_box_mass`].
    pub log_box: f64,
    /// Coordinates resting on a fit bound with the gradient pushing outward.
    pub n_active: usize,
    /// The configuration score.
    ///
    /// ```text
    /// score = -i_div + log_prior + p/2 * log(2*pi) - logdet/2 + log_box
    /// ```
    ///
    /// The first four terms are the regular Laplace expression, the integral
    /// of the quadratic model over all of `R^p`; `log_box` restricts it to
    /// the prior's support.
    ///
    /// # What the curvature is
    ///
    /// The fit's expected Fisher information `J^T W J` plus the width prior's
    /// clipped curvature. Three things it is not: it is not the observed
    /// Hessian of the log posterior; it is not exact, because clipping a
    /// heavy-tailed prior's curvature at zero is a deliberate approximation;
    /// and it carries no optimizer damping -- `lambda` and the Coleman-Li
    /// diagonals never enter it. So this is an *approximate* Laplace score.
    /// Calling a difference of two of them "the log Bayes factor" would
    /// overstate it, and this module never does.
    pub score: f64,
    pub status: ScoreStatus,
    pub fit: FitInfo,
    /// The penalized objective the fit actually minimized, `i_div + penalty`.
    /// Used to choose between restarts of the SAME hypothesis, where the
    /// prior's discrete terms are identical and only the fit quality differs.
    pub objective: f64,
}

impl Hypothesis {
    pub fn k(&self) -> usize {
        psf::n_emitters_var(&self.theta)
    }
}

/// Reusable storage for fits, rendering, proposals and factorizations.
///
/// Trial configurations allocate no frame-sized array: everything they touch
/// is region-sized and lives here for the whole transaction. Separate
/// instances share no mutable state, so two prepared engines can run
/// concurrently.
pub struct GroupWorkspace {
    fit: FitWorkspace,
    chol: Chol,
    factors: psf::Factors,
    model: Vec<f64>,
    resid: Vec<f64>,
    theta: Vec<f64>,
    sigmas: Vec<f64>,
    scratch: Vec<f64>,
    cond_scratch: Vec<f64>,
    inv_diag: Vec<f64>,
    /// Matched-filter response over the region, and the separable PSF factors
    /// it is built from; see `light_outside_box`.
    zscore: Vec<f64>,
    ey_all: Vec<f64>,
    ex_all: Vec<f64>,
    /// Start vector for a restart, reused so a restart allocates nothing.
    start: Vec<f64>,
}

impl Default for GroupWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl GroupWorkspace {
    pub fn new() -> Self {
        Self {
            fit: FitWorkspace::new(),
            // Sized for `4*(K_MAX+1)+1` up front rather than grown inside a
            // comparison; see `linalg::P_MAX_VAR`'s capacity audit.
            chol: Chol::new(linalg::P_MAX_VAR),
            factors: psf::Factors::new(0, 0, 0),
            model: Vec::new(),
            resid: Vec::new(),
            theta: Vec::new(),
            sigmas: Vec::new(),
            scratch: Vec::new(),
            cond_scratch: Vec::new(),
            inv_diag: Vec::new(),
            zscore: Vec::new(),
            ey_all: Vec::new(),
            ex_all: Vec::new(),
            start: Vec::new(),
        }
    }

    fn ensure(&mut self, ctx: &GroupContext, p: usize) {
        let n = ctx.n_pixels();
        self.chol.ensure(p);
        if self.model.len() < n {
            self.model.resize(n, 0.0);
            self.resid.resize(n, 0.0);
        }
        if self.inv_diag.len() < p {
            self.inv_diag.resize(p, 0.0);
        }
    }

    /// The last fit's penalized gradient, for a diagnostic referee.
    pub fn last_gradient(&self, p: usize) -> Vec<f64> {
        self.fit.gradient(p).to_vec()
    }

    /// Render the model for `theta` over the region, halo included.
    fn render(&mut self, ctx: &GroupContext, theta: &[f64]) {
        let n = ctx.n_pixels();
        self.factors.ensure(ctx.h, ctx.w, psf::n_emitters_var(theta).max(1));
        psf::model_var_sigma_ax(
            theta,
            &ctx.ay,
            &ctx.ax,
            Some(&ctx.halo),
            &mut self.factors,
            &mut self.model[..n],
        );
    }

    /// `obs - model(theta)` over the region.
    fn residual(&mut self, ctx: &GroupContext, theta: &[f64]) {
        let n = ctx.n_pixels();
        self.render(ctx, theta);
        for i in 0..n {
            self.resid[i] = ctx.obs[i] - self.model[i];
        }
    }
}

/// Fit one configuration and score it, in a fixed context.
///
/// The single entry point for every hypothesis, incumbent included, so that
/// the fit budget, the prior, the region and the validity rules are identical
/// on both sides of every comparison by construction rather than by review.
pub fn fit_and_score(
    ctx: &GroupContext,
    ws: &mut GroupWorkspace,
    theta0: &[f64],
    opts: FitOpts,
) -> Hypothesis {
    let k = psf::n_emitters_var(theta0);
    let p = 4 * k + 1;
    ws.ensure(ctx, p);

    let (lo, hi) = ctx.bounds.arrays(k);
    let bounds = Bounds::new(&lo, &hi);
    let penalty = ctx.prior.penalty();

    let info = lmcl::fit_var_sigma_prior(
        &mut ws.fit,
        theta0,
        ctx.h,
        ctx.w,
        &ctx.obs,
        &bounds,
        Some(&ctx.halo),
        opts,
        penalty,
    );

    let theta = ws.fit.theta().to_vec();
    let curvature = ws.fit.fisher(p).to_vec();
    let log_prior = ctx.prior.log_config(&theta, &mut ws.sigmas);

    let (logdet, cond, pd) = linalg::logdet_cond(&curvature, p, &mut ws.chol, &mut ws.cond_scratch);

    // The box correction reads `diag(F^-1)` off the factor `logdet_cond` just
    // left in `ws.chol`, and the KKT multipliers off the fit's own gradient.
    let mut n_active = 0;
    let log_box = if pd {
        ws.inv_diag.resize(p.max(ws.inv_diag.len()), 0.0);
        ws.chol.inv_diag(&mut ws.inv_diag[..p], &mut ws.scratch);
        let (slo, shi) = score_support(ctx, &theta, &lo, &hi);
        log_box_mass(
            &theta,
            ws.fit.gradient(p),
            &ws.inv_diag[..p],
            &lo,
            &hi,
            &slo,
            &shi,
            &mut n_active,
        )
    } else {
        f64::NAN
    };

    // The penalized objective the optimizer minimized, reconstructed so that
    // restarts of the same hypothesis can be ranked. `i_div` is data-only.
    let objective = info.i_div - log_prior_continuous(ctx, &theta, &mut ws.sigmas);

    let status = classify(&theta, &info, pd, cond, log_prior, logdet, log_box);
    let score = if status.is_supported() {
        -info.i_div + log_prior + 0.5 * p as f64 * (2.0 * std::f64::consts::PI).ln() - 0.5 * logdet
            + log_box
    } else {
        f64::NAN
    };

    Hypothesis {
        theta,
        curvature,
        p,
        i_div: info.i_div,
        log_prior,
        logdet,
        cond,
        log_box,
        n_active,
        score,
        status,
        fit: info,
        objective,
    }
}

/// The support a configuration's score integrates over: the fit bounds, with
/// each width narrowed to its class interval. See
/// [`WidthPrior::class_interval`].
pub fn score_support(ctx: &GroupContext, theta: &[f64], lo: &[f64], hi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (mut slo, mut shi) = (lo.to_vec(), hi.to_vec());
    for e in 0..psf::n_emitters_var(theta) {
        let q = 4 + 4 * e;
        let (a, b) = ctx.prior.width.class_interval(theta[q], lo[q], hi[q]);
        slo[q] = a;
        shi[q] = b;
    }
    (slo, shi)
}

/// The part of `log p(configuration)` the FIT was penalized by: the continuous
/// densities only, without the count and class terms.
///
/// Needed to reconstruct the optimizer's own objective from the data-only
/// `i_div` it returns. The count terms are excluded because they jump at the
/// class boundary and the optimizer never saw them.
fn log_prior_continuous(ctx: &GroupContext, theta: &[f64], sigmas: &mut Vec<f64>) -> f64 {
    let k = psf::n_emitters_var(theta);
    let pen = ctx.prior.penalty();
    let mut v = 0.0;
    if pen.flux.is_some() {
        v += ctx
            .prior
            .flux
            .log_config((0..k).map(|i| psf::amp_var(theta, i)));
    }
    if pen.width.is_some() {
        sigmas.clear();
        sigmas.extend((0..k).map(|i| psf::sigma_var(theta, i)));
        v += sigmas.iter().map(|&s| ctx.prior.width.logpdf(s)).sum::<f64>();
    }
    v
}

/// The symmetric validity policy. Applied to every hypothesis identically.
///
/// Where the mode sits is not a validity question any more: a mode on a bound
/// is scored by [`log_box_mass`]. What remains here is whether the expansion
/// point is a mode at all, and whether its curvature is a number.
fn classify(
    theta: &[f64],
    info: &FitInfo,
    positive_definite: bool,
    cond: f64,
    log_prior: f64,
    logdet: f64,
    log_box: f64,
) -> ScoreStatus {
    if !theta.iter().all(|v| v.is_finite())
        || !info.i_div.is_finite()
        || !log_prior.is_finite()
        || !logdet.is_finite()
    {
        return ScoreStatus::NonFinite;
    }
    if !info.converged {
        return ScoreStatus::Nonstationary;
    }
    if !positive_definite {
        return ScoreStatus::Singular;
    }
    if !(cond <= COND_LIMIT) {
        return ScoreStatus::IllConditioned;
    }
    if !log_box.is_finite() {
        return ScoreStatus::NonFinite;
    }
    ScoreStatus::Supported
}

// ---------------------------------------------------------------------------
// The search
// ---------------------------------------------------------------------------

/// Smallest score difference this search will act on, in nats.
///
/// A *numerical* resolution, not a scientific detection threshold. The
/// detection decision is the sign of the gain under the prior, which already
/// charges the Occam factor; this constant only says how much of a difference
/// is real arithmetic rather than the noise floor of two independently
/// converged fits.
///
/// # What the measurement found, and what it did not
///
/// `scripts/measure_group_score.py` rescores one configuration from perturbed
/// starts. The naive reading of that -- "the spread is the resolution" -- is
/// wrong by orders of magnitude, and the way it is wrong is worth recording.
///
/// The measured spread is exactly LINEAR in the perturbation over six decades
/// (1e-8 -> 9.9e-6 nats, 1e-7 -> 9.9e-5, ... 1e-3 -> 1.0). Linear, not
/// quadratic, because the score is evaluated at the fit's stopping point and
/// the fit is stationary for `I - log pi` while the score ALSO carries
/// `-logdet(F)/2`, whose gradient there is not zero. So a returned parameter
/// anywhere inside the stationarity band moves the score at first order.
///
/// Past about 1% the linearity is something else entirely: the fits stop
/// reaching the same optimum. Ten starts 1% apart reached ten different
/// converged optima spread 5.0 nats, and re-running each at 3000 iterations
/// with `tol_obj = 1e-14` moved none of them. That is multimodality, not
/// resolution, and no tolerance can absorb it -- it is what
/// [`GroupSettings::incumbent_restarts`] and [`perturbed_start`] are for.
///
/// So this constant does one narrow job: it stops a move being committed on a
/// gain the fit's own stopping tolerance could have manufactured. 1e-4 nats
/// covers the score's first-order sensitivity out to a 1e-7 relative
/// displacement, which is two decades beyond where the stationarity check
/// leaves a converged fit. It is NOT a detection threshold and NOT a claim
/// that two scores 1e-3 apart mean different things -- the scientific
/// separation is the sign of the gain under the prior, whose Occam term is
/// worth nats, not fractions of one.
pub const SCORE_TOL: f64 = 1e-4;

/// How far a restart steps each parameter, in units of its own conditional
/// standard error.
///
/// Large enough to leave the basin the previous fit settled in, small enough
/// that the start still describes the same configuration -- a source stepped
/// by several standard errors is a different proposal, and the search has move
/// constructors for those. One standard error is by construction the scale
/// over which the objective changes by ~1/2 nat, which is the scale the basins
/// measured in [`perturbed_start`] are separated on.
pub const STARTUP_SE_FRAC: f64 = 1.0;

/// Gain band around zero within which a comparison is re-run at the escalated
/// fit budget before it is decided.
///
/// A comparison this close is settled by fit quality rather than by the
/// models, and the proposal is the side that started further from its optimum
/// -- `lmcl::fit`'s convergence note measures that truncation is not symmetric
/// noise but a systematic bias toward the smaller model. Escalating both sides
/// removes the asymmetry rather than compensating for it.
pub const ESCALATE_BAND: f64 = 1.0;

/// How a transaction ended.
///
/// Note what is absent: nothing here says "converged" and nothing says
/// "optimal". A capped search reports its cap, and a search that ran out of
/// improving proposals reports exactly that -- over the proposals it generated,
/// in one context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchStatus {
    /// No generated proposal improved the score by more than [`SCORE_TOL`].
    NoImprovingProposal,
    /// A move or fit budget expired with proposals still untried.
    BudgetExhausted,
    /// A comparison could not be settled: an incumbent or the only improving
    /// alternative stayed unsupported through its restart budget.
    UnresolvedComparison,
    /// The committed configuration no longer fits the context it was compared
    /// in. The caller must rebuild and may then run another transaction.
    ContextRebuildRequired,
}

impl SearchStatus {
    pub fn reason(self) -> &'static str {
        match self {
            SearchStatus::NoImprovingProposal => "no_improving_proposal",
            SearchStatus::BudgetExhausted => "budget_exhausted",
            SearchStatus::UnresolvedComparison => "unresolved_comparison",
            SearchStatus::ContextRebuildRequired => "context_rebuild_required",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveKind {
    Birth,
    Split,
    Removal,
}

impl MoveKind {
    pub fn name(self) -> &'static str {
        match self {
            MoveKind::Birth => "birth",
            MoveKind::Split => "split",
            MoveKind::Removal => "removal",
        }
    }
}

/// One committed move.
#[derive(Clone, Copy, Debug)]
pub struct AcceptedMove {
    pub kind: MoveKind,
    /// The emitter the move acted on: the parent of a split, the emitter
    /// removed. `None` for a birth.
    pub target: Option<EmitterId>,
    /// `score(candidate) - score(incumbent)`, in the same context.
    pub gain: f64,
    /// The part of `gain` contributed by the box restriction of the two
    /// Laplace volumes, `log_box(candidate) - log_box(incumbent)`. For
    /// diagnostics: `gain - box_gain` is what the regular expression would
    /// have said about the same two fits.
    pub box_gain: f64,
    /// The score after the commit.
    pub score: f64,
}

/// Work spent and conditions met, for development. Adapting a normal result
/// requires none of this: acceptance is already decided.
#[derive(Clone, Debug, Default)]
pub struct GroupDiagnostics {
    pub n_fits: usize,
    pub n_restarts: usize,
    pub fits_by_kind: [usize; 3],
    pub proposals_generated: usize,
    /// Hypotheses that came back unsupported, by reason.
    pub unsupported: Vec<(&'static str, usize)>,
    /// Supported hypotheses whose mode rests on at least one bound, so their
    /// score depended on the one-sided volume in [`log_box_mass`].
    pub boundary_scored: usize,
    /// The free set had to be truncated to `k_max`.
    pub capacity_limited: bool,
    /// The region or position box met the frame edge.
    pub edge_clipped: bool,
    /// A committed emitter reached its position bound -- the drift allowance
    /// in [`DRIFT_FACTOR`] was active, so the geometry was binding.
    pub position_bound_active: bool,
    /// A residual peak above [`OUTSIDE_PEAK_SIGMA`] lies inside the region but
    /// outside the admissible position box, so no hypothesis here could
    /// propose for it.
    pub residual_peak_outside_box: bool,
    /// Where that light is: the matched-filter maxima outside the position
    /// box in the transaction's last proposal round, GLOBAL coordinates.
    pub outside_peaks: Vec<(f64, f64)>,
    /// The incumbent's own refit could not be scored at entry.
    pub incumbent_unsupported: bool,
    pub incumbent_status: Option<ScoreStatus>,
}

impl GroupDiagnostics {
    fn note(&mut self, h: &Hypothesis) {
        if h.status.is_supported() {
            self.boundary_scored += (h.n_active > 0) as usize;
            return;
        }
        let key = h.status.reason();
        if let Some(e) = self.unsupported.iter_mut().find(|e| e.0 == key) {
            e.1 += 1;
        } else {
            self.unsupported.push((key, 1));
        }
    }
}

/// What a transaction returns.
#[derive(Clone, Debug)]
pub struct GroupOutcome {
    /// The committed configuration, in global coordinates.
    pub state: GroupState,
    /// Ids whose parameters changed, including newly created ones.
    pub changed: Vec<EmitterId>,
    /// Ids no longer in the model. A split retires its parent id and mints two
    /// children, so the parent appears here and both children in `changed`.
    pub removed: Vec<EmitterId>,
    pub status: SearchStatus,
    /// The committed configuration's score, when it has one.
    pub score: Option<f64>,
    /// Why it does or does not. Kept separate from `status`: a search can end
    /// cleanly on a supported score, or end unresolved with no score at all.
    pub score_status: ScoreStatus,
    /// The committed configuration's fit. Separate from the search status for
    /// the same reason.
    pub fit: FitInfo,
    /// Conditional standard errors for the committed parameters, in the fit's
    /// own `[b, (A, y, x, sigma) * K]` order, or `None` when the curvature does
    /// not support them.
    ///
    /// Conditional on this context: on the frozen halo, on the background
    /// parameterization and on `K`. It is not a marginal uncertainty over the
    /// number of sources.
    pub uncertainty: Option<Vec<f64>>,
    pub trace: Vec<AcceptedMove>,
    pub diag: GroupDiagnostics,
    /// The context version this outcome was produced in. A caller holding a
    /// cached score from a different version must discard it.
    pub version: u64,
}

/// Search tuning. Budgets are per transaction.
#[derive(Clone, Debug)]
pub struct GroupSettings {
    /// Compute limit on the free set. A birth or split may cross it once, to
    /// keep the `K+1` alternative testable; the transaction then ends with
    /// [`SearchStatus::ContextRebuildRequired`] rather than refusing the move.
    pub k_max: usize,
    pub max_moves: usize,
    pub max_fits: usize,
    /// The budget every hypothesis gets first. Symmetric by construction.
    pub fit: FitOpts,
    /// The budget a stalled or close comparison is re-run at, on both sides.
    pub escalated_fit: FitOpts,
    /// Restarts a proposal gets on its first evaluation. Kept small: most
    /// proposals lose by a wide margin and paying multi-start for all of them
    /// multiplies the transaction's cost by the restart count.
    pub max_restarts: usize,
    /// Restarts the incumbent gets. It is one hypothesis and it is the
    /// baseline every gain is measured against, so a bad basin here makes
    /// every move look good; it is the one place multi-start is always worth
    /// paying for.
    pub incumbent_restarts: usize,
    /// Restarts BOTH sides of a close comparison get before it is decided.
    pub escalate_restarts: usize,
    pub min_gain: f64,
    pub escalate_band: f64,
    /// Split displacements, in units of the parent's OWN width.
    pub split_disps: Vec<f64>,
    /// Multipliers on `sigma0` for alternative width starts.
    pub width_starts: Vec<f64>,
    /// Residual maxima to seed births from, per iteration.
    pub max_birth_seeds: usize,
    /// Family-wise rate for the outside-the-box residual test. A setting
    /// rather than a bare constant so its operating point can be swept; see
    /// [`OUTSIDE_PEAK_ALPHA`].
    pub outside_peak_alpha: f64,
}

impl Default for GroupSettings {
    fn default() -> Self {
        Self {
            k_max: linalg::K_MAX,
            max_moves: 8,
            max_fits: 600,
            fit: FitOpts::default(),
            escalated_fit: FitOpts {
                max_iter: 300,
                tol_obj: 1e-10,
                ..FitOpts::default()
            },
            max_restarts: 1,
            incumbent_restarts: 3,
            escalate_restarts: 3,
            min_gain: SCORE_TOL,
            escalate_band: ESCALATE_BAND,
            split_disps: moves::SPLIT_DISPS.to_vec(),
            // Narrow and broad. A source that widened to absorb a neighbour
            // leaves no residual peak to propose into, so the alternative has
            // to be seeded from the other side -- children started AT the PSF
            // width, against a parent that is not.
            width_starts: vec![1.0, 1.5],
            max_birth_seeds: 3,
            outside_peak_alpha: OUTSIDE_PEAK_ALPHA,
        }
    }
}

struct Candidate {
    kind: MoveKind,
    /// Index into the incumbent's emitter vector that the move acted on: the
    /// parent of a split, the emitter a removal dropped. `None` for a birth.
    ///
    /// An index rather than an id, because it is only meaningful against the
    /// incumbent this proposal was generated from -- and because which *id*
    /// the move ends up retiring is not decided here. See [`assign_ids`].
    target_index: Option<usize>,
    theta0: Vec<f64>,
}

impl Candidate {
    /// How many genuinely new sources this move introduces.
    fn n_new(&self) -> usize {
        match self.kind {
            MoveKind::Birth => 1,
            MoveKind::Split => 2,
            MoveKind::Removal => 0,
        }
    }
}

/// Attach identities to a committed configuration.
///
/// # Why this is not "the vector slot it landed in"
///
/// A `GroupState` is a SET of emitters, and every hypothesis is jointly
/// refitted, so the fitted vector's order carries no information about which
/// object is which. Two removal proposals that drop different members of a
/// collapsed pair converge to the same one-source configuration: the pair is
/// exchangeable, so which proposal wins is decided by rounding, and reading the
/// surviving id off the vector slot would make a real source appear to vanish
/// and a spurious one to teleport onto it. Identity has to follow the object.
///
/// So a commit matches the fitted emitters back to the pre-move ones by
/// proximity, greedily nearest-pair first, and mints ids only for the
/// remainder. `exclude` is the split parent, which is deliberately NOT
/// matchable: neither child IS the parent -- they are two sources where the
/// model held one -- so its id retires and both children are new.
///
/// Returns `(ids aligned with new_theta, ids no longer present)`.
fn assign_ids(
    prev_theta: &[f64],
    prev_ids: &[EmitterId],
    exclude: Option<usize>,
    new_theta: &[f64],
    n_new: usize,
    ids: &mut IdAllocator,
) -> (Vec<EmitterId>, Vec<EmitterId>) {
    let k_new = psf::n_emitters_var(new_theta);
    let k_prev = psf::n_emitters_var(prev_theta);
    let carry = k_new - n_new;

    let mut pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(k_new * k_prev);
    for i in 0..k_new {
        let (y, x) = (psf::cy_var(new_theta, i), psf::cx_var(new_theta, i));
        for j in 0..k_prev {
            if exclude == Some(j) {
                continue;
            }
            let (dy, dx) = (psf::cy_var(prev_theta, j) - y, psf::cx_var(prev_theta, j) - x);
            pairs.push((dy * dy + dx * dx, i, j));
        }
    }
    // Nearest first; indices break ties so the assignment is deterministic.
    pairs.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });

    let mut out: Vec<Option<EmitterId>> = vec![None; k_new];
    let mut used_prev = vec![false; k_prev];
    let mut assigned = 0usize;
    for &(_, i, j) in &pairs {
        if assigned == carry {
            break;
        }
        if out[i].is_some() || used_prev[j] {
            continue;
        }
        out[i] = Some(prev_ids[j]);
        used_prev[j] = true;
        assigned += 1;
    }
    let final_ids: Vec<EmitterId> = out
        .into_iter()
        .map(|o| o.unwrap_or_else(|| ids.mint()))
        .collect();
    let removed = prev_ids
        .iter()
        .enumerate()
        .filter(|(j, _)| !used_prev[*j])
        .map(|(_, &id)| id)
        .collect();
    (final_ids, removed)
}

/// Optimize one local group of interacting emitters, and commit the result.
///
/// Owns neighbourhood usage, proposals, fits, scoring and local acceptance.
/// No Python numerical or decision callback is involved at any point.
///
/// The loop: refit the incumbent at the transaction's budget; generate every
/// move type from that same incumbent; jointly refit each candidate including
/// its surviving neighbours; take the best valid strict improvement; commit it
/// atomically; regenerate. It ends when no generated proposal improves the
/// score, when a comparison cannot be settled, when the context needs
/// rebuilding, or when a budget expires.
pub fn search_group(
    ctx: &GroupContext,
    ws: &mut GroupWorkspace,
    entry: &GroupState,
    ids: &mut IdAllocator,
    settings: &GroupSettings,
) -> GroupOutcome {
    let mut diag = GroupDiagnostics {
        capacity_limited: ctx.capacity_limited,
        edge_clipped: ctx.edge_clipped,
        ..Default::default()
    };
    let mut trace = Vec::new();

    let mut live_ids: Vec<EmitterId> = entry.emitters.iter().map(|e| e.id).collect();
    // Defensive: a caller that hands in a stale allocator must not be able to
    // mint an id that is already in this group.
    for id in &live_ids {
        ids.reserve_through(*id);
    }
    let mut removed: Vec<EmitterId> = Vec::new();
    let entry_ids: Vec<EmitterId> = live_ids.clone();

    ctx.pack_state(entry, &mut ws.theta);
    let entry_theta = ws.theta.clone();

    // The incumbent's refit. `evaluate` keeps the best valid evaluated state
    // across restarts, so a failed restart cannot overwrite a good fit.
    let mut incumbent = evaluate(
        ctx,
        ws,
        &entry_theta,
        settings,
        &mut diag,
        MoveKind::Removal,
        settings.incumbent_restarts,
    );
    // A refit that made the penalized objective worse is not an improvement at
    // fixed K; keep the state we came in with. This is an estimation step, not
    // a model comparison, so it needs no Laplace score to be decided.
    if !(incumbent.objective.is_finite()) {
        incumbent = fit_and_score(ctx, ws, &entry_theta, FitOpts { max_iter: 0, ..settings.fit });
    }
    diag.incumbent_status = Some(incumbent.status);
    diag.incumbent_unsupported = !incumbent.status.is_supported();

    let mut status = SearchStatus::NoImprovingProposal;

    for _move_index in 0..settings.max_moves {
        if diag.n_fits >= settings.max_fits {
            status = SearchStatus::BudgetExhausted;
            break;
        }
        if !incumbent.status.is_supported() {
            // The incumbent cannot be scored, so no gain against it means
            // anything. It does not get an arbitrary finite score, and it does
            // not get replaced automatically either.
            status = SearchStatus::UnresolvedComparison;
            break;
        }

        let mut outside = Vec::new();
        let proposals = generate(ctx, ws, &incumbent, settings, &mut outside);
        diag.residual_peak_outside_box |= !outside.is_empty();
        // The latest round's, so they describe the residual of the state the
        // transaction ends in rather than one it has since changed.
        diag.outside_peaks = outside.iter().map(|&(y, x)| ctx.to_global(y, x)).collect();
        diag.proposals_generated += proposals.len();
        if proposals.is_empty() {
            status = SearchStatus::NoImprovingProposal;
            break;
        }

        let mut best: Option<(Candidate, Hypothesis)> = None;
        let mut unresolved_improver = false;
        for cand in proposals {
            if diag.n_fits >= settings.max_fits {
                status = SearchStatus::BudgetExhausted;
                break;
            }
            let hyp = evaluate(
                ctx,
                ws,
                &cand.theta0,
                settings,
                &mut diag,
                cand.kind,
                settings.max_restarts,
            );
            if !hyp.status.is_supported() {
                // Track whether an unsupported hypothesis was at least trying
                // to improve, so the transaction can say so honestly.
                if hyp.objective < incumbent.objective {
                    unresolved_improver = true;
                }
                continue;
            }
            if best.as_ref().is_none_or(|(_, b)| hyp.score > b.score) {
                best = Some((cand, hyp));
            }
        }
        if status == SearchStatus::BudgetExhausted {
            break;
        }

        let Some((cand, mut hyp)) = best else {
            status = if unresolved_improver {
                SearchStatus::UnresolvedComparison
            } else {
                SearchStatus::NoImprovingProposal
            };
            break;
        };

        // A comparison decided inside the escalation band is decided by fit
        // quality, not by the models. Re-run BOTH sides at the larger budget.
        let mut gain = hyp.score - incumbent.score;
        if gain.abs() < settings.escalate_band {
            let inc2 = escalate(ctx, ws, &incumbent, settings, &mut diag, MoveKind::Removal);
            let cand2 = escalate(ctx, ws, &hyp, settings, &mut diag, cand.kind);
            if inc2.status.is_supported() && cand2.status.is_supported() {
                incumbent = inc2;
                hyp = cand2;
                gain = hyp.score - incumbent.score;
            } else if !inc2.status.is_supported() || !cand2.status.is_supported() {
                status = SearchStatus::UnresolvedComparison;
                break;
            }
        }

        // Strict improvement, on the resolution the score is actually known
        // to. A tie keeps the incumbent.
        if !(gain > settings.min_gain) {
            status = SearchStatus::NoImprovingProposal;
            break;
        }

        // Atomic commit: every affected parameter at once, or nothing.
        debug_assert!(hyp.score > incumbent.score);
        let exclude = if cand.kind == MoveKind::Split {
            cand.target_index
        } else {
            None
        };
        let (new_ids, gone) = assign_ids(
            &incumbent.theta,
            &live_ids,
            exclude,
            &hyp.theta,
            cand.n_new(),
            ids,
        );
        // What a move retires is decided by the assignment, not by the vector
        // slot the proposal dropped: an exchangeable pair makes those two
        // different answers, and only the first is about the objects.
        let target = match cand.kind {
            MoveKind::Birth => None,
            MoveKind::Split => cand.target_index.map(|i| live_ids[i]),
            MoveKind::Removal => gone.first().copied(),
        };
        removed.extend_from_slice(&gone);
        live_ids = new_ids;
        trace.push(AcceptedMove {
            kind: cand.kind,
            target,
            gain,
            box_gain: hyp.log_box - incumbent.log_box,
            score: hyp.score,
        });
        incumbent = hyp;

        // Regenerate against the committed state -- never a stale residual or
        // a stale neighbour fit. Removal followed by a split, or the reverse,
        // is permitted: each is supported by the same comparison rule.

        if live_ids.len() > settings.k_max {
            status = SearchStatus::ContextRebuildRequired;
            break;
        }
        if at_position_bound(ctx, &incumbent.theta) {
            // The committed state is pinned against where this context lets
            // a source go: the region was binding, not generous.
            //
            // What is deliberately NOT a trigger: a frozen neighbour within
            // the link radius of a free emitter. The free set is chosen by
            // distance to the FOCUS, so a rebuilt context freezes that same
            // neighbour again -- in a dense field the request could never be
            // satisfied, and on a 39x39 frame at 0.034/px^2 it ended 71 of 80
            // transactions. The comparison is conditional on frozen
            // neighbours through the halo, and revisiting them is the frame
            // schedule's job, not a reason to rebuild this one.
            diag.position_bound_active = true;
            status = SearchStatus::ContextRebuildRequired;
            break;
        }
        status = SearchStatus::NoImprovingProposal;
    }
    if trace.len() >= settings.max_moves && status == SearchStatus::NoImprovingProposal {
        status = SearchStatus::BudgetExhausted;
    }
    if status == SearchStatus::NoImprovingProposal && at_position_bound(ctx, &incumbent.theta) {
        // Scored, but pinned against the edge of where this context lets a
        // source go -- the region was binding, not generous. Before boundary
        // modes were scored this state ended `unresolved_comparison` and
        // never reached here; a committed move asks the same question inside
        // the loop.
        diag.position_bound_active = true;
        status = SearchStatus::ContextRebuildRequired;
    }
    if status == SearchStatus::NoImprovingProposal && diag.residual_peak_outside_box {
        // Not "nothing improves": nothing *here* improves, and there is light
        // this context cannot place a source on.
        status = SearchStatus::ContextRebuildRequired;
    }

    let state = ctx.unpack(&incumbent.theta, &live_ids);
    let changed = changed_ids(entry, &state, &entry_ids);
    let uncertainty = if incumbent.status.is_supported() {
        let p = incumbent.p;
        if ws.chol.factor(&incumbent.curvature, p) {
            ws.inv_diag.resize(p.max(ws.inv_diag.len()), 0.0);
            ws.chol.inv_diag(&mut ws.inv_diag[..p], &mut ws.scratch);
            Some(ws.inv_diag[..p].iter().map(|v| v.max(0.0).sqrt()).collect())
        } else {
            None
        }
    } else {
        None
    };

    GroupOutcome {
        score: incumbent.status.is_supported().then_some(incumbent.score),
        score_status: incumbent.status,
        fit: incumbent.fit,
        state,
        changed,
        removed,
        status,
        uncertainty,
        trace,
        diag,
        version: ctx.version,
    }
}

/// A deterministic alternative start for one hypothesis, `index` selecting it.
///
/// # Why restarts have to move, not just iterate longer
///
/// Measured on a crowded K=3 group: ten starts perturbed by 1% reached ten
/// DIFFERENT converged optima, spread 5.0 nats in the data term alone -- and
/// re-running each at 3000 iterations with `tol_obj = 1e-14` moved none of
/// them, landing on the same ten values in ~21 iterations. A bigger iteration
/// budget cannot escape a basin. So an escalation that only iterates harder
/// re-certifies the answer it already had, and the comparison it was meant to
/// settle stays settled by whichever side got the luckier initialization --
/// exactly the bias `lmcl::fit`'s convergence note warns about, with the
/// asymmetry moved from truncation to basin choice.
///
/// Each free parameter is stepped by `STARTUP_SE_FRAC` of its OWN conditional
/// standard error, so the step is scale-free across background, flux, position
/// and width without a table of per-parameter scales. Signs come from the bits
/// of `index`, which makes the starts deterministic, reproducible and
/// genuinely different from one another.
///
/// The background is left alone: it is shared by every hypothesis, is nearly
/// orthogonal to the emitters over a small region, and moving it perturbs
/// every emitter at once rather than exploring the degeneracy that matters.
fn perturbed_start(
    ctx: &GroupContext,
    ws: &mut GroupWorkspace,
    hyp: &Hypothesis,
    index: usize,
    out: &mut Vec<f64>,
) {
    out.clear();
    out.extend_from_slice(&hyp.theta);
    let p = hyp.p;
    ws.inv_diag.resize(p.max(ws.inv_diag.len()), 0.0);
    let have_se = ws.chol.factor(&hyp.curvature, p) && {
        ws.chol.inv_diag(&mut ws.inv_diag[..p], &mut ws.scratch);
        true
    };
    let (lo, hi) = ctx.bounds.arrays(psf::n_emitters_var(&hyp.theta));
    for q in 1..p {
        let sign = if (index >> ((q - 1) % 31)) & 1 == 1 {
            1.0
        } else {
            -1.0
        };
        let se = if have_se && ws.inv_diag[q] > 0.0 && ws.inv_diag[q].is_finite() {
            ws.inv_diag[q].sqrt()
        } else {
            // No usable curvature: fall back to a fraction of the parameter's
            // own admissible range, which is the only scale left.
            0.02 * (hi[q] - lo[q])
        };
        let step = sign * STARTUP_SE_FRAC * se;
        // Kept strictly inside, for the reason `lmcl::Interior` exists.
        let margin = 1e-6 * (hi[q] - lo[q]);
        out[q] = (out[q] + step).clamp(lo[q] + margin, hi[q] - margin);
    }
}

/// Fit one hypothesis, retaining the best valid evaluated state across the
/// restart budget.
///
/// The escalation policy is common to every move type and to the incumbent:
/// the first attempt gets `settings.fit`; if it comes back unsupported, the
/// next attempts continue from the parameters it reached at
/// `settings.escalated_fit`. A restart that ends worse than the attempt before
/// it is discarded -- non-convergence is not evidence against whichever side
/// happened to receive the worse initialization.
fn evaluate(
    ctx: &GroupContext,
    ws: &mut GroupWorkspace,
    theta0: &[f64],
    settings: &GroupSettings,
    diag: &mut GroupDiagnostics,
    kind: MoveKind,
    restarts: usize,
) -> Hypothesis {
    let mut best = fit_and_score(ctx, ws, theta0, settings.fit);
    diag.n_fits += 1;
    diag.fits_by_kind[kind as usize] += 1;
    diag.note(&best);

    for attempt in 0..restarts {
        // Attempt 0 continues from where the fit stopped, at the larger
        // budget: that is the right answer when the fit was merely truncated,
        // and no answer at all when it was not -- a supported fit is already
        // stationary, and continuing it only re-certifies the same point.
        // Later attempts MOVE, because a truncated fit and a fit in the wrong
        // basin are different failures and only one of them is fixed by
        // iterating. See `perturbed_start`.
        if attempt == 0 && best.status.is_supported() {
            continue;
        }
        let mut start = std::mem::take(&mut ws.start);
        if attempt == 0 {
            start.clear();
            start.extend_from_slice(&best.theta);
        } else {
            perturbed_start(ctx, ws, &best, attempt, &mut start);
        }
        let next = fit_and_score(ctx, ws, &start, settings.escalated_fit);
        ws.start = start;
        diag.n_fits += 1;
        diag.n_restarts += 1;
        diag.fits_by_kind[kind as usize] += 1;
        diag.note(&next);
        best = keep_better(best, next);
    }
    best
}

/// The best valid evaluated state of two attempts at the SAME hypothesis.
///
/// A supported fit beats an unsupported one whatever their objectives, because
/// an unsupported one has no score to compare with. Between two supported
/// fits the lower penalized objective wins -- that is the quantity both were
/// minimizing, and it is comparable in a way two Laplace scores of the same
/// configuration at different modes are also comparable but noisier.
/// A failed restart never overwrites a good fit.
fn keep_better(best: Hypothesis, next: Hypothesis) -> Hypothesis {
    let take = match (best.status.is_supported(), next.status.is_supported()) {
        (false, true) => true,
        (true, false) => false,
        _ => next.objective < best.objective,
    };
    if take { next } else { best }
}

/// Re-run one already-fitted hypothesis under the full restart policy.
///
/// The same routine both sides of a close comparison go through, which is what
/// makes the escalation symmetric: it is one function, not two that are meant
/// to agree.
fn escalate(
    ctx: &GroupContext,
    ws: &mut GroupWorkspace,
    from: &Hypothesis,
    settings: &GroupSettings,
    diag: &mut GroupDiagnostics,
    kind: MoveKind,
) -> Hypothesis {
    let mut start = std::mem::take(&mut ws.start);
    start.clear();
    start.extend_from_slice(&from.theta);
    let next = evaluate(ctx, ws, &start, settings, diag, kind, settings.escalate_restarts);
    ws.start = start;
    keep_better(from.clone(), next)
}

/// Every move type, all generated from the SAME incumbent.
///
/// Births, splits and removals compete against one another and against the
/// incumbent under one score. In particular a merged-pair seed may initialize
/// a removal hypothesis; it does not acquire an acceptance rule of its own.
fn generate(
    ctx: &GroupContext,
    ws: &mut GroupWorkspace,
    incumbent: &Hypothesis,
    settings: &GroupSettings,
    outside: &mut Vec<(f64, f64)>,
) -> Vec<Candidate> {
    let theta = &incumbent.theta;
    let k = psf::n_emitters_var(theta);
    let mut out = Vec::new();

    ws.residual(ctx, theta);
    // Asked whatever `k` is: a group capped at `k_max` must still say what it
    // is leaving behind rather than reporting that nothing improves.
    *outside = light_outside_box(ctx, ws, ctx.sigma0, settings.outside_peak_alpha);

    // --- Births. One extra emitter is allowed to cross `k_max`, so the K+1
    // alternative stays testable; the transaction then asks for a rebuild.
    if k < settings.k_max + 1 {
        let mut seeds: Vec<Seed> = Vec::new();
        for s in &ctx.seeds {
            let (ly, lx) = ctx.to_local(s.y, s.x);
            if in_box(ctx, ly, lx) {
                seeds.push(Seed {
                    y: ly,
                    x: lx,
                    flux: s.flux,
                });
            }
        }
        seeds.extend(residual_peaks(ctx, ws, settings.max_birth_seeds));
        for s in seeds {
            for &mult in &settings.width_starts {
                let sigma = (mult * ctx.sigma0).clamp(ctx.bounds.sigma_lo, ctx.bounds.sigma_hi);
                let flux = if s.flux > 0.0 {
                    s.flux
                } else {
                    peak_flux(ctx, ws, s.y, s.x, sigma)
                };
                out.push(Candidate {
                    kind: MoveKind::Birth,
                    target_index: None,
                    theta0: moves::birth_var(theta, flux, s.y, s.x, sigma),
                });
            }
        }
    }

    // --- Splits, width-aware, at several initial separations.
    if k < settings.k_max + 1 {
        for e in 0..k {
            let (u, _strength) =
                moves::residual_axis_var(theta, e, 0.0, 0.0, ctx.h, ctx.w, &ws.resid[..ctx.n_pixels()]);
            let parent_sigma = psf::sigma_var(theta, e);
            for &disp in &settings.split_disps {
                let base = moves::split_var(theta, e, u, disp * parent_sigma);
                // Children at the parent's width: the proposal and the
                // incumbent start in the same basin.
                push_split(&mut out, ctx, &base, e, None);
                // Children at the PSF width: the alternative for a parent that
                // widened to absorb its neighbour, where there is no residual
                // peak left to propose into.
                if (parent_sigma - ctx.sigma0).abs() > 1e-9 {
                    push_split(&mut out, ctx, &base, e, Some(ctx.sigma0));
                }
            }
        }
    }

    // --- Removals.
    if k > 0 {
        for e in 0..k {
            out.push(Candidate {
                kind: MoveKind::Removal,
                target_index: Some(e),
                theta0: moves::remove_var(theta, e),
            });
        }
    }
    out
}

/// One split proposal, optionally with both children restarted at a given
/// width. The parent's id is retired at commit time; see [`assign_ids`].
fn push_split(
    out: &mut Vec<Candidate>,
    ctx: &GroupContext,
    base: &[f64],
    parent: usize,
    child_sigma: Option<f64>,
) {
    let mut theta0 = base.to_vec();
    if let Some(s) = child_sigma {
        let s = s.clamp(ctx.bounds.sigma_lo, ctx.bounds.sigma_hi);
        let n = psf::n_emitters_var(&theta0);
        theta0[4 * (n - 1)] = s;
        theta0[4 * n] = s;
    }
    out.push(Candidate {
        kind: MoveKind::Split,
        target_index: Some(parent),
        theta0,
    });
}

/// Family-wise false-alarm rate for the outside-the-box residual test.
///
/// This is a SEEDER-grade question, not a decision rule, and the costs are
/// lopsided: a false positive spends one rebuilt context and reaches the same
/// answer, while a false negative loses a source outright, because no
/// hypothesis in THIS context can ever place one there. So it is deliberately
/// loose, in the same spirit as `calibrate.SEED_ALPHA`.
///
/// But the nominal rate is not the realized one. `calibrate.py` records why:
/// the Bonferroni count assumes the response is standard normal at
/// `area/win^2` independent points, and taking local maxima of a SMOOTH field
/// samples its supremum instead -- so the realized rate runs several times the
/// nominal. Measured here by `scripts/measure_group_score.py --part outside`,
/// on isolated in-focus sources where nothing is in fact outside the box, and
/// against a genuine neighbour placed in the annulus:
///
/// | alpha | false rebuilds | 1500 e- | 300 | 100 | 60 | 40 |
/// |---|---|---|---|---|---|---|
/// | 5e-2 | 0.325 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |
/// | 1e-2 | 0.075 | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 |
/// | **1e-3** | **0.000** | 1.00 | 1.00 | 1.00 | 1.00 | 0.58 |
/// | 1e-4 | 0.000 | 1.00 | 1.00 | 1.00 | 0.93 | 0.35 |
/// | 1e-5 | 0.000 | 1.00 | 1.00 | 1.00 | 0.85 | 0.23 |
///
/// (40 trials per cell; the neighbour sits 8 px out, in the annulus.) 1e-3 is
/// the knee: the loosest value with no false rebuilds at all, and the tightest
/// that still finds every 60 e- neighbour. At 40 e- total flux the peak is
/// ~4 e- against a background of 5, so the degradation there is the source
/// becoming undetectable rather than the threshold refusing it.
///
/// The value is chosen from that table, not from the nominal rate -- which is
/// 30x looser than what it delivers.
pub const OUTSIDE_PEAK_ALPHA: f64 = 1e-3;

/// Is there light inside the region but outside the admissible position box
/// that this context cannot propose a source for?
///
/// # Why a matched filter and not a pixel
///
/// The first version of this tested one pixel's residual against
/// `3 * sqrt(m)`. That is a ~1e-3 per-pixel false-alarm rate applied to
/// several hundred pixels at once, so it fired on ordinary Poisson noise
/// around an isolated source and asked for a rebuild on frames where nothing
/// was missing. A source is not one pixel: it spreads over `2*pi*sigma^2`, and
/// the statistic that matches it is the Poisson-weighted amplitude estimate
/// divided by its own standard error,
///
/// ```text
/// z = (sum_i g_i r_i / m_i) / sqrt(sum_i g_i^2 / m_i)
/// ```
///
/// with `g` the unit-flux PSF centred on the candidate. That is the same
/// `A/SE` quantity `_amplitude_var` and `PRUNE_TAU` are written in.
///
/// The cut is `calibrate.seed_threshold`'s: `Phi^-1(1 - alpha/n)` over the
/// number of INDEPENDENT tests, `n_outside / win^2` with
/// `win = 2*ceil(sigma)+1` -- a threshold that scales with the region and the
/// PSF rather than a constant that cannot be right on two of either.
///
/// Returns every such response maximum, in LOCAL coordinates, so the caller
/// can say WHERE the light is rather than only that it exists: a frame
/// schedule answers it by looking there, not by rebuilding this group.
fn light_outside_box(
    ctx: &GroupContext,
    ws: &mut GroupWorkspace,
    sigma: f64,
    alpha: f64,
) -> Vec<(f64, f64)> {
    let mut found = Vec::new();
    let (h, w) = (ctx.h, ctx.w);
    if h < 3 || w < 3 {
        return found;
    }
    let rad = (3.0 * sigma).ceil() as usize;

    // Separable unit-flux PSF factors for EVERY candidate centre, computed
    // once: `ey[k*h + i]` is row `i`'s factor for a source centred on row `k`.
    // The alternative -- rebuilding the two axes inside the pixel loop -- is
    // the same arithmetic `psf::Factors` exists to stop being repeated.
    let centres_y: Vec<f64> = (0..h).map(|v| v as f64).collect();
    let centres_x: Vec<f64> = (0..w).map(|v| v as f64).collect();
    ws.ey_all.clear();
    ws.ey_all.resize(h * h, 0.0);
    ws.ex_all.clear();
    ws.ex_all.resize(w * w, 0.0);
    psf::shape_axis(&ctx.ay, &centres_y, sigma, &mut ws.ey_all);
    psf::shape_axis(&ctx.ax, &centres_x, sigma, &mut ws.ex_all);

    ws.zscore.clear();
    ws.zscore.resize(h * w, f64::NEG_INFINITY);
    let mut n_outside = 0usize;
    for pr in 0..h {
        // The window is CLIPPED at the region border rather than skipped.
        // Everything outside the position box is, by construction, within
        // `BBOX_PAD * sigma_hi` of that border -- so skipping partial windows
        // would blind this test to most of the annulus it exists to watch.
        // A clipped window is still a valid weighted estimate; it just has
        // less information, and `den` carries that.
        let (r0, r1) = (pr.saturating_sub(rad), (pr + rad + 1).min(h));
        for pc in 0..w {
            if !in_box(ctx, pr as f64, pc as f64) {
                n_outside += 1;
            }
            let (c0, c1) = (pc.saturating_sub(rad), (pc + rad + 1).min(w));
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for r in r0..r1 {
                let gy = ws.ey_all[pr * h + r];
                for c in c0..c1 {
                    let g = gy * ws.ex_all[pc * w + c];
                    let m = ws.model[r * w + c].max(1e-9);
                    num += g * ws.resid[r * w + c] / m;
                    den += g * g / m;
                }
            }
            ws.zscore[pr * w + pc] = if den > 0.0 {
                num / den.sqrt()
            } else {
                f64::NEG_INFINITY
            };
        }
    }
    if n_outside == 0 {
        return found;
    }
    let win = (2 * sigma.ceil() as usize + 1).max(1) as f64;
    let n_tests = (n_outside as f64 / (win * win)).max(1.0);
    let z_min = norm_isf(alpha / n_tests);

    // Local maxima of the RESPONSE, not of the raw residual -- the same shape
    // as `find_candidates`. Maxima of the raw residual would be pixels the
    // filter then reads as biased high, and the test count above would not
    // describe them.
    for pr in 1..h - 1 {
        for pc in 1..w - 1 {
            let z = ws.zscore[pr * w + pc];
            if z <= z_min || in_box(ctx, pr as f64, pc as f64) {
                continue;
            }
            let mut is_max = true;
            'nb: for dr in 0..3 {
                for dc in 0..3 {
                    if dr == 1 && dc == 1 {
                        continue;
                    }
                    if ws.zscore[(pr + dr - 1) * w + (pc + dc - 1)] >= z {
                        is_max = false;
                        break 'nb;
                    }
                }
            }
            if is_max {
                found.push((pr as f64, pc as f64));
            }
        }
    }
    found
}

/// Standard-normal upper-tail inverse, `Phi^-1(1 - p)`.
///
/// Public so it can be checked against `scipy.special.ndtri`'s values rather
/// than trusted.
///
/// `calibrate._norm_isf` uses `scipy.special.ndtri`; here it is Newton on
/// `erfc`, which needs only the `libm::erf` this crate already depends on for
/// the PSF itself. It is used once per proposal round to set a seeder-grade
/// threshold, so a few Newton steps are free and no accuracy argument is
/// needed beyond "the threshold is right to more digits than it is known to".
pub fn norm_isf(p: f64) -> f64 {
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    // Bisection rather than Newton. Newton on `Phi_upper(z) - p` divides by
    // `phi(z)`, which out in the tail is smaller than the residual it is
    // dividing, so a step from any plausible start can overshoot by whole
    // units and has to be safeguarded back into a bracket anyway. Bisection is
    // that bracket with none of the bookkeeping, it cannot diverge, and this
    // runs once per proposal round. 80 halvings of [0, 40] resolve `z` far
    // below the last digit anyone reads off it.
    let (mut lo, mut hi) = (0.0f64, 40.0f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if 0.5 * libm::erfc(mid / std::f64::consts::SQRT_2) > p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Strict local maxima of the residual inside the position box, strongest
/// first. Seeds only: they say where to look, never what is there.
///
fn residual_peaks(ctx: &GroupContext, ws: &GroupWorkspace, limit: usize) -> Vec<Seed> {
    let (h, w) = (ctx.h, ctx.w);
    let mut peaks: Vec<(f64, usize, usize)> = Vec::new();
    for r in 1..h.saturating_sub(1) {
        for c in 1..w.saturating_sub(1) {
            let v = ws.resid[r * w + c];
            if v <= 0.0 || !in_box(ctx, r as f64, c as f64) {
                continue;
            }
            if is_local_max(ws, r, c, w) {
                peaks.push((v, r, c));
            }
        }
    }
    peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    peaks.truncate(limit);
    peaks
        .into_iter()
        .map(|(_, r, c)| Seed {
            y: r as f64,
            x: c as f64,
            flux: 0.0,
        })
        .collect()
}

/// Strictly greater than all eight neighbours.
fn is_local_max(ws: &GroupWorkspace, r: usize, c: usize, w: usize) -> bool {
    let v = ws.resid[r * w + c];
    for dr in 0..3 {
        for dc in 0..3 {
            if dr == 1 && dc == 1 {
                continue;
            }
            if ws.resid[(r + dr - 1) * w + (c + dc - 1)] >= v {
                return false;
            }
        }
    }
    true
}

/// A total-flux start from the residual height at a seed.
///
/// `A` is total flux, not peak height, so an observed peak is divided by
/// `peak_factor`. Floored at the amplitude bound so the start is feasible.
fn peak_flux(ctx: &GroupContext, ws: &GroupWorkspace, y: f64, x: f64, sigma: f64) -> f64 {
    let r = (y.round().max(0.0) as usize).min(ctx.h - 1);
    let c = (x.round().max(0.0) as usize).min(ctx.w - 1);
    let peak = ws.resid[r * ctx.w + c].max(0.0);
    (peak / psf::peak_factor(sigma)).clamp(ctx.bounds.a_min * 2.0, ctx.bounds.a_max * 0.5)
}

#[inline]
fn in_box(ctx: &GroupContext, y: f64, x: f64) -> bool {
    y >= ctx.bounds.y_lo && y <= ctx.bounds.y_hi && x >= ctx.bounds.x_lo && x <= ctx.bounds.x_hi
}

/// Does any emitter rest on the transaction's position box?
fn at_position_bound(ctx: &GroupContext, theta: &[f64]) -> bool {
    let b = &ctx.bounds;
    let ry = (b.y_hi - b.y_lo).max(1e-12);
    let rx = (b.x_hi - b.x_lo).max(1e-12);
    (0..psf::n_emitters_var(theta)).any(|e| {
        let (y, x) = (psf::cy_var(theta, e), psf::cx_var(theta, e));
        (y - b.y_lo) / ry < BOUND_TOL
            || (b.y_hi - y) / ry < BOUND_TOL
            || (x - b.x_lo) / rx < BOUND_TOL
            || (b.x_hi - x) / rx < BOUND_TOL
    })
}

/// Ids whose parameters differ from the entry state, plus every new id.
fn changed_ids(entry: &GroupState, state: &GroupState, entry_ids: &[EmitterId]) -> Vec<EmitterId> {
    state
        .emitters
        .iter()
        .filter(|e| {
            if !entry_ids.contains(&e.id) {
                return true;
            }
            match entry.index_of(e.id) {
                Some(i) => entry.emitters[i] != **e,
                None => true,
            }
        })
        .map(|e| e.id)
        .collect()
}
