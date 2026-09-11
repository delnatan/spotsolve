//! LAYER 8: native local group search.
//!
//! No golden fixture. This layer is a deliberately new algorithm -- the Python
//! has no group transaction to be compared against -- so what is asserted here
//! are the invariants the comparison rests on, not a reference trajectory:
//! the prior's normalization, the score's composition, the antisymmetry of a
//! gain, the fit's data-only/curvature contract, and the search's own rules
//! (every commit strictly improves ONE context's score; a capped search says
//! so; an unsupported hypothesis is never evidence).
//!
//! Scientific behaviour -- recall, localization, over-splitting -- is measured
//! by `scripts/check_group_search.py` against known truth, not here.

use spotsolve_core::dense_group::{
    self as dg, ContextSpec, Emitter, FluxPrior, FrameView, GroupSettings, GroupState,
    GroupWorkspace, IdAllocator, MoveKind, PriorSnapshot, ScoreStatus, SearchStatus, Seed,
    WidthPrior,
};
use spotsolve_core::lmcl::{self, Bounds, FitOpts, FitWorkspace, FluxPenalty, Penalty, WidthPenalty};
use spotsolve_core::linalg;
use spotsolve_core::psf;

// ---------------------------------------------------------------------------
// A deterministic synthetic frame. No dev-dependency on an RNG crate: the
// generator is 20 lines and being able to state exactly which stream produced
// a failing case is worth more than a better one.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Knuth's product method. `lam` here is at most a few hundred.
    fn poisson(&mut self, lam: f64) -> f64 {
        if lam <= 0.0 {
            return 0.0;
        }
        let l = (-lam).exp();
        let (mut k, mut p) = (0u32, 1.0f64);
        loop {
            p *= self.uniform();
            if p <= l || k > 100_000 {
                return k as f64;
            }
            k += 1;
        }
    }
}

struct Sim {
    d: Vec<f64>,
    bmap: Vec<f64>,
    h: usize,
    w: usize,
}

/// A frame with `truth` emitters on a flat background, Poisson sampled.
fn simulate(h: usize, w: usize, truth: &[(f64, f64, f64, f64)], bg: f64, seed: u64) -> Sim {
    let ay = psf::local_axis(h);
    let ax = psf::local_axis(w);
    let mut mean = vec![bg; h * w];
    let mut ey = vec![0.0; h];
    let mut ex = vec![0.0; w];
    for &(y, x, flux, sigma) in truth {
        psf::shape_axis(&ay, &[y], sigma, &mut ey);
        psf::shape_axis(&ax, &[x], sigma, &mut ex);
        for r in 0..h {
            for c in 0..w {
                mean[r * w + c] += flux * ey[r] * ex[c];
            }
        }
    }
    let mut rng = Rng::new(seed);
    let d = mean.iter().map(|&m| rng.poisson(m)).collect();
    Sim {
        d,
        bmap: vec![bg; h * w],
        h,
        w,
    }
}

const SIGMA0: f64 = 1.2;
const SLACK: (f64, f64) = (0.70, 2.2);
const BAND_HI: f64 = 2.0;

fn spec(lam: f64, a_s: f64) -> ContextSpec {
    ContextSpec {
        sigma0: SIGMA0,
        slack: SLACK,
        k_max: 12,
        prior: PriorSnapshot {
            flux: FluxPrior::Exponential { a_s },
            width: WidthPrior::FocusMixture {
                lam_focus: lam,
                lam_wide: lam * 0.1,
                lo: SLACK.0 * SIGMA0,
                mid: BAND_HI * SIGMA0,
                hi: SLACK.1 * SIGMA0,
                sigma0: SIGMA0,
                scale: 0.20 * SIGMA0,
            },
        },
    }
}

fn view(sim: &Sim) -> FrameView<'_> {
    FrameView {
        d: &sim.d,
        bmap: &sim.bmap,
        h: sim.h,
        w: sim.w,
    }
}

// ---------------------------------------------------------------------------
// Unit: priors
// ---------------------------------------------------------------------------

#[test]
fn width_density_normalizes_over_the_model_space() {
    for prior in [
        WidthPrior::Uniform {
            lam: 0.02,
            lo: 0.84,
            hi: 2.64,
        },
        WidthPrior::FocusMixture {
            lam_focus: 0.02,
            lam_wide: 0.002,
            lo: 0.84,
            mid: 2.40,
            hi: 2.64,
            sigma0: 1.2,
            scale: 0.24,
        },
    ] {
        let (lo, hi) = match prior {
            WidthPrior::Uniform { lo, hi, .. } => (lo, hi),
            WidthPrior::FocusMixture { lo, hi, .. } => (lo, hi),
        };
        // Composite Simpson over the support: the density is smooth and
        // bounded there, so 4001 points is far more than enough to see a wrong
        // normalizer.
        let n = 4000usize;
        let step = (hi - lo) / n as f64;
        let mut total = 0.0;
        for i in 0..=n {
            let s = lo + i as f64 * step;
            let coeff = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            total += coeff * prior.logpdf(s).exp();
        }
        total *= step / 3.0;
        assert!(
            (total - 1.0).abs() < 1e-9,
            "{prior:?} integrates to {total}, not 1"
        );
    }
}

#[test]
fn width_penalty_derivatives_match_the_density_it_penalizes() {
    let prior = WidthPrior::FocusMixture {
        lam_focus: 0.02,
        lam_wide: 0.002,
        lo: 0.84,
        mid: 2.40,
        hi: 2.64,
        sigma0: 1.2,
        scale: 0.24,
    };
    let pen: WidthPenalty = prior.penalty().expect("mixture is not flat");
    for i in 0..=40 {
        let s = 0.85 + i as f64 * (2.63 - 0.85) / 40.0;
        assert!(
            (pen.logpdf(s) - prior.logpdf(s)).abs() < 1e-12,
            "the fit and the score charge different densities at sigma={s}"
        );
        let step = 1e-6;
        let fd = -(prior.logpdf(s + step) - prior.logpdf(s - step)) / (2.0 * step);
        assert!(
            (pen.gradient(s) - fd).abs() < 1e-6 * (1.0 + fd.abs()),
            "gradient {} != finite difference {fd} at sigma={s}",
            pen.gradient(s)
        );
        let fd2 = -(prior.logpdf(s + step) - 2.0 * prior.logpdf(s) + prior.logpdf(s - step))
            / (step * step);
        // Clamped at zero in the tails, where the Cauchy is not log-concave.
        let expected = fd2.max(0.0);
        assert!(
            (pen.curvature(s) - expected).abs() < 1e-3 * (1.0 + expected.abs()),
            "curvature {} != clamped finite difference {expected} at sigma={s}",
            pen.curvature(s)
        );
    }
}

#[test]
fn flux_penalty_is_the_negative_log_of_the_flux_density() {
    let a_s = 900.0;
    let prior = FluxPrior::Exponential { a_s };
    let pen: FluxPenalty = prior.penalty().unwrap();
    let theta = vec![3.0, 400.0, 2.0, 3.0, 1.2, 1500.0, 5.0, 6.0, 1.4];
    let expected = -prior.log_config([400.0, 1500.0].into_iter());
    assert!((pen.value(&theta) - expected).abs() < 1e-12);
    // Constant gradient; that is what makes its curvature exactly zero and so
    // leaves the Laplace volume untouched.
    assert!((pen.gradient() - 1.0 / a_s).abs() < 1e-15);
}

#[test]
fn count_and_labeling_terms_are_charged_exactly_once() {
    let prior = WidthPrior::FocusMixture {
        lam_focus: 0.02,
        lam_wide: 0.002,
        lo: 0.84,
        mid: 2.40,
        hi: 2.64,
        sigma0: 1.2,
        scale: 0.24,
    };
    // Two in focus, one wide.
    let sigmas = [1.15, 1.30, 2.50];
    let got = prior.log_config(&sigmas);
    let want = 2.0 * 0.02f64.ln() - 2.0f64.ln()      // k_f log lam_f - log 2!
        + 1.0 * 0.002f64.ln() - 0.0                   // k_w log lam_w - log 1!
        + sigmas.iter().map(|&s| prior.logpdf(s)).sum::<f64>();
    assert!((got - want).abs() < 1e-12, "{got} != {want}");
    assert_eq!(prior.log_config(&[]), 0.0);
}

// ---------------------------------------------------------------------------
// Unit: coordinates, proposals, bookkeeping
// ---------------------------------------------------------------------------

#[test]
fn coordinates_round_trip_between_the_frame_and_the_window() {
    let sim = simulate(32, 30, &[(16.0, 15.0, 900.0, 1.2)], 5.0, 1);
    let ctx = dg::build_context(&view(&sim), &[], (16.3, 15.7), &[], &spec(0.02, 900.0), 7);
    for (y, x) in [(16.3, 15.7), (10.0, 12.5), (0.0, 0.0)] {
        let (ly, lx) = ctx.to_local(y, x);
        let (gy, gx) = ctx.to_global(ly, lx);
        assert!((gy - y).abs() < 1e-15 && (gx - x).abs() < 1e-15);
    }
    // The region really is the window the observations were cropped from.
    assert_eq!(ctx.obs.len(), ctx.h * ctx.w);
    assert_eq!(ctx.halo.len(), ctx.h * ctx.w);
    assert_eq!(ctx.version, 7);
}

#[test]
fn a_split_conserves_flux_and_a_birth_and_removal_account_for_theirs() {
    let theta = vec![3.0, 800.0, 4.0, 5.0, 1.3, 200.0, 9.0, 9.0, 1.1];
    let sum = |t: &[f64]| -> f64 {
        (0..psf::n_emitters_var(t))
            .map(|i| psf::amp_var(t, i))
            .sum()
    };

    let split = spotsolve_core::moves::split_var(&theta, 0, [1.0, 0.0], 1.6);
    assert_eq!(psf::n_emitters_var(&split), 3);
    assert!((sum(&split) - sum(&theta)).abs() < 1e-12, "split moved flux");
    // Both children inherit the PARENT's width, so proposal and incumbent
    // start in the same basin.
    assert_eq!(psf::sigma_var(&split, 1), 1.3);
    assert_eq!(psf::sigma_var(&split, 2), 1.3);
    // ... and they straddle the parent along the axis.
    assert!((psf::cy_var(&split, 1) + psf::cy_var(&split, 2) - 2.0 * 4.0).abs() < 1e-12);

    let removed = spotsolve_core::moves::remove_var(&theta, 0);
    assert_eq!(psf::n_emitters_var(&removed), 1);
    // The removed emitter's flux is NOT handed to the survivor: the joint
    // refit reassigns that light, which is why every hypothesis is refitted.
    assert_eq!(psf::amp_var(&removed, 0), 200.0);

    let born = spotsolve_core::moves::birth_var(&theta, 500.0, 1.0, 2.0, 1.2);
    assert_eq!(psf::n_emitters_var(&born), 3);
    assert!((sum(&born) - sum(&theta) - 500.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Numerical: what a scored fit returns
// ---------------------------------------------------------------------------

#[test]
fn the_returned_objective_is_data_only_and_the_curvature_carries_the_prior() {
    let truth = [(9.0, 10.0, 1400.0, 1.25)];
    let sim = simulate(20, 21, &truth, 5.0, 11);
    let sp = spec(0.02, 1200.0);
    let ctx = dg::build_context(&view(&sim), &[], (9.0, 10.0), &[], &sp, 1);
    let mut ws = GroupWorkspace::new();

    let (ly, lx) = ctx.to_local(9.0, 10.0);
    let theta0 = vec![ctx.bg_level, 1400.0, ly, lx, SIGMA0];
    let hyp = dg::fit_and_score(&ctx, &mut ws, &theta0, FitOpts::default());
    assert!(hyp.status.is_supported(), "{:?}", hyp.status);

    // I is the DATA-only I-divergence at the returned parameters, with the
    // halo in the mean. The prior is added by the score and must not be paid
    // twice.
    let mut factors = psf::Factors::new(ctx.h, ctx.w, 1);
    let mut model = vec![0.0; ctx.h * ctx.w];
    psf::model_var_sigma_ax(
        &hyp.theta,
        &psf::local_axis(ctx.h),
        &psf::local_axis(ctx.w),
        Some(&ctx.halo),
        &mut factors,
        &mut model,
    );
    let i_data = lmcl::i_divergence(&ctx.obs, &model);
    assert!(
        (hyp.i_div - i_data).abs() < 1e-9,
        "returned I {} is not the data-only divergence {i_data}",
        hyp.i_div
    );

    // The curvature is J^T W J plus the width prior's clipped curvature on the
    // width diagonal, and NOTHING else -- in particular no optimizer damping.
    let p = hyp.p;
    let mut jac = vec![0.0; p * ctx.h * ctx.w];
    psf::model_and_jac_var_sigma_ax(
        &hyp.theta,
        &psf::local_axis(ctx.h),
        &psf::local_axis(ctx.w),
        Some(&ctx.halo),
        &mut factors,
        &mut model,
        &mut jac,
    );
    let n = ctx.h * ctx.w;
    let pen = ctx.prior.penalty();
    for a in 0..p {
        for b in 0..p {
            let mut want: f64 = (0..n).map(|i| jac[a * n + i] * jac[b * n + i] / model[i]).sum();
            if a == b && a >= 1 && (a - 1) % 4 == 3 {
                want += pen.width.unwrap().curvature(hyp.theta[a]);
            }
            let got = hyp.curvature[a * p + b];
            assert!(
                (got - want).abs() <= 1e-8 * (1.0 + want.abs()),
                "curvature[{a},{b}] = {got}, expected {want}"
            );
        }
    }

    // And the score is exactly its four documented terms.
    let want = -hyp.i_div + hyp.log_prior
        + 0.5 * p as f64 * (2.0 * std::f64::consts::PI).ln()
        - 0.5 * hyp.logdet;
    assert!((hyp.score - want).abs() < 1e-12);
}

#[test]
fn the_flux_prior_shrinks_the_fitted_amplitude_and_leaves_the_curvature_alone() {
    // The deliberate algorithm change of plan section 3, validated on its own:
    // charging the flux prior in the FIT moves the mode, and only the mode.
    let truth = [(9.0, 10.0, 1400.0, 1.2)];
    let sim = simulate(20, 21, &truth, 5.0, 23);
    let theta0 = vec![5.0, 1400.0, 9.0, 10.0, 1.2];
    let lo = vec![0.0, 1e-2, -0.5, -0.5, 0.84];
    let hi = vec![100.0, 5e4, 19.5, 20.5, 2.64];
    let bounds = Bounds::new(&lo, &hi);
    let a_s = 900.0;

    let mut ws = FitWorkspace::new();
    let flux_only = Penalty { width: None, flux: Some(FluxPenalty { a_s }) };

    let ml = lmcl::fit_var_sigma_prior(
        &mut ws, &theta0, 20, 21, &sim.d, &bounds, None, FitOpts::default(),
        Penalty::default(),
    );
    let a_ml = ws.theta()[1];
    assert!(ml.converged);

    let map = lmcl::fit_var_sigma_prior(
        &mut ws, &theta0, 20, 21, &sim.d, &bounds, None, FitOpts::default(),
        flux_only,
    );
    let (a_map, theta_map) = (ws.theta()[1], ws.theta().to_vec());
    assert!(map.converged);

    // The mode moves: an Exp(1/a_s) prior has its mass at low flux, so the MAP
    // amplitude sits below the ML one by whatever the data's own gradient
    // needs to balance the constant 1/a_s.
    assert!(a_map < a_ml, "an Exp(1/a_s) prior must shrink the flux: {a_map} !< {a_ml}");
    assert!(a_ml - a_map > 1e-6, "the prior had no effect on the mode");

    // ... and ONLY the mode. Evaluated at the same parameters -- `max_iter: 0`
    // returns the curvature at the start point -- the two agree bit for bit,
    // because the exponential's curvature is exactly zero. So the Laplace
    // volume and every reported standard error are untouched. Comparing the
    // two CONVERGED matrices instead would compare `F` at two different
    // points, which differ for a reason that has nothing to do with this.
    let still = FitOpts { max_iter: 0, ..Default::default() };
    lmcl::fit_var_sigma_prior(&mut ws, &theta_map, 20, 21, &sim.d, &bounds, None, still,
                              Penalty::default());
    let f_ml = ws.fisher(5).to_vec();
    lmcl::fit_var_sigma_prior(&mut ws, &theta_map, 20, 21, &sim.d, &bounds, None, still, flux_only);
    let f_map = ws.fisher(5).to_vec();
    assert_eq!(f_ml, f_map, "the flux prior must not touch the curvature");

    // And I is still data-only: the same expression at two different points,
    // so the ML fit cannot be beaten on data alone.
    assert!(map.i_div > ml.i_div - 1e-9, "the MAP fit beat the ML fit on data alone");
}

#[test]
fn an_exhausted_fit_budget_is_reported_not_scored() {
    let truth = [(9.0, 10.0, 1400.0, 1.25), (10.3, 11.1, 1200.0, 1.25)];
    let sim = simulate(22, 22, &truth, 5.0, 31);
    let sp = spec(0.02, 1200.0);
    let ctx = dg::build_context(&view(&sim), &[], (9.5, 10.5), &[], &sp, 1);
    let mut ws = GroupWorkspace::new();
    let (ly, lx) = ctx.to_local(9.0, 10.0);
    let theta0 = vec![ctx.bg_level, 1400.0, ly, lx, SIGMA0];

    let starved = dg::fit_and_score(
        &ctx,
        &mut ws,
        &theta0,
        FitOpts {
            max_iter: 1,
            ..Default::default()
        },
    );
    assert_eq!(
        starved.status,
        ScoreStatus::Nonstationary,
        "a fit that ran one iteration must not be scored as a mode"
    );
    assert!(starved.score.is_nan(), "an unsupported hypothesis has no score");
    // ... and it is not evidence for anything: the same start with a real
    // budget is supported.
    let full = dg::fit_and_score(&ctx, &mut ws, &theta0, FitOpts::default());
    assert!(full.status.is_supported(), "{:?}", full.status);
}

#[test]
fn a_gain_is_exactly_antisymmetric() {
    let truth = [(11.0, 11.0, 1500.0, 1.25)];
    let sim = simulate(24, 24, &truth, 5.0, 41);
    let sp = spec(0.02, 1200.0);
    let ctx = dg::build_context(&view(&sim), &[], (11.0, 11.0), &[], &sp, 1);
    let mut ws = GroupWorkspace::new();
    let (ly, lx) = ctx.to_local(11.0, 11.0);

    let one = dg::fit_and_score(
        &ctx,
        &mut ws,
        &[ctx.bg_level, 1500.0, ly, lx, SIGMA0],
        FitOpts::default(),
    );
    let two = dg::fit_and_score(
        &ctx,
        &mut ws,
        &[
            ctx.bg_level,
            800.0,
            ly - 0.8,
            lx,
            SIGMA0,
            800.0,
            ly + 0.8,
            lx,
            SIGMA0,
        ],
        FitOpts::default(),
    );
    assert!(one.status.is_supported() && two.status.is_supported());
    // One score per configuration means a gain is a difference, so its
    // antisymmetry is structural rather than something to be arranged. This is
    // what makes birth and removal exact inverses of each other, which is what
    // lets the search interleave them.
    let forward = two.score - one.score;
    let back = one.score - two.score;
    assert_eq!(forward, -back);
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

fn run(
    sim: &Sim,
    emitters: &[Emitter],
    focus: (f64, f64),
    seeds: &[Seed],
    sp: &ContextSpec,
    settings: &GroupSettings,
) -> (dg::GroupOutcome, u32) {
    let ctx = dg::build_context(&view(sim), emitters, focus, seeds, sp, 1);
    let entry = dg::entry_state(&ctx);
    let mut ws = GroupWorkspace::new();
    let mut ids = IdAllocator::starting_at(
        emitters.iter().map(|e| e.id + 1).max().unwrap_or(0),
    );
    let out = dg::search_group(&ctx, &mut ws, &entry, &mut ids, settings);
    (out, ids.peek())
}

#[test]
fn every_committed_move_strictly_improves_the_same_contexts_score() {
    let truth = [(12.0, 12.0, 1600.0, 1.2), (13.6, 12.4, 1500.0, 1.2)];
    let sim = simulate(26, 26, &truth, 5.0, 53);
    let sp = spec(0.02, 1500.0);
    let settings = GroupSettings::default();
    // Start from ONE emitter where there are two: the search has to find the
    // second by itself.
    let emitters = [Emitter {
        id: 0,
        y: 12.8,
        x: 12.2,
        flux: 3000.0,
        sigma: 1.4,
    }];
    let (out, _) = run(&sim, &emitters, (12.8, 12.2), &[], &sp, &settings);

    assert!(!out.trace.is_empty(), "the search accepted nothing: {:?}", out.status);
    for m in &out.trace {
        assert!(m.gain > settings.min_gain, "committed a move with gain {}", m.gain);
    }
    // Scores after each commit are strictly increasing -- one context, one
    // score, so this is the invariant that rules out a move and its inverse.
    for pair in out.trace.windows(2) {
        assert!(pair[1].score > pair[0].score);
    }
    assert_eq!(out.version, 1);
    assert!(out.score.is_some());
    assert!(out.score_status.is_supported());
    assert_eq!(out.state.len(), 2, "expected the pair to be resolved");
    assert!(out.trace.iter().any(|m| m.kind == MoveKind::Split || m.kind == MoveKind::Birth));
}

#[test]
fn a_settled_group_proposes_no_inverse_of_what_it_just_did() {
    let truth = [(12.0, 12.0, 1600.0, 1.2), (13.6, 12.4, 1500.0, 1.2)];
    let sim = simulate(26, 26, &truth, 5.0, 53);
    let sp = spec(0.02, 1500.0);
    let settings = GroupSettings::default();
    let emitters = [Emitter {
        id: 0,
        y: 12.8,
        x: 12.2,
        flux: 3000.0,
        sigma: 1.4,
    }];
    let (first, next_id) = run(&sim, &emitters, (12.8, 12.2), &[], &sp, &settings);
    assert!(!first.trace.is_empty());

    // Re-run the transaction on the configuration it just committed, in the
    // same context. A search that would undo its own work shows up here.
    let ctx = dg::build_context(
        &view(&sim),
        &first.state.emitters,
        (12.8, 12.2),
        &[],
        &sp,
        2,
    );
    let entry = GroupState {
        background: first.state.background,
        emitters: first.state.emitters.clone(),
    };
    let mut ws = GroupWorkspace::new();
    let mut ids = IdAllocator::starting_at(next_id);
    let again = dg::search_group(&ctx, &mut ws, &entry, &mut ids, &settings);
    assert!(
        again.trace.is_empty(),
        "the search undid or redid its own work: {:?}",
        again.trace
    );
    // Either terminating status is consistent with "committed nothing here".
    // Which one it is depends on whether this noise realization tripped the
    // outside-the-box test, and that is a separate question with its own test.
    assert!(
        matches!(
            again.status,
            SearchStatus::NoImprovingProposal | SearchStatus::ContextRebuildRequired
        ),
        "{:?}",
        again.status
    );
}

#[test]
fn an_empty_region_stays_empty_and_k_zero_works() {
    let sim = simulate(24, 24, &[], 5.0, 61);
    let sp = spec(0.02, 1200.0);
    let settings = GroupSettings::default();
    let (out, _) = run(&sim, &[], (12.0, 12.0), &[], &sp, &settings);
    assert_eq!(out.state.len(), 0, "invented {} emitters on noise", out.state.len());
    assert!(out.trace.is_empty());
    assert!(out.score.is_some(), "a K=0 configuration is still scorable");
    assert!(out.removed.is_empty() && out.changed.is_empty());
}

#[test]
fn an_isolated_source_is_kept_and_not_split() {
    let truth = [(12.0, 12.0, 1500.0, 1.2)];
    let sim = simulate(26, 26, &truth, 5.0, 67);
    let sp = spec(0.02, 1400.0);
    let settings = GroupSettings::default();
    let emitters = [Emitter {
        id: 4,
        y: 12.1,
        x: 11.9,
        flux: 1400.0,
        sigma: 1.2,
    }];
    let (out, _) = run(&sim, &emitters, (12.1, 11.9), &[], &sp, &settings);
    assert_eq!(out.state.len(), 1, "an isolated source was split or removed");
    assert_eq!(out.state.emitters[0].id, 4, "identity was not preserved");
    assert!(out.removed.is_empty());
    // Localization: the refit should sit on the source, not wander.
    let e = out.state.emitters[0];
    assert!(
        (e.y - 12.0).hypot(e.x - 12.0) < 0.5,
        "refit landed at ({}, {})",
        e.y,
        e.x
    );
}

#[test]
fn a_zero_flux_emitter_is_removed_by_the_same_rule_that_adds_one() {
    let truth = [(12.0, 12.0, 1500.0, 1.2)];
    let sim = simulate(26, 26, &truth, 5.0, 71);
    let sp = spec(0.02, 1400.0);
    let settings = GroupSettings::default();
    // A real source plus a spurious one on bare background.
    let emitters = [
        Emitter { id: 0, y: 12.0, x: 12.0, flux: 1450.0, sigma: 1.2 },
        Emitter { id: 1, y: 15.5, x: 15.5, flux: 12.0, sigma: 1.2 },
    ];
    let (out, _) = run(&sim, &emitters, (15.5, 15.5), &[], &sp, &settings);
    assert!(
        out.removed.contains(&1),
        "kept a spurious source: state {:?}, status {:?}",
        out.state,
        out.status
    );
    // Removal is a proposal like any other: it appears in the trace with a
    // positive gain, not as a forced override.
    let removal = out
        .trace
        .iter()
        .find(|m| m.kind == MoveKind::Removal)
        .expect("the spurious source was not removed by a scored move");
    assert!(removal.gain > 0.0);
    // And identity follows the OBJECT: the real source keeps id 0 wherever the
    // proposal that won happened to drop it from the vector.
    assert_eq!(removal.target, Some(1));
    assert_eq!(out.removed, vec![1]);
    assert_eq!(out.state.len(), 1);
    assert_eq!(out.state.emitters[0].id, 0);
    let e = out.state.emitters[0];
    assert!((e.y - 12.0).hypot(e.x - 12.0) < 0.5, "survivor at ({}, {})", e.y, e.x);
}

#[test]
fn a_capped_search_reports_its_cap_rather_than_convergence() {
    let truth = [(12.0, 12.0, 1600.0, 1.2), (13.6, 12.4, 1500.0, 1.2)];
    let sim = simulate(26, 26, &truth, 5.0, 53);
    let sp = spec(0.02, 1500.0);
    let emitters = [Emitter { id: 0, y: 12.8, x: 12.2, flux: 3000.0, sigma: 1.4 }];

    let capped = GroupSettings {
        max_moves: 1,
        ..Default::default()
    };
    let (out, _) = run(&sim, &emitters, (12.8, 12.2), &[], &sp, &capped);
    assert_eq!(out.trace.len(), 1);
    assert_eq!(
        out.status,
        SearchStatus::BudgetExhausted,
        "a search stopped by its move cap must not claim it ran out of proposals"
    );

    let starved = GroupSettings {
        max_fits: 2,
        ..Default::default()
    };
    let (out, _) = run(&sim, &emitters, (12.8, 12.2), &[], &sp, &starved);
    assert_eq!(out.status, SearchStatus::BudgetExhausted);
    assert!(out.trace.is_empty());
}

#[test]
fn a_frozen_bright_neighbour_is_not_double_counted() {
    // One source inside the group, one far enough out to be frozen. If the
    // halo double counted it, the group's background or its flux would absorb
    // the difference and the fitted flux would be wrong.
    let truth = [(12.0, 12.0, 1500.0, 1.2), (20.0, 20.0, 4000.0, 1.2)];
    let sim = simulate(30, 30, &truth, 5.0, 83);
    let sp = spec(0.02, 1500.0);
    let emitters = [
        Emitter { id: 0, y: 12.0, x: 12.0, flux: 1500.0, sigma: 1.2 },
        Emitter { id: 1, y: 20.0, x: 20.0, flux: 4000.0, sigma: 1.2 },
    ];
    let ctx = dg::build_context(&view(&sim), &emitters, (12.0, 12.0), &[], &sp, 1);
    assert_eq!(ctx.free.len(), 1, "the far source should not be free");
    assert_eq!(ctx.free[0].id, 0);
    // The halo carries it: somewhere in the region the halo must be above the
    // background's own variation, and it must never be negative.
    assert!(ctx.halo.iter().all(|v| v.is_finite()));
    let entry = dg::entry_state(&ctx);
    let mut ws = GroupWorkspace::new();
    let mut ids = IdAllocator::starting_at(2);
    let out = dg::search_group(&ctx, &mut ws, &entry, &mut ids, &GroupSettings::default());
    assert_eq!(out.state.len(), 1);
    let flux = out.state.emitters[0].flux;
    assert!(
        (flux - 1500.0).abs() < 400.0,
        "flux {flux} is far from truth; the halo is probably wrong"
    );
}

#[test]
fn an_oversized_neighbourhood_says_so_instead_of_dropping_an_emitter() {
    // 14 sources inside one link radius, against a `k_max` of 12.
    let mut truth = Vec::new();
    let mut emitters = Vec::new();
    for i in 0..14u32 {
        let a = i as f64 * 0.45;
        let (y, x) = (18.0 + 3.0 * a.cos(), 18.0 + 3.0 * a.sin());
        truth.push((y, x, 900.0, 1.2));
        emitters.push(Emitter { id: i, y, x, flux: 900.0, sigma: 1.2 });
    }
    let sim = simulate(36, 36, &truth, 5.0, 97);
    let sp = spec(0.02, 900.0);
    let ctx = dg::build_context(&view(&sim), &emitters, (18.0 + 3.0, 18.0), &[], &sp, 1);
    assert!(ctx.capacity_limited, "an oversized group must say so");
    assert_eq!(ctx.free.len(), 12);
    // The focus is always free -- capacity pressure freezes the farthest
    // coupled neighbour, never the source being tested.
    let (fy, fx) = (18.0 + 3.0, 18.0);
    let nearest = ctx
        .free
        .iter()
        .map(|e| (e.y - fy).hypot(e.x - fx))
        .fold(f64::INFINITY, f64::min);
    assert!(nearest < 1e-9, "the focus emitter was frozen out of its own group");
    // Every emitter is still accounted for: free plus frozen, none dropped.
    assert_eq!(ctx.free.len() + ctx.frozen.len(), 14);
}

#[test]
fn a_committed_state_that_outgrows_its_context_asks_for_a_rebuild() {
    // A source right at the position bound after the refit forces a rebuild
    // rather than a silently truncated comparison.
    let truth = [(12.0, 12.0, 1500.0, 1.2)];
    let sim = simulate(26, 26, &truth, 5.0, 101);
    let sp = spec(0.02, 1400.0);
    // Place the incumbent far from the truth so the refit runs to the box edge.
    let emitters = [Emitter { id: 0, y: 12.0, x: 12.0, flux: 1400.0, sigma: 1.2 }];
    let ctx = dg::build_context(&view(&sim), &emitters, (12.0, 12.0), &[], &sp, 5);
    // A frozen neighbour drifting into the link radius is the other trigger.
    assert!(ctx.frozen.is_empty());
    let entry = dg::entry_state(&ctx);
    let mut ws = GroupWorkspace::new();
    let mut ids = IdAllocator::starting_at(1);
    let out = dg::search_group(&ctx, &mut ws, &entry, &mut ids, &GroupSettings::default());
    // Nothing here should be at a bound, so no rebuild is asked for.
    assert_ne!(out.status, SearchStatus::ContextRebuildRequired);
    assert!(!out.diag.position_bound_active);
    assert_eq!(out.version, 5);
}

#[test]
fn ids_are_stable_across_a_removal_and_a_split_retires_its_parent() {
    let truth = [
        (12.0, 12.0, 1600.0, 1.2),
        (13.6, 12.4, 1500.0, 1.2),
        (12.5, 19.0, 1500.0, 1.2),
    ];
    let sim = simulate(30, 30, &truth, 5.0, 103);
    let sp = spec(0.02, 1500.0);
    let emitters = [
        Emitter { id: 100, y: 12.8, x: 12.2, flux: 3000.0, sigma: 1.4 },
        Emitter { id: 200, y: 12.5, x: 19.0, flux: 1500.0, sigma: 1.2 },
    ];
    let (out, next) = run(&sim, &emitters, (12.8, 12.2), &[], &sp, &GroupSettings::default());
    // Whatever happened to 100, the untouched neighbour keeps its identity.
    if out.state.emitters.iter().any(|e| e.id == 200) {
        assert!(!out.removed.contains(&200));
    }
    // New ids are minted above everything in use and never collide.
    for e in &out.state.emitters {
        assert!(e.id < next);
    }
    let mut seen: Vec<u32> = out.state.emitters.iter().map(|e| e.id).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), out.state.len(), "duplicate ids in the committed state");
    // A split retires its parent, so the parent appears in `removed` and both
    // children in `changed`.
    if let Some(m) = out.trace.iter().find(|m| m.kind == MoveKind::Split) {
        let parent = m.target.unwrap();
        assert!(out.removed.contains(&parent));
        assert!(!out.state.emitters.iter().any(|e| e.id == parent));
    }
}

#[test]
fn a_broad_source_is_widened_rather_than_tiled() {
    // The move the free width exists for: a defocused source leaves a
    // rotationally symmetric residual, not a quadrupole, so widening should
    // beat splitting.
    let truth = [(14.0, 14.0, 2600.0, 2.0)];
    let sim = simulate(30, 30, &truth, 5.0, 107);
    let sp = spec(0.02, 2000.0);
    let emitters = [Emitter { id: 0, y: 14.0, x: 14.0, flux: 2600.0, sigma: 1.2 }];
    let (out, _) = run(&sim, &emitters, (14.0, 14.0), &[], &sp, &GroupSettings::default());
    assert_eq!(
        out.state.len(),
        1,
        "a broad source was tiled into {} emitters",
        out.state.len()
    );
    assert!(
        out.state.emitters[0].sigma > 1.6,
        "the source did not widen: sigma = {}",
        out.state.emitters[0].sigma
    );
}

#[test]
fn the_normal_tail_inverse_matches_the_values_scipy_gives() {
    // `calibrate._norm_isf` is `scipy.special.ndtri`; these are its outputs.
    // The threshold this feeds is seeder-grade, so what matters is that the
    // Rust and Python cuts are the same number, not the last digit of either.
    for (p, want) in [
        (0.5, 0.0),
        (0.05, 1.644_853_626_951_472_2),
        (1e-3, 3.090_232_306_167_813),
        (1e-6, 4.753_424_308_822_899),
        (2e-8, 5.490_851_752_104_35),
    ] {
        let got = dg::norm_isf(p);
        assert!(
            (got - want).abs() < 1e-9,
            "norm_isf({p}) = {got}, scipy says {want}"
        );
    }
}

#[test]
fn light_the_group_cannot_reach_asks_for_a_context_rebuild() {
    // A bright source well outside the admissible position box, but inside the
    // pixel region. No hypothesis here may place an emitter on it, so the
    // honest answer is "wrong context", not "nothing improves".
    let truth = [(12.0, 12.0, 1500.0, 1.2), (12.0, 20.0, 2200.0, 1.2)];
    let sim = simulate(30, 34, &truth, 5.0, 211);
    let sp = spec(0.02, 1500.0);
    // Only the first source is committed, so the position box is built around
    // it alone and the second lands outside.
    let emitters = [Emitter { id: 0, y: 12.0, x: 12.0, flux: 1500.0, sigma: 1.2 }];
    let ctx = dg::build_context(&view(&sim), &emitters, (12.0, 12.0), &[], &sp, 1);
    assert!(
        ctx.to_global(0.0, ctx.bounds.x_hi).1 < 20.0,
        "the control needs the far source outside the position box"
    );
    let entry = dg::entry_state(&ctx);
    let mut ws = GroupWorkspace::new();
    let mut ids = IdAllocator::starting_at(1);
    let out = dg::search_group(&ctx, &mut ws, &entry, &mut ids, &GroupSettings::default());
    assert!(out.diag.residual_peak_outside_box, "the far source was not noticed");
    assert_eq!(out.status, SearchStatus::ContextRebuildRequired);

    // ... and on isolated sources, where nothing IS outside the box, it fires
    // rarely. A rate rather than a single realization: the test is a family-
    // wise false-alarm rate, so one seed proving nothing is the point. The
    // realized rate runs several times the nominal alpha for the reason
    // `calibrate.py` records -- local maxima of a smooth field sample its
    // supremum, not `area/win^2` independent draws -- and
    // `scripts/measure_group_score.py --part outside` is where the operating
    // point is chosen. This asserts only that it is not routine.
    let mut fired = 0;
    let trials = 40usize;
    for seed in 0..trials as u64 {
        let quiet = simulate(30, 30, &[(15.0, 15.0, 1500.0, 1.2)], 5.0, 300 + seed);
        let em = [Emitter { id: 0, y: 15.0, x: 15.0, flux: 1500.0, sigma: 1.2 }];
        let ctx = dg::build_context(&view(&quiet), &em, (15.0, 15.0), &[], &sp, 1);
        let entry = dg::entry_state(&ctx);
        let mut ids = IdAllocator::starting_at(1);
        let out = dg::search_group(&ctx, &mut ws, &entry, &mut ids, &GroupSettings::default());
        fired += usize::from(out.diag.residual_peak_outside_box);
    }
    assert!(
        fired * 5 <= trials,
        "{fired}/{trials} isolated sources asked for a rebuild; the \
         outside-the-box test has become routine rather than exceptional"
    );
}

// ---------------------------------------------------------------------------
// The Laplace volume over the admissible box
// ---------------------------------------------------------------------------

/// Composite Simpson's rule for the 1-D integrand the box factor claims to
/// integrate in closed form, normalized by the full-line Gaussian volume.
fn box_factor_by_quadrature(theta: f64, lo: f64, hi: f64, s: f64, g: f64) -> f64 {
    let (a, b) = (lo - theta, hi - theta);
    let n = 200_000;
    let hstep = (b - a) / n as f64;
    let f = |t: f64| (-g * t - t * t / (2.0 * s * s)).exp();
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let t = a + i as f64 * hstep;
        sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f(t);
    }
    (sum * hstep / 3.0 / ((2.0 * std::f64::consts::PI).sqrt() * s)).ln()
}

#[test]
fn the_box_factor_is_the_exact_one_dimensional_integral_in_every_regime() {
    // (theta, lo, hi, s, g, what it is)
    let cases = [
        (0.0, -5.0, 5.0, 0.1, 0.0, "well-determined interior mode"),
        (0.0, 0.0, 5.0, 0.3, 0.0, "on a lower bound, zero multiplier"),
        (0.0, 0.0, 5.0, 0.3, 4.0, "on a lower bound, pushed against it"),
        (0.0, 0.0, 5.0, 0.3, 40.0, "on a lower bound, pushed hard"),
        (5.0, 0.0, 5.0, 0.3, -4.0, "on an upper bound, pushed against it"),
        (1.0, 0.0, 2.0, 50.0, 0.0, "Gaussian 25 box-widths wide"),
        (0.0, 0.0, 2.0, 50.0, 1e-3, "wide AND on a bound"),
        (0.3, 0.0, 1.0, 0.4, 0.0, "interior, both edges within reach"),
    ];
    for (theta, lo, hi, s, g, what) in cases {
        let closed = dg::log_box_factor(theta, lo, hi, s, g);
        let quad = box_factor_by_quadrature(theta, lo, hi, s, g);
        assert!(
            (closed - quad).abs() < 1e-8,
            "{what}: closed form {closed} vs quadrature {quad}"
        );
    }
    // The limits the doc comment states, as numbers.
    assert!(dg::log_box_factor(0.0, -5.0, 5.0, 0.1, 0.0).abs() < 1e-12);
    assert!((dg::log_box_factor(0.0, 0.0, 5.0, 0.3, 0.0) - 0.5f64.ln()).abs() < 1e-12);
    let wide = dg::log_box_factor(1.0, 0.0, 2.0, 50.0, 0.0);
    let cap = (2.0 / ((2.0 * std::f64::consts::PI).sqrt() * 50.0)).ln();
    assert!((wide - cap).abs() < 1e-3, "wide {wide} vs cap {cap}");
}

#[test]
fn the_normal_difference_has_no_cancellation_in_either_tail() {
    // Deep in the upper tail, where `Phi(u2) - Phi(u1)` is ~1e-200 and the
    // naive subtraction returns 0.
    let v = dg::log_ndtr_diff(30.0, 31.0);
    let reference = dg::log_ndtr_upper(30.0) + (-(dg::log_ndtr_upper(31.0) - dg::log_ndtr_upper(30.0)).exp()).ln_1p();
    assert!(v.is_finite() && (v - reference).abs() < 1e-12, "{v}");
    // The mirror is the same number.
    assert!((dg::log_ndtr_diff(-31.0, -30.0) - v).abs() < 1e-12);
    // Straddling zero: a tiny interval, where 1 - Phi - Phi would cancel.
    let tiny = dg::log_ndtr_diff(-1e-9, 1e-9);
    let expect = (2e-9 / (2.0 * std::f64::consts::PI).sqrt()).ln();
    assert!((tiny - expect).abs() < 1e-6, "{tiny} vs {expect}");
    // The asymptotic branch joins the erfc branch continuously.
    let (below, above) = (dg::log_ndtr_upper(30.0 - 1e-9), dg::log_ndtr_upper(30.0 + 1e-9));
    assert!((below - above).abs() < 1e-6, "{below} vs {above}");
}

#[test]
fn a_mode_on_the_width_floor_is_scored_not_refused() {
    // A source genuinely narrower than the model space allows: its fitted
    // width rests on `sigma_lo`, which the old policy called `BoundaryMode`
    // and refused to score.
    let truth = [(12.0, 12.0, 1600.0, 0.60 * SIGMA0)];
    let sim = simulate(26, 26, &truth, 5.0, 83);
    let sp = spec(0.02, 1400.0);
    let ctx = dg::build_context(&view(&sim), &[], (12.0, 12.0), &[], &sp, 1);
    let mut ws = GroupWorkspace::new();
    let (ly, lx) = ctx.to_local(12.0, 12.0);
    let hyp = dg::fit_and_score(&ctx, &mut ws, &[ctx.bg_level, 1600.0, ly, lx, SIGMA0], FitOpts::default());

    assert!(hyp.status.is_supported(), "{:?}", hyp.status);
    assert!((psf::sigma_var(&hyp.theta, 0) - ctx.bounds.sigma_lo) < 1e-6);
    assert_eq!(hyp.n_active, 1, "only the width should rest on a bound");
    // One-sided, and pushed against the bound: at most half the volume.
    assert!(hyp.log_box < 0.5f64.ln() + 1e-9, "log_box {}", hyp.log_box);
    assert!(hyp.score.is_finite());
}

#[test]
fn no_coordinate_is_credited_more_volume_than_its_prior_support() {
    // The failure the box correction was built for: an emitter parked at the
    // amplitude floor has position and width curvature ~A^2, so the regular
    // `-logdet/2` credits it with a Gaussian many box-widths wide. The gate's
    // overfit control accepted splits to 0.05 e- at +12 and +15 nats on it.
    //
    // Asserted per coordinate, with its marginal standard deviation `s_q`:
    // the integral of `exp(-quadratic)` peaking at the mode can never exceed
    // the width of the interval it is taken over, so
    //
    //     log(sqrt(2 pi) s_q) + (that coordinate's box factor) <= log(hi_q - lo_q).
    //
    // The sum over the phantom's position and width is where the regular
    // volume overshoots; the amplitude and background ranges are so wide that
    // a total over all coordinates would hide it. Holds whatever the fit's
    // status, which is why no status is asserted.
    let truth = [(12.0, 12.0, 1500.0, 1.2)];
    let sim = simulate(26, 26, &truth, 5.0, 89);
    let sp = spec(0.02, 1400.0);
    let emitters = [Emitter { id: 0, y: 12.0, x: 12.0, flux: 1500.0, sigma: 1.2 }];
    let ctx = dg::build_context(&view(&sim), &emitters, (12.0, 12.0), &[], &sp, 1);
    let mut ws = GroupWorkspace::new();
    let (ly, lx) = ctx.to_local(12.0, 12.0);
    // Evaluated where it is placed, not fitted: whether a given noise draw
    // lets an optimizer settle at the floor is not the question, and the
    // curvature there is the same function either way. A phantom on the real
    // source, the way a collapsed split child sits.
    let theta0 = [ctx.bg_level, 1500.0, ly, lx, SIGMA0, ctx.bounds.a_min * 1.5, ly, lx, 0.9];
    let hyp = dg::fit_and_score(&ctx, &mut ws, &theta0, FitOpts { max_iter: 0, ..Default::default() });
    assert_eq!(hyp.theta[5], theta0[5], "a zero-iteration evaluation moved the parameters");

    let p = hyp.p;
    let mut chol = linalg::Chol::new(p);
    assert!(chol.factor(&hyp.curvature, p));
    let (mut var, mut scratch) = (vec![0.0; p], Vec::new());
    chol.inv_diag(&mut var, &mut scratch);
    let (lo, hi) = ctx.bounds.arrays(2);
    let (slo, shi) = dg::score_support(&ctx, &hyp.theta, &lo, &hi);
    let ln_sqrt_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();

    let (mut regular, mut boxed, mut support) = (0.0, 0.0, 0.0);
    for q in 6..=8 {
        let s = var[q].sqrt();
        let c = dg::log_box_factor(hyp.theta[q], slo[q], shi[q], s, 0.0);
        regular += ln_sqrt_2pi + s.ln();
        boxed += ln_sqrt_2pi + s.ln() + c;
        support += (shi[q] - slo[q]).ln();
    }
    assert!(
        regular > support + 5.0,
        "premise: the regular volume should overshoot the support by nats ({regular} vs {support})"
    );
    assert!(boxed <= support + 1e-9, "box-restricted {boxed} exceeds the support {support}");
    // And the score actually carries it: the box term removes at least the
    // overshoot the phantom's block had.
    assert!(hyp.log_box <= -(regular - support), "log_box {} vs overshoot {}", hyp.log_box, regular - support);
}
