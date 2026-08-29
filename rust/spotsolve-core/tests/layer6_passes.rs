//! LAYER 6: the four passes, against `tests/fixtures/06_passes.json`.
//!
//! Run in the order ADD -> SPLIT -> REFINE -> PRUNE: that is `detect`'s own
//! order and the passes are not independent of it.
//!
//! Emitters are matched by nearest neighbour rather than by index. A split's
//! two children may be listed in either order (the principal axis is defined
//! only up to sign), and a one-emitter difference renumbers everything after
//! it, so index-wise comparison would report noise instead of the defect.

mod common;

use common::*;
use spotsolve_core::evidence::Prior;
use spotsolve_core::passes::{self, Emitters, Frame, Solver};
use spotsolve_core::render;
use serde_json::Value;

fn emitters(case: &Value, key: &str) -> Emitters {
    let (n, two, pos) = mat_at(case, key);
    assert_eq!(two, 2, "{key} is not (N, 2)");
    let amp = vec_at(case, &key.replace("positions", "amplitudes"));
    assert_eq!(amp.len(), n);
    Emitters::from_parts(pos, amp)
}

fn sub_emitters(case: &Value, block: &str) -> Emitters {
    let b = &case[block];
    let (n, two, pos) = mat_at(b, "positions");
    assert_eq!(two, 2);
    let amp = vec_at(b, "amplitudes");
    assert_eq!(amp.len(), n);
    Emitters::from_parts(pos, amp)
}

/// Positions closer than this are treated as one unresolvable pair.
///
/// Below ~1 sigma a pair's Fisher matrix is nearly singular along its
/// separation direction: the likelihood is flat along that ridge, so where the
/// optimizer stops on it is set by the last ulp, and the two implementations
/// legitimately stop in different places. README section 15 records this as the
/// largest known open defect of the algorithm itself.
const UNRESOLVED_SIGMA: f64 = 1.0;

/// Tolerance for an emitter with no sub-sigma neighbour. These are determined,
/// and they agree to 1e-10 in practice.
const TOL_RESOLVED_PX: f64 = 1e-6;

/// Tolerance for an emitter inside a sub-sigma pair, along the flat direction.
///
/// Measured: the worst such disagreement in these fixtures is 3.6e-3 px, on a
/// pair at 0.39 sigma. For scale, an honest SE for such a pair is ~0.354 px, so
/// this bound is still seven times tighter than the noise -- and a real
/// decision error looks nothing like it: when a `COND_GUARD` mismatch made the
/// port refuse a split the Python accepted, the resulting positions moved by
/// 0.77 px, twenty times this.
const TOL_UNRESOLVED_PX: f64 = 0.05;

/// Tolerance for every other emitter in a configuration that contains a
/// sub-sigma pair somewhere.
///
/// The leak is not bounded by any radius, so this is deliberately not a
/// geometric test. A degenerate pair is frozen as a constant into every patch
/// within `HALO_FACTOR * sigma` of it, and `prune`'s faintest-first write-back
/// cascade then carries the perturbation further still -- that cascade is the
/// point of the pass, not a side effect (README section 9). Trying to bound it
/// geometrically means chasing one emitter at a time.
///
/// Both numbers this sits between are measured. The largest leak outside a pair
/// in these fixtures is 5.3e-6 px, so this allows 200x margin; a real decision
/// change -- the `COND_GUARD` mismatch that made the port refuse a split --
/// moved positions by 0.77 px, 770x this. There is no ambiguity about which
/// side of the bound a genuine regression lands on.
const TOL_CONTAMINATED_PX: f64 = 1e-3;

/// Nearest-neighbour match, with the tolerance chosen per emitter by whether it
/// is identifiable at all. Also checks total flux, which IS determined even
/// where the split between a degenerate pair's members is not.
/// `before` is the configuration the pass was HANDED. Contamination is a
/// property of the input as much as the output: `prune` removing one member of
/// a degenerate pair leaves a survivor with no sub-sigma neighbour, which looks
/// fully determined but has absorbed the pair's ambiguity along with its flux.
/// README section 15 records that as the algorithm's largest open defect, so
/// the test has to know about it rather than be surprised by it.
fn check(got: &Emitters, want: &Emitters, before: &Emitters, sigma: f64, what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: N differs ({} vs {}). A different count means a different \
         DECISION somewhere -- chase that in the layer below; pairwise \
         comparison here would be meaningless.",
        got.len(),
        want.len()
    );
    // Members of a sub-sigma pair, in either the input or the output.
    let pair_sites: Vec<(f64, f64)> = [before, got]
        .iter()
        .flat_map(|em| {
            (0..em.len()).filter_map(move |i| {
                let close = (0..em.len()).any(|k| {
                    k != i
                        && (em.y(i) - em.y(k)).hypot(em.x(i) - em.x(k)) < UNRESOLVED_SIGMA * sigma
                });
                close.then(|| (em.y(i), em.x(i)))
            })
        })
        .collect();
    let in_pair: Vec<bool> = (0..got.len())
        .map(|i| {
            pair_sites
                .iter()
                .any(|&(y, x)| (got.y(i) - y).hypot(got.x(i) - x) < UNRESOLVED_SIGMA * sigma)
        })
        .collect();

    for i in 0..got.len() {
        let mut best = (f64::INFINITY, usize::MAX);
        for j in 0..want.len() {
            let d = (got.y(i) - want.y(j)).hypot(got.x(i) - want.x(j));
            if d < best.0 {
                best = (d, j);
            }
        }
        let (tol, why) = if in_pair[i] {
            (TOL_UNRESOLVED_PX, "is inside a sub-sigma pair")
        } else if !pair_sites.is_empty() {
            (TOL_CONTAMINATED_PX, "shares a configuration with a sub-sigma pair")
        } else {
            (TOL_RESOLVED_PX, "is fully determined")
        };
        assert!(
            best.0 <= tol,
            "{what}: emitter at ({:.4}, {:.4}) is {:.3e} px from its match, over \
             the {tol:.0e} px bound for an emitter that {why}",
            got.y(i),
            got.x(i),
            best.0
        );
        let unresolved = in_pair[i];
        if !unresolved {
            let (ga, wa) = (got.amp[i], want.amp[best.1]);
            assert!(
                (ga - wa).abs() / wa.abs().max(1.0) <= 1e-4,
                "{what}: amplitude {ga:.3} vs {wa:.3} at ({:.4}, {:.4})",
                got.y(i),
                got.x(i)
            );
        }
    }
    // Total flux is determined even where its division between the members of a
    // degenerate pair is not, so it is asserted unconditionally.
    let (gf, wf): (f64, f64) = (got.amp.iter().sum(), want.amp.iter().sum());
    assert!(
        (gf - wf).abs() / wf.max(1.0) <= 1e-4,
        "{what}: total flux {gf:.3} vs {wf:.3} -- flux has been misassigned, \
         which a per-emitter comparison inside a degenerate pair would hide"
    );
}

#[test]
fn passes_match_the_fixture() {
    let fx = load("06_passes");
    let mut s = Solver::new();

    for case in fx.cases() {
        let seed = usize_at(case, "seed");
        let (h, w, d_e) = mat_at(case, "d_e");
        let (_, _, bmap) = mat_at(case, "bmap");
        let sigma = f64_at(case, "sigma");
        let frame = Frame {
            d_e: &d_e,
            bmap: &bmap,
            h,
            w,
            sigma,
            k_max: usize_at(case, "k_max"),
        };
        let prior = Prior { lam: f64_at(case, "lam"), a_s: f64_at(case, "A_s") };

        // --- ADD -------------------------------------------------------
        let mut em = emitters(case, "positions");
        let add = &case["add"];
        let (n_cand, two, cand) = mat_at(add, "cand");
        assert_eq!(two, 2);
        let camp = vec_at(add, "camp");
        assert_eq!(camp.len(), n_cand);

        let before = em.clone();
        let n_added = passes::add_pass(&mut s, &frame, &mut em, &cand, &camp, prior);
        assert_eq!(n_added, usize_at(add, "n_added"), "seed {seed}: ADD accepted a different count");
        check(&em, &sub_emitters(case, "add"), &before, sigma, &format!("seed {seed} ADD"));

        // --- SPLIT -----------------------------------------------------
        let (_, _, model) = mat_at(&case["split"], "model");
        let before = em.clone();
        let n_split = passes::split_pass(&mut s, &frame, &mut em, &model, prior);
        assert_eq!(
            n_split,
            usize_at(&case["split"], "n_split"),
            "seed {seed}: SPLIT accepted a different count"
        );
        check(&em, &sub_emitters(case, "split"), &before, sigma, &format!("seed {seed} SPLIT"));

        // --- REFINE ----------------------------------------------------
        let before = em.clone();
        let se = passes::refine(&mut s, &frame, &mut em, 200, usize_at(&case["refine"], "max_sweeps"), passes::REFINE_TOL);
        check(&em, &sub_emitters(case, "refine"), &before, sigma, &format!("seed {seed} REFINE"));
        assert_eq!(se.len(), 3 * em.len(), "se has the wrong shape");
        // The fixture stores -1.0 where the Python reported NaN.
        let (_, _, want_se) = mat_at(&case["refine"], "se");
        let n_finite_want = want_se.iter().filter(|v| **v >= 0.0).count();
        let n_finite_got = se.iter().filter(|v| v.is_finite()).count();
        assert!(
            n_finite_got >= n_finite_want,
            "seed {seed}: REFINE reported fewer usable standard errors ({n_finite_got}) \
             than the Python ({n_finite_want})"
        );

        // --- PRUNE -----------------------------------------------------
        let before = em.clone();
        let n_before = em.len();
        let n_removed = passes::prune(&mut s, &frame, &mut em, prior);
        assert_eq!(n_before - n_removed, em.len(), "prune's own count disagrees with its output");
        check(&em, &sub_emitters(case, "prune"), &before, sigma, &format!("seed {seed} PRUNE"));
    }
}

/// ADD and SPLIT take `K -> K+1` and PRUNE only removes. The two loops do not
/// alternate, so no move can undo another and termination is structural rather
/// than a tolerance being chased.
#[test]
fn add_and_split_are_monotone_and_prune_only_removes() {
    let fx = load("06_passes");
    let mut s = Solver::new();
    let case = &fx.cases()[0];
    let (h, w, d_e) = mat_at(case, "d_e");
    let (_, _, bmap) = mat_at(case, "bmap");
    let frame = Frame {
        d_e: &d_e,
        bmap: &bmap,
        h,
        w,
        sigma: f64_at(case, "sigma"),
        k_max: usize_at(case, "k_max"),
    };
    let prior = Prior { lam: f64_at(case, "lam"), a_s: f64_at(case, "A_s") };

    let mut em = emitters(case, "positions");
    let n0 = em.len();
    let add = &case["add"];
    let (_, _, cand) = mat_at(add, "cand");
    let n_added = passes::add_pass(&mut s, &frame, &mut em, &cand, &vec_at(add, "camp"), prior);
    assert_eq!(em.len(), n0 + n_added, "ADD is not monotone in N");

    let (_, _, model) = mat_at(&case["split"], "model");
    let n1 = em.len();
    let n_split = passes::split_pass(&mut s, &frame, &mut em, &model, prior);
    assert_eq!(em.len(), n1 + n_split, "SPLIT is not monotone in N");

    let n2 = em.len();
    passes::refine(&mut s, &frame, &mut em, 200, 1, passes::REFINE_TOL);
    assert_eq!(em.len(), n2, "REFINE changed N -- it must propose nothing");

    let n_removed = passes::prune(&mut s, &frame, &mut em, prior);
    assert_eq!(em.len(), n2 - n_removed, "PRUNE did not only remove");
}

/// The global model must agree with what the passes actually fit. A drift here
/// would make `find_candidates` propose on a residual the search never sees.
#[test]
fn rendered_model_reproduces_the_fixture_model() {
    let fx = load("06_passes");
    let case = &fx.cases()[0];
    let (h, w, bmap) = mat_at(case, "bmap");
    let em = sub_emitters(case, "add");
    let (_, _, want) = mat_at(&case["split"], "model");

    let mut got = render::render_model(
        &em.pos,
        &em.amp,
        em.len(),
        f64_at(case, "sigma"),
        h,
        w,
        0.0,
        render::RENDER_TRUNCATE,
    );
    for (g, b) in got.iter_mut().zip(bmap.iter()) {
        *g += b;
    }
    assert_all_rel(&got, &want, 1e-12, "render(positions, amplitudes, sigma, bmap)");
}
