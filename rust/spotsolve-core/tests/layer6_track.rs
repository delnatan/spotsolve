//! The linker and its parameter estimator against `tracksolve`, the Python
//! reference they were ported from.
//!
//! `tests/fixtures/08_track.json` is three simulated movies in pixels and
//! frames, spanning step/nearest-neighbour ratios 0.11, 0.31 and 0.53 -- the
//! GEM dataset measures about 0.3 -- with their tracksolve `mode="lap"`
//! linking and the parameters tracksolve's empirical-Bayes loop fitted to
//! them. It is FROZEN, like every other fixture here.
//!
//! Two different claims are checked, at two different strengths:
//!
//! - **Linking, given parameters: identical.** Both sides solve the same
//!   assignment exactly, so with the same scores the answer is the same
//!   labelling, detection for detection.
//! - **The parameter fit: close.** The estimator's one deliberate departure
//!   is the wrong-neighbour density, a kernel density on log distance that
//!   the port evaluates by binning rather than by the exact O(N^2) sum
//!   scipy uses. That moves the fitted numbers by ~1e-4 relative, and a
//!   different fitted parameter can then flip a genuinely borderline link,
//!   so the linking from a self-fitted parameter set is compared as a switch
//!   rate rather than row by row.
//!
//! The reference was generated with tracksolve's gate reading the INFLATED
//! CRLB, which the port does and tracksolve (in the version ported) does not:
//! its `gate.accept` sees the raw `se^2` while its filter sees
//! `se^2 * se_inflate`, so above an inflation of 1 its gate is tighter than
//! its own model and the gate's miss-rate guarantee does not hold.

mod common;

use common::{f64_at, load, vec_at};
use spotsolve_core::track::{Detections, Params, link};
use spotsolve_core::trackparams::fit;

fn detections(case: &serde_json::Value) -> Detections {
    let frame: Vec<i64> = case["frame"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let (y, x) = (vec_at(case, "y"), vec_at(case, "x"));
    let (sy, sx) = (vec_at(case, "se_y"), vec_at(case, "se_x"));
    let mut pos = Vec::with_capacity(2 * y.len());
    let mut se = Vec::with_capacity(2 * y.len());
    for i in 0..y.len() {
        pos.push(y[i]);
        pos.push(x[i]);
        se.push(sy[i]);
        se.push(sx[i]);
    }
    Detections::new(&frame, &pos, &se).expect("fixture detections are valid")
}

fn reference_params(case: &serde_json::Value) -> Params {
    let p = &case["params"];
    Params {
        d_grid: vec_at(p, "d_grid"),
        d_logprior: vec_at(p, "d_logprior"),
        p_cont: f64_at(p, "p_cont"),
        lam_birth: f64_at(p, "lam_birth"),
        se_inflate: f64_at(p, "se_inflate"),
    }
}

/// Switches per 100 links against the truth implied by the reference
/// labelling: of the consecutive pairs this linking claims, the share the
/// reference does not also claim. Used only where an exact match is not the
/// right standard (see the module docs).
fn disagreement(d: &Detections, got: &[u32], want: &[u32]) -> f64 {
    let pairs = |t: &[u32]| -> std::collections::HashSet<(usize, usize)> {
        let mut by_track: std::collections::HashMap<u32, Vec<usize>> = Default::default();
        for (r, &id) in t.iter().enumerate() {
            by_track.entry(id).or_default().push(r);
        }
        let mut out = std::collections::HashSet::new();
        for rows in by_track.values() {
            for w in rows.windows(2) {
                out.insert((w[0], w[1]));
            }
        }
        out
    };
    let _ = d;
    let (a, b) = (pairs(got), pairs(want));
    if a.is_empty() {
        return 0.0;
    }
    100.0 * (1.0 - a.intersection(&b).count() as f64 / a.len() as f64)
}

#[test]
fn linking_is_identical_to_tracksolve_given_its_parameters() {
    let fx = load("08_track");
    for case in fx.cases() {
        let name = case["name"].as_str().unwrap();
        let d = detections(case);
        let p = reference_params(case);
        p.check().unwrap();
        let got = link(&d, &p);

        let want: Vec<u32> = case["track_id"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            got.n_tracks as usize,
            case["n_tracks"].as_u64().unwrap() as usize,
            "{name}: track count"
        );
        // `Detections` sorts by frame; the fixture is already in that order,
        // and `order` maps back either way.
        let mut mine = vec![0u32; want.len()];
        for (r, &id) in got.track.iter().enumerate() {
            mine[d.order[r]] = id;
        }
        let bad: Vec<usize> = (0..want.len()).filter(|&i| mine[i] != want[i]).collect();
        assert!(
            bad.is_empty(),
            "{name}: {} of {} detections got a different track id (first: row {}, \
             got {}, want {})",
            bad.len(),
            want.len(),
            bad[0],
            mine[bad[0]],
            want[bad[0]]
        );
    }
}

#[test]
fn the_fitted_parameters_match_tracksolves() {
    let fx = load("08_track");
    for case in fx.cases() {
        let name = case["name"].as_str().unwrap();
        let d = detections(case);
        let (p, traj) = fit(&d);
        let want = reference_params(case);

        assert_eq!(p.d_grid.len(), want.d_grid.len(), "{name}: grid size");
        for (i, (&g, &w)) in p.d_grid.iter().zip(&want.d_grid).enumerate() {
            assert!(
                (g - w).abs() <= 1e-9 * w.max(1e-12),
                "{name}: d_grid[{i}] {g} vs {w}"
            );
        }
        // The prior is a distribution: compare it as one, in total variation.
        let tv: f64 = p
            .d_logprior
            .iter()
            .zip(&want.d_logprior)
            .map(|(a, b)| (a.exp() - b.exp()).abs())
            .sum::<f64>()
            / 2.0;
        assert!(
            tv < 0.02,
            "{name}: D prior differs by {tv} in total variation"
        );
        for (got, wanted, what, tol) in [
            (p.p_cont, want.p_cont, "p_cont", 1e-3),
            (p.lam_birth, want.lam_birth, "lam_birth", 5e-2),
            (p.se_inflate, want.se_inflate, "se_inflate", 2e-2),
        ] {
            let rel = (got - wanted).abs() / wanted.abs().max(1e-12);
            assert!(rel <= tol, "{name}: {what} {got} vs {wanted} (rel {rel})");
        }
        assert_eq!(traj.len(), 4, "{name}: initialize plus three iterations");

        // And the linking those parameters produce must agree with the
        // reference linking to well under one switch per 100 links.
        let got = link(&d, &p);
        let want_ids: Vec<u32> = case["track_id"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let mut mine = vec![0u32; want_ids.len()];
        for (r, &id) in got.track.iter().enumerate() {
            mine[d.order[r]] = id;
        }
        let disagree = disagreement(&d, &mine, &want_ids);
        assert!(
            disagree < 0.5,
            "{name}: self-fitted linking differs from the reference on \
             {disagree:.2} per 100 links"
        );
    }
}
