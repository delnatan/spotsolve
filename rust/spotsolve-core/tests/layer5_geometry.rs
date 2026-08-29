//! LAYER 5: patch decomposition, window selection, the emitter free-mask and
//! model rendering, against `tests/fixtures/05_geometry.json`.
//!
//! This is the layer the port deliberately *changes*: `cKDTree` +
//! `connected_components` become a uniform grid plus union-find, and the
//! `O(N*H*W)` mask sweep becomes an `O(N*sigma^2)` stamp [P9]. Same answers by
//! a different algorithm -- which is exactly when a golden fixture earns its
//! keep. The indices and bboxes are integers, so these are exact assertions
//! with no tolerance to spend.

mod common;

use common::*;
use spotsolve_core::grid::EmitterGrid;
use spotsolve_core::patches::{self, BBox, HALO_FACTOR};
use spotsolve_core::render;
use serde_json::Value;

fn positions_flat(case: &Value) -> (Vec<f64>, usize) {
    let (n, two, p) = mat_at(case, "positions");
    assert_eq!(two, 2);
    (p, n)
}

fn u32s(v: &Value, key: &str) -> Vec<u32> {
    v[key].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

#[test]
fn patch_decomposition_matches_the_fixture() {
    let fx = load("05_geometry");
    for case in fx.cases() {
        let (pos, n) = positions_flat(case);
        let (h, w) = (usize_at(case, "H"), usize_at(case, "W"));
        let sigma = f64_at(case, "sigma");
        let k_max = usize_at(case, "k_max");

        let got = patches::build_patches(&pos, n, sigma, h, w, k_max);
        let want = case["patches"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "different number of patches");

        // Patch ORDER is not part of the contract -- within one refine sweep
        // the patches are independent, because each builds its halo from the
        // sweep's input state rather than from its own neighbours' updates.
        // So compare as a set, keyed by the free-index list.
        let mut got_keyed: Vec<(Vec<u32>, &patches::Patch)> = got
            .iter()
            .map(|p| {
                let mut v = p.indices.clone();
                v.sort_unstable();
                (v, p)
            })
            .collect();
        got_keyed.sort_by(|a, b| a.0.cmp(&b.0));
        let mut want_keyed: Vec<(Vec<u32>, &Value)> =
            want.iter().map(|p| (u32s(p, "indices"), p)).collect();
        want_keyed.sort_by(|a, b| a.0.cmp(&b.0));

        for ((gi, gp), (wi, wp)) in got_keyed.iter().zip(&want_keyed) {
            assert_eq!(gi, wi, "patch membership differs");
            let mut gf = gp.frozen.clone();
            gf.sort_unstable();
            assert_eq!(&gf, &u32s(wp, "frozen_indices"), "frozen halo differs for {gi:?}");
            assert_eq!(
                gp.bbox,
                BBox {
                    y0: usize_at(wp, "y0"),
                    x0: usize_at(wp, "x0"),
                    y1: usize_at(wp, "y1"),
                    x1: usize_at(wp, "x1"),
                },
                "bbox differs for {gi:?}"
            );
        }

        // Every emitter belongs to exactly one patch, and no patch exceeds
        // k_max -- the invariants the joint Fisher matrix's size depends on.
        let mut seen = vec![0usize; n];
        for p in &got {
            assert!(p.indices.len() <= k_max, "patch of {} exceeds k_max", p.indices.len());
            for &i in &p.indices {
                seen[i as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "patches are not a partition");
    }
}

/// `free` is ordered by distance to the candidate, not by index, and that order
/// decides the theta layout of the fit. Asserted positionally, unlike the
/// frozen list.
#[test]
fn window_selection_matches_the_fixture() {
    let fx = load("05_geometry");
    for case in fx.cases() {
        let (pos, n) = positions_flat(case);
        let (h, w) = (usize_at(case, "H"), usize_at(case, "W"));
        let sigma = f64_at(case, "sigma");
        let k_max = usize_at(case, "k_max");
        let grid = EmitterGrid::build(&pos, n, h, w, HALO_FACTOR * sigma);
        let mut scratch = Vec::new();

        for win in case["windows"].as_array().unwrap() {
            let c = vec_at(win, "cand");
            let (free, frozen, bbox) =
                patches::window(&pos, n, c[0], c[1], sigma, h, w, k_max, &grid, &mut scratch);
            assert_eq!(free, u32s(win, "free"), "free set (or its order) differs at {c:?}");
            let mut f = frozen.clone();
            f.sort_unstable();
            assert_eq!(f, u32s(win, "frozen"), "frozen halo differs at {c:?}");
            assert_eq!(
                bbox,
                BBox {
                    y0: usize_at(win, "y0"),
                    x0: usize_at(win, "x0"),
                    y1: usize_at(win, "y1"),
                    x1: usize_at(win, "x1"),
                },
                "bbox differs at {c:?}"
            );
        }
    }
}

/// The stamped mask must agree with the swept one pixel for pixel -- it is the
/// same predicate, visited in a different order.
#[test]
fn free_mask_matches_the_fixture() {
    let fx = load("05_geometry");
    for case in fx.cases() {
        let (pos, n) = positions_flat(case);
        let (h, w) = (usize_at(case, "H"), usize_at(case, "W"));
        let (_, _, want) = mat_at(case, "free_mask");
        let got = render::emitter_free_mask(
            &pos,
            n,
            f64_at(case, "sigma"),
            f64_at(case, "mask_radius"),
            h,
            w,
        );
        for i in 0..h * w {
            assert_eq!(
                got[i],
                want[i] != 0.0,
                "mask differs at pixel ({}, {})",
                i / w,
                i % w
            );
        }
    }
}

#[test]
fn render_model_matches_the_fixture() {
    let fx = load("05_geometry");
    for case in fx.cases() {
        let (pos, n) = positions_flat(case);
        let (h, w) = (usize_at(case, "H"), usize_at(case, "W"));
        let amp = vec_at(case, "amplitudes");
        let (_, _, want) = mat_at(case, "render");
        let got = render::render_model(
            &pos,
            &amp,
            n,
            f64_at(case, "sigma"),
            h,
            w,
            0.0,
            f64_at(case, "render_truncate"),
        );
        assert_all_rel(&got, &want, 1e-13, "render_model");
    }
}

/// The frozen halo is rendered WITHOUT truncation, unlike the global model.
/// Truncating it would reintroduce the pedestal the 5-sigma radius exists to
/// remove -- at 3 sigma a patch could be handed an unmodelled 3.1 e- offset on
/// a 4 e- background.
#[test]
fn halo_is_untruncated_and_local() {
    let (h, w, sigma) = (9usize, 9usize, 1.2);
    // One emitter well outside the patch: at 8 px it is beyond 4*sigma = 4.8,
    // so a truncated render would contribute exactly nothing.
    let pos = [(-8.0f64), 4.0];
    let amp = [2000.0];
    let mut out = Vec::new();
    render::halo_image(&pos, &amp, &[0], sigma, 0, 0, h, w, &mut out);
    let total: f64 = out.iter().sum();
    assert!(total > 0.0, "the halo truncated an emitter it must not have");

    // Local coordinates: shifting the patch origin and the emitter together
    // must leave the rendered halo unchanged.
    let pos2 = [(-8.0f64) + 20.0, 4.0 + 30.0];
    let mut out2 = Vec::new();
    render::halo_image(&pos2, &amp, &[0], sigma, 20, 30, h, w, &mut out2);
    assert_eq!(out, out2, "halo_image is not translation-invariant in local coords");
}

/// `residual_axis` and `split` against the fixture's unresolved-pair cases.
///
/// The principal axis is defined only up to sign, so the axis is compared as
/// `|u . u_py| == 1` and the split's children as a set. `strength` is
/// sign-independent and is compared directly -- it is what `split_pass` ranks
/// on, and that ranking is load-bearing.
#[test]
fn residual_axis_and_split_match_the_fixture() {
    use spotsolve_core::moves;
    let fx = load("05_geometry");
    for case in fx.cases() {
        let (h, w) = (usize_at(case, "H"), usize_at(case, "W"));
        let sigma = f64_at(case, "sigma");
        for mv in case["moves"].as_array().unwrap() {
            let theta = vec_at(mv, "theta");
            let (_, _, resid) = mat_at(mv, "resid");
            let want_u = vec_at(mv, "u");
            let want_s = f64_at(mv, "strength");

            let (u, s) = moves::residual_axis(
                spotsolve_core::psf::cy(&theta, 0),
                spotsolve_core::psf::cx(&theta, 0),
                spotsolve_core::psf::amp(&theta, 0),
                0.0,
                0.0,
                h,
                w,
                sigma,
                &resid,
            );
            assert_rel(s, want_s, 1e-10, "residual_axis strength");
            let dot = (u[0] * want_u[0] + u[1] * want_u[1]).abs();
            assert_abs(dot, 1.0, 1e-9, "residual_axis principal axis");
            assert_abs(u[0] * u[0] + u[1] * u[1], 1.0, 1e-12, "axis is not a unit vector");
            assert!(u[0] > 0.0 || (u[0] == 0.0 && u[1] >= 0.0), "canonical sign not applied");

            let got = moves::split(&theta, 0, u, f64_at(mv, "disp") * sigma);
            let want = vec_at(mv, "split");
            assert_eq!(got.len(), want.len(), "split produced a different K");
            assert_abs(got[0], want[0], 1e-12, "split background");
            // Children as a set: a flipped `u` swaps them.
            for a in 0..2 {
                let best = (0..2)
                    .map(|b| {
                        (spotsolve_core::psf::cy(&got, a) - spotsolve_core::psf::cy(&want, b)).abs()
                            + (spotsolve_core::psf::cx(&got, a) - spotsolve_core::psf::cx(&want, b)).abs()
                    })
                    .fold(f64::INFINITY, f64::min);
                assert!(best < 1e-9, "split child {a} has no match in the fixture");
            }
        }
    }
}

/// A split conserves total flux exactly, which is what makes the amplitude
/// prior term collapse to `-log(A_s)`. If flux leaked, the Bayes factor would
/// charge a split for flux it never added.
#[test]
fn split_conserves_flux_and_is_symmetric() {
    use spotsolve_core::{moves, psf};
    let theta = psf::pack(4.0, &[900.0, 1500.0], &[5.0, 9.0], &[6.0, 2.0]);
    let u = [0.6, 0.8];
    let disp = 1.44;
    let out = moves::split(&theta, 1, u, disp);

    assert_eq!(psf::n_emitters(&out), 3);
    let before: f64 = (0..2).map(|k| psf::amp(&theta, k)).sum();
    let after: f64 = (0..3).map(|k| psf::amp(&out, k)).sum();
    assert_abs(after, before, 1e-12, "split did not conserve flux");

    // The untouched emitter keeps its place, and the children are appended.
    assert_abs(psf::amp(&out, 0), 900.0, 1e-12, "untouched emitter moved");
    // Children straddle the parent, half a displacement each way.
    let mid_y = 0.5 * (psf::cy(&out, 1) + psf::cy(&out, 2));
    let mid_x = 0.5 * (psf::cx(&out, 1) + psf::cx(&out, 2));
    assert_abs(mid_y, 9.0, 1e-12, "children are not centred on the parent");
    assert_abs(mid_x, 2.0, 1e-12, "children are not centred on the parent");
    let sep = ((psf::cy(&out, 1) - psf::cy(&out, 2)).powi(2)
        + (psf::cx(&out, 1) - psf::cx(&out, 2)).powi(2))
    .sqrt();
    assert_abs(sep, disp, 1e-12, "children are not `disp` apart");

    // The amplitude floor keeps a split of a near-zero emitter well posed.
    let faint = psf::pack(0.0, &[1e-12], &[3.0], &[3.0]);
    let out = moves::split(&faint, 0, u, disp);
    assert!(psf::amp(&out, 0) >= moves::A_MIN, "split fell below the amplitude floor");
}
