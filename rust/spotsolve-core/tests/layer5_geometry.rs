//! LAYER 5: patch decomposition, the emitter free-mask and model rendering,
//! against `tests/fixtures/05_geometry.json`.
//!
//! This is the layer the port deliberately *changes*: `cKDTree` +
//! `connected_components` become a uniform grid plus union-find, and the
//! `O(N*H*W)` mask sweep becomes an `O(N*sigma^2)` stamp [P9]. Same answers by
//! a different algorithm -- which is exactly when a golden fixture earns its
//! keep. The indices and bboxes are integers, so these are exact assertions
//! with no tolerance to spend.

mod common;

use common::*;
use spotsolve_core::patches::{self, BBox};
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
        // The width-aware render, every emitter at the fixture's one width.
        let sig = vec![f64_at(case, "sigma"); n];
        let got = render::render_model(
            &pos,
            &amp,
            &sig,
            h,
            w,
            &vec![0.0; h * w],
            f64_at(case, "render_truncate"),
        );
        assert_all_rel(&got, &want, 1e-13, "render_model");
    }
}
