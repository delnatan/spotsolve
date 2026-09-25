//! LAYER 5: model rendering against `tests/fixtures/05_geometry.json`.

mod common;

use common::*;
use spotsolve_core::render;
use serde_json::Value;

fn positions_flat(case: &Value) -> (Vec<f64>, usize) {
    let (n, two, p) = mat_at(case, "positions");
    assert_eq!(two, 2);
    (p, n)
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
