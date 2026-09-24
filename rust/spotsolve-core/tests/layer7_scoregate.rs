//! The one-pass score-gated search against the frozen Python prototype
//! (`tests/fixtures/10_scoregate.json`, written by
//! `scripts/make_scoregate_fixture.py` at the prototype's last commit).
//!
//! The median background is exact (whole ADU). Filters differ from scipy only
//! in summation order, so u and phi agree to rounding and seeds exactly.
//! Every fit is the same LM fitter from the same start, so on the synthetic
//! and bead cases each emitter agrees to 1e-5.
//!
//! The GEM frame holds objects of 1e5-2e5 ADU that both implementations
//! split into near-coincident components (flux Fisher fractions below 0.01).
//! Their split lies on a ridge of the likelihood, and rounding-level input
//! differences settle it differently, which then reaches later windows as
//! neighbour light: 74 of 246 emitters move. There the contract is the
//! decision counts and the fitted model image, within 2 noise sd (measured
//! worst: 1.42).

mod common;

use spotsolve_core::boxsearch::{self as bs, Settings, Workspace};
use spotsolve_core::render::render_model;

#[test]
fn the_port_reproduces_the_prototype() {
    let fx = common::load("10_scoregate");
    let fp = fx.root["fp_per_mpx"].as_f64().unwrap();
    let slack = common::vec_at(&fx.root, "slack");
    for case in fx.cases() {
        let name = case["name"].as_str().unwrap();
        let shape = common::vec_at(case, "shape");
        let (h, w) = (shape[0] as usize, shape[1] as usize);
        let frame = common::vec_at(case, "frame");
        let (sigma, offset) = (common::f64_at(case, "sigma"), common::f64_at(case, "offset"));
        let s = Settings { sigma, fp_per_mpx: fp, slack: (slack[0], slack[1]) };
        let d: Vec<f64> = frame.iter().map(|v| v - offset).collect();
        let o = bs::one_pass(&d, h, w, None, &s, &mut Workspace::new());

        common::assert_rel(o.u, common::f64_at(case, "u"), 1e-12, &format!("{name}: u"));
        common::assert_rel(o.dispersion, common::f64_at(case, "phi"), 1e-9, &format!("{name}: phi"));
        if case.get("background").is_some() {
            assert_eq!(o.background, common::vec_at(case, "background"), "{name}: background");
        }
        assert_eq!(o.n_seeds, case["seeds"].as_array().unwrap().len(), "{name}: seeds");
        let (o_amp, o_pos, o_sig): (Vec<f64>, Vec<f64>, Vec<f64>) = (
            o.ems.iter().map(|e| e[0]).collect(),
            o.ems.iter().flat_map(|e| [e[1], e[2]]).collect(),
            o.ems.iter().map(|e| e[3]).collect(),
        );
        let want: Vec<Vec<f64>> = case["emitters"].as_array().unwrap().iter()
            .map(|e| e.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
            .collect();
        assert_eq!(o_amp.len(), want.len(), "{name}: emitters");
        assert_eq!(o.fits, common::usize_at(case, "fits"), "{name}: fits");
        assert_eq!(o.lr_fail, common::usize_at(case, "lr_fail"), "{name}: lr_fail");
        if name == "gem" {
            let pos: Vec<f64> = want.iter().flat_map(|e| [e[1], e[2]]).collect();
            let amp: Vec<f64> = want.iter().map(|e| e[0]).collect();
            let sig: Vec<f64> = want.iter().map(|e| e[3]).collect();
            let m0 = render_model(&pos, &amp, &sig, h, w, &o.background, 6.0);
            let m1 = render_model(&o_pos, &o_amp, &o_sig, h, w, &o.background, 6.0);
            let worst = (0..h * w)
                .map(|i| (m1[i] - m0[i]).abs() / (o.dispersion * m0[i].max(1.0)).sqrt())
                .fold(0.0, f64::max);
            assert!(worst < 2.0, "gem: model differs by {worst} noise sd");
            continue;
        }
        for (k, e) in want.iter().enumerate() {
            common::assert_rel(o_amp[k], e[0], 1e-5, &format!("{name}: A[{k}]"));
            common::assert_abs(o_pos[2 * k], e[1], 1e-5, &format!("{name}: y[{k}]"));
            common::assert_abs(o_pos[2 * k + 1], e[2], 1e-5, &format!("{name}: x[{k}]"));
            common::assert_abs(o_sig[k], e[3], 1e-5, &format!("{name}: sigma[{k}]"));
        }
    }
}
