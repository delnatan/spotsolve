//! The joint model against the frozen Python prototype's stages
//! (`tests/fixtures/11_joint.json`, written by `scripts/make_joint_fixture.py`).
//!
//! Every stage starts from the prototype's own one-pass emitters `e0`, so
//! one-pass drift (layer 7) cannot leak in. Nodes, pair couplings, groups and
//! kappa are deterministic linear algebra and agree to rounding. One round
//! (with or without count changes) is a sequence of the same LM fits from the
//! same starts, so emitters agree to 1e-5 except on GEM, whose coincident
//! splits are chaotic (see layer 7); GEM is checked for stages only.
//!
//! These run [`Config::prototype`]. The fast default is held to the
//! prototype's converged answer statistically (last test).

mod common;

use serde_json::Value;
use spotsolve_core::boxsearch::{self as bs, Em, Settings, Workspace};
use spotsolve_core::joint::{Config, Joint};

fn ems_at(v: &Value, key: &str) -> Vec<Em> {
    v[key].as_array().unwrap().iter()
        .map(|e| {
            let e: Vec<f64> = e.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            [e[0], e[1], e[2], e[3]]
        })
        .collect()
}

fn frames() -> std::collections::HashMap<String, Vec<f64>> {
    let fx = common::load("10_scoregate");
    fx.cases().iter().map(|c| (c["name"].as_str().unwrap().to_string(), common::vec_at(c, "frame"))).collect()
}

fn assert_ems(got: &[Em], want: &[Em], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: emitters");
    for (k, (g, e)) in got.iter().zip(want).enumerate() {
        common::assert_rel(g[0], e[0], 1e-5, &format!("{what}: A[{k}]"));
        for c in 1..4 {
            common::assert_abs(g[c], e[c], 1e-5, &format!("{what}: [{k}][{c}]"));
        }
    }
}

#[test]
fn the_joint_stages_reproduce_the_prototype() {
    let fx = common::load("11_joint");
    let src = frames();
    for case in fx.cases() {
        let name = case["name"].as_str().unwrap();
        let shape = common::vec_at(case, "shape");
        let (h, w) = (shape[0] as usize, shape[1] as usize);
        let frame = if case.get("frame").is_some() { common::vec_at(case, "frame") } else { src[name].clone() };
        let (sigma, offset) = (common::f64_at(case, "sigma"), common::f64_at(case, "offset"));
        let s = Settings { sigma, fp_per_mpx: bs::FP_PER_MPX, slack: bs::SLACK };
        let d: Vec<f64> = frame.iter().map(|v| v - offset).collect();
        let bg0 = bs::median_background(&d, h, w);
        let phi = bs::dispersion(&d, h, w);
        let u = bs::threshold(sigma, s.fp_per_mpx);
        let e0 = ems_at(case, "e0");
        let mut ws = Workspace::new();
        let fresh = |ws: &mut Workspace| {
            let mut jm = Joint::new(&d, h, w, &e0, &bg0, phi, u, &s, None, ws);
            jm.config = Config::prototype();
            jm
        };

        let jm = fresh(&mut ws);
        common::assert_all_rel(&jm.nodes.beta, &common::vec_at(case, "beta_pre"), 1e-8, &format!("{name}: beta_pre"));
        let model = jm.model();
        let pairs = jm.pairs(&model, &mut ws);
        let want = case["pairs"].as_array().unwrap();
        assert_eq!(pairs.len(), want.len(), "{name}: pairs");
        for (k, (p, q)) in pairs.iter().zip(want).enumerate() {
            let q: Vec<f64> = q.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            assert_eq!((p.0, p.1), (q[0] as usize, q[1] as usize), "{name}: pair {k}");
            common::assert_abs(p.2, q[2], 1e-9, &format!("{name}: rho2[{k}]"));
        }
        let groups: Vec<Vec<usize>> = case["groups"].as_array().unwrap().iter()
            .map(|g| g.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect())
            .collect();
        assert_eq!(jm.groups(&model, &mut ws), groups, "{name}: groups");
        common::assert_rel(jm.kappa(), common::f64_at(case, "kappa"), 1e-9, &format!("{name}: kappa"));
        if name == "gem" {
            continue;
        }

        for (key, add) in [("fit_round", false), ("add_round", true)] {
            let r = &case[key];
            let what = format!("{name} {key}");
            let mut jm = fresh(&mut ws);
            jm.round(add, &mut ws);
            assert_ems(&jm.ems, &ems_at(r, "emitters"), &what);
            common::assert_all_rel(&jm.nodes.beta, &common::vec_at(r, "beta"), 1e-6, &format!("{what}: beta"));
            assert_eq!(jm.stats.fits, common::usize_at(r, "fits"), "{what}: fits");
            assert_eq!(jm.stats.adds, common::usize_at(r, "adds"), "{what}: adds");
            assert_eq!(jm.stats.removed, common::usize_at(r, "removed"), "{what}: removed");
            assert_eq!(jm.stats.lr_fail, common::usize_at(r, "lr_fail"), "{what}: lr_fail");
            common::assert_rel(jm.stats.kappa, common::f64_at(r, "kappa"), 1e-9, &format!("{what}: kappa"));
        }
    }
}

/// The frozen configuration run to convergence from the same `e0`. Measured
/// on this fixture, every case (GEM included) reproduced the prototype's
/// decisions exactly and its emitters to 2e-7 px, so the decisions are pinned
/// here and positions held to 1e-5 px, with GEM given 1e-4 for its ridges.
#[test]
fn the_converged_model_reproduces_the_prototype() {
    let fx = common::load("11_joint");
    let src = frames();
    for case in fx.cases() {
        let name = case["name"].as_str().unwrap();
        let shape = common::vec_at(case, "shape");
        let (h, w) = (shape[0] as usize, shape[1] as usize);
        let frame = if case.get("frame").is_some() { common::vec_at(case, "frame") } else { src[name].clone() };
        let (sigma, offset) = (common::f64_at(case, "sigma"), common::f64_at(case, "offset"));
        let s = Settings { sigma, fp_per_mpx: bs::FP_PER_MPX, slack: bs::SLACK };
        let d: Vec<f64> = frame.iter().map(|v| v - offset).collect();
        let bg0 = bs::median_background(&d, h, w);
        let (phi, u) = (bs::dispersion(&d, h, w), bs::threshold(sigma, s.fp_per_mpx));
        let mut ws = Workspace::new();
        let mut jm = Joint::new(&d, h, w, &ems_at(case, "e0"), &bg0, phi, u, &s, None, &mut ws);
        jm.config = Config::prototype();
        jm.run(&mut ws);
        let r = &case["full"];
        for (key, got) in [("outer", jm.stats.outer), ("adds", jm.stats.adds), ("removed", jm.stats.removed),
                           ("fits", jm.stats.fits), ("lr_fail", jm.stats.lr_fail)] {
            assert_eq!(got, common::usize_at(r, key), "{name} full: {key}");
        }
        common::assert_rel(jm.stats.kappa, common::f64_at(r, "kappa"), 1e-9, &format!("{name} full: kappa"));
        let want = ems_at(r, "emitters");
        assert_eq!(jm.ems.len(), want.len(), "{name} full: emitters");
        let tol = if name == "gem" { 1e-4 } else { 1e-5 };
        for (k, (g, e)) in jm.ems.iter().zip(&want).enumerate() {
            for c in 1..3 {
                common::assert_abs(g[c], e[c], tol, &format!("{name} full: [{k}][{c}]"));
            }
        }
    }
}

/// The fast default against the prototype's converged answer: count within
/// 2% (+1) and emitters within 0.5 px of one of its. Measured: every count
/// identical; agreement 1.000 on every case but GEM, 0.939 there, where the
/// 1e5-ADU objects' coincident splits settle differently (see layer 7).
#[test]
fn the_fast_default_agrees_with_the_converged_prototype() {
    let fx = common::load("11_joint");
    let src = frames();
    for case in fx.cases() {
        let name = case["name"].as_str().unwrap();
        let shape = common::vec_at(case, "shape");
        let (h, w) = (shape[0] as usize, shape[1] as usize);
        let frame = if case.get("frame").is_some() { common::vec_at(case, "frame") } else { src[name].clone() };
        let (sigma, offset) = (common::f64_at(case, "sigma"), common::f64_at(case, "offset"));
        let s = Settings { sigma, fp_per_mpx: bs::FP_PER_MPX, slack: bs::SLACK };
        let d: Vec<f64> = frame.iter().map(|v| v - offset).collect();
        let bg0 = bs::median_background(&d, h, w);
        let (phi, u) = (bs::dispersion(&d, h, w), bs::threshold(sigma, s.fp_per_mpx));
        let mut ws = Workspace::new();
        let mut jm = Joint::new(&d, h, w, &ems_at(case, "e0"), &bg0, phi, u, &s, None, &mut ws);
        jm.run(&mut ws);
        let want = ems_at(&case["full"], "emitters");
        let near = |a: &[Em], b: &[Em]| a.iter()
            .filter(|e| b.iter().any(|g| (g[1] - e[1]).hypot(g[2] - e[2]) < 0.5)).count();
        let (n0, n1) = (want.len(), jm.ems.len());
        let agree = if n0 + n1 == 0 { 1.0 } else { (near(&want, &jm.ems) + near(&jm.ems, &want)) as f64 / (n0 + n1) as f64 };
        println!("{name}: N {n1} vs {n0}, agreement {agree:.3}, fits {} vs {}", jm.stats.fits, common::usize_at(&case["full"], "fits"));
        assert!(n1.abs_diff(n0) as f64 <= 0.02 * n0 as f64 + 1.0, "{name}: N {n1} vs {n0}");
        let floor = if name == "gem" { 0.9 } else { 0.999 };
        assert!(agree >= floor, "{name}: agreement {agree}");
    }
}
