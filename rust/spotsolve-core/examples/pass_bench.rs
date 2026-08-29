//! Wall time for each pass on the fixture's states, against the Python's own.
//! Run: `cargo run --release --example pass_bench`.

use spotsolve_core::evidence::Prior;
use spotsolve_core::passes::{self, Emitters, Frame, Solver};
use std::time::Instant;

fn mat(v: &serde_json::Value) -> (usize, usize, Vec<f64>) {
    let rows = v.as_array().unwrap();
    let nc = rows[0].as_array().unwrap().len();
    (rows.len(), nc, rows.iter().flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap())).collect())
}
fn vec_(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn main() {
    let txt = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/06_passes.json"),
    ).unwrap();
    let root: serde_json::Value = serde_json::from_str(&txt).unwrap();
    println!("{:>6} {:>5} {:>10} {:>10} {:>10} {:>10}", "seed", "N", "add ms", "split ms", "refine ms", "prune ms");
    for case in root["cases"].as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let (h, w, d_e) = mat(&case["d_e"]);
        let (_, _, bmap) = mat(&case["bmap"]);
        let frame = Frame { d_e: &d_e, bmap: &bmap, h, w,
            sigma: case["sigma"].as_f64().unwrap(),
            k_max: case["k_max"].as_u64().unwrap() as usize };
        let prior = Prior { lam: case["lam"].as_f64().unwrap(), a_s: case["A_s"].as_f64().unwrap() };
        let (_, _, cand) = mat(&case["add"]["cand"]);
        let camp = vec_(&case["add"]["camp"]);
        let (_, _, model) = mat(&case["split"]["model"]);
        let (_, _, p0) = mat(&case["positions"]);
        let a0 = vec_(&case["amplitudes"]);
        let mut s = Solver::new();

        let mut t = Instant::now();
        let mut em = Emitters::from_parts(p0.clone(), a0.clone());
        passes::add_pass(&mut s, &frame, &mut em, &cand, &camp, prior);
        let t_add = t.elapsed().as_secs_f64() * 1e3;

        t = Instant::now();
        passes::split_pass(&mut s, &frame, &mut em, &model, prior);
        let t_split = t.elapsed().as_secs_f64() * 1e3;

        t = Instant::now();
        passes::refine(&mut s, &frame, &mut em, 200, 1, passes::REFINE_TOL);
        let t_refine = t.elapsed().as_secs_f64() * 1e3;

        let n = em.len();
        t = Instant::now();
        passes::prune(&mut s, &frame, &mut em, prior);
        let t_prune = t.elapsed().as_secs_f64() * 1e3;

        println!("{seed:>6} {n:>5} {t_add:>10.1} {t_split:>10.1} {t_refine:>10.1} {t_prune:>10.1}");
    }
}
