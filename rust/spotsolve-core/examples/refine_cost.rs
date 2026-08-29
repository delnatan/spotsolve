//! Where does REFINE's time actually go?
//!
//! `refine` is ~51% of a 512x512 frame, and end-to-end its sweeps past the
//! first buy nothing measurable. This measures the sweep at the level the
//! optimizer sees it: how many LM iterations each patch fit costs, how that
//! decays across sweeps, how many patches are still moving, and how many were
//! **bit-reproducible** -- reading inputs identical to the previous sweep, so
//! that skipping them would have been exact rather than an approximation.
//!
//! Run: `cargo run --release --example refine_cost`.

use spotsolve_core::passes::{self, Emitters, Frame, PatchCost, Solver};
use std::time::Instant;

fn mat(v: &serde_json::Value) -> (usize, usize, Vec<f64>) {
    let rows = v.as_array().unwrap();
    let nc = rows[0].as_array().unwrap().len();
    (rows.len(), nc, rows.iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
        .collect())
}
fn vec_(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn pct(v: &mut Vec<usize>, q: f64) -> usize {
    v.sort_unstable();
    if v.is_empty() { return 0; }
    v[(((v.len() - 1) as f64) * q).round() as usize]
}

fn main() {
    let txt = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/06_passes.json"),
    ).unwrap();
    let root: serde_json::Value = serde_json::from_str(&txt).unwrap();
    const NS: usize = 8;

    let tol_obj: f64 = std::env::args().nth(1)
        .map(|a| a.parse().expect("tol_obj"))
        .unwrap_or(passes::REFINE_TOL_OBJ);
    println!("tol_obj = {tol_obj:e}");
    for case in root["cases"].as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let (h, w, d_e) = mat(&case["d_e"]);
        let (_, _, bmap) = mat(&case["bmap"]);
        let frame = Frame { d_e: &d_e, bmap: &bmap, h, w,
            sigma: case["sigma"].as_f64().unwrap(),
            k_max: case["k_max"].as_u64().unwrap() as usize };
        let (_, _, pos) = mat(&case["split"]["positions"]);
        let mut em = Emitters::from_parts(pos, vec_(&case["split"]["amplitudes"]));
        let mut next = em.clone();
        let mut se = vec![f64::NAN; 3 * em.len()];
        let mut cost: Vec<PatchCost> = Vec::new();
        let mut prev: Option<Emitters> = None;
        let mut s = Solver::new();

        println!("\n=== seed {seed}: N={}, {h}x{w} ===", em.len());
        println!("{:>5} {:>8} {:>7} {:>7} {:>7} {:>7} {:>6} {:>8} {:>9} {:>10} {:>8}",
                 "sweep", "patches", "it sum", "it med", "it p90", "it max",
                 "!conv", "its", "at floor", "floor its", "ms");
        for k in 0..NS {
            let t = Instant::now();
            passes::refine_sweep_costed(&mut s, &frame, &em, 200, &mut next,
                                        &mut se, prev.as_ref(), &mut cost, tol_obj, None, None, 0.0, None, false);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let mut its: Vec<usize> = cost.iter().map(|c| c.n_iter).collect();
            let sum: usize = its.iter().sum();
            let nconv = cost.iter().filter(|c| !c.converged).count();
            // What the non-converged fits COST, which is the number that
            // bounds any "give up early" change.
            let bad_it: usize = cost.iter().filter(|c| !c.converged)
                .map(|c| c.n_iter).sum();
            let nmov = cost.iter().filter(|c| c.moved > passes::REFINE_TOL).count();
            let nrep = cost.iter().filter(|c| c.reproducible).count();
            // Emitters pinned at the amplitude floor, and what the groups
            // holding them cost.
            let nfloor: usize = cost.iter().map(|c| c.n_at_floor).sum();
            let floor_it: usize = cost.iter().filter(|c| c.n_at_floor > 0)
                .map(|c| c.n_iter).sum();
            let (med, p90) = (pct(&mut its, 0.5), pct(&mut its, 0.9));
            let mx = *its.iter().max().unwrap_or(&0);
            let _ = (nmov, nrep);
            println!("{k:>5} {:>8} {sum:>7} {med:>7} {p90:>7} {mx:>7} {nconv:>6} {:>7.1}% {nfloor:>8} {:>9.1}% {ms:>8.1}",
                     cost.len(), 100.0 * bad_it as f64 / sum.max(1) as f64,
                     100.0 * floor_it as f64 / sum.max(1) as f64);
            prev = Some(em.clone());
            std::mem::swap(&mut em, &mut next);
        }
    }
}
