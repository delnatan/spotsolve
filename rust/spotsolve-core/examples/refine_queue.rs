//! Does the group-wise work queue actually drain?
//!
//! REFINE's groups are spatially independent by construction, so a group whose
//! own emitters and frozen halo have not moved cannot change. Scheduling on
//! that should let quiet regions drop out and let the loop terminate. This
//! measures whether they do, against the threshold that decides "has moved" --
//! which is the whole question, since the optimizer's own noise moves every
//! emitter a little on every refit.
//!
//! Run: `cargo run --release --example refine_queue`.

use spotsolve_core::passes::{self, Emitters, Frame, PatchCost, Solver};
use std::time::Instant;

const GS: bool = true;

fn mat(v: &serde_json::Value) -> (usize, usize, Vec<f64>) {
    let rows = v.as_array().unwrap();
    let nc = rows[0].as_array().unwrap().len();
    (rows.len(), nc, rows.iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap())).collect())
}
fn vec_(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn main() {
    let txt = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/06_passes.json"),
    ).unwrap();
    let root: serde_json::Value = serde_json::from_str(&txt).unwrap();

    for case in root["cases"].as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let (h, w, d_e) = mat(&case["d_e"]);
        let (_, _, bmap) = mat(&case["bmap"]);
        let frame = Frame { d_e: &d_e, bmap: &bmap, h, w,
            sigma: case["sigma"].as_f64().unwrap(),
            k_max: case["k_max"].as_u64().unwrap() as usize };
        let (_, _, pos) = mat(&case["split"]["positions"]);
        let em0 = Emitters::from_parts(pos, vec_(&case["split"]["amplitudes"]));
        let n = em0.len();
        println!("\n=== seed {seed}: N={n}, {h}x{w} ===");
        println!("{:>10}  {:>44}  {:>9} {:>8}", "move_eps", "groups fitted per pass", "LM iters", "ms");

        for &eps in &[0.0f64, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1] {
            let mut em = em0.clone();
            let mut next = em.clone();
            let mut se = vec![f64::NAN; 3 * n];
            let mut cost: Vec<PatchCost> = Vec::new();
            let mut dirty = vec![true; n];
            let mut moved = vec![false; n];
            let (mut hist, mut iters) = (Vec::new(), 0usize);
            let t = Instant::now();
            for _ in 0..8 {
                let fitted = passes::refine_sweep_costed(
                    &mut Solver::new(), &frame, &em, 200, &mut next, &mut se, None,
                    &mut cost, passes::REFINE_TOL_OBJ, Some(&dirty), Some(&mut moved), eps, None, GS);
                iters += cost.iter().map(|c| c.n_iter).sum::<usize>();
                hist.push(fitted);
                std::mem::swap(&mut em, &mut next);
                if fitted == 0 || !moved.iter().any(|&m| m) { break; }
                dirty.copy_from_slice(&moved);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let cells: Vec<String> = hist.iter().map(|f| format!("{f}")).collect();
            println!("{eps:>10.0e}  {:>44}  {iters:>9} {ms:>8.1}", cells.join(" "));
        }
    }
}
