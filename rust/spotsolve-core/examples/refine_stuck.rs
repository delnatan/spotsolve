//! What is a REFINE fit that never converges?
//!
//! One such group costs 35-40% of a whole sweep's LM iterations. This asks what
//! they are, because "non-identifiable" should mean a direction of the Fisher
//! matrix carrying no information -- a statement about `F`, not about the
//! optimizer having a bad day.
//!
//! Run: `cargo run --release --example refine_stuck`.

use spotsolve_core::passes::{self, Emitters, Frame, PatchCost, Solver};

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
        let sigma = case["sigma"].as_f64().unwrap();
        let frame = Frame { d_e: &d_e, bmap: &bmap, h, w, sigma,
            k_max: case["k_max"].as_u64().unwrap() as usize };
        let (_, _, pos) = mat(&case["split"]["positions"]);
        let mut em = Emitters::from_parts(pos, vec_(&case["split"]["amplitudes"]));
        let n = em.len();
        let mut next = em.clone();
        let mut se = vec![f64::NAN; 3 * n];
        let mut cost: Vec<PatchCost> = Vec::new();
        let mut s = Solver::new();
        println!("\n=== seed {seed}: N={n}, {h}x{w}, COND_GUARD = {:.0e} ===",
                 spotsolve_core::evidence::COND_GUARD);
        println!("{:>5} {:>6} {:>3} {:>5} {:>8} {:>12} {:>12} {:>6}",
                 "sweep", "iters", "K", "conv", "pair px", "scaled cond",
                 "bound frac", "which");
        let kind = |k: u8| match k { 0 => "bg", 1 => "A", 2 => "y", _ => "x" };

        for k in 0..8 {
            passes::refine_sweep_costed(&mut s, &frame, &em, 200, &mut next, &mut se,
                                        None, &mut cost, passes::REFINE_TOL_OBJ,
                                        None, None, 0.0, None, false);
            // Every fit that stopped without converging, plus the worst-
            // conditioned one that DID converge, for contrast.
            let mut worst_ok: Option<&PatchCost> = None;
            for c in cost.iter() {
                if !c.converged {
                    println!("{k:>5} {:>6} {:>3} {:>5} {:>8.3} {:>12.3e} {:>12.2e} {:>6}",
                             c.n_iter, c.k, "NO", c.min_pair_px, c.scaled_cond,
                             c.min_bound_frac, kind(c.min_bound_kind));
                } else if worst_ok.map_or(true, |b| c.scaled_cond > b.scaled_cond) {
                    worst_ok = Some(c);
                }
            }
            if let Some(c) = worst_ok {
                println!("{k:>5} {:>6} {:>3} {:>5} {:>8.3} {:>12.3e} {:>12.2e} {:>6}  <- worst conv",
                         c.n_iter, c.k, "yes", c.min_pair_px, c.scaled_cond,
                         c.min_bound_frac, kind(c.min_bound_kind));
            }
            std::mem::swap(&mut em, &mut next);
        }
    }
}
