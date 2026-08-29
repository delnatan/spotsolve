//! What does REFINE's objective tolerance cost, in units of a standard error?
//!
//! `tol_obj` is a predicted decrease in **nats**, but REFINE's output is a
//! position in **pixels** with a reported CRLB. Near the optimum
//! `I(t) ~ I_min + 0.5 dt' F dt`, so stopping when the predicted decrease falls
//! below `tol` leaves the parameter about `sqrt(2*tol)` standard errors short.
//! That algebra is the argument for loosening it; this measures whether it
//! actually holds, by refitting every patch against a `tol_obj = 1e-12`
//! reference and reporting the shift in units of that patch's own reported SE.
//!
//! Run: `cargo run --release --example refine_tol`.

use spotsolve_core::passes::{self, Emitters, Frame, PatchCost, Solver};

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
fn quant(v: &mut Vec<f64>, q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { return f64::NAN; }
    v[(((v.len() - 1) as f64) * q).round() as usize]
}

fn main() {
    let txt = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/06_passes.json"),
    ).unwrap();
    let root: serde_json::Value = serde_json::from_str(&txt).unwrap();
    let tols = [1e-8f64, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];

    for case in root["cases"].as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let (h, w, d_e) = mat(&case["d_e"]);
        let (_, _, bmap) = mat(&case["bmap"]);
        let frame = Frame { d_e: &d_e, bmap: &bmap, h, w,
            sigma: case["sigma"].as_f64().unwrap(),
            k_max: case["k_max"].as_u64().unwrap() as usize };
        let (_, _, pos) = mat(&case["split"]["positions"]);
        let em = Emitters::from_parts(pos, vec_(&case["split"]["amplitudes"]));
        let n = em.len();
        let mut s = Solver::new();
        let mut cost: Vec<PatchCost> = Vec::new();

        // Reference: one sweep, converged far past anything downstream reads.
        let mut refr = em.clone();
        let mut se_ref = vec![f64::NAN; 3 * n];
        passes::refine_sweep_costed(&mut s, &frame, &em, 400, &mut refr,
                                    &mut se_ref, None, &mut cost, 1e-12, None, None, 0.0, None, false);
        let it_ref: usize = cost.iter().map(|c| c.n_iter).sum();

        println!("\n=== seed {seed}: N={n}, {h}x{w} (reference tol_obj = 1e-12, \
                  {it_ref} LM iterations) ===");
        println!("{:>9} {:>8} {:>7} {:>11} {:>11} {:>11} {:>11}",
                 "tol_obj", "iters", "vs ref", "sqrt(2*tol)", "med d/SE", "p95 d/SE", "max d/SE");
        for &t in &tols {
            let mut out = em.clone();
            let mut se = vec![f64::NAN; 3 * n];
            passes::refine_sweep_costed(&mut s, &frame, &em, 400, &mut out,
                                        &mut se, None, &mut cost, t, None, None, 0.0, None, false);
            let its: usize = cost.iter().map(|c| c.n_iter).sum();
            // Position shift from the reference, in units of the reported SE.
            let mut rel: Vec<f64> = Vec::with_capacity(n);
            for i in 0..n {
                let (sy, sx) = (se_ref[3 * i + 1], se_ref[3 * i + 2]);
                if !sy.is_finite() || !sx.is_finite() || sy <= 0.0 || sx <= 0.0 { continue; }
                let dy = (out.pos[2 * i] - refr.pos[2 * i]) / sy;
                let dx = (out.pos[2 * i + 1] - refr.pos[2 * i + 1]) / sx;
                rel.push(dy.hypot(dx));
            }
            let (med, p95, mx) = (quant(&mut rel, 0.5), quant(&mut rel, 0.95), quant(&mut rel, 1.0));
            println!("{t:>9.0e} {its:>8} {:>6.2}x {:>11.4} {med:>11.4} {p95:>11.4} {mx:>11.4}",
                     its as f64 / it_ref as f64, (2.0 * t).sqrt());
        }
    }
}
