//! What do REFINE's last 150 LM iterations actually buy?
//!
//! A handful of groups run the full `max_iter` without converging and cost
//! 10-40% of a sweep. They are not stalled -- they are still taking accepted
//! steps, crawling along a direction the data carries no information about.
//! The emitters in them are NOT disposable: 25 of 28 survive to the final
//! output. So the question is not whether to abandon them but what an
//! iteration budget costs, and the honest unit for that is the same one
//! `refine_tol.rs` uses -- the shift in each emitter's position measured
//! against the standard error that same fit reports.
//!
//! Run: `cargo run --release --example refine_budget`.

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
    let budgets = [10usize, 25, 50, 100, 200, 400];

    for case in root["cases"].as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let (h, w, d_e) = mat(&case["d_e"]);
        let (_, _, bmap) = mat(&case["bmap"]);
        let sigma = case["sigma"].as_f64().unwrap();
        let frame = Frame { d_e: &d_e, bmap: &bmap, h, w, sigma,
            k_max: case["k_max"].as_u64().unwrap() as usize };
        let (_, _, pos) = mat(&case["split"]["positions"]);
        let em = Emitters::from_parts(pos, vec_(&case["split"]["amplitudes"]));
        let n = em.len();
        let mut s = Solver::new();
        let mut cost: Vec<PatchCost> = Vec::new();

        // The stuck groups do not exist on sweep 0 -- they appear as the
        // configuration drifts -- so the budget has to be judged over the whole
        // sweep sequence REFINE actually runs, not over one pass.
        const NS: usize = 8;
        let run = |s: &mut Solver, cost: &mut Vec<PatchCost>, budget: usize|
            -> (Emitters, Vec<f64>, usize, usize) {
            let mut cur = em.clone();
            let mut nxt = em.clone();
            let mut se = vec![f64::NAN; 3 * n];
            let (mut its, mut unconv) = (0usize, 0usize);
            for _ in 0..NS {
                passes::refine_sweep_costed(s, &frame, &cur, budget, &mut nxt, &mut se,
                                            None, cost, passes::REFINE_TOL_OBJ,
                                            None, None, 0.0, None, false);
                its += cost.iter().map(|c| c.n_iter).sum::<usize>();
                unconv += cost.iter().filter(|c| !c.converged).count();
                std::mem::swap(&mut cur, &mut nxt);
            }
            (cur, se, its, unconv)
        };
        let (refr, se_ref, it_ref, unconv_ref) = run(&mut s, &mut cost, 4000);
        println!("\n=== seed {seed}: N={n}, {h}x{w}, {NS} sweeps \
                  (reference budget 4000: {it_ref} iters, {unconv_ref} unconverged) ===");
        println!("{:>8} {:>8} {:>7} {:>6} {:>11} {:>11} {:>11}",
                 "max_iter", "iters", "vs ref", "!conv", "med d/SE", "p95 d/SE", "max d/SE");

        for &b in &budgets {
            let (out, _se, its, nconv) = run(&mut s, &mut cost, b);
            let mut rel: Vec<f64> = Vec::with_capacity(n);
            for i in 0..n {
                let (sy, sx) = (se_ref[3 * i + 1], se_ref[3 * i + 2]);
                if !sy.is_finite() || !sx.is_finite() || sy <= 0.0 || sx <= 0.0 { continue; }
                let dy = (out.pos[2 * i] - refr.pos[2 * i]) / sy;
                let dx = (out.pos[2 * i + 1] - refr.pos[2 * i + 1]) / sx;
                rel.push(dy.hypot(dx));
            }
            let (med, p95, mx) = (quant(&mut rel, 0.5), quant(&mut rel, 0.95), quant(&mut rel, 1.0));
            println!("{b:>8} {its:>8} {:>6.2}x {nconv:>6} {med:>11.2e} {p95:>11.2e} {mx:>11.4}",
                     its as f64 / it_ref as f64);
        }
    }
}
