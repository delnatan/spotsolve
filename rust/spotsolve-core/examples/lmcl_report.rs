//! How closely this optimizer reproduces the Python's converged fits, and how
//! fast it is. Not a test -- a measurement, so the numbers claimed in the port
//! notes stay checkable. Run: `cargo run --release --example lmcl_report`.

use spotsolve_core::lmcl::{self, Bounds, FitOpts, FitWorkspace};
use std::time::Instant;

fn main() {
    let txt = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/02_lmga.json"),
    )
    .unwrap();
    let root: serde_json::Value = serde_json::from_str(&txt).unwrap();

    println!(
        "{:>3} {:>6} {:>7} {:>16} {:>12} {:>10}",
        "K", "px", "n_iter", "I", "dI vs py", "us/fit"
    );
    let mut ws = FitWorkspace::new();
    for case in root["var_sigma_cases"].as_array().unwrap() {
        let k = case["K"].as_u64().unwrap() as usize;
        let (h, w) = (case["h"].as_u64().unwrap() as usize, case["w"].as_u64().unwrap() as usize);
        let d: Vec<f64> = case["data"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let g = |key: &str| -> Vec<f64> {
            case[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
        };
        let (theta0, lo, hi) = (g("theta0"), g("lower"), g("upper"));
        let bounds = Bounds::new(&lo, &hi);
        let want_i = case["ml"]["I"].as_f64().unwrap();

        let info = lmcl::fit_var_sigma(&mut ws, &theta0, h, w, &d, &bounds, None, FitOpts::default());

        let reps = 2000;
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(lmcl::fit_var_sigma(
                &mut ws, &theta0, h, w, &d, &bounds, None, FitOpts::default(),
            ));
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

        println!(
            "{k:>3} {:>6} {:>7} {:>16.8} {:>12.2e} {us:>10.2}",
            h * w,
            info.n_iter,
            info.i_div,
            (info.i_div - want_i).abs()
        );
    }
}
