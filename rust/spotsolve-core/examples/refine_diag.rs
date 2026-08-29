//! Diagnostic: how far REFINE lands from the Python, per emitter, starting
//! from the fixture's own post-SPLIT state.

use spotsolve_core::passes::{self, Emitters, Frame, Solver};

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
    for case in root["cases"].as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap();
        let (h, w, d_e) = mat(&case["d_e"]);
        let (_, _, bmap) = mat(&case["bmap"]);
        let frame = Frame { d_e: &d_e, bmap: &bmap, h, w,
            sigma: case["sigma"].as_f64().unwrap(),
            k_max: case["k_max"].as_u64().unwrap() as usize };

        let (_, _, pos) = mat(&case["split"]["positions"]);
        let mut em = Emitters::from_parts(pos, vec_(&case["split"]["amplitudes"]));
        let mut s = Solver::new();
        passes::refine(&mut s, &frame, &mut em, 200, 1, passes::REFINE_TOL);

        let (nw, _, wpos) = mat(&case["refine"]["positions"]);
        let wamp = vec_(&case["refine"]["amplitudes"]);
        let mut errs: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..em.len() {
            let mut best = (f64::INFINITY, usize::MAX);
            for j in 0..nw {
                let d = (em.y(i) - wpos[2 * j]).hypot(em.x(i) - wpos[2 * j + 1]);
                if d < best.0 { best = (d, j); }
            }
            errs.push((best.0, i, best.1));
        }
        errs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("\nseed {seed}: N={}  worst position errors", em.len());
        println!("{:>10} {:>9} {:>9} {:>10} {:>10} {:>8}", "err px", "y", "x", "A rust", "A python", "dA/A");
        for &(d, i, j) in errs.iter().take(5) {
            println!("{d:>10.3e} {:>9.4} {:>9.4} {:>10.1} {:>10.1} {:>8.1e}",
                     em.y(i), em.x(i), em.amp[i], wamp[j],
                     (em.amp[i] - wamp[j]).abs() / wamp[j]);
        }
        let over = errs.iter().filter(|e| e.0 > 1e-6).count();
        println!("emitters over 1e-6 px: {over} of {}", em.len());
        // Nearest-neighbour distance of the worst offender: crowding is the
        // expected amplifier.
        let (_, iw, _) = errs[0];
        let mut nn = f64::INFINITY;
        for k in 0..em.len() {
            if k != iw { nn = nn.min((em.y(iw) - em.y(k)).hypot(em.x(iw) - em.x(k))); }
        }
        println!("worst offender's nearest neighbour: {nn:.4} px ({:.2} sigma)",
                 nn / case["sigma"].as_f64().unwrap());
    }
}
