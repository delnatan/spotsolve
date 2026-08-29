//! Diagnostic: the SPLIT ranking and outcome on fixture case 0, starting from
//! the fixture's own post-ADD state so ADD cannot contribute any difference.

use spotsolve_core::evidence::Prior;
use spotsolve_core::moves;
use spotsolve_core::passes::{self, Emitters, Frame, Solver};
use spotsolve_core::patches::BBOX_PAD;

fn mat(v: &serde_json::Value) -> (usize, usize, Vec<f64>) {
    let rows = v.as_array().unwrap();
    let nc = rows[0].as_array().unwrap().len();
    let data = rows
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
        .collect();
    (rows.len(), nc, data)
}

fn main() {
    let txt = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/06_passes.json"),
    )
    .unwrap();
    let root: serde_json::Value = serde_json::from_str(&txt).unwrap();
    let case = &root["cases"][0];
    let (h, w, d_e) = mat(&case["d_e"]);
    let (_, _, bmap) = mat(&case["bmap"]);
    let sigma = case["sigma"].as_f64().unwrap();
    let (_, _, model) = mat(&case["split"]["model"]);
    let (n0, _, pos) = mat(&case["add"]["positions"]);
    let amp: Vec<f64> =
        case["add"]["amplitudes"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();

    let frame = Frame { d_e: &d_e, bmap: &bmap, h, w, sigma, k_max: 12 };
    let prior =
        Prior { lam: case["lam"].as_f64().unwrap(), a_s: case["A_s"].as_f64().unwrap() };
    let em0 = Emitters::from_parts(pos, amp);

    // The ranking, computed exactly as split_pass does.
    let pad = (BBOX_PAD * sigma).ceil() as i64;
    let mut st: Vec<(f64, usize)> = Vec::new();
    let mut win = Vec::new();
    for i in 0..n0 {
        let (cy, cx) = (em0.y(i), em0.x(i));
        let y0 = ((cy as i64) - pad).max(0) as usize;
        let x0 = ((cx as i64) - pad).max(0) as usize;
        let y1 = ((((cy as i64) + pad + 1).max(0)) as usize).min(h);
        let x1 = ((((cx as i64) + pad + 1).max(0)) as usize).min(w);
        win.clear();
        for r in y0..y1 {
            win.extend((x0..x1).map(|c| d_e[r * w + c] - model[r * w + c]));
        }
        let (_, s) = moves::residual_axis(
            cy, cx, em0.amp[i], y0 as f64, x0 as f64, y1 - y0, x1 - x0, sigma, &win,
        );
        st.push((s, i));
    }
    let mut order = st.clone();
    order.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    println!("n0 = {n0}");
    print!("top 8 by strength: [");
    for (s, i) in order.iter().take(8) {
        print!("({i}, {s:.5}), ");
    }
    println!("]");
    println!("n with strength>0: {}", st.iter().filter(|(s, _)| *s > 0.0).count());

    let mut em = em0.clone();
    let mut solver = Solver::new();
    let n_split = passes::split_pass(&mut solver, &frame, &mut em, &model, prior);
    println!("n_split = {n_split} (fixture: {})", case["split"]["n_split"]);

    // Which emitters ended up somewhere the fixture does not have.
    let (_, _, wpos) = mat(&case["split"]["positions"]);
    let nw = wpos.len() / 2;
    let mut bad = 0;
    for i in 0..em.len() {
        let mut best = f64::INFINITY;
        for j in 0..nw {
            best = best.min((em.y(i) - wpos[2 * j]).hypot(em.x(i) - wpos[2 * j + 1]));
        }
        if best > 1e-6 {
            if bad < 8 {
                println!("  unmatched: got ({:.4}, {:.4}) A={:.1} nearest {:.4} px",
                         em.y(i), em.x(i), em.amp[i], best);
            }
            bad += 1;
        }
    }
    println!("unmatched: {bad} of {}", em.len());

    // Both configurations in the neighbourhood the difference sits in.
    println!("\n-- emitters within 3 px of (32.0, 30.4) --");
    println!("{:>8} {:>9} {:>9} {:>10}", "which", "y", "x", "A");
    let mut mine: Vec<(f64, f64, f64)> = (0..em.len())
        .map(|i| (em.y(i), em.x(i), em.amp[i]))
        .filter(|(y, x, _)| (y - 32.0).hypot(x - 30.4) < 3.0)
        .collect();
    mine.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (y, x, a) in &mine {
        println!("{:>8} {y:>9.4} {x:>9.4} {a:>10.1}", "rust");
    }
    let wamp: Vec<f64> = case["split"]["amplitudes"].as_array().unwrap().iter()
        .map(|v| v.as_f64().unwrap()).collect();
    let mut theirs: Vec<(f64, f64, f64)> = (0..nw)
        .map(|j| (wpos[2 * j], wpos[2 * j + 1], wamp[j]))
        .filter(|(y, x, _)| (y - 32.0).hypot(x - 30.4) < 3.0)
        .collect();
    theirs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (y, x, a) in &theirs {
        println!("{:>8} {y:>9.4} {x:>9.4} {a:>10.1}", "python");
    }
    println!("total flux  rust {:.1}   python {:.1}",
             mine.iter().map(|t| t.2).sum::<f64>(),
             theirs.iter().map(|t| t.2).sum::<f64>());
}
