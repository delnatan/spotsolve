//! What `localize` achieves against what the data allow, on simulated
//! Poisson frames with known emitters:
//!
//! - `iso`: isolated emitters by oracle SNR and width -- recall, position
//!   error over the Cramer-Rao bound, width bias, false emitters, time;
//! - `pairs`: equal pairs by separation -- how many detections each yields;
//! - `fields`: random fields by density -- recall by nearest-neighbour
//!   distance and oracle SNR, precision, time;
//! - `noise`: false emitters per 10^6 pixels of pure noise, against the
//!   `fp_per_mpx` promise.
//!
//! The oracle SNR is the flux over its standard error for a matched filter at
//! the emitter's true width, with a free local level: the best any detector
//! can do for that emitter alone.
//!
//! ```sh
//! cargo run --release --example characterize -- [iso] [pairs] [fields] [noise]
//! ```

use spotsolve_core::detect::{self as bs, Settings};
use spotsolve_core::psf;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn uni(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uni().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uni()).cos()
    }
    fn poisson(&mut self, lam: f64) -> f64 {
        if lam > 500.0 {
            return (lam + lam.sqrt() * self.normal()).round().max(0.0);
        }
        let l = (-lam).exp();
        let (mut k, mut p) = (0.0, 1.0);
        loop {
            p *= self.uni();
            if p <= l {
                return k;
            }
            k += 1.0;
        }
    }
}

#[derive(Clone, Copy)]
struct T {
    a: f64,
    y: f64,
    x: f64,
    s: f64,
}

fn render(h: usize, w: usize, bg: f64, ts: &[T]) -> Vec<f64> {
    let mut theta = vec![bg];
    for t in ts {
        theta.extend_from_slice(&[t.a, t.y, t.x, t.s]);
    }
    let mut m = vec![0.0; h * w];
    psf::model_var_sigma_ax(&theta, &psf::local_axis(h), &psf::local_axis(w), None, &mut psf::Factors::new(h, w, ts.len().max(1)), &mut m);
    m
}

/// Profile of a unit-flux emitter at (y, x, s) on a (2R+1)^2 box around it:
/// values and derivatives (y, x, s) per pixel.
fn profile(y: f64, x: f64, s: f64, r: isize) -> Vec<[f64; 4]> {
    let cy = y.round() as isize;
    let cx = x.round() as isize;
    let ay: Vec<f64> = (cy - r..=cy + r).map(|v| v as f64).collect();
    let ax: Vec<f64> = (cx - r..=cx + r).map(|v| v as f64).collect();
    let n = ay.len();
    let (mut ey, mut dy, mut sy) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let (mut ex, mut dx, mut sx) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    psf::factors_axis_sigma(&ay, &[y], s, &mut ey, &mut dy, &mut sy);
    psf::factors_axis_sigma(&ax, &[x], s, &mut ex, &mut dx, &mut sx);
    let mut out = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            out.push([ey[i] * ex[j], dy[i] * ex[j], ey[i] * dx[j], sy[i] * ex[j] + ey[i] * sx[j]]);
        }
    }
    out
}

/// Oracle z of an isolated emitter: flux over its SE with the level free and
/// everything else known -- the best a matched filter at the true width does.
fn oracle_z(a: f64, s: f64, bg: f64) -> f64 {
    let p = profile(0.0, 0.0, s, (5.0 * s).ceil() as isize);
    let (mut gg, mut g1, mut n1) = (0.0, 0.0, 0.0);
    for v in &p {
        gg += v[0] * v[0] / bg;
        g1 += v[0] / bg;
        n1 += 1.0 / bg;
    }
    a * (gg - g1 * g1 / n1).sqrt()
}

/// CRLB of (A, y, x, s) with a free level, isolated emitter; returns sd of y.
fn crlb_pos(a: f64, s: f64, bg: f64) -> (f64, f64) {
    let p = profile(0.3, 0.1, s, (5.0 * s).ceil() as isize);
    let mut f = [[0.0; 5]; 5];
    for v in &p {
        let m = bg + a * v[0];
        let j = [1.0, v[0], a * v[1], a * v[2], a * v[3]];
        for q in 0..5 {
            for r in 0..5 {
                f[q][r] += j[q] * j[r] / m;
            }
        }
    }
    let inv = inv5(f);
    (inv[2][2].sqrt(), inv[4][4].sqrt())
}

fn inv5(mut a: [[f64; 5]; 5]) -> [[f64; 5]; 5] {
    let mut b = [[0.0; 5]; 5];
    for i in 0..5 {
        b[i][i] = 1.0;
    }
    for c in 0..5 {
        let p = (c..5).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, p);
        b.swap(c, p);
        let d = a[c][c];
        for k in 0..5 {
            a[c][k] /= d;
            b[c][k] /= d;
        }
        for r in 0..5 {
            if r != c {
                let f = a[r][c];
                for k in 0..5 {
                    a[r][k] -= f * a[c][k];
                    b[r][k] -= f * b[c][k];
                }
            }
        }
    }
    b
}

/// One-to-one greedy match within `tol` px: per truth, Some(detection).
fn matching(o: &bs::Output, truth: &[T], tol: f64) -> Vec<Option<usize>> {
    let n = o.amp.len();
    let mut pairs = Vec::new();
    for (t, p) in truth.iter().enumerate() {
        for k in 0..n {
            let d = (o.pos[2 * k] - p.y).hypot(o.pos[2 * k + 1] - p.x);
            if d < tol {
                pairs.push((d, t, k));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut ut = vec![None; truth.len()];
    let mut uk = vec![false; n];
    for (_, t, k) in pairs {
        if ut[t].is_none() && !uk[k] {
            ut[t] = Some(k);
            uk[k] = true;
        }
    }
    ut
}

fn settings(sigma: f64) -> Settings {
    Settings { sigma, fp_per_mpx: bs::FP_PER_MPX, slack: bs::SLACK }
}

fn isolated(rng: &mut Rng, sref: f64) {
    println!("\n== isolated emitters, bg 20, sigma_ref {sref}: recall | pos rmse/CRLB | width bias/CRLB | FP per frame | ms");
    println!("{:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>6}", "s/sref", "z_or", "recall", "pos/crlb", "sbias/cr", "FP", "ms");
    let (h, w, sp, bg) = (256usize, 256usize, 16.0, 20.0);
    for ratio in [0.8, 1.0, 1.3, 1.6, 2.0] {
        let s = sref * ratio;
        for z in [4.0, 5.0, 6.0, 8.0, 12.0, 20.0] {
            let a = z / oracle_z(1.0, s, bg);
            let (mut tp, mut nt, mut fp, mut se_pos, mut se_s, mut n_m, mut ms) = (0, 0, 0, 0.0, 0.0, 0, 0.0);
            let (cr_pos, cr_s) = crlb_pos(a, s, bg);
            for _ in 0..3 {
                let mut ts = Vec::new();
                let mut yy = 12.0;
                while yy < h as f64 - 12.0 {
                    let mut xx = 12.0;
                    while xx < w as f64 - 12.0 {
                        ts.push(T { a, y: yy + rng.uni() - 0.5, x: xx + rng.uni() - 0.5, s });
                        xx += sp;
                    }
                    yy += sp;
                }
                let m = render(h, w, bg, &ts);
                let d: Vec<f64> = m.iter().map(|&v| rng.poisson(v)).collect();
                let t0 = Instant::now();
                let o = bs::localize(&d, h, w, None, &settings(sref));
                ms += t0.elapsed().as_secs_f64() * 1e3 / 3.0;
                let mt = matching(&o, &ts, 1.5 * s.max(1.0));
                nt += ts.len();
                for (t, k) in mt.iter().enumerate() {
                    if let Some(k) = *k {
                        tp += 1;
                        let (dy, dx) = (o.pos[2 * k] - ts[t].y, o.pos[2 * k + 1] - ts[t].x);
                        se_pos += 0.5 * (dy * dy + dx * dx);
                        se_s += o.sig[k] - s;
                        n_m += 1;
                    }
                }
                fp += o.amp.len() - mt.iter().filter(|k| k.is_some()).count();
            }
            println!(
                "{:>6.2} {:>6.1} {:>8.3} {:>8.2} {:>8.2} {:>8.1} {:>6.1}",
                ratio,
                z,
                tp as f64 / nt as f64,
                (se_pos / n_m.max(1) as f64).sqrt() / cr_pos,
                se_s / n_m.max(1) as f64 / cr_s,
                fp as f64 / 3.0,
                ms
            );
        }
    }
}

fn pairs(rng: &mut Rng, sref: f64) {
    println!("\n== equal pairs, width sref, oracle z 15 each: fraction of pairs yielding 0/1/2/3+ detections; recall");
    let (h, w, sp, bg) = (240usize, 240usize, 24.0, 20.0);
    let a = 15.0 / oracle_z(1.0, sref, bg);
    for sep in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0] {
        let mut cnt = [0usize; 4];
        let (mut tp, mut nt) = (0, 0);
        for _ in 0..3 {
            let mut ts = Vec::new();
            let mut centres = Vec::new();
            let mut yy = 12.0;
            while yy < h as f64 - 12.0 {
                let mut xx = 12.0;
                while xx < w as f64 - 12.0 {
                    let th = rng.uni() * std::f64::consts::PI;
                    let (cy, cx) = (yy + rng.uni() - 0.5, xx + rng.uni() - 0.5);
                    let (dy, dx) = (0.5 * sep * sref * th.sin(), 0.5 * sep * sref * th.cos());
                    ts.push(T { a, y: cy + dy, x: cx + dx, s: sref });
                    ts.push(T { a, y: cy - dy, x: cx - dx, s: sref });
                    centres.push((cy, cx));
                    xx += sp;
                }
                yy += sp;
            }
            let d: Vec<f64> = render(h, w, bg, &ts).iter().map(|&v| rng.poisson(v)).collect();
            let o = bs::localize(&d, h, w, None, &settings(sref));
            for &(cy, cx) in &centres {
                let k = (0..o.amp.len()).filter(|&k| (o.pos[2 * k] - cy).hypot(o.pos[2 * k + 1] - cx) < 6.0).count();
                cnt[k.min(3)] += 1;
            }
            let mt = matching(&o, &ts, 0.5 * sep * sref + 0.5);
            tp += mt.iter().filter(|k| k.is_some()).count();
            nt += ts.len();
        }
        let tot: usize = cnt.iter().sum();
        println!(
            "sep {:>4.2} sref: 0:{:.2} 1:{:.2} 2:{:.2} 3+:{:.2}  recall(within sep/2+0.5) {:.3}",
            sep,
            cnt[0] as f64 / tot as f64,
            cnt[1] as f64 / tot as f64,
            cnt[2] as f64 / tot as f64,
            cnt[3] as f64 / tot as f64,
            tp as f64 / nt as f64
        );
    }
}

fn fields(rng: &mut Rng, sref: f64, spread: (f64, f64), label: &str) {
    println!("\n== random fields 256x256, bg 20, flux 150-3000, width sref*U{spread:?} ({label})");
    let (h, w, bg) = (256usize, 256usize, 20.0);
    for dens in [0.005, 0.015, 0.03] {
        let n = (dens * (h * w) as f64).round() as usize;
        let ts: Vec<T> = (0..n)
            .map(|_| T {
                a: 150.0 + 2850.0 * rng.uni(),
                y: 4.0 + rng.uni() * (h as f64 - 9.0),
                x: 4.0 + rng.uni() * (w as f64 - 9.0),
                s: sref * (spread.0 + (spread.1 - spread.0) * rng.uni()),
            })
            .collect();
        let d: Vec<f64> = render(h, w, bg, &ts).iter().map(|&v| rng.poisson(v)).collect();
        let t0 = Instant::now();
        let o = bs::localize(&d, h, w, None, &settings(sref));
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let mt = matching(&o, &ts, 1.0);
        let tp = mt.iter().filter(|k| k.is_some()).count();
        // Misses by oracle z and nearest-neighbour distance (in sref).
        let nn: Vec<f64> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i).map(|j| (ts[i].y - ts[j].y).hypot(ts[i].x - ts[j].x)).fold(f64::INFINITY, f64::min) / sref)
            .collect();
        let mut tab = [[0usize; 2]; 4];
        for i in 0..n {
            let b = if nn[i] < 1.5 { 0 } else if nn[i] < 2.5 { 1 } else if nn[i] < 4.0 { 2 } else { 3 };
            tab[b][0] += 1;
            tab[b][1] += mt[i].is_some() as usize;
        }
        let zo: Vec<f64> = ts.iter().map(|t| oracle_z(t.a, t.s, bg)).collect();
        let mut zt = [[0usize; 2]; 4];
        for i in 0..n {
            if nn[i] < 4.0 {
                continue;
            }
            let b = if zo[i] < 6.0 { 0 } else if zo[i] < 8.0 { 1 } else if zo[i] < 12.0 { 2 } else { 3 };
            zt[b][0] += 1;
            zt[b][1] += mt[i].is_some() as usize;
        }
        println!(
            "dens {dens}: N {} of {n}, recall {:.3} precision {:.3}, {ms:.0} ms ({:.2} us/px), seeds {} fits {} outer {} adds {} removed {} lr_fail {} kappa {:.2}",
            o.amp.len(),
            tp as f64 / n as f64,
            tp as f64 / o.amp.len().max(1) as f64,
            ms * 1e3 / (h * w) as f64,
            o.n_seeds,
            o.fits,
            o.outer,
            o.adds,
            o.removed,
            o.lr_fail,
            o.kappa
        );
        let f = |t: [usize; 2]| format!("{}/{}", t[1], t[0]);
        println!("   recall by NN dist (sref) <1.5: {}  1.5-2.5: {}  2.5-4: {}  >4: {}", f(tab[0]), f(tab[1]), f(tab[2]), f(tab[3]));
        println!("   isolated (NN>4) recall by oracle z <6: {}  6-8: {}  8-12: {}  >12: {}", f(zt[0]), f(zt[1]), f(zt[2]), f(zt[3]));
    }
}

fn noise(rng: &mut Rng) {
    println!("\n== pure noise, bg 20, 8 frames 256x256");
    for sref in [1.0, 1.2, 1.6, 2.0] {
        let (h, w) = (256usize, 256usize);
        let mut n = 0;
        let mut ms = 0.0;
        for _ in 0..8 {
            let d: Vec<f64> = (0..h * w).map(|_| rng.poisson(20.0)).collect();
            let t0 = Instant::now();
            n += bs::localize(&d, h, w, None, &settings(sref)).amp.len();
            ms += t0.elapsed().as_secs_f64() * 1e3 / 8.0;
        }
        println!("sref {sref}: {:.1} FP/Mpx (target {}), {ms:.1} ms/frame", n as f64 / (8.0 * (h * w) as f64) * 1e6, bs::FP_PER_MPX);
    }
}

fn main() {
    let which: Vec<String> = std::env::args().skip(1).collect();
    let on = |s: &str| which.is_empty() || which.iter().any(|w| w == s);
    let mut rng = Rng(2026);
    let sref = 1.2;
    if on("iso") {
        isolated(&mut rng, sref);
    }
    if on("pairs") {
        pairs(&mut rng, sref);
    }
    if on("fields") {
        fields(&mut rng, sref, (0.8, 1.2), "near focus");
        fields(&mut rng, sref, (0.8, 2.0), "defocus spread");
    }
    if on("noise") {
        noise(&mut rng);
    }
}
