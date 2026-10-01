//! Shared by the layer tests: the golden fixtures in `tests/fixtures/*.json`
//! (frozen; floats at 17 significant digits, so they round-trip f64), and a
//! deterministic random source.

#![allow(dead_code)]

use serde_json::Value;
use std::path::PathBuf;

pub struct Fixture {
    pub root: Value,
}

pub fn load(name: &str) -> Fixture {
    let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "..", "tests", "fixtures"]
        .iter()
        .collect::<PathBuf>()
        .join(format!("{name}.json"));
    let txt = std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}. The fixtures are tracked in git; restore them.",
            p.display()
        )
    });
    Fixture {
        root: serde_json::from_str(&txt).expect("fixture is not valid JSON"),
    }
}

impl Fixture {
    pub fn cases(&self) -> &Vec<Value> {
        self.root["cases"]
            .as_array()
            .expect("fixture has no `cases` array")
    }
}

pub fn f64_at(v: &Value, key: &str) -> f64 {
    num(&v[key])
}

pub fn usize_at(v: &Value, key: &str) -> usize {
    v[key]
        .as_u64()
        .unwrap_or_else(|| panic!("`{key}` is not an integer")) as usize
}

pub fn vec_at(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap_or_else(|| panic!("`{key}` is not an array"))
        .iter()
        .map(num)
        .collect()
}

/// A 2-D fixture array, returned row-major as `(rows, cols, data)`.
pub fn mat_at(v: &Value, key: &str) -> (usize, usize, Vec<f64>) {
    let rows = v[key]
        .as_array()
        .unwrap_or_else(|| panic!("`{key}` is not an array"));
    let nr = rows.len();
    let nc = if nr == 0 {
        0
    } else {
        rows[0].as_array().expect("not 2-D").len()
    };
    let mut data = Vec::with_capacity(nr * nc);
    for r in rows {
        let row = r.as_array().expect("ragged fixture array");
        assert_eq!(row.len(), nc, "ragged fixture array in `{key}`");
        data.extend(row.iter().map(num));
    }
    (nr, nc, data)
}

/// A number, or NaN / Infinity written as strings (JSON has neither).
fn num(v: &Value) -> f64 {
    if let Some(x) = v.as_f64() {
        return x;
    }
    match v.as_str() {
        Some("NaN") => f64::NAN,
        Some("Infinity") => f64::INFINITY,
        Some("-Infinity") => f64::NEG_INFINITY,
        _ => panic!("not a number: {v}"),
    }
}

#[track_caller]
pub fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let scale = want.abs().max(1.0);
    let err = (got - want).abs() / scale;
    assert!(
        err <= tol,
        "{what}: got {got:.17e}, want {want:.17e}, rel err {err:.3e} > {tol:.3e}"
    );
}

/// Elementwise relative comparison, reporting the worst entry.
#[track_caller]
pub fn assert_all_rel(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} vs {}",
        got.len(),
        want.len()
    );
    let mut worst = (0usize, 0.0f64);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let e = (g - w).abs() / w.abs().max(1.0);
        if e > worst.1 {
            worst = (i, e);
        }
    }
    assert!(
        worst.1 <= tol,
        "{what}: worst at [{}] got {:.17e}, want {:.17e}, rel err {:.3e} > {:.3e}",
        worst.0,
        got[worst.0],
        want[worst.0],
        worst.1,
        tol
    );
}

/// An LCG: deterministic across platforms.
pub struct Rng(pub u64);

impl Rng {
    /// Uniform on (0, 1).
    pub fn uni(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Knuth's product method, for means up to a few hundred.
    pub fn poisson(&mut self, lam: f64) -> f64 {
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
