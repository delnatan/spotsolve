//! Loader for the golden fixtures in `tests/fixtures/*.json`. They are
//! FROZEN: the Python reference and `scripts/make_fixtures.py` that wrote
//! them were retired on 2026-09-11 (both last present in commit `ea6b17f`).
//! Every float there is emitted at 17 significant digits, so it round-trips
//! f64 exactly and the early layers can be compared bit for bit
//! (PORTING_NOTES.md section 16).
//!
//! Port bottom up and assert at each layer. Each fixture's own `compare` field
//! states how exactly that layer can be reproduced; the layers genuinely
//! differ, and pretending otherwise wastes days.

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
    /// The fixture's own statement of how exactly this layer reproduces.
    /// Printed on failure so the tolerance and its reasoning stay together.
    pub fn compare(&self) -> &str {
        self.root["compare"].as_str().unwrap_or("")
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

pub fn bool_at(v: &Value, key: &str) -> bool {
    v[key]
        .as_bool()
        .unwrap_or_else(|| panic!("`{key}` is not a bool"))
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

/// JSON has no NaN or Infinity, so the fixture encoder wrote them as the bare
/// strings Python's `json` produces. Accept both.
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

// ----------------------------------------------------------------- asserts

#[track_caller]
pub fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let scale = want.abs().max(1.0);
    let err = (got - want).abs() / scale;
    assert!(
        err <= tol,
        "{what}: got {got:.17e}, want {want:.17e}, rel err {err:.3e} > {tol:.3e}"
    );
}

#[track_caller]
pub fn assert_abs(got: f64, want: f64, tol: f64, what: &str) {
    let err = (got - want).abs();
    assert!(
        err <= tol,
        "{what}: got {got:.17e}, want {want:.17e}, abs err {err:.3e} > {tol:.3e}"
    );
}

/// Elementwise relative comparison, reporting the worst entry rather than the
/// first -- when a whole Jacobian is wrong, the worst one localizes the bug.
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
