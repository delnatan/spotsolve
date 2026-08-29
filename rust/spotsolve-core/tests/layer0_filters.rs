//! Layer 0: the separable filters and order statistics, against
//! `tests/fixtures/07_filters.json`.
//!
//! Below every other layer. These are pure arithmetic on identical inputs, so
//! the tolerance is 1e-12 absolute -- a port that only matches to 1e-6 has a
//! convention wrong, not a rounding difference. The convention most likely to
//! be wrong is the order-2 kernel's normalization; see the `kernels1d` case.

mod common;

use spotsolve_core::filters::{
    gaussian_filter, gaussian_kernel1d, gaussian_laplace, kernel_radius, maximum_filter,
    uniform_filter, Mode,
};

const TOL: f64 = 1e-12;

/// Flatten a fixture value that may be a flat list or a list of rows. The
/// encoder writes 2-D arrays as nested rows, 1-D as a flat list.
fn f64s(v: &serde_json::Value) -> Vec<f64> {
    let a = v.as_array().expect("expected a JSON array");
    if a.first().map(|x| x.is_array()).unwrap_or(false) {
        a.iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
            .collect()
    } else {
        a.iter().map(|x| x.as_f64().unwrap()).collect()
    }
}

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length {} vs {}", got.len(), want.len());
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= TOL,
        "{what}: max |diff| {worst:.3e} at index {at} (got {}, want {}), tol {TOL:.0e}",
        got[at], want[at]
    );
}

#[test]
fn gaussian_kernels_match_scipy_including_the_derivative_normalization() {
    let fx = common::load("07_filters");
    for case in fx.root["kernels1d"].as_array().unwrap() {
        let sigma = case["sigma"].as_f64().unwrap();
        let order = case["order"].as_u64().unwrap() as usize;
        let radius = case["radius"].as_u64().unwrap() as usize;
        assert_eq!(
            kernel_radius(sigma),
            radius,
            "radius rule disagrees at sigma={sigma}: scipy uses int(4*sigma + 0.5)"
        );
        let want = f64s(&case["taps"]);
        let got = gaussian_kernel1d(sigma, order, radius);
        assert_close(&got, &want, &format!("kernel sigma={sigma} order={order}"));
    }
}

/// The trap this layer exists to catch: a truncated order-2 kernel does NOT
/// sum to zero, because scipy normalizes the order-0 kernel and only then
/// differentiates. "Fixing" that rescales the whole LoG response against a
/// fixed detection threshold.
#[test]
fn truncated_second_derivative_kernel_does_not_sum_to_zero() {
    let k = gaussian_kernel1d(0.6, 2, kernel_radius(0.6));
    let s: f64 = k.iter().sum();
    assert!(
        (s - -0.06497).abs() < 1e-4,
        "order-2 kernel at sigma=0.6 should sum to about -6.5e-2, got {s:.6e}; \
         a sum of 0 means the kernel was re-normalized after differentiation"
    );
}

#[test]
fn two_d_filters_match_the_fixture() {
    let fx = common::load("07_filters");
    let img = f64s(&fx.root["image"]["data"]);
    let h = fx.root["image"]["h"].as_u64().unwrap() as usize;
    let w = fx.root["image"]["w"].as_u64().unwrap() as usize;

    for case in fx.root["filters2d"].as_array().unwrap() {
        let op = case["op"].as_str().unwrap();
        let mode = match case["mode"].as_str().unwrap() {
            "reflect" => Mode::Reflect,
            "nearest" => Mode::Nearest,
            m => panic!("unknown mode {m}"),
        };
        let want = f64s(&case["result"]);
        let got = match op {
            "gaussian_laplace" => {
                gaussian_laplace(&img, h, w, case["sigma"].as_f64().unwrap(), mode)
            }
            "gaussian_filter" => {
                gaussian_filter(&img, h, w, case["sigma"].as_f64().unwrap(), mode)
            }
            "uniform_filter" => {
                uniform_filter(&img, h, w, case["size"].as_u64().unwrap() as usize, mode)
            }
            "maximum_filter" => {
                maximum_filter(&img, h, w, case["size"].as_u64().unwrap() as usize, mode)
            }
            o => panic!("unknown op {o}"),
        };
        assert_close(&got, &want, &format!("{op} mode={:?}", mode));
    }
}

/// A filter wider than the array. `background_map` reaches this on a small
/// frame, and it is where a naive index reflection goes wrong.
#[test]
fn filters_wider_than_the_array_still_match() {
    let fx = common::load("07_filters");
    let o = &fx.root["oversize"];
    let img = f64s(&o["data"]);
    let (h, w) = (o["h"].as_u64().unwrap() as usize, o["w"].as_u64().unwrap() as usize);
    assert_close(&uniform_filter(&img, h, w, 9, Mode::Nearest),
                 &f64s(&o["uniform_9_nearest"]), "uniform_9_nearest");
    assert_close(&maximum_filter(&img, h, w, 9, Mode::Reflect),
                 &f64s(&o["maximum_9_reflect"]), "maximum_9_reflect");
    assert_close(&gaussian_filter(&img, h, w, 3.0, Mode::Nearest),
                 &f64s(&o["gaussian_3_nearest"]), "gaussian_3_nearest");
}
