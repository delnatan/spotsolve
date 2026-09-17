//! Golden fits from spotfitlm's original C implementation, including covariance.
mod common;
use spotsolve_core::aguet::{fit_patch, FitWorkspace};

#[test]
fn fits_match_spotfitlm_parameters_objectives_and_observed_covariance() {
    let fixture = common::load("09_aguet");
    let mut ws = FitWorkspace::default();
    for (index, case) in fixture.root["fits"].as_array().unwrap().iter().enumerate() {
        let rows = case["data"].as_array().unwrap();
        let data: Vec<f64> = rows
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let result = fit_patch(
            &mut ws,
            &data,
            rows.len(),
            case["sigma0"].as_f64().unwrap(),
            case["itermax"].as_u64().unwrap() as usize,
        );
        assert_eq!(
            result.status as i64,
            case["status"].as_i64().unwrap(),
            "case {index}"
        );
        for (actual, expected) in result.theta.iter().zip(case["theta"].as_array().unwrap()) {
            let expected = expected.as_f64().unwrap();
            assert!(
                (actual - expected).abs() < 1e-6 * expected.abs().max(1.0),
                "case {index}: theta {actual} != {expected}"
            );
        }
        let expected = case["objective"].as_f64().unwrap();
        assert!(
            (result.objective - expected).abs() < 1e-7,
            "case {index}: objective {} != {expected}",
            result.objective
        );
        if let Some(cov) = case["covariance"].as_array() {
            for (actual, expected) in result
                .covariance
                .iter()
                .zip(cov.iter().flat_map(|row| row.as_array().unwrap()))
            {
                let expected = expected.as_f64().unwrap();
                assert!(
                    (actual - expected).abs() < 2e-6 * expected.abs().max(1e-4),
                    "case {index}: covariance {actual} != {expected}"
                );
            }
        }
    }
}
