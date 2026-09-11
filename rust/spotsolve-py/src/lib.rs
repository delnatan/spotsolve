//! PyO3 bindings for `spotsolve-core`, exposed to Python as `spotsolve_rs`.
//!
//! The boundary sits at the FRAME: `box_localize` and `box_localize_stack`
//! (see `boxsearch`) and `localize_sparse` each run a whole frame natively
//! with the GIL released. `lmcl_fit_var_sigma` exposes one fit, for the
//! Python reference (`box.py`) and for the fitter's own tests.
//!
//! Arrays must be C-contiguous f64; `as_slice()` fails loudly otherwise rather
//! than silently transposing.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spotsolve_core::{lmcl, sparse};

mod boxsearch;

type Arr1 = Py<PyArray1<f64>>;
type Arr2 = Py<PyArray2<f64>>;

fn slice2<'a>(a: &'a PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "`{name}` must be a C-contiguous float64 array; pass \
             np.ascontiguousarray(x, dtype=float)"
        ))
    })
}

fn slice1<'a>(a: &'a PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("`{name}` must be a contiguous float64 array")))
}

/// One bounded maximum-likelihood variable-sigma fit. Returns
/// `(theta, I, F, n_iter, converged, stalled)`.
///
/// `theta0` has length `4K+1`: `[b, A0, y0, x0, sigma0, ...]`. `halo` is the
/// parameter-free local contribution.
#[pyfunction]
#[pyo3(signature = (theta0, h, w, d, halo, lower, upper, max_iter=180, *, tol_obj=1e-8))]
#[allow(clippy::too_many_arguments)]
fn lmcl_fit_var_sigma(
    py: Python<'_>,
    theta0: PyReadonlyArray1<'_, f64>,
    h: usize,
    w: usize,
    d: PyReadonlyArray2<'_, f64>,
    halo: PyReadonlyArray2<'_, f64>,
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
    max_iter: usize,
    tol_obj: f64,
) -> PyResult<(Arr1, f64, Arr2, usize, bool, bool)> {
    let t = slice1(&theta0, "theta0")?;
    if t.len() % 4 != 1 {
        return Err(PyValueError::new_err("`theta0` must have length 4K+1"));
    }
    let data = slice2(&d, "d")?;
    let halo = slice2(&halo, "halo")?;
    let lo = slice1(&lower, "lower")?;
    let hi = slice1(&upper, "upper")?;
    if lo.len() != t.len()
        || hi.len() != t.len()
        || t.iter().any(|v| !v.is_finite())
        || lo
            .iter()
            .zip(hi)
            .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l >= u)
        || lo.iter().skip(4).step_by(4).any(|&s| s <= 0.0)
    {
        return Err(PyValueError::new_err(
            "expected finite theta and matching, ordered bounds with positive sigma limits",
        ));
    }
    if !tol_obj.is_finite() || tol_obj <= 0.0 {
        return Err(PyValueError::new_err("`tol_obj` must be positive and finite"));
    }
    if h == 0 || w == 0 || d.shape() != [h, w] || data.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("`d` does not match (h, w)"));
    }
    if halo.len() != h * w || halo.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("`halo` does not match (h, w)"));
    }
    let bounds = lmcl::Bounds::new(lo, hi);
    let p = t.len();
    let mut ws = lmcl::FitWorkspace::new();
    let info = lmcl::fit_var_sigma(
        &mut ws,
        t,
        h,
        w,
        data,
        &bounds,
        Some(halo),
        lmcl::FitOpts {
            max_iter,
            tol_obj,
            ..Default::default()
        },
    );
    Ok((
        ws.theta().to_vec().into_pyarray(py).unbind(),
        info.i_div,
        ws.fisher(p).to_vec().into_pyarray(py).reshape([p, p])?.unbind(),
        info.n_iter,
        info.converged,
        info.stalled,
    ))
}

/// Aguet significance pass followed by independent one-emitter fits.
///
/// This deliberately assumes sparse emitters. It performs no add/split/prune
/// loop and no joint fitting of overlapping candidates.
#[pyfunction]
#[pyo3(signature = (data, sigma, alpha=0.05, fit_sigma=false, sigma_bounds=(0.7, 2.2), fit_radius_sigma=4.0, max_iter=100))]
#[allow(clippy::too_many_arguments)]
fn localize_sparse<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    sigma: f64,
    alpha: f64,
    fit_sigma: bool,
    sigma_bounds: (f64, f64),
    fit_radius_sigma: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let shape = [data.shape()[0], data.shape()[1]];
    let values = slice2(&data, "data")?;
    let width = if fit_sigma {
        sparse::Width::Fitted {
            lower_ratio: sigma_bounds.0,
            upper_ratio: sigma_bounds.1,
        }
    } else {
        sparse::Width::Fixed
    };
    let mut workspace = sparse::Workspace::new();
    let result = py
        .detach(|| {
            sparse::localize(
                values,
                shape,
                sparse::Options {
                    sigma,
                    alpha,
                    width,
                    fit_radius_sigma,
                    max_iter,
                },
                &mut workspace,
            )
        })
        .map_err(PyValueError::new_err)?;
    let n = result.localizations.len();
    let positions = result
        .localizations
        .iter()
        .flat_map(|fit| [fit.y, fit.x])
        .collect::<Vec<_>>();
    let standard_errors = result
        .localizations
        .iter()
        .flat_map(|fit| [fit.se_flux, fit.se_y, fit.se_x])
        .collect::<Vec<_>>();
    let output = pyo3::types::PyDict::new(py);
    output.set_item("positions", positions.into_pyarray(py).reshape([n, 2])?)?;
    output.set_item(
        "amplitudes",
        result
            .localizations
            .iter()
            .map(|fit| fit.flux)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    output.set_item(
        "fit_sigma",
        result
            .localizations
            .iter()
            .map(|fit| fit.sigma)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    output.set_item("se", standard_errors.into_pyarray(py).reshape([n, 3])?)?;
    output.set_item(
        "test_statistic",
        result
            .localizations
            .iter()
            .map(|fit| fit.test_statistic)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    output.set_item(
        "p_value",
        result
            .localizations
            .iter()
            .map(|fit| fit.p_value)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    output.set_item(
        "iterations",
        result
            .localizations
            .iter()
            .map(|fit| fit.iterations)
            .collect::<Vec<_>>(),
    )?;
    output.set_item(
        "status",
        result
            .localizations
            .iter()
            .map(|fit| fit.status)
            .collect::<Vec<_>>(),
    )?;
    output.set_item("background", result.background)?;
    output.set_item("candidate_count", result.candidate_count)?;
    output.set_item("model_image", result.model.into_pyarray(py).reshape(shape)?)?;
    output.set_item("residual", result.residual.into_pyarray(py).reshape(shape)?)?;
    Ok(output)
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn spotsolve_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    boxsearch::register(m)?;
    m.add_function(wrap_pyfunction!(lmcl_fit_var_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(localize_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
