//! PyO3 bindings for `spotsolve-core`, exposed to Python as `spotsolve_rs`.
//!
//! The boundary sits at the FRAME: `box_localize` and `box_localize_stack`
//! (see `boxsearch`) run a whole frame natively with the GIL released.
//! `lmcl_fit_var_sigma` exposes one fit, for the fitter's own tests.
//!
//! Arrays must be C-contiguous f64; `as_slice()` fails loudly otherwise rather
//! than silently transposing.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spotsolve_core::lmcl;

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

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn spotsolve_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    boxsearch::register(m)?;
    m.add_function(wrap_pyfunction!(lmcl_fit_var_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
