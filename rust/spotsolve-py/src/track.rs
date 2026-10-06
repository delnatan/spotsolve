//! Bindings for the native linker, `spotsolve_core::track`.
//!
//! The boundary sits at the MOVIE: `track_link` takes every row's frame,
//! position and localization error and returns a track id per row, in input
//! order, with the GIL released.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spotsolve_core::track;

/// Link `(y, x)` positions frame to frame, with `(se_y, se_x)` errors and
/// steps up to `max_step` pixels. Returns a track id per row, in input order.
#[pyfunction]
#[pyo3(signature = (frame, positions, errors, max_step))]
fn track_link(
    py: Python<'_>,
    frame: PyReadonlyArray1<'_, i64>,
    positions: PyReadonlyArray2<'_, f64>,
    errors: PyReadonlyArray2<'_, f64>,
    max_step: f64,
) -> PyResult<Py<PyArray1<u32>>> {
    if positions.shape().get(1) != Some(&2) {
        return Err(PyValueError::new_err("`positions` must have shape (N, 2), as (y, x)"));
    }
    if errors.shape().get(1) != Some(&2) {
        return Err(PyValueError::new_err("`errors` must have shape (N, 2), as (se_y, se_x)"));
    }
    let err = |n: &str| PyValueError::new_err(format!("`{n}` must be a C-contiguous array"));
    let f = frame.as_slice().map_err(|_| err("frame"))?.to_vec();
    let p = positions.as_slice().map_err(|_| err("positions"))?.to_vec();
    let e = errors.as_slice().map_err(|_| err("errors"))?.to_vec();
    let l = py
        .detach(|| track::link(&f, &p, &e, max_step))
        .map_err(PyValueError::new_err)?;
    Ok(l.track.into_pyarray(py).unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(track_link, m)?)?;
    Ok(())
}
