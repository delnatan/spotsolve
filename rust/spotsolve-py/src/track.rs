//! Bindings for the native linker, `spotsolve_core::track`.
//!
//! The boundary sits at the MOVIE: `track_link` takes every row's frame,
//! position and localization error and returns a track id per row, in input
//! order, and optionally each row's link probability, with the GIL released.
//! `track_search_radius` reads the search radius from the movie.

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use spotsolve_core::track;

type Rows = (Vec<i64>, Vec<f64>, Vec<f64>);

fn rows(
    frame: &PyReadonlyArray1<'_, i64>,
    positions: &PyReadonlyArray2<'_, f64>,
    errors: &PyReadonlyArray2<'_, f64>,
) -> PyResult<Rows> {
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
    Ok((f, p, e))
}

/// Link `(y, x)` positions frame to frame, with `(se_y, se_x)` errors,
/// within a search `radius` in pixels (`None`: read from the movie, as
/// `track_search_radius` does), and optionally each row's detection
/// `strength`, by which the linker reads its continuation per class, and
/// the `switch_cost`, the log odds a link must reach.
/// Returns a track id per row, in input order, and with `link_p` also the
/// probability of the link into each row (`NaN` where a track starts).
#[pyfunction]
#[pyo3(signature = (frame, positions, errors, radius=None, strength=None, link_p=false, switch_cost=track::SWITCH_COST))]
fn track_link(
    py: Python<'_>,
    frame: PyReadonlyArray1<'_, i64>,
    positions: PyReadonlyArray2<'_, f64>,
    errors: PyReadonlyArray2<'_, f64>,
    radius: Option<f64>,
    strength: Option<PyReadonlyArray1<'_, f64>>,
    link_p: bool,
    switch_cost: f64,
) -> PyResult<Py<PyAny>> {
    let (f, p, e) = rows(&frame, &positions, &errors)?;
    let z = strength
        .map(|z| z.as_slice().map(|v| v.to_vec()))
        .transpose()
        .map_err(|_| PyValueError::new_err("`strength` must be a C-contiguous array"))?;
    let l = py
        .detach(|| {
            let r = match radius {
                Some(r) => r,
                None => track::search_radius(&f, &p, &e)?,
            };
            track::link_with(&f, &p, &e, z.as_deref(), r, switch_cost)
        })
        .map_err(PyValueError::new_err)?;
    let track = l.track.into_pyarray(py);
    if link_p {
        Ok((track, l.link_p.into_pyarray(py)).into_pyobject(py)?.into_any().unbind())
    } else {
        Ok(track.into_any().unbind())
    }
}

/// The search radius the linker reads from the movie, without links: from
/// next-frame against own-frame pair distances, the fastest diffusion
/// coefficient the movie holds, with a margin.
#[pyfunction]
fn track_search_radius(
    py: Python<'_>,
    frame: PyReadonlyArray1<'_, i64>,
    positions: PyReadonlyArray2<'_, f64>,
    errors: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let (f, p, e) = rows(&frame, &positions, &errors)?;
    py.detach(|| track::search_radius(&f, &p, &e)).map_err(PyValueError::new_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(track_link, m)?)?;
    m.add_function(wrap_pyfunction!(track_search_radius, m)?)?;
    m.add("TRACK_SWITCH_COST", track::SWITCH_COST)?;
    Ok(())
}
