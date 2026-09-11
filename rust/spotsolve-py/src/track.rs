//! Bindings for the native linker, `spotsolve_core::track`.
//!
//! The boundary sits at the MOVIE: `track_fit` estimates every parameter from
//! a whole table of localizations and `track_link` links it, both with the
//! GIL released. Python only moves columns in and a track id out; the sort by
//! frame, the validation and all the arithmetic happen here.
//!
//! Positions and errors are in pixels and time in frames, so `d_grid` is
//! px^2/frame and `lam_birth` is per px^2 per frame. The scores are log
//! likelihood ratios and therefore invariant to that choice of units.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spotsolve_core::track as tk;
use spotsolve_core::trackparams as tp;

fn inputs<'a>(
    frame: &'a PyReadonlyArray1<'_, i64>,
    pos: &'a PyReadonlyArray2<'_, f64>,
    se: &'a PyReadonlyArray2<'_, f64>,
) -> PyResult<(&'a [i64], &'a [f64], &'a [f64])> {
    let err = |n: &str| {
        PyValueError::new_err(format!(
            "`{n}` must be a C-contiguous array of the right dtype; pass \
             np.ascontiguousarray(x)"
        ))
    };
    if pos.shape().get(1) != Some(&2) || se.shape().get(1) != Some(&2) {
        return Err(PyValueError::new_err(
            "`positions` and `errors` must have shape (N, 2), as (y, x)",
        ));
    }
    Ok((
        frame.as_slice().map_err(|_| err("frame"))?,
        pos.as_slice().map_err(|_| err("positions"))?,
        se.as_slice().map_err(|_| err("errors"))?,
    ))
}

fn detections(f: &[i64], p: &[f64], s: &[f64]) -> PyResult<tk::Detections> {
    tk::Detections::new(f, p, s).map_err(PyValueError::new_err)
}

fn params(
    d_grid: Vec<f64>,
    d_logprior: Vec<f64>,
    p_cont: f64,
    lam_birth: f64,
    se_inflate: f64,
) -> PyResult<tk::Params> {
    let p = tk::Params {
        d_grid,
        d_logprior,
        p_cont,
        lam_birth,
        se_inflate,
    };
    p.check().map_err(PyValueError::new_err)?;
    Ok(p)
}

/// Estimate every linking parameter from the localizations themselves.
///
/// Returns `d_grid`, `d_logprior`, `p_cont`, `lam_birth`, `se_inflate` and
/// `trajectory`, one entry per iteration of the empirical-Bayes loop.
#[pyfunction]
#[pyo3(signature = (frame, positions, errors))]
fn track_fit<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray1<'_, i64>,
    positions: PyReadonlyArray2<'_, f64>,
    errors: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (f, p, s) = inputs(&frame, &positions, &errors)?;
    let d = detections(f, p, s)?;
    let (fitted, traj) = py.detach(|| tp::fit(&d));

    let out = PyDict::new(py);
    out.set_item("d_grid", fitted.d_grid.into_pyarray(py))?;
    out.set_item("d_logprior", fitted.d_logprior.into_pyarray(py))?;
    out.set_item("p_cont", fitted.p_cont)?;
    out.set_item("lam_birth", fitted.lam_birth)?;
    out.set_item("se_inflate", fitted.se_inflate)?;
    let steps = PyList::empty(py);
    for s in traj {
        let e = PyDict::new(py);
        e.set_item("label", s.label)?;
        e.set_item("p_cont", s.p_cont)?;
        e.set_item("lam_birth", s.lam_birth)?;
        e.set_item("se_inflate", s.se_inflate)?;
        e.set_item("d_mean", s.d_mean)?;
        e.set_item("d_immobile", s.d_immobile)?;
        e.set_item("n_tracks", s.n_tracks)?;
        steps.append(e)?;
    }
    out.set_item("trajectory", steps)?;
    Ok(out)
}

/// Link one movie. Returns a track id per input row, in input order.
#[pyfunction]
#[pyo3(signature = (frame, positions, errors, d_grid, d_logprior, p_cont, lam_birth, se_inflate))]
#[allow(clippy::too_many_arguments)]
fn track_link(
    py: Python<'_>,
    frame: PyReadonlyArray1<'_, i64>,
    positions: PyReadonlyArray2<'_, f64>,
    errors: PyReadonlyArray2<'_, f64>,
    d_grid: PyReadonlyArray1<'_, f64>,
    d_logprior: PyReadonlyArray1<'_, f64>,
    p_cont: f64,
    lam_birth: f64,
    se_inflate: f64,
) -> PyResult<Py<PyArray1<u32>>> {
    let (f, p, s) = inputs(&frame, &positions, &errors)?;
    let d = detections(f, p, s)?;
    let grid = d_grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("`d_grid` must be contiguous float64"))?;
    let prior = d_logprior
        .as_slice()
        .map_err(|_| PyValueError::new_err("`d_logprior` must be contiguous float64"))?;
    let pp = params(grid.to_vec(), prior.to_vec(), p_cont, lam_birth, se_inflate)?;

    let ids = py.detach(|| {
        let l = tk::link(&d, &pp);
        let mut out = vec![0u32; d.n_dets()];
        for (r, &id) in l.track.iter().enumerate() {
            out[d.order[r]] = id;
        }
        out
    });
    Ok(ids.into_pyarray(py).unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(track_fit, m)?)?;
    m.add_function(wrap_pyfunction!(track_link, m)?)?;
    m.add("TRACK_GATE_ALPHA", tk::GATE_ALPHA)?;
    m.add("TRACK_D_GRID_N", tk::D_GRID_N)?;
    m.add("TRACK_EM_ITERS", tp::EM_ITERS)?;
    Ok(())
}
