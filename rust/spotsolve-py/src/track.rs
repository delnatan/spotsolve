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

/// Track ids per input row, and the brightness noise `(tau2, q)` if used.
type Linked = (Py<PyArray1<u32>>, Option<(f64, f64)>);
type Scored = (
    Py<PyArray1<u32>>,
    Option<(f64, f64)>,
    Option<(Vec<Option<f64>>, Vec<bool>)>,
);

/// Link one movie. Returns a track id per input row, in input order, and
/// the brightness model's `(tau2, q)` when one was used.
///
/// `brightness` is `None` (positions only) or `(log_flux, log_flux_var)` per
/// row: brightness as a second cue, its noise estimated from the movie
/// (`trackparams::flux_model`).
#[pyfunction]
#[pyo3(signature = (frame, positions, errors, d_grid, d_logprior, p_cont, lam_birth, se_inflate, brightness=None))]
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
    brightness: Option<(Vec<f64>, Vec<f64>)>,
) -> PyResult<Linked> {
    let (ids, noise, _) = track_link_scored(
        py, frame, positions, errors, d_grid, d_logprior, p_cont, lam_birth, se_inflate,
        brightness, 0.0, false,
    )?;
    Ok((ids, noise))
}

/// Link with an optional exclusion-margin cutoff and incoming diagnostics.
#[pyfunction]
#[pyo3(signature = (frame, positions, errors, d_grid, d_logprior, p_cont, lam_birth, se_inflate, brightness=None, min_link_margin=0.0, diagnostics=false))]
#[allow(clippy::too_many_arguments)]
fn track_link_scored(
    py: Python<'_>,
    frame: PyReadonlyArray1<'_, i64>,
    positions: PyReadonlyArray2<'_, f64>,
    errors: PyReadonlyArray2<'_, f64>,
    d_grid: PyReadonlyArray1<'_, f64>,
    d_logprior: PyReadonlyArray1<'_, f64>,
    p_cont: f64,
    lam_birth: f64,
    se_inflate: f64,
    brightness: Option<(Vec<f64>, Vec<f64>)>,
    min_link_margin: f64,
    diagnostics: bool,
) -> PyResult<Scored> {
    if !min_link_margin.is_finite() || min_link_margin < 0.0 {
        return Err(PyValueError::new_err(
            "min_link_margin must be finite and non-negative",
        ));
    }
    let (f, p, s) = inputs(&frame, &positions, &errors)?;
    let d = detections(f, p, s)?;
    let grid = d_grid
        .as_slice()
        .map_err(|_| PyValueError::new_err("`d_grid` must be contiguous float64"))?;
    let prior = d_logprior
        .as_slice()
        .map_err(|_| PyValueError::new_err("`d_logprior` must be contiguous float64"))?;
    let pp = params(grid.to_vec(), prior.to_vec(), p_cont, lam_birth, se_inflate)?;

    if let Some((lf, var)) = &brightness {
        if lf.len() != d.n_dets() || var.len() != d.n_dets() {
            return Err(PyValueError::new_err(
                "brightness arrays must have one value per row",
            ));
        }
        if lf.iter().chain(var).any(|v| !v.is_finite()) || var.iter().any(|&v| v < 0.0) {
            return Err(PyValueError::new_err(
                "brightness must be finite, with non-negative variances",
            ));
        }
    }
    let (ids, noise, diag) = py.detach(|| {
        let fx = brightness.map(|(lf, var)| tp::flux_model(&d, &pp, lf, var));
        let (l, diag) = tk::link_scored(&d, &pp, fx.as_ref(), min_link_margin, diagnostics);
        let mut out = vec![0u32; d.n_dets()];
        for (r, &id) in l.track.iter().enumerate() {
            out[d.order[r]] = id;
        }
        let diag = diag.map(|diag| {
            let mut margin = vec![None; d.n_dets()];
            let mut rejected = vec![false; d.n_dets()];
            for (r, &original) in d.order.iter().enumerate() {
                margin[original] = diag.margin[r];
                rejected[original] = diag.rejected[r];
            }
            (margin, rejected)
        });
        (out, fx.map(|f| (f.tau2, f.q)), diag)
    });
    Ok((ids.into_pyarray(py).unbind(), noise, diag))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(track_fit, m)?)?;
    m.add_function(wrap_pyfunction!(track_link, m)?)?;
    m.add_function(wrap_pyfunction!(track_link_scored, m)?)?;
    m.add("TRACK_GATE_ALPHA", tk::GATE_ALPHA)?;
    m.add("TRACK_D_GRID_N", tk::D_GRID_N)?;
    m.add("TRACK_EM_ITERS", tp::EM_ITERS)?;
    Ok(())
}
