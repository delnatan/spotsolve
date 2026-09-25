//! Bindings for the native detector, `spotsolve_core::detect`.
//!
//! Inputs are copied before releasing the GIL. Outputs contain every selected
//! emitter, diagnostic flags, a background surface and frame metadata.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spotsolve_core::detect;

type Arr1 = Py<PyArray1<f64>>;
type Arr2 = Py<PyArray2<f64>>;
/// `(positions, amplitudes, sigmas, se, sigma_se, flags, background, info)`.
type Frame<'py> = (Arr2, Arr1, Arr1, Arr2, Arr1, Py<PyArray1<u8>>, Arr2, Bound<'py, PyDict>);

fn settings(sigma: f64, fp_per_mpx: f64, slack: (f64, f64)) -> PyResult<detect::Settings> {
    if !(sigma.is_finite() && sigma > 0.0) {
        return Err(PyValueError::new_err("`sigma` must be positive"));
    }
    if !(slack.0 > 0.0 && slack.0 < slack.1 && slack.1.is_finite()) {
        return Err(PyValueError::new_err("`slack` must be 0 < lo < hi"));
    }
    if !(fp_per_mpx.is_finite() && fp_per_mpx > 0.0) {
        return Err(PyValueError::new_err("`fp_per_mpx` must be positive and finite"));
    }
    Ok(detect::Settings { sigma, fp_per_mpx, slack })
}

fn check_offset(offset: f64) -> PyResult<()> {
    if !offset.is_finite() {
        return Err(PyValueError::new_err("`offset` must be finite"));
    }
    Ok(())
}

fn roi_slice<'a>(
    roi: &'a Option<PyReadonlyArray2<'_, bool>>,
    h: usize,
    w: usize,
) -> PyResult<Option<&'a [bool]>> {
    match roi {
        None => Ok(None),
        Some(m) => {
            if m.shape() != [h, w] {
                return Err(PyValueError::new_err("`roi` does not match the frame"));
            }
            Ok(Some(m.as_slice().map_err(|_| {
                PyValueError::new_err("`roi` must be a C-contiguous bool array")
            })?))
        }
    }
}

fn give<'py>(py: Python<'py>, o: detect::Output, h: usize, w: usize) -> PyResult<Frame<'py>> {
    let n = o.amp.len();
    let info = PyDict::new(py);
    info.set_item("fitted_background", o.fitted_background.into_pyarray(py))?;
    info.set_item("dispersion", o.dispersion)?;
    info.set_item("u", o.u)?;
    info.set_item("seeds", o.n_seeds)?;
    info.set_item("fits", o.fits)?;
    info.set_item("lr_fail", o.lr_fail)?;
    info.set_item("adds", o.adds)?;
    info.set_item("removed", o.removed)?;
    info.set_item("outer", o.outer)?;
    info.set_item("kappa", o.kappa)?;
    info.set_item("fisher_fraction", o.fisher_fraction.into_pyarray(py).reshape([n, 4])?)?;
    Ok((
        o.pos.into_pyarray(py).reshape([n, 2])?.unbind(),
        o.amp.into_pyarray(py).unbind(),
        o.sig.into_pyarray(py).unbind(),
        o.se.into_pyarray(py).reshape([n, 3])?.unbind(),
        o.se_sig.into_pyarray(py).unbind(),
        o.flags.into_pyarray(py).unbind(),
        o.background.into_pyarray(py).reshape([h, w])?.unbind(),
        info,
    ))
}

/// Localize one raw frame. Everything is in ADU above `offset`; the
/// background and dispersion are measured from the frame.
#[pyfunction]
#[pyo3(signature = (raw, sigma, offset=0.0, *, roi=None, fp_per_mpx=detect::FP_PER_MPX, slack=detect::SLACK))]
#[allow(clippy::too_many_arguments)]
fn detect_localize<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray2<'_, f64>,
    sigma: f64,
    offset: f64,
    roi: Option<PyReadonlyArray2<'_, bool>>,
    fp_per_mpx: f64,
    slack: (f64, f64),
) -> PyResult<Frame<'py>> {
    let (h, w) = (raw.shape()[0], raw.shape()[1]);
    let r = raw
        .as_slice()
        .map_err(|_| PyValueError::new_err("`raw` must be a C-contiguous float64 array"))?
        .to_vec();
    if h == 0 || w == 0 || r.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("`raw` must be a finite, non-empty image"));
    }
    check_offset(offset)?;
    let roi = roi_slice(&roi, h, w)?.map(|m| m.to_vec());
    let s = settings(sigma, fp_per_mpx, slack)?;
    let o = py.detach(|| detect::localize_raw(&r, h, w, offset, roi.as_deref(), &s));
    give(py, o, h, w)
}

/// Localize every frame of a raw `(T, H, W)` stack on `n_threads` workers,
/// each frame exactly as `detect_localize` would. Returns one tuple per frame,
/// in frame order.
#[pyfunction]
#[pyo3(signature = (raw, sigma, offset=0.0, *, roi=None, fp_per_mpx=detect::FP_PER_MPX, slack=detect::SLACK, n_threads=1))]
#[allow(clippy::too_many_arguments)]
fn detect_localize_stack<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray3<'_, f64>,
    sigma: f64,
    offset: f64,
    roi: Option<PyReadonlyArray2<'_, bool>>,
    fp_per_mpx: f64,
    slack: (f64, f64),
    n_threads: usize,
) -> PyResult<Vec<Frame<'py>>> {
    let sh = raw.shape();
    let (n, h, w) = (sh[0], sh[1], sh[2]);
    let r = raw
        .as_slice()
        .map_err(|_| PyValueError::new_err("`raw` must be a C-contiguous float64 array"))?;
    if h == 0 || w == 0 || r.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("`raw` must be finite and non-empty"));
    }
    check_offset(offset)?;
    let roi = roi_slice(&roi, h, w)?.map(|m| m.to_vec());
    let s = settings(sigma, fp_per_mpx, slack)?;
    let r = r.to_vec();
    let outs = py.detach(|| {
        detect::localize_stack(&r, n, h, w, offset, roi.as_deref(), &s, n_threads.max(1))
    });
    outs.into_iter().map(|o| give(py, o, h, w)).collect()
}

/// `background` plus every emitter at its own width, truncated at
/// `truncate` of its sigma.
#[pyfunction]
#[pyo3(signature = (positions, amplitudes, sigmas, background, truncate=4.0))]
fn detect_render(
    py: Python<'_>,
    positions: PyReadonlyArray2<'_, f64>,
    amplitudes: PyReadonlyArray1<'_, f64>,
    sigmas: PyReadonlyArray1<'_, f64>,
    background: PyReadonlyArray2<'_, f64>,
    truncate: f64,
) -> PyResult<Arr2> {
    let (h, w) = (background.shape()[0], background.shape()[1]);
    let err = |name: &str| PyValueError::new_err(format!("`{name}` must be contiguous float64"));
    let pos = positions.as_slice().map_err(|_| err("positions"))?;
    let amp = amplitudes.as_slice().map_err(|_| err("amplitudes"))?;
    let sig = sigmas.as_slice().map_err(|_| err("sigmas"))?;
    let bg = background.as_slice().map_err(|_| err("background"))?;
    if pos.len() != 2 * amp.len() || sig.len() != amp.len() {
        return Err(PyValueError::new_err("positions, amplitudes and sigmas disagree"));
    }
    Ok(spotsolve_core::render::render_model(pos, amp, sig, h, w, bg, truncate)
        .into_pyarray(py)
        .reshape([h, w])?
        .unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("DETECT_OUTPUT_VERSION", 5)?;
    m.add_function(wrap_pyfunction!(detect_localize, m)?)?;
    m.add_function(wrap_pyfunction!(detect_localize_stack, m)?)?;
    m.add_function(wrap_pyfunction!(detect_render, m)?)?;
    // The detector's defaults, read by `spotsolve.native` so Python states
    // no second copy of them.
    m.add("DETECT_SLACK", detect::SLACK)?;
    m.add("DETECT_FP_PER_MPX", detect::FP_PER_MPX)?;
    Ok(())
}
