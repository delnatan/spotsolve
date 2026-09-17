//! Bindings for the native box search, `spotsolve_core::boxsearch`.
//!
//! Inputs are copied once, the GIL is released for the whole search, and the
//! answer comes back as owned arrays: every fitted emitter with its class
//! (0 focus, 1 narrow, 2 wide, 3 edge), the background surface, and an
//! `info` dict of the frame's measured dispersion and the work counts.
//! Python only arranges these into a `Localizations`.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spotsolve_core::boxsearch as bs;

type Arr1 = Py<PyArray1<f64>>;
type Arr2 = Py<PyArray2<f64>>;
/// `(positions, amplitudes, sigmas, se, sigma_se, classes, background, info)`.
type Frame<'py> = (Arr2, Arr1, Arr1, Arr2, Arr1, Py<PyArray1<u8>>, Arr2, Bound<'py, PyDict>);

#[allow(clippy::too_many_arguments)]
fn settings(
    sigma: f64,
    k_max: usize,
    threshold: Option<f64>,
    selection: &str,
    count_penalty: f64,
    slack: (f64, f64),
    sweeps: usize,
    polish: bool,
    band: Option<(f64, f64)>,
) -> PyResult<bs::Settings> {
    if !(sigma.is_finite() && sigma > 0.0) {
        return Err(PyValueError::new_err("`sigma` must be positive"));
    }
    if !(slack.0 > 0.0 && slack.0 < slack.1 && slack.1.is_finite()) {
        return Err(PyValueError::new_err("`slack` must be 0 < lo < hi"));
    }
    if k_max == 0 {
        return Err(PyValueError::new_err("`k_max` must be at least 1"));
    }
    let threshold = threshold.unwrap_or(bs::PEAK_Z);
    if !threshold.is_finite() {
        return Err(PyValueError::new_err("`threshold` must be finite"));
    }
    let selection = match selection {
        "fixed" => bs::Selection::Fixed,
        "bic" => bs::Selection::Bic,
        _ => return Err(PyValueError::new_err("`selection` must be 'fixed' or 'bic'")),
    };
    if !count_penalty.is_finite() || count_penalty < 0.0 {
        return Err(PyValueError::new_err("`count_penalty` must be finite and non-negative"));
    }
    Ok(bs::Settings {
        sigma,
        k_max,
        threshold,
        selection,
        count_penalty,
        slack,
        sweeps,
        polish,
        band,
    })
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

fn give<'py>(py: Python<'py>, o: bs::Output, h: usize, w: usize) -> PyResult<Frame<'py>> {
    let n = o.amp.len();
    let info = PyDict::new(py);
    info.set_item("dispersion", o.dispersion)?;
    info.set_item("candidates", o.n_candidates)?;
    info.set_item("boxes", o.n_boxes)?;
    info.set_item("search_fits", o.search_fits)?;
    info.set_item("polish_fits", o.polish_fits)?;
    info.set_item("selection_fits", o.selection_fits)?;
    info.set_item("fisher_fraction", o.fisher_fraction.into_pyarray(py).reshape([n, 4])?)?;
    let class: Vec<u8> = o.class.iter().map(|&c| c as u8).collect();
    Ok((
        o.pos.into_pyarray(py).reshape([n, 2])?.unbind(),
        o.amp.into_pyarray(py).unbind(),
        o.sig.into_pyarray(py).unbind(),
        o.se.into_pyarray(py).reshape([n, 3])?.unbind(),
        o.se_sig.into_pyarray(py).unbind(),
        class.into_pyarray(py).unbind(),
        o.background.into_pyarray(py).reshape([h, w])?.unbind(),
        info,
    ))
}

/// Localize one raw frame. Everything is in ADU above `offset`; the noise is
/// measured from the frame.
#[pyfunction]
#[pyo3(signature = (raw, sigma, offset=0.0, *, roi=None, k_max=bs::K_MAX, threshold=None, selection="fixed", count_penalty=0.0, slack=bs::SLACK, band=Some(bs::BAND), sweeps=bs::SWEEPS, polish=true))]
#[allow(clippy::too_many_arguments)]
fn box_localize<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray2<'_, f64>,
    sigma: f64,
    offset: f64,
    roi: Option<PyReadonlyArray2<'_, bool>>,
    k_max: usize,
    threshold: Option<f64>,
    selection: &str,
    count_penalty: f64,
    slack: (f64, f64),
    band: Option<(f64, f64)>,
    sweeps: usize,
    polish: bool,
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
    let s = settings(sigma, k_max, threshold, selection, count_penalty, slack, sweeps, polish, band)?;
    let o = py.detach(|| {
        let (mut ws, mut d) = (bs::Workspace::new(), Vec::new());
        bs::localize_raw(&r, h, w, offset, roi.as_deref(), &s, &mut ws, &mut d)
    });
    give(py, o, h, w)
}

/// Localize every frame of a raw `(T, H, W)` stack on `n_threads` workers,
/// each frame exactly as `box_localize` would. Returns one tuple per frame,
/// in frame order.
#[pyfunction]
#[pyo3(signature = (raw, sigma, offset=0.0, *, roi=None, k_max=bs::K_MAX, threshold=None, selection="fixed", count_penalty=0.0, slack=bs::SLACK, band=Some(bs::BAND), sweeps=bs::SWEEPS, polish=true, n_threads=1))]
#[allow(clippy::too_many_arguments)]
fn box_localize_stack<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray3<'_, f64>,
    sigma: f64,
    offset: f64,
    roi: Option<PyReadonlyArray2<'_, bool>>,
    k_max: usize,
    threshold: Option<f64>,
    selection: &str,
    count_penalty: f64,
    slack: (f64, f64),
    band: Option<(f64, f64)>,
    sweeps: usize,
    polish: bool,
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
    let s = settings(sigma, k_max, threshold, selection, count_penalty, slack, sweeps, polish, band)?;
    let r = r.to_vec();
    let outs = py.detach(|| {
        bs::localize_stack(&r, n, h, w, offset, roi.as_deref(), &s, n_threads.max(1))
    });
    outs.into_iter().map(|o| give(py, o, h, w)).collect()
}

/// `background` plus every emitter at its own width, truncated at
/// `truncate` of its sigma.
#[pyfunction]
#[pyo3(signature = (positions, amplitudes, sigmas, background, truncate=4.0))]
fn box_render(
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
    m.add_function(wrap_pyfunction!(box_localize, m)?)?;
    m.add_function(wrap_pyfunction!(box_localize_stack, m)?)?;
    m.add_function(wrap_pyfunction!(box_render, m)?)?;
    // The detector's defaults, read by `spotsolve.native` so Python states
    // no second copy of them.
    m.add("BOX_SLACK", bs::SLACK)?;
    m.add("BOX_BAND", bs::BAND)?;
    m.add("BOX_K_MAX", bs::K_MAX)?;
    m.add("BOX_ADD_NATS", bs::ADD_NATS)?;
    m.add("BOX_PEAK_Z", bs::PEAK_Z)?;
    m.add("BOX_BAND_Z", bs::BAND_Z)?;
    Ok(())
}
