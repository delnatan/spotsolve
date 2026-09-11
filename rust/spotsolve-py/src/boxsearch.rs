//! Bindings for the native box search, `spotsolve_core::boxsearch`.
//!
//! Inputs are copied once, the GIL is released for the whole search, and the
//! answer comes back as owned arrays: every fitted emitter with its class
//! (0 focus, 1 narrow, 2 wide, 3 edge), the background surface, and an
//! `info` dict of the gain used and the work counts. Python only arranges
//! these into a `DetectResult`.

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
/// `(positions, amplitudes, sigmas, se, classes, background, info)`.
type Frame<'py> = (Arr2, Arr1, Arr1, Arr2, Py<PyArray1<u8>>, Arr2, Bound<'py, PyDict>);

#[allow(clippy::too_many_arguments)]
fn settings(
    h: usize,
    w: usize,
    sigma: f64,
    k_max: usize,
    threshold: Option<f64>,
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
    let threshold = threshold.unwrap_or_else(|| bs::seed_threshold(h, w, sigma, bs::SEED_ALPHA));
    if !threshold.is_finite() {
        return Err(PyValueError::new_err("`threshold` must be finite"));
    }
    Ok(bs::Settings {
        sigma,
        k_max,
        threshold,
        slack,
        sweeps,
        polish,
        band,
    })
}

fn check_gain(gain: Option<f64>, offset: f64, read_noise: f64) -> PyResult<()> {
    if gain.is_some_and(|g| !(g.is_finite() && g > 0.0)) {
        return Err(PyValueError::new_err("`gain` must be positive"));
    }
    if !offset.is_finite() || !read_noise.is_finite() || read_noise < 0.0 {
        return Err(PyValueError::new_err("`offset` and `read_noise` must be finite"));
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
    info.set_item("gain", o.gain)?;
    info.set_item("candidates", o.n_candidates)?;
    info.set_item("boxes", o.n_boxes)?;
    info.set_item("search_fits", o.search_fits)?;
    info.set_item("polish_fits", o.polish_fits)?;
    let class: Vec<u8> = o.class.iter().map(|&c| c as u8).collect();
    Ok((
        o.pos.into_pyarray(py).reshape([n, 2])?.unbind(),
        o.amp.into_pyarray(py).unbind(),
        o.sig.into_pyarray(py).unbind(),
        o.se.into_pyarray(py).reshape([n, 3])?.unbind(),
        class.into_pyarray(py).unbind(),
        o.background.into_pyarray(py).reshape([h, w])?.unbind(),
        info,
    ))
}

/// Localize one raw frame. `gain=None` estimates it from the frame; the
/// background is returned in photoelectrons, `read_noise^2` included.
#[pyfunction]
#[pyo3(signature = (raw, sigma, offset=0.0, gain=None, *, read_noise=0.0, roi=None, k_max=12, threshold=None, slack=(0.7, 2.2), band=Some((0.8, 2.0)), sweeps=2, polish=true))]
#[allow(clippy::too_many_arguments)]
fn box_localize<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray2<'_, f64>,
    sigma: f64,
    offset: f64,
    gain: Option<f64>,
    read_noise: f64,
    roi: Option<PyReadonlyArray2<'_, bool>>,
    k_max: usize,
    threshold: Option<f64>,
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
    check_gain(gain, offset, read_noise)?;
    let roi = roi_slice(&roi, h, w)?.map(|m| m.to_vec());
    let s = settings(h, w, sigma, k_max, threshold, slack, sweeps, polish, band)?;
    let o = py.detach(|| {
        let (mut ws, mut d) = (bs::Workspace::new(), Vec::new());
        let shift = read_noise * read_noise;
        bs::localize_raw(&r, h, w, offset, gain, shift, roi.as_deref(), &s, &mut ws, &mut d)
    });
    give(py, o, h, w)
}

/// Localize every frame of a raw `(T, H, W)` stack on `n_threads` workers,
/// each frame exactly as `box_localize` would. `gain=None` estimates one per
/// frame. Returns one tuple per frame, in frame order.
#[pyfunction]
#[pyo3(signature = (raw, sigma, offset=0.0, gain=None, *, read_noise=0.0, roi=None, k_max=12, threshold=None, slack=(0.7, 2.2), band=Some((0.8, 2.0)), sweeps=2, polish=true, n_threads=1))]
#[allow(clippy::too_many_arguments)]
fn box_localize_stack<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray3<'_, f64>,
    sigma: f64,
    offset: f64,
    gain: Option<f64>,
    read_noise: f64,
    roi: Option<PyReadonlyArray2<'_, bool>>,
    k_max: usize,
    threshold: Option<f64>,
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
    check_gain(gain, offset, read_noise)?;
    let roi = roi_slice(&roi, h, w)?.map(|m| m.to_vec());
    let s = settings(h, w, sigma, k_max, threshold, slack, sweeps, polish, band)?;
    let r = r.to_vec();
    let outs = py.detach(|| {
        let shift = read_noise * read_noise;
        bs::localize_stack(&r, n, h, w, offset, gain, shift, roi.as_deref(), &s, n_threads.max(1))
    });
    outs.into_iter().map(|o| give(py, o, h, w)).collect()
}

/// `background` plus every emitter at its own width, truncated at
/// `truncate` of its sigma. `calibrate.render_model` with per-emitter widths.
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
    // The constants `box.py` and `core.py` must agree with, so a drift shows
    // up as a test failure rather than as an algorithmic difference.
    m.add("BOX_ADD_NATS", bs::ADD_NATS)?;
    m.add("BOX_OWN_RADIUS", bs::OWN_RADIUS)?;
    m.add("BOX_SWEEPS", bs::SWEEPS)?;
    m.add("BOX_FIT_TOL_OBJ", bs::FIT_TOL_OBJ)?;
    m.add("BOX_FIT_MAX_ITER", bs::FIT_MAX_ITER)?;
    m.add("BOX_POLISH_SWEEPS", bs::POLISH_SWEEPS)?;
    m.add("BOX_POLISH_MAX_ITER", bs::POLISH_MAX_ITER)?;
    m.add("BOX_POLISH_TOL_OBJ", bs::POLISH_TOL_OBJ)?;
    m.add("BOX_POLISH_MOVE_TOL", bs::POLISH_MOVE_TOL)?;
    m.add("BOX_BG_KERNEL", bs::BG_KERNEL)?;
    m.add("BOX_BG_FLOOR", bs::BG_FLOOR)?;
    m.add("BOX_BG_MASK_RADIUS", bs::BG_MASK_RADIUS)?;
    m.add("BOX_BG_MIN_PIXELS", bs::BG_MIN_PIXELS)?;
    m.add("BOX_SEED_ALPHA", bs::SEED_ALPHA)?;
    m.add("BOX_A_MIN", bs::A_MIN)?;
    m.add("BOX_A_MIN_REL", bs::A_MIN_REL)?;
    m.add("BOX_EDGE_MARGIN", bs::EDGE_MARGIN)?;
    m.add("BOX_GAIN_FRAC", bs::GAIN_FRAC)?;
    Ok(())
}
