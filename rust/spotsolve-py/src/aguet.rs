//! Thin array boundary for the sparse detector; all frame work releases the GIL.
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use spotsolve_core::aguet;

type Frame<'py> = (
    Py<PyArray2<f64>>,
    Py<PyArray3<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Bound<'py, PyDict>,
);

fn give<'py>(py: Python<'py>, o: aguet::Output, h: usize, w: usize) -> PyResult<Frame<'py>> {
    let n = o.spots.len();
    let params: Vec<f64> = o.spots.iter().flat_map(|s| s.fit.theta).collect();
    let covariance: Vec<f64> = o.spots.iter().flat_map(|s| s.fit.covariance).collect();
    let seeds: Vec<f64> = o
        .spots
        .iter()
        .flat_map(|s| s.seed.map(|v| v as f64))
        .collect();
    let info = PyDict::new(py);
    info.set_item("candidates", o.candidates)?;
    info.set_item("fits", o.candidates)?;
    info.set_item("processed_pixels", o.processed_pixels)?;
    info.set_item("failures", o.failures)?;
    info.set_item(
        "iterations",
        o.spots.iter().map(|s| s.fit.iterations).collect::<Vec<_>>(),
    )?;
    info.set_item(
        "objective",
        o.spots.iter().map(|s| s.fit.objective).collect::<Vec<_>>(),
    )?;
    Ok((
        params.into_pyarray(py).reshape([n, 5])?.unbind(),
        covariance.into_pyarray(py).reshape([n, 5, 5])?.unbind(),
        seeds.into_pyarray(py).reshape([n, 2])?.unbind(),
        o.background.into_pyarray(py).reshape([h, w])?.unbind(),
        info,
    ))
}

#[pyfunction]
#[pyo3(signature = (raw, sigma, cutoff, *, boxsize=9, itermax=50, offset=0.0, roi=None, n_threads=1))]
#[allow(clippy::too_many_arguments)]
fn aguet_localize_stack<'py>(
    py: Python<'py>,
    raw: PyReadonlyArray3<'_, f64>,
    sigma: f64,
    cutoff: f64,
    boxsize: usize,
    itermax: usize,
    offset: f64,
    roi: Option<PyReadonlyArray2<'_, bool>>,
    n_threads: usize,
) -> PyResult<Vec<Frame<'py>>> {
    let (n, h, w) = (raw.shape()[0], raw.shape()[1], raw.shape()[2]);
    let data = raw
        .as_slice()
        .map_err(|_| PyValueError::new_err("expected contiguous float64 input"))?;
    if h == 0
        || w == 0
        || !sigma.is_finite()
        || sigma <= 0.0
        || sigma > h.max(w) as f64
        || !cutoff.is_finite()
        || !offset.is_finite()
        || boxsize < 3
        || boxsize % 2 == 0
        || itermax == 0
        || n_threads == 0
        || data
            .iter()
            .any(|v| !v.is_finite() || !(v - offset).is_finite())
    {
        return Err(PyValueError::new_err("invalid Aguet image or settings"));
    }
    let mask = match roi {
        Some(m) => {
            if m.shape() != [h, w] {
                return Err(PyValueError::new_err("roi does not match frame"));
            }
            Some(
                m.as_slice()
                    .map_err(|_| PyValueError::new_err("roi must be contiguous"))?
                    .to_vec(),
            )
        }
        None => None,
    };
    let data = data.to_vec();
    let settings = aguet::Settings::new(sigma, boxsize, itermax, cutoff);
    let output = py.detach(|| {
        if n == 1 {
            vec![aguet::localize(
                &data,
                h,
                w,
                offset,
                mask.as_deref(),
                &settings,
                &mut aguet::Workspace::default(),
            )]
        } else {
            aguet::localize_stack(
                &data,
                n,
                h,
                w,
                offset,
                mask.as_deref(),
                &settings,
                n_threads,
            )
        }
    });
    output.into_iter().map(|o| give(py, o, h, w)).collect()
}

/// Sampled-Gaussian diagnostic rendering, deliberately separate from box_render.
#[pyfunction]
fn aguet_render<'py>(
    py: Python<'py>,
    params: PyReadonlyArray2<'_, f64>,
    background: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    if params.shape()[1] != 5 {
        return Err(PyValueError::new_err("params must be (N, 5)"));
    }
    let theta = params.as_slice()?;
    let (h, w) = (background.shape()[0], background.shape()[1]);
    if h == 0 || w == 0 {
        return Err(PyValueError::new_err(
            "background dimensions must be nonzero",
        ));
    }
    if theta.iter().any(|v| !v.is_finite()) || theta.chunks_exact(5).any(|t| t[2] <= 0.0) {
        return Err(PyValueError::new_err("invalid fitted parameters"));
    }
    let mut model = background.as_slice()?.to_vec();
    for t in theta.chunks_exact(5) {
        let radius = 4.0 * t[2];
        let y0 = (t[1] - radius).floor().max(0.0) as usize;
        let y1 = ((t[1] + radius).ceil().max(0.0) as usize).min(h.saturating_sub(1));
        let x0 = (t[0] - radius).floor().max(0.0) as usize;
        let x1 = ((t[0] + radius).ceil().max(0.0) as usize).min(w.saturating_sub(1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                model[y * w + x] += t[3]
                    * (-((x as f64 - t[0]).powi(2) + (y as f64 - t[1]).powi(2))
                        / (2.0 * t[2] * t[2]))
                        .exp();
            }
        }
    }
    Ok(model.into_pyarray(py).reshape([h, w])?.unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aguet_localize_stack, m)?)?;
    m.add_function(wrap_pyfunction!(aguet_render, m)?)?;
    Ok(())
}
