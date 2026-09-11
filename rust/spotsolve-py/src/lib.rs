//! PyO3 bindings for `spotsolve-core`, exposed to Python as `spotsolve_rs`.
//!
//! # Where the boundary sits
//!
//! At **pass** granularity, not per fit and not per frame. `core.py` keeps
//! `detect`'s round loop, `find_candidates` and `background_map`'s
//! convolutions, and calls in here ~35 times per frame. Every `scipy.ndimage`
//! call in the pipeline is per-round rather than per-fit and profiles at
//! 0.1-0.3%, so reimplementing scipy's kernel construction and its three
//! boundary modes would buy 0.3% for the most fidelity-fragile code in the
//! port. The one filtering-adjacent thing that does move is
//! [`emitter_free_mask`], which is `O(N*H*W)` as written in Python.
//!
//! # Zero copy
//!
//! `d_e` and `bmap` are read on every pass and are `(H, W)` f64. They arrive as
//! `PyReadonlyArray2` and are borrowed, never `.to_owned()`. At 512x512 that is
//! the difference between free and megabytes of copying per round.
//!
//! Arrays must be C-contiguous f64; `as_slice()` fails loudly otherwise rather
//! than silently transposing.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use spotsolve_core::evidence::{Evidence, Prior};
use spotsolve_core::passes::{self, Emitters, Frame, Solver};
use spotsolve_core::{linalg, lmcl, psf, render, sparse};

mod dense_group;
mod inference;

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

/// Take `(positions, amplitudes)` as the core's flat representation.
fn take_emitters(
    positions: &PyReadonlyArray2<'_, f64>,
    amplitudes: &PyReadonlyArray1<'_, f64>,
) -> PyResult<Emitters> {
    let pos = slice2(positions, "positions")?;
    let amp = slice1(amplitudes, "amplitudes")?;
    if positions.shape()[1] != 2 {
        return Err(PyValueError::new_err("`positions` must have shape (N, 2)"));
    }
    if positions.shape()[0] != amp.len() {
        return Err(PyValueError::new_err(format!(
            "`positions` has {} rows but `amplitudes` has {}",
            positions.shape()[0],
            amp.len()
        )));
    }
    Ok(Emitters::from_parts(pos.to_vec(), amp.to_vec()))
}

fn give_emitters(py: Python<'_>, em: Emitters) -> (Arr2, Arr1) {
    let n = em.len();
    let pos = em.pos.into_pyarray(py).reshape([n, 2]).unwrap().unbind();
    let amp = em.amp.into_pyarray(py).unbind();
    (pos, amp)
}

fn make_frame<'a>(
    d_e: &'a [f64],
    bmap: &'a [f64],
    shape: [usize; 2],
    sigma: f64,
    k_max: usize,
) -> PyResult<Frame<'a>> {
    let (h, w) = (shape[0], shape[1]);
    if bmap.len() != h * w {
        return Err(PyValueError::new_err(
            "`bmap` must have the same shape as `d_e`",
        ));
    }
    if !(sigma > 0.0) {
        return Err(PyValueError::new_err("`sigma` must be positive"));
    }
    Ok(Frame {
        d_e,
        bmap,
        h,
        w,
        sigma,
        k_max,
    })
}

// ------------------------------------------------------------------ passes

/// One ADD pass over a candidate list.
///
/// Returns `(positions, amplitudes, n_added)`. The candidate list comes from
/// `find_candidates` in Python; the in-loop proximity re-check lives here
/// because it reads the positions an earlier acceptance in *this* pass wrote.
#[pyfunction]
#[pyo3(signature = (d_e, bmap, positions, amplitudes, cand, cand_amp, sigma, lam, a_s, k_max=12))]
#[allow(clippy::too_many_arguments)]
fn add_pass(
    py: Python<'_>,
    d_e: PyReadonlyArray2<'_, f64>,
    bmap: PyReadonlyArray2<'_, f64>,
    positions: PyReadonlyArray2<'_, f64>,
    amplitudes: PyReadonlyArray1<'_, f64>,
    cand: PyReadonlyArray2<'_, f64>,
    cand_amp: PyReadonlyArray1<'_, f64>,
    sigma: f64,
    lam: f64,
    a_s: f64,
    k_max: usize,
) -> PyResult<(Arr2, Arr1, usize)> {
    let d = slice2(&d_e, "d_e")?;
    let b = slice2(&bmap, "bmap")?;
    let frame = make_frame(d, b, [d_e.shape()[0], d_e.shape()[1]], sigma, k_max)?;
    let mut em = take_emitters(&positions, &amplitudes)?;
    let c = slice2(&cand, "cand")?;
    let ca = slice1(&cand_amp, "cand_amp")?;
    if cand.shape()[1] != 2 || cand.shape()[0] != ca.len() {
        return Err(PyValueError::new_err(
            "`cand` must be (M, 2) matching `cand_amp`",
        ));
    }
    let mut s = Solver::new();
    let n = py.detach(|| passes::add_pass(&mut s, &frame, &mut em, c, ca, Prior { lam, a_s }));
    let (p, a) = give_emitters(py, em);
    Ok((p, a, n))
}

/// One SPLIT pass, most pair-like first.
///
/// `model` is the current full model image, rendered by the caller. The
/// ordering is not an optimization detail: a split accepted early changes its
/// neighbours, so it decides which configuration later proposals are scored
/// against.
#[pyfunction]
#[pyo3(signature = (d_e, bmap, positions, amplitudes, model, sigma, lam, a_s, k_max=12))]
#[allow(clippy::too_many_arguments)]
fn split_pass(
    py: Python<'_>,
    d_e: PyReadonlyArray2<'_, f64>,
    bmap: PyReadonlyArray2<'_, f64>,
    positions: PyReadonlyArray2<'_, f64>,
    amplitudes: PyReadonlyArray1<'_, f64>,
    model: PyReadonlyArray2<'_, f64>,
    sigma: f64,
    lam: f64,
    a_s: f64,
    k_max: usize,
) -> PyResult<(Arr2, Arr1, usize)> {
    let d = slice2(&d_e, "d_e")?;
    let b = slice2(&bmap, "bmap")?;
    let m = slice2(&model, "model")?;
    let frame = make_frame(d, b, [d_e.shape()[0], d_e.shape()[1]], sigma, k_max)?;
    if m.len() != frame.h * frame.w {
        return Err(PyValueError::new_err(
            "`model` must have the same shape as `d_e`",
        ));
    }
    let mut em = take_emitters(&positions, &amplitudes)?;
    let mut s = Solver::new();
    let n = py.detach(|| passes::split_pass(&mut s, &frame, &mut em, m, Prior { lam, a_s }));
    let (p, a) = give_emitters(py, em);
    Ok((p, a, n))
}

/// Joint re-fit at fixed N in connected groups, plus per-emitter CRLBs.
///
/// Returns `(positions, amplitudes, se)` with `se` an `(N, 3)` array of
/// `(SE_A, SE_y, SE_x)`, NaN where the Fisher matrix was singular. Proposes
/// nothing.
///
/// **Signature-compatible with `spotsolve.refine`**, because `crlb.py`'s oracle arm
/// calls it directly with the true positions at the true N -- and without that
/// arm a pull spread is uninterpretable.
#[pyfunction]
#[pyo3(signature = (d_e, positions, amplitudes, sigma, bmap, k_max=12, max_iter=passes::REFINE_MAX_ITER, max_sweeps=4, tol=passes::REFINE_TOL))]
#[allow(clippy::too_many_arguments)]
fn refine(
    py: Python<'_>,
    d_e: PyReadonlyArray2<'_, f64>,
    positions: PyReadonlyArray2<'_, f64>,
    amplitudes: PyReadonlyArray1<'_, f64>,
    sigma: f64,
    bmap: PyReadonlyArray2<'_, f64>,
    k_max: usize,
    max_iter: usize,
    max_sweeps: usize,
    tol: f64,
) -> PyResult<(Arr2, Arr1, Arr2)> {
    let d = slice2(&d_e, "d_e")?;
    let b = slice2(&bmap, "bmap")?;
    let frame = make_frame(d, b, [d_e.shape()[0], d_e.shape()[1]], sigma, k_max)?;
    let mut em = take_emitters(&positions, &amplitudes)?;
    let se = py.detach(|| {
        passes::refine(
            &mut Solver::new(),
            &frame,
            &mut em,
            max_iter,
            max_sweeps,
            tol,
        )
    });
    let n = em.len();
    let (p, a) = give_emitters(py, em);
    let se = se.into_pyarray(py).reshape([n, 3]).unwrap().unbind();
    Ok((p, a, se))
}

/// One PRUNE pass, faintest first, with write-back onto survivors.
///
/// Returns `(positions, amplitudes, n_removed)`.
#[pyfunction]
#[pyo3(signature = (d_e, bmap, positions, amplitudes, sigma, lam, a_s, k_max=12))]
#[allow(clippy::too_many_arguments)]
fn prune(
    py: Python<'_>,
    d_e: PyReadonlyArray2<'_, f64>,
    bmap: PyReadonlyArray2<'_, f64>,
    positions: PyReadonlyArray2<'_, f64>,
    amplitudes: PyReadonlyArray1<'_, f64>,
    sigma: f64,
    lam: f64,
    a_s: f64,
    k_max: usize,
) -> PyResult<(Arr2, Arr1, usize)> {
    let d = slice2(&d_e, "d_e")?;
    let b = slice2(&bmap, "bmap")?;
    let frame = make_frame(d, b, [d_e.shape()[0], d_e.shape()[1]], sigma, k_max)?;
    let mut em = take_emitters(&positions, &amplitudes)?;
    let mut s = Solver::new();
    let n = py.detach(|| passes::prune(&mut s, &frame, &mut em, Prior { lam, a_s }));
    let (p, a) = give_emitters(py, em);
    Ok((p, a, n))
}

// ------------------------------------------------------------- image pieces

/// Global model image: `background` plus every emitter's PSF, summed.
#[pyfunction]
#[pyo3(signature = (positions, amplitudes, sigma, shape, background=0.0, truncate=render::RENDER_TRUNCATE))]
fn render_model(
    py: Python<'_>,
    positions: PyReadonlyArray2<'_, f64>,
    amplitudes: PyReadonlyArray1<'_, f64>,
    sigma: f64,
    shape: (usize, usize),
    background: f64,
    truncate: f64,
) -> PyResult<Arr2> {
    let em = take_emitters(&positions, &amplitudes)?;
    let (h, w) = shape;
    let m = render::render_model(
        &em.pos,
        &em.amp,
        em.len(),
        sigma,
        h,
        w,
        background,
        truncate,
    );
    Ok(m.into_pyarray(py).reshape([h, w]).unwrap().unbind())
}

/// Pixels no emitter reaches: `True` where every emitter is farther than
/// `radius_factor * sigma`.
///
/// Exactly `free &= (yy-cy)**2 + (xx-cx)**2 > r2` accumulated over emitters, but
/// stamped rather than swept -- `O(N*sigma^2)` instead of `O(N*H*W)`. This is
/// the one piece of `background_map` that moves to Rust; its convolutions stay
/// in scipy.
#[pyfunction]
#[pyo3(signature = (positions, sigma, shape, radius_factor=3.0))]
fn emitter_free_mask(
    py: Python<'_>,
    positions: PyReadonlyArray2<'_, f64>,
    sigma: f64,
    shape: (usize, usize),
    radius_factor: f64,
) -> PyResult<Py<PyArray2<bool>>> {
    let pos = slice2(&positions, "positions")?;
    if positions.shape()[1] != 2 {
        return Err(PyValueError::new_err("`positions` must have shape (N, 2)"));
    }
    let (h, w) = shape;
    let m = render::emitter_free_mask(pos, positions.shape()[0], sigma, radius_factor, h, w);
    Ok(m.into_pyarray(py).reshape([h, w]).unwrap().unbind())
}

// --------------------------------------------------------- per-layer probes
//
// Exposed so the golden fixtures can be asserted from pytest as well as from
// `cargo test`, and so a discrepancy can be localized from the Python side
// without a Rust toolchain.

/// `(model, jacobian)` on a local `h x w` pixel grid. The Jacobian is returned
/// as `(h*w, 3K+1)` to match `psf.jac(...).reshape(-1, p)`, transposing from the
/// core's own parameter-major layout.
#[pyfunction]
fn psf_model_jac(
    py: Python<'_>,
    theta: PyReadonlyArray1<'_, f64>,
    h: usize,
    w: usize,
    sigma: f64,
) -> PyResult<(Arr2, Arr2)> {
    let t = slice1(&theta, "theta")?;
    if t.len() % 3 != 1 {
        return Err(PyValueError::new_err("`theta` must have length 3K+1"));
    }
    let (n, p) = (h * w, t.len());
    let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
    let mut f = psf::Factors::new(h, w, psf::n_emitters(t).max(1));
    let (mut m, mut j) = (vec![0.0; n], vec![0.0; p * n]);
    psf::model_and_jac_ax(t, &ay, &ax, sigma, None, &mut f, &mut m, &mut j);
    let jt: Vec<f64> = (0..n)
        .flat_map(|i| (0..p).map(move |q| (q, i)))
        .map(|(q, i)| j[q * n + i])
        .collect();
    Ok((
        m.into_pyarray(py).reshape([h, w]).unwrap().unbind(),
        jt.into_pyarray(py).reshape([n, p]).unwrap().unbind(),
    ))
}

/// One bounded fit. Returns `(theta, I, F, n_iter, converged, stalled)`.
#[pyfunction]
#[pyo3(signature = (theta0, h, w, sigma, d, lower, upper, max_iter=100))]
#[allow(clippy::too_many_arguments)]
fn lmcl_fit(
    py: Python<'_>,
    theta0: PyReadonlyArray1<'_, f64>,
    h: usize,
    w: usize,
    sigma: f64,
    d: PyReadonlyArray2<'_, f64>,
    lower: PyReadonlyArray1<'_, f64>,
    upper: PyReadonlyArray1<'_, f64>,
    max_iter: usize,
) -> PyResult<(Arr1, f64, Arr2, usize, bool, bool)> {
    let t = slice1(&theta0, "theta0")?;
    let data = slice2(&d, "d")?;
    let bounds = lmcl::Bounds::new(slice1(&lower, "lower")?, slice1(&upper, "upper")?);
    if data.len() != h * w {
        return Err(PyValueError::new_err("`d` does not match (h, w)"));
    }
    let p = t.len();
    let mut ws = lmcl::FitWorkspace::new();
    let info = lmcl::fit(
        &mut ws,
        t,
        h,
        w,
        sigma,
        data,
        &bounds,
        None,
        lmcl::FitOpts {
            max_iter,
            ..Default::default()
        },
    );
    Ok((
        ws.theta().to_vec().into_pyarray(py).unbind(),
        info.i_div,
        ws.fisher(p)
            .to_vec()
            .into_pyarray(py)
            .reshape([p, p])
            .unwrap()
            .unbind(),
        info.n_iter,
        info.converged,
        info.stalled,
    ))
}

/// One bounded variable-sigma fit.
///
/// `theta0` has length `4K+1`: `[b, A0, y0, x0, sigma0, ...]`. `halo` is the
/// parameter-free local contribution. `width_prior` carries the Cauchy
/// (center, scale, log normalizer); no Python callbacks run inside the fit.
#[pyfunction]
#[pyo3(signature = (theta0, h, w, d, halo, lower, upper, max_iter=180, *, tol_obj=1e-8, width_prior=None))]
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
    width_prior: Option<(f64, f64, f64)>,
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
        return Err(PyValueError::new_err(
            "`tol_obj` must be positive and finite",
        ));
    }
    let penalty = match width_prior {
        Some((sigma0, scale, log_z)) => {
            if !sigma0.is_finite()
                || sigma0 <= 0.0
                || !scale.is_finite()
                || scale <= 0.0
                || !log_z.is_finite()
            {
                return Err(PyValueError::new_err("invalid Cauchy width prior"));
            }
            Some(lmcl::WidthPenalty {
                sigma0,
                scale,
                log_z,
            })
        }
        None => None,
    };
    let bounds = lmcl::Bounds::new(lo, hi);
    if h == 0 || w == 0 || d.shape() != [h, w] || data.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("`d` does not match (h, w)"));
    }
    if halo.len() != h * w || halo.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("`halo` does not match (h, w)"));
    }
    let p = t.len();
    let mut ws = lmcl::FitWorkspace::new();
    let info = lmcl::fit_var_sigma_map(
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
        penalty,
    );
    Ok((
        ws.theta().to_vec().into_pyarray(py).unbind(),
        info.i_div,
        ws.fisher(p)
            .to_vec()
            .into_pyarray(py)
            .reshape([p, p])
            .unwrap()
            .unbind(),
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

/// `(log|F|, scaled condition number, ok)`.
#[pyfunction]
fn logdet_cond(f: PyReadonlyArray2<'_, f64>) -> PyResult<(f64, f64, bool)> {
    let a = slice2(&f, "F")?;
    let n = f.shape()[0];
    if f.shape()[1] != n {
        return Err(PyValueError::new_err("`F` must be square"));
    }
    let (mut chol, mut scratch) = (linalg::Chol::new(n), Vec::new());
    Ok(linalg::logdet_cond(a, n, &mut chol, &mut scratch))
}

/// `log BF` for `K_before -> K_before + 1`, and the scaled condition number of
/// the larger model's Fisher matrix.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn log_bf_add(
    i_before: f64,
    i_after: f64,
    f_before: PyReadonlyArray2<'_, f64>,
    f_after: PyReadonlyArray2<'_, f64>,
    sum_a_before: f64,
    sum_a_after: f64,
    k_before: usize,
    lam: f64,
    a_s: f64,
) -> PyResult<(f64, f64)> {
    let (fb, fa) = (slice2(&f_before, "F_before")?, slice2(&f_after, "F_after")?);
    let (nb, na) = (f_before.shape()[0], f_after.shape()[0]);
    Ok(Evidence::new().log_bf_add(
        i_before,
        i_after,
        fb,
        fa,
        nb,
        na,
        sum_a_before,
        sum_a_after,
        k_before,
        Prior { lam, a_s },
        None,
    ))
}

/// `log BF` for `K_full -> K_full - 1`. The exact negation of [`log_bf_add`].
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn log_bf_remove(
    i_full: f64,
    i_reduced: f64,
    f_full: PyReadonlyArray2<'_, f64>,
    f_reduced: PyReadonlyArray2<'_, f64>,
    sum_a_full: f64,
    sum_a_reduced: f64,
    k_full: usize,
    lam: f64,
    a_s: f64,
) -> PyResult<f64> {
    if k_full == 0 {
        return Err(PyRuntimeError::new_err("cannot remove from an empty model"));
    }
    let (ff, fr) = (slice2(&f_full, "F_full")?, slice2(&f_reduced, "F_reduced")?);
    let (nf, nr) = (f_full.shape()[0], f_reduced.shape()[0]);
    Ok(Evidence::new().log_bf_remove(
        i_full,
        i_reduced,
        ff,
        fr,
        nf,
        nr,
        sum_a_full,
        sum_a_reduced,
        k_full,
        Prior { lam, a_s },
        None,
    ))
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn spotsolve_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<inference::CalibratedModel>()?;
    dense_group::register(m)?;
    m.add_function(wrap_pyfunction!(add_pass, m)?)?;
    m.add_function(wrap_pyfunction!(split_pass, m)?)?;
    m.add_function(wrap_pyfunction!(refine, m)?)?;
    m.add_function(wrap_pyfunction!(prune, m)?)?;
    m.add_function(wrap_pyfunction!(render_model, m)?)?;
    m.add_function(wrap_pyfunction!(emitter_free_mask, m)?)?;
    m.add_function(wrap_pyfunction!(psf_model_jac, m)?)?;
    m.add_function(wrap_pyfunction!(lmcl_fit, m)?)?;
    m.add_function(wrap_pyfunction!(lmcl_fit_var_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(localize_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(logdet_cond, m)?)?;
    m.add_function(wrap_pyfunction!(log_bf_add, m)?)?;
    m.add_function(wrap_pyfunction!(log_bf_remove, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    // The constants the Python driver must agree with, so a drift is visible
    // rather than silent.
    m.add("PRUNE_TAU", passes::PRUNE_TAU)?;
    m.add("COND_GUARD", spotsolve_core::evidence::COND_GUARD)?;
    m.add("REFINE_TOL", passes::REFINE_TOL)?;
    m.add("REFINE_TOL_OBJ", passes::REFINE_TOL_OBJ)?;
    m.add("REFINE_MAX_ITER", passes::REFINE_MAX_ITER)?;
    m.add("EVIDENCE_TOL_OBJ", passes::EVIDENCE_TOL_OBJ)?;
    Ok(())
}
