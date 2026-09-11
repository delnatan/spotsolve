//! Binding for the native local group search.
//!
//! # One operation, not a dispatcher
//!
//! The variable-width path used to cross this boundary once per *fit*: Python
//! built each window, each proposal and each comparison, and called Rust to
//! minimize. [`DenseGroupEngine::search_group`] crosses it once per *group
//! transaction* -- Rust builds the neighbourhood, generates births, splits and
//! removals, fits and scores them all, and commits the winner. There is no
//! numerical or decision callback into Python inside a call.
//!
//! # Ownership and concurrency
//!
//! The engine copies the frame once, at construction, so a transaction holds
//! no borrow of a numpy array and the GIL is released for the whole search.
//! Each instance owns its own workspace and id allocator; two instances share
//! no mutable solver state and may run concurrently.
//!
//! # Where validation happens
//!
//! Here, and only here. Shapes, contiguity, finiteness, bound ordering and
//! every settings value are checked at this boundary so the core can assume
//! them. In particular an unsupported prior is rejected with a message naming
//! what is supported, rather than silently substituted -- a detector that
//! quietly changed its prior would change what exists.

use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spotsolve_core::dense_group as dg;
use spotsolve_core::lmcl::FitOpts;

fn slice1<'a>(a: &'a PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("`{name}` must be a contiguous float64 array")))
}

fn slice2<'a>(a: &'a PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "`{name}` must be a C-contiguous float64 array; pass \
             np.ascontiguousarray(x, dtype=float)"
        ))
    })
}

/// Translate a Python-side width prior into the native one.
///
/// `kind` is `"uniform"` or `"focus_mixture"`; anything else is refused. The
/// normalizer is NOT taken from Python -- the core computes its own, so the
/// density the fit is penalized by and the density the score charges are the
/// same object by construction rather than by two implementations agreeing.
fn width_prior(kind: &str, p: &[f64]) -> PyResult<dg::WidthPrior> {
    match kind {
        "uniform" => {
            if p.len() != 3 {
                return Err(PyValueError::new_err(
                    "uniform width prior takes (lam, lo, hi)",
                ));
            }
            if !(p[1] < p[2]) || p[0] <= 0.0 {
                return Err(PyValueError::new_err(
                    "uniform width prior needs lo < hi and lam > 0",
                ));
            }
            Ok(dg::WidthPrior::Uniform {
                lam: p[0],
                lo: p[1],
                hi: p[2],
            })
        }
        "focus_mixture" => {
            if p.len() != 7 {
                return Err(PyValueError::new_err(
                    "focus mixture width prior takes \
                     (lam_focus, lam_wide, lo, mid, hi, sigma0, scale)",
                ));
            }
            if !(p[2] < p[3] && p[3] < p[4]) {
                return Err(PyValueError::new_err(
                    "focus mixture needs lo < mid < hi: `mid` is the class \
                     boundary, and a boundary on a bound leaves one class \
                     empty by construction",
                ));
            }
            if p[0] <= 0.0 || p[1] <= 0.0 || p[5] <= 0.0 || p[6] <= 0.0 {
                return Err(PyValueError::new_err(
                    "focus mixture needs positive rates, sigma0 and scale",
                ));
            }
            Ok(dg::WidthPrior::FocusMixture {
                lam_focus: p[0],
                lam_wide: p[1],
                lo: p[2],
                mid: p[3],
                hi: p[4],
                sigma0: p[5],
                scale: p[6],
            })
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported width prior {other:?}; the native group search \
             implements 'uniform' (prior.UniformWidth) and 'focus_mixture' \
             (prior.FocusMixtureWidth). A custom WidthPrior subclass is \
             refused rather than approximated by one of these."
        ))),
    }
}

/// A prepared group engine: one frame, one prior snapshot, reusable storage.
#[pyclass(name = "DenseGroupEngine")]
pub struct DenseGroupEngine {
    d: Vec<f64>,
    bmap: Vec<f64>,
    h: usize,
    w: usize,
    spec: dg::ContextSpec,
    ws: dg::GroupWorkspace,
    ids: dg::IdAllocator,
    version: u64,
}

#[pymethods]
impl DenseGroupEngine {
    /// `d_e` and `bmap` are copied once here, so later calls hold no borrow.
    #[new]
    #[pyo3(signature = (d_e, bmap, sigma, slack, k_max, a_s, width_kind, width_params, next_id=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        d_e: PyReadonlyArray2<'_, f64>,
        bmap: PyReadonlyArray2<'_, f64>,
        sigma: f64,
        slack: (f64, f64),
        k_max: usize,
        a_s: f64,
        width_kind: &str,
        width_params: Vec<f64>,
        next_id: u32,
    ) -> PyResult<Self> {
        let d = slice2(&d_e, "d_e")?;
        let b = slice2(&bmap, "bmap")?;
        let (h, w) = (d_e.shape()[0], d_e.shape()[1]);
        if bmap.shape() != d_e.shape() {
            return Err(PyValueError::new_err("`bmap` must have `d_e`'s shape"));
        }
        if h == 0 || w == 0 {
            return Err(PyValueError::new_err("`d_e` must be non-empty"));
        }
        if d.iter().chain(b.iter()).any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("`d_e` and `bmap` must be finite"));
        }
        if !(sigma > 0.0 && sigma.is_finite()) {
            return Err(PyValueError::new_err("`sigma` must be positive"));
        }
        if !(slack.0 > 0.0 && slack.0 < slack.1 && slack.1.is_finite()) {
            return Err(PyValueError::new_err("`slack` must be 0 < lo < hi"));
        }
        if k_max == 0 || k_max > spotsolve_core::linalg::K_MAX {
            return Err(PyValueError::new_err(format!(
                "`k_max` must be in 1..={}; the group workspace is sized for \
                 that plus the K+1 alternative",
                spotsolve_core::linalg::K_MAX
            )));
        }
        if !(a_s > 0.0 && a_s.is_finite()) {
            return Err(PyValueError::new_err(
                "`a_s` (the exponential flux scale) must be positive; the \
                 native group search implements prior.ExponentialFlux only",
            ));
        }
        let width = width_prior(width_kind, &width_params)?;
        Ok(Self {
            d: d.to_vec(),
            bmap: b.to_vec(),
            h,
            w,
            spec: dg::ContextSpec {
                sigma0: sigma,
                slack,
                k_max,
                prior: dg::PriorSnapshot {
                    flux: dg::FluxPrior::Exponential { a_s },
                    width,
                },
            },
            ws: dg::GroupWorkspace::new(),
            ids: dg::IdAllocator::starting_at(next_id),
            version: 0,
        })
    }

    /// Replace the prior snapshot between frame epochs, and bump the version.
    ///
    /// Bumping is not optional bookkeeping: a score computed under the old
    /// prior is not comparable with one computed under the new one, and the
    /// version is how a caller holding a cached score finds that out.
    #[pyo3(signature = (a_s, width_kind, width_params))]
    fn set_prior(&mut self, a_s: f64, width_kind: &str, width_params: Vec<f64>) -> PyResult<()> {
        if !(a_s > 0.0 && a_s.is_finite()) {
            return Err(PyValueError::new_err("`a_s` must be positive"));
        }
        self.spec.prior = dg::PriorSnapshot {
            flux: dg::FluxPrior::Exponential { a_s },
            width: width_prior(width_kind, &width_params)?,
        };
        self.version += 1;
        Ok(())
    }

    /// Re-read the background surface. Bumps the version for the same reason.
    fn set_background(&mut self, bmap: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let b = slice2(&bmap, "bmap")?;
        if bmap.shape() != [self.h, self.w] {
            return Err(PyValueError::new_err("`bmap` must keep the frame's shape"));
        }
        if b.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("`bmap` must be finite"));
        }
        self.bmap.copy_from_slice(b);
        self.version += 1;
        Ok(())
    }

    /// The current context version. Discard cached scores from another one.
    #[getter]
    fn version(&self) -> u64 {
        self.version
    }

    /// The next id that will be minted. Carry it across engines on one frame.
    #[getter]
    fn next_id(&self) -> u32 {
        self.ids.peek()
    }

    /// Optimize the local group around `focus`, and commit the result.
    ///
    /// `positions`, `amplitudes`, `sigmas` and `ids` describe EVERY committed
    /// emitter on the frame, not just the group: which of them are free, which
    /// are frozen into the halo and which are irrelevant is decided natively,
    /// from width-aware support. They must include nuisance sources outside
    /// the reporting band -- filtering those out here would put their light
    /// back into the residual the search reads.
    ///
    /// `seeds` are candidate positions with no statistical standing; they say
    /// where to look and never what is there. `seed_amps <= 0` means "take the
    /// start from the residual".
    #[pyo3(signature = (
        positions, amplitudes, sigmas, ids, focus, seeds=None, seed_amps=None, *,
        max_moves=8, max_fits=600, max_iter=100, tol_obj=1e-8,
        escalated_max_iter=300, escalated_tol_obj=1e-10, max_restarts=1,
        incumbent_restarts=3, escalate_restarts=3,
        min_gain=dg::SCORE_TOL, escalate_band=dg::ESCALATE_BAND,
        split_disps=None, width_starts=None, max_birth_seeds=3,
        outside_peak_alpha=dg::OUTSIDE_PEAK_ALPHA
    ))]
    #[allow(clippy::too_many_arguments)]
    fn search_group<'py>(
        &mut self,
        py: Python<'py>,
        positions: PyReadonlyArray2<'py, f64>,
        amplitudes: PyReadonlyArray1<'py, f64>,
        sigmas: PyReadonlyArray1<'py, f64>,
        ids: PyReadonlyArray1<'py, u32>,
        focus: (f64, f64),
        seeds: Option<PyReadonlyArray2<'py, f64>>,
        seed_amps: Option<PyReadonlyArray1<'py, f64>>,
        max_moves: usize,
        max_fits: usize,
        max_iter: usize,
        tol_obj: f64,
        escalated_max_iter: usize,
        escalated_tol_obj: f64,
        max_restarts: usize,
        incumbent_restarts: usize,
        escalate_restarts: usize,
        min_gain: f64,
        escalate_band: f64,
        split_disps: Option<Vec<f64>>,
        width_starts: Option<Vec<f64>>,
        max_birth_seeds: usize,
        outside_peak_alpha: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        // --- Inputs, copied so the GIL can be released for the search.
        if !focus.0.is_finite() || !focus.1.is_finite() {
            return Err(PyValueError::new_err("`focus` must be finite"));
        }
        let emitters = self.read_emitters(&positions, &amplitudes, &sigmas, &ids)?;
        let seed_list = self.read_seeds(&seeds, seed_amps.as_ref())?;

        // --- Settings.
        if max_moves == 0 || max_fits == 0 || max_iter == 0 || escalated_max_iter == 0 {
            return Err(PyValueError::new_err("every budget must be positive"));
        }
        if !(tol_obj > 0.0) || !(escalated_tol_obj > 0.0) {
            return Err(PyValueError::new_err("tolerances must be positive"));
        }
        if !(min_gain > 0.0 && min_gain.is_finite()) {
            return Err(PyValueError::new_err(
                "`min_gain` must be positive: it is the numerical resolution \
                 of the score, and a non-positive one would accept a move that \
                 improved nothing",
            ));
        }
        if !(escalate_band >= 0.0 && escalate_band.is_finite()) {
            return Err(PyValueError::new_err("`escalate_band` must be >= 0"));
        }
        if !(outside_peak_alpha > 0.0 && outside_peak_alpha < 1.0) {
            return Err(PyValueError::new_err(
                "`outside_peak_alpha` must be in (0, 1)",
            ));
        }
        let disps = split_disps.unwrap_or_else(|| dg::GroupSettings::default().split_disps);
        let starts = width_starts.unwrap_or_else(|| dg::GroupSettings::default().width_starts);
        if disps.iter().any(|&v| !(v > 0.0 && v.is_finite()))
            || starts.iter().any(|&v| !(v > 0.0 && v.is_finite()))
        {
            return Err(PyValueError::new_err(
                "`split_disps` and `width_starts` must be positive and finite",
            ));
        }
        let settings = dg::GroupSettings {
            k_max: self.spec.k_max,
            max_moves,
            max_fits,
            fit: FitOpts {
                max_iter,
                tol_obj,
                ..Default::default()
            },
            escalated_fit: FitOpts {
                max_iter: escalated_max_iter,
                tol_obj: escalated_tol_obj,
                ..Default::default()
            },
            max_restarts,
            incumbent_restarts,
            escalate_restarts,
            min_gain,
            escalate_band,
            split_disps: disps,
            width_starts: starts,
            max_birth_seeds,
            outside_peak_alpha,
        };

        // --- The search itself, with the GIL released.
        let (h, w) = (self.h, self.w);
        let version = self.version;
        let spec = self.spec;
        let (outcome, ctx_free, ctx_frozen) = py.detach(|| {
            let frame = dg::FrameView {
                d: &self.d,
                bmap: &self.bmap,
                h,
                w,
            };
            let ctx = dg::build_context(&frame, &emitters, focus, &seed_list, &spec, version);
            let entry = dg::entry_state(&ctx);
            let free = ctx.free.len();
            let frozen = ctx.frozen.len();
            let out = dg::search_group(&ctx, &mut self.ws, &entry, &mut self.ids, &settings);
            (out, free, frozen)
        });

        // --- Owned result arrays.
        let k = outcome.state.len();
        let mut out_pos = Vec::with_capacity(2 * k);
        let mut out_amp = Vec::with_capacity(k);
        let mut out_sig = Vec::with_capacity(k);
        let mut out_ids = Vec::with_capacity(k);
        for e in &outcome.state.emitters {
            out_pos.push(e.y);
            out_pos.push(e.x);
            out_amp.push(e.flux);
            out_sig.push(e.sigma);
            out_ids.push(e.id);
        }

        let d = PyDict::new(py);
        d.set_item(
            "positions",
            out_pos.into_pyarray(py).reshape([k, 2]).unwrap(),
        )?;
        d.set_item("amplitudes", out_amp.into_pyarray(py))?;
        d.set_item("sigmas", out_sig.into_pyarray(py))?;
        d.set_item("ids", out_ids.into_pyarray(py))?;
        d.set_item("background", outcome.state.background)?;
        d.set_item("changed", outcome.changed.clone().into_pyarray(py))?;
        d.set_item("removed", outcome.removed.clone().into_pyarray(py))?;
        d.set_item("status", outcome.status.reason())?;
        d.set_item("score", outcome.score)?;
        d.set_item("score_status", outcome.score_status.reason())?;
        d.set_item("version", outcome.version)?;

        // Fit status, kept separate from search status: a search can end
        // cleanly on a converged fit or end unresolved with none.
        let fit = PyDict::new(py);
        fit.set_item("i_div", outcome.fit.i_div)?;
        fit.set_item("n_iter", outcome.fit.n_iter)?;
        fit.set_item("converged", outcome.fit.converged)?;
        fit.set_item("stalled", outcome.fit.stalled)?;
        d.set_item("fit", fit)?;

        d.set_item(
            "uncertainty",
            outcome.uncertainty.map(|u| u.into_pyarray(py)),
        )?;

        let trace = PyList::empty(py);
        for m in &outcome.trace {
            let e = PyDict::new(py);
            e.set_item("kind", m.kind.name())?;
            e.set_item("target", m.target)?;
            e.set_item("gain", m.gain)?;
            e.set_item("box_gain", m.box_gain)?;
            e.set_item("score", m.score)?;
            trace.append(e)?;
        }
        d.set_item("trace", trace)?;

        let g = &outcome.diag;
        let diag = PyDict::new(py);
        diag.set_item("n_fits", g.n_fits)?;
        diag.set_item("n_restarts", g.n_restarts)?;
        diag.set_item("fits_birth", g.fits_by_kind[0])?;
        diag.set_item("fits_split", g.fits_by_kind[1])?;
        diag.set_item("fits_removal", g.fits_by_kind[2])?;
        diag.set_item("proposals_generated", g.proposals_generated)?;
        diag.set_item("capacity_limited", g.capacity_limited)?;
        diag.set_item("edge_clipped", g.edge_clipped)?;
        diag.set_item("position_bound_active", g.position_bound_active)?;
        diag.set_item("residual_peak_outside_box", g.residual_peak_outside_box)?;
        let peaks: Vec<f64> = g.outside_peaks.iter().flat_map(|&(y, x)| [y, x]).collect();
        let n_peaks = g.outside_peaks.len();
        diag.set_item("outside_peaks", peaks.into_pyarray(py).reshape([n_peaks, 2]).unwrap())?;
        diag.set_item("incumbent_unsupported", g.incumbent_unsupported)?;
        diag.set_item("boundary_scored", g.boundary_scored)?;
        diag.set_item(
            "incumbent_status",
            g.incumbent_status.map(|s| s.reason()),
        )?;
        diag.set_item("n_free", ctx_free)?;
        diag.set_item("n_frozen", ctx_frozen)?;
        let unsupported = PyDict::new(py);
        for (reason, count) in &g.unsupported {
            unsupported.set_item(reason, count)?;
        }
        diag.set_item("unsupported", unsupported)?;
        d.set_item("diagnostics", diag)?;

        Ok(d)
    }

    /// Fit and score ONE configuration, and return every term of the score.
    ///
    /// The development window onto the comparison: the data-only objective,
    /// the configuration prior, the Laplace volume's two pieces, the scaled
    /// condition number, the stationarity flags and the validity reason. It
    /// runs exactly the code path `search_group` scores a hypothesis with --
    /// a separate scoring routine for diagnostics would eventually disagree
    /// with the one that decides.
    ///
    /// Adapting a normal result never needs this. Nothing in the pipeline
    /// calls it; `scripts/measure_group_score.py` and the control scripts do.
    #[pyo3(signature = (positions, amplitudes, sigmas, ids, focus, seeds=None,
                        *, max_iter=100, tol_obj=1e-8))]
    #[allow(clippy::too_many_arguments)]
    fn score_state<'py>(
        &mut self,
        py: Python<'py>,
        positions: PyReadonlyArray2<'py, f64>,
        amplitudes: PyReadonlyArray1<'py, f64>,
        sigmas: PyReadonlyArray1<'py, f64>,
        ids: PyReadonlyArray1<'py, u32>,
        focus: (f64, f64),
        seeds: Option<PyReadonlyArray2<'py, f64>>,
        max_iter: usize,
        tol_obj: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let emitters = self.read_emitters(&positions, &amplitudes, &sigmas, &ids)?;
        let seed_list = self.read_seeds(&seeds, None)?;
        if max_iter == 0 || !(tol_obj > 0.0) {
            return Err(PyValueError::new_err("`max_iter` and `tol_obj` must be positive"));
        }
        let (h, w, version, spec) = (self.h, self.w, self.version, self.spec);
        let opts = FitOpts { max_iter, tol_obj, ..Default::default() };
        let (hyp, theta_len, k, region) = py.detach(|| {
            let frame = dg::FrameView { d: &self.d, bmap: &self.bmap, h, w };
            let ctx = dg::build_context(&frame, &emitters, focus, &seed_list, &spec, version);
            let entry = dg::entry_state(&ctx);
            let mut theta = Vec::new();
            ctx.pack_state(&entry, &mut theta);
            let k = entry.len();
            let hyp = dg::fit_and_score(&ctx, &mut self.ws, &theta, opts);
            // Everything an independent referee needs to integrate the SAME
            // posterior over the SAME support: the region, its observations
            // and halo, the fit bounds and the score's support.
            let (lo, hi) = ctx.bounds.arrays(hyp.k());
            let (slo, shi) = dg::score_support(&ctx, &hyp.theta, &lo, &hi);
            let grad = self.ws.last_gradient(hyp.p);
            let region = (ctx.y0, ctx.x0, ctx.h, ctx.w, ctx.obs.clone(), ctx.halo.clone(), lo, hi, slo, shi, grad);
            (hyp, theta.len(), k, region)
        });
        let (y0, x0, rh, rw, obs, halo, lo, hi, slo, shi, grad) = region;
        let d = PyDict::new(py);
        d.set_item("score", if hyp.status.is_supported() { Some(hyp.score) } else { None })?;
        d.set_item("status", hyp.status.reason())?;
        d.set_item("i_div", hyp.i_div)?;
        d.set_item("log_prior", hyp.log_prior)?;
        d.set_item("logdet", hyp.logdet)?;
        d.set_item("cond", hyp.cond)?;
        d.set_item("p", hyp.p)?;
        d.set_item("k", k)?;
        d.set_item("theta_len", theta_len)?;
        d.set_item("objective", hyp.objective)?;
        d.set_item("converged", hyp.fit.converged)?;
        d.set_item("stalled", hyp.fit.stalled)?;
        d.set_item("n_iter", hyp.fit.n_iter)?;
        d.set_item(
            "log_volume",
            0.5 * hyp.p as f64 * (2.0 * std::f64::consts::PI).ln() - 0.5 * hyp.logdet,
        )?;
        d.set_item("theta", hyp.theta.clone().into_pyarray(py))?;
        d.set_item("log_box", hyp.log_box)?;
        d.set_item("n_active", hyp.n_active)?;
        d.set_item("curvature", hyp.curvature.clone().into_pyarray(py).reshape([hyp.p, hyp.p]).unwrap())?;
        d.set_item("gradient", grad.into_pyarray(py))?;
        d.set_item("fit_lo", lo.into_pyarray(py))?;
        d.set_item("fit_hi", hi.into_pyarray(py))?;
        d.set_item("score_lo", slo.into_pyarray(py))?;
        d.set_item("score_hi", shi.into_pyarray(py))?;
        d.set_item("region", (y0, x0, rh, rw))?;
        d.set_item("obs", obs.into_pyarray(py).reshape([rh, rw]).unwrap())?;
        d.set_item("halo", halo.into_pyarray(py).reshape([rh, rw]).unwrap())?;
        Ok(d)
    }
}

impl DenseGroupEngine {
    /// Validate and copy the committed-emitter arrays. Shared by both entry
    /// points so their contract cannot drift apart.
    fn read_emitters(
        &self,
        positions: &PyReadonlyArray2<'_, f64>,
        amplitudes: &PyReadonlyArray1<'_, f64>,
        sigmas: &PyReadonlyArray1<'_, f64>,
        ids: &PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Vec<dg::Emitter>> {
        let pos = slice2(positions, "positions")?;
        let amp = slice1(amplitudes, "amplitudes")?;
        let sig = slice1(sigmas, "sigmas")?;
        let id = ids
            .as_slice()
            .map_err(|_| PyValueError::new_err("`ids` must be a contiguous uint32 array"))?;
        let n = amp.len();
        if positions.shape() != [n, 2] || sig.len() != n || id.len() != n {
            return Err(PyValueError::new_err(
                "`positions` must be (N, 2) and `amplitudes`, `sigmas`, `ids` \
                 must all have length N",
            ));
        }
        if pos.iter().chain(amp).chain(sig).any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "positions, amplitudes and sigmas must be finite",
            ));
        }
        if sig.iter().any(|&s| s <= 0.0) || amp.iter().any(|&a| a <= 0.0) {
            return Err(PyValueError::new_err(
                "amplitudes and sigmas must be positive",
            ));
        }
        let mut seen = id.to_vec();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != n {
            return Err(PyValueError::new_err("`ids` must be unique"));
        }
        Ok((0..n)
            .map(|i| dg::Emitter {
                id: id[i],
                y: pos[2 * i],
                x: pos[2 * i + 1],
                flux: amp[i],
                sigma: sig[i],
            })
            .collect())
    }

    fn read_seeds(
        &self,
        seeds: &Option<PyReadonlyArray2<'_, f64>>,
        seed_amps: Option<&PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Vec<dg::Seed>> {
        let Some(s) = seeds else {
            return Ok(Vec::new());
        };
        let sv = slice2(s, "seeds")?;
        let m = s.shape()[0];
        if s.shape()[1] != 2 {
            return Err(PyValueError::new_err("`seeds` must be (M, 2)"));
        }
        if sv.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("`seeds` must be finite"));
        }
        let a: Option<&[f64]> = match seed_amps {
            None => None,
            Some(v) => {
                let a = slice1(v, "seed_amps")?;
                if a.len() != m {
                    return Err(PyValueError::new_err(
                        "`seed_amps` must have one entry per seed",
                    ));
                }
                Some(a)
            }
        };
        Ok((0..m)
            .map(|i| dg::Seed {
                y: sv[2 * i],
                x: sv[2 * i + 1],
                flux: a.map_or(0.0, |v| v[i].max(0.0)),
            })
            .collect())
    }
}

/// `log p(K, classes, widths)` for one configuration, from the native prior.
///
/// Exposed so the Python side can assert that this really is
/// `prior.WidthPrior.log_config` rather than assuming it. Nothing in the
/// pipeline calls it.
#[pyfunction]
pub fn group_width_log_config(
    sigmas: PyReadonlyArray1<'_, f64>,
    width_kind: &str,
    width_params: Vec<f64>,
) -> PyResult<f64> {
    let s = slice1(&sigmas, "sigmas")?;
    Ok(width_prior(width_kind, &width_params)?.log_config(s))
}

/// `sum_k log g(A_k)` for one configuration, from the native flux prior.
#[pyfunction]
pub fn group_flux_log_config(amplitudes: PyReadonlyArray1<'_, f64>, a_s: f64) -> PyResult<f64> {
    let a = slice1(&amplitudes, "amplitudes")?;
    if !(a_s > 0.0 && a_s.is_finite()) {
        return Err(PyValueError::new_err("`a_s` must be positive"));
    }
    Ok(dg::FluxPrior::Exponential { a_s }.log_config(a.iter().copied()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DenseGroupEngine>()?;
    m.add_function(wrap_pyfunction!(group_width_log_config, m)?)?;
    m.add_function(wrap_pyfunction!(group_flux_log_config, m)?)?;
    m.add("GROUP_SCORE_TOL", dg::SCORE_TOL)?;
    m.add("GROUP_ESCALATE_BAND", dg::ESCALATE_BAND)?;
    m.add("GROUP_COND_LIMIT", dg::COND_LIMIT)?;
    m.add("GROUP_BOUND_TOL", dg::BOUND_TOL)?;
    m.add("GROUP_DRIFT_FACTOR", dg::DRIFT_FACTOR)?;
    m.add("GROUP_OUTSIDE_PEAK_ALPHA", dg::OUTSIDE_PEAK_ALPHA)?;
    m.add("GROUP_STARTUP_SE_FRAC", dg::STARTUP_SE_FRAC)?;
    m.add("GROUP_P_MAX_VAR", spotsolve_core::linalg::P_MAX_VAR)?;
    Ok(())
}
