//! Persistent calibration/model and reusable component workspace.
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spotsolve_core::inference::{Calibration, Model, Workspace};
use spotsolve_core::{affine, geometry, search, uncertainty};

/// Separate instances may run concurrently. A mutable call borrows this
/// instance's workspace exclusively; returned arrays own their storage.
#[pyclass(name = "CalibratedModel")]
pub struct CalibratedModel {
    model: Model,
    workspace: Workspace,
    solver: affine::Workspace,
    geometry: geometry::Workspace,
    uncertainty: uncertainty::Workspace,
    constraints: Vec<f64>,
    constraint_rhs: Vec<f64>,
}

#[pymethods]
impl CalibratedModel {
    #[new]
    fn new(
        depth: PyReadonlyArray1<'_, f64>,
        offsets: PyReadonlyArray1<'_, f64>,
        coefficients: PyReadonlyArray3<'_, f64>,
        shape: [usize; 2],
        focus_bounds: [f64; 4],
        defocus_bounds: [f64; 2],
    ) -> PyResult<Self> {
        let z = depth
            .as_slice()
            .map_err(|_| PyValueError::new_err("depth must be contiguous float64"))?;
        let xy = offsets
            .as_slice()
            .map_err(|_| PyValueError::new_err("offsets must be contiguous float64"))?;
        if coefficients.shape() != [z.len() + 4, xy.len() + 4, xy.len() + 4] {
            return Err(PyValueError::new_err("incorrect padded coefficient shape"));
        }
        let c = coefficients
            .as_slice()
            .map_err(|_| PyValueError::new_err("coefficients must be C-contiguous float64"))?;
        // One calibration copy establishes ownership across GIL-free calls.
        let calibration = Calibration::new(z, xy, c.to_vec()).map_err(PyValueError::new_err)?;
        let model = Model::new(shape, focus_bounds, defocus_bounds, calibration)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            model,
            workspace: Workspace::default(),
            solver: affine::Workspace::new(19),
            geometry: geometry::Workspace::default(),
            uncertainty: uncertainty::Workspace::default(),
            constraints: Vec::new(),
            constraint_rhs: Vec::new(),
        })
    }

    fn evaluate<'py>(
        &mut self,
        py: Python<'py>,
        theta: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray3<f64>>)> {
        // At most 26 values. Copy so other Python threads cannot mutate the
        // input while the GIL is released.
        let theta = theta
            .as_slice()
            .map_err(|_| PyValueError::new_err("theta must be contiguous float64"))?
            .to_vec();
        py.detach(|| self.model.evaluate(&theta, &mut self.workspace))
            .map_err(PyValueError::new_err)?;
        let [h, w] = self.model.shape;
        let mean = self
            .workspace
            .mean
            .clone()
            .into_pyarray(py)
            .reshape([h, w])?
            .unbind();
        let jac = self
            .workspace
            .jacobian
            .clone()
            .into_pyarray(py)
            .reshape([h, w, theta.len()])?
            .unbind();
        Ok((mean, jac))
    }

    fn affine_statistics<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        theta: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(f64, Py<numpy::PyArray1<f64>>, Py<PyArray2<f64>>)> {
        if data.shape() != self.model.shape {
            return Err(PyValueError::new_err("data shape must match model"));
        }
        let data = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("data must be C-contiguous float64"))?
            .to_vec();
        let theta = theta
            .as_slice()
            .map_err(|_| PyValueError::new_err("theta must be contiguous float64"))?
            .to_vec();
        let value = py
            .detach(|| {
                self.model
                    .affine_statistics(&data, &theta, &mut self.workspace)
            })
            .map_err(PyValueError::new_err)?;
        let n = self.workspace.gradient.len();
        let gradient = self.workspace.gradient.clone().into_pyarray(py).unbind();
        let hessian = self
            .workspace
            .hessian
            .clone()
            .into_pyarray(py)
            .reshape([n, n])?
            .unbind();
        Ok((value, gradient, hessian))
    }

    fn prepare_affine<'py>(
        &mut self,
        py: Python<'py>,
        theta: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Py<numpy::PyArray1<f64>>> {
        let theta = theta
            .as_slice()
            .map_err(|_| PyValueError::new_err("theta must be contiguous float64"))?
            .to_vec();
        let initial = py
            .detach(|| self.model.prepare_affine(&theta, &mut self.workspace))
            .map_err(PyValueError::new_err)?;
        Ok(initial.into_pyarray(py).unbind())
    }

    fn affine_at<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        coefficients: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(f64, Py<numpy::PyArray1<f64>>, Py<PyArray2<f64>>)> {
        if data.shape() != self.model.shape {
            return Err(PyValueError::new_err("data shape must match model"));
        }
        let data = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("data must be C-contiguous float64"))?
            .to_vec();
        let coefficients = coefficients
            .as_slice()
            .map_err(|_| PyValueError::new_err("coefficients must be contiguous float64"))?
            .to_vec();
        let value = py
            .detach(|| {
                self.model
                    .affine_at(&data, &coefficients, &mut self.workspace)
            })
            .map_err(PyValueError::new_err)?;
        let n = self.workspace.gradient.len();
        Ok((
            value,
            self.workspace.gradient.clone().into_pyarray(py).unbind(),
            self.workspace
                .hessian
                .clone()
                .into_pyarray(py)
                .reshape([n, n])?
                .unbind(),
        ))
    }
    /// Solve at explicitly prepared geometry. Data crosses the boundary once;
    /// all Newton, active-face and line-search iterations execute in Rust.
    #[pyo3(signature = (data, initial, lower, upper, tolerance=1e-7, max_iter=160))]
    fn solve_affine<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        initial: PyReadonlyArray1<'py, f64>,
        lower: PyReadonlyArray1<'py, f64>,
        upper: PyReadonlyArray1<'py, f64>,
        tolerance: f64,
        max_iter: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if data.shape() != self.model.shape {
            return Err(PyValueError::new_err("data shape must match model"));
        }
        let copy = |v: PyReadonlyArray1<'py, f64>| -> PyResult<Vec<f64>> {
            Ok(v.as_slice()
                .map_err(|_| PyValueError::new_err("parameters must be contiguous float64"))?
                .to_vec())
        };
        let initial = copy(initial)?;
        let lower = copy(lower)?;
        let upper = copy(upper)?;
        let data = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("data must be C-contiguous float64"))?
            .to_vec();
        let report = py
            .detach(|| {
                if self.workspace.affine.columns == 0
                    || initial.len() != self.workspace.affine.columns
                    || lower.len() != initial.len()
                {
                    return Err(
                        "prepare geometry and supply compatible affine coefficients".to_string()
                    );
                }
                spotsolve_core::inference::prepare_constraints(
                    &self.model,
                    &lower,
                    &upper,
                    &mut self.constraints,
                    &mut self.constraint_rhs,
                )?;
                let problem = affine::Problem {
                    mean: &self.workspace.affine,
                    observations: &data,
                    constraints: &self.constraints,
                    rhs: &self.constraint_rhs,
                };
                affine::solve(
                    &problem,
                    &initial,
                    affine::Options {
                        tolerance,
                        max_iter,
                    },
                    &mut self.solver,
                )
            })
            .map_err(PyValueError::new_err)?;
        let result = PyDict::new(py);
        result.set_item(
            "coefficients",
            self.solver.coefficients.clone().into_pyarray(py),
        )?;
        result.set_item(
            "multipliers",
            self.solver.multipliers.clone().into_pyarray(py),
        )?;
        result.set_item("objective", report.objective)?;
        result.set_item("kkt", report.kkt)?;
        result.set_item("feasibility", report.feasibility)?;
        result.set_item("iterations", report.iterations)?;
        result.set_item("hessian_evaluations", report.hessian_evaluations)?;
        result.set_item("line_search_evaluations", report.line_search_evaluations)?;
        result.set_item("status", report.status.name())?;
        result.set_item("converged", report.status == affine::Status::Converged)?;
        Ok(result)
    }
    #[pyo3(signature = (data, theta, lower, upper, tolerance=1e-7, max_iter=160))]
    fn profile_at<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        theta: PyReadonlyArray1<'py, f64>,
        lower: PyReadonlyArray1<'py, f64>,
        upper: PyReadonlyArray1<'py, f64>,
        tolerance: f64,
        max_iter: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if data.shape() != self.model.shape {
            return Err(PyValueError::new_err("data shape must match model"));
        }
        let data = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let theta = theta
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let lower = lower
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let upper = upper
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let result = py
            .detach(|| {
                let problem = geometry::Problem {
                    model: &self.model,
                    data: &data,
                    lower: &lower,
                    upper: &upper,
                };
                geometry::prepare(&problem, &theta, &mut self.geometry)?;
                geometry::profile(
                    &problem,
                    &theta,
                    affine::Options {
                        tolerance,
                        max_iter,
                    },
                    &mut self.geometry,
                )
            })
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item(
            "theta",
            self.geometry.theta[..theta.len()].to_vec().into_pyarray(py),
        )?;
        if result.converged {
            out.set_item(
                "gradient",
                self.geometry.gradient[..theta.len()]
                    .to_vec()
                    .into_pyarray(py),
            )?;
        } else {
            out.set_item("gradient", py.None())?;
        }
        out.set_item("objective", result.objective)?;
        out.set_item("inner_kkt", result.kkt)?;
        out.set_item("inner_iterations", result.iterations)?;
        out.set_item("converged", result.converged)?;
        Ok(out)
    }

    #[pyo3(signature = (data, theta, lower, upper, max_iter=400, gtol=1e-6, inner_max_iter=160))]
    fn fit_geometry<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        theta: PyReadonlyArray1<'py, f64>,
        lower: PyReadonlyArray1<'py, f64>,
        upper: PyReadonlyArray1<'py, f64>,
        max_iter: usize,
        gtol: f64,
        inner_max_iter: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if data.shape() != self.model.shape {
            return Err(PyValueError::new_err("data shape must match model"));
        }
        let data = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let theta = theta
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let lower = lower
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let upper = upper
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let result = py
            .detach(|| {
                let problem = geometry::Problem {
                    model: &self.model,
                    data: &data,
                    lower: &lower,
                    upper: &upper,
                };
                geometry::optimize(
                    &problem,
                    &theta,
                    &geometry::Options {
                        max_iter,
                        gtol,
                        inner: affine::Options {
                            tolerance: 1e-7_f64.min(0.1 * gtol),
                            max_iter: inner_max_iter,
                        },
                    },
                    &mut self.geometry,
                )
            })
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item(
            "theta",
            self.geometry.theta[..theta.len()].to_vec().into_pyarray(py),
        )?;
        if result.status != "inner_failure" {
            out.set_item(
                "gradient",
                self.geometry.gradient[..theta.len()]
                    .to_vec()
                    .into_pyarray(py),
            )?;
        } else {
            out.set_item("gradient", py.None())?;
        }
        out.set_item("objective", result.objective)?;
        out.set_item("geometry_kkt", result.geometry_kkt)?;
        out.set_item("inner_kkt", result.inner_kkt)?;
        out.set_item("iterations", result.iterations)?;
        out.set_item("evaluations", result.evaluations)?;
        out.set_item("inner_iterations", result.inner_iterations)?;
        out.set_item("inner_failures", result.inner_failures)?;
        out.set_item("status", result.status)?;
        out.set_item("converged", result.status == "converged")?;
        Ok(out)
    }
    fn observed_curvature<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        theta: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if data.shape() != self.model.shape {
            return Err(PyValueError::new_err("data shape must match model"));
        }
        let data = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let theta = theta
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        py.detach(|| uncertainty::observed(&self.model, &data, &theta, &mut self.uncertainty))
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        let p = theta.len();
        out.set_item(
            "gradient",
            self.uncertainty.gradient.clone().into_pyarray(py),
        )?;
        out.set_item(
            "hessian",
            self.uncertainty
                .hessian
                .clone()
                .into_pyarray(py)
                .reshape([p, p])?,
        )?;
        Ok(out)
    }

    fn position_uncertainty<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        theta: PyReadonlyArray1<'py, f64>,
        lower: PyReadonlyArray1<'py, f64>,
        upper: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if data.shape() != self.model.shape {
            return Err(PyValueError::new_err("data shape must match model"));
        }
        let data = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let theta = theta
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let lower = lower
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let upper = upper
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let result = py
            .detach(|| {
                uncertainty::positions(
                    &self.model,
                    &data,
                    &theta,
                    &lower,
                    &upper,
                    &mut self.uncertainty,
                )
            })
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("status", result.status)?;
        out.set_item("broad_conditioned_absent", result.broad_conditioned_absent)?;
        out.set_item("gradient_max", result.gradient_max)?;
        if result.status == "conditional_observed_hessian" {
            let r = result.positions;
            out.set_item(
                "covariance",
                self.uncertainty
                    .covariance
                    .clone()
                    .into_pyarray(py)
                    .reshape([r, r])?,
            )?;
            out.set_item(
                "nuisance_fixed_covariance",
                self.uncertainty
                    .fixed_covariance
                    .clone()
                    .into_pyarray(py)
                    .reshape([r, r])?,
            )?;
        } else {
            out.set_item("covariance", py.None())?;
            out.set_item("nuisance_fixed_covariance", py.None())?;
        }
        Ok(out)
    }
    #[pyo3(signature = (data, candidate_centres, seed_sigma=1., max_iter=400, screen_iter=48, keep_screened=4, gtol=1e-6, proposal_method="moments", proposal_alpha=0.05))]
    fn fit_component<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        candidate_centres: PyReadonlyArray2<'py, f64>,
        seed_sigma: f64,
        max_iter: usize,
        screen_iter: usize,
        keep_screened: usize,
        gtol: f64,
        proposal_method: &str,
        proposal_alpha: f64,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        if data.shape() != self.model.shape || candidate_centres.shape()[1] != 2 {
            return Err(PyValueError::new_err(
                "data must match model; candidate_centres must be (n,2)",
            ));
        }
        let proposals = match proposal_method {
            "moments" => search::Proposals::Moments,
            "aguet" => search::Proposals::Aguet {
                alpha: proposal_alpha,
            },
            _ => {
                return Err(PyValueError::new_err(
                    "proposal_method must be moments or aguet",
                ));
            }
        };
        // Copy once at the boundary so Python cannot mutate inputs during fitting.
        let data = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .to_vec();
        let centres = candidate_centres
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?
            .chunks_exact(2)
            .map(|v| [v[0], v[1]])
            .collect::<Vec<_>>();
        let result = py
            .detach(|| {
                search::fit_component(
                    &self.model,
                    &data,
                    &centres,
                    seed_sigma,
                    &search::Options {
                        proposals,
                        max_iter,
                        screen_iter,
                        keep_screened,
                        gtol,
                    },
                    &mut self.geometry,
                    &mut self.workspace,
                )
            })
            .map_err(PyValueError::new_err)?;
        result
            .into_iter()
            .map(|fit| {
                let out = PyDict::new(py);
                out.set_item("proposal_centres", fit.proposal_centres)?;
                out.set_item("theta", fit.theta.into_pyarray(py))?;
                out.set_item("objective", fit.objective)?;
                out.set_item("kkt", fit.kkt)?;
                out.set_item("status", fit.status)?;
                out.set_item("boundary", fit.boundary)?;
                out.set_item("starts", fit.starts)?;
                out.set_item("evaluations", fit.evaluations)?;
                out.set_item("inner_iterations", fit.inner_iterations)?;
                out.set_item("inner_failures", fit.inner_failures)?;
                Ok(out)
            })
            .collect()
    }

    fn parameter_bounds<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f64>,
        count: usize,
    ) -> PyResult<(Py<numpy::PyArray1<f64>>, Py<numpy::PyArray1<f64>>)> {
        if data.shape() != self.model.shape {
            return Err(PyValueError::new_err("data shape must match model"));
        }
        let data = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be contiguous float64"))?;
        let (lo, hi) = search::bounds(&self.model, data, count).map_err(PyValueError::new_err)?;
        Ok((lo.into_pyarray(py).unbind(), hi.into_pyarray(py).unbind()))
    }
}
