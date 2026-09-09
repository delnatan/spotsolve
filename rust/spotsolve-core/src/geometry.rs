//! Profiled geometry fitting: borrowed problem data, reusable workspace, and
//! free numerical functions. Dense BFGS is only a step metric for <=7 geometry
//! coordinates, never a localization covariance.
use crate::{affine, inference};

pub struct Problem<'a> {
    pub model: &'a inference::Model,
    pub data: &'a [f64],
    pub lower: &'a [f64],
    pub upper: &'a [f64],
}

pub struct Options {
    pub max_iter: usize,
    pub gtol: f64,
    pub inner: affine::Options,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_iter: 400,
            gtol: 1e-6,
            inner: affine::Options::default(),
        }
    }
}

pub struct Profile {
    pub objective: f64,
    pub kkt: f64,
    pub iterations: usize,
    pub converged: bool,
}

pub struct Report {
    pub objective: f64,
    pub geometry_kkt: f64,
    pub inner_kkt: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub inner_iterations: usize,
    pub inner_failures: usize,
    pub status: &'static str,
}

pub struct Workspace {
    pub theta: [f64; 26],
    pub gradient: [f64; 26],
    pub parameters: usize,
    model: inference::Workspace,
    linear: affine::Workspace,
    constraints: Vec<f64>,
    rhs: Vec<f64>,
    affine_ids: [usize; 19],
    geometry_ids: [usize; 7],
    affine_lower: [f64; 19],
    affine_upper: [f64; 19],
    inverse: [f64; 49],
}
impl Default for Workspace {
    fn default() -> Self {
        Self {
            theta: [0.; 26],
            gradient: [0.; 26],
            parameters: 0,
            model: inference::Workspace::default(),
            linear: affine::Workspace::new(19),
            constraints: Vec::new(),
            rhs: Vec::new(),
            affine_ids: [0; 19],
            geometry_ids: [0; 7],
            affine_lower: [0.; 19],
            affine_upper: [0.; 19],
            inverse: [0.; 49],
        }
    }
}

/// Prepare constraints once per image/start call, independently of geometry.
pub fn prepare(problem: &Problem, initial: &[f64], w: &mut Workspace) -> Result<(), String> {
    w.parameters = 0;
    let count = inference::Model::count(initial)?;
    let p = initial.len();
    let a = 17 + count;
    if problem.lower.len() != p
        || problem.upper.len() != p
        || problem.data.len() != problem.model.shape[0] * problem.model.shape[1]
        || problem.data.iter().any(|v| !v.is_finite() || *v < 0.)
        || initial
            .iter()
            .zip(problem.lower)
            .zip(problem.upper)
            .any(|((v, lo), hi)| {
                !v.is_finite() || !lo.is_finite() || !hi.is_finite() || lo > hi || v < lo || v > hi
            })
    {
        return Err(
            "finite compatible data, bounds and feasible initial parameters required".into(),
        );
    }
    for j in 0..17 {
        w.affine_ids[j] = j;
    }
    w.geometry_ids[..3].copy_from_slice(&[17, 18, 19]);
    for k in 0..count {
        w.affine_ids[17 + k] = 20 + 3 * k;
        w.geometry_ids[3 + 2 * k] = 21 + 3 * k;
        w.geometry_ids[4 + 2 * k] = 22 + 3 * k;
    }
    let [h, width] = problem.model.shape;
    let [y0, x0, y1, x1] = problem.model.focus_bounds;
    for (i, &id) in w.geometry_ids[..3 + 2 * count].iter().enumerate() {
        let (lo, hi) = match i {
            0 => (-0.5, h as f64 - 0.5),
            1 => (-0.5, width as f64 - 0.5),
            2 => (
                problem.model.defocus_bounds[0],
                problem.model.defocus_bounds[1],
            ),
            _ if i % 2 == 1 => (y0, y1),
            _ => (x0, x1),
        };
        if problem.lower[id] < lo || problem.upper[id] > hi {
            return Err("geometry bounds must lie inside the model bounds".into());
        }
    }
    for j in 0..a {
        w.affine_lower[j] = problem.lower[w.affine_ids[j]];
        w.affine_upper[j] = problem.upper[w.affine_ids[j]];
    }
    inference::prepare_constraints(
        problem.model,
        &w.affine_lower[..a],
        &w.affine_upper[..a],
        &mut w.constraints,
        &mut w.rhs,
    )?;
    w.parameters = p;
    Ok(())
}

/// Profile the affine block and rescale cached unit geometry derivatives. The
/// full gradient is meaningful only when the inner solve reports convergence.
pub fn profile(
    problem: &Problem,
    theta: &[f64],
    options: affine::Options,
    w: &mut Workspace,
) -> Result<Profile, String> {
    let p = w.parameters;
    if p == 0 || theta.len() != p {
        return Err("prepare a compatible geometry problem first".into());
    }
    let count = (p - 20) / 3;
    let a = 17 + count;
    let mut initial = [0.; 19];
    problem
        .model
        .prepare_affine_into(theta, &mut w.model, &mut initial[..a])?;
    let linear_problem = affine::Problem {
        mean: &w.model.affine,
        observations: problem.data,
        constraints: &w.constraints,
        rhs: &w.rhs,
    };
    let result = affine::solve(&linear_problem, &initial[..a], options, &mut w.linear)?;
    w.theta[..p].copy_from_slice(theta);
    for j in 0..a {
        w.theta[w.affine_ids[j]] = w.linear.coefficients[j];
    }
    w.gradient.fill(f64::NAN);
    let converged = result.status == affine::Status::Converged;
    if converged {
        w.gradient.fill(0.);
        // The inner solver already computed these at the accepted point.
        for j in 0..a {
            w.gradient[w.affine_ids[j]] = w.linear.gradient[j];
        }
        let mut amplitudes = [1.; 26];
        amplitudes[17] = w.theta[16];
        amplitudes[18] = w.theta[16];
        amplitudes[19] = w.theta[16];
        for k in 0..count {
            amplitudes[21 + 3 * k] = w.theta[20 + 3 * k];
            amplitudes[22 + 3 * k] = w.theta[20 + 3 * k];
        }
        for (i, &y) in problem.data.iter().enumerate() {
            let residual = 1. - y / w.linear.mean[i];
            for &j in &w.geometry_ids[..3 + 2 * count] {
                w.gradient[j] += w.model.jacobian[i * p + j] * amplitudes[j] * residual;
            }
        }
    }
    if converged && w.gradient[..p].iter().any(|v| !v.is_finite()) {
        return Err("nonfinite profiled geometry gradient".into());
    }
    Ok(Profile {
        objective: result.objective,
        kkt: result.kkt,
        iterations: result.iterations,
        converged,
    })
}

fn blocked(x: f64, g: f64, lo: f64, hi: f64) -> bool {
    lo == hi || (x <= lo + 1e-10 && g > 0.) || (x >= hi - 1e-10 && g < 0.)
}

fn projected(theta: &[f64], gradient: &[f64], lower: &[f64], upper: &[f64], ids: &[usize]) -> f64 {
    ids.iter()
        .map(|&j| {
            if blocked(theta[j], gradient[j], lower[j], upper[j]) {
                0.
            } else {
                gradient[j].abs()
            }
        })
        .fold(0., f64::max)
}

fn reset(inverse: &mut [f64; 49], n: usize) {
    inverse.fill(0.);
    for j in 0..n {
        inverse[j * n + j] = 1.;
    }
}

/// One bounded local search from one supplied seed. Proposal generation,
/// multi-start screening and absent-broad residual starts are separate layers.
pub fn optimize(
    problem: &Problem,
    initial: &[f64],
    options: &Options,
    w: &mut Workspace,
) -> Result<Report, String> {
    if options.max_iter == 0 || !options.gtol.is_finite() || options.gtol <= 0. {
        return Err("positive geometry tolerance and iteration budget required".into());
    }
    prepare(problem, initial, w)?;
    let p = initial.len();
    let n = 3 + 2 * ((p - 20) / 3);
    let ids = w.geometry_ids;
    let mut current = profile(problem, initial, options.inner, w)?;
    let mut accepted = w.theta;
    let mut gradient = w.gradient;
    let mut evaluations = 1;
    let mut inner_iterations = current.iterations;
    let mut inner_failures = usize::from(!current.converged);
    // Lateral coordinates use pixels; depth uses the represented defocus span.
    let mut scale = [1.; 7];
    scale[2] = problem.model.defocus_bounds[1] - problem.model.defocus_bounds[0];
    reset(&mut w.inverse, n);
    for iteration in 0..=options.max_iter {
        let kkt = if current.converged {
            projected(
                &accepted,
                &gradient,
                problem.lower,
                problem.upper,
                &ids[..n],
            )
        } else {
            f64::INFINITY
        };
        let report = |status| Report {
            objective: current.objective,
            geometry_kkt: kkt,
            inner_kkt: current.kkt,
            iterations: iteration,
            evaluations,
            inner_iterations,
            inner_failures,
            status,
        };
        w.theta = accepted;
        w.gradient = gradient;
        if !current.converged {
            return Ok(report("inner_failure"));
        }
        if kkt <= options.gtol {
            return Ok(report("converged"));
        }
        if iteration == options.max_iter {
            return Ok(report("iteration_limit"));
        }
        let mut g = [0.; 7];
        for j in 0..n {
            let id = ids[j];
            g[j] = gradient[id] * scale[j];
            if blocked(
                accepted[id],
                gradient[id],
                problem.lower[id],
                problem.upper[id],
            ) {
                g[j] = 0.;
            }
        }
        let mut found = None;
        // If the quasi-Newton direction fails, discard its metric and try the
        // projected gradient. Neither path reuses an unsuccessful inner gradient.
        for attempt in 0..2 {
            if attempt == 1 {
                reset(&mut w.inverse, n);
            }
            let mut direction = [0.; 7];
            for j in 0..n {
                direction[j] = -(0..n).map(|k| w.inverse[j * n + k] * g[k]).sum::<f64>();
                let id = ids[j];
                if blocked(
                    accepted[id],
                    gradient[id],
                    problem.lower[id],
                    problem.upper[id],
                ) || (accepted[id] <= problem.lower[id] + 1e-10 && direction[j] < 0.)
                    || (accepted[id] >= problem.upper[id] - 1e-10 && direction[j] > 0.)
                {
                    direction[j] = 0.;
                }
            }
            let largest = direction[..n].iter().fold(1.0_f64, |v, x| v.max(x.abs()));
            for d in &mut direction[..n] {
                *d /= largest;
            }
            let mut alpha = 1.;
            for _ in 0..30 {
                let mut trial = accepted;
                let mut descent = 0.;
                let mut distance = 0.0_f64;
                for j in 0..n {
                    let id = ids[j];
                    trial[id] = (accepted[id] + alpha * scale[j] * direction[j])
                        .clamp(problem.lower[id], problem.upper[id]);
                    let delta = trial[id] - accepted[id];
                    descent += gradient[id] * delta;
                    distance = distance.max(delta.abs());
                }
                if descent >= 0. || distance < 1e-14 {
                    break;
                }
                let result = profile(problem, &trial[..p], options.inner, w)?;
                evaluations += 1;
                inner_iterations += result.iterations;
                if !result.converged {
                    inner_failures += 1;
                }
                if result.converged
                    && result.objective <= current.objective + 1e-4 * descent + 1e-12
                {
                    found = Some(result);
                    break;
                }
                alpha *= 0.5;
            }
            if found.is_some() {
                break;
            }
        }
        let Some(next) = found else {
            w.theta = accepted;
            w.gradient = gradient;
            return Ok(Report {
                objective: current.objective,
                geometry_kkt: kkt,
                inner_kkt: current.kkt,
                iterations: iteration,
                evaluations,
                inner_iterations,
                inner_failures,
                status: "no_progress",
            });
        };
        let mut s = [0.; 7];
        let mut y = [0.; 7];
        let mut hy = [0.; 7];
        let mut face_changed = false;
        for j in 0..n {
            let id = ids[j];
            let old_blocked = blocked(
                accepted[id],
                gradient[id],
                problem.lower[id],
                problem.upper[id],
            );
            let new_blocked = blocked(
                w.theta[id],
                w.gradient[id],
                problem.lower[id],
                problem.upper[id],
            );
            face_changed |= old_blocked != new_blocked;
            if !old_blocked && !new_blocked {
                s[j] = (w.theta[id] - accepted[id]) / scale[j];
                y[j] = (w.gradient[id] - gradient[id]) * scale[j];
            }
        }
        let sy = (0..n).map(|j| s[j] * y[j]).sum::<f64>();
        let norm = (s[..n].iter().map(|v| v * v).sum::<f64>()
            * y[..n].iter().map(|v| v * v).sum::<f64>())
        .sqrt();
        if !face_changed && sy > 1e-10 * norm.max(1e-20) {
            for j in 0..n {
                hy[j] = (0..n).map(|k| w.inverse[j * n + k] * y[k]).sum();
            }
            let yhy = (0..n).map(|j| y[j] * hy[j]).sum::<f64>();
            for j in 0..n {
                for k in 0..n {
                    w.inverse[j * n + k] +=
                        (1. + yhy / sy) * s[j] * s[k] / sy - (s[j] * hy[k] + hy[j] * s[k]) / sy;
                }
            }
        } else {
            reset(&mut w.inverse, n);
        }
        accepted = w.theta;
        gradient = w.gradient;
        current = next;
    }
    unreachable!()
}

/// Recompute constrained stationarity at a raw fallback without moving it.
pub(crate) fn stationarity(
    problem: &Problem,
    theta: &[f64],
    w: &mut Workspace,
) -> Result<f64, String> {
    prepare(problem, theta, w)?;
    let p = theta.len();
    let count = (p - 20) / 3;
    let a = 17 + count;
    let mut initial = [0.; 19];
    problem
        .model
        .prepare_affine_into(theta, &mut w.model, &mut initial[..a])?;
    let linear = affine::Problem {
        mean: &w.model.affine,
        observations: problem.data,
        constraints: &w.constraints,
        rhs: &w.rhs,
    };
    let kkt = affine::stationarity(&linear, &initial[..a], &mut w.linear);
    // Unit-flux Jacobians from prepare_affine need the original amplitudes.
    w.gradient.fill(0.);
    for (i, &y) in problem.data.iter().enumerate() {
        let residual = 1. - y / w.linear.mean[i];
        for (k, &id) in w.geometry_ids[..3 + 2 * count].iter().enumerate() {
            let amplitude = if k < 3 {
                theta[16]
            } else {
                theta[20 + 3 * ((k - 3) / 2)]
            };
            w.gradient[id] += residual * w.model.jacobian[i * p + id] * amplitude;
        }
    }
    Ok(kkt.max(projected(
        theta,
        &w.gradient,
        problem.lower,
        problem.upper,
        &w.geometry_ids[..3 + 2 * count],
    )))
}
