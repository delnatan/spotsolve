//! Fixed-geometry Poisson inference. Flat problem data and reusable scratch are
//! separate from the algorithms; no PSF, Python, or detector dependencies.
use crate::linalg::Chol;

/// Pixel-major affine mean: offset + basis * coefficients.
#[derive(Default)]
pub struct AffineData {
    pub columns: usize,
    pub basis: Vec<f64>,
    pub offset: Vec<f64>,
}

/// Borrowed problem view. Constraints are row-major C*a >= rhs (including boxes).
pub struct Problem<'a> {
    pub mean: &'a AffineData,
    pub observations: &'a [f64],
    pub constraints: &'a [f64],
    pub rhs: &'a [f64],
}

#[derive(Clone, Copy)]
pub struct Options {
    pub tolerance: f64,
    pub max_iter: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            tolerance: 1e-7,
            max_iter: 160,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Status {
    Converged,
    IterationLimit,
    NoProgress,
    SingularFace,
}
impl Status {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::IterationLimit => "iteration_limit",
            Self::NoProgress => "no_progress",
            Self::SingularFace => "singular_face",
        }
    }
}

pub struct Report {
    pub objective: f64,
    pub kkt: f64,
    pub feasibility: f64,
    pub iterations: usize,
    pub hessian_evaluations: usize,
    pub line_search_evaluations: usize,
    pub status: Status,
}

/// One allocation per fit capacity. Coefficients and multipliers are outputs;
/// scratch is retained for successive geometries and images.
pub struct Workspace {
    pub coefficients: Vec<f64>,
    pub multipliers: Vec<f64>,
    // Accepted-point outputs reused by the geometry envelope calculation.
    pub(crate) mean: Vec<f64>,
    pub(crate) gradient: Vec<f64>,
    hessian: Vec<f64>,
    trial: Vec<f64>,
    direction: Vec<f64>,
    slack: Vec<f64>,
    active: Vec<usize>,
    q: Vec<f64>,
    r: Vec<f64>,
    tangent: Vec<f64>,
    reduced: Vec<f64>,
    scale: Vec<f64>,
    rhs: Vec<f64>,
    lambda: Vec<f64>,
    scratch: Vec<f64>,
    chol: Chol,
    dual: DualWorkspace,
}
impl Workspace {
    pub fn new(capacity: usize) -> Self {
        Self {
            coefficients: Vec::new(),
            multipliers: Vec::new(),
            mean: Vec::new(),
            gradient: Vec::new(),
            hessian: Vec::new(),
            trial: Vec::new(),
            direction: Vec::new(),
            slack: Vec::new(),
            active: Vec::with_capacity(capacity),
            q: Vec::new(),
            r: Vec::new(),
            tangent: Vec::new(),
            reduced: Vec::new(),
            scale: Vec::new(),
            rhs: Vec::new(),
            lambda: Vec::new(),
            scratch: Vec::new(),
            chol: Chol::new(capacity),
            dual: DualWorkspace::default(),
        }
    }
    fn size(&mut self, pixels: usize, n: usize, m: usize) {
        self.coefficients.resize(n, 0.0);
        self.multipliers.resize(m, 0.0);
        self.mean.resize(pixels, 0.0);
        self.gradient.resize(n, 0.0);
        self.hessian.resize(n * n, 0.0);
        self.trial.resize(n, 0.0);
        self.direction.resize(n, 0.0);
        self.slack.resize(m, 0.0);
        self.q.resize(n * n, 0.0);
        self.r.resize(n * n, 0.0);
        self.tangent.resize(n * n, 0.0);
        self.reduced.resize(n * n, 0.0);
        self.scale.resize(n, 0.0);
        self.rhs.resize(n, 0.0);
        self.lambda.resize(n, 0.0);
        self.scratch.resize(n, 0.0);
        self.dual.size(n, m);
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One saturated-constant-removed Poisson term, for validated nonnegative y
/// and positive mean. Shared by fitting and original-start scoring.
#[inline]
pub(crate) fn poisson_term(y: f64, mu: f64) -> f64 {
    if y == 0. {
        mu
    } else {
        let delta = (mu - y) / y;
        if delta.abs() < 0.5 {
            y * (delta - delta.ln_1p())
        } else {
            y * (y / mu).ln() + mu - y
        }
    }
}

/// Value-only pass for line search; no Jacobian or Hessian reconstruction.
pub fn value(data: &AffineData, y: &[f64], a: &[f64], mean: &mut [f64]) -> f64 {
    let n = data.columns;
    let mut value = 0.0;
    for i in 0..y.len() {
        let mu = data.offset[i] + dot(&data.basis[i * n..(i + 1) * n], a);
        if !mu.is_finite() || mu <= 0.0 {
            return f64::INFINITY;
        }
        mean[i] = mu;
        value += poisson_term(y[i], mu);
    }
    value
}

pub fn derivatives(data: &AffineData, y: &[f64], mean: &[f64], g: &mut [f64], h: &mut [f64]) {
    let n = data.columns;
    g.fill(0.0);
    h.fill(0.0);
    for i in 0..y.len() {
        let residual = 1.0 - y[i] / mean[i];
        let weight = (y[i] / mean[i]) / mean[i];
        let row = &data.basis[i * n..(i + 1) * n];
        for j in 0..n {
            g[j] += row[j] * residual;
            for k in 0..=j {
                h[j * n + k] += row[j] * weight * row[k];
            }
        }
    }
    for j in 0..n {
        for k in 0..j {
            h[k * n + j] = h[j * n + k];
        }
    }
}

// Twice-reorthogonalized Gram-Schmidt: at most 19 coordinates in this model.
// Active normals retain their original scaling for multiplier interpretation.
fn orthogonalize(v: &mut [f64], q: &[f64], rows: usize, n: usize) {
    for _ in 0..2 {
        for i in 0..rows {
            let row = &q[i * n..(i + 1) * n];
            let projection = dot(v, row);
            for j in 0..n {
                v[j] -= projection * row[j];
            }
        }
    }
}

fn face(problem: &Problem, w: &mut Workspace) -> bool {
    let n = problem.mean.columns;
    let k = w.active.len();
    if k == 0 {
        w.q.fill(0.0);
        for i in 0..n {
            w.q[i * n + i] = 1.0;
        }
        w.tangent.copy_from_slice(&w.q);
        return true;
    }
    w.r.fill(0.0);
    for i in 0..k {
        let row = &problem.constraints[w.active[i] * n..(w.active[i] + 1) * n];
        w.scratch.copy_from_slice(row);
        orthogonalize(&mut w.scratch, &w.q, i, n);
        let norm = dot(&w.scratch, &w.scratch).sqrt();
        if norm <= 1e-10 * dot(row, row).sqrt() {
            return false;
        }
        for j in 0..n {
            w.q[i * n + j] = w.scratch[j] / norm;
        }
        for j in 0..=i {
            w.r[i * n + j] = dot(row, &w.q[j * n..(j + 1) * n]);
        }
    }
    // Complete an orthonormal basis; tangent vectors follow the active normals.
    let mut rows = k;
    for axis in 0..n {
        if rows == n {
            break;
        }
        w.scratch.fill(0.0);
        w.scratch[axis] = 1.0;
        orthogonalize(&mut w.scratch, &w.q, rows, n);
        let norm = dot(&w.scratch, &w.scratch).sqrt();
        if norm > 1e-8 {
            for j in 0..n {
                w.q[rows * n + j] = w.scratch[j] / norm;
            }
            rows += 1;
        }
    }
    if rows != n {
        return false;
    }
    w.tangent[..(n - k) * n].copy_from_slice(&w.q[k * n..n * n]);
    true
}

fn certificate(problem: &Problem, w: &mut Workspace) -> (f64, f64, Option<usize>, f64) {
    let n = problem.mean.columns;
    let k = w.active.len();
    // R^T lambda = Q*g, since C_active = R*Q.
    for i in (0..k).rev() {
        let mut v = dot(&w.q[i * n..(i + 1) * n], &w.gradient);
        for j in i + 1..k {
            v -= w.r[j * n + i] * w.lambda[j];
        }
        w.lambda[i] = v / w.r[i * n + i];
    }
    w.multipliers.fill(0.0);
    let mut remove = None;
    let mut negative = 0.0;
    w.scratch.copy_from_slice(&w.gradient);
    for i in 0..k {
        if w.lambda[i] < negative {
            negative = w.lambda[i];
            remove = Some(i);
        }
        let id = w.active[i];
        // A nonnegative dual vector supplies an actual KKT certificate even
        // before the working set is correct. Do not declare success on step size.
        let lambda = w.lambda[i].max(0.0);
        w.multipliers[id] = lambda;
        for j in 0..n {
            w.scratch[j] -= lambda * problem.constraints[id * n + j];
        }
    }
    let mut kkt = w.scratch.iter().fold(0.0_f64, |v, x| v.max(x.abs()));
    let mut feasibility = 0.0_f64;
    for i in 0..problem.rhs.len() {
        w.slack[i] =
            dot(&problem.constraints[i * n..(i + 1) * n], &w.coefficients) - problem.rhs[i];
        feasibility = feasibility.max(-w.slack[i]);
        kkt = kkt.max((w.multipliers[i] * w.slack[i]).abs());
    }
    let tangent_residual = (k..n)
        .map(|i| dot(&w.q[i * n..(i + 1) * n], &w.gradient).abs())
        .fold(0.0_f64, f64::max);
    (kkt.max(feasibility), feasibility, remove, tangent_residual)
}

/// NNLS on active constraint normals supplies a certificate at degenerate
/// vertices, where one independent working face need not span the normal cone.
/// Storage and iteration are independent of the primal Newton working set.
#[derive(Default)]
struct DualWorkspace {
    ids: Vec<usize>,
    q: Vec<f64>,
    r: Vec<f64>,
    weights: Vec<f64>,
    candidate: Vec<f64>,
    residual: Vec<f64>,
    scratch: Vec<f64>,
    norms: Vec<f64>,
}

impl DualWorkspace {
    fn size(&mut self, n: usize, m: usize) {
        self.ids.clear();
        self.ids.reserve(n);
        self.q.resize(n * n, 0.0);
        self.r.resize(n * n, 0.0);
        self.weights.resize(n, 0.0);
        self.candidate.resize(n, 0.0);
        self.residual.resize(n, 0.0);
        self.scratch.resize(n, 0.0);
        self.norms.resize(m, 0.0);
    }
}

fn cone_certificate(
    problem: &Problem,
    g: &[f64],
    slack: &[f64],
    out: &mut [f64],
    w: &mut DualWorkspace,
    tolerance: f64,
) -> f64 {
    let n = problem.mean.columns;
    let m = problem.rhs.len();
    w.ids.clear();
    for i in 0..m {
        w.norms[i] = dot(
            &problem.constraints[i * n..(i + 1) * n],
            &problem.constraints[i * n..(i + 1) * n],
        )
        .sqrt();
    }
    out.fill(0.0);
    w.residual.copy_from_slice(g);
    for _ in 0..10 * (m + n) {
        let mut best = None;
        let mut score = tolerance * 0.01;
        for i in 0..m {
            if slack[i] > 1e-8 || w.norms[i] == 0.0 || w.ids.contains(&i) {
                continue;
            }
            let row = &problem.constraints[i * n..(i + 1) * n];
            let correlation = dot(row, &w.residual) / w.norms[i];
            if correlation > score {
                w.scratch.copy_from_slice(row);
                orthogonalize(&mut w.scratch, &w.q, w.ids.len(), n);
                if dot(&w.scratch, &w.scratch).sqrt() > 1e-10 * w.norms[i] {
                    best = Some(i);
                    score = correlation;
                }
            }
        }
        let Some(id) = best else {
            break;
        };
        if w.ids.len() == n {
            break;
        }
        w.weights[w.ids.len()] = 0.0;
        w.ids.push(id);
        // Solve unconstrained least squares on the passive normals, moving
        // toward it only until a multiplier hits zero; remove and refactor.
        for _ in 0..2 * n + 1 {
            let k = w.ids.len();
            w.r.fill(0.0);
            for i in 0..k {
                let row = &problem.constraints[w.ids[i] * n..(w.ids[i] + 1) * n];
                for j in 0..n {
                    w.scratch[j] = row[j] / w.norms[w.ids[i]];
                }
                orthogonalize(&mut w.scratch, &w.q, i, n);
                let norm = dot(&w.scratch, &w.scratch).sqrt();
                for j in 0..n {
                    w.q[i * n + j] = w.scratch[j] / norm;
                }
                for j in 0..=i {
                    w.r[i * n + j] = dot(row, &w.q[j * n..(j + 1) * n]) / w.norms[w.ids[i]];
                }
            }
            for i in (0..k).rev() {
                let mut v = dot(&w.q[i * n..(i + 1) * n], g);
                for j in i + 1..k {
                    v -= w.r[j * n + i] * w.candidate[j];
                }
                w.candidate[i] = v / w.r[i * n + i];
            }
            if w.candidate[..k].iter().all(|v| *v > 0.0) {
                w.weights[..k].copy_from_slice(&w.candidate[..k]);
                break;
            }
            let mut alpha = 1.0_f64;
            let mut remove = 0;
            for i in 0..k {
                if w.candidate[i] <= 0.0 {
                    let ratio =
                        w.weights[i] / (w.weights[i] - w.candidate[i]).max(f64::MIN_POSITIVE);
                    if ratio <= alpha {
                        alpha = ratio;
                        remove = i;
                    }
                }
            }
            for i in 0..k {
                w.weights[i] += alpha * (w.candidate[i] - w.weights[i]);
            }
            w.ids.remove(remove);
            for i in remove..k - 1 {
                w.weights[i] = w.weights[i + 1];
            }
        }
        w.residual.copy_from_slice(g);
        out.fill(0.0);
        for (j, &id) in w.ids.iter().enumerate() {
            out[id] = w.weights[j].max(0.0) / w.norms[id];
            for i in 0..n {
                w.residual[i] -= out[id] * problem.constraints[id * n + i];
            }
        }
        if w.residual.iter().all(|v| v.abs() <= tolerance * 0.1) {
            break;
        }
    }
    let mut kkt = w.residual.iter().fold(0.0_f64, |v, x| v.max(x.abs()));
    for i in 0..m {
        kkt = kkt.max(-slack[i]).max((out[i] * slack[i]).abs());
    }
    kkt
}

/// Certificate at an untouched feasible seed, without optimizing its amplitudes.
/// Used only when a raw nested hypothesis wins the multi-start search.
pub(crate) fn stationarity(problem: &Problem, initial: &[f64], w: &mut Workspace) -> f64 {
    let n = problem.mean.columns;
    w.size(problem.observations.len(), n, problem.rhs.len());
    value(problem.mean, problem.observations, initial, &mut w.mean);
    w.gradient.fill(0.);
    for (i, &y) in problem.observations.iter().enumerate() {
        for j in 0..n {
            w.gradient[j] += problem.mean.basis[i * n + j] * (1. - y / w.mean[i]);
        }
    }
    for i in 0..problem.rhs.len() {
        w.slack[i] = dot(&problem.constraints[i * n..(i + 1) * n], initial) - problem.rhs[i];
    }
    cone_certificate(
        problem,
        &w.gradient,
        &w.slack,
        &mut w.multipliers,
        &mut w.dual,
        1e-7,
    )
}

/// Feasible active-face damped Newton. All iterations stay in Rust. The only
/// ridge is a dimensionless step safeguard, never part of likelihood/curvature.
/// Rank-deficient problems need agreement of mean, objective and KKT, not a
/// unique coefficient vector. A failed stationarity check is an explicit status.
pub fn solve(
    problem: &Problem,
    initial: &[f64],
    options: Options,
    w: &mut Workspace,
) -> Result<Report, String> {
    let n = problem.mean.columns;
    let pixels = problem.observations.len();
    let m = problem.rhs.len();
    if n == 0
        || n * n > w.chol.capacity()
        || initial.len() != n
        || pixels == 0
        || problem.mean.basis.len() != pixels * n
        || problem.mean.offset.len() != pixels
        || problem.constraints.len() != m * n
        || !options.tolerance.is_finite()
        || options.tolerance <= 0.0
        || options.max_iter == 0
        || initial
            .iter()
            .chain(&problem.mean.basis)
            .chain(&problem.mean.offset)
            .chain(problem.constraints)
            .chain(problem.rhs)
            .any(|x| !x.is_finite())
        || problem
            .observations
            .iter()
            .any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err("incompatible/nonfinite affine problem or solver options".into());
    }
    w.size(pixels, n, m);
    w.coefficients.copy_from_slice(initial);
    w.active.clear();
    for i in 0..m {
        let slack = dot(&problem.constraints[i * n..(i + 1) * n], initial) - problem.rhs[i];
        if slack < -1e-8 {
            return Err("affine solve requires a feasible initial point".into());
        }
    }
    let mut objective = value(
        problem.mean,
        problem.observations,
        &w.coefficients,
        &mut w.mean,
    );
    if !objective.is_finite() {
        return Err("initial Poisson mean must be finite and positive".into());
    }
    let mut derivatives_dirty = true;
    let mut hessian_evaluations = 0;
    let mut line_search_evaluations = 0;
    for iteration in 0..=options.max_iter {
        // Adding/releasing a face without moving coefficients changes neither
        // gradient nor Hessian. Keep those pixel reductions cached as well.
        if derivatives_dirty {
            derivatives(
                problem.mean,
                problem.observations,
                &w.mean,
                &mut w.gradient,
                &mut w.hessian,
            );
            hessian_evaluations += 1;
            derivatives_dirty = false;
        }
        if !face(problem, w) {
            return Err("dependent active constraint normals".into());
        }
        let (mut kkt, feasibility, remove, tangent_residual) = certificate(problem, w);
        // Only invoke the full normal-cone calculation at nearly stationary
        // faces; interior iterations need no dual solve or constraint factorization.
        if kkt > options.tolerance && tangent_residual <= options.tolerance {
            kkt = cone_certificate(
                problem,
                &w.gradient,
                &w.slack,
                &mut w.multipliers,
                &mut w.dual,
                options.tolerance,
            );
        }
        let report = move |status| Report {
            objective,
            kkt,
            feasibility,
            iterations: iteration,
            hessian_evaluations,
            line_search_evaluations,
            status,
        };
        if !kkt.is_finite() || w.gradient.iter().chain(&w.hessian).any(|x| !x.is_finite()) {
            return Err("nonfinite Poisson derivatives".into());
        }
        if kkt <= options.tolerance {
            return Ok(report(Status::Converged));
        }
        if iteration == options.max_iter {
            return Ok(report(Status::IterationLimit));
        }
        if tangent_residual <= options.tolerance {
            if let Some(i) = remove {
                w.active.remove(i);
                continue;
            }
        }
        let d = n - w.active.len();
        if d == 0 {
            return Ok(report(Status::NoProgress));
        }
        // T H T^T; diagonal scaling in tangent coordinates protects the solve
        // from mixed brightness/background units without changing its solution.
        if d == n {
            // Interior fits need no null-space construction or congruence.
            w.reduced.copy_from_slice(&w.hessian);
            for i in 0..n {
                w.rhs[i] = -w.gradient[i];
            }
        } else {
            for i in 0..d {
                for col in 0..n {
                    w.scratch[col] = (0..n)
                        .map(|j| w.tangent[i * n + j] * w.hessian[j * n + col])
                        .sum();
                }
                for j in 0..=i {
                    let v = dot(&w.scratch, &w.tangent[j * n..(j + 1) * n]);
                    w.reduced[i * d + j] = v;
                    w.reduced[j * d + i] = v;
                }
                w.rhs[i] = -dot(&w.tangent[i * n..(i + 1) * n], &w.gradient);
            }
        }
        for i in 0..d {
            w.scale[i] = w.reduced[i * d + i].max(1e-12).sqrt();
        }
        for i in 0..d {
            for j in 0..d {
                w.reduced[i * d + j] /= w.scale[i] * w.scale[j];
            }
            w.reduced[i * d + i] += 1e-10;
            w.rhs[i] /= w.scale[i];
        }
        if !w.chol.factor(&w.reduced[..d * d], d) {
            return Ok(report(Status::SingularFace));
        }
        w.chol.solve_in_place(&mut w.rhs[..d]);
        if d == n {
            for j in 0..n {
                w.direction[j] = w.rhs[j] / w.scale[j];
            }
        } else {
            for j in 0..n {
                w.direction[j] = (0..d)
                    .map(|i| w.tangent[i * n + j] * w.rhs[i] / w.scale[i])
                    .sum();
            }
        }
        // Newton curvature may vanish (all observations zero) or be singular.
        // Rescaling a direction leaves its descent and blocking face unchanged,
        // and avoids enormous trial steps amplifying roundoff in tangent normals.
        let largest = w.direction.iter().fold(0.0_f64, |v, x| v.max(x.abs()));
        let radius = w.coefficients.iter().fold(1.0_f64, |v, x| v.max(x.abs()));
        if largest > radius {
            for step in &mut w.direction {
                *step *= radius / largest;
            }
        }
        let descent = dot(&w.gradient, &w.direction);
        if !descent.is_finite() || descent >= 0.0 {
            return Ok(report(Status::NoProgress));
        }
        let mut alpha = 1.0_f64;
        let mut blocker = None;
        for i in 0..m {
            if w.active.contains(&i) {
                continue;
            }
            let row = &problem.constraints[i * n..(i + 1) * n];
            let change = dot(row, &w.direction);
            if change < -1e-12 {
                let limit = w.slack[i].max(0.0) / -change;
                if limit < alpha {
                    alpha = limit;
                    blocker = Some(i);
                }
            }
        }
        if let Some(i) = blocker {
            w.active.push(i);
            let independent = face(problem, w);
            w.active.pop();
            if !independent {
                return Ok(report(Status::SingularFace));
            }
        }
        let mut accepted = false;
        let boundary_alpha = alpha;
        for _ in 0..40 {
            for j in 0..n {
                w.trial[j] = w.coefficients[j] + alpha * w.direction[j];
            }
            line_search_evaluations += 1;
            let trial_value = value(problem.mean, problem.observations, &w.trial, &mut w.mean);
            if trial_value.is_finite() && trial_value <= objective + 1e-4 * alpha * descent + 1e-12
            {
                derivatives_dirty = w.coefficients != w.trial;
                w.coefficients.copy_from_slice(&w.trial);
                objective = trial_value;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            // Restore statistics to the accepted point for the returned report.
            value(
                problem.mean,
                problem.observations,
                &w.coefficients,
                &mut w.mean,
            );
            let mut result = report(Status::NoProgress);
            result.line_search_evaluations = line_search_evaluations;
            return Ok(result);
        }
        if alpha == boundary_alpha {
            if let Some(i) = blocker {
                w.active.push(i);
            }
        }
    }
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pixel_boundary_has_a_nonzero_gradient_and_positive_multiplier() {
        let mean = AffineData {
            columns: 2,
            basis: vec![1., 0., 1., 1., 1., 2.],
            offset: vec![1.; 3],
        };
        let problem = Problem {
            mean: &mean,
            observations: &[0., 0., 12.],
            constraints: &mean.basis,
            rhs: &[0.; 3],
        };
        let mut w = Workspace::new(2);
        let report = solve(&problem, &[1., 1.], Options::default(), &mut w).unwrap();
        assert_eq!(
            report.status,
            Status::Converged,
            "{} {:?}",
            report.kkt,
            w.coefficients
        );
        assert!(w.coefficients[0].abs() < 1e-8);
        assert!((w.coefficients[1] - 3.5).abs() < 1e-7);
        assert!((w.multipliers[0] - 1.5).abs() < 1e-7);
    }
    #[test]
    fn rank_deficiency_does_not_require_unique_coefficients() {
        let mean = AffineData {
            columns: 2,
            basis: vec![1., 1., 1., 1.],
            offset: vec![1.; 2],
        };
        let problem = Problem {
            mean: &mean,
            observations: &[5., 5.],
            constraints: &[1., 0., 0., 1.],
            rhs: &[0.; 2],
        };
        let mut w = Workspace::new(2);
        let report = solve(&problem, &[1., 1.], Options::default(), &mut w).unwrap();
        assert_eq!(report.status, Status::Converged);
        assert!((w.coefficients.iter().sum::<f64>() - 4.).abs() < 1e-7);
    }
    #[test]
    fn zero_counts_and_redundant_constraints() {
        let mean = AffineData {
            columns: 2,
            basis: vec![1., 0., 0., 1.],
            offset: vec![1e-4; 2],
        };
        let problem = Problem {
            mean: &mean,
            observations: &[0.; 2],
            constraints: &[1., 0., 0., 1., 1., 1., 2., 2.],
            rhs: &[0.; 4],
        };
        let mut w = Workspace::new(2);
        let report = solve(&problem, &[1., 1.], Options::default(), &mut w).unwrap();
        assert_eq!(
            report.status,
            Status::Converged,
            "{} {:?}",
            report.kkt,
            w.coefficients
        );
        assert!((report.objective - 2e-4).abs() < 1e-10);
        assert!(w.coefficients.iter().all(|a| a.abs() < 1e-10));
        let pointer = w.hessian.as_ptr();
        solve(&problem, &[2., 3.], Options::default(), &mut w).unwrap();
        assert_eq!(pointer, w.hessian.as_ptr());
    }
    #[test]
    fn boundary_start_can_leave_the_face_and_upper_bounds_are_enforced() {
        let mean = AffineData {
            columns: 2,
            basis: vec![1., 0., 0., 1.],
            offset: vec![1.; 2],
        };
        let problem = Problem {
            mean: &mean,
            observations: &[4., 20.],
            constraints: &[1., 0., 0., 1., -1., 0., 0., -1.],
            rhs: &[0., 0., -5., -5.],
        };
        let mut w = Workspace::new(2);
        let report = solve(&problem, &[0., 0.], Options::default(), &mut w).unwrap();
        assert_eq!(report.status, Status::Converged);
        assert!((w.coefficients[0] - 3.).abs() < 1e-7);
        assert!((w.coefficients[1] - 5.).abs() < 1e-7);
        assert!(w.multipliers[3] > 0.);
    }
    #[test]
    fn invalid_and_exhausted_problems_are_not_successes() {
        let mean = AffineData {
            columns: 1,
            basis: vec![1.],
            offset: vec![1.],
        };
        let problem = Problem {
            mean: &mean,
            observations: &[50.],
            constraints: &[1.],
            rhs: &[0.],
        };
        let mut w = Workspace::new(1);
        assert!(solve(&problem, &[-1.], Options::default(), &mut w).is_err());
        let report = solve(
            &problem,
            &[1.],
            Options {
                max_iter: 1,
                ..Options::default()
            },
            &mut w,
        )
        .unwrap();
        assert_eq!(report.status, Status::IterationLimit);
        assert!(report.kkt > 1e-7);
    }
}
