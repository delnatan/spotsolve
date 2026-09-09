//! Observed Poisson curvature and conditional position uncertainty. The model
//! supplies compact source-block second derivatives; algorithms operate on flat
//! arrays and a separate reusable workspace, never the optimizer's BFGS metric.
use crate::{inference, linalg};

pub struct Workspace {
    pub model: inference::Workspace,
    pub gradient: Vec<f64>,
    pub hessian: Vec<f64>,
    pub covariance: Vec<f64>,
    pub fixed_covariance: Vec<f64>,
    scaled: [f64; 676],
    nuisance: [f64; 676],
    cross: [f64; 104],
    schur: [f64; 16],
    fixed: [f64; 16],
    rhs: [f64; 26],
    solution: [f64; 26],
    scale: [f64; 26],
    eigenvalues: Vec<f64>,
    eigenwork: Vec<f64>,
    chol: linalg::Chol,
}
impl Default for Workspace {
    fn default() -> Self {
        Self {
            model: inference::Workspace::default(),
            gradient: Vec::new(),
            hessian: Vec::new(),
            covariance: Vec::new(),
            fixed_covariance: Vec::new(),
            scaled: [0.; 676],
            nuisance: [0.; 676],
            cross: [0.; 104],
            schur: [0.; 16],
            fixed: [0.; 16],
            rhs: [0.; 26],
            solution: [0.; 26],
            scale: [0.; 26],
            eigenvalues: Vec::new(),
            eigenwork: Vec::new(),
            chol: linalg::Chol::new(26),
        }
    }
}

pub struct Report {
    pub status: &'static str,
    pub positions: usize,
    pub gradient_max: Option<f64>,
    pub broad_conditioned_absent: bool,
}

/// H = J^T diag(y/mu^2) J + sum_i (1-y_i/mu_i) * d2mu_i.
/// Only the second term has compact per-source support; the first retains all
/// source/background/neighbor coupling. No ridge, Fisher replacement or clipping.
pub fn observed(
    model: &inference::Model,
    data: &[f64],
    theta: &[f64],
    w: &mut Workspace,
) -> Result<(), String> {
    if data.len() != model.shape[0] * model.shape[1]
        || data.iter().any(|v| !v.is_finite() || *v < 0.)
    {
        return Err("data must be finite nonnegative pixels matching the model".into());
    }
    model.evaluate_second(theta, &mut w.model)?;
    let p = theta.len();
    let count = (p - 20) / 3;
    let stride = 9 + 5 * count;
    w.gradient.resize(p, 0.);
    w.gradient.fill(0.);
    w.hessian.resize(p * p, 0.);
    w.hessian.fill(0.);
    for (pixel, &y) in data.iter().enumerate() {
        let mu = w.model.mean[pixel];
        if mu <= 0. {
            return Err("Poisson mean must be positive".into());
        }
        let residual = 1. - y / mu;
        let weight = (y / mu) / mu;
        let row = &w.model.jacobian[pixel * p..(pixel + 1) * p];
        for i in 0..p {
            w.gradient[i] += row[i] * residual;
            for j in 0..=i {
                w.hessian[i * p + j] += row[i] * weight * row[j];
            }
        }
        let mut offset = pixel * stride;
        for source in 0..=count {
            let (base, n) = if source == 0 {
                (16, 4)
            } else {
                (20 + 3 * (source - 1), 3)
            };
            for i in 1..n {
                for j in 0..=i {
                    w.hessian[(base + i) * p + base + j] += residual * w.model.second[offset];
                    offset += 1;
                }
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            w.hessian[j * p + i] = w.hessian[i * p + j];
        }
    }
    if w.gradient.iter().chain(&w.hessian).any(|v| !v.is_finite()) {
        return Err("nonfinite observed Poisson curvature".into());
    }
    Ok(())
}

/// Conditional observed-Hessian covariance, matching the retained Python
/// boundary and scaled-eigenvalue guards. Recomputes stationarity from pixels.
/// A numerically absent broad component is conditioned absent, not marginalized.
/// The tiny-flux guard uses the geometry active-bound tolerance (1e-10 in
/// flux/1000 units), plus a per-pixel mean perturbation guard. Curvature and
/// stationarity are recomputed at zero flux; this is not a presence decision.
pub fn positions(
    model: &inference::Model,
    data: &[f64],
    theta: &[f64],
    lower: &[f64],
    upper: &[f64],
    w: &mut Workspace,
) -> Result<Report, String> {
    w.covariance.clear();
    w.fixed_covariance.clear();
    let count = inference::Model::count(theta)?;
    let p = theta.len();
    let r = 2 * count;
    if lower.len() != p
        || upper.len() != p
        || data.len() != model.shape[0] * model.shape[1]
        || data.iter().any(|v| !v.is_finite() || *v < 0.)
        || theta.iter().zip(lower).zip(upper).any(|((v, lo), hi)| {
            !v.is_finite()
                || !lo.is_finite()
                || !hi.is_finite()
                || lo > hi
                || *v < lo - 1e-8
                || *v > hi + 1e-8
        })
    {
        return Err("finite compatible data, parameters and bounds required".into());
    }
    for id in std::iter::once(16).chain((0..count).map(|k| 20 + 3 * k)) {
        if lower[id] < 0. || theta[id] < -1e-8 {
            return Err("source flux bounds must be nonnegative".into());
        }
    }
    // Canonicalize only numerical zero on a feasible absent-broad face.
    // The mean guard also protects calibrations with unusually large responses.
    let mut canonical = [0.; 26];
    canonical[..p].copy_from_slice(theta);
    let mut broad_conditioned_absent = count > 0 && lower[16] == 0. && theta[16] == 0.;
    if count > 0 && lower[16] == 0. && theta[16] != 0. && theta[16].abs() <= 1e-10 {
        model.evaluate(theta, &mut w.model)?;
        broad_conditioned_absent = w.model.mean.iter().enumerate().all(|(pixel, mu)| {
            let contribution = theta[16] * w.model.jacobian[pixel * p + 16];
            contribution.abs() <= 1e-10 * mu.abs().max(1.)
        });
    }
    if broad_conditioned_absent {
        canonical[16] = 0.;
    }
    let theta = &canonical[..p];
    let report = |status, gradient_max| Report {
        broad_conditioned_absent,
        status,
        positions: r,
        gradient_max,
    };
    if count == 0 {
        return Ok(report("no_focused_sources", None));
    }
    let mut ids = [0; 26];
    let mut n = 0;
    for j in 0..p {
        if broad_conditioned_absent && (16..20).contains(&j) {
            continue;
        }
        ids[n] = j;
        n += 1;
    }
    let min_background = model
        .background
        .iter()
        .map(|row| row.iter().zip(theta).map(|(b, a)| b * a).sum::<f64>())
        .fold(f64::INFINITY, f64::min);
    if min_background < -1e-8 {
        return Err("background rates must be nonnegative".into());
    }
    if min_background <= 1e-7 {
        return Ok(report("background_pixel_boundary", None));
    }
    if ids[..n]
        .iter()
        .any(|&j| (theta[j] - lower[j]).min(upper[j] - theta[j]) <= 1e-7)
    {
        return Ok(report("parameter_boundary", None));
    }
    observed(model, data, theta, w)?;
    let stationarity = (0..p)
        .map(|j| {
            let g = w.gradient[j];
            if (theta[j] <= lower[j] + 1e-10 && g > 0.) || (theta[j] >= upper[j] - 1e-10 && g < 0.)
            {
                0.
            } else {
                g.abs()
            }
        })
        .fold(0., f64::max);
    if stationarity > 1e-3 {
        return Ok(report("nonstationary_fit", Some(stationarity)));
    }
    for j in 0..n {
        let diag = w.hessian[ids[j] * p + ids[j]];
        if diag <= 0. {
            return Ok(report("nonpositive_curvature", Some(stationarity)));
        }
        w.scale[j] = diag.sqrt();
    }
    for i in 0..n {
        for j in 0..n {
            w.scaled[i * n + j] = w.hessian[ids[i] * p + ids[j]] / (w.scale[i] * w.scale[j]);
        }
    }
    if !linalg::sym_eigvals(&w.scaled[..n * n], n, &mut w.eigenvalues, &mut w.eigenwork)
        || w.eigenvalues[0] <= 1e-8
    {
        return Ok(report(
            "singular_or_nonpositive_curvature",
            Some(stationarity),
        ));
    }
    let mut position_ids = [0; 4];
    let mut nuisance_ids = [0; 26];
    let mut nr = 0;
    let mut nn = 0;
    for (i, &id) in ids[..n].iter().enumerate() {
        if id >= 20 && (id - 20) % 3 != 0 {
            position_ids[nr] = i;
            nr += 1;
        } else {
            nuisance_ids[nn] = i;
            nn += 1;
        }
    }
    for i in 0..nn {
        for j in 0..nn {
            w.nuisance[i * nn + j] = w.scaled[nuisance_ids[i] * n + nuisance_ids[j]];
        }
    }
    if !w.chol.factor(&w.nuisance[..nn * nn], nn) {
        return Ok(report(
            "singular_or_nonpositive_curvature",
            Some(stationarity),
        ));
    }
    for i in 0..r {
        for j in 0..nn {
            w.rhs[j] = w.scaled[nuisance_ids[j] * n + position_ids[i]];
        }
        w.chol.solve(&w.rhs[..nn], &mut w.solution[..nn]);
        w.cross[i * nn..(i + 1) * nn].copy_from_slice(&w.solution[..nn]);
    }
    for i in 0..r {
        for j in 0..r {
            w.fixed[i * r + j] = w.scaled[position_ids[i] * n + position_ids[j]];
            w.schur[i * r + j] = w.fixed[i * r + j]
                - (0..nn)
                    .map(|k| w.scaled[position_ids[i] * n + nuisance_ids[k]] * w.cross[j * nn + k])
                    .sum::<f64>();
        }
    }
    if !w.chol.factor(&w.schur[..r * r], r) {
        return Ok(report(
            "singular_or_nonpositive_curvature",
            Some(stationarity),
        ));
    }
    w.covariance.resize(r * r, 0.);
    for j in 0..r {
        w.rhs[..r].fill(0.);
        w.rhs[j] = 1.;
        w.chol.solve(&w.rhs[..r], &mut w.solution[..r]);
        for i in 0..r {
            w.covariance[i * r + j] =
                w.solution[i] / (w.scale[position_ids[i]] * w.scale[position_ids[j]]);
        }
    }
    if !w.chol.factor(&w.fixed[..r * r], r) {
        w.covariance.clear();
        return Ok(report(
            "singular_or_nonpositive_curvature",
            Some(stationarity),
        ));
    }
    w.fixed_covariance.resize(r * r, 0.);
    for j in 0..r {
        w.rhs[..r].fill(0.);
        w.rhs[j] = 1.;
        w.chol.solve(&w.rhs[..r], &mut w.solution[..r]);
        for i in 0..r {
            w.fixed_covariance[i * r + j] =
                w.solution[i] / (w.scale[position_ids[i]] * w.scale[position_ids[j]]);
        }
    }
    if w.covariance
        .iter()
        .chain(&w.fixed_covariance)
        .any(|v| !v.is_finite())
    {
        w.covariance.clear();
        w.fixed_covariance.clear();
        return Ok(report("nonfinite_covariance", Some(stationarity)));
    }
    Ok(report("conditional_observed_hessian", Some(stationarity)))
}
