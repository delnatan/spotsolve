//! Complete calibrated K=0/1/2 search. Rust owns proposals, screening, residual
//! restarts and nesting. Flat parameter records are separate from algorithms;
//! one numerical workspace is reused across all local fits.
use crate::{affine, geometry, inference, sparse};
use std::f64::consts::PI;

/// Experimental initial focused-center policy; never a count acceptance rule.
#[derive(Clone, Copy, Default)]
pub enum Proposals {
    #[default]
    Moments,
    Aguet {
        alpha: f64,
    },
}

pub struct Options {
    pub proposals: Proposals,
    pub max_iter: usize,
    pub screen_iter: usize,
    pub keep_screened: usize,
    pub gtol: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            proposals: Proposals::default(),
            max_iter: 400,
            screen_iter: 48,
            keep_screened: 4,
            gtol: 1e-6,
        }
    }
}

#[derive(Clone)]
struct Candidate {
    theta: [f64; 26],
    objective: f64,
    kkt: f64,
    status: &'static str,
}
pub struct CountFit {
    pub proposal_centres: Vec<[f64; 2]>,
    pub theta: Vec<f64>,
    pub objective: f64,
    pub kkt: f64,
    pub status: &'static str,
    pub boundary: Vec<usize>,
    pub starts: usize,
    pub evaluations: usize,
    pub inner_iterations: usize,
    pub inner_failures: usize,
}

/// Bounds are derived from the observed pixels, with signed background
/// coefficients. Sampled-background positivity is enforced by the inner solver.
pub fn bounds(
    model: &inference::Model,
    data: &[f64],
    count: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if count > 2
        || data.len() != model.shape[0] * model.shape[1]
        || data.iter().any(|v| !v.is_finite() || *v < 0.)
    {
        return Err("finite nonnegative data matching model and count 0/1/2 required".into());
    }
    let flux = (data.iter().sum::<f64>() * 5.).max(1000.) / 1000.;
    let background = (data.iter().copied().fold(0., f64::max) * 10.).max(100.) / 10.;
    if !flux.is_finite() || !background.is_finite() {
        return Err("data magnitude overflows parameter bounds".into());
    }
    let mut lo = vec![-background; 16];
    let mut hi = vec![background; 16];
    lo.extend([0., -0.5, -0.5, model.defocus_bounds[0]]);
    hi.extend([
        flux,
        model.shape[0] as f64 - 0.5,
        model.shape[1] as f64 - 0.5,
        model.defocus_bounds[1],
    ]);
    let [y0, x0, y1, x1] = model.focus_bounds;
    for _ in 0..count {
        lo.extend([0., y0, x0]);
        hi.extend([flux, y1, x1]);
    }
    Ok((lo, hi))
}

fn unique(points: &mut Vec<[f64; 2]>) {
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    points.dedup();
}
fn clip(point: [f64; 2], model: &inference::Model) -> [f64; 2] {
    let [y0, x0, y1, x1] = model.focus_bounds;
    [point[0].clamp(y0, y1), point[1].clamp(x0, x1)]
}
fn moments(data: &[f64], width: usize, background: f64) -> ([f64; 2], f64, f64) {
    let mut total = 0.;
    let mut centre = [0.; 2];
    let mut peak = 0;
    for (i, &y) in data.iter().enumerate() {
        if y > data[peak] {
            peak = i;
        }
        let v = (y - background).max(0.);
        total += v;
        centre[0] += v * (i / width) as f64;
        centre[1] += v * (i % width) as f64;
    }
    if total <= 1e-12 {
        return ([(peak / width) as f64, (peak % width) as f64], 0., 1.);
    }
    centre.iter_mut().for_each(|v| *v /= total);
    let (mut yy, mut yx, mut xx) = (0., 0., 0.);
    for (i, &y) in data.iter().enumerate() {
        let v = (y - background).max(0.);
        let dy = (i / width) as f64 - centre[0];
        let dx = (i % width) as f64 - centre[1];
        yy += v * dy * dy;
        yx += v * dy * dx;
        xx += v * dx * dx;
    }
    // Principal axis in (y,x); deterministic first eigenvector in the isotropic tie.
    let angle = if yy == xx && yx == 0. {
        PI / 2.
    } else {
        (0.5 * (2. * yx).atan2(xx - yy)).rem_euclid(PI)
    };
    (centre, angle, total)
}

/// Correlate an asymmetric calibrated response with score and information in
/// one pass. The observed image stays fixed; the tested emitter center moves.
fn residual_peak(shape: [usize; 2], data: &[f64], mean: &[f64], kernel: &[f64]) -> [f64; 2] {
    let [h, w] = shape;
    let kw = 2 * w - 1;
    let mut best = f64::NEG_INFINITY;
    let mut point = [0.; 2];
    let inverse: Vec<_> = mean.iter().map(|mu| 1. / mu).collect();
    let score: Vec<_> = data.iter().zip(&inverse).map(|(y, v)| y * v - 1.).collect();
    for cy in 0..h {
        for cx in 0..w {
            let (mut numerator, mut information) = (0., 0.);
            for py in 0..h {
                for px in 0..w {
                    let q = kernel[(h - 1 + py - cy) * kw + w - 1 + px - cx];
                    let i = py * w + px;
                    numerator += score[i] * q;
                    information += inverse[i] * q * q;
                }
            }
            let value = numerator / information.sqrt();
            if value > best {
                best = value;
                point = [cy as f64, cx as f64];
            }
        }
    }
    point
}

fn fit_count(
    model: &inference::Model,
    data: &[f64],
    count: usize,
    mut starts: Vec<[f64; 26]>,
    kernels: &[Vec<f64>; 3],
    options: &Options,
    w: &mut geometry::Workspace,
    render: &mut inference::Workspace,
) -> Result<CountFit, String> {
    let (lo, hi) = bounds(model, data, count)?;
    let p = lo.len();
    for theta in &mut starts {
        for j in 0..p {
            theta[j] = theta[j].clamp(lo[j], hi[j]);
        }
    }
    let problem = geometry::Problem {
        model,
        data,
        lower: &lo,
        upper: &hi,
    };
    let mut evaluations = 0;
    let mut inner_iterations = 0;
    let mut inner_failures = 0;
    let mut optimize = |theta: &[f64; 26], budget| -> Result<Candidate, String> {
        // Active affine faces can return roundoff just outside a box bound.
        // Every internally recycled seed needs the same projection as raw starts.
        let mut feasible = *theta;
        for j in 0..p {
            feasible[j] = feasible[j].clamp(lo[j], hi[j]);
        }
        let report = geometry::optimize(
            &problem,
            &feasible[..p],
            &geometry::Options {
                max_iter: budget,
                gtol: options.gtol,
                inner: affine::Options {
                    max_iter: 160,
                    tolerance: 1e-7_f64.min(options.gtol * 0.1),
                },
            },
            w,
        )?;
        evaluations += report.evaluations;
        inner_iterations += report.inner_iterations;
        inner_failures += report.inner_failures;
        Ok(Candidate {
            theta: w.theta,
            objective: report.objective,
            kkt: report.inner_kkt.max(report.geometry_kkt),
            status: report.status,
        })
    };
    let mut screened = starts
        .iter()
        .map(|s| optimize(s, options.screen_iter.min(options.max_iter)))
        .collect::<Result<Vec<_>, _>>()?;
    screened.sort_by(|a, b| a.objective.total_cmp(&b.objective));
    let mut refined = screened
        .iter()
        .take(options.keep_screened)
        .map(|s| optimize(&s.theta, options.max_iter))
        .collect::<Result<Vec<_>, _>>()?;
    if count > 0 {
        let best = screened
            .iter()
            .chain(&refined)
            .min_by(|a, b| a.objective.total_cmp(&b.objective))
            .unwrap()
            .theta;
        let mut absent = best;
        absent[16] = 0.;
        model.evaluate(&absent[..p], render)?;
        for (j, kernel) in kernels.iter().enumerate() {
            let point = residual_peak(model.shape, data, &render.mean, kernel);
            let mut seed = best;
            seed[17..19].copy_from_slice(&point);
            seed[19] = model.defocus_bounds[0]
                + j as f64 * 0.5 * (model.defocus_bounds[1] - model.defocus_bounds[0]);
            refined.push(optimize(&seed, options.max_iter)?);
        }
    }
    // Retain raw nested hypotheses even if a local optimizer loses accuracy.
    // Strict comparisons keep stable screening/raw/refinement tie ordering.
    let mut best = screened.remove(0);
    for theta in &starts {
        model.evaluate(&theta[..p], render)?;
        let objective = data
            .iter()
            .zip(&render.mean)
            .map(|(&y, &mu)| affine::poisson_term(y, mu))
            .sum();
        if objective < best.objective {
            best = Candidate {
                theta: *theta,
                objective,
                kkt: f64::NAN,
                status: "raw_start",
            };
        }
    }
    for candidate in refined {
        if candidate.objective < best.objective {
            best = candidate;
        }
    }
    if best.status == "raw_start" {
        best.kkt = geometry::stationarity(&problem, &best.theta[..p], w)?;
    }
    let boundary = (0..p)
        .filter(|&j| {
            (best.theta[j] - lo[j]).min(hi[j] - best.theta[j]) / (hi[j] - lo[j]).max(1.) < 1e-6
        })
        .collect();
    Ok(CountFit {
        proposal_centres: Vec::new(),
        theta: best.theta[..p].to_vec(),
        objective: best.objective,
        kkt: best.kkt,
        status: best.status,
        boundary,
        starts: starts.len() + if count > 0 { 3 } else { 0 },
        evaluations,
        inner_iterations,
        inner_failures,
    })
}

pub fn fit_component(
    model: &inference::Model,
    data: &[f64],
    candidate_centres: &[[f64; 2]],
    seed_sigma: f64,
    options: &Options,
    w: &mut geometry::Workspace,
    render: &mut inference::Workspace,
) -> Result<[CountFit; 3], String> {
    bounds(model, data, 0)?;
    if options.max_iter == 0
        || options.screen_iter == 0
        || options.keep_screened == 0
        || !options.gtol.is_finite()
        || options.gtol <= 0.
        || !seed_sigma.is_finite()
        || seed_sigma <= 0.
        || candidate_centres.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(
            "positive budgets, tolerance, seed scale and finite candidate centers required".into(),
        );
    }
    if let Proposals::Aguet { alpha } = options.proposals {
        if !alpha.is_finite() || !(0. < alpha && alpha < 1.) {
            return Err("proposal_alpha must be finite and between zero and one".into());
        }
        // Bound filter support before allocating kernels. Huge scales carry no
        // useful local-peak information on this patch.
        if seed_sigma > model.shape[0].max(model.shape[1]) as f64 {
            return Err("Aguet seed_sigma must not exceed the patch size".into());
        }
    }
    let [h, width] = model.shape;
    let [y0, x0, y1, x1] = model.focus_bounds;
    let mut peak = None;
    for (i, &y) in data.iter().enumerate() {
        let cy = (i / width) as f64;
        let cx = (i % width) as f64;
        if cy >= y0 && cy <= y1 && cx >= x0 && cx <= x1 && peak.is_none_or(|j| y > data[j]) {
            peak = Some(i);
        }
    }
    let peak = peak.ok_or("focus_bounds must contain at least one pixel center")?;
    let peak = [(peak / width) as f64, (peak % width) as f64];
    let mut sorted = data.to_vec();
    sorted.sort_by(f64::total_cmp);
    let t = (data.len() - 1) as f64 * 0.2;
    let left = t.floor() as usize;
    let right = t.ceil() as usize;
    let b0 = (sorted[left] + (t - left as f64) * (sorted[right] - sorted[left])).max(1e-3);
    let (moment, axis, excess) = moments(data, width, b0);
    let midpoint = [(h - 1) as f64 / 2., (width - 1) as f64 / 2.];
    let mut centres = vec![midpoint, moment, peak];
    unique(&mut centres);
    let depths = [
        model.defocus_bounds[0],
        (model.defocus_bounds[0] + model.defocus_bounds[1]) / 2.,
        model.defocus_bounds[1],
    ];
    let kernels = [
        model.broad_kernel(depths[0])?,
        model.broad_kernel(depths[1])?,
        model.broad_kernel(depths[2])?,
    ];
    let mut absent = [0.; 26];
    absent[..16].fill(b0 / 10.);
    absent[17..19].copy_from_slice(&midpoint);
    absent[19] = depths[0];
    let mut starts = Vec::new();
    for centre in centres {
        for z in depths {
            let mut theta = absent;
            theta[16] = excess.max(1.) / 1000.;
            theta[17..19].copy_from_slice(&centre);
            theta[19] = z;
            starts.push(theta);
        }
    }
    starts.push(absent);
    let zero = fit_count(model, data, 0, starts, &kernels, options, w, render)?;
    let mut points = match options.proposals {
        Proposals::Moments => vec![clip(peak, model), clip(moment, model)],
        Proposals::Aguet { alpha } => sparse::aguet_candidates(data, h, width, seed_sigma, alpha)
            .into_iter()
            .map(|candidate| {
                [
                    (candidate.index / width) as f64,
                    (candidate.index % width) as f64,
                ]
            })
            // Do not turn peaks outside the owned region into edge proposals.
            .filter(|point| *point == clip(*point, model))
            .collect(),
    };
    points.extend(candidate_centres.iter().map(|p| clip(*p, model)));
    unique(&mut points);
    let focus_flux =
        (sorted[sorted.len() - 1] - b0).max(1.) * 2. * PI * seed_sigma * seed_sigma / 1000.;
    if !focus_flux.is_finite() {
        return Err("seed scale overflows initial flux".into());
    }
    let mut initial = [0.; 26];
    initial[..20].copy_from_slice(&zero.theta);
    initial[21..23].copy_from_slice(&clip(peak, model));
    let mut starts = vec![initial];
    for point in &points {
        for nuisance in [&initial, &absent] {
            let mut theta = *nuisance;
            theta[20] = focus_flux;
            theta[21..23].copy_from_slice(point);
            starts.push(theta);
        }
    }
    let one = fit_count(model, data, 1, starts, &kernels, options, w, render)?;
    initial[..23].copy_from_slice(&one.theta);
    initial[23] = 0.;
    let centre = [one.theta[21], one.theta[22]];
    initial[24..26].copy_from_slice(&centre);
    let total = one.theta[20];
    let mut starts = vec![initial];
    for q in [0.5, 0.8] {
        for j in 0..4 {
            let angle = axis + j as f64 * PI / 4.;
            for &polarity in if q == 0.5 { &[1.][..] } else { &[1., -1.][..] } {
                for separation in [0.5, 1., 1.75] {
                    let vector = [
                        polarity * separation * seed_sigma * angle.sin(),
                        polarity * separation * seed_sigma * angle.cos(),
                    ];
                    let first = clip(
                        [
                            centre[0] + (1. - q) * vector[0],
                            centre[1] + (1. - q) * vector[1],
                        ],
                        model,
                    );
                    let second = clip(
                        [centre[0] - q * vector[0], centre[1] - q * vector[1]],
                        model,
                    );
                    let mut theta = initial;
                    theta[20] = total * q;
                    theta[23] = total * (1. - q);
                    theta[21..23].copy_from_slice(&first);
                    theta[24..26].copy_from_slice(&second);
                    starts.push(theta);
                }
            }
        }
    }
    for (j, point) in points.iter().enumerate() {
        for other in &points[j + 1..] {
            let mut theta = initial;
            theta[20] = total / 2.;
            theta[23] = total / 2.;
            theta[21..23].copy_from_slice(point);
            theta[24..26].copy_from_slice(other);
            starts.push(theta);
        }
    }
    let two = fit_count(model, data, 2, starts, &kernels, options, w, render)?;
    let mut fits = [zero, one, two];
    for fit in &mut fits {
        fit.proposal_centres = points.clone();
    }
    Ok(fits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(focus_bounds: [f64; 4]) -> inference::Model {
        let depth = [0., 0.2, 0.4, 0.6];
        let offsets = [-4., -2., 0., 2., 4.];
        let calibration =
            inference::Calibration::new(&depth, &offsets, vec![-4.; 8 * 9 * 9]).unwrap();
        inference::Model::new([3, 4], focus_bounds, [0.25, 0.55], calibration).unwrap()
    }

    #[test]
    fn asymmetric_residual_response_moves_the_center_opposite_the_offset() {
        // Response brightest one row below and one column left of its emitter.
        // A bright pixel at (1,2) therefore proposes an emitter at (0,3).
        // Rectangular/even image dimensions catch correlation alignment errors.
        let mut kernel = vec![0.01; 5 * 7];
        kernel[3 * 7 + 2] = 1.;
        let mut data = vec![1.; 12];
        data[6] = 10.;
        assert_eq!(residual_peak([3, 4], &data, &[1.; 12], &kernel), [0., 3.]);
    }

    #[test]
    fn weighted_axis_and_blank_image_have_deterministic_starts() {
        let mut data = vec![0.; 12];
        data[1] = 3.;
        data[9] = 3.;
        let (centre, axis, total) = moments(&data, 4, 0.);
        assert_eq!(centre, [1., 1.]);
        assert_eq!(total, 6.);
        assert!((axis - PI / 2.).abs() < 1e-12);
        assert_eq!(moments(&[0.; 12], 4, 0.), ([0., 0.], 0., 1.));
    }

    #[test]
    fn raw_zero_flux_boundary_is_certified_without_moving_the_seed() {
        let model = model([0., 0., 2., 3.]);
        let data = [0.; 12];
        let (lo, hi) = bounds(&model, &data, 1).unwrap();
        let mut theta = [0.; 23];
        theta[17] = 1.;
        theta[18] = 1.;
        theta[19] = 0.4;
        theta[21] = 1.;
        theta[22] = 1.;
        let problem = geometry::Problem {
            model: &model,
            data: &data,
            lower: &lo,
            upper: &hi,
        };
        let mut workspace = geometry::Workspace::default();
        let kkt = geometry::stationarity(&problem, &theta, &mut workspace).unwrap();
        assert!(kkt < 1e-8, "{kkt}");
        assert_eq!(theta[16], 0.);
        assert_eq!(theta[20], 0.);
    }

    #[test]
    fn component_rejects_focus_bounds_without_any_pixel_center() {
        let model = model([0.1, 0.1, 0.2, 0.2]);
        let result = fit_component(
            &model,
            &[1.; 12],
            &[],
            1.,
            &Options::default(),
            &mut geometry::Workspace::default(),
            &mut inference::Workspace::default(),
        );
        assert!(result.err().unwrap().contains("pixel center"));
    }
}
