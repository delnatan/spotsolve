//! Single-pass localization for sparse emitter fields.
//!
//! Candidate detection and independent one-emitter fits are deliberately kept
//! separate from the interacting-source search. This is a reference method for
//! fields where candidate fitting windows do not materially overlap.

use crate::{filters, linalg, lmcl, psf, statistics};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Width {
    Fixed,
    Fitted { lower_ratio: f64, upper_ratio: f64 },
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub sigma: f64,
    pub alpha: f64,
    pub width: Width,
    pub fit_radius_sigma: f64,
    pub max_iter: usize,
}

#[derive(Clone, Debug)]
pub struct Localization {
    pub y: f64,
    pub x: f64,
    pub flux: f64,
    pub sigma: f64,
    pub se_flux: f64,
    pub se_y: f64,
    pub se_x: f64,
    pub test_statistic: f64,
    pub p_value: f64,
    pub iterations: usize,
    pub status: &'static str,
}

pub struct SparseResult {
    pub localizations: Vec<Localization>,
    pub candidate_count: usize,
    pub background: f64,
    pub model: Vec<f64>,
    pub residual: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub index: usize,
    pub test_statistic: f64,
    pub p_value: f64,
}

pub struct Workspace {
    fit: lmcl::FitWorkspace,
    patch: Vec<f64>,
    halo: Vec<f64>,
    inverse_diagonal: [f64; 5],
    inverse_scratch: Vec<f64>,
    chol: linalg::Chol,
    factors: psf::Factors,
}

impl Workspace {
    pub fn new() -> Self {
        Self {
            fit: lmcl::FitWorkspace::new(),
            patch: Vec::new(),
            halo: Vec::new(),
            inverse_diagonal: [f64::NAN; 5],
            inverse_scratch: Vec::new(),
            chol: linalg::Chol::new(5),
            factors: psf::Factors::new(0, 0, 0),
        }
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

fn median(data: &[f64]) -> f64 {
    let mut values = data.to_vec();
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 0 {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    } else {
        values[n / 2]
    }
}

/// Aguet fixed-PSF significance test intersected with LoG local maxima.
///
/// This stage is independent of sparse fitting and can also propose sites to
/// a dense joint-model search. The joint model remains responsible for its
/// own source-count decisions.
pub fn aguet_candidates(
    data: &[f64],
    h: usize,
    w: usize,
    sigma: f64,
    alpha: f64,
) -> Vec<Candidate> {
    // Aguet's fixed-centre linear regression: A*g + c. The unnormalized
    // Gaussian is intentional because A is peak height in the noise-floor
    // hypothesis below.
    let radius = (4.0 * sigma).ceil() as usize;
    let kernel = (-(radius as isize)..=radius as isize)
        .map(|x| (-0.5 * (x as f64 / sigma).powi(2)).exp())
        .collect::<Vec<_>>();
    let ones = vec![1.0; 2 * radius + 1];
    let squared = data.iter().map(|value| value * value).collect::<Vec<_>>();
    let fg = filters::separable_filter(data, h, w, &kernel, &kernel, filters::Mode::Reflect);
    let fu = filters::separable_filter(data, h, w, &ones, &ones, filters::Mode::Reflect);
    let fu2 = filters::separable_filter(&squared, h, w, &ones, &ones, filters::Mode::Reflect);
    let n = ((2 * radius + 1) * (2 * radius + 1)) as f64;
    let gsum = kernel.iter().sum::<f64>().powi(2);
    let g2sum = kernel
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .powi(2);
    let denominator = g2sum - gsum * gsum / n;
    let c00 = 1.0 / denominator;
    let k_level = statistics::normal_quantile(1.0 - alpha / 2.0);

    // Curvature only establishes which location represents a significant
    // region. It does not decide significance.
    let log_response = filters::gaussian_laplace(data, h, w, sigma, filters::Mode::Reflect)
        .into_iter()
        .map(|value| -value / filters::log_kernel_l2(sigma))
        .collect::<Vec<_>>();
    let window = 2 * sigma.ceil() as usize + 1;
    let local_max = filters::maximum_filter(&log_response, h, w, window, filters::Mode::Reflect);
    let border = window;
    let mut output = log_response
        .iter()
        .zip(local_max)
        .enumerate()
        .filter_map(|(index, (&response, maximum))| {
            let y = index / w;
            let x = index % w;
            if response != maximum || y < border || y + border >= h || x < border || x + border >= w
            {
                return None;
            }
            let amplitude = (fg[index] - gsum * fu[index] / n) / denominator;
            let background = (fu[index] - amplitude * gsum) / n;
            let rss = (amplitude * amplitude * g2sum
                - 2.0 * amplitude * (fg[index] - background * gsum)
                + fu2[index]
                - 2.0 * background * fu[index]
                + n * background * background)
                .max(0.0);
            if rss <= 0.0 {
                return None;
            }
            let sigma_a2 = rss / (n - 3.0) * c00;
            let sigma_res = (rss / (n - 1.0)).sqrt();
            let se_sigma_c = sigma_res / (2.0 * (n - 1.0)).sqrt() * k_level;
            let sigma_c2 = se_sigma_c * se_sigma_c;
            let combined2 = (sigma_a2 + sigma_c2) / n;
            let degrees_of_freedom =
                (n - 1.0) * (sigma_a2 + sigma_c2).powi(2) / (sigma_a2.powi(2) + sigma_c2.powi(2));
            if combined2 <= 0.0 || !degrees_of_freedom.is_finite() {
                return None;
            }
            let statistic = (amplitude - sigma_res * k_level) / combined2.sqrt();
            let p_value = statistics::student_t_sf(statistic, degrees_of_freedom);
            (p_value < alpha).then_some(Candidate {
                index,
                test_statistic: statistic,
                p_value,
            })
        })
        .collect::<Vec<_>>();
    output.sort_by(|a, b| {
        b.test_statistic
            .total_cmp(&a.test_statistic)
            .then(a.index.cmp(&b.index))
    });
    output
}

fn uncertainty(
    fisher: &[f64],
    theta: &[f64],
    lower: &[f64],
    upper: &[f64],
    converged: bool,
    workspace: &mut Workspace,
) -> (&'static str, [f64; 3]) {
    let p = theta.len();
    if !converged {
        return ("optimizer_not_converged", [f64::NAN; 3]);
    }
    if theta
        .iter()
        .zip(lower)
        .zip(upper)
        .any(|((&value, &lo), &hi)| (value - lo).min(hi - value) / (hi - lo).max(1.0) < 1e-6)
    {
        return ("parameter_boundary", [f64::NAN; 3]);
    }
    if !workspace.chol.factor(fisher, p) {
        return ("singular_information", [f64::NAN; 3]);
    }
    workspace.chol.inv_diag(
        &mut workspace.inverse_diagonal[..p],
        &mut workspace.inverse_scratch,
    );
    let values = [
        workspace.inverse_diagonal[1],
        workspace.inverse_diagonal[2],
        workspace.inverse_diagonal[3],
    ];
    if values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return ("singular_information", [f64::NAN; 3]);
    }
    (
        "conditional_fisher",
        [values[0].sqrt(), values[1].sqrt(), values[2].sqrt()],
    )
}

fn fit_candidate(
    data: &[f64],
    shape: [usize; 2],
    candidate: usize,
    test_statistic: f64,
    p_value: f64,
    background: f64,
    options: Options,
    workspace: &mut Workspace,
) -> Localization {
    let [h, w] = shape;
    let cy = candidate / w;
    let cx = candidate % w;
    let sigma_hi = match options.width {
        Width::Fixed => options.sigma,
        Width::Fitted { upper_ratio, .. } => options.sigma * upper_ratio,
    };
    let radius = (options.fit_radius_sigma * sigma_hi).ceil() as usize;
    let y0 = cy.saturating_sub(radius);
    let x0 = cx.saturating_sub(radius);
    let y1 = (cy + radius + 1).min(h);
    let x1 = (cx + radius + 1).min(w);
    let ph = y1 - y0;
    let pw = x1 - x0;
    workspace.patch.clear();
    workspace.patch.reserve(ph * pw);
    for y in y0..y1 {
        workspace
            .patch
            .extend_from_slice(&data[y * w + x0..y * w + x1]);
    }
    workspace.halo.resize(ph * pw, 0.0);
    workspace.halo.fill(0.0);

    let local_y = (cy - y0) as f64;
    let local_x = (cx - x0) as f64;
    let peak = (data[candidate] - background).max(1.0);
    let flux = peak / psf::peak_factor(options.sigma);
    let max_pixel = workspace.patch.iter().copied().fold(0.0, f64::max);
    let flux_hi = (workspace.patch.iter().map(|v| v.max(0.0)).sum::<f64>() * 5.0).max(1000.0);
    let background_hi = (10.0 * max_pixel).max(10.0 * background).max(100.0);
    let position_radius = options.sigma.max(1.0);
    let y_lo = (local_y - position_radius).max(-0.5);
    let y_hi = (local_y + position_radius).min(ph as f64 - 0.5);
    let x_lo = (local_x - position_radius).max(-0.5);
    let x_hi = (local_x + position_radius).min(pw as f64 - 0.5);

    let (theta, lower, upper) = match options.width {
        Width::Fixed => (
            vec![background, flux, local_y, local_x],
            vec![1e-9, 0.0, y_lo, x_lo],
            vec![background_hi, flux_hi, y_hi, x_hi],
        ),
        Width::Fitted {
            lower_ratio,
            upper_ratio,
        } => (
            vec![background, flux, local_y, local_x, options.sigma],
            vec![1e-9, 0.0, y_lo, x_lo, options.sigma * lower_ratio],
            vec![
                background_hi,
                flux_hi,
                y_hi,
                x_hi,
                options.sigma * upper_ratio,
            ],
        ),
    };
    let bounds = lmcl::Bounds::new(&lower, &upper);
    let fit_options = lmcl::FitOpts {
        max_iter: options.max_iter,
        ..Default::default()
    };
    let info = match options.width {
        Width::Fixed => lmcl::fit(
            &mut workspace.fit,
            &theta,
            ph,
            pw,
            options.sigma,
            &workspace.patch,
            &bounds,
            Some(&workspace.halo),
            fit_options,
        ),
        Width::Fitted { .. } => lmcl::fit_var_sigma(
            &mut workspace.fit,
            &theta,
            ph,
            pw,
            &workspace.patch,
            &bounds,
            Some(&workspace.halo),
            fit_options,
        ),
    };
    let fitted = workspace.fit.theta().to_vec();
    let fisher = workspace.fit.fisher(fitted.len()).to_vec();
    let (status, se) = uncertainty(&fisher, &fitted, &lower, &upper, info.converged, workspace);
    Localization {
        y: fitted[2] + y0 as f64,
        x: fitted[3] + x0 as f64,
        flux: fitted[1],
        sigma: if fitted.len() == 5 {
            fitted[4]
        } else {
            options.sigma
        },
        se_flux: se[0],
        se_y: se[1],
        se_x: se[2],
        test_statistic,
        p_value,
        iterations: info.n_iter,
        status,
    }
}

pub fn localize(
    data: &[f64],
    shape: [usize; 2],
    options: Options,
    workspace: &mut Workspace,
) -> Result<SparseResult, String> {
    let [h, w] = shape;
    let valid_width = match options.width {
        Width::Fixed => true,
        Width::Fitted {
            lower_ratio,
            upper_ratio,
        } => {
            lower_ratio.is_finite()
                && upper_ratio.is_finite()
                && lower_ratio > 0.0
                && lower_ratio < 1.0
                && upper_ratio > 1.0
        }
    };
    if h < 3
        || w < 3
        || data.len() != h * w
        || data.iter().any(|v| !v.is_finite())
        || !options.sigma.is_finite()
        || options.sigma <= 0.0
        || !options.alpha.is_finite()
        || options.alpha <= 0.0
        || options.alpha >= 1.0
        || !options.fit_radius_sigma.is_finite()
        || options.fit_radius_sigma < 2.0
        || options.max_iter == 0
        || !valid_width
    {
        return Err("invalid sparse-localization image or options".into());
    }
    let background = median(data).max(1e-6);
    let peaks = aguet_candidates(data, h, w, options.sigma, options.alpha);
    let candidate_count = peaks.len();
    let localizations = peaks
        .into_iter()
        .map(|candidate| {
            fit_candidate(
                data,
                shape,
                candidate.index,
                candidate.test_statistic,
                candidate.p_value,
                background,
                options,
                workspace,
            )
        })
        .filter(|fit| fit.status == "conditional_fisher")
        .collect::<Vec<_>>();

    let amplitudes = localizations.iter().map(|v| v.flux).collect::<Vec<_>>();
    let ys = localizations.iter().map(|v| v.y).collect::<Vec<_>>();
    let xs = localizations.iter().map(|v| v.x).collect::<Vec<_>>();
    let sigmas = localizations.iter().map(|v| v.sigma).collect::<Vec<_>>();
    let mut model = vec![background; h * w];
    if !localizations.is_empty() {
        workspace.factors.ensure(h, w, localizations.len());
        let ay = (0..h).map(|v| v as f64).collect::<Vec<_>>();
        let ax = (0..w).map(|v| v as f64).collect::<Vec<_>>();
        let theta = match options.width {
            Width::Fixed => psf::pack(background, &amplitudes, &ys, &xs),
            Width::Fitted { .. } => psf::pack_var(background, &amplitudes, &ys, &xs, &sigmas),
        };
        match options.width {
            Width::Fixed => psf::model_ax(
                &theta,
                &ay,
                &ax,
                options.sigma,
                None,
                &mut workspace.factors,
                &mut model,
            ),
            Width::Fitted { .. } => {
                psf::model_var_sigma_ax(&theta, &ay, &ax, None, &mut workspace.factors, &mut model)
            }
        }
    }
    let residual = data.iter().zip(&model).map(|(d, m)| d - m).collect();
    Ok(SparseResult {
        localizations,
        candidate_count,
        background,
        model,
        residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(shape: [usize; 2], sigma: f64, flux: f64, y: f64, x: f64) -> Vec<f64> {
        let [h, w] = shape;
        let ay = (0..h).map(|v| v as f64).collect::<Vec<_>>();
        let ax = (0..w).map(|v| v as f64).collect::<Vec<_>>();
        let theta = psf::pack(4.0, &[flux], &[y], &[x]);
        let mut model = vec![0.0; h * w];
        let mut jacobian = vec![0.0; theta.len() * h * w];
        psf::model_and_jac_ax(
            &theta,
            &ay,
            &ax,
            sigma,
            None,
            &mut psf::Factors::new(h, w, 1),
            &mut model,
            &mut jacobian,
        );
        model
    }

    #[test]
    fn fixed_width_single_pass_recovers_an_isolated_emitter() {
        let shape = [25, 25];
        let data = render(shape, 1.2, 900.0, 12.2, 11.7);
        let result = localize(
            &data,
            shape,
            Options {
                sigma: 1.2,
                alpha: 0.05,
                width: Width::Fixed,
                fit_radius_sigma: 4.0,
                max_iter: 100,
            },
            &mut Workspace::new(),
        )
        .unwrap();
        assert_eq!(result.localizations.len(), 1);
        let fit = &result.localizations[0];
        assert!((fit.y - 12.2).abs() < 1e-5);
        assert!((fit.x - 11.7).abs() < 1e-5);
        assert!((fit.flux - 900.0).abs() < 1e-3);
        assert_eq!(fit.status, "conditional_fisher");
    }

    #[test]
    fn fitted_width_recovers_an_isolated_broad_gaussian() {
        let shape = [31, 31];
        let data = render(shape, 1.65, 1200.0, 15.25, 14.6);
        let result = localize(
            &data,
            shape,
            Options {
                sigma: 1.2,
                alpha: 0.05,
                width: Width::Fitted {
                    lower_ratio: 0.7,
                    upper_ratio: 2.0,
                },
                fit_radius_sigma: 4.0,
                max_iter: 180,
            },
            &mut Workspace::new(),
        )
        .unwrap();
        assert_eq!(result.localizations.len(), 1);
        let fit = &result.localizations[0];
        assert!((fit.sigma - 1.65).abs() < 1e-4);
        assert!((fit.y - 15.25).abs() < 1e-4);
        assert_eq!(fit.status, "conditional_fisher");
    }

    #[test]
    fn blank_image_has_no_candidates() {
        let result = localize(
            &[4.0; 81],
            [9, 9],
            Options {
                sigma: 1.2,
                alpha: 0.05,
                width: Width::Fixed,
                fit_radius_sigma: 4.0,
                max_iter: 100,
            },
            &mut Workspace::new(),
        )
        .unwrap();
        assert!(result.localizations.is_empty());
        assert!(result.residual.iter().all(|value| *value == 0.0));
    }
}
