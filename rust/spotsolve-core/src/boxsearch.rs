//! Group-based spot detection in ADU above the camera offset.
//!
//! 1. Estimate local noise and find LoG candidates.
//! 2. Estimate a smooth background from pixels outside candidate support.
//! 3. Group nearby candidates; fit each group against fixed neighboring light.
//! 4. Refine the selected emitters jointly and classify their fitted widths.
//!
//! `Selection::Fixed` accepts additions/removals by a fixed deviance cost.
//! `Selection::Bic` scores counts along forward and backward paths, then
//! rechecks removals after refinement. See docs/COUNT_SELECTION.md.
//!
//! During a count comparison, pixels, background shape, neighboring light
//! and dispersion stay fixed. Only the group's emitters and background level
//! are fitted. Widths float within `slack`; there is no width prior.
//!
//! Historical measurements and rejected approaches are preserved in
//! `docs/archive/DETECTOR_DESIGN_NOTES.md`.

use crate::filters::{self, Mode};
use crate::grid::EmitterGrid;
use crate::linalg::Chol;
use crate::lmcl::{self, Bounds, FitOpts, FitWorkspace};
use crate::patches::{self, HALO_FACTOR};
use crate::psf;
use crate::render;
use crate::statistics;

/// Allowed fitted widths, as multiples of `sigma`. Wider-than-reportable
/// sources remain in the model so their light does not become extra spots.
pub const SLACK: (f64, f64) = (0.70, 2.2);
/// Widths reported as detections, as multiples of `sigma`: the REPORTING
/// BAND, a downstream contract about what the caller is handed. Its upper
/// edge is how far out of focus an emitter may still be reported: a point
/// source images at 1.26x the in-focus width at |z| = 0.25 um, 1.95x at 0.35
/// and 3.2x at 0.50, so 2.0 is |z| < ~0.36 um.
pub const BAND: (f64, f64) = (0.80, 2.0);
/// Most emitters one box fits jointly. `patches::K_MAX`.
pub const K_MAX: usize = crate::patches::K_MAX;

/// Base addition/removal cost in dispersion-scaled I-divergence for fixed
/// selection. The actual cost is `(ADD_NATS + count_penalty) * phi`.
pub const ADD_NATS: f64 = 10.0;
/// Maximum proposal distance from an owned candidate, in units of sigma.
/// Ownership also excludes pixels closer to another group's candidate.
pub const OWN_RADIUS: f64 = 3.0;
/// Search sweeps. Later sweeps use the neighbors fitted in earlier visits.
pub const SWEEPS: usize = 2;
/// Search-fit objective tolerance, in nats.
pub const FIT_TOL_OBJ: f64 = 1e-6;
pub const FIT_MAX_ITER: usize = 100;
/// Most sweeps of the polish; see `polish` for why the queue does not
/// drain on its own.
pub const POLISH_SWEEPS: usize = 4;
/// Iteration budget per refinement fit; reaching it is not convergence.
pub const POLISH_MAX_ITER: usize = 50;
/// Refinement-fit objective tolerance, in nats.
pub const POLISH_TOL_OBJ: f64 = 1e-6;
/// px. An emitter that moved less than this in a polish sweep does not
/// dirty the groups that read it.
pub const POLISH_MOVE_TOL: f64 = 1e-3;
/// Side, px, of the window [`background_map`] averages over.
pub const BG_KERNEL: usize = 25;
/// ADU. `W = 1/m` is singular at `m = 0`; this is far below one count.
pub const BG_FLOOR: f64 = 1e-3;
/// sigma. Emitter support excluded from the background estimate.
pub const BG_MASK_RADIUS: f64 = 3.0;
/// Unmasked pixels a window needs before its local mean is believed.
pub const BG_MIN_PIXELS: f64 = 25.0;
/// LoG threshold in local noise standard deviations, used for both initial
/// candidates and residual proposals. Passing it proposes a fit; count
/// selection still decides whether to retain the emitter.
pub const PEAK_Z: f64 = 2.75;
/// Numerical amplitude floor: `max(A_MIN, A_MIN_REL * A_max)`.
/// The relative term keeps position/width information from becoming singular
/// at near-zero flux. This is a numerical guard, not a detection threshold.
pub const A_MIN: f64 = 1e-4;
pub const A_MIN_REL: f64 = 1e-6;
/// sigma. An out-of-band fit this near the frame border is `Edge`, not a
/// width flag. On beads_60x_still (in-focus beads on the coverslip, sigma0
/// 1.0 px) the fits the border cut off sat 0-0.5 px from it; the two narrow
/// interior fits sat 1.9 px and further in.
pub const EDGE_MARGIN: f64 = 1.0;
/// px. The window of [`noise_map`]'s two local medians, tied to the
/// background's: both describe the frame at the scale haze varies on.
pub const NOISE_WIN: usize = BG_KERNEL;
/// Spacing of the exact median grid, in pixels; interpolate between nodes.
pub const NOISE_STRIDE: usize = 12;
/// ADU^2. Integer-valued camera data can not be less variable than its own
/// quantization, `1/12`. Only a noise-free or constant image reaches it.
pub const NOISE_VAR_FLOOR: f64 = 1.0 / 12.0;
/// Width-reporting tolerance in standard errors. An out-of-band width at
/// the upper fitting bound is too wide, regardless of its uncertainty.
pub const BAND_Z: f64 = 2.0;
/// Relative. A width within this fraction of `SLACK.1 * sigma` is on the bound.
pub const BOUND_TOL: f64 = 1e-3;

/// What a fitted emitter is reported as, by [`classify`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Class {
    /// Width inside the reporting band: a detection.
    Focus = 0,
    /// Narrower than the band, away from the border.
    Narrow = 1,
    /// Wider than the band, away from the border.
    Wide = 2,
    /// Out of band within `EDGE_MARGIN` sigma of the border, which cuts it:
    /// not an interior width measurement at all.
    Edge = 3,
}

/// Experimental count selection. BIC is a penalized, dispersion-scaled
/// deviance score here, not a calibrated Bayes factor or false-positive rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Fixed,
    Bic,
}

/// What a caller chooses per frame.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// In-focus PSF width, px.
    pub sigma: f64,
    pub k_max: usize,
    /// The cut on the LoG statistic, in local sds, for FIND and for every
    /// placement in a box; defaults to [`PEAK_Z`].
    pub threshold: f64,
    pub selection: Selection,
    /// Extra cost per emitter, in units of I / phi. A count prior
    /// proportional to exp(-count_penalty * K) motivates this term.
    pub count_penalty: f64,
    /// Widths a fit may take, as multiples of `sigma`.
    pub slack: (f64, f64),
    pub sweeps: usize,
    pub polish: bool,
    /// Widths reported as detections, as multiples of `sigma`; `None`
    /// reports every fit.
    pub band: Option<(f64, f64)>,
}

/// One frame's answer. Every fitted emitter, in or out of any reporting band
/// -- the caller classifies. Positions are global `(y, x)`.
#[derive(Clone, Debug, Default)]
pub struct Output {
    /// `2N`, row-major `(y, x)`.
    pub pos: Vec<f64>,
    pub amp: Vec<f64>,
    pub sig: Vec<f64>,
    /// `3N`: SE of `(A, y, x)` from the polish's Fisher matrix, scaled by the
    /// local dispersion; NaN without.
    pub se: Vec<f64>,
    /// `N`: SE of each fitted width, likewise.
    pub se_sig: Vec<f64>,
    /// `N`: each emitter's [`Class`].
    pub class: Vec<Class>,
    /// The median of [`noise_map`]'s dispersion over the searched pixels:
    /// pixel variance per unit of signal, ADU. For a camera it is about the
    /// gain plus `gain^2 * read_noise^2 / background`.
    pub dispersion: f64,
    /// `H*W`, ADU above the offset.
    pub background: Vec<f64>,
    pub n_candidates: usize,
    pub n_boxes: usize,
    pub search_fits: usize,
    pub polish_fits: usize,
    pub selection_fits: usize,
}

/// Reusable per-thread storage: one per worker, never shared.
pub struct Workspace {
    fit: FitWorkspace,
    f: psf::Factors,
    chol: Chol,
    scratch: Vec<f64>,
    model: Vec<f64>,
}

impl Workspace {
    pub fn new() -> Self {
        Self {
            fit: FitWorkspace::new(),
            f: psf::Factors::new(1, 1, 1),
            chol: Chol::new(1),
            scratch: Vec::new(),
            model: Vec::new(),
        }
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// `np.percentile(v, q)`, linear interpolation, `q` in `[0, 100]`.
pub fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let pos = (s.len() - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (pos - lo as f64) * (s[hi] - s[lo])
}

/// The median of `a` (`h*w`) over the `win`-px square centred on each pixel,
/// the array's edge clamped outward.
///
/// Exact at the nodes of a grid [`NOISE_STRIDE`] apart and bilinear between
/// them (see its note for why that is enough). The nodes are the rows and
/// columns whose GLOBAL index `oy + r`, `ox + c` is a multiple of the stride,
/// plus the array's own first and last: an ROI crop of the frame then shares
/// every interior node with the whole frame, and only pixels within a stride
/// of the crop's edge -- inside [`crop_margin`] -- can tell the difference.
pub fn local_median(a: &[f64], h: usize, w: usize, win: usize, oy: usize, ox: usize) -> Vec<f64> {
    let nodes = |n: usize, o: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).filter(|&i| i == 0 || i == n - 1 || (o + i) % NOISE_STRIDE == 0).collect();
        v.dedup();
        v
    };
    let (ny, nx) = (nodes(h, oy), nodes(w, ox));
    let r = (win / 2) as isize;
    let mut buf = Vec::with_capacity(win * win);
    let mut g = vec![0.0; ny.len() * nx.len()];
    for (iy, &y) in ny.iter().enumerate() {
        for (ix, &x) in nx.iter().enumerate() {
            buf.clear();
            for dy in -r..=r {
                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                for dx in -r..=r {
                    let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    buf.push(a[yy * w + xx]);
                }
            }
            let mid = buf.len() / 2;
            let (_, m, _) = buf.select_nth_unstable_by(mid, f64::total_cmp);
            g[iy * nx.len() + ix] = *m;
        }
    }
    // For each row (column): the node interval it falls in and its weight.
    let interp = |n: usize, nodes: &[usize]| -> Vec<(usize, f64)> {
        (0..n)
            .map(|i| {
                if nodes.len() == 1 {
                    return (0, 0.0);
                }
                let k = nodes.partition_point(|&v| v <= i).clamp(1, nodes.len() - 1) - 1;
                let (a0, a1) = (nodes[k] as f64, nodes[k + 1] as f64);
                (k, ((i as f64 - a0) / (a1 - a0)).clamp(0.0, 1.0))
            })
            .collect()
    };
    let (wy, wx) = (interp(h, &ny), interp(w, &nx));
    let nxl = nx.len();
    let at = |iy: usize, ix: usize| g[iy.min(ny.len() - 1) * nxl + ix.min(nxl - 1)];
    let mut out = vec![0.0; h * w];
    for (r_, &(ky, fy)) in wy.iter().enumerate() {
        for (c, &(kx, fx)) in wx.iter().enumerate() {
            let top = (1.0 - fx) * at(ky, kx) + fx * at(ky, kx + 1);
            let bot = (1.0 - fx) * at(ky + 1, kx) + fx * at(ky + 1, kx + 1);
            out[r_ * w + c] = (1.0 - fy) * top + fy * bot;
        }
    }
    out
}

/// Estimate `(sd, phi)` per pixel, where `phi = sd^2 / local_median(data)`.
/// The separable fourth-difference filter removes smooth image structure;
/// a local median of its squared output estimates noise variance. Only valid
/// filter pixels contribute; boundary values are extended outward.
/// `(oy, ox)` anchors the median grid to global frame coordinates.
pub fn noise_map(d: &[f64], h: usize, w: usize, oy: usize, ox: usize) -> (Vec<f64>, Vec<f64>) {
    let med = local_median(d, h, w, NOISE_WIN, oy, ox);
    let var = if h < 5 || w < 5 {
        // Too small to filter: Poisson in ADU, `sd^2 = median`, is the most
        // that can be said.
        med.iter().map(|m| m.max(NOISE_VAR_FLOOR)).collect::<Vec<f64>>()
    } else {
        const K: [f64; 5] = [1.0, -4.0, 6.0, -4.0, 1.0];
        let (vh, vw) = (h - 4, w - 4);
        let mut tmp = vec![0.0; h * vw];
        for r in 0..h {
            for c in 0..vw {
                tmp[r * vw + c] = (0..5).map(|j| K[j] * d[r * w + c + j]).sum();
            }
        }
        let mut sq = vec![0.0; vh * vw];
        for r in 0..vh {
            for c in 0..vw {
                let v: f64 = (0..5).map(|i| K[i] * tmp[(r + i) * vw + c]).sum();
                sq[r * vw + c] = v * v;
            }
        }
        let norm = statistics::CHI2_1_MEDIAN * 70.0 * 70.0;
        let m = local_median(&sq, vh, vw, NOISE_WIN, oy + 2, ox + 2);
        let mut var = vec![0.0; h * w];
        for r in 0..h {
            let rr = r.clamp(2, h - 3) - 2;
            for c in 0..w {
                let cc = c.clamp(2, w - 3) - 2;
                var[r * w + c] = (m[rr * vw + cc] / norm).max(NOISE_VAR_FLOOR);
            }
        }
        var
    };
    let sd = var.iter().map(|v| v.sqrt()).collect();
    let phi = var.iter().zip(&med).map(|(v, m)| v / m.max(1e-6)).collect();
    (sd, phi)
}

/// Each emitter's [`Class`]. In the band it is a detection, and so is a
/// width out of it by no more than [`BAND_Z`] of its own SE `se_sig` (a NaN
/// SE never flags), unless it sits on the upper `slack` bound. Otherwise,
/// near the border it is `Edge`, else `Narrow` or `Wide` by which side it
/// fell: a source the border cuts is not an interior width measurement, so a
/// width flag always means an interior fit.
#[allow(clippy::too_many_arguments)]
pub fn classify(
    pos: &[f64],
    sig: &[f64],
    se_sig: &[f64],
    h: usize,
    w: usize,
    sigma: f64,
    slack: (f64, f64),
    band: Option<(f64, f64)>,
) -> Vec<Class> {
    (0..sig.len())
        .map(|k| {
            let Some((lo, hi)) = band else {
                return Class::Focus;
            };
            let (s, e) = (sig[k], se_sig.get(k).copied().unwrap_or(f64::NAN));
            let margin = if e.is_finite() { BAND_Z * e } else { f64::INFINITY };
            let narrow = lo * sigma - s > margin;
            let pinned = s >= slack.1 * sigma * (1.0 - BOUND_TOL);
            let wide = s > hi * sigma && (s - hi * sigma > margin || pinned);
            if !narrow && !wide {
                return Class::Focus;
            }
            let (y, x) = (pos[2 * k], pos[2 * k + 1]);
            let border = (y + 0.5)
                .min(h as f64 - 0.5 - y)
                .min(x + 0.5)
                .min(w as f64 - 0.5 - x);
            if border <= EDGE_MARGIN * sigma {
                Class::Edge
            } else if narrow {
                Class::Narrow
            } else {
                Class::Wide
            }
        })
        .collect()
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 0 {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    } else {
        s[n / 2]
    }
}

/// The values `keep` selects, or `None` to mean "use the array itself".
///
/// `None` for no mask, and also for a mask that selects nothing: an empty
/// selection is no statistic at all, and the whole array is a better answer
/// than a panic.
fn masked_values(v: &[f64], keep: Option<&[bool]>) -> Option<Vec<f64>> {
    let m = keep?;
    let sel: Vec<f64> = v.iter().zip(m).filter(|&(_, &k)| k).map(|(&x, _)| x).collect();
    (!sel.is_empty()).then_some(sel)
}

/// Context needed around an ROI: the largest support of candidate finding,
/// fit boxes, background estimation and noise estimation. Sequential filter
/// supports add; independent stages take the maximum.
fn crop_margin(sigma: f64) -> usize {
    let find = sigma.ceil() as usize + filters::kernel_radius(sigma);
    let boxes = (patches::BBOX_PAD * sigma).ceil() as usize + 1;
    let bg = 2 * (BG_KERNEL / 2) + filters::kernel_radius(BG_KERNEL as f64 / 6.0);
    let noise = 2 + NOISE_WIN / 2 + NOISE_STRIDE;
    find.max(boxes).max(bg).max(noise)
}

/// Widen `[lo, hi)` to at least `want` pixels without leaving `[0, n)`, and
/// without ever giving up ground it already held.
fn widen(lo: usize, hi: usize, want: usize, n: usize) -> (usize, usize) {
    let want = want.min(n);
    if hi - lo >= want {
        return (lo, hi);
    }
    let mid = (lo + hi) / 2;
    let start = mid.saturating_sub(want / 2).min(n - want);
    (start.min(lo), (start + want).max(hi))
}

/// The sub-frame [`localize`] does its whole-frame work on: the ROI's
/// bounding box plus [`crop_margin`], clamped to the frame. `None` when the
/// ROI selects no pixel at all.
///
/// Also at least `3 * BG_KERNEL` px on a side where the frame allows, because
/// [`background_map`] narrows its kernel on an array too small to hold it
/// (`BG_KERNEL.min(3.max(min(h, w) / 3))`). Without the floor a thin ROI
/// would silently get a different background kernel from the one the frame
/// would have used, which is exactly the margin's promise broken. It costs
/// nothing in the ordinary case: the margin alone already gives 83 px.
fn roi_crop(roi: &[bool], h: usize, w: usize, sigma: f64) -> Option<patches::BBox> {
    let (mut y0, mut y1) = (usize::MAX, 0usize);
    let (mut x0, mut x1) = (usize::MAX, 0usize);
    for r in 0..h {
        for c in 0..w {
            if roi[r * w + c] {
                y0 = y0.min(r);
                y1 = y1.max(r + 1);
                x0 = x0.min(c);
                x1 = x1.max(c + 1);
            }
        }
    }
    if y0 == usize::MAX {
        return None;
    }
    let m = crop_margin(sigma);
    let side = 3 * BG_KERNEL;
    let (y0, y1) = widen(y0.saturating_sub(m), (y1 + m).min(h), side, h);
    let (x0, x1) = widen(x0.saturating_sub(m), (x1 + m).min(w), side, w);
    Some(patches::BBox { y0, x0, y1, x1 })
}

/// Copy `bb` out of an `h*w` array, row by row.
fn crop<T: Copy>(v: &[T], w: usize, bb: &patches::BBox) -> Vec<T> {
    let mut out = Vec::with_capacity(bb.n_pixels());
    for r in bb.y0..bb.y1 {
        out.extend_from_slice(&v[r * w + bb.x0..r * w + bb.x1]);
    }
    out
}

/// Find LoG peaks against a flat `level`, returning `(positions, amplitudes,
/// strengths)` in decreasing strength order. Divide by local pixel noise and
/// the filter's L2 norm so `threshold` is in standard deviations.
/// LoG supplies candidate positions; model selection decides their counts.
pub fn find_candidates(
    d: &[f64],
    h: usize,
    w: usize,
    level: f64,
    sigma: f64,
    sd: &[f64],
    threshold: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let resid: Vec<f64> = d.iter().map(|&v| v - level).collect();
    let l2 = filters::log_kernel_l2(sigma);
    let mut log_f = filters::gaussian_laplace(&resid, h, w, sigma, Mode::Reflect);
    for (v, e) in log_f.iter_mut().zip(sd) {
        *v = -*v / (l2 * e);
    }
    let win = 2 * sigma.ceil() as usize + 1;
    let mx = filters::maximum_filter(&log_f, h, w, win, Mode::Reflect);
    let pf = psf::peak_factor(sigma);
    let mut found: Vec<(f64, usize)> = (0..h * w)
        .filter(|&i| log_f[i] == mx[i] && log_f[i] > threshold)
        .map(|i| (log_f[i], i))
        .collect();
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut pos = Vec::with_capacity(2 * found.len());
    let mut amp = Vec::with_capacity(found.len());
    let mut strength = Vec::with_capacity(found.len());
    for &(s, i) in &found {
        pos.push((i / w) as f64);
        pos.push((i % w) as f64);
        amp.push((d[i] - level).max(1e-2) / pf);
        strength.push(s);
    }
    (pos, amp, strength)
}

/// Estimate a smooth background from pixels outside candidate support:
/// two local means with upper-tail clipping at 3 local sds, then smoothing.
/// Use the data rather than fit residuals to avoid feedback from emitter fits.
///
/// Sparse windows use the median of unmasked pixels, or the 10th percentile
/// if too few remain. The ROI limits these fallback statistics, not the local
/// filter's context. Return `(surface, fallback)` for scattering into a frame.
pub fn background_map(
    d: &[f64],
    h: usize,
    w: usize,
    cand: &[f64],
    sigma: f64,
    sd: &[f64],
    roi: Option<&[bool]>,
) -> (Vec<f64>, f64) {
    let n = cand.len() / 2;
    let mut k = BG_KERNEL.min(3usize.max(h.min(w) / 3));
    if k % 2 == 0 {
        k += 1;
    }
    let free = render::emitter_free_mask(cand, n, sigma, BG_MASK_RADIUS, h, w);
    let inside = masked_values(d, roi);
    let stat = inside.as_deref().unwrap_or(d);
    let fallback = if n == 0 {
        median(stat)
    } else {
        let kept: Vec<f64> = (0..h * w)
            .filter(|&i| free[i] && roi.is_none_or(|m| m[i]))
            .map(|i| d[i])
            .collect();
        if kept.len() as f64 >= 16f64.max(0.02 * stat.len() as f64) {
            median(&kept)
        } else {
            percentile(stat, 10.0)
        }
    };
    let local_mean = |mask: &[bool]| -> (Vec<f64>, Vec<f64>) {
        let masked: Vec<f64> = (0..h * w).map(|i| if mask[i] { d[i] } else { 0.0 }).collect();
        let ones: Vec<f64> = mask.iter().map(|&m| if m { 1.0 } else { 0.0 }).collect();
        let num = filters::uniform_filter(&masked, h, w, k, Mode::Nearest);
        let den = filters::uniform_filter(&ones, h, w, k, Mode::Nearest);
        let b = num.iter().zip(&den).map(|(a, c)| a / c.max(1e-9)).collect();
        let cnt = den.iter().map(|c| c * (k * k) as f64).collect();
        (b, cnt)
    };
    let (b1, _) = local_mean(&free);
    let keep: Vec<bool> = (0..h * w)
        .map(|i| free[i] && d[i] <= b1[i] + 3.0 * sd[i])
        .collect();
    let (b2, cnt) = local_mean(&keep);
    let b3: Vec<f64> = (0..h * w)
        .map(|i| if cnt[i] >= BG_MIN_PIXELS { b2[i] } else { fallback })
        .collect();
    let mut out = filters::gaussian_filter(&b3, h, w, k as f64 / 6.0, Mode::Nearest);
    for v in out.iter_mut() {
        *v = v.max(BG_FLOOR);
    }
    (out, fallback.max(BG_FLOOR))
}

/// An emitter: `[A, y, x, sigma]`.
type Em = [f64; 4];

/// [`noise_map`]'s two maps over the whole frame.
struct Noise {
    w: usize,
    sd: Vec<f64>,
    phi: Vec<f64>,
}

/// A window's pixels and the parameter-free part of its model.
///
/// The background map enters split into a free `level` (its median here,
/// where the fit's `b` starts) and a known `shape`, added like a frozen
/// emitter. The split keeps `b` strictly interior: folding the whole surface
/// into the known term would leave `b` wanting to be 0, its lower bound, and
/// Coleman-Li collapses every coordinate's step when one parameter sits on a
/// bound (see [`lmcl`]).
struct Window {
    y0: usize,
    x0: usize,
    h: usize,
    w: usize,
    sub: Vec<f64>,
    /// Frozen neighbours plus the background map's shape.
    halo: Vec<f64>,
    /// The background map's median here; where the free level starts.
    level: f64,
    /// Background map's shape, `bmap - level`.
    shape: Vec<f64>,
    /// The local pixel sd, ADU, and the median dispersion over the window --
    /// the scale of its nats. Only a search window reads them ([`Window::noise`]).
    sd: Vec<f64>,
    phi: f64,
}

impl Window {
    fn new(d: &[f64], fw: usize, bmap: &[f64], bb: &patches::BBox) -> Self {
        let (h, w) = (bb.h(), bb.w());
        let mut sub = Vec::with_capacity(h * w);
        let mut bg = Vec::with_capacity(h * w);
        for r in bb.y0..bb.y1 {
            sub.extend_from_slice(&d[r * fw + bb.x0..r * fw + bb.x1]);
            bg.extend_from_slice(&bmap[r * fw + bb.x0..r * fw + bb.x1]);
        }
        let level = median(&bg);
        let shape: Vec<f64> = bg.iter().map(|v| v - level).collect();
        Self {
            y0: bb.y0,
            x0: bb.x0,
            h,
            w,
            sub,
            halo: shape.clone(),
            level,
            shape,
            sd: Vec::new(),
            phi: f64::NAN,
        }
    }

    /// Attach the noise a search reads: the window's sds and median `phi`.
    fn noise(mut self, noise: &Noise) -> Self {
        let mut phi = Vec::with_capacity(self.h * self.w);
        for r in self.y0..self.y0 + self.h {
            let row = r * noise.w + self.x0..r * noise.w + self.x0 + self.w;
            self.sd.extend_from_slice(&noise.sd[row.clone()]);
            phi.extend_from_slice(&noise.phi[row]);
        }
        self.phi = median(&phi);
        self
    }

    /// `halo <- (emitters, rendered here) + shape`. `ems` are global.
    fn set_halo(&mut self, ems: &[Em], f: &mut psf::Factors) {
        let n = self.h * self.w;
        self.halo.clear();
        self.halo.resize(n, 0.0);
        if !ems.is_empty() {
            let mut theta = Vec::with_capacity(4 * ems.len() + 1);
            theta.push(0.0);
            for e in ems {
                theta.extend_from_slice(&[e[0], e[1] - self.y0 as f64, e[2] - self.x0 as f64, e[3]]);
            }
            let (ay, ax) = (psf::local_axis(self.h), psf::local_axis(self.w));
            f.ensure(self.h, self.w, ems.len());
            psf::model_var_sigma_ax(&theta, &ay, &ax, None, f, &mut self.halo);
        }
        for (v, s) in self.halo.iter_mut().zip(&self.shape) {
            *v += s;
        }
    }
}

/// One converged window fit: its data-only I and parameters, local coords.
#[derive(Clone)]
struct Fitted {
    i_div: f64,
    b: f64,
    em: Vec<Em>,
}

/// One bounded free-width ML fit: bounds from the window's peak, the start
/// pulled 1e-9 inside them.
///
/// `a_max` is raised by `slack.1^2`: it comes from the window's peak through
/// `peak_factor(sigma)`, and a source `n` times the PSF width carries the
/// same flux at `1/n^2` of the peak, so the in-focus bound would clip exactly
/// the defocused emitters the slack exists for. Positions stay inside the
/// window: letting a centre leave it was tried -- it lets the fit put rim
/// flux where it came from -- and measurably lost real detections elsewhere.
fn fit_window(
    ws: &mut Workspace,
    win: &Window,
    b: f64,
    em: &[Em],
    s: &Settings,
    max_iter: usize,
    tol_obj: f64,
) -> Fitted {
    let k = em.len();
    let smax = win.sub.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v)).max(1.0);
    let b_max = (4.0 * smax).max(10.0);
    let a_max = 8.0 * smax / psf::peak_factor(s.sigma) * s.slack.1 * s.slack.1;
    let a_min = A_MIN.max(A_MIN_REL * a_max);
    let (s_lo, s_hi) = (s.slack.0 * s.sigma, s.slack.1 * s.sigma);
    let mut lo = Vec::with_capacity(4 * k + 1);
    let mut hi = Vec::with_capacity(4 * k + 1);
    lo.push(0.0);
    hi.push(b_max);
    for _ in 0..k {
        lo.extend_from_slice(&[a_min, -0.5, -0.5, s_lo]);
        hi.extend_from_slice(&[a_max, win.h as f64 - 0.5, win.w as f64 - 0.5, s_hi]);
    }
    let mut th0 = Vec::with_capacity(4 * k + 1);
    th0.push(b);
    for e in em {
        th0.extend_from_slice(&[e[0], e[1], e[2], e[3].clamp(s_lo, s_hi)]);
    }
    for q in 0..th0.len() {
        th0[q] = th0[q].clamp(lo[q] + 1e-9, hi[q] - 1e-9);
    }
    let bounds = Bounds::new(&lo, &hi);
    let info = lmcl::fit_var_sigma(
        &mut ws.fit,
        &th0,
        win.h,
        win.w,
        &win.sub,
        &bounds,
        Some(&win.halo),
        FitOpts {
            max_iter,
            tol_obj,
            ..Default::default()
        },
    );
    let t = ws.fit.theta();
    Fitted {
        i_div: info.i_div,
        b: t[0],
        em: (0..k)
            .map(|j| [t[1 + 4 * j], t[2 + 4 * j], t[3 + 4 * j], t[4 + 4 * j]])
            .collect(),
    }
}

/// Propose `(y, x, amplitude)` at the strongest owned residual LoG peak
/// above the threshold. Screening residuals limits satellites caused by
/// noise or imperfect PSF fits. The negated comparison rejects NaN scores.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn placement(
    ws: &mut Workspace,
    win: &Window,
    owned: &[bool],
    state: &Fitted,
    s: &Settings,
    l2: f64,
) -> Option<(f64, f64, f64)> {
    let n = win.h * win.w;
    let mut theta = Vec::with_capacity(4 * state.em.len() + 1);
    theta.push(state.b);
    for e in &state.em {
        theta.extend_from_slice(e);
    }
    let (ay, ax) = (psf::local_axis(win.h), psf::local_axis(win.w));
    ws.f.ensure(win.h, win.w, state.em.len().max(1));
    ws.model.clear();
    ws.model.resize(n, 0.0);
    psf::model_var_sigma_ax(&theta, &ay, &ax, Some(&win.halo), &mut ws.f, &mut ws.model);
    let resid: Vec<f64> = (0..n).map(|i| win.sub[i] - ws.model[i]).collect();
    let log_f = filters::gaussian_laplace(&resid, win.h, win.w, s.sigma, Mode::Nearest);
    let mut best: Option<(f64, usize)> = None;
    for i in 0..n {
        if owned[i] {
            let v = -log_f[i] / (l2 * win.sd[i]);
            if best.is_none_or(|(b, _)| v > b) {
                best = Some((v, i));
            }
        }
    }
    let (v, i) = best?;
    if !(v > s.threshold) {
        return None;
    }
    let resid = win.sub[i] - ws.model[i];
    Some(((i / win.w) as f64, (i % win.w) as f64, resid.max(1e-2) / psf::peak_factor(s.sigma)))
}

/// One box decided from K = 0: FORWARD placements while each pays
/// `ADD_NATS * phi`, then BACKWARD removals while one costs less. Returns the
/// box's emitters (local) and the fits spent.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn search_box(
    ws: &mut Workspace,
    win: &Window,
    owned: &[bool],
    s: &Settings,
    l2: f64,
) -> (Vec<Em>, usize) {
    if s.selection == Selection::Bic {
        return search_box_bic(ws, win, owned, s, l2);
    }
    let mut fits = 1usize;
    let cost = (ADD_NATS + s.count_penalty) * win.phi;
    let mut state = fit_window(ws, win, win.level, &[], s, FIT_MAX_ITER, FIT_TOL_OBJ);
    while state.em.len() < s.k_max {
        let Some((y, x, a0)) = placement(ws, win, owned, &state, s, l2) else {
            break;
        };
        let mut em = state.em.clone();
        em.push([a0, y, x, s.sigma]);
        let trial = fit_window(ws, win, state.b, &em, s, FIT_MAX_ITER, FIT_TOL_OBJ);
        fits += 1;
        if !(state.i_div - trial.i_div > cost) {
            break;
        }
        state = trial;
    }
    // Not at K = 1: that removal is the K = 0 fit FORWARD already beat.
    // Measured: output bit-identical, 11-25% fewer fits.
    while state.em.len() > 1 {
        let mut best: Option<Fitted> = None;
        for drop in 0..state.em.len() {
            let em: Vec<Em> = (0..state.em.len())
                .filter(|&j| j != drop)
                .map(|j| state.em[j])
                .collect();
            let reduced = fit_window(ws, win, state.b, &em, s, FIT_MAX_ITER, FIT_TOL_OBJ);
            fits += 1;
            if best.as_ref().is_none_or(|b| reduced.i_div < b.i_div) {
                best = Some(reduced);
            }
        }
        let best = best.expect("K >= 2 has removals");
        if !(best.i_div - state.i_div < cost) {
            break;
        }
        state = best;
    }
    (state.em, fits)
}

/// Half of the BIC-inspired score, with the common background penalty
/// omitted: I / phi + K * (2 ln(n_pixels) + lambda). Each emitter has four
/// free parameters. Pixels, background shape, halo and phi MUST stay fixed
/// throughout a comparison; filtered pixels and photon counts are not n.
fn count_score(state: &Fitted, win: &Window, s: &Settings) -> f64 {
    state.i_div / win.phi
        + state.em.len() as f64 * (2.0 * ((win.h * win.w) as f64).ln() + s.count_penalty)
}

fn prefer_count(trial: &Fitted, best: &Fitted, win: &Window, s: &Settings) -> bool {
    let (a, b) = (count_score(trial, win, s), count_score(best, win, s));
    a.is_finite()
        && (!b.is_finite() || a < b || (a == b && trial.em.len() < best.em.len()))
}

/// Follow a deletion path all the way to K=1, refitting EVERY possible
/// single deletion at each step. Keep the best score encountered, including
/// the separately fitted K=0 model. Unlike greedy threshold pruning, an
/// intermediate count that scores poorly does not stop the search.
fn select_reductions(
    ws: &mut Workspace,
    win: &Window,
    mut state: Fitted,
    mut best: Fitted,
    s: &Settings,
) -> (Fitted, usize) {
    let mut fits = 0;
    while state.em.len() > 1 {
        let mut reduced_best: Option<Fitted> = None;
        for drop in 0..state.em.len() {
            let em: Vec<Em> = state.em.iter().enumerate()
                .filter(|(j, _)| *j != drop).map(|(_, e)| *e).collect();
            let trial = fit_window(ws, win, state.b, &em, s, FIT_MAX_ITER, FIT_TOL_OBJ);
            fits += 1;
            if trial.i_div.is_finite()
                && reduced_best.as_ref().is_none_or(|b| trial.i_div < b.i_div)
            {
                reduced_best = Some(trial);
            }
        }
        let Some(reduced) = reduced_best else { break };
        if prefer_count(&reduced, &best, win, s) {
            best = reduced.clone();
        }
        state = reduced;
    }
    (best, fits)
}

/// Candidate counts from an ascending residual-placement path and a
/// descending deletion path, not an exhaustive search over configurations.
/// LoG still limits proposals. Continue past a failed complexity test so
/// e.g. K=2 can win even if K=1 did not. Stop at k_max, no residual proposal,
/// or a failed likelihood improvement (a numerical/search failure).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn search_box_bic(
    ws: &mut Workspace, win: &Window, owned: &[bool], s: &Settings, l2: f64,
) -> (Vec<Em>, usize) {
    let mut state = fit_window(ws, win, win.level, &[], s, FIT_MAX_ITER, FIT_TOL_OBJ);
    let mut best = state.clone();
    let mut fits = 1;
    while state.em.len() < s.k_max {
        let Some((y, x, a)) = placement(ws, win, owned, &state, s, l2) else { break };
        let mut em = state.em.clone();
        em.push([a, y, x, s.sigma]);
        let trial = fit_window(ws, win, state.b, &em, s, FIT_MAX_ITER, FIT_TOL_OBJ);
        fits += 1;
        if !trial.i_div.is_finite() || !(trial.i_div < state.i_div) { break; }
        if prefer_count(&trial, &best, win, s) { best = trial.clone(); }
        state = trial;
    }
    let (best, reduction_fits) = select_reductions(ws, win, state, best, s);
    (best.em, fits + reduction_fits)
}

/// Collect neighboring emitters that reach this box, preserving visit order.
fn collect_halo(
    bb: &patches::BBox,
    neighbours: &[usize],
    held: &[Vec<Em>],
    sigma: f64,
    out: &mut Vec<Em>,
) {
    let (y0, y1) = (bb.y0 as f64, (bb.y1 - 1) as f64);
    let (x0, x1) = (bb.x0 as f64, (bb.x1 - 1) as f64);
    out.clear();
    for &j in neighbours {
        for e in &held[j] {
            let dy = e[1] - e[1].clamp(y0, y1);
            let dx = e[2] - e[2].clamp(x0, x1);
            if dy.hypot(dx) <= HALO_FACTOR * e[3].max(sigma) {
                out.push(*e);
            }
        }
    }
}

/// Pixels a box may place on: within `OWN_RADIUS*sigma` of its own nearest
/// candidate, no farther from it than from any other candidate, and in the
/// ROI. Only candidates within that radius of the box can take a pixel from
/// it, so `grid` is asked for those alone.
#[allow(clippy::too_many_arguments)]
fn owned_mask(
    bb: &patches::BBox,
    own: &[u32],
    cand: &[f64],
    grid: &EmitterGrid,
    roi: Option<&[bool]>,
    fw: usize,
    sigma: f64,
    near: &mut Vec<u32>,
) -> Vec<bool> {
    let r = OWN_RADIUS * sigma;
    grid.query_rect(
        bb.y0 as f64,
        bb.x0 as f64,
        (bb.y1 - 1) as f64,
        (bb.x1 - 1) as f64,
        r,
        near,
    );
    near.retain(|i| !own.contains(i));
    let mut owned = Vec::with_capacity(bb.n_pixels());
    for py in bb.y0..bb.y1 {
        for px in bb.x0..bb.x1 {
            let dist = |i: u32| {
                (py as f64 - cand[2 * i as usize]).hypot(px as f64 - cand[2 * i as usize + 1])
            };
            let d_own = own.iter().map(|&i| dist(i)).fold(f64::INFINITY, f64::min);
            let d_oth = near.iter().map(|&i| dist(i)).fold(f64::INFINITY, f64::min);
            let in_roi = roi.is_none_or(|m| m[py * fw + px]);
            owned.push(d_own <= r && d_own <= d_oth && in_roi);
        }
    }
    owned
}

/// Localize an offset-subtracted frame in ADU. An ROI limits proposals;
/// preprocessing uses its bounding box plus enough context for all filters.
/// Return global coordinates and a full-frame background map. Empty ROIs
/// return no detections. ROI pixels set the background fallback and reported
/// dispersion.
pub fn localize(
    d: &[f64],
    h: usize,
    w: usize,
    roi: Option<&[bool]>,
    s: &Settings,
    ws: &mut Workspace,
) -> Output {
    assert_eq!(d.len(), h * w);
    let bb = match roi {
        None => patches::BBox { y0: 0, x0: 0, y1: h, x1: w },
        Some(m) => match roi_crop(m, h, w, s.sigma) {
            Some(bb) => bb,
            // An ROI selecting nothing asks for nothing.
            None => {
                return Output {
                    background: vec![BG_FLOOR; h * w],
                    dispersion: f64::NAN,
                    ..Output::default()
                }
            }
        },
    };
    let whole = bb.h() == h && bb.w() == w;
    let (cw, ch) = (bb.w(), bb.h());
    let dsub = if whole { Vec::new() } else { crop(d, w, &bb) };
    let dc: &[f64] = if whole { d } else { &dsub };
    let rsub = match roi {
        Some(m) if !whole => Some(crop(m, w, &bb)),
        _ => None,
    };
    let rc: Option<&[bool]> = match (roi, &rsub) {
        (Some(m), None) => Some(m),
        (_, Some(v)) => Some(v),
        (None, None) => None,
    };

    let level = masked_values(dc, rc);
    let b0 = percentile(level.as_deref().unwrap_or(dc), 10.0).max(BG_FLOOR);
    let (sd_c, phi_c) = noise_map(dc, ch, cw, bb.y0, bb.x0);
    let dispersion = median(masked_values(&phi_c, rc).as_deref().unwrap_or(&phi_c));
    let (cand_all, amp_all, str_all) = find_candidates(dc, ch, cw, b0, s.sigma, &sd_c, s.threshold);
    let (bsub, fill) = background_map(dc, ch, cw, &cand_all, s.sigma, &sd_c, rc);
    // Back to the frame. Outside the crop nothing was estimated, so each map
    // carries a fill rather than a hole: no fit reads it -- every box lies
    // inside the crop by `crop_margin` -- but callers render the background.
    let to_frame = |sub: Vec<f64>, fill: f64| -> Vec<f64> {
        if whole {
            return sub;
        }
        let mut full = vec![fill; h * w];
        for r in 0..ch {
            full[(bb.y0 + r) * w + bb.x0..(bb.y0 + r) * w + bb.x1]
                .copy_from_slice(&sub[r * cw..(r + 1) * cw]);
        }
        full
    };
    let bmap = to_frame(bsub, fill);
    let sd_fill = median(&sd_c);
    let noise = Noise { w, sd: to_frame(sd_c, sd_fill), phi: to_frame(phi_c, dispersion) };

    let (mut cand, mut camp, mut strength) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..amp_all.len() {
        let (y, x) = (cand_all[2 * j] + bb.y0 as f64, cand_all[2 * j + 1] + bb.x0 as f64);
        if roi.is_none_or(|m| m[y as usize * w + x as usize]) {
            cand.extend_from_slice(&[y, x]);
            camp.push(amp_all[j]);
            strength.push(str_all[j]);
        }
    }
    let nc = camp.len();
    let mut boxes = patches::build_patches(&cand, nc, s.sigma, h, w, s.k_max);
    // Brightest first, so the strongest light is already fitted when its
    // neighbours read it through their halos. Stable, so ties keep FIND's
    // order.
    let peak = |p: &patches::Patch| {
        p.indices
            .iter()
            .map(|&i| strength[i as usize])
            .fold(f64::NEG_INFINITY, f64::max)
    };
    boxes.sort_by(|a, b| peak(b).total_cmp(&peak(a)));
    let nb = boxes.len();

    // Once per frame: each box's pixels, ownership and the boxes that can
    // reach it. A held emitter stays inside its own box's fit bounds, within
    // half a pixel of the rectangle, at a width of at most slack.1 * sigma;
    // the halo then admits it only within HALO_FACTOR widths of this
    // box's pixel rectangle. Boxes farther than that can never contribute.
    let grid = EmitterGrid::build(&cand, nc, h, w, (OWN_RADIUS * s.sigma).max(1.0));
    let mut near = Vec::new();
    let mut wins: Vec<Window> = Vec::with_capacity(nb);
    let mut owned: Vec<Vec<bool>> = Vec::with_capacity(nb);
    for p in &boxes {
        wins.push(Window::new(d, w, &bmap, &p.bbox).noise(&noise));
        owned.push(owned_mask(&p.bbox, &p.indices, &cand, &grid, roi, w, s.sigma, &mut near));
    }
    let reach = HALO_FACTOR * s.slack.1.max(1.0) * s.sigma;
    let gap = |a0: f64, a1: f64, b0: f64, b1: f64| (b0 - a1).max(a0 - b1).max(0.0);
    let neighbours: Vec<Vec<usize>> = (0..nb)
        .map(|i| {
            let bi = &boxes[i].bbox;
            (0..nb)
                .filter(|&j| j != i)
                .filter(|&j| {
                    let bj = &boxes[j].bbox;
                    let gy = gap(
                        bi.y0 as f64,
                        (bi.y1 - 1) as f64,
                        bj.y0 as f64 - 0.5,
                        bj.y1 as f64 - 0.5,
                    );
                    let gx = gap(
                        bi.x0 as f64,
                        (bi.x1 - 1) as f64,
                        bj.x0 as f64 - 0.5,
                        bj.x1 as f64 - 0.5,
                    );
                    gy.hypot(gx) <= reach
                })
                .collect()
        })
        .collect();

    // What each box holds, global coords. Until a box is first decided, its
    // candidates stand in for it as in-focus seeds.
    let mut held: Vec<Vec<Em>> = boxes
        .iter()
        .map(|p| {
            p.indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    [camp[i], cand[2 * i], cand[2 * i + 1], s.sigma]
                })
                .collect()
        })
        .collect();

    let l2 = filters::log_kernel_l2(s.sigma);
    let mut search_fits = 0usize;
    let mut ems: Vec<Em> = Vec::new();
    // The halo each box was last decided against. A box is a pure function of
    // its pixels and its halo, so when a later sweep hands it the same halo
    // bit for bit its answer cannot change and it is not re-decided. On GEM
    // frames that is a box with no neighbour in reach, or whose neighbours'
    // emitters all came back where they were.
    let mut seen: Vec<Option<Vec<Em>>> = vec![None; nb];
    for _ in 0..s.sweeps {
        for i in 0..nb {
            let bb = boxes[i].bbox;
            collect_halo(&bb, &neighbours[i], &held, s.sigma, &mut ems);
            if seen[i].as_deref() == Some(ems.as_slice()) {
                continue;
            }
            wins[i].set_halo(&ems, &mut ws.f);
            let (local, fits) = search_box(ws, &wins[i], &owned[i], s, l2);
            search_fits += fits;
            let (y0, x0) = (bb.y0 as f64, bb.x0 as f64);
            held[i] = local.iter().map(|e| [e[0], e[1] + y0, e[2] + x0, e[3]]).collect();
            seen[i] = Some(ems.clone());
        }
    }

    let all: Vec<Em> = held.iter().flatten().copied().collect();
    let mut n = all.len();
    let mut pos: Vec<f64> = all.iter().flat_map(|e| [e[1], e[2]]).collect();
    let mut amp: Vec<f64> = all.iter().map(|e| e[0]).collect();
    let mut sig: Vec<f64> = all.iter().map(|e| e[3]).collect();
    let mut se4 = vec![f64::NAN; 4 * n];
    let mut polish_fits = if s.polish && n > 0 {
        polish(d, h, w, &bmap, &mut pos, &mut amp, &mut sig, &mut se4, s, ws)
    } else {
        0
    };
    let mut selection_fits = 0;
    if s.selection == Selection::Bic && s.polish && n > 0 {
        // Bring the refined parameters back to their original groups. Keep
        // those groups' pixels and noise fixed for the final comparisons.
        let mut k = 0;
        for group in &mut held {
            for e in group {
                *e = [amp[k], pos[2 * k], pos[2 * k + 1], sig[k]];
                k += 1;
            }
        }
        for i in 0..nb {
            if held[i].is_empty() { continue; }
            let bb = boxes[i].bbox;
            let (y0, x0) = (bb.y0 as f64, bb.x0 as f64);
            collect_halo(&bb, &neighbours[i], &held, s.sigma, &mut ems);
            wins[i].set_halo(&ems, &mut ws.f);
            let start: Vec<Em> = held[i].iter()
                .map(|e| [e[0], e[1] - y0, e[2] - x0, e[3]]).collect();
            let win = &wins[i];
            let state = fit_window(ws, win, win.level, &start, s, FIT_MAX_ITER, FIT_TOL_OBJ);
            let mut best = fit_window(ws, win, win.level, &[], s, FIT_MAX_ITER, FIT_TOL_OBJ);
            selection_fits += 2;
            if prefer_count(&state, &best, win, s) { best = state.clone(); }
            let (best, reduction_fits) = select_reductions(ws, win, state, best, s);
            selection_fits += reduction_fits;
            held[i] = best.em.iter().map(|e| [e[0], e[1] + y0, e[2] + x0, e[3]]).collect();
        }
        let all: Vec<Em> = held.iter().flatten().copied().collect();
        n = all.len();
        pos = all.iter().flat_map(|e| [e[1], e[2]]).collect();
        amp = all.iter().map(|e| e[0]).collect();
        sig = all.iter().map(|e| e[3]).collect();
        // Refresh parameters and uncertainty after selection. This is one
        // bounded selection pass, not a claim of a global fixed point.
        se4 = vec![f64::NAN; 4 * n];
        if n > 0 {
            polish_fits += polish(d, h, w, &bmap, &mut pos, &mut amp, &mut sig, &mut se4, s, ws);
        }
    }
    // The Fisher matrix treated the ADU as Poisson counts, whose variance is
    // the mean; the pixel's is `phi` times that. So every variance is `phi`
    // times too small, read at the emitter.
    let mut se = Vec::with_capacity(3 * n);
    let mut se_sig = Vec::with_capacity(n);
    for k in 0..n {
        let py = (pos[2 * k].round().max(0.0) as usize).min(h - 1);
        let px = (pos[2 * k + 1].round().max(0.0) as usize).min(w - 1);
        let scale = noise.phi[py * w + px].sqrt();
        se.extend((0..3).map(|c| se4[4 * k + c] * scale));
        se_sig.push(se4[4 * k + 3] * scale);
    }
    let class = classify(&pos, &sig, &se_sig, h, w, s.sigma, s.slack, s.band);
    Output {
        pos,
        amp,
        sig,
        se,
        se_sig,
        class,
        dispersion,
        background: bmap,
        n_candidates: nc,
        n_boxes: nb,
        search_fits,
        polish_fits,
        selection_fits,
    }
}

/// Refine at fixed emitter count and estimate uncertainties from each fit's
/// Fisher matrix. All patches in a sweep read the same input state (Jacobi).
/// Rebuild patches each sweep; revisit those whose free or frozen emitters
/// moved. `POLISH_SWEEPS` bounds the work if overlapping groups keep moving.
#[allow(clippy::too_many_arguments)]
fn polish(
    d: &[f64],
    h: usize,
    w: usize,
    bmap: &[f64],
    pos: &mut Vec<f64>,
    amp: &mut Vec<f64>,
    sig: &mut Vec<f64>,
    se: &mut [f64],
    s: &Settings,
    ws: &mut Workspace,
) -> usize {
    let n = amp.len();
    let mut dirty = vec![true; n];
    let mut fits = 0usize;
    let mut ems: Vec<Em> = Vec::new();
    let mut var = Vec::new();
    for _ in 0..POLISH_SWEEPS {
        let pset = patches::build_patches(pos, n, s.sigma, h, w, s.k_max);
        let (mut opos, mut oamp, mut osig) = (pos.clone(), amp.clone(), sig.clone());
        let mut moved = vec![false; n];
        let mut n_fitted = 0usize;
        for p in &pset {
            let touched = p.indices.iter().chain(&p.frozen);
            if !touched.into_iter().any(|&i| dirty[i as usize]) {
                continue;
            }
            n_fitted += 1;
            let mut win = Window::new(d, w, bmap, &p.bbox);
            ems.clear();
            ems.extend(p.frozen.iter().map(|&i| {
                let i = i as usize;
                [amp[i], pos[2 * i], pos[2 * i + 1], sig[i]]
            }));
            win.set_halo(&ems, &mut ws.f);
            let (y0, x0) = (p.bbox.y0 as f64, p.bbox.x0 as f64);
            let start: Vec<Em> = p
                .indices
                .iter()
                .map(|&i| {
                    let i = i as usize;
                    [amp[i], pos[2 * i] - y0, pos[2 * i + 1] - x0, sig[i]]
                })
                .collect();
            let r = fit_window(ws, &win, win.level, &start, s, POLISH_MAX_ITER, POLISH_TOL_OBJ);
            fits += 1;
            for (j, &i) in p.indices.iter().enumerate() {
                let i = i as usize;
                let e = r.em[j];
                oamp[i] = e[0];
                opos[2 * i] = e[1] + y0;
                opos[2 * i + 1] = e[2] + x0;
                osig[i] = e[3];
                moved[i] = (opos[2 * i] - pos[2 * i]).hypot(opos[2 * i + 1] - pos[2 * i + 1])
                    > POLISH_MOVE_TOL;
            }
            // SEs from the Fisher matrix of the fit whose parameters are
            // reported. An indefinite matrix leaves the previous SEs.
            let pdim = 4 * p.indices.len() + 1;
            ws.chol.ensure(pdim);
            if ws.chol.factor(ws.fit.fisher(pdim), pdim) {
                var.resize(pdim, 0.0);
                ws.chol.inv_diag(&mut var, &mut ws.scratch);
                for (j, &i) in p.indices.iter().enumerate() {
                    for c in 0..4 {
                        let v = var[1 + 4 * j + c];
                        se[4 * i as usize + c] = if v > 0.0 { v.sqrt() } else { f64::NAN };
                    }
                }
            }
        }
        *pos = opos;
        *amp = oamp;
        *sig = osig;
        if n_fitted == 0 || !moved.iter().any(|&m| m) {
            break;
        }
        dirty = moved;
    }
    fits
}

#[allow(clippy::too_many_arguments)]
/// Localize one raw frame, `d = raw - offset` in ADU. No gain and no read
/// noise: [`noise_map`] measures what they would have said.
pub fn localize_raw(
    raw: &[f64],
    h: usize,
    w: usize,
    offset: f64,
    roi: Option<&[bool]>,
    s: &Settings,
    ws: &mut Workspace,
    d: &mut Vec<f64>,
) -> Output {
    d.clear();
    d.extend(raw.iter().map(|&r| r - offset));
    localize(d, h, w, roi, s, ws)
}

/// Localize every frame of a stack on `n_threads` workers. Frames are
/// independent, so each worker takes the next undone frame and keeps its own
/// [`Workspace`]; the output is in frame order whatever the scheduling.
///
/// `raw` is `n*H*W`; each frame goes through [`localize_raw`].
#[allow(clippy::too_many_arguments)]
pub fn localize_stack(
    raw: &[f64],
    n: usize,
    h: usize,
    w: usize,
    offset: f64,
    roi: Option<&[bool]>,
    s: &Settings,
    n_threads: usize,
) -> Vec<Output> {
    assert_eq!(raw.len(), n * h * w);
    crate::frames::map(n, n_threads,
        || (Workspace::new(), Vec::with_capacity(h * w)),
        |t, (ws, d)| {
            let frame = &raw[t * h * w..(t + 1) * h * w];
            localize_raw(frame, h, w, offset, roi, s, ws, d)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic standard normals: an LCG through Box-Muller.
    fn normals(n: usize, mut state: u64) -> Vec<f64> {
        let mut uni = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| (-2.0 * uni().ln()).sqrt() * (2.0 * std::f64::consts::PI * uni()).cos())
            .collect()
    }

    #[test]
    fn percentile_interpolates_as_numpy_does() {
        let v = [4.0, 1.0, 3.0, 2.0, 10.0];
        // np.percentile([1, 2, 3, 4, 10], q) for q = 0, 10, 50, 90, 100.
        for (q, want) in [(0.0, 1.0), (10.0, 1.4), (50.0, 3.0), (90.0, 7.6), (100.0, 10.0)] {
            assert!((percentile(&v, q) - want).abs() < 1e-12, "q={q}");
        }
    }

    #[test]
    fn the_noise_map_reads_white_noise_and_scales_with_the_image() {
        let (h, w) = (96, 80);
        let z = normals(h * w, 7);
        // Mean 50, sd 3: phi = 9 / 50.
        let d: Vec<f64> = z.iter().map(|v| 50.0 + 3.0 * v).collect();
        let (sd, phi) = noise_map(&d, h, w, 0, 0);
        let msd = median(&sd);
        assert!((msd - 3.0).abs() < 0.1, "sd {msd}");
        assert!((median(&phi) - 9.0 / 50.0).abs() < 0.015);
        // Every decision divides by this map, so it must scale with the data.
        let d7: Vec<f64> = d.iter().map(|v| 7.0 * v).collect();
        let (sd7, phi7) = noise_map(&d7, h, w, 0, 0);
        for i in 0..h * w {
            assert!((sd7[i] - 7.0 * sd[i]).abs() < 1e-9 * sd7[i]);
            assert!((phi7[i] - 7.0 * phi[i]).abs() < 1e-9 * phi7[i]);
        }
    }

    #[test]
    fn local_median_is_exact_on_its_nodes_and_anchored_to_the_frame() {
        let (h, w) = (40, 50);
        let a = normals(h * w, 3);
        let m = local_median(&a, h, w, 5, 0, 0);
        let r = 2isize;
        for &(y, x) in &[(0usize, 0usize), (12, 24), (24, 36), (39, 49)] {
            let mut v = Vec::new();
            for dy in -r..=r {
                for dx in -r..=r {
                    let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                    let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    v.push(a[yy * w + xx]);
                }
            }
            assert!((m[y * w + x] - median(&v)).abs() < 1e-12, "node ({y}, {x})");
        }
        // A crop whose origin is (12, 12) shares the frame's interior nodes.
        let sub: Vec<f64> = (12..40).flat_map(|r| a[r * w + 12..r * w + 50].to_vec()).collect();
        let ms = local_median(&sub, 28, 38, 5, 12, 12);
        assert!((ms[12 * 38 + 12] - m[24 * w + 24]).abs() < 1e-12);
    }

    #[test]
    fn bic_deletion_path_refits_neighbours_and_reaches_the_single_source() {
        let (h, w) = (17, 17);
        let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
        let mut data = vec![0.0; h * w];
        psf::model_var_sigma_ax(
            &[20.0, 1200.0, 8.2, 8.4, 1.2], &ay, &ax, None,
            &mut psf::Factors::new(h, w, 1), &mut data,
        );
        // Known noise for this deterministic model-selection test. Real
        // frames use noise_map; a noiseless image cannot estimate its noise.
        let win = Window {
            y0: 0, x0: 0, h, w, sub: data, halo: vec![0.0; h * w],
            shape: vec![0.0; h * w], level: 20.0,
            sd: vec![20.0_f64.sqrt(); h * w], phi: 1.0,
        };
        let s = Settings {
            sigma: 1.2, k_max: 4, threshold: PEAK_Z,
            selection: Selection::Bic, count_penalty: 2.0,
            slack: SLACK, sweeps: SWEEPS, polish: true, band: None,
        };
        let mut ws = Workspace::new();
        let mut best = fit_window(&mut ws, &win, 20.0, &[], &s, FIT_MAX_ITER, FIT_TOL_OBJ);
        let state = fit_window(
            &mut ws, &win, 20.0,
            &[[600.0, 8.0, 7.7, 1.2], [550.0, 8.5, 9.0, 1.2],
              [50.0, 5.0, 5.0, 1.2]],
            &s, FIT_MAX_ITER, FIT_TOL_OBJ,
        );
        if prefer_count(&state, &best, &win, &s) { best = state.clone(); }
        let (best, fits) = select_reductions(&mut ws, &win, state, best, &s);
        assert_eq!(fits, 5); // all three removals, then both removals
        assert_eq!(best.em.len(), 1);
        assert!((best.em[0][1] - 8.2).hypot(best.em[0][2] - 8.4) < 0.01);
        assert!((best.em[0][0] - 1200.0).abs() < 1.0);
        let strict = Settings { count_penalty: 1e6, ..s };
        let (em, _) = search_box_bic(
            &mut ws, &win, &vec![true; h * w], &strict,
            filters::log_kernel_l2(s.sigma),
        );
        assert!(em.is_empty());
    }

    #[test]
    fn an_isolated_emitter_is_found_once_where_it_is() {
        let (h, w, sigma) = (31, 33, 1.2);
        let theta = [0.0, 1500.0, 14.3, 17.6, 1.3];
        let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
        let mut f = psf::Factors::new(h, w, 1);
        let mut d = vec![0.0; h * w];
        psf::model_var_sigma_ax(&theta, &ay, &ax, None, &mut f, &mut d);
        // Gaussian noise at the Poisson variance on a background of 10. The
        // detector measures its noise from the frame, so a noise-free frame
        // is not a meaningful input any more.
        for (v, z) in d.iter_mut().zip(normals(h * w, 11)) {
            *v += 10.0;
            *v += v.sqrt() * z;
        }
        let s = Settings {
            sigma,
            k_max: 12,
            threshold: PEAK_Z,
            selection: Selection::Fixed,
            count_penalty: 0.0,
            slack: (0.7, 2.2),
            sweeps: SWEEPS,
            polish: true,
            band: Some((0.8, 2.0)),
        };
        let o = localize(&d, h, w, None, &s, &mut Workspace::new());
        // Once near the truth. The noise may also buy a faint fit elsewhere:
        // with this seed, a 24-count spike on the frame's top row at the
        // lowest width, which pays its nats once the fit converges.
        let near: Vec<usize> = (0..o.amp.len())
            .filter(|&k| (o.pos[2 * k] - 14.3).hypot(o.pos[2 * k + 1] - 17.6) < 3.0)
            .collect();
        assert_eq!(near.len(), 1, "found {:?} amp {:?}", o.pos, o.amp);
        let k = near[0];
        assert!((0..o.amp.len()).all(|j| j == k || o.amp[j] < 0.05 * o.amp[k]), "amp {:?}", o.amp);
        assert_eq!(o.class[k], Class::Focus);
        assert!(o.se.iter().chain(&o.se_sig).all(|v| v.is_finite() && *v > 0.0));
        assert!((o.dispersion - 1.0).abs() < 0.3, "dispersion {}", o.dispersion);
        let (se_a, se_y, se_x) = (o.se[3 * k], o.se[3 * k + 1], o.se[3 * k + 2]);
        assert!((o.pos[2 * k] - 14.3).abs() < 3.0 * se_y, "y {}", o.pos[2 * k]);
        assert!((o.pos[2 * k + 1] - 17.6).abs() < 3.0 * se_x, "x {}", o.pos[2 * k + 1]);
        assert!((o.amp[k] - 1500.0).abs() < 3.0 * se_a, "A {}", o.amp[k]);
        assert!((o.sig[k] - 1.3).abs() < 3.0 * o.se_sig[k], "sigma {}", o.sig[k]);
    }
}
