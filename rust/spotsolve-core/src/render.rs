//! Rendering emitters into images, and the mask of pixels no emitter reaches.
//!
//! Ports `calibrate.py::render_model`, `patches.py::build_halo_image` and the
//! masking loop shared by `core.py::background_map` and
//! `calibrate.py::robust_background`.
//!
//! # Why the mask is here and the convolutions are not
//!
//! The background surface's *filtering* stays in Python: `scipy.ndimage` is
//! already C, every filter call is per-round rather than per-fit, and
//! `background_map` profiles at 0.1-0.3%. Its **mask** is a different animal.
//! Written as `for each emitter: free &= (yy-cy)^2 + (xx-cx)^2 > r2` it touches
//! every pixel for every emitter -- `O(N*H*W)`. That is inside the 0.3% at
//! 128x128 with N=901 and emphatically outside it at 512x512, and it is the
//! same super-linear shape [P9] identifies in `_window`. Stamping only the
//! pixels within the radius makes it `O(N*sigma^2)`, and the result is
//! identical pixel for pixel.

use crate::psf;

/// Truncation radius for the global model, in sigma.
///
/// Each emitter is rendered only within this many sigma of its centre, so cost
/// is linear in emitter count rather than `N*H*W`. It must stay at least as
/// wide as the frozen halo's own radius, so the halo never omits flux the
/// global model includes.
pub const RENDER_TRUNCATE: f64 = 4.0;

/// Global model image: `background` plus every emitter's PSF, summed.
///
/// Contributions from overlapping emitters **add**, which is what the physics
/// says and what every patch fit assumes locally. (An earlier Python version
/// stitched per-patch models by *averaging* them where bounding boxes
/// overlapped; that produced block artifacts, a systematically high model, and
/// a residual with median -3 in normalized units.)
pub fn render_model(
    pos: &[f64],
    amp: &[f64],
    n: usize,
    sigma: f64,
    h: usize,
    w: usize,
    background: f64,
    truncate: f64,
) -> Vec<f64> {
    let mut m = vec![background; h * w];
    if n == 0 {
        return m;
    }
    let rad = (truncate * sigma).ceil() as i64;
    let mut f = psf::Factors::new(1, 1, 1);
    let (mut ay, mut ax, mut sub) = (Vec::new(), Vec::new(), Vec::new());

    for k in 0..n {
        let (cy, cx) = (pos[2 * k], pos[2 * k + 1]);
        let y0 = (cy.floor() as i64 - rad).max(0) as usize;
        let y1 = (((cy.ceil() as i64 + rad + 1).max(0)) as usize).min(h);
        let x0 = (cx.floor() as i64 - rad).max(0) as usize;
        let x1 = (((cx.ceil() as i64 + rad + 1).max(0)) as usize).min(w);
        if y1 <= y0 || x1 <= x0 {
            continue;
        }
        // GLOBAL pixel-centre axes: the emitter's centre is in global
        // coordinates, so the grid must be too.
        ay.clear();
        ay.extend((y0..y1).map(|v| v as f64));
        ax.clear();
        ax.extend((x0..x1).map(|v| v as f64));
        sub.clear();
        sub.resize((y1 - y0) * (x1 - x0), 0.0);
        f.ensure(ay.len(), ax.len(), 1);
        let theta = [0.0, amp[k], cy, cx];
        psf::model_ax(&theta, &ay, &ax, sigma, None, &mut f, &mut sub);
        for r in y0..y1 {
            for c in x0..x1 {
                m[r * w + c] += sub[(r - y0) * (x1 - x0) + (c - x0)];
            }
        }
    }
    m
}

/// The parameter-free contribution of frozen (out-of-patch) emitters, in the
/// patch's LOCAL coordinates.
///
/// **Not truncated**, unlike [`render_model`]: the patch is small, the halo is
/// the thing that keeps flux from being double-counted at patch borders, and
/// truncating it would reintroduce exactly the pedestal the 5-sigma radius
/// exists to eliminate. The patch's own background `b` is a free parameter
/// handled separately in its theta vector.
pub fn halo_image(
    pos: &[f64],
    amp: &[f64],
    frozen: &[u32],
    sigma: f64,
    y0: usize,
    x0: usize,
    h: usize,
    w: usize,
    out: &mut Vec<f64>,
) {
    out.clear();
    out.resize(h * w, 0.0);
    if frozen.is_empty() {
        return;
    }
    let mut theta = Vec::with_capacity(3 * frozen.len() + 1);
    theta.push(0.0);
    for &i in frozen {
        theta.push(amp[i as usize]);
        theta.push(pos[2 * i as usize] - y0 as f64);
        theta.push(pos[2 * i as usize + 1] - x0 as f64);
    }
    let (ay, ax) = (psf::local_axis(h), psf::local_axis(w));
    let mut f = psf::Factors::new(h, w, frozen.len());
    psf::model_ax(&theta, &ay, &ax, sigma, None, &mut f, out);
}

/// Pixels no emitter reaches: `true` where every emitter is farther than
/// `radius_factor * sigma`.
///
/// Exactly `free &= (yy-cy)^2 + (xx-cx)^2 > r2` accumulated over emitters, but
/// stamped rather than swept -- see the module docs.
pub fn emitter_free_mask(
    pos: &[f64],
    n: usize,
    sigma: f64,
    radius_factor: f64,
    h: usize,
    w: usize,
) -> Vec<bool> {
    let mut free = vec![true; h * w];
    let r = radius_factor * sigma;
    let r2 = r * r;
    for k in 0..n {
        let (cy, cx) = (pos[2 * k], pos[2 * k + 1]);
        let y0 = (cy - r).ceil().max(0.0) as usize;
        let y1 = (((cy + r).floor().max(-1.0) as i64 + 1).max(0) as usize).min(h);
        let x0 = (cx - r).ceil().max(0.0) as usize;
        let x1 = (((cx + r).floor().max(-1.0) as i64 + 1).max(0) as usize).min(w);
        for r_ in y0..y1 {
            let dy = r_ as f64 - cy;
            let dy2 = dy * dy;
            for c in x0..x1 {
                let dx = c as f64 - cx;
                if dy2 + dx * dx <= r2 {
                    free[r_ * w + c] = false;
                }
            }
        }
    }
    free
}
