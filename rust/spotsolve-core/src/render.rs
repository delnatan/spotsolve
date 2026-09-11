//! Rendering emitters into images, and the mask of pixels no emitter reaches.
//!
//! Ported from the retired Python reference's `calibrate.render_model` and
//! the masking loop of its `background_map`.
//!
//! Contributions from overlapping emitters ADD, which is what the physics says
//! and what every patch fit assumes locally. (An early version stitched
//! per-patch models by AVERAGING them where boxes overlapped and filled
//! patch-free pixels with the image median; that produced block artifacts, a
//! systematically high model and a residual with median -3 in normalized
//! units.)
//!
//! # Why the mask is stamped
//!
//! Written as `for each emitter: free &= (yy-cy)^2 + (xx-cx)^2 > r2` it touches
//! every pixel for every emitter -- `O(N*H*W)`. That is inside the 0.3% at
//! 128x128 with N=901 and emphatically outside it at 512x512, and it is the
//! same super-linear shape [P9] identifies in `_window`. Stamping only the
//! pixels within the radius makes it `O(N*sigma^2)`, and the result is
//! identical pixel for pixel.

use crate::psf;

/// Truncation radius for the model image, in each emitter's own sigma.
///
/// Each emitter is rendered only within this many sigma of its centre, so cost
/// is linear in emitter count rather than `N*H*W`.
pub const RENDER_TRUNCATE: f64 = 4.0;

/// The model image: `background` plus every emitter at its own width, each
/// rendered only within `truncate` of its own sigma.
#[allow(clippy::too_many_arguments)]
pub fn render_model(
    pos: &[f64],
    amp: &[f64],
    sig: &[f64],
    h: usize,
    w: usize,
    background: &[f64],
    truncate: f64,
) -> Vec<f64> {
    debug_assert_eq!(background.len(), h * w);
    let mut m = background.to_vec();
    let mut f = psf::Factors::new(1, 1, 1);
    let mut sub = Vec::new();
    for k in 0..amp.len() {
        let (cy, cx, sk) = (pos[2 * k], pos[2 * k + 1], sig[k]);
        let rad = (truncate * sk).ceil() as i64;
        let y0 = (cy.floor() as i64 - rad).max(0) as usize;
        let y1 = ((cy.ceil() as i64 + rad + 1).max(0) as usize).min(h);
        let x0 = (cx.floor() as i64 - rad).max(0) as usize;
        let x1 = ((cx.ceil() as i64 + rad + 1).max(0) as usize).min(w);
        if y1 <= y0 || x1 <= x0 {
            continue;
        }
        let ay: Vec<f64> = (y0..y1).map(|v| v as f64).collect();
        let ax: Vec<f64> = (x0..x1).map(|v| v as f64).collect();
        sub.clear();
        sub.resize(ay.len() * ax.len(), 0.0);
        f.ensure(ay.len(), ax.len(), 1);
        psf::model_ax(&[0.0, amp[k], cy, cx], &ay, &ax, sk, None, &mut f, &mut sub);
        for r in y0..y1 {
            for c in x0..x1 {
                m[r * w + c] += sub[(r - y0) * (x1 - x0) + (c - x0)];
            }
        }
    }
    m
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
