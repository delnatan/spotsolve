//! Rendering emitters into images.
//!
//! Contributions from overlapping emitters add, as the physics says. Each
//! emitter is rendered only near its centre, so cost is linear in emitter
//! count rather than `N*H*W`.

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
