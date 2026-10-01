//! Model images.

use crate::psf;

/// Each emitter is rendered within this many of its widths of its centre.
pub const RENDER_TRUNCATE: f64 = 4.0;

/// `background` (`h x w`) plus every emitter at its own width, each within
/// `truncate` widths of its centre. `pos` is `2N` row-major `(y, x)`.
pub fn render_model(pos: &[f64], amp: &[f64], sig: &[f64], h: usize, w: usize, background: &[f64], truncate: f64) -> Vec<f64> {
    assert_eq!(background.len(), h * w);
    let mut m = background.to_vec();
    let mut sub = Vec::new();
    for k in 0..amp.len() {
        let (cy, cx, s) = (pos[2 * k], pos[2 * k + 1], sig[k]);
        let rad = (truncate * s).ceil();
        let span = |c: f64, n: usize| ((c.floor() - rad).max(0.0) as usize, ((c.ceil() + rad + 1.0).max(0.0) as usize).min(n));
        let ((y0, y1), (x0, x1)) = (span(cy, h), span(cx, w));
        if y1 <= y0 || x1 <= x0 {
            continue;
        }
        let ay: Vec<f64> = (y0..y1).map(|v| v as f64).collect();
        let ax: Vec<f64> = (x0..x1).map(|v| v as f64).collect();
        sub.clear();
        sub.resize(ay.len() * ax.len(), 0.0);
        psf::add_emitter(&mut sub, &ay, &ax, amp[k], cy, cx, s);
        for (r, row) in (y0..y1).zip(sub.chunks_exact(ax.len())) {
            for (v, d) in m[r * w + x0..r * w + x1].iter_mut().zip(row) {
                *v += d;
            }
        }
    }
    m
}
