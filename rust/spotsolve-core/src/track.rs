//! Frame-to-frame linking by least squared displacement (Crocker & Grier
//! 1996, *J. Colloid Interface Sci.* 179:298).
//!
//! Between consecutive frames the links minimize
//!
//! ```text
//!     sum over links of d^2  +  R^2 * (tracks ended)
//! ```
//!
//! where `R` is the largest step a particle may take. For Brownian particles
//! with one diffusion coefficient and uniform localization error, every step
//! has the same Gaussian variance, so the summed squared displacement is the
//! negative log likelihood of the assignment up to a scale, and `R^2` prices
//! a track end in the same units. `R` is a physical setting, not a fitted
//! parameter: it trades broken tracks (too small) against identity switches
//! (too large).
//!
//! Ending a track costs `R^2`, so a link of length `d` gains `R^2 - d^2` over
//! leaving both ends alone; the frame's assignment is a maximum-gain matching
//! over pairs closer than `R`, solved exactly by [`lap`]. A frame with no
//! detections ends every track. Units are the caller's (pixels).

use crate::lap;

/// Track id per detection, in input order, numbered by first appearance.
pub struct Linking {
    pub track: Vec<u32>,
    pub n_tracks: u32,
}

/// One frame's detections bucketed into square cells of side `R`, sorted by
/// `(cell row, cell column)`. A disc of radius `R` touches at most three
/// adjacent cells in each of three cell rows, and each such run of three is
/// contiguous in this order.
struct Cells {
    keys: Vec<(i64, i64, usize)>,
}

impl Cells {
    /// `keys` hold each detection's index within `rows`.
    fn new(rows: &[usize], pos: &[f64], r: f64) -> Self {
        let mut keys: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(k, &i)| ((pos[2 * i] / r).floor() as i64, (pos[2 * i + 1] / r).floor() as i64, k))
            .collect();
        keys.sort_unstable();
        Cells { keys }
    }

    fn near(&self, q: [f64; 2], r: f64, mut f: impl FnMut(usize)) {
        let (cy, cx) = ((q[0] / r).floor() as i64, (q[1] / r).floor() as i64);
        for iy in [cy.saturating_sub(1), cy, cy.saturating_add(1)] {
            let lo = self.keys.partition_point(|k| (k.0, k.1) < (iy, cx.saturating_sub(1)));
            let hi = self.keys.partition_point(|k| (k.0, k.1) <= (iy, cx.saturating_add(1)));
            for k in &self.keys[lo..hi] {
                f(k.2);
            }
        }
    }
}

/// Links `(y, x)` positions (`pos` is `(N, 2)` row-major) by frame index.
/// Only consecutive frames are linked.
pub fn link(frame: &[i64], pos: &[f64], max_step: f64) -> Result<Linking, String> {
    let n = frame.len();
    if pos.len() != 2 * n {
        return Err(format!("frame has {n} rows but positions have {}", pos.len() / 2));
    }
    if !(max_step.is_finite() && max_step > 0.0) {
        return Err(format!("max_step must be positive and finite, got {max_step}"));
    }
    if let Some(i) = (0..2 * n).find(|&i| !pos[i].is_finite()) {
        return Err(format!("row {} has a non-finite position", i / 2));
    }
    let r2 = max_step * max_step;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| frame[i]);

    let mut track = vec![0u32; n];
    let mut n_tracks = 0u32;
    let mut prev: &[usize] = &[];
    let mut prev_frame = i64::MIN;
    let mut start = 0;
    while start < n {
        let f = frame[order[start]];
        let end = start + order[start..].partition_point(|&i| frame[i] == f);
        let cur = &order[start..end];
        let mut taken = vec![false; cur.len()];
        if !prev.is_empty() && prev_frame.checked_add(1) == Some(f) {
            let cells = Cells::new(cur, pos, max_step);
            let mut row_ptr = vec![0usize];
            let (mut cols, mut gain) = (Vec::new(), Vec::new());
            let mut edges: Vec<(usize, f64)> = Vec::new();
            for &a in prev {
                let q = [pos[2 * a], pos[2 * a + 1]];
                edges.clear();
                cells.near(q, max_step, |k| {
                    let b = cur[k];
                    let (dy, dx) = (pos[2 * b] - q[0], pos[2 * b + 1] - q[1]);
                    let g = r2 - (dy * dy + dx * dx);
                    if g > 0.0 {
                        edges.push((k, g));
                    }
                });
                edges.sort_unstable_by_key(|e| e.0);
                for &(c, g) in &edges {
                    cols.push(c);
                    gain.push(g);
                }
                row_ptr.push(cols.len());
            }
            for (&a, m) in prev.iter().zip(lap::max_gain_matching(cur.len(), &row_ptr, &cols, &gain)) {
                if let Some(c) = m {
                    track[cur[c]] = track[a];
                    taken[c] = true;
                }
            }
        }
        for (k, &i) in cur.iter().enumerate() {
            if !taken[k] {
                track[i] = n_tracks;
                n_tracks += 1;
            }
        }
        prev = cur;
        prev_frame = f;
        start = end;
    }
    Ok(Linking { track, n_tracks })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Two frames' total cost, `sum d^2 + R^2 * ended`, for a linking.
    fn cost(frame: &[i64], pos: &[f64], r: f64, l: &Linking) -> f64 {
        let n0 = frame.iter().filter(|&&f| f == 0).count();
        let mut total = 0.0;
        for a in 0..n0 {
            let next = (n0..frame.len()).find(|&b| l.track[b] == l.track[a]);
            total += match next {
                Some(b) => (pos[2 * a] - pos[2 * b]).powi(2) + (pos[2 * a + 1] - pos[2 * b + 1]).powi(2),
                None => r * r,
            };
        }
        total
    }

    /// Every partial matching of `n0` rows to `n1` columns within `r`.
    fn brute(pos: &[f64], n0: usize, n1: usize, r: f64) -> f64 {
        fn go(a: usize, used: &mut Vec<bool>, pos: &[f64], n0: usize, n1: usize, r: f64) -> f64 {
            if a == n0 {
                return 0.0;
            }
            let mut best = r * r + go(a + 1, used, pos, n0, n1, r);
            for b in 0..n1 {
                let j = n0 + b;
                let d2 = (pos[2 * a] - pos[2 * j]).powi(2) + (pos[2 * a + 1] - pos[2 * j + 1]).powi(2);
                if !used[b] && d2 < r * r {
                    used[b] = true;
                    best = best.min(d2 + go(a + 1, used, pos, n0, n1, r));
                    used[b] = false;
                }
            }
            best
        }
        go(0, &mut vec![false; n1], pos, n0, n1, r)
    }

    #[test]
    fn each_frame_pair_is_the_least_squares_assignment() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for _ in 0..300 {
            let (n0, n1) = (1 + (rng.uniform() * 6.0) as usize, 1 + (rng.uniform() * 6.0) as usize);
            let frame: Vec<i64> = (0..n0 + n1).map(|i| (i >= n0) as i64).collect();
            let pos: Vec<f64> = (0..2 * (n0 + n1)).map(|_| rng.uniform() * 4.0).collect();
            let r = 0.5 + rng.uniform() * 2.0;
            let l = link(&frame, &pos, r).unwrap();
            let got = cost(&frame, &pos, r, &l);
            let want = brute(&pos, n0, n1, r);
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }

    #[test]
    fn a_step_longer_than_max_step_is_never_a_link() {
        let l = link(&[0, 1, 2], &[0.0, 0.0, 0.0, 1.9, 0.0, 4.0], 2.0).unwrap();
        assert_eq!(l.track, vec![0, 0, 1]);
    }

    #[test]
    fn a_contested_detection_goes_to_the_assignment_with_least_total_motion() {
        // Nearest-first would take the 0.5 px link and end the other track
        // (0.25 + 1.44); the optimum links both (0.49 + 0.81).
        let pos = [0.0, 0.0, 0.0, 1.2, 0.0, 0.7, 0.0, 2.1];
        let l = link(&[0, 0, 1, 1], &pos, 1.2).unwrap();
        assert_eq!(l.track[2], l.track[0]);
        assert_eq!(l.track[3], l.track[1]);
    }

    #[test]
    fn an_empty_frame_ends_every_track_and_order_does_not_matter() {
        let l = link(&[0, 2], &[0.0, 0.0, 0.0, 0.1], 1.0).unwrap();
        assert_eq!(l.n_tracks, 2);
        let frame = [3, 1, 2, 1, 3, 2];
        let pos = [0.0, 0.2, 5.0, 5.0, 0.0, 0.1, 0.0, 0.0, 5.1, 5.0, 5.0, 5.2];
        let l = link(&frame, &pos, 1.0).unwrap();
        assert_eq!(l.track, vec![1, 0, 1, 1, 0, 0]);
    }

    #[test]
    fn bad_inputs_are_refused() {
        assert!(link(&[0], &[0.0], 1.0).is_err());
        assert!(link(&[0], &[0.0, f64::NAN], 1.0).is_err());
        assert!(link(&[0], &[0.0, 0.0], 0.0).is_err());
        assert!(link(&[], &[], 1.0).unwrap().n_tracks == 0);
    }
}
