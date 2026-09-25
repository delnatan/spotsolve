//! Sparse linear assignment: the exact solver under the frame-to-frame linker.
//!
//! # The problem it solves
//!
//! Jaqaman et al. (2008, *Nat. Methods* 5:695) make one frame's linking a
//! square assignment by augmenting the `T x N` link block with the
//! alternatives -- terminate a track, start a track at a detection:
//!
//! ```text
//!             detections   terminate
//!     T   [    link      |  diag(term)  ]
//!     N   [  diag(birth) |      0       ]
//! ```
//!
//! With the bottom-right block ZERO (not u-track's transpose of the link
//! block, which counts every link twice -- tracksolve's `assign` docstring
//! has the 2x2 counterexample), the objective of any assignment is exactly
//! `sum(term) + sum(birth) + sum over links of g[t, d]`, where
//!
//! ```text
//!     g[t, d] = link[t, d] - term[t] - birth[d]
//! ```
//!
//! is the gain of linking over leaving both ends alone. So the augmented
//! square problem IS a maximum-gain bipartite matching over the gated pairs,
//! with no requirement that anything be matched. That problem stays sparse:
//! give every row a private "unmatched" column of gain 0, and it becomes a
//! rectangular assignment in which every row has its gated edges plus one
//! more. The dense `(T+N)^2` matrix -- and its zero block, which is what
//! makes the augmented form dense no matter how tight the gate -- never
//! exists.
//!
//! # The algorithm
//!
//! Shortest augmenting paths with dual potentials: the Jonker-Volgenant
//! family, in the form scipy's `linear_sum_assignment` uses (Crouse 2016,
//! *IEEE TAES* 52:1679), with the dense column scan replaced by Dijkstra on a
//! binary heap over the columns actually reachable. Rows are inserted one at
//! a time; each insertion finds the cheapest alternating path from the new
//! row to a free column, updates the potentials so every reduced cost stays
//! non-negative, and flips the path. Only rows already inserted are ever
//! re-scanned, and their reduced costs are non-negative by the invariant, so
//! Dijkstra is exact; the new row's own edges may be negative, which only
//! shifts every path by the same constant.
//!
//! A row can always reach its own private column, so the problem is never
//! infeasible and no sentinel "forbidden" cost exists anywhere: a pair the
//! gate rejected is simply not an edge.
//!
//! Exactness is the whole point -- tracksolve measured that greedy matching
//! over the SAME scores costs 1-2 switches per 100 links at step/NN 0.5 -- so
//! the tests below check it against brute-force enumeration of every partial
//! matching, not against another solver.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

const NONE: usize = usize::MAX;

/// One tentative column distance in the heap. `BinaryHeap` is a max-heap,
/// so the order is reversed on distance. Among equal distances a FREE column
/// wins (scipy's rule: stop at the first free column rather than walk
/// further along an equally cheap path), then the lower index, so the result
/// is deterministic.
#[derive(PartialEq)]
struct Entry {
    dist: f64,
    free: bool,
    col: usize,
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        o.dist
            .total_cmp(&self.dist)
            .then(self.free.cmp(&o.free))
            .then(o.col.cmp(&self.col))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Maximum-gain matching of rows to `n_cols` columns over sparse edges.
///
/// Row `i`'s edges are `cols[row_ptr[i]..row_ptr[i+1]]` with gains
/// `gain[...]` (CSR; at most one edge per row-column pair). Returns, for each
/// row, its matched column, or `None` when it is better left unmatched.
/// An edge of gain `<= 0` is never needed -- leaving both ends unmatched is
/// at least as good -- so callers may omit those.
pub fn max_gain_matching(
    n_cols: usize,
    row_ptr: &[usize],
    cols: &[usize],
    gain: &[f64],
) -> Vec<Option<usize>> {
    solve(n_cols, row_ptr, cols, gain)
        .into_iter()
        .map(|j| (j < n_cols).then_some(j))
        .collect()
}

/// Each row's column; `n_cols + row` is that row's private unmatched column.
fn solve(n_cols: usize, row_ptr: &[usize], cols: &[usize], gain: &[f64]) -> Vec<usize> {
    let n_rows = row_ptr.len().saturating_sub(1);
    // Real columns, then row i's private "unmatched" column at n_cols + i.
    let nc = n_cols + n_rows;
    let mut u = vec![0.0f64; n_rows];
    let mut v = vec![0.0f64; nc];
    let mut col4row = vec![NONE; n_rows];
    let mut row4col = vec![NONE; nc];
    let mut dist = vec![f64::INFINITY; nc];
    let mut path = vec![NONE; nc];
    let mut done = vec![false; nc];
    let mut touched: Vec<usize> = Vec::new();
    let mut sr: Vec<usize> = Vec::new();
    let mut sc: Vec<usize> = Vec::new();
    let mut heap = BinaryHeap::new();

    for cur in 0..n_rows {
        if row_ptr[cur] == row_ptr[cur + 1] {
            // Nothing to link: its private column is free and costs 0.
            col4row[cur] = n_cols + cur;
            row4col[n_cols + cur] = cur;
            continue;
        }
        let mut i = cur;
        let mut min_val = 0.0f64;
        let sink = loop {
            sr.push(i);
            {
                let mut relax = |j: usize, cost: f64| {
                    if done[j] {
                        return;
                    }
                    let r = min_val + cost - u[i] - v[j];
                    if r < dist[j] {
                        if dist[j] == f64::INFINITY {
                            touched.push(j);
                        }
                        dist[j] = r;
                        path[j] = i;
                        heap.push(Entry {
                            dist: r,
                            free: row4col[j] == NONE,
                            col: j,
                        });
                    }
                };
                for k in row_ptr[i]..row_ptr[i + 1] {
                    relax(cols[k], -gain[k]);
                }
                relax(n_cols + i, 0.0);
            }

            let j = loop {
                let e = heap
                    .pop()
                    .expect("a row always reaches its own private column");
                if !done[e.col] && e.dist <= dist[e.col] {
                    break e.col;
                }
            };
            min_val = dist[j];
            done[j] = true;
            sc.push(j);
            if row4col[j] == NONE {
                break j;
            }
            i = row4col[j];
        };

        u[cur] += min_val;
        for &i in &sr {
            if i != cur {
                u[i] += min_val - dist[col4row[i]];
            }
        }
        for &j in &sc {
            v[j] -= min_val - dist[j];
        }

        let mut j = sink;
        loop {
            let i = path[j];
            row4col[j] = i;
            std::mem::swap(&mut col4row[i], &mut j);
            if i == cur {
                break;
            }
        }

        for &j in &touched {
            dist[j] = f64::INFINITY;
            done[j] = false;
        }
        touched.clear();
        sr.clear();
        sc.clear();
        heap.clear();
    }

    col4row
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

    fn total(m: &[Option<usize>], row_ptr: &[usize], cols: &[usize], gain: &[f64]) -> f64 {
        m.iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let c = (*c)?;
                let k = (row_ptr[i]..row_ptr[i + 1]).find(|&k| cols[k] == c)?;
                Some(gain[k])
            })
            .sum()
    }

    /// Best total over every partial matching, by exhaustive recursion.
    fn brute(
        i: usize,
        used: &mut Vec<bool>,
        row_ptr: &[usize],
        cols: &[usize],
        gain: &[f64],
    ) -> f64 {
        if i + 1 == row_ptr.len() {
            return 0.0;
        }
        let mut best = brute(i + 1, used, row_ptr, cols, gain);
        for k in row_ptr[i]..row_ptr[i + 1] {
            let c = cols[k];
            if !used[c] {
                used[c] = true;
                best = best.max(gain[k] + brute(i + 1, used, row_ptr, cols, gain));
                used[c] = false;
            }
        }
        best
    }

    /// Random sparse problems with mixed-sign gains, against enumeration.
    /// Mixed signs matter: a negative-gain edge must never be taken, and a
    /// positive one must sometimes be given up for a better pair elsewhere.
    #[test]
    fn matches_brute_force_on_random_problems() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for trial in 0..3000 {
            let n_rows = 1 + (rng.uniform() * 7.0) as usize;
            let n_cols = 1 + (rng.uniform() * 7.0) as usize;
            let density = 0.2 + 0.8 * rng.uniform();
            let mut row_ptr = vec![0];
            let (mut cols, mut gain) = (vec![], vec![]);
            for _ in 0..n_rows {
                for c in 0..n_cols {
                    if rng.uniform() < density {
                        cols.push(c);
                        gain.push(4.0 * rng.uniform() - 1.0);
                    }
                }
                row_ptr.push(cols.len());
            }
            let m = max_gain_matching(n_cols, &row_ptr, &cols, &gain);
            let mut seen = vec![false; n_cols];
            for c in m.iter().flatten() {
                assert!(!seen[*c], "trial {trial}: column {c} used twice");
                seen[*c] = true;
            }
            let got = total(&m, &row_ptr, &cols, &gain);
            let want = brute(0, &mut vec![false; n_cols], &row_ptr, &cols, &gain);
            assert!(
                (got - want).abs() < 1e-9,
                "trial {trial}: got {got}, optimum {want}"
            );
        }
    }

    /// The classic trap for nearest-neighbour linking: each row's best column
    /// is the same one, and the optimum gives it to the row that loses most
    /// without it.
    #[test]
    fn gives_a_contested_column_to_the_row_that_needs_it() {
        // row 0: col 0 (5), col 1 (4); row 1: col 0 (5) only.
        let m = max_gain_matching(2, &[0, 2, 3], &[0, 1, 0], &[5.0, 4.0, 5.0]);
        assert_eq!(m, vec![Some(1), Some(0)]);
    }

    #[test]
    fn empty_and_edgeless_rows() {
        assert!(max_gain_matching(3, &[0], &[], &[]).is_empty());
        assert_eq!(max_gain_matching(0, &[0, 0, 0], &[], &[]), vec![None, None]);
        assert_eq!(
            max_gain_matching(1, &[0, 0, 1], &[0], &[-2.0]),
            vec![None, None]
        );
    }
}
