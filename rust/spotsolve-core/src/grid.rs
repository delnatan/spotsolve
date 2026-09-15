//! A uniform-cell spatial index over the committed emitters.
//!
//! # Why this exists
//!
//! `spotsolve` is linear in area at fixed density -- measured us/px is flat from
//! 62x62 to 128x128 -- with exactly one exception, and it is a data-structure
//! problem rather than an algorithmic one. `_window` measures a candidate
//! against **every** committed emitter, so it is `O(N)` per proposal and
//! `O(N^2)` per round:
//!
//! | size | N | `_window` share |
//! |------|-----|-----|
//! | 39 | 84 | 1.2% |
//! | 62 | 211 | 1.3% |
//! | 96 | 507 | 2.1% |
//! | 128 | 901 | 3.0% |
//!
//! Small then; not small at 512x512. The box search's ownership masks and
//! patch decomposition have the same shape, and use this index [P9].
//!
//! # Layout
//!
//! Positions are `(N, 2)` row-major `(y, x)`, i.e. a flat slice where emitter
//! `i` is `pos[2*i]`, `pos[2*i + 1]`. That is numpy's own layout, so it crosses
//! the PyO3 boundary with no copy.
//!
//! Cells are CSR: `items[starts[c] .. starts[c+1]]` are the emitters in cell
//! `c`. The cell side is `HALO_FACTOR * sigma`, which is the largest radius any
//! query uses, so every query is a small fixed neighbourhood walk.
//!
//! # Determinism
//!
//! A query visits cells in raster order, so candidates come out grouped by
//! cell, not in ascending emitter index. Callers that need a defined order must
//! impose it -- and several do, because ordering is load-bearing in this
//! pipeline. [`EmitterGrid::query_rect`] sorts, because every caller wants that.

/// Emitters bucketed on a uniform grid.
///
/// Buckets are per-cell `Vec`s rather than a packed CSR array, because the ADD
/// pass **mutates** the committed set as it goes: every acceptance writes back
/// its whole window's refit and appends one emitter. Rebuilding a packed index
/// after each acceptance would be `O(N)` per acceptance and `O(N^2)` per pass
/// -- exactly the cost this index exists to remove. [`EmitterGrid::relocate`]
/// and [`EmitterGrid::insert`] keep it current in `O(1)`.
///
/// Cell contents are **not** kept sorted; [`EmitterGrid::query_rect`] sorts its
/// output instead, which is where a defined order is actually needed.
///
/// Indices are positions into the caller's arrays. Any operation that
/// renumbers emitters (compacting after a prune, say) invalidates the grid and
/// requires a rebuild.
pub struct EmitterGrid {
    cell: f64,
    ny: usize,
    nx: usize,
    cells: Vec<Vec<u32>>,
}

impl EmitterGrid {
    /// Bucket `n` emitters over an `h x w` image with the given cell side.
    ///
    /// `cell` should be the largest query radius in use (`HALO_FACTOR * sigma`),
    /// so no query needs to walk more than a 3x3 neighbourhood of cells.
    pub fn build(pos: &[f64], n: usize, h: usize, w: usize, cell: f64) -> Self {
        debug_assert!(pos.len() >= 2 * n);
        let cell = cell.max(1e-6);
        let ny = ((h as f64 / cell).ceil() as usize).max(1);
        let nx = ((w as f64 / cell).ceil() as usize).max(1);
        let mut g = Self {
            cell,
            ny,
            nx,
            cells: vec![Vec::new(); ny * nx],
        };
        for i in 0..n {
            g.insert(i as u32, pos[2 * i], pos[2 * i + 1]);
        }
        g
    }

    #[inline]
    fn cell_of(&self, y: f64, x: f64) -> usize {
        let iy = ((y / self.cell).floor().max(0.0) as usize).min(self.ny - 1);
        let ix = ((x / self.cell).floor().max(0.0) as usize).min(self.nx - 1);
        iy * self.nx + ix
    }

    pub fn insert(&mut self, i: u32, y: f64, x: f64) {
        let c = self.cell_of(y, x);
        self.cells[c].push(i);
    }

    pub fn remove(&mut self, i: u32, y: f64, x: f64) {
        let c = self.cell_of(y, x);
        if let Some(k) = self.cells[c].iter().position(|&v| v == i) {
            self.cells[c].swap_remove(k);
        }
    }

    /// Move emitter `i` from one position to another. A no-op when the two
    /// positions fall in the same cell, which is the common case: a refit moves
    /// an emitter a fraction of a pixel and the cell is `5*sigma` across.
    pub fn relocate(&mut self, i: u32, old_y: f64, old_x: f64, new_y: f64, new_x: f64) {
        let (from, to) = (self.cell_of(old_y, old_x), self.cell_of(new_y, new_x));
        if from == to {
            return;
        }
        if let Some(k) = self.cells[from].iter().position(|&v| v == i) {
            self.cells[from].swap_remove(k);
        }
        self.cells[to].push(i);
    }

    /// Emitters whose cell overlaps the pixel rectangle `[y0, y1] x [x0, x1]`
    /// grown by `margin`, written to `out` in ascending index order.
    ///
    /// This is a superset of the emitters actually within `margin` of the
    /// rectangle -- the caller must still apply the exact distance test. The
    /// index only replaces the scan over all `N`.
    pub fn query_rect(&self, y0: f64, x0: f64, y1: f64, x1: f64, margin: f64, out: &mut Vec<u32>) {
        out.clear();
        let lo_y = (((y0 - margin) / self.cell).floor().max(0.0) as usize).min(self.ny - 1);
        let hi_y = (((y1 + margin) / self.cell).floor().max(0.0) as usize).min(self.ny - 1);
        let lo_x = (((x0 - margin) / self.cell).floor().max(0.0) as usize).min(self.nx - 1);
        let hi_x = (((x1 + margin) / self.cell).floor().max(0.0) as usize).min(self.nx - 1);
        for iy in lo_y..=hi_y {
            for ix in lo_x..=hi_x {
                out.extend_from_slice(&self.cells[iy * self.nx + ix]);
            }
        }
        out.sort_unstable();
    }

    /// Emitters within `radius` of the point `(y, x)`, in ascending index
    /// order. Exact: the distance test is applied here.
    pub fn query_disc(&self, pos: &[f64], y: f64, x: f64, radius: f64, out: &mut Vec<u32>) {
        self.query_rect(y, x, y, x, radius, out);
        let r2 = radius * radius;
        out.retain(|&i| {
            let (dy, dx) = (pos[2 * i as usize] - y, pos[2 * i as usize + 1] - x);
            dy * dy + dx * dx <= r2
        });
    }
}
