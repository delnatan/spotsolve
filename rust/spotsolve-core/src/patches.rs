//! Group emitters into bounded joint-fit patches with fixed neighboring light.
//!
//! `LINK_FACTOR` sets connectivity, `K_MAX` caps free emitters, `HALO_FACTOR`
//! includes frozen neighbors, and `BBOX_PAD` supplies pixel context. Frozen
//! emitters contribute to the model but not to the fitted covariance.
//! These radii use the reference sigma; wide sources can extend beyond them.
//! Geometry fixtures verify the partition and coordinate conventions.

use crate::grid::EmitterGrid;

pub const LINK_FACTOR: f64 = 2.5;
pub const HALO_FACTOR: f64 = 5.0;
pub const BBOX_PAD: f64 = 3.0;
pub const K_MAX: usize = 12;

/// A pixel bounding box: `y0`/`x0` inclusive, `y1`/`x1` exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BBox {
    pub y0: usize,
    pub x0: usize,
    pub y1: usize,
    pub x1: usize,
}

impl BBox {
    #[inline]
    pub fn h(&self) -> usize {
        self.y1 - self.y0
    }
    #[inline]
    pub fn w(&self) -> usize {
        self.x1 - self.x0
    }
    #[inline]
    pub fn n_pixels(&self) -> usize {
        self.h() * self.w()
    }
}

/// A group of emitters fitted jointly, plus its pixel bounding box.
///
/// `indices` are the FREE emitters; `frozen` are neighbours just outside the
/// group, held fixed and folded into the model as a constant halo so flux is
/// not double-counted at patch borders and the joint Hessian stays bounded.
#[derive(Clone, Debug)]
pub struct Patch {
    pub indices: Vec<u32>,
    pub frozen: Vec<u32>,
    pub bbox: BBox,
}

fn bbox_for(ys: &[f64], xs: &[f64], pad: f64, h: usize, w: usize) -> BBox {
    let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..ys.len() {
        ymin = ymin.min(ys[i]);
        ymax = ymax.max(ys[i]);
        xmin = xmin.min(xs[i]);
        xmax = xmax.max(xs[i]);
    }
    BBox {
        y0: (ymin - pad).floor().max(0.0) as usize,
        x0: (xmin - pad).floor().max(0.0) as usize,
        y1: (((ymax + pad).ceil() as i64 + 1).max(0) as usize).min(h),
        x1: (((xmax + pad).ceil() as i64 + 1).max(0) as usize).min(w),
    }
}

struct UnionFind(Vec<u32>);

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind((0..n as u32).collect())
    }
    fn find(&mut self, mut i: u32) -> u32 {
        while self.0[i as usize] != i {
            self.0[i as usize] = self.0[self.0[i as usize] as usize];
            i = self.0[i as usize];
        }
        i
    }
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Always attach the larger root to the smaller, so the
            // representative of a component is its lowest member regardless of
            // the order pairs arrive in.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.0[hi as usize] = lo;
        }
    }
}

/// Group `n` emitters into jointly-fittable patches.
///
/// Components larger than `k_max` are bisected; see [`split_component`].
pub fn build_patches(
    pos: &[f64],
    n: usize,
    sigma: f64,
    h: usize,
    w: usize,
    k_max: usize,
) -> Vec<Patch> {
    if n == 0 {
        return Vec::new();
    }
    let halo_r = HALO_FACTOR * sigma;
    let link_r = LINK_FACTOR * sigma;
    let grid = EmitterGrid::build(pos, n, h, w, halo_r);

    let mut uf = UnionFind::new(n);
    let mut cand = Vec::new();
    for i in 0..n {
        grid.query_disc(pos, pos[2 * i], pos[2 * i + 1], link_r, &mut cand);
        for &j in &cand {
            if j as usize != i {
                uf.union(i as u32, j);
            }
        }
    }

    // Components, each with its members in ascending index order -- which is
    // what `np.nonzero(labels == c)` produces, and what `split_component`
    // assumes.
    let mut root_of: Vec<u32> = (0..n).map(|i| uf.find(i as u32)).collect();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| (root_of[i as usize], i));
    let mut patches = Vec::new();
    let mut start = 0usize;
    while start < order.len() {
        let r = root_of[order[start] as usize];
        let mut end = start;
        while end < order.len() && root_of[order[end] as usize] == r {
            end += 1;
        }
        let comp: Vec<u32> = order[start..end].to_vec();
        if comp.len() > k_max {
            for sub in split_component(&comp, pos, k_max) {
                patches.push(finalize(sub, pos, &grid, halo_r, BBOX_PAD * sigma, h, w));
            }
        } else {
            patches.push(finalize(comp, pos, &grid, halo_r, BBOX_PAD * sigma, h, w));
        }
        start = end;
    }
    root_of.clear();
    patches
}

/// Recursively bisect an oversized component by a spatial median split along
/// its longer axis, until every piece is at most `k_max`.
///
/// A proxy for "cut the weakest graph edge": splitting along the axis of
/// greatest spread tends to cut through the sparsest part of a spatially
/// clustered component.
fn split_component(idx: &[u32], pos: &[f64], k_max: usize) -> Vec<Vec<u32>> {
    if idx.len() <= k_max {
        return vec![idx.to_vec()];
    }
    let (mut ylo, mut yhi) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut xlo, mut xhi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &i in idx {
        ylo = ylo.min(pos[2 * i as usize]);
        yhi = yhi.max(pos[2 * i as usize]);
        xlo = xlo.min(pos[2 * i as usize + 1]);
        xhi = xhi.max(pos[2 * i as usize + 1]);
    }
    let axis = if (yhi - ylo) >= (xhi - xlo) { 0 } else { 1 };
    let mut order = idx.to_vec();
    // Stable, with the emitter index as an explicit tiebreak. numpy's argsort
    // defaults to an unstable quicksort, so matching it on ties is neither
    // possible nor desirable; being deterministic is.
    order.sort_by(|&a, &b| {
        let (va, vb) = (pos[2 * a as usize + axis], pos[2 * b as usize + axis]);
        va.partial_cmp(&vb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mid = order.len() / 2;
    let mut out = split_component(&order[..mid], pos, k_max);
    out.extend(split_component(&order[mid..], pos, k_max));
    out
}

fn finalize(
    idx: Vec<u32>,
    pos: &[f64],
    grid: &EmitterGrid,
    halo_r: f64,
    bbox_pad: f64,
    h: usize,
    w: usize,
) -> Patch {
    let ys: Vec<f64> = idx.iter().map(|&i| pos[2 * i as usize]).collect();
    let xs: Vec<f64> = idx.iter().map(|&i| pos[2 * i as usize + 1]).collect();
    let bbox = bbox_for(&ys, &xs, bbox_pad, h, w);
    let mut cand = Vec::new();
    let frozen = frozen_around(pos, grid, &bbox, halo_r, &idx, &mut cand);
    Patch {
        indices: idx,
        frozen,
        bbox,
    }
}

/// Emitters within `halo_r` of the bbox *rectangle* (not of its centre), less
/// those already free in `exclude`. Ascending index order.
fn frozen_around(
    pos: &[f64],
    grid: &EmitterGrid,
    bbox: &BBox,
    halo_r: f64,
    exclude: &[u32],
    cand: &mut Vec<u32>,
) -> Vec<u32> {
    grid.query_rect(
        bbox.y0 as f64,
        bbox.x0 as f64,
        (bbox.y1 - 1) as f64,
        (bbox.x1 - 1) as f64,
        halo_r,
        cand,
    );
    cand.iter()
        .copied()
        .filter(|i| !exclude.contains(i))
        .filter(|&i| dist_to_bbox(pos, i as usize, bbox) <= halo_r)
        .collect()
}

/// Distance from an emitter to the closed pixel rectangle `[y0, y1-1] x [x0, x1-1]`.
fn dist_to_bbox(pos: &[f64], i: usize, b: &BBox) -> f64 {
    let (y, x) = (pos[2 * i], pos[2 * i + 1]);
    let py = y.clamp(b.y0 as f64, (b.y1 - 1) as f64);
    let px = x.clamp(b.x0 as f64, (b.x1 - 1) as f64);
    ((y - py).powi(2) + (x - px).powi(2)).sqrt()
}
