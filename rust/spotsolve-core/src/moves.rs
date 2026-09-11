//! Proposal constructors: pure transformations of a parameter vector.
//!
//! Ports `moves.py`. Each takes a `theta` and returns a new `theta` with one
//! more emitter. Nothing here fits, scores, or decides -- scoring lives in
//! `evidence` and the accept/reject loop in `passes`.
//!
//! Keeping proposals separate matters because they are where this method's
//! domain knowledge sits: a good proposal is what lets the optimizer reach the
//! two-emitter basin at all, and it is the part most worth testing on its own.

/// The amplitude floor a proposal may not go below.
///
/// A hard zero makes that emitter's position block of the Fisher matrix
/// identically zero, so `F` is singular and every evidence term built from it
/// is meaningless. A small positive floor keeps the fit well posed; an emitter
/// that does not want to be there is removed by `prune`, on the evidence,
/// rather than by silently collapsing.
pub const A_MIN: f64 = 1e-4;

/// Displacements, in sigma, at which a split is proposed.
///
/// These two bracket the only band still in question. Below ~1 sigma a pair is
/// not identifiable and the evidence refuses it; beyond ~2 sigma FIND already
/// produces a separate LoG peak and the split is redundant.
pub const SPLIT_DISPS: [f64; 2] = [1.0, 1.6];

/// Unit vector along the residual quadrupole around one emitter, and a
/// scale-free measure of how much that emitter looks like an unresolved pair.
///
/// # What this recovers
///
/// Two emitters closer than about 1.5 sigma are fitted well by one brighter
/// PSF, so their residual has **no peak** for a LoG filter to find -- it has a
/// *quadrupole*: negative in the middle, positive on two lobes along the pair
/// axis. The eigenvector of the PSF-weighted second moment of that residual
/// recovers the axis, which is where a split should be proposed. This is the
/// move FIND structurally cannot make.
///
/// `resid` is the incumbent's residual over an `h x w` window whose pixel (0,0)
/// is at `(y_origin, x_origin)`; `cy`/`cx` must be in that same frame. The
/// caller renders the residual, because it does not depend on which emitter is
/// being ranked while this is called once per emitter.
///
/// # Sign
///
/// The principal axis is defined only up to sign, and this returns the
/// canonical one (first non-zero component positive) rather than whatever
/// LAPACK happens to produce. Flipping it merely swaps which child of a split
/// is listed first, so it cannot change the configuration proposed -- but it
/// would change array ordering downstream, and ordering is load-bearing here.
///
/// `strength` is the largest eigenvalue normalized by the emitter's flux, and
/// is sign-independent.
///
/// # Known weakness
///
/// Background curvature produces a quadrupole too. On a strongly structured
/// background the ranking degrades and this move's advantage shrinks to near
/// zero. A projected score (marginal against the incumbent's full parameter
/// block) would plausibly do better -- but it must not be reused as a screen or
/// a seed.
#[allow(clippy::too_many_arguments)]
pub fn residual_axis(
    cy: f64,
    cx: f64,
    amplitude: f64,
    y_origin: f64,
    x_origin: f64,
    h: usize,
    w: usize,
    sigma: f64,
    resid: &[f64],
) -> ([f64; 2], f64) {
    debug_assert_eq!(resid.len(), h * w);
    let two_s2 = 2.0 * (1.5 * sigma) * (1.5 * sigma);
    let (mut m00, mut m01, mut m11) = (0.0f64, 0.0f64, 0.0f64);
    for r in 0..h {
        let dy = (r as f64 + y_origin) - cy;
        for c in 0..w {
            let dx = (c as f64 + x_origin) - cx;
            let wgt = (-(dy * dy + dx * dx) / two_s2).exp();
            let wr = wgt * resid[r * w + c];
            m00 += wr * dy * dy;
            m01 += wr * dy * dx;
            m11 += wr * dx * dx;
        }
    }
    if !(m00.is_finite() && m01.is_finite() && m11.is_finite()) {
        return ([1.0, 0.0], 0.0);
    }

    // Largest eigenvalue and eigenvector of the symmetric 2x2 [[m00, m01],
    // [m01, m11]], in closed form -- no general eigensolver at this size.
    let tr = m00 + m11;
    let diff = m00 - m11;
    let root = (diff * diff + 4.0 * m01 * m01).sqrt();
    let lambda = 0.5 * (tr + root);
    let mut u = if m01 != 0.0 {
        [m01, lambda - m00]
    } else if m00 >= m11 {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let norm = (u[0] * u[0] + u[1] * u[1]).sqrt();
    if norm > 0.0 {
        u[0] /= norm;
        u[1] /= norm;
    } else {
        u = [1.0, 0.0];
    }
    // Canonical sign; see the doc comment.
    if u[0] < 0.0 || (u[0] == 0.0 && u[1] < 0.0) {
        u[0] = -u[0];
        u[1] = -u[1];
    }
    (u, lambda / amplitude.max(1e-12))
}

/// Replace emitter `k` with two at `c_k +/- (disp/2) * u`, each of half its
/// flux.
///
/// Total flux is conserved by construction, which is what makes the amplitude
/// prior term in `log_bf_add` collapse to `-log(A_s)` -- charging a split
/// `A_child/A_s` as well would over-penalize precisely the move that resolves
/// close pairs.
///
/// The untouched emitters keep their order and the two children are appended,
/// so the returned vector is `[free without k] + [child0, child1]`.
pub fn split(theta: &[f64], k: usize, u: [f64; 2], disp: f64) -> Vec<f64> {
    let n = crate::psf::n_emitters(theta);
    debug_assert!(k < n);
    let mut out = Vec::with_capacity(3 * (n + 1) + 1);
    out.push(theta[0]);
    for j in 0..n {
        if j != k {
            out.push(crate::psf::amp(theta, j));
            out.push(crate::psf::cy(theta, j));
            out.push(crate::psf::cx(theta, j));
        }
    }
    let half = (crate::psf::amp(theta, k) / 2.0).max(A_MIN);
    let (cy, cx) = (crate::psf::cy(theta, k), crate::psf::cx(theta, k));
    for s in [1.0f64, -1.0] {
        out.push(half);
        out.push(cy + s * 0.5 * disp * u[0]);
        out.push(cx + s * 0.5 * disp * u[1]);
    }
    out
}

// ---------------------------------------------------------------------------
// Width-aware constructors, `theta = [b, (A, y, x, sigma) * K]`
// ---------------------------------------------------------------------------
//
// Ports `moves.py`'s `residual_axis_var` / `split_var` and adds the two
// constructors the group search needs that the pass-based search never had a
// use for: a birth and a removal as *proposals* rather than as pass outcomes.
//
// Nothing here fits, scores or decides. In particular a removal constructor is
// not a pruning rule: it builds the K-1 vector, and `dense_group` scores it by
// exactly the same expression it scores a birth by.

/// [`residual_axis`] on a variable-width theta, weighted at emitter `k`'s OWN
/// width.
///
/// The weight decides which pixels count as "around this emitter". Weighting a
/// defocused source at the in-focus width sees only its core -- exactly the
/// region where a broadened PSF and an unresolved pair look most alike -- so
/// the ranking this feeds would put defocused singles at the top, which is the
/// failure the free width exists to remove.
#[allow(clippy::too_many_arguments)]
pub fn residual_axis_var(
    theta: &[f64],
    k: usize,
    y_origin: f64,
    x_origin: f64,
    h: usize,
    w: usize,
    resid: &[f64],
) -> ([f64; 2], f64) {
    residual_axis(
        crate::psf::cy_var(theta, k),
        crate::psf::cx_var(theta, k),
        crate::psf::amp_var(theta, k),
        y_origin,
        x_origin,
        h,
        w,
        crate::psf::sigma_var(theta, k),
        resid,
    )
}

/// Replace emitter `k` with two at `c_k +/- (disp/2) * u`, each of half its
/// flux. Both children inherit the PARENT's width.
///
/// Not the in-focus width: a proposal and the incumbent it is compared against
/// must start in the same basin, because the comparison differences their two
/// objectives and a proposal started worse is under-credited -- the bias
/// `lmcl::fit`'s convergence note describes.
///
/// The untouched emitters keep their order and the two children are appended,
/// so the returned vector is `[free without k] + [child0, child1]`.
pub fn split_var(theta: &[f64], k: usize, u: [f64; 2], disp: f64) -> Vec<f64> {
    let n = crate::psf::n_emitters_var(theta);
    debug_assert!(k < n);
    let mut out = Vec::with_capacity(4 * (n + 1) + 1);
    out.push(theta[0]);
    for j in 0..n {
        if j != k {
            push_emitter_var(&mut out, theta, j);
        }
    }
    let half = (crate::psf::amp_var(theta, k) / 2.0).max(A_MIN);
    let (cy, cx) = (crate::psf::cy_var(theta, k), crate::psf::cx_var(theta, k));
    let sigma = crate::psf::sigma_var(theta, k);
    for s in [1.0f64, -1.0] {
        out.push(half);
        out.push(cy + s * 0.5 * disp * u[0]);
        out.push(cx + s * 0.5 * disp * u[1]);
        out.push(sigma);
    }
    out
}

/// Drop emitter `k`, keeping every other emitter's order.
///
/// The flux it carried is NOT redistributed. Handing it to a neighbour would
/// make the removal hypothesis a different move -- a merge -- started from a
/// state the joint refit did not choose, and the two would then be compared as
/// though they were the same proposal. The refit reassigns that light itself,
/// which is the whole reason every hypothesis is jointly refitted.
pub fn remove_var(theta: &[f64], k: usize) -> Vec<f64> {
    let n = crate::psf::n_emitters_var(theta);
    debug_assert!(k < n);
    let mut out = Vec::with_capacity(4 * (n - 1) + 1);
    out.push(theta[0]);
    for j in 0..n {
        if j != k {
            push_emitter_var(&mut out, theta, j);
        }
    }
    out
}

/// Append one emitter at `(y, x)` with flux `a` and width `sigma`.
pub fn birth_var(theta: &[f64], a: f64, y: f64, x: f64, sigma: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(theta.len() + 4);
    out.extend_from_slice(theta);
    out.push(a.max(A_MIN));
    out.push(y);
    out.push(x);
    out.push(sigma);
    out
}

#[inline]
fn push_emitter_var(out: &mut Vec<f64>, theta: &[f64], j: usize) {
    out.push(crate::psf::amp_var(theta, j));
    out.push(crate::psf::cy_var(theta, j));
    out.push(crate::psf::cx_var(theta, j));
    out.push(crate::psf::sigma_var(theta, j));
}
