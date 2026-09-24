"""PROTOTYPE (2026-09-24), frozen: the joint-model localizer the Rust port
(`rust/spotsolve-core/src/joint.rs`) reproduces. Frozen configuration:
KG 12, PRE_BG, OMEGA 1.5, EMP_NULL, REMOVE with REMOVE_U = threshold(sigma),
starting from the one-pass result, add_after_converged, max_outer 80, tol 1e-2.

Joint-model localizer prototype.

Model: m = B(beta) + sum_k A_k g(p; y_k, x_k, w_k), B bilinear on a TILE-px
node lattice. One likelihood for the whole frame; blocks:
  - coupled groups of emitters (connected components at LINK * mean width),
    each fitted by LM with everything else fixed (halo);
  - background nodes, Poisson IRLS with emitters fixed.
Groups are weakly coupled to each other and to the nodes by construction, so
the outer block loop contracts fast; inside a group LM handles the coupling.
Count: score-gated adds on each group's region (efficient score, same gate).
"""
import time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import spotsolve_rs as _rs
from spotsolve import scoregate as sg, psf

TILE = 16
LINK = 4.0        # x mean width: pairs closer than this are fitted together
PAD = 3.0         # x max width: group box margin (support)
OWN = sg.OWN      # sigma: adds are tested within this of a member


def bilinear(H, W, tile=TILE):
    """Sparse (H*W, n_nodes) interpolation matrix and node grid shape."""
    ny, nx = int(np.ceil((H - 1) / tile)) + 1, int(np.ceil((W - 1) / tile)) + 1
    yy, xx = np.mgrid[:H, :W]
    fy, fx = yy / tile, xx / tile
    iy, ix = np.minimum(fy.astype(int), ny - 2) if ny > 1 else np.zeros_like(yy), \
        np.minimum(fx.astype(int), nx - 2) if nx > 1 else np.zeros_like(xx)
    ty, tx = fy - iy, fx - ix
    rows, cols, vals = [], [], []
    p = (yy * W + xx).ravel()
    for dy, dx, wv in [(0, 0, (1 - ty) * (1 - tx)), (1, 0, ty * (1 - tx)), (0, 1, (1 - ty) * tx), (1, 1, ty * tx)]:
        n = np.minimum(iy + dy, ny - 1) * nx + np.minimum(ix + dx, nx - 1)
        rows.append(p); cols.append(n.ravel()); vals.append(wv.ravel())
    M = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(H * W, ny * nx))
    return M


def render_box(E, y0, x0, h, w):
    if len(E) == 0:
        return np.zeros((h, w))
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w].astype(float)
    return psf.model_var_sigma(psf.pack_var_sigma(0.0, E[:, 0], E[:, 1], E[:, 2], E[:, 3]), yy, xx)


def render_all(E, H, W):
    out = np.zeros((H, W))
    for e in E:
        r = int(np.ceil(4 * e[3])) + 1
        y0, x0 = max(int(e[1]) - r, 0), max(int(e[2]) - r, 0)
        y1, x1 = min(int(e[1]) + r + 2, H), min(int(e[2]) + r + 2, W)
        out[y0:y1, x0:x1] += render_box(e[None], y0, x0, y1 - y0, x1 - x0)
    return out


ADD_FREE_LEVEL = False
EMP_NULL = False
REMOVE = False     # backward test after convergence, same calibrated threshold
REMOVE_U = None    # removal threshold; None = u*kappa (same as adds), else this z
PSF_ERR = False    # null variance phi*m + (alpha*L)^2, alpha measured near emitters
GROUP_FREE_LEVEL = False
OMEGA = 1.0        # over-relaxation of the node update
PRE_BG = False     # fit nodes to the initial emitters before the first round


PROFILE = None   # radial excess variance per unit flux^2, bins of 0.5 sigma (set by wing_variance)
WING_EDGES = np.arange(0, 9.5, 0.5)


def wing_render(E, P, sigma, H, W):
    """sum_k F_k^2 P(r_k / sigma) on the frame."""
    out = np.zeros((H, W))
    R = int(np.ceil(9 * sigma))
    for e in E:
        y0, x0 = max(int(e[1]) - R, 0), max(int(e[2]) - R, 0)
        y1, x1 = min(int(e[1]) + R + 2, H), min(int(e[2]) + R + 2, W)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(yy - e[1], xx - e[2]) / sigma
        b = np.clip(np.digitize(rr, WING_EDGES) - 1, 0, len(P) - 1)
        out[y0:y1, x0:x1] += e[0] ** 2 * np.where(rr < 9, P[b], 0.0)
    return out


def wing_variance(d, m, E, phi, sigma, n_min=5):
    """Measure the PSF wing profile from bright isolated in-focus emitters and render
    its variance for every emitter. None when the frame has too few such emitters."""
    global PROFILE
    H, W = d.shape
    res = d - m
    dnn = cKDTree(E[:, 1:3]).query(E[:, 1:3], k=2)[0][:, 1] if len(E) > 1 else np.full(len(E), np.inf)
    F = E[:, 0]
    sel = np.nonzero((F > np.percentile(F, 75)) & (dnn > 7 * sigma) & (E[:, 3] < 1.3 * sigma)
                     & (E[:, 1] > 10) & (E[:, 1] < H - 10) & (E[:, 2] > 10) & (E[:, 2] < W - 10))[0]
    if len(sel) < n_min:
        PROFILE = None
        return None
    nb = len(WING_EDGES) - 1
    S2 = np.zeros(nb); N = np.zeros(nb)
    yy, xx = np.mgrid[:H, :W]
    for k in sel:
        rr = np.hypot(yy - E[k, 1], xx - E[k, 2]) / sigma; s_ = rr < 9
        b = np.digitize(rr[s_], WING_EDGES) - 1
        S2 += np.bincount(b, (res[s_] ** 2 - phi * m[s_]) / F[k] ** 2, nb); N += np.bincount(b, None, nb)
    P = np.where(N > 0, S2 / np.maximum(N, 1), 0.0)
    P = np.convolve(np.r_[P[0], P, P[-1]], np.ones(3) / 3, mode="valid")
    P[WING_EDGES[:-1] < 1.5] = 0.0            # the core belongs to the fit
    PROFILE = np.maximum(P, 0.0)
    return wing_render(E, PROFILE, sigma, H, W)


def bg_step(M, beta, d, light, omega):
    H, W = d.shape
    b = beta.copy()
    for _ in range(3):
        m = np.maximum((M @ b).reshape(H, W) + light, 1e-3).ravel()
        Wt = sparse.diags(1.0 / m)
        b = np.maximum(b + sparse.linalg.spsolve((M.T @ Wt @ M).tocsc(), M.T @ ((d.ravel() - m) / m)), 1e-3)
    b = np.maximum(beta + omega * (b - beta), 1e-3)
    return b, (M @ b).reshape(H, W)


KG = None          # group size cap; None = distance components (old)
RHO_MIN = 0.05     # pairs coupled weaker than this are never linked


def pair_rho2(E, pairs, model):
    """Squared first canonical correlation of each pair's (A, y, x, w) blocks,
    from their Fisher matrix on the union of their 3-width supports."""
    H, W = model.shape
    out = np.empty(len(pairs))
    for n, (i, j) in enumerate(pairs):
        P = E[[i, j]]
        r = 3.0 * P[:, 3].max()
        y0, x0 = max(int(P[:, 1].min() - r), 0), max(int(P[:, 2].min() - r), 0)
        y1, x1 = min(int(P[:, 1].max() + r) + 2, H), min(int(P[:, 2].max() + r) + 2, W)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
        J = psf.jac_var_sigma(psf.pack_var_sigma(0.0, P[:, 0], P[:, 1], P[:, 2], P[:, 3]), yy, xx)
        J = J.reshape(-1, 9)[:, 1:] / np.sqrt(model[y0:y1, x0:x1].reshape(-1, 1))
        F = J.T @ J
        try:
            M = np.linalg.solve(F[:4, :4], F[:4, 4:]) @ np.linalg.solve(F[4:, 4:], F[4:, :4])
            out[n] = float(np.max(np.abs(np.linalg.eigvals(M))))
        except np.linalg.LinAlgError:
            out[n] = 1.0
    return out


LAST_CUT = []


def groups(E, sigma, model=None):
    if KG is None or model is None:
        return groups_distance(E, sigma)
    n = len(E)
    if n == 0:
        return []
    pairs = cKDTree(E[:, 1:3]).query_pairs(6.0 * E[:, 3].max(), output_type="ndarray")
    parent = np.arange(n); size = np.ones(n, int)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    cut = []
    if len(pairs):
        rho = pair_rho2(E, pairs, model)
        for k in np.argsort(-rho):
            if rho[k] < RHO_MIN:
                break
            a, b = find(pairs[k, 0]), find(pairs[k, 1])
            if a == b:
                continue
            if size[a] + size[b] > KG:
                cut.append(rho[k]); continue
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a; size[a] += size[b]
    LAST_CUT.append(max(cut) if cut else 0.0)
    lab = np.array([find(a) for a in range(n)])
    return [np.nonzero(lab == g)[0] for g in np.unique(lab)]


def groups_distance(E, sigma):
    if len(E) == 0:
        return []
    tree = cKDTree(E[:, 1:3])
    wmax = E[:, 3].max()
    pairs = tree.query_pairs(LINK * wmax, output_type="ndarray")
    if len(pairs):
        dist = np.hypot(*(E[pairs[:, 0], 1:3] - E[pairs[:, 1], 1:3]).T)
        keep = dist < LINK * 0.5 * (E[pairs[:, 0], 3] + E[pairs[:, 1], 3])
        pairs = pairs[keep]
    n = len(E)
    G = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else sparse.coo_matrix((n, n))
    _, lab = connected_components(G, directed=False)
    return [np.nonzero(lab == g)[0] for g in range(lab.max() + 1)]


def bounds(k, smax, sigma, h, w, slack=sg.SLACK):
    a_max = 8.0 * smax / float(psf.peak_factor(sigma)) * slack[1] ** 2
    lo = np.r_[-1e-6, np.tile([max(1e-4, 1e-6 * a_max), -0.5, -0.5, slack[0] * sigma], k)]
    hi = np.r_[1e-6, np.tile([a_max, h - 0.5, w - 0.5, slack[1] * sigma], k)]
    return lo, hi


class Stats:
    def __init__(self):
        self.fits = self.adds = self.lr_fail = self.outer = 0
        self.max_group = 0
        self.capped = 0
        self.kappa = 1.0
        self.alpha = 0.0
        self.removed = 0


def localize(frame, sigma, *, offset=0.0, fp_per_mpx=sg.FP_PER_MPX, u=None, max_outer=30, tol=1e-3,
             adds=True, verbose=False, bg_fixed=None, add_after_converged=False, max_iter=100, E0=None):
    u = sg.threshold(sigma, fp_per_mpx) if u is None else float(u)
    d = np.ascontiguousarray(frame, float) - offset
    H, W = d.shape
    st = Stats()
    bg0 = sg.median_background(d)
    phi = sg.estimate_dispersion(d)
    z0 = sg.detection_map(d - bg0, phi * np.maximum(bg0, 1e-3), sigma)
    seeds, _ = sg.find_seeds(z0, sigma, u)
    M = bilinear(H, W)
    beta = sparse.linalg.lsqr(M, bg0.ravel())[0]
    bgs = (M @ beta).reshape(H, W)
    if bg_fixed is not None:
        bgs = bg0.copy() if isinstance(bg_fixed, str) else np.full((H, W), float(bg_fixed))
    g0 = sg.psf_kernel(sigma)
    # initial amplitudes: matched-filter flux estimate at the seed
    num = sg.ndi.correlate(d - bg0, g0, mode="reflect") / (g0 * g0).sum()
    E = E0.copy() if E0 is not None else np.array([[max(num[int(y), int(x)], 1.0), y, x, sigma] for y, x in seeds]).reshape(-1, 4)
    smax = max(float(d.max()), 1.0)
    gain = 0.5 * u * u * phi
    light = render_all(E, H, W)
    if PRE_BG and bg_fixed is None:
        beta, bgs = bg_step(M, beta, d, light, 1.0)
    prev = np.inf
    phase_add = False
    wingvar = None
    for it in range(max_outer):
        st.outer += 1
        n_add = 0
        tot_rem = 0
        ue = u
        if EMP_NULL and (not add_after_converged or phase_add):
            m_ = np.maximum(bgs + light, 1e-3)
            zr = sg.detection_map(d - m_, phi * m_, sigma)
            far = np.ones((H, W), bool)
            if len(E):
                dist, _ = cKDTree(E[:, 1:3]).query(np.c_[np.mgrid[:H, :W][0].ravel(), np.mgrid[:H, :W][1].ravel()])
                far = dist.reshape(H, W) > 3 * sigma
            zf = zr[4:-4, 4:-4][far[4:-4, 4:-4]]
            kappa = 1.4826 * np.median(np.abs(zf - np.median(zf))) if zf.size > 100 else 1.0
            st.kappa = max(1.0, kappa)
            ue = u * st.kappa
            if PSF_ERR and len(E):
                wingvar = wing_variance(d, m_, E, phi, sigma)
                st.alpha = float(wingvar.max()) if wingvar is not None else 0.0
        gain = 0.5 * ue * ue * phi
        for gidx in groups(E, sigma, np.maximum(bgs + light, 1e-3)):
            G = E[gidx]
            st.max_group = max(st.max_group, len(gidx))
            pad = int(np.ceil(max(PAD * G[:, 3].max(), (OWN + sg.SUPPORT) * sigma if adds else 0)))
            y0, x0 = max(int(G[:, 1].min()) - pad, 0), max(int(G[:, 2].min()) - pad, 0)
            y1, x1 = min(int(G[:, 1].max()) + pad + 2, H), min(int(G[:, 2].max()) + pad + 2, W)
            h, w = y1 - y0, x1 - x0
            own_light = render_box(G, y0, x0, h, w)
            halo = np.ascontiguousarray(bgs[y0:y1, x0:x1] + light[y0:y1, x0:x1] - own_light)
            sub = np.ascontiguousarray(d[y0:y1, x0:x1])
            loc = G.copy(); loc[:, 1] -= y0; loc[:, 2] -= x0

            def fit(L, free=False, lv=0.0):
                lo, hi = bounds(len(L), smax, sigma, h, w)
                if free:
                    lo[0], hi[0] = -0.9 * float(halo.min()), float(sub.max()) + 10.0
                th = np.clip(np.r_[lv, L.ravel()], lo + 1e-9, hi - 1e-9)
                th, idiv, _, nit, conv, _ = _rs.lmcl_fit_var_sigma(th, h, w, sub, halo, lo, hi, max_iter, tol_obj=1e-6)
                st.fits += 1; st.capped += nit >= max_iter
                th = np.asarray(th)
                return (th[1:].reshape(-1, 4), float(idiv), th[0]) if free else (th[1:].reshape(-1, 4), float(idiv))

            if GROUP_FREE_LEVEL:
                loc, idiv, _ = fit(loc, True)
            else:
                loc, idiv = fit(loc)
            n_rem = 0
            if adds and (not add_after_converged or phase_add):
                g2 = g0 * g0 / (g0 * g0).sum()
                mbox = np.maximum(bgs[y0:y1, x0:x1] + light[y0:y1, x0:x1], 1e-3)
                def phi_eff(excl=None):
                    if wingvar is None:
                        return np.full((h, w), phi)
                    vb = wingvar[y0:y1, x0:x1]
                    if excl is not None:
                        vb = np.maximum(vb - excl, 0.0)
                    return sg.ndi.correlate(phi + vb / mbox, g2, mode="nearest")
                def own_wing(k):
                    if wingvar is None:
                        return None
                    return wing_render(np.c_[loc[k:k+1, 0], loc[k:k+1, 1] + y0, loc[k:k+1, 2] + x0, loc[k:k+1, 3]], PROFILE, sigma, H, W)[y0:y1, x0:x1]
                ids = list(range(len(loc)))
                while REMOVE and len(loc):
                    best = None
                    for k in range(len(loc)):
                        L2, i2 = fit(np.delete(loc, k, 0))
                        pk = phi_eff(own_wing(k))[int(np.clip(round(loc[k, 1]), 0, h - 1)), int(np.clip(round(loc[k, 2]), 0, w - 1))]
                        ur = ue if REMOVE_U is None else REMOVE_U
                        marg = (i2 - idiv) - 0.5 * ur * ur * pk
                        if best is None or marg < best[0]:
                            best = (marg, k, L2, i2)
                    if best[0] >= 0:
                        break
                    k = best[1]; loc, idiv = best[2], best[3]; ids.pop(k)
                    st.removed += 1; n_rem += 1
                yy, xx = np.mgrid[:h, :w]
                if len(loc) == 0:
                    near = np.zeros((h, w), bool)
                else:
                    near = np.min(np.hypot(yy[..., None] - loc[:, 1], xx[..., None] - loc[:, 2]), -1) <= OWN * sigma
                pe = phi_eff()
                lv = 0.0
                if ADD_FREE_LEVEL:
                    _, idiv, lv = fit(loc, True)
                while near.any() and len(loc) < len(gidx) + sg.K_MAX:
                    win = sg.Window(y0, x0, sub, None, 0.0, near, phi, halo)
                    state = sg.Fit(idiv, np.r_[lv, loc.ravel()])
                    z, a = sg.efficient_score(win, state, g0, sg.Stats())
                    z = np.where(near, z * np.sqrt(phi / pe), -np.inf)
                    p = np.unravel_index(np.argmax(z), z.shape)
                    if not z[p] > ue:
                        break
                    cand = np.r_[loc, [[a[p], p[0], p[1], sigma]]]
                    if ADD_FREE_LEVEL:
                        trial, ti, tlv = fit(cand, True, lv)
                    else:
                        trial, ti = fit(cand)
                    if not idiv - ti > 0.5 * ue * ue * pe[p]:
                        st.lr_fail += 1
                        break
                    st.adds += 1; n_add += 1
                    loc, idiv = trial, ti
                    ids.append(-1)
                    if ADD_FREE_LEVEL:
                        lv = tlv
            if adds and ADD_FREE_LEVEL and (not add_after_converged or phase_add):
                loc = fit(loc, GROUP_FREE_LEVEL)[0]   # back to the node background
            new = loc.copy(); new[:, 1] += y0; new[:, 2] += x0
            light[y0:y1, x0:x1] += render_box(new, y0, x0, h, w) - own_light
            # extra light outside the box is ignored in `light` (support < pad)
            if adds and (not add_after_converged or phase_add):
                for i, gi in enumerate(gidx):
                    E[gi] = new[ids.index(i)] if i in ids else [0.0, E[gi, 1], E[gi, 2], E[gi, 3]]
                E = np.r_[E, new[[j for j, q in enumerate(ids) if q == -1]]]
            else:
                E[gidx] = new[:len(gidx)]
                E = np.r_[E, new[len(gidx):]]
            tot_rem += n_rem
        E = E[E[:, 0] > 0]
        light = render_all(E, H, W)
        if bg_fixed is None:
            beta, bgs = bg_step(M, beta, d, light, OMEGA)
        m = np.maximum(bgs + light, 1e-3)
        obj = float(np.sum(m - d + np.where(d > 0, d * np.log(np.maximum(d, 1e-12) / m), 0.0))) / phi
        if verbose:
            print(f"  outer {it}: K {len(E)} adds {n_add} obj {obj:.3f} d {prev - obj:.3g}")
        conv = prev - obj < tol * max(len(E), 1)
        if verbose:
            print(f"     capped fits so far {st.capped}")
        if add_after_converged:
            if phase_add:
                phase_add = False
                if n_add == 0 and tot_rem == 0:
                    break
            elif conv:
                phase_add = True
        elif n_add == 0 and conv:
            break
        prev = obj
    return E, bgs, phi, st, u
