"""Freeze joint-model prototype stages as the Rust port's parity contract.

Writes tests/fixtures/11_joint.json. Frames come from 10_scoregate.json by
case name, plus one crowded referee cell (test_localize cell 18) embedded
here. Each case starts from the prototype's one-pass emitters `e0` and
records, in order:

- `beta_pre`: background nodes after the pre-fit (least squares to the
  median map, then one IRLS step against `e0`'s light);
- `pairs` `(i, j, rho2)` and `groups` at that state, and `kappa`;
- `fit_round`: emitters and nodes after one Gauss-Seidel round without adds;
- `add_round`: the same for one round with removal and adds;
- `full`: the frozen configuration run to convergence.

Stages up to one round are deterministic and are compared exactly; `full`
iterates a chaotic map on crowded frames and is compared statistically.
Run once from the prototype (scripts/jointfit_prototype.py).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import jointfit_prototype as jf  # noqa: E402
from spotsolve import scoregate as sg  # noqa: E402
from spotsolve.simulate import simulate  # noqa: E402

jf.KG, jf.PRE_BG, jf.OMEGA, jf.EMP_NULL, jf.REMOVE = 12, True, 1.5, True, True
MAX_OUTER, TOL = 80, 1e-2


def frames():
    src = json.load(open(ROOT / "tests" / "fixtures" / "10_scoregate.json"))
    for c in src["cases"]:
        f = np.array(c["frame"], float).reshape(c["shape"])
        yield c["name"], f, c["sigma"], c["offset"], None
    sim = simulate(shape=(64, 64), density=0.034, amplitude_range=(900.0, 1900.0),
                   sigma=1.2, sigma_spread=0.2, seed=18)
    yield "crowded", sim.image.astype(float), 1.2, 0.0, sim.image.astype(int).ravel().tolist()


def kappa(d, m, E, phi, sigma):
    """The prototype's empirical-null scale, copied from `localize`."""
    H, W = d.shape
    zr = sg.detection_map(d - m, phi * m, sigma)
    far = np.ones((H, W), bool)
    if len(E):
        dist, _ = cKDTree(E[:, 1:3]).query(np.c_[np.mgrid[:H, :W][0].ravel(), np.mgrid[:H, :W][1].ravel()])
        far = dist.reshape(H, W) > 3 * sigma
    zf = zr[4:-4, 4:-4][far[4:-4, 4:-4]]
    k = 1.4826 * np.median(np.abs(zf - np.median(zf))) if zf.size > 100 else 1.0
    return max(1.0, float(k))


def nodes_of(M, bgs):
    return np.linalg.lstsq(M.toarray(), bgs.ravel(), rcond=None)[0]


def main():
    out = {"tile": jf.TILE, "kg": jf.KG, "rho_min": jf.RHO_MIN, "omega": jf.OMEGA,
           "max_outer": MAX_OUTER, "tol": TOL, "cases": []}
    for name, f, sigma, offset, embed in frames():
        jf.REMOVE_U = sg.threshold(sigma)
        d = f - offset
        H, W = d.shape
        r = sg.localize(f, sigma, offset=offset)
        E0 = np.c_[r.amplitudes, r.positions, r.fit_sigma].reshape(-1, 4)
        phi = sg.estimate_dispersion(d)
        M = jf.bilinear(H, W)
        beta = sparse.linalg.lsqr(M, sg.median_background(d).ravel())[0]
        light = jf.render_all(E0, H, W)
        beta, bgs = jf.bg_step(M, beta, d, light, 1.0)
        model = np.maximum(bgs + light, 1e-3)
        case = {"name": name, "sigma": sigma, "offset": offset, "shape": [H, W],
                "e0": E0.tolist(), "beta_pre": beta.tolist(), "kappa": kappa(d, model, E0, phi, sigma)}
        if embed is not None:
            case["frame"] = embed
        if len(E0) > 1:
            pairs = cKDTree(E0[:, 1:3]).query_pairs(6.0 * E0[:, 3].max(), output_type="ndarray")
            pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
            rho = jf.pair_rho2(E0, pairs, model) if len(pairs) else np.zeros(0)
            case["pairs"] = np.c_[pairs, rho].tolist()
        else:
            case["pairs"] = []
        case["groups"] = [g.tolist() for g in jf.groups(E0, sigma, model)]
        if name != "gem":
            for key, aac in [("fit_round", True), ("add_round", False)]:
                E, bgs, _, st, _ = jf.localize(f, sigma, offset=offset, E0=E0, add_after_converged=aac,
                                               max_outer=1, tol=TOL)
                case[key] = {"emitters": E.tolist(), "beta": nodes_of(M, bgs).tolist(), "fits": st.fits,
                             "adds": st.adds, "removed": st.removed, "lr_fail": st.lr_fail, "kappa": st.kappa}
        E, bgs, _, st, _ = jf.localize(f, sigma, offset=offset, E0=E0, add_after_converged=True,
                                       max_outer=MAX_OUTER, tol=TOL)
        case["full"] = {"emitters": E.tolist(), "beta": nodes_of(M, bgs).tolist(), "fits": st.fits,
                        "adds": st.adds, "removed": st.removed, "lr_fail": st.lr_fail,
                        "kappa": st.kappa, "outer": st.outer, "max_group": st.max_group}
        print(f"{name}: e0 {len(E0)} -> {len(E)}, outer {st.outer}, adds {st.adds}, removed {st.removed}, "
              f"kappa {st.kappa:.3f}, fits {st.fits}", flush=True)
        out["cases"].append(case)
    path = ROOT / "tests" / "fixtures" / "11_joint.json"
    path.write_text(json.dumps(out))
    print("wrote", path)


if __name__ == "__main__":
    main()
