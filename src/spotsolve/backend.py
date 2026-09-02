"""Pass-level dispatch between the Python reference and the Rust core.

`detect`'s round loop, `find_candidates` and `background_map`'s convolutions
stay in Python for both backends. What differs is exactly four passes plus two
image helpers, which is what makes `bench.py --methods spotsolve,spotsolve-rs` a
controlled comparison: identical candidates, identical background surfaces, and
only the passes swapped.

    from spotsolve import backend
    be = backend.get("rs")          # or "py"

Every backend takes and returns the same things:

    add_pass(d_e, bmap, pos, amp, cand, camp, sigma, lam, A_s, k_max)
                                            -> (pos, amp, n_added)
    split_pass(d_e, bmap, pos, amp, model, sigma, lam, A_s, k_max)
                                            -> (pos, amp, n_split)
    refine(d_e, pos, amp, sigma, bmap, k_max, max_sweeps)  -> (pos, amp, se)
    prune(d_e, bmap, pos, amp, sigma, lam, A_s, k_max)
                                            -> (pos, amp, n_removed)
    fit_var_sigma(theta0, h, w, d, halo, lower, upper, max_iter)
                                            -> FitResult
    render_model(pos, amp, sigma, shape, background)       -> (H, W)
    emitter_free_mask(pos, sigma, shape, radius_factor)    -> (H, W) bool

Positions are `(N, 2)` float64 `(y, x)`; amplitudes are total flux in
photoelectrons. Everything is in photoelectrons -- see `calibrate.py`.
"""

import numpy as np

__all__ = ["get", "available", "PythonBackend", "RustBackend"]


def _f2(a):
    """A C-contiguous float64 (N, 2), including when N is 0."""
    a = np.ascontiguousarray(np.asarray(a, dtype=float))
    return a.reshape(0, 2) if a.size == 0 else np.atleast_2d(a)


def _f1(a):
    return np.ascontiguousarray(np.asarray(a, dtype=float).ravel())


def _img(a, name):
    a = np.ascontiguousarray(np.asarray(a, dtype=float))
    if a.ndim != 2:
        raise ValueError(f"`{name}` must be 2-D, got shape {a.shape}")
    return a


class PythonBackend:
    """The reference implementation. The oracle every fixture is built from."""

    name = "py"

    def add_pass(self, d_e, bmap, pos, amp, cand, camp, sigma, lam, A_s, k_max):
        from . import core
        return core._add_pass(d_e, bmap, pos, amp, cand, camp,
                                sigma, lam, A_s, k_max)

    def split_pass(self, d_e, bmap, pos, amp, model, sigma, lam, A_s, k_max):
        from . import core
        return core._split_pass(d_e, pos, amp, bmap, sigma, lam, A_s,
                                  k_max, model)

    def refine(self, d_e, pos, amp, sigma, bmap, k_max, max_sweeps):
        from . import core
        return core.refine(d_e, pos, amp, sigma, bmap, k_max=k_max,
                             max_sweeps=max_sweeps)

    def prune(self, d_e, bmap, pos, amp, sigma, lam, A_s, k_max):
        from . import core
        n0 = len(pos)
        pos, amp = core._prune(d_e, pos, amp, bmap, sigma, lam, A_s, k_max)
        return pos, amp, n0 - len(pos)

    def fit_var_sigma(self, theta0, h, w, d, halo, lower, upper, max_iter):
        from . import core, lmga
        yy, xx = np.mgrid[0:h, 0:w]
        return lmga.fit(theta0, yy.astype(float), xx.astype(float), 1.0, d,
                        lower, upper, halo=halo, max_iter=max_iter,
                        tol_obj=core.EVIDENCE_TOL_OBJ,
                        free_sigma="per_emitter")

    def render_model(self, pos, amp, sigma, shape, background):
        from . import calibrate
        return calibrate.render_model(pos, amp, sigma, shape, background)

    def emitter_free_mask(self, pos, sigma, shape, radius_factor):
        from . import calibrate
        return calibrate.emitter_free_mask(shape, _f2(pos), sigma,
                                           radius_factor)


class RustBackend:
    """`spotsolve_rs`, the maturin-built extension in `rust/`.

    Constructing this asserts that the constants shared across the boundary
    still agree. They are defined in both languages, and a silent drift in
    `PRUNE_TAU` -- the pipeline's only precision/recall dial -- would look like
    an algorithmic difference rather than a build problem.
    """

    name = "rs"

    def __init__(self):
        from . import core
        import spotsolve_rs
        self._rs = spotsolve_rs
        for const in ("PRUNE_TAU", "REFINE_TOL", "REFINE_TOL_OBJ",
                      "EVIDENCE_TOL_OBJ", "REFINE_MAX_ITER"):
            py, rs = getattr(core, const), getattr(spotsolve_rs, const)
            if py != rs:
                raise RuntimeError(
                    f"{const} differs between core.py ({py}) and spotsolve_rs "
                    f"({rs}); rebuild the extension with `maturin develop`")
        from . import evidence
        if evidence.COND_GUARD != spotsolve_rs.COND_GUARD:
            raise RuntimeError("COND_GUARD differs between Python and Rust")

    def add_pass(self, d_e, bmap, pos, amp, cand, camp, sigma, lam, A_s, k_max):
        return self._rs.add_pass(_img(d_e, "d_e"), _img(bmap, "bmap"),
                                 _f2(pos), _f1(amp), _f2(cand), _f1(camp),
                                 sigma, lam, A_s, k_max)

    def split_pass(self, d_e, bmap, pos, amp, model, sigma, lam, A_s, k_max):
        return self._rs.split_pass(_img(d_e, "d_e"), _img(bmap, "bmap"),
                                   _f2(pos), _f1(amp), _img(model, "model"),
                                   sigma, lam, A_s, k_max)

    def refine(self, d_e, pos, amp, sigma, bmap, k_max, max_sweeps):
        from . import core
        return self._rs.refine(_img(d_e, "d_e"), _f2(pos), _f1(amp), sigma,
                               _img(bmap, "bmap"), k_max, core.REFINE_MAX_ITER,
                               max_sweeps, self._rs.REFINE_TOL)

    def prune(self, d_e, bmap, pos, amp, sigma, lam, A_s, k_max):
        return self._rs.prune(_img(d_e, "d_e"), _img(bmap, "bmap"),
                              _f2(pos), _f1(amp), sigma, lam, A_s, k_max)

    def fit_var_sigma(self, theta0, h, w, d, halo, lower, upper, max_iter):
        from .structs import FitResult
        theta, i_div, fisher, n_iter, converged, stalled = \
            self._rs.lmcl_fit_var_sigma(
                _f1(theta0), int(h), int(w), _img(d, "d"), _img(halo, "halo"),
                _f1(lower), _f1(upper), int(max_iter))
        return FitResult(theta=theta, I=i_div, F=fisher, n_iter=n_iter,
                         converged=converged, stalled=stalled)

    def render_model(self, pos, amp, sigma, shape, background):
        return self._rs.render_model(_f2(pos), _f1(amp), sigma,
                                     (int(shape[0]), int(shape[1])),
                                     float(background))

    def emitter_free_mask(self, pos, sigma, shape, radius_factor):
        return self._rs.emitter_free_mask(_f2(pos), sigma,
                                          (int(shape[0]), int(shape[1])),
                                          radius_factor)


_CACHE = {}


def get(name="py"):
    """Backend by name: "py" (the reference) or "rs" (the Rust extension)."""
    if isinstance(name, (PythonBackend, RustBackend)):
        return name
    if name not in ("py", "rs"):
        raise ValueError(f"unknown backend {name!r}; expected 'py' or 'rs'")
    if name not in _CACHE:
        _CACHE[name] = PythonBackend() if name == "py" else RustBackend()
    return _CACHE[name]


def available():
    """Backend names that can actually be constructed here."""
    out = ["py"]
    try:
        get("rs")
        out.append("rs")
    except Exception:
        pass
    return out
