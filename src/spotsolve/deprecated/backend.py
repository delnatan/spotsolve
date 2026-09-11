"""Fit-level dispatch for the Python reference, `box.localize_boxes`.

The detector itself (`spotsolve.localize`) runs whole frames in Rust and does
not come through here. The reference does: its search and polish are Python,
and each bounded window fit goes to one of two interchangeable fitters,

    fit_var_sigma(theta0, h, w, d, halo, lower, upper, max_iter,
                  tol_obj=...)                              -> FitResult

"py" is `lmga.fit`, the Python optimizer every fixture is generated from;
"rs" is `lmcl.rs`, the fitter the native detector uses.

    from spotsolve import backend
    be = backend.get("rs")          # or "py"

Positions are `(N, 2)` float64 `(y, x)`; amplitudes are total flux in
photoelectrons. Everything is in photoelectrons -- see `calibrate.py`.
"""

import numpy as np

__all__ = ["get", "available", "PythonBackend", "RustBackend"]


def _f1(a):
    return np.ascontiguousarray(np.asarray(a, dtype=float).ravel())


def _img(a, name):
    a = np.ascontiguousarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"`{name}` must be 2-D, got shape {a.shape}")
    return a


class PythonBackend:
    """The Python reference fitter, `lmga.fit`."""

    name = "py"

    def fit_var_sigma(self, theta0, h, w, d, halo, lower, upper, max_iter,
                      *, tol_obj=1e-8):
        from . import lmga
        yy, xx = np.mgrid[0:h, 0:w]
        return lmga.fit(theta0, yy.astype(float), xx.astype(float), 1.0, d,
                        lower, upper, halo=halo, max_iter=max_iter,
                        tol_obj=tol_obj, free_sigma="per_emitter")


class RustBackend:
    """`spotsolve_rs`, the maturin-built extension in `rust/`."""

    name = "rs"

    def __init__(self):
        import spotsolve_rs
        self._rs = spotsolve_rs

    def fit_var_sigma(self, theta0, h, w, d, halo, lower, upper, max_iter,
                      *, tol_obj=1e-8):
        from .structs import FitResult
        halo = np.broadcast_to(np.asarray(halo, dtype=float), (h, w))
        theta, i_div, fisher, n_iter, converged, stalled = \
            self._rs.lmcl_fit_var_sigma(
                _f1(theta0), int(h), int(w), _img(d, "d"), _img(halo, "halo"),
                _f1(lower), _f1(upper), int(max_iter), tol_obj=float(tol_obj))
        return FitResult(theta=theta, I=i_div, F=fisher, n_iter=n_iter,
                         converged=converged, stalled=stalled)


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
