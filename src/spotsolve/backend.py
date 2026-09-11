"""Pass-level dispatch between the Python reference and the Rust core.

`detect`'s round loop, `find_candidates` and `background_map`'s convolutions
stay in Python for both backends. The fixed-width path dispatches four passes plus two image helpers. The
variable-width path currently dispatches each ML/MAP fit to Rust while keeping
its width-aware passes in Python. New variable-width numerical work belongs
in Rust; Python is an explicit reference, not a parity target.

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

One entry point is Rust-only, because it is a Rust-only algorithm:

    group_engine(d_e, bmap, sigma, slack, k_max, wprior, flux_prior)
                                            -> DenseGroupEngine

`DenseGroupEngine.search_group` owns local neighbourhood construction,
proposals, fits, scoring and acceptance for one group. See
`docs/RUST_GROUP_SEARCH_PLAN.md`.

Positions are `(N, 2)` float64 `(y, x)`; amplitudes are total flux in
photoelectrons. Everything is in photoelectrons -- see `calibrate.py`.
"""

import numpy as np

__all__ = ["get", "available", "PythonBackend", "RustBackend",
           "native_prior_spec"]


def native_prior_spec(wprior, flux_prior):
    """`(a_s, width_kind, width_params)` for the native group search.

    Translates the two prior classes the Rust core implements and REFUSES
    everything else, rather than approximating it by one of them. That is not
    defensiveness: the prior is what prices one more emitter, so a detector
    that quietly substituted a different one would report different objects and
    say nothing about it.

    A curved flux prior is refused for a second reason on top of that -- it
    contributes a `Lambda` block on the amplitudes that the Laplace volume does
    not carry. `prior.py`'s header records that this is exactly why the NPMLE
    estimator was built, measured and removed rather than wired in.

    The width prior's normalizer is deliberately NOT passed across: Rust
    computes its own from `(lo, hi, sigma0, scale)`, so the density the native
    fit is penalized by and the density the native score charges are one
    object. `tests/test_dense_group.py` asserts the two agree.
    """
    from . import prior as prior_mod

    if isinstance(flux_prior, (int, float, np.floating, np.integer)):
        a_s = float(flux_prior)
    elif type(flux_prior) is prior_mod.ExponentialFlux:
        a_s = float(flux_prior.A_s)
    else:
        raise TypeError(
            f"the native group search implements prior.ExponentialFlux only, "
            f"got {type(flux_prior).__name__}. A curved flux prior also needs "
            f"the amplitude curvature term the Laplace volume omits; see "
            f"prior.py.")

    if type(wprior) is prior_mod.UniformWidth:
        return a_s, "uniform", [wprior.lam, wprior.lo, wprior.hi]
    if type(wprior) is prior_mod.FocusMixtureWidth:
        return a_s, "focus_mixture", [
            wprior.lam_focus, wprior.lam_wide, wprior.lo, wprior.mid,
            wprior.hi, wprior.sigma0, wprior.scale]
    raise TypeError(
        f"the native group search implements prior.UniformWidth and "
        f"prior.FocusMixtureWidth, got {type(wprior).__name__}")


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
    """The explicit Python reference for numerical and scientific controls."""

    name = "py"

    # The four passes carry per-emitter widths internally; this contract does
    # not, because native pass-level dispatch is fixed-width. Every emitter is
    # handed in at the PSF width and the widths that come back are discarded,
    # which is exactly the fixed-width pipeline. `core.detect(slack=...)`
    # bypasses this backend and calls the passes directly -- see its `impl`
    # note. The variable-width path dispatches each numerical fit through
    # `fit_var_sigma` while keeping the width-aware passes in Python.

    def add_pass(self, d_e, bmap, pos, amp, cand, camp, sigma, lam, A_s, k_max):
        from . import core
        sig = np.full(len(np.asarray(amp).ravel()), float(sigma))
        pos, amp, _, n = core._add_pass(d_e, bmap, pos, amp, sig, cand, camp,
                                        sigma, lam, A_s, k_max)
        return pos, amp, n

    def split_pass(self, d_e, bmap, pos, amp, model, sigma, lam, A_s, k_max):
        from . import core
        sig = np.full(len(np.asarray(amp).ravel()), float(sigma))
        pos, amp, _, n = core._split_pass(d_e, pos, amp, sig, bmap, sigma,
                                          lam, A_s, k_max, model)
        return pos, amp, n

    def refine(self, d_e, pos, amp, sigma, bmap, k_max, max_sweeps):
        from . import core
        return core.refine(d_e, pos, amp, sigma, bmap, k_max=k_max,
                             max_sweeps=max_sweeps)[:3]

    def prune(self, d_e, bmap, pos, amp, sigma, lam, A_s, k_max):
        from . import core
        n0 = len(pos)
        sig = np.full(len(np.asarray(amp).ravel()), float(sigma))
        pos, amp, _ = core._prune(d_e, pos, amp, sig, bmap, sigma, lam, A_s,
                                  k_max)
        return pos, amp, n0 - len(pos)

    def fit_var_sigma(self, theta0, h, w, d, halo, lower, upper, max_iter,
                      *, tol_obj=1e-8, wprior=None):
        from . import core, lmga
        yy, xx = np.mgrid[0:h, 0:w]
        return lmga.fit(theta0, yy.astype(float), xx.astype(float), 1.0, d,
                        lower, upper, halo=halo, max_iter=max_iter,
                        tol_obj=tol_obj, free_sigma="per_emitter",
                        penalty=None if (wprior is None or wprior.is_flat
                                         or len(theta0) == 1)
                        else core._WidthPenalty(wprior, (len(theta0) - 1) // 4))

    def render_model(self, pos, amp, sigma, shape, background):
        from . import calibrate
        return calibrate.render_model(pos, amp, sigma, shape, background)

    def emitter_free_mask(self, pos, sigma, shape, radius_factor):
        from . import calibrate
        return calibrate.emitter_free_mask(shape, _f2(pos), sigma,
                                           radius_factor)

    def group_engine(self, *args, **kwargs):
        """Not available: the group search has no Python implementation.

        Deliberately. A second implementation of a NEW algorithm is not a
        reference -- there is nothing yet for it to be a reference to -- and
        two implementations of one search would have to be kept in step by hand
        while the search is still being designed. `impl="py"` remains the
        reference for the numerical fitter and for the pass-based passes, which
        is where a reference earns its keep.
        """
        raise NotImplementedError(
            "the native group search has no Python reference implementation; "
            "use backend.get('rs').group_engine(...)")


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

    def fit_var_sigma(self, theta0, h, w, d, halo, lower, upper, max_iter,
                      *, tol_obj=1e-8, wprior=None):
        from .structs import FitResult
        from .prior import FocusMixtureWidth
        width_prior = None
        if wprior is not None and not wprior.is_flat and len(theta0) > 1:
            if type(wprior) is not FocusMixtureWidth:
                raise TypeError("Rust fitting supports FocusMixtureWidth or flat width priors")
            width_prior = (wprior.sigma0, wprior.scale, wprior._logZ)
        halo = np.broadcast_to(np.asarray(halo, dtype=float), (h, w))
        theta, i_div, fisher, n_iter, converged, stalled = \
            self._rs.lmcl_fit_var_sigma(
                _f1(theta0), int(h), int(w), _img(d, "d"), _img(halo, "halo"),
                _f1(lower), _f1(upper), int(max_iter),
                tol_obj=float(tol_obj), width_prior=width_prior)
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

    def group_engine(self, d_e, bmap, sigma, slack, k_max, wprior,
                     flux_prior, next_id=0):
        """A prepared native group engine over one frame.

        The frame is copied once here; each `search_group` call then holds no
        borrow on it and runs with the GIL released. The engine owns its
        workspace and its id allocator, so one engine per frame -- not one per
        group -- is what keeps ids unique and the buffers warm.
        """
        a_s, kind, params = native_prior_spec(wprior, flux_prior)
        return self._rs.DenseGroupEngine(
            _img(d_e, "d_e"), _img(bmap, "bmap"), float(sigma),
            (float(slack[0]), float(slack[1])), int(k_max), a_s, kind,
            [float(v) for v in params], int(next_id))


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
