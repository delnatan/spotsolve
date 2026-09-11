"""The native group search, across the Python boundary.

What is checked here is the BOUNDARY, not the algorithm -- the algorithm's own
invariants live in `rust/spotsolve-core/tests/layer8_dense_group.rs` and its
scientific behaviour in `scripts/check_group_search.py`. So: that the native
priors really are `prior.py`'s, that an unsupported prior is refused rather
than approximated, that the arrays and settings are validated, that the result
contract is what a caller can adapt, and -- the one that matters most -- that
NO Python numerical or decision callback is reached during a transaction.
"""

import numpy as np
import pytest

from spotsolve import backend, calibrate, prior

rs = pytest.importorskip("spotsolve_rs")

SIGMA = 1.2
SLACK = (0.70, 2.2)
BAND = (0.80, 2.0)


def width_prior(lam=0.02):
    return prior.FocusMixtureWidth(lam, lam * 0.1, SLACK[0] * SIGMA,
                                   BAND[1] * SIGMA, SLACK[1] * SIGMA, SIGMA)


def sim(truth, shape=(30, 30), background=5.0, seed=7):
    pos = np.array([[t[0], t[1]] for t in truth], float).reshape(-1, 2)
    amp = np.array([t[2] for t in truth], float)
    sig = np.array([t[3] for t in truth], float)
    mean = calibrate.render_model(pos, amp, sig, shape, background)
    image = np.random.default_rng(seed).poisson(mean).astype(float)
    return image, np.full(shape, background), pos, amp, sig


def engine(image, bmap, a_s=1400.0, k_max=12, wp=None, next_id=0):
    return backend.get("rs").group_engine(
        image, bmap, SIGMA, SLACK, k_max, wp or width_prior(), a_s,
        next_id=next_id)


def entry(emitters):
    """`(positions, amplitudes, sigmas, ids)` from `(y, x, flux, sigma)` rows."""
    a = np.array(emitters, float).reshape(-1, 4)
    return (np.ascontiguousarray(a[:, :2]), np.ascontiguousarray(a[:, 2]),
            np.ascontiguousarray(a[:, 3]),
            np.arange(len(a), dtype=np.uint32))


# ---------------------------------------------------------------------------
# The priors really are `prior.py`'s
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigmas", [
    [1.2], [1.05, 1.31, 1.44], [1.1, 2.5], [2.45, 2.55], [],
])
def test_native_width_prior_is_the_python_one(sigmas):
    # Including the NORMALIZER, which is deliberately not passed across the
    # boundary: Rust computes its own from (lo, hi, sigma0, scale) so the
    # density the fit is penalized by and the density the score charges cannot
    # drift apart. This is what makes that safe.
    wp = width_prior()
    _, kind, params = backend.native_prior_spec(wp, 1400.0)
    got = rs.group_width_log_config(
        np.ascontiguousarray(np.array(sigmas, float)), kind, params)
    assert got == pytest.approx(wp.log_config(sigmas), rel=0, abs=1e-10)


def test_native_flux_prior_is_the_python_one():
    a_s = 1234.5
    amps = np.array([100.0, 900.0, 2500.0])
    want = float(np.sum(prior.ExponentialFlux(a_s).logpdf(amps)))
    assert rs.group_flux_log_config(amps, a_s) == pytest.approx(want, abs=1e-10)


def test_uniform_width_prior_crosses_too():
    wp = prior.UniformWidth(0.02, SLACK[0] * SIGMA, SLACK[1] * SIGMA)
    _, kind, params = backend.native_prior_spec(wp, 900.0)
    assert kind == "uniform"
    s = np.array([1.0, 1.3, 2.0])
    assert rs.group_width_log_config(s, kind, params) == \
        pytest.approx(wp.log_config(s), abs=1e-10)


def test_an_unsupported_prior_is_refused_rather_than_approximated():
    class MyWidth(prior.WidthPrior):
        def logpdf(self, s): return np.zeros(np.shape(s))
        def curvature(self, s): return np.zeros(np.shape(s))
        def log_config(self, s): return 0.0

    class MyFlux(prior.FluxPrior):
        def logpdf(self, a): return np.zeros(np.shape(a))

    with pytest.raises(TypeError, match="UniformWidth"):
        backend.native_prior_spec(MyWidth(), 900.0)
    with pytest.raises(TypeError, match="ExponentialFlux"):
        backend.native_prior_spec(width_prior(), MyFlux())
    # A SUBCLASS of a supported prior is refused too: it may have overridden
    # the density, and the native side would silently use the base one.
    class Sneaky(prior.FocusMixtureWidth):
        def logpdf(self, s): return np.full(np.shape(s), -1.0)
    with pytest.raises(TypeError):
        backend.native_prior_spec(
            Sneaky(0.02, 0.002, 0.84, 2.4, 2.64, 1.2), 900.0)


def test_the_engine_refuses_a_malformed_prior_at_the_boundary():
    image, bmap, *_ = sim([(15.0, 15.0, 1400.0, SIGMA)])
    with pytest.raises(ValueError, match="lo < mid < hi"):
        rs.DenseGroupEngine(image, bmap, SIGMA, SLACK, 12, 900.0,
                            "focus_mixture", [0.02, 0.002, 2.4, 0.84, 2.64,
                                              1.2, 0.24])
    with pytest.raises(ValueError, match="unsupported width prior"):
        rs.DenseGroupEngine(image, bmap, SIGMA, SLACK, 12, 900.0,
                            "npmle", [1.0])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_arrays_and_settings_are_validated_at_the_boundary():
    image, bmap, *_ = sim([(15.0, 15.0, 1400.0, SIGMA)])
    eng = engine(image, bmap)
    pos, amp, sig, ids = entry([(15.0, 15.0, 1400.0, SIGMA)])

    with pytest.raises(ValueError, match="length N"):
        eng.search_group(pos, amp[:0], sig, ids, (15.0, 15.0))
    with pytest.raises(ValueError, match="unique"):
        eng.search_group(np.vstack([pos, pos]), np.concatenate([amp, amp]),
                         np.concatenate([sig, sig]),
                         np.array([0, 0], np.uint32), (15.0, 15.0))
    with pytest.raises(ValueError, match="positive"):
        eng.search_group(pos, -amp, sig, ids, (15.0, 15.0))
    with pytest.raises(ValueError, match="finite"):
        eng.search_group(pos, amp, sig * np.nan, ids, (15.0, 15.0))
    with pytest.raises(ValueError, match="budget"):
        eng.search_group(pos, amp, sig, ids, (15.0, 15.0), max_moves=0)
    with pytest.raises(ValueError, match="min_gain"):
        eng.search_group(pos, amp, sig, ids, (15.0, 15.0), min_gain=0.0)
    with pytest.raises(ValueError, match="k_max"):
        engine(image, bmap, k_max=999)
    with pytest.raises(ValueError, match="bmap"):
        backend.get("rs").group_engine(image, bmap[:5], SIGMA, SLACK, 12,
                                       width_prior(), 900.0)


def test_an_awkwardly_laid_out_array_is_read_correctly_not_transposed():
    """A Fortran-ordered or strided input must give the SAME answer.

    `PyReadonlyArray2` converts a non-C-contiguous input to C order on
    extraction rather than rejecting it, so the failure mode to guard against
    is not an error that never comes -- it is a silent transposition. An
    asymmetric frame catches that: a transposed 26x34 image would not even have
    the right shape, and a transposed square one would move every source.
    """
    truth = [(12.0, 20.0, 1600.0, SIGMA)]
    image, bmap, tp, *_ = sim(truth, shape=(26, 34), seed=41)
    args = (*entry([(12.0, 20.0, 1600.0, SIGMA)]), (12.0, 20.0))
    want = engine(image, bmap).search_group(*args)
    for awkward in (np.asfortranarray(image),
                    np.repeat(image, 2, axis=1)[:, ::2]):
        got = engine(awkward, bmap).search_group(*args)
        np.testing.assert_allclose(got["positions"], want["positions"],
                                   rtol=0, atol=0)
        np.testing.assert_allclose(got["amplitudes"], want["amplitudes"],
                                   rtol=0, atol=0)


# ---------------------------------------------------------------------------
# The result contract
# ---------------------------------------------------------------------------

def test_the_outcome_is_adaptable_without_reading_diagnostics():
    image, bmap, tp, ta, ts = sim([(15.0, 15.0, 1500.0, SIGMA)], seed=11)
    eng = engine(image, bmap, next_id=1)
    out = eng.search_group(*entry([(15.0, 15.0, 1500.0, SIGMA)]), (15.0, 15.0))

    assert out["positions"].shape == (len(out["amplitudes"]), 2)
    assert out["ids"].dtype == np.uint32
    assert len(set(out["ids"].tolist())) == len(out["ids"])
    assert out["status"] in {"no_improving_proposal", "budget_exhausted",
                             "unresolved_comparison",
                             "context_rebuild_required"}
    assert out["score_status"] in {"supported", "nonstationary",
                                   "singular_curvature",
                                   "ill_conditioned_curvature",
                                   "boundary_mode", "non_finite"}
    # Fit status is SEPARATE from search status.
    assert set(out["fit"]) == {"i_div", "n_iter", "converged", "stalled"}
    # A supported outcome carries a finite score and conditional errors sized
    # to the fitted vector.
    if out["score_status"] == "supported":
        assert np.isfinite(out["score"])
        assert len(out["uncertainty"]) == 4 * len(out["amplitudes"]) + 1
    # Nothing invents an infinite score, ever.
    assert out["score"] is None or np.isfinite(out["score"])
    # The one source is kept, near truth.
    assert len(out["positions"]) == 1
    assert np.linalg.norm(out["positions"][0] - tp[0]) < 0.5


def test_a_committed_move_is_traced_with_its_own_gain():
    # A merged pair: the search has to split it, and the trace has to say so.
    image, bmap, tp, *_ = sim(
        [(14.0, 14.0, 1600.0, SIGMA), (15.5, 14.4, 1500.0, SIGMA)], seed=13)
    eng = engine(image, bmap, a_s=1500.0, next_id=1)
    out = eng.search_group(*entry([(14.75, 14.2, 3100.0, 1.35)]), (14.75, 14.2))
    assert out["trace"], f"nothing accepted: {out['status']}"
    for move in out["trace"]:
        assert move["kind"] in {"birth", "split", "removal"}
        assert move["gain"] > rs.GROUP_SCORE_TOL
        assert np.isfinite(move["score"])
    scores = [m["score"] for m in out["trace"]]
    assert scores == sorted(scores), "committed scores must increase"


def test_the_version_travels_with_the_outcome_and_moves_with_the_prior():
    image, bmap, *_ = sim([(15.0, 15.0, 1400.0, SIGMA)])
    eng = engine(image, bmap)
    args = (*entry([(15.0, 15.0, 1400.0, SIGMA)]), (15.0, 15.0))
    assert eng.search_group(*args)["version"] == eng.version
    v0 = eng.version
    _, kind, params = backend.native_prior_spec(width_prior(lam=0.05), 900.0)
    eng.set_prior(900.0, kind, params)
    assert eng.version > v0
    # A score cached from `v0` is not comparable with one from here; the
    # version is how a caller finds that out rather than by noticing later.
    assert eng.search_group(*args)["version"] == eng.version
    eng.set_background(np.full(image.shape, 6.0))
    assert eng.search_group(*args)["version"] > v0 + 1


def test_ids_are_minted_above_what_is_in_use_and_never_reused():
    image, bmap, *_ = sim(
        [(14.0, 14.0, 1600.0, SIGMA), (15.5, 14.4, 1500.0, SIGMA)], seed=17)
    eng = engine(image, bmap, a_s=1500.0, next_id=100)
    pos, amp, sig, _ = entry([(14.75, 14.2, 3100.0, 1.35)])
    ids = np.array([42], np.uint32)
    out = eng.search_group(pos, amp, sig, ids, (14.75, 14.2))
    minted = [i for i in out["ids"].tolist() if i != 42]
    assert all(i >= 100 for i in minted), out["ids"]
    assert eng.next_id > max(out["ids"].tolist(), default=0)
    # Every entering id either survives or is reported removed.
    assert 42 in set(out["ids"].tolist()) | set(out["removed"].tolist())


# ---------------------------------------------------------------------------
# No Python callback runs inside a transaction
# ---------------------------------------------------------------------------

def test_the_native_path_needs_no_python_numerics_or_decisions(monkeypatch):
    """The stage's definition of done, made executable.

    Every Python routine the old variable-width path called per fit or per move
    is replaced with something that raises. If the transaction still completes,
    nothing inside it went back through Python.
    """
    from spotsolve import core, evidence, lmga, moves, psf

    def forbidden(name):
        def f(*a, **k):
            raise AssertionError(
                f"the native group search called Python's {name}")
        return f

    for module, names in [
        (lmga, ["fit", "i_divergence"]),
        (evidence, ["log_bf_add", "log_bf_remove", "logdet", "logdet_cond"]),
        (moves, ["residual_axis", "residual_axis_var", "split", "split_var"]),
        (core, ["_try_add", "_try_split", "_prune", "_fit_any", "_window",
                "_fit_window", "_add_pass", "_split_pass", "refine"]),
        (psf, ["model_var_sigma", "jac_var_sigma"]),
    ]:
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden(f"{module.__name__}.{name}"))

    image, bmap, tp, *_ = sim(
        [(14.0, 14.0, 1600.0, SIGMA), (15.6, 14.4, 1500.0, SIGMA)], seed=19)
    eng = engine(image, bmap, a_s=1500.0, next_id=1)
    out = eng.search_group(*entry([(14.8, 14.2, 3100.0, 1.35)]), (14.8, 14.2))
    assert out["status"] in {"no_improving_proposal", "budget_exhausted",
                             "unresolved_comparison",
                             "context_rebuild_required"}
    assert len(out["positions"]) >= 1


def test_detect_with_group_search_needs_no_python_numerics_or_decisions(
        monkeypatch):
    """The same guard, one level up: a whole `detect(search="groups")`.

    The epoch loop may run FIND and the background surface in Python; it may
    not fit, score, construct a move, refine or prune.
    """
    from spotsolve import core, detect, evidence, lmga, moves, psf

    image, bmap, *_ = sim(
        [(10.0, 10.0, 1500.0, SIGMA), (11.5, 10.6, 1300.0, SIGMA),
         (20.0, 19.0, 1700.0, SIGMA)], shape=(30, 30), seed=41)

    def forbidden(name):
        def f(*a, **k):
            raise AssertionError(f"search='groups' called Python's {name}")
        return f

    for module, names in [
        (lmga, ["fit", "i_divergence"]),
        (evidence, ["log_bf_add", "log_bf_remove", "logdet", "logdet_cond"]),
        (moves, ["residual_axis", "residual_axis_var", "split", "split_var"]),
        (core, ["_try_add", "_try_split", "_prune", "_fit_any", "_window",
                "_fit_window", "_add_pass", "_split_pass", "refine"]),
        (psf, ["model_var_sigma", "jac_var_sigma"]),
    ]:
        for name in names:
            if hasattr(module, name):
                monkeypatch.setattr(module, name,
                                    forbidden(f"{module.__name__}.{name}"))

    r = detect(image, sigma=SIGMA, gain=1.0, impl="rs", search="groups",
               verbose=0)
    assert len(r.positions) >= 2
    assert r.se.shape == (len(r.positions), 3)
    assert all("transactions" in h for h in r.history)


def test_group_search_refuses_what_it_does_not_implement():
    from spotsolve import detect
    image = np.full((12, 12), 5.0)
    with pytest.raises(ValueError, match="impl='rs'"):
        detect(image, gain=1.0, impl="py", search="groups", verbose=0)
    with pytest.raises(ValueError, match="variable width"):
        detect(image, gain=1.0, impl="rs", search="groups", slack=None,
               verbose=0)
    with pytest.raises(ValueError, match="search must be"):
        detect(image, gain=1.0, impl="rs", search="both", verbose=0)


# ---------------------------------------------------------------------------
# Diagnostics, and the constants the comparison rests on
# ---------------------------------------------------------------------------

def test_the_score_components_add_up_to_the_score():
    image, bmap, *_ = sim([(15.0, 15.0, 1500.0, SIGMA)], seed=23)
    eng = engine(image, bmap)
    r = eng.score_state(*entry([(15.0, 15.0, 1500.0, SIGMA)]), (15.0, 15.0))
    assert r["status"] == "supported"
    want = (-r["i_div"] + r["log_prior"]
            + 0.5 * r["p"] * np.log(2 * np.pi) - 0.5 * r["logdet"]
            + r["log_box"])
    assert r["score"] == pytest.approx(want, rel=0, abs=1e-9)
    # The box term is a probability of the Laplace Gaussian's mass inside the
    # prior's support: never positive, and nothing for a source well inside.
    assert r["log_box"] <= 1e-12
    assert r["log_box"] > -1e-3
    assert r["log_volume"] == pytest.approx(
        0.5 * r["p"] * np.log(2 * np.pi) - 0.5 * r["logdet"], abs=1e-12)
    assert r["p"] == 4 * r["k"] + 1
    # `log_prior` is the configuration prior from prior.py, exactly.
    wp = width_prior()
    _, kind, params = backend.native_prior_spec(wp, 1400.0)
    theta = np.asarray(r["theta"])
    assert r["log_prior"] == pytest.approx(
        rs.group_flux_log_config(np.ascontiguousarray(theta[1::4]), 1400.0)
        + rs.group_width_log_config(np.ascontiguousarray(theta[4::4]),
                                    kind, params), abs=1e-9)


def test_an_exhausted_budget_reports_itself_instead_of_being_scored():
    image, bmap, *_ = sim([(15.0, 15.0, 1500.0, SIGMA)], seed=29)
    eng = engine(image, bmap)
    r = eng.score_state(*entry([(12.0, 12.0, 400.0, SIGMA)]), (12.0, 12.0),
                        max_iter=1)
    assert r["status"] == "nonstationary"
    assert r["score"] is None, "an unsupported hypothesis has no score"


def test_the_capacity_audit_covers_the_variable_width_layout():
    # `linalg::P_MAX` is 3*K_MAX+1 and describes the fixed-width passes. A
    # group transaction is 4*K+1 with room for the K+1 alternative, which does
    # not fit there -- the reason `P_MAX_VAR` exists.
    assert rs.GROUP_P_MAX_VAR == 4 * (12 + 1) + 1
    assert rs.GROUP_P_MAX_VAR > 3 * 12 + 1


def test_the_documented_constants_are_the_ones_that_ship():
    assert rs.GROUP_SCORE_TOL > 0
    assert rs.GROUP_COND_LIMIT >= 1e3
    assert 0 < rs.GROUP_OUTSIDE_PEAK_ALPHA < 1
    assert rs.GROUP_BOUND_TOL > 0
    assert rs.GROUP_DRIFT_FACTOR > 0
    assert rs.GROUP_STARTUP_SE_FRAC > 0


def test_diagnostics_report_the_work_and_the_geometry():
    image, bmap, *_ = sim(
        [(14.0, 14.0, 1600.0, SIGMA), (15.6, 14.4, 1500.0, SIGMA)], seed=31)
    eng = engine(image, bmap, a_s=1500.0, next_id=1)
    out = eng.search_group(*entry([(14.8, 14.2, 3100.0, 1.35)]), (14.8, 14.2))
    d = out["diagnostics"]
    assert d["n_fits"] >= 1
    # Every fit is charged to exactly one move type. The incumbent's own
    # refits and restarts are charged to `removal`, which is the move whose
    # hypothesis it is: `K` against `K-1`.
    assert d["n_fits"] == d["fits_birth"] + d["fits_split"] + d["fits_removal"]
    assert d["n_restarts"] <= d["n_fits"]
    assert d["proposals_generated"] >= 1
    assert d["n_free"] >= 1 and d["n_frozen"] >= 0
    for flag in ("capacity_limited", "edge_clipped", "position_bound_active",
                 "incumbent_unsupported", "residual_peak_outside_box"):
        assert isinstance(d[flag], bool)
    assert isinstance(d["unsupported"], dict)


def test_separate_engines_share_no_state():
    image, bmap, *_ = sim([(15.0, 15.0, 1500.0, SIGMA)], seed=37)
    a, b = engine(image, bmap), engine(image, bmap)
    args = (*entry([(15.0, 15.0, 1500.0, SIGMA)]), (15.0, 15.0))
    first = a.search_group(*args)
    _ = b.search_group(*args)
    np.testing.assert_allclose(a.search_group(*args)["positions"],
                               first["positions"], rtol=0, atol=0)
