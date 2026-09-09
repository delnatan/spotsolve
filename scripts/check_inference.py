#!/usr/bin/env python3
"""Check the retained Rust-port fixture and time one local profiling step.

The two-pass calculation is an independent validation/timing reference only;
it is not an alternative algorithm exposed by spotsolve.inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from spotsolve.inference import FocusedModel, PixelPSFBank, CountFit, prepare_rust, fit_component, FitOptions
from spotsolve.inference.reference import position_uncertainty, fit_component as reference_fit_component, FitOptions as ReferenceOptions
from spotsolve.inference.reference.profiled import profile_at, solve_poisson_linear
from spotsolve.inference._numerics import _poisson_objective


def load_contract(path):
    with np.load(path) as arrays:
        values = dict(arrays)
    bank = PixelPSFBank(values["depth_um"], values["offsets_px"], values["responses"])
    model = FocusedModel(tuple(values["shape"]), tuple(values["focus_bounds"]), bank,
                         tuple(values["defocus_bounds_um"]), float(values["seed_sigma"]))
    return model, values


def two_pass_profile(data, model, count, theta):
    ids = model.affine_indices(count)
    reference = theta.copy(); reference[ids] = 0
    offset, jac = model.evaluate(count, reference)
    lo, hi = model.parameter_bounds(data, count)
    linear = solve_poisson_linear(data, jac.reshape(-1, len(theta))[:, ids], offset,
        theta[ids], lo[ids], hi[ids], constraints=model.rate_constraints(count)[:, ids])
    fitted = theta.copy(); fitted[ids] = linear.coefficients
    mean, jac = model.evaluate(count, fitted)
    gradient = jac.reshape(-1, len(theta)).T @ (1-data.ravel()/mean.ravel())
    return fitted, linear, gradient


def check_contract(model, values):
    count = int(values["count"])
    theta, data = values["theta"], values["data"]
    mean, jac = model.evaluate(count, theta)
    np.testing.assert_allclose(mean, values["expected_mean"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(jac, values["expected_jacobian"], rtol=1e-8, atol=1e-8)
    fitted, result, gradient = profile_at(data, model, count, values["trial"])
    assert result.converged
    np.testing.assert_allclose(fitted, values["expected_profile_theta"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.objective, values["expected_profile_objective"], atol=1e-8)
    np.testing.assert_allclose(gradient, values["expected_profile_gradient"], rtol=1e-5, atol=1e-6)
    fit = CountFit(count, theta, 0., True, 0., (), 0, 1, model.nuisance_size)
    uncertainty = position_uncertainty(data, model, fit)
    assert uncertainty.status == "conditional_observed_hessian"
    np.testing.assert_allclose(uncertainty.covariance, values["expected_covariance"], rtol=1e-5, atol=1e-8)
    # Check against the pre-pruning two-evaluation algebra as well as the fixture.
    old_theta, old_result, old_gradient = two_pass_profile(data, model, count, values["trial"])
    np.testing.assert_allclose(fitted, old_theta, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result.objective, old_result.objective, atol=1e-8)
    np.testing.assert_allclose(gradient, old_gradient, rtol=1e-5, atol=1e-6)


def check_geometry(native, model, values, repeats):
    """One local seed, not the count fitter's screened/residual proposal search."""
    from scipy.optimize import minimize
    count = int(values["count"])
    truth = values["theta"]
    data = model.evaluate(count, truth)[0]
    lower, upper = model.parameter_bounds(data, count)
    theta = truth.copy(); theta[17:20] += [.2, -.2, .02]
    for k in range(count): theta[21+3*k:23+3*k] += [.15, -.1]
    ids = np.setdiff1d(np.arange(len(theta)), model.affine_indices(count))
    def python_fit():
        current = theta.copy()
        evaluations = 0
        def objective(x):
            nonlocal current, evaluations
            current[ids] = x
            current, linear, gradient = profile_at(data, model, count, current)
            assert linear.converged
            evaluations += 1
            return linear.objective, gradient[ids]
        fit = minimize(objective, theta[ids], jac=True, method="L-BFGS-B",
            bounds=list(zip(lower[ids], upper[ids])),
            options=dict(maxiter=400, gtol=1e-6, ftol=1e-13, maxls=30))
        value, gradient = objective(fit.x)
        return dict(objective=value, evaluations=evaluations, geometry_kkt=float(np.max(np.abs(gradient))))
    times = {"python": [], "rust": []}
    for _ in range(min(repeats, 3)):
        start = perf_counter(); reference = python_fit(); times["python"].append(perf_counter()-start)
        start = perf_counter(); fitted = native.fit_geometry(data, theta, lower, upper)
        times["rust"].append(perf_counter()-start)
        assert fitted["converged"], (fitted["status"], fitted["geometry_kkt"])
        np.testing.assert_allclose(fitted["objective"], reference["objective"], atol=1e-8)
        np.testing.assert_allclose(fitted["theta"][ids], truth[ids], atol=1e-5)
    medians = {key:float(np.median(v)) for key,v in times.items()}
    return dict(check="passed", scope="One displaced local seed, no proposal search or uncertainty; includes Python boundary copies",
        repeats=len(times["rust"]), median_seconds=medians, speed_ratio=medians["python"]/medians["rust"],
        rust={key:fitted[key] for key in ("objective","geometry_kkt","inner_kkt","iterations","evaluations","inner_failures")},
        python=reference)


def check_uncertainty(native, model, values, repeats):
    count = int(values["count"])
    theta, data = values["theta"], values["data"]
    lower, upper = model.parameter_bounds(data, count)
    fit = CountFit(count, theta, 0., True, 0., (), 0, 1, model.nuisance_size)
    times = {"python": [], "rust": []}
    for _ in range(min(repeats, 3)):
        start = perf_counter(); reference = position_uncertainty(data, model, fit)
        times["python"].append(perf_counter()-start)
        start = perf_counter(); result = native.position_uncertainty(data, theta, lower, upper)
        times["rust"].append(perf_counter()-start)
        assert result["status"] == reference.status == "conditional_observed_hessian"
        np.testing.assert_allclose(result["covariance"], values["expected_covariance"], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(result["nuisance_fixed_covariance"], reference.nuisance_fixed_covariance, rtol=1e-5, atol=1e-8)
    medians = {key:float(np.median(v)) for key,v in times.items()}
    return dict(check="passed", scope="Conditional joint position covariance at the same interior fit; includes boundary copies, excludes fitting/calibration",
        repeats=len(times["rust"]), median_seconds=medians, speed_ratio=medians["python"]/medians["rust"],
        model_evaluations={"python_finite_difference":2*len(theta), "rust_analytic":1},
        covariance_max_abs_error=float(np.max(np.abs(result["covariance"]-values["expected_covariance"]))),
        status=result["status"])


def check_counts(model, values):
    """One paired complete K=0/1/2 search, including preparation and proposals."""
    data = values["data"]
    centres = values["theta"][20:].reshape(-1,3)[:,1:]
    options = FitOptions(max_iter=200, screen_iter=24, keep_screened=2)
    results, times = {}, {}
    for backend, fitter, settings in (("python", reference_fit_component, ReferenceOptions(**vars(options))),
                                       ("rust", fit_component, options)):
        start = perf_counter()
        results[backend] = fitter(data, model, candidate_centres=centres,
                                        options=settings)
        times[backend] = perf_counter()-start
    python, rust = results["python"], results["rust"]
    np.testing.assert_allclose([f.objective for f in rust.fits], [f.objective for f in python.fits], atol=2e-5)
    assert min(rust.likelihood_gains) >= -1e-8
    assert [f.starts for f in rust.fits] == [f.starts for f in python.fits]
    return dict(check="passed", scope="One complete local K=0/1/2 search with identical proposal rules and budgets; includes native preparation, excludes uncertainty/full-frame detection",
        repeats=1, seconds=times, speed_ratio=times["python"]/times["rust"],
        options=vars(options), starts=[f.starts for f in rust.fits],
        objectives={key:[float(f.objective) for f in result.fits] for key,result in results.items()},
        native_stationarity=[float(f.projected_gradient_max) for f in rust.fits])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT/"tests/fixtures/08_inference.npz")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--rust", action="store_true", help="require and check the calibrated Rust kernels")
    parser.add_argument("--output", type=Path, default=ROOT/"reference/inference-check/results.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    model, values = load_contract(args.fixture)
    check_contract(model, values)
    native = prepare_rust(model) if args.rust else None
    if native is not None:
        mean, jac = native.evaluate(values["theta"])
        np.testing.assert_allclose(mean, values["expected_mean"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(jac, values["expected_jacobian"], rtol=1e-8, atol=1e-8)
        theta = values["trial"]
        mean, jac = model.evaluate(int(values["count"]), theta)
        basis = jac.reshape(-1, len(theta))[:, model.affine_indices(int(values["count"]))]
        value, gradient, hessian = native.affine_statistics(values["data"], theta)
        np.testing.assert_allclose(value, _poisson_objective(values["data"], mean), atol=1e-8)
        np.testing.assert_allclose(gradient, basis.T @ (1-values["data"].ravel()/mean.ravel()), atol=1e-8)
        np.testing.assert_allclose(hessian,
            basis.T @ ((values["data"].ravel()/mean.ravel()**2)[:, None]*basis), rtol=1e-8, atol=1e-8)
        ids = model.affine_indices(int(values["count"]))
        lo, hi = model.parameter_bounds(values["data"], int(values["count"]))
        lower, upper = lo[ids], hi[ids]
        coefficients = native.prepare_affine(theta)
        solved = native.solve_affine(values["data"], coefficients, lower, upper)
        assert solved["converged"], (solved["status"], solved["kkt"])
        np.testing.assert_allclose(solved["coefficients"], values["expected_profile_theta"][ids], atol=1e-6)
        np.testing.assert_allclose(solved["objective"], values["expected_profile_objective"], atol=1e-8)
        profiled = native.profile_at(values["data"], theta, lo, hi)
        assert profiled["converged"]
        np.testing.assert_allclose(profiled["gradient"], values["expected_profile_gradient"], atol=1e-6)
    times = {"two_pass_reference": [], "retained_one_pass": []}
    count = int(values["count"])
    # Interleave timings after warmup to reduce order/initialization effects.
    for _ in range(args.repeats):
        for label, implementation in (("two_pass_reference", two_pass_profile),
                                      ("retained_one_pass", profile_at)):
            start = perf_counter()
            implementation(values["data"], model, count, values["trial"])
            times[label].append(perf_counter()-start)
    medians = {label: float(np.median(samples)) for label, samples in times.items()}
    report = dict(check="passed", scope="One fixed-geometry profiling step; not detector throughput",
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [Path(__file__).resolve(), *sorted((ROOT/"src/spotsolve/inference").rglob("*.py"))]},
        repeats=args.repeats, median_seconds=medians,
        speed_ratio=medians["two_pass_reference"]/medians["retained_one_pass"],
        psf_evaluations_per_profile={"two_pass_reference": 2*(count+1), "retained_one_pass": count+1})
    if native is not None:
        samples = {"python_model": [], "rust_model": [], "rust_affine_statistics": [], "rust_cached_affine_statistics": [], "python_inner_solve": [], "rust_inner_solve": []}
        coefficients = native.prepare_affine(values["trial"])
        constraints = model.rate_constraints(count)[:, ids]
        offset = np.full(values["data"].size, 1e-4)
        for _ in range(args.repeats):
            for label, operation in (
                ("python_model", lambda: model.evaluate(count, values["trial"])),
                ("rust_model", lambda: native.evaluate(values["trial"])),
                ("rust_affine_statistics", lambda: native.affine_statistics(values["data"], values["trial"])),
                ("rust_cached_affine_statistics", lambda: native.affine_at(values["data"], coefficients)),
                ("python_inner_solve", lambda: solve_poisson_linear(values["data"], basis, offset,
                    coefficients, lower, upper, constraints=constraints)),
                ("rust_inner_solve", lambda: native.solve_affine(values["data"], coefficients, lower, upper))):
                start = perf_counter(); operation()
                samples[label].append(perf_counter()-start)
        report["rust"] = dict(check="passed", scope="Fixed-geometry kernels and complete constrained inner solve, including boundary copies",
            inner_iterations=solved["iterations"], inner_kkt=solved["kkt"],
            median_seconds={key: float(np.median(v)) for key, v in samples.items()})
        report["geometry"] = check_geometry(native, model, values, args.repeats)
        report["uncertainty"] = check_uncertainty(native, model, values, args.repeats)
        report["counts"] = check_counts(model, values)
        for path in (ROOT/"rust/spotsolve-core/src/search.rs", ROOT/"rust/spotsolve-core/src/uncertainty.rs", ROOT/"rust/spotsolve-core/src/geometry.rs", ROOT/"rust/spotsolve-core/src/inference.rs", ROOT/"rust/spotsolve-core/src/affine.rs", ROOT/"rust/spotsolve-py/src/inference.rs"):
            report["sources"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
