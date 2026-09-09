#!/usr/bin/env python3
"""Paired native initial-proposal controls; no source-count acceptance rule.

Uses the immutable synthetic calibration, the same observed pixels and search
budgets for both policies. No truth centers are supplied to either search.
"""
import argparse
import hashlib
import json
from itertools import permutations
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from scripts.check_inference import load_contract
from spotsolve.inference import prepare_rust


def controls(model, values):
    base = values['theta'].copy()
    base[:16] = .5
    base[16:20] = [0., 10., 10., .4]
    base[20:26] = [1.5, 10., 9., 1.5, 10., 11.]
    for name, count in [('isolated', 1), ('focused_on_broad', 1),
                        ('equal_pair', 2), ('unequal_pair', 2),
                        ('broad_only', 0), ('noisy_haze', 2)]:
        theta = base[:20 + 3 * count].copy()
        if name in ('focused_on_broad', 'broad_only', 'noisy_haze'):
            theta[16] = 6.
        if name == 'unequal_pair':
            theta[23] = .45
        if name == 'noisy_haze':
            theta[:16] = np.repeat([0., .2, .6, .8], 4)
        positions = theta[20:].reshape(-1, 3)[:, 1:]
        y0, x0, y1, x1 = model.focus_bounds
        assert np.all(positions >= [y0, x0]) and np.all(positions <= [y1, x1])
        mean = model.evaluate(count, theta)[0]
        # Each paired policy sees exactly the same fixed draw.
        data = np.random.default_rng(42).poisson(mean).astype(float) if name == 'noisy_haze' else mean
        yield name, count, theta, np.ascontiguousarray(data)


def position_error(points, truth):
    if not len(truth):
        return None
    return min(float(np.max(np.linalg.norm(points[list(order)] - truth, axis=1)))
               for order in permutations(range(len(points))))


def compare(model, values):
    options = dict(max_iter=200, screen_iter=24, keep_screened=2, gtol=1e-6)
    native = prepare_rust(model)
    rows = []
    for name, count, theta, data in controls(model, values):
        truth = theta[20:].reshape(-1, 3)[:, 1:]
        policies = {}
        for policy in ('moments', 'aguet'):
            started = perf_counter()
            fits = native.fit_component(data, np.empty((0, 2)), seed_sigma=model.seed_sigma,
                                        proposal_method=policy, proposal_alpha=.05, **options)
            elapsed = perf_counter() - started
            objectives = np.array([fit['objective'] for fit in fits])
            assert np.min(-np.diff(objectives)) >= -1e-8
            proposals = np.asarray(fits[0]['proposal_centres']).reshape(-1, 2)
            distances = (np.min(np.linalg.norm(truth[:, None] - proposals[None], axis=2), axis=1).tolist()
                         if len(truth) and len(proposals) else None)
            for k, fit in enumerate(fits):
                lo, hi = native.parameter_bounds(data, k)
                assert np.min(fit['theta'] - lo) >= -1e-8
                assert np.min(hi - fit['theta']) >= -1e-8
                assert np.min(model.rate_constraints(k) @ fit['theta']) >= -1e-8
            lo, hi = native.parameter_bounds(data, count)
            uncertainty = native.position_uncertainty(data, fits[count]['theta'], lo, hi)
            policies[policy] = dict(proposal_centres=proposals.tolist(),
                nearest_proposal_distance_px=distances, objectives=objectives.tolist(),
                likelihood_gains=(-np.diff(objectives)).tolist(),
                starts=[fit['starts'] for fit in fits], statuses=[fit['status'] for fit in fits],
                stationarity=[fit['kkt'] for fit in fits], seconds=elapsed,
                broad_flux_photons=[float(1000 * fit['theta'][16]) for fit in fits],
                truth_count_position_error_px=position_error(fits[count]['theta'][20:].reshape(-1, 3)[:, 1:], truth),
                truth_count_uncertainty_status=uncertainty['status'],
                broad_conditioned_absent=uncertainty['broad_conditioned_absent'])
        rows.append(dict(scene=name, truth_count=count, policies=policies,
                         aguet_minus_moments_objective=(np.array(policies['aguet']['objectives']) -
                                                       policies['moments']['objectives']).tolist()))
        print(name, rows[-1]['aguet_minus_moments_objective'], flush=True)
    return dict(scope='Small paired initial-proposal controls, not count selection or probability calibration',
                options=options, proposal_alpha=.05, noisy_seed=42, scenes=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reference/dense-proposals/results.json')
    args = parser.parse_args()
    fixture = ROOT / 'tests/fixtures/08_inference.npz'
    model, values = load_contract(fixture)
    result = compare(model, values)
    result['fixture_sha256'] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    paths = [Path(__file__).resolve(), ROOT / 'rust/spotsolve-core/src/search.rs',
             ROOT / 'rust/spotsolve-core/src/sparse.rs',
             ROOT / 'rust/spotsolve-core/src/uncertainty.rs', ROOT / 'rust/spotsolve-py/src/inference.rs']
    result['sources'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
