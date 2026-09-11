"""The package imports, and its public surface is what `__init__` advertises."""

import spotsolve


def test_public_names_exist():
    missing = [n for n in spotsolve.__all__ if not hasattr(spotsolve, n)]
    assert not missing, f"__all__ names not bound: {missing}"


def test_submodules_import():
    for name in ("native", "results", "calibration", "aggregates", "loctable",
                 "audit", "metrics", "simulate", "psf"):
        __import__(f"spotsolve.{name}")
