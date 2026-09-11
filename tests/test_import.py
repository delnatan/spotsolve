"""The package imports, and its public surface is what `__init__` advertises."""

import spotsolve


def test_public_names_exist():
    missing = [n for n in spotsolve.__all__ if not hasattr(spotsolve, n)]
    assert not missing, f"__all__ names not bound: {missing}"


def test_submodules_import():
    for name in ("native", "results", "calibration", "aggregates", "loctable",
                 "audit", "metrics", "simulate", "psf",
                 "deprecated.box", "deprecated.core", "deprecated.lmga",
                 "deprecated.backend", "deprecated.patches",
                 "deprecated.calibrate", "deprecated.structs"):
        __import__(f"spotsolve.{name}")


def test_production_does_not_import_the_reference():
    import subprocess, sys
    code = ("import sys, spotsolve, spotsolve.loctable; "
            "print([m for m in sys.modules if m.startswith('spotsolve.deprecated')])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "[]", out
