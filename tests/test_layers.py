"""Run the layer verification scripts and require a clean exit.

`verify_*.py` are standalone: each prints a PASS/FAIL line per check and exits
non-zero if any failed, so they can be run by hand while working on a layer

    python tests/verify_psf.py

This wrapper is what makes `pytest` see them. They are run as subprocesses
rather than imported because they assert at module level and call `sys.exit`
-- importing one would take the test session down with it.
"""

import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = sorted(HERE.glob("verify_*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_layer(script):
    r = subprocess.run([sys.executable, str(script)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"{script.name} reported failures:\n{r.stdout}{r.stderr}")
