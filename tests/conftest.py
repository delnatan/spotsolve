"""`--runslow` gates the checks that integrate rather than differentiate.

Only one check is slow, and it is slow for a good reason: `verify_evidence`
validates the Laplace approximation against a brute-force 4-D quadrature of the
same posterior, which takes ~2 minutes. That is the check that would catch a
wrong evidence, so CI runs it -- but it should not stand between a working
change and a green `pytest`.
"""

import pytest

SLOW = {"verify_evidence"}


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run the brute-force quadrature checks (~2 min)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs --runslow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow; pass --runslow")
    for item in items:
        if any(name in item.name for name in SLOW):
            item.add_marker(skip)
