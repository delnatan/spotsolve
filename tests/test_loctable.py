"""The derived `peak` column: what it equals, and where it stops being flux."""

import numpy as np
import polars as pl

from spotsolve import loctable, psf
from spotsolve.results import REJECT_DTYPE, Localizations


def _result(flux, fit_sigma, sigma=1.45, info=None):
    n = len(flux)
    return Localizations(
        positions=np.column_stack((np.full(n, 20.0), np.arange(n, dtype=float) + 20)),
        amplitudes=np.asarray(flux, float),
        se=np.tile([1.0, 0.1, 0.1], (n, 1)),
        fit_sigma=np.asarray(fit_sigma, float),
        sigma_se=np.full(n, 0.05),
        rejects=np.empty(0, dtype=REJECT_DTYPE),
        background=np.full((64, 64), 60.0),
        sigma=sigma, dispersion=2.0, info=info or {},
    )


def test_peak_is_the_models_own_centre_pixel():
    """The definition, checked against the renderer rather than the formula.

    An emitter sitting exactly on a pixel centre puts `peak` in that pixel, so
    this is exact, not a tolerance on a sub-pixel interpolation.
    """
    yy, xx = (a * 1.0 for a in np.mgrid[0:41, 0:41])
    for sigma in (1.0, 1.45, 2.0):
        flux = 1234.0
        m = psf.model(psf.pack(0.0, [flux], [20.0], [20.0]), yy, xx, sigma)
        assert abs(m[20, 20] - _result([flux], [sigma]).peak[0]) < 1e-9


def test_aguet_peak_is_its_own_fitted_amplitude():
    """Aguet's sampled Gaussian reads the continuous peak, so `peak` inverts
    its own `2*pi*sigma**2` conversion exactly and recovers the fit parameter."""
    fitted_peak, sigma = 91.0, 1.45
    flux = fitted_peak * 2 * np.pi * sigma ** 2
    got = _result([flux], [sigma], info={"method": "aguet"}).peak[0]
    assert abs(got - fitted_peak) < 1e-9
    # The two conventions must NOT silently agree -- 4% apart at sigma 1.45.
    integrated = _result([flux], [sigma]).peak[0]
    assert 1.035 < got / integrated < 1.045


def test_peak_carries_width_where_flux_does_not():
    """Equal flux at different widths is equal flux and unequal peak: the
    reason `peak` is a reading aid and `flux` stays the quantity to cut on."""
    locs, _, _ = loctable.frame_tables(_result([1000.0, 1000.0], [1.0, 2.0]), 0)
    assert locs["flux"][0] == locs["flux"][1]
    assert locs["peak"][0] / locs["peak"][1] > 3.5


def test_peak_is_present_and_typed_in_both_tables():
    rej = np.array([(5.0, 6.0, 900.0, 2.9, 2.0, "too_wide")], dtype=REJECT_DTYPE)
    res = _result([1000.0], [1.45])
    res = Localizations(**{**res.__dict__, "rejects": rej})
    locs, _, _ = loctable.frame_tables(res, 0)
    rt = loctable.reject_table(res, 0)
    assert list(locs.columns) == list(loctable.LOCALIZATION_SCHEMA)
    assert list(rt.columns) == list(loctable.REJECT_SCHEMA)
    assert locs.schema["peak"] == pl.Float64
    assert abs(rt["peak"][0] - 900.0 * psf.peak_factor(2.9)) < 1e-9
