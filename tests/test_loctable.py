"""The derived `peak` column: what it equals, and where it stops being flux."""

import numpy as np
import polars as pl
import pytest

from spotsolve import loctable, psf
from spotsolve.results import FitFlag, Localizations


def _result(flux, fit_sigma, sigma=1.45, info=None):
    n = len(flux)
    return Localizations(
        positions=np.column_stack((np.full(n, 20.0), np.arange(n, dtype=float) + 20)),
        amplitudes=np.asarray(flux, float),
        se=np.tile([1.0, 0.1, 0.1], (n, 1)),
        fit_sigma=np.asarray(fit_sigma, float),
        sigma_se=np.full(n, 0.05),
        flags=np.zeros(n, dtype=np.uint8),
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
    locs, _ = loctable.frame_tables(_result([1000.0, 1000.0], [1.0, 2.0]), 0)
    assert locs["flux"][0] == locs["flux"][1]
    assert locs["peak"][0] / locs["peak"][1] > 3.5


def test_tables_preserve_measurements_flags_and_uncertainties():
    result = _result([100., 10000.], [0.75, 3.0], info={
        "fisher_fraction": np.array([[.1, .2, .3, .4], [.5, .6, .7, .8]]),
        "fitted_background": [21., 22.],
    })
    result.flags[:] = [0, int(FitFlag.EDGE | FitFlag.AT_BOUND)]
    locs, summary = loctable.frame_tables(result, 2, pixel_size=.1, loc_id0=12)
    assert locs.height == len(result)
    assert list(locs.columns) == list(loctable.LOCALIZATION_SCHEMA)
    assert locs.schema["flags"] == pl.UInt8
    np.testing.assert_array_equal(locs["flags"], result.flags)
    np.testing.assert_array_equal(locs["sigma_se"], result.sigma_se)
    np.testing.assert_array_equal(locs["fisher_sigma"], [.4, .8])
    np.testing.assert_array_equal(locs["bg"], [21., 22.])
    np.testing.assert_array_equal(locs["peak"], result.peak)
    np.testing.assert_array_equal(locs["loc_id"], [12, 13])
    assert summary["n_locs"][0] == 2 and summary["n_flagged"][0] == 1
    assert "is_aggregate" not in locs.columns


def test_empty_tables_keep_schema_and_missing_diagnostics_are_nan():
    empty, summary = loctable.frame_tables(_result([], []), 0)
    assert empty.schema == loctable.LOCALIZATION_SCHEMA
    assert summary["n_locs"][0] == summary["n_flagged"][0] == 0
    locs, _ = loctable.frame_tables(_result([100.], [1.]), 0)
    assert np.isnan(locs["fisher_sigma"][0])


def test_quality_filter_excludes_invalid_coordinates_and_preserves_rows():
    locs = pl.DataFrame({
        "loc_id": [8, 3, 7, 1, 5, 9],
        "x": [0., float("nan"), 2., 3., 4., 5.],
        "y": [0., 1., 2., 3., 4., 5.],
        "se_x": [0.3, 0.3, 0., None, float("inf"), 0.6],
        "se_y": [0.4, 0.4, 0.4, 0.4, 0.4, 0.8],
        "se_pos": [100.] * 6,  # stale derived values must not decide the cut
    })
    valid = loctable.filter_quality(locs)
    assert valid.equals(locs[[0, 5]])
    precise = loctable.filter_quality(locs, max_se_pos=0.5)
    assert precise.equals(locs[[0]])
    assert precise.schema == locs.schema
    assert loctable.filter_quality(locs.head(0), max_se_pos=0.5).equals(locs.head(0))


def test_quality_filter_flux_is_optional_and_uses_uncertainty():
    locs, _ = loctable.frame_tables(_result([6., 6., 6., 6., 6.], [1.45] * 5), 0)
    locs = locs.with_columns(
        pl.Series("se_flux", [2., 3., 0., float("nan"), None]),
        pl.lit(100.).alias("flux_snr"),  # stale derived SNR
    )
    assert loctable.filter_quality(locs).equals(locs)
    assert loctable.filter_quality(locs, min_flux_snr=3.).equals(locs[[0]])
    assert loctable.filter_quality(locs, max_se_pos=0.1, min_flux_snr=3.).is_empty()
    assert loctable.filter_quality(locs, max_se_pos=0.2, min_flux_snr=3.).equals(locs[[0]])


def test_quality_filter_coordinate_units_and_extreme_precision_values():
    locs, _ = loctable.frame_tables(_result([100.], [1.45]), 0)
    original = loctable.filter_quality(locs, max_se_pos=0.2)
    converted = locs.with_columns([pl.col(c) * 0.104 for c in ("x", "y", "se_x", "se_y")])
    assert loctable.filter_quality(converted, max_se_pos=0.0208)["loc_id"].equals(original["loc_id"])
    tiny = locs.with_columns(pl.lit(1e-200).alias("se_x"), pl.lit(1e-200).alias("se_y"))
    assert loctable.filter_quality(tiny, max_se_pos=1e-200).is_empty()
    huge = locs.with_columns(pl.lit(1e200).alias("se_x"), pl.lit(1e200).alias("se_y"))
    assert loctable.filter_quality(huge, max_se_pos=1.5e200).equals(huge)


@pytest.mark.parametrize("cutoff", [0, -1, float("nan"), float("inf"), True, "0.5", [0.5]])
def test_quality_filter_invalid_cutoffs(cutoff):
    locs, _ = loctable.frame_tables(_result([100.], [1.45]), 0)
    for name in ("max_se_pos", "min_flux_snr"):
        with pytest.raises(ValueError, match=f"{name} must be positive and finite"):
            loctable.filter_quality(locs, **{name: cutoff})
