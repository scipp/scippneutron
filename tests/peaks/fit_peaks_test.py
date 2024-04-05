# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
import scipp.testing

from scippneutron.peaks import fit_peaks
from scippneutron.peaks.model import GaussianModel, PolynomialModel


def test_fit_peaks_rejects_descending_coord():
    data = sc.DataArray(sc.ones(sizes={'x': 10}), coords={'x': -sc.arange('x', 10.0)})
    with pytest.raises(sc.CoordError):
        fit_peaks(data, sc.array(dims=['x'], values=[5.0]), windows=sc.scalar(1.0))


def test_fit_peaks_rejects_disordered_coord():
    rng = np.random.default_rng(93)
    data = sc.DataArray(sc.ones(sizes={'x': 10}), coords={'x': sc.arange('x', 10.0)})
    data = data[list(rng.permutation(len(data)))]
    with pytest.raises(sc.CoordError):
        fit_peaks(data, sc.array(dims=['x'], values=[5.0]), windows=sc.scalar(1.0))


def test_fit_peaks_finds_single_gaussian_peak_linear_background():
    rng = np.random.default_rng(663)
    model = GaussianModel(prefix='peak_') + PolynomialModel(degree=1, prefix='bkg_')
    real_params = {
        'peak_amplitude': sc.scalar(2.7, unit='K*m'),
        'peak_loc': sc.scalar(14.0, unit='m'),
        'peak_scale': sc.scalar(2.8, unit='m'),
        'bkg_a0': sc.scalar(-4.9, unit='K'),
        'bkg_a1': sc.scalar(1.8, unit='K/m'),
    }

    x = sc.linspace('x', 3.0, 20.0, 100, unit='m')
    y = model(x, **real_params)
    y.values += rng.normal(0.0, 0.01, len(y))
    y.variances = y.values / 3.0
    data = sc.DataArray(y, coords={'x': x})

    [result] = fit_peaks(
        data,
        peak_estimates=sc.array(dims=['x'], values=[13.0], unit='m'),
        windows=sc.scalar(15.0, unit='m'),
    )
    assert result.success
    sc.testing.assert_allclose(
        sc.values(result.popt['peak_amplitude']),
        real_params['peak_amplitude'],
        rtol=sc.scalar(0.1),
    )
    sc.testing.assert_allclose(
        sc.values(result.popt['peak_loc']), real_params['peak_loc'], rtol=sc.scalar(0.1)
    )
    sc.testing.assert_allclose(
        sc.values(result.popt['peak_scale']),
        real_params['peak_scale'],
        rtol=sc.scalar(0.1),
    )
    sc.testing.assert_allclose(
        sc.values(result.popt['bkg_a0']), real_params['bkg_a0'], rtol=sc.scalar(0.1)
    )
    sc.testing.assert_allclose(
        sc.values(result.popt['bkg_a1']), real_params['bkg_a1'], rtol=sc.scalar(0.1)
    )

    lo = result.best_fit.coords['x'].min()
    hi = result.best_fit.coords['x'].max()
    data_in_window = data['x', lo : hi + sc.scalar(1e-8, unit='m')]
    sc.testing.assert_allclose(
        sc.values(result.best_fit), data_in_window, rtol=sc.scalar(0.1)
    )
