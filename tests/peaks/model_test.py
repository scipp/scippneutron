# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import math

import numpy as np
import scipp as sc
import scipp.testing

from scippneutron.peaks import model


def test_linear_guess_params():
    offset = sc.scalar(0.2, unit='cm')
    slope = sc.scalar(-5.1, unit='cm/s')

    x = sc.linspace('xx', 0.8, 4.3, 100, unit='s')
    y = offset + slope * x
    data = sc.DataArray(y, coords={'xx': x})

    m = model.LinearModel(prefix='')
    params = m.guess(data)
    assert params.keys() == {'slope', 'offset'}
    sc.testing.assert_allclose(params['slope'], slope, atol=sc.scalar(0.1, unit='cm/s'))
    sc.testing.assert_allclose(params['offset'], offset, atol=sc.scalar(0.1, unit='cm'))


def test_linear_guess_params_prefix():
    offset = sc.scalar(0.2, unit='cm')
    slope = sc.scalar(-5.1, unit='cm/s')

    x = sc.linspace('xx', 0.8, 4.3, 100, unit='s')
    y = offset + slope * x
    data = sc.DataArray(y, coords={'xx': x})

    m = model.LinearModel(prefix='linear_')
    params = m.guess(data)
    assert params.keys() == {'linear_slope', 'linear_offset'}
    sc.testing.assert_allclose(
        params['linear_slope'], slope, atol=sc.scalar(0.1, unit='cm/s')
    )
    sc.testing.assert_allclose(
        params['linear_offset'], offset, atol=sc.scalar(0.1, unit='cm')
    )


def test_gaussian_guess_params_linspace():
    amplitude = sc.scalar(2.8, unit='kg')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')

    x = sc.linspace('xx', -4.0, 5.0, 200, unit='m')
    y = (
        amplitude
        / (math.sqrt(2 * math.pi) * scale)
        * sc.exp(-((x - loc) ** 2) / (2 * scale**2))
    )
    data = sc.DataArray(y, coords={'xx': x})

    m = model.GaussianModel(prefix='')
    params = m.guess(data)
    assert params.keys() == {'amplitude', 'loc', 'scale'}
    sc.testing.assert_allclose(
        params['amplitude'], amplitude, atol=sc.scalar(0.5, unit='kg')
    )
    sc.testing.assert_allclose(params['loc'], loc, atol=sc.scalar(0.5, unit='m'))
    sc.testing.assert_allclose(params['scale'], scale, atol=sc.scalar(0.5, unit='m'))


def test_gaussian_guess_params_logspace():
    amplitude = sc.scalar(0.02, unit='kg')
    loc = sc.scalar(10.2, unit='m')
    scale = sc.scalar(2.6, unit='m')

    x = sc.geomspace('xx', 2.0, 20.0, 200, unit='m')
    y = (
        amplitude
        / (math.sqrt(2 * math.pi) * scale)
        * sc.exp(-((x - loc) ** 2) / (2 * scale**2))
    )
    data = sc.DataArray(y, coords={'xx': x})

    m = model.GaussianModel(prefix='')
    params = m.guess(data)
    assert params.keys() == {'amplitude', 'loc', 'scale'}
    sc.testing.assert_allclose(
        params['amplitude'], amplitude, atol=sc.scalar(0.05, unit='kg')
    )
    sc.testing.assert_allclose(params['loc'], loc, atol=sc.scalar(0.5, unit='m'))
    sc.testing.assert_allclose(params['scale'], scale, atol=sc.scalar(0.5, unit='m'))


def test_gaussian_guess_params_disordered():
    rng = np.random.default_rng(772)

    amplitude = sc.scalar(2.8, unit='kg')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')

    x = sc.linspace('xx', -4.0, 5.0, 200, unit='m')
    y = (
        amplitude
        / (math.sqrt(2 * math.pi) * scale)
        * sc.exp(-((x - loc) ** 2) / (2 * scale**2))
    )
    data = sc.DataArray(y, coords={'xx': x})
    data = data[list(rng.permutation(len(data)))]

    m = model.GaussianModel(prefix='')
    params = m.guess(data)
    assert params.keys() == {'amplitude', 'loc', 'scale'}
    sc.testing.assert_allclose(
        params['amplitude'], amplitude, atol=sc.scalar(0.5, unit='kg')
    )
    sc.testing.assert_allclose(params['loc'], loc, atol=sc.scalar(0.5, unit='m'))
    sc.testing.assert_allclose(params['scale'], scale, atol=sc.scalar(0.5, unit='m'))


def test_gaussian_guess_params_small_array():
    amplitude = sc.scalar(2.8, unit='kg')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')

    x = sc.linspace('xx', 0.2, 0.5, 2, unit='m')
    y = (
        amplitude
        / (math.sqrt(2 * math.pi) * scale)
        * sc.exp(-((x - loc) ** 2) / (2 * scale**2))
    )
    data = sc.DataArray(y, coords={'xx': x})

    m = model.GaussianModel(prefix='')
    params = m.guess(data)
    assert params.keys() == {'amplitude', 'loc', 'scale'}
    sc.testing.assert_allclose(
        params['amplitude'], amplitude, atol=sc.scalar(5.0, unit='kg')
    )
    sc.testing.assert_allclose(params['loc'], loc, atol=sc.scalar(0.5, unit='m'))
    sc.testing.assert_allclose(params['scale'], scale, atol=sc.scalar(0.5, unit='m'))
