# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import math

import numpy as np
import pytest
import scipp as sc
import scipp.testing

from scippneutron.peaks import model


@pytest.mark.parametrize('prefix', ('', 'lorentz_'))
def test_polynomial_degree_1_call(prefix: str):
    a0 = sc.scalar(0.2, unit='cm')
    a1 = sc.scalar(-5.1, unit='cm/s')
    params = {f'{prefix}a0': a0, f'{prefix}a1': a1}
    m = model.PolynomialModel(degree=1, prefix=prefix)

    x = sc.linspace('xx', 0.8, 4.3, 100, unit='s')
    expected = a0 + a1 * x
    actual = m(x, **params)
    sc.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('prefix', ('', 'lorentz_'))
def test_polynomial_degree_2_call(prefix: str):
    a0 = sc.scalar(-1.3, unit='K')
    a1 = sc.scalar(0.03, unit='K/m')
    a2 = sc.scalar(1.7, unit='K/m^2')
    params = {f'{prefix}a0': a0, f'{prefix}a1': a1, f'{prefix}a2': a2}
    m = model.PolynomialModel(degree=2, prefix=prefix)

    x = sc.linspace('xx', 100.0, 234.0, 100, unit='m')
    expected = a0 + a1 * x + a2 * x**2
    actual = m(x, **params)
    sc.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('prefix', ('', 'lorentz_'))
def test_gaussian_call(prefix: str):
    amplitude = sc.scalar(2.8, unit='kg')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')
    params = {
        f'{prefix}amplitude': amplitude,
        f'{prefix}loc': loc,
        f'{prefix}scale': scale,
    }
    m = model.GaussianModel(prefix=prefix)

    x = sc.linspace('xx', -2.0, 3.5, 200, unit='m')
    expected = (
        amplitude
        / (math.sqrt(2 * math.pi) * scale)
        * sc.exp(-((x - loc) ** 2) / (2 * scale**2))
    )
    actual = m(x, **params)
    sc.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('prefix', ('', 'lorentz_'))
def test_lorentzian_call(prefix: str):
    amplitude = sc.scalar(2.8, unit='kg')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')
    params = {
        f'{prefix}amplitude': amplitude,
        f'{prefix}loc': loc,
        f'{prefix}scale': scale,
    }

    m = model.LorentzianModel(prefix=prefix)

    x = sc.linspace('xx', -4.0, 5.0, 200, unit='m')
    expected = amplitude / np.pi * scale / ((x - loc) ** 2 + scale**2)
    actual = m(x, **params)
    sc.testing.assert_allclose(actual, expected)


def test_polynomial_degree_1_guess_params():
    a0 = sc.scalar(0.2, unit='cm')
    a1 = sc.scalar(-5.1, unit='cm/s')

    x = sc.linspace('xx', 0.8, 4.3, 100, unit='s')
    y = a0 + a1 * x
    data = sc.DataArray(y, coords={'xx': x})

    m = model.PolynomialModel(degree=1, prefix='')
    params = m.guess(data)
    assert params.keys() == {'a0', 'a1'}
    sc.testing.assert_allclose(params['a0'], a0, atol=sc.scalar(0.1, unit='cm'))
    sc.testing.assert_allclose(params['a1'], a1, atol=sc.scalar(0.1, unit='cm/s'))


def test_polynomial_degree_1_guess_params_prefix():
    a0 = sc.scalar(0.2, unit='cm')
    a1 = sc.scalar(-5.1, unit='cm/s')

    x = sc.linspace('xx', 0.8, 4.3, 100, unit='s')
    y = a0 + a1 * x
    data = sc.DataArray(y, coords={'xx': x})

    m = model.PolynomialModel(degree=1, prefix='linear_')
    params = m.guess(data)
    assert params.keys() == {'linear_a1', 'linear_a0'}
    sc.testing.assert_allclose(params['linear_a0'], a0, atol=sc.scalar(0.1, unit='cm'))
    sc.testing.assert_allclose(
        params['linear_a1'], a1, atol=sc.scalar(0.1, unit='cm/s')
    )


def test_polynomial_degree_2_guess_params():
    a0 = sc.scalar(-1.3, unit='K')
    a1 = sc.scalar(0.03, unit='K/m')
    a2 = sc.scalar(1.7, unit='K/m^2')

    x = sc.linspace('xx', 100.0, 234.0, 100, unit='m')
    y = a0 + a1 * x + a2 * x**2
    data = sc.DataArray(y, coords={'xx': x})

    m = model.PolynomialModel(degree=2, prefix='')
    params = m.guess(data)
    assert params.keys() == {'a0', 'a1', 'a2'}
    sc.testing.assert_allclose(params['a0'], a0, atol=sc.scalar(0.1, unit='K'))
    sc.testing.assert_allclose(params['a1'], a1, atol=sc.scalar(0.1, unit='K/m'))
    sc.testing.assert_allclose(params['a2'], a2, atol=sc.scalar(0.1, unit='K/m^2'))


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


def test_gaussian_guess_params_prefix():
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

    m = model.GaussianModel(prefix='normal-')
    params = m.guess(data)
    assert params.keys() == {'normal-amplitude', 'normal-loc', 'normal-scale'}
    sc.testing.assert_allclose(
        params['normal-amplitude'], amplitude, atol=sc.scalar(0.5, unit='kg')
    )
    sc.testing.assert_allclose(params['normal-loc'], loc, atol=sc.scalar(0.5, unit='m'))
    sc.testing.assert_allclose(
        params['normal-scale'], scale, atol=sc.scalar(0.5, unit='m')
    )


def test_lorentzian_guess_params_linspace():
    amplitude = sc.scalar(2.8, unit='kg')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')

    x = sc.linspace('xx', -4.0, 5.0, 200, unit='m')
    y = amplitude / np.pi * scale / ((x - loc) ** 2 + scale**2)
    data = sc.DataArray(y, coords={'xx': x})

    m = model.LorentzianModel(prefix='')
    params = m.guess(data)
    assert params.keys() == {'amplitude', 'loc', 'scale'}
    sc.testing.assert_allclose(
        params['amplitude'], amplitude, atol=sc.scalar(0.6, unit='kg')
    )
    sc.testing.assert_allclose(params['loc'], loc, atol=sc.scalar(0.5, unit='m'))
    sc.testing.assert_allclose(params['scale'], scale, atol=sc.scalar(0.5, unit='m'))
