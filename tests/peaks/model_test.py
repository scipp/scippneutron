# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import math

import numpy as np
import pytest
import scipp as sc
import scipp.testing

from scippneutron.peaks import model


@pytest.mark.parametrize('prefix', ['', 'lorentz_'])
def test_polynomial_degree_1_call(prefix: str):
    a0 = sc.scalar(0.2, unit='cm')
    a1 = sc.scalar(-5.1, unit='cm/s')
    params = {f'{prefix}a0': a0, f'{prefix}a1': a1}
    m = model.PolynomialModel(degree=1, prefix=prefix)

    x = sc.linspace('xx', 0.8, 4.3, 100, unit='s')
    expected = a0 + a1 * x
    actual = m(x, **params)
    sc.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('prefix', ['', 'lorentz_'])
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


@pytest.mark.parametrize('prefix', ['', 'lorentz_'])
def test_polynomial_degree_3_call(prefix: str):
    a0 = sc.scalar(-1.3, unit='K')
    a1 = sc.scalar(0.03, unit='K/m')
    a2 = sc.scalar(1.7, unit='K/m^2')
    a3 = sc.scalar(0.9, unit='K/m^3')
    params = {
        f'{prefix}a0': a0,
        f'{prefix}a1': a1,
        f'{prefix}a2': a2,
        f'{prefix}a3': a3,
    }
    m = model.PolynomialModel(degree=3, prefix=prefix)

    x = sc.linspace('xx', 100.0, 234.0, 100, unit='m')
    expected = a0 + a1 * x + a2 * x**2 + a3 * x**3
    actual = m(x, **params)
    sc.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('prefix', ['', 'lorentz_'])
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


@pytest.mark.parametrize('prefix', ['', 'lorentz_'])
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


@pytest.mark.parametrize('prefix', ['', 'lorentz_'])
@pytest.mark.parametrize('fraction', [0.0, 1.0, 0.5, 0.3])
def test_pseudo_voigt_call(prefix: str, fraction: float):
    amplitude = sc.scalar(2.8, unit='kg*m')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')
    fraction = sc.scalar(fraction)
    params = {
        f'{prefix}amplitude': amplitude,
        f'{prefix}loc': loc,
        f'{prefix}scale': scale,
        f'{prefix}fraction': fraction,
    }

    m = model.PseudoVoigtModel(prefix=prefix)

    x = sc.linspace('xx', -4.0, 5.0, 200, unit='m')
    scale_g = scale / math.sqrt(2 * math.log(2))
    gaussian = (
        amplitude
        / (math.sqrt(2 * math.pi) * scale_g)
        * sc.exp(-((x - loc) ** 2) / (2 * scale_g**2))
    )
    lorentzian = amplitude / np.pi * scale / ((x - loc) ** 2 + scale**2)
    expected = fraction * lorentzian + (1 - fraction) * gaussian
    actual = m(x, **params)
    sc.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('prefix', ['', 'pre_'])
def test_composite_call(prefix: str):
    amplitude = sc.scalar(2.8, unit='kg*m')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')
    a0 = sc.scalar(-0.4, unit='kg')
    a1 = sc.scalar(0.05, unit='kg/m')
    params = {
        f'{prefix}n_amplitude': amplitude,
        f'{prefix}n_loc': loc,
        f'{prefix}n_scale': scale,
        f'{prefix}p_a0': a0,
        f'{prefix}p_a1': a1,
    }
    m = model.CompositeModel(
        model.PolynomialModel(degree=1, prefix='p_'),
        model.GaussianModel(prefix='n_'),
        prefix=prefix,
    )

    x = sc.linspace('xx', -4.0, 5.0, 200, unit='m')
    expected_p = a0 + a1 * x
    expected_n = (
        amplitude
        / (math.sqrt(2 * math.pi) * scale)
        * sc.exp(-((x - loc) ** 2) / (2 * scale**2))
    )
    expected = expected_p + expected_n
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


def test_lorentzian_guess_params():
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


@pytest.mark.parametrize('fraction', [0.0, 1.0, 0.5, 0.3])
def test_pseudo_voigt_guess_params(fraction: float):
    amplitude = sc.scalar(2.8, unit='kg')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')
    fraction = sc.scalar(fraction)

    x = sc.linspace('xx', -4.0, 5.0, 200, unit='m')
    scale_g = scale / math.sqrt(2 * math.log(2))
    gaussian = (
        amplitude
        / (math.sqrt(2 * math.pi) * scale_g)
        * sc.exp(-((x - loc) ** 2) / (2 * scale_g**2))
    )
    lorentzian = amplitude / np.pi * scale / ((x - loc) ** 2 + scale**2)
    y = (1 - fraction) * gaussian + fraction * lorentzian
    data = sc.DataArray(y, coords={'xx': x})

    m = model.PseudoVoigtModel(prefix='')
    params = m.guess(data)
    assert params.keys() == {'amplitude', 'loc', 'scale', 'fraction'}
    sc.testing.assert_allclose(
        params['amplitude'], amplitude, atol=sc.scalar(0.6, unit='kg')
    )
    sc.testing.assert_allclose(params['loc'], loc, atol=sc.scalar(0.5, unit='m'))
    sc.testing.assert_allclose(params['scale'], scale, atol=sc.scalar(0.5, unit='m'))
    sc.testing.assert_identical(params['fraction'], sc.scalar(0.5))


def test_composite_guess_params():
    amplitude = sc.scalar(2.8, unit='kg*m')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')
    a0 = sc.scalar(-0.4, unit='kg')
    a1 = sc.scalar(0.05, unit='kg/m')
    x = sc.linspace('xx', -4.0, 5.0, 200, unit='m')
    y_p = a0 + a1 * x
    y_n = (
        amplitude
        / (math.sqrt(2 * math.pi) * scale)
        * sc.exp(-((x - loc) ** 2) / (2 * scale**2))
    )
    y = y_p + y_n
    data = sc.DataArray(y, coords={'xx': x})

    m = model.CompositeModel(
        model.PolynomialModel(degree=1, prefix='p_'),
        model.GaussianModel(prefix='n_'),
        prefix='',
    )
    params = m.guess(data)
    assert params.keys() == {'p_a0', 'p_a1', 'n_amplitude', 'n_loc', 'n_scale'}
    sc.testing.assert_allclose(params['p_a0'], a0, atol=sc.scalar(0.5, unit='kg'))
    sc.testing.assert_allclose(params['p_a1'], a1, atol=sc.scalar(0.5, unit='kg/m'))
    sc.testing.assert_allclose(
        params['n_amplitude'], amplitude, atol=sc.scalar(0.6, unit='kg*m')
    )
    sc.testing.assert_allclose(params['n_loc'], loc, atol=sc.scalar(0.5, unit='m'))
    sc.testing.assert_allclose(params['n_scale'], scale, atol=sc.scalar(0.5, unit='m'))


def test_composite_from_sum():
    amplitude = sc.scalar(2.8, unit='kg*m')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')
    a0 = sc.scalar(-0.4, unit='kg')
    a1 = sc.scalar(0.05, unit='kg/m')
    params = {
        'n_amplitude': amplitude,
        'n_loc': loc,
        'n_scale': scale,
        'p_a0': a0,
        'p_a1': a1,
    }
    m = model.PolynomialModel(degree=1, prefix='p_') + model.GaussianModel(prefix='n_')

    x = sc.linspace('xx', -4.0, 5.0, 200, unit='m')
    expected_p = a0 + a1 * x
    expected_n = (
        amplitude
        / (math.sqrt(2 * math.pi) * scale)
        * sc.exp(-((x - loc) ** 2) / (2 * scale**2))
    )
    expected = expected_p + expected_n
    actual = m(x, **params)
    sc.testing.assert_allclose(actual, expected)


def test_composite_without_prefixes():
    amplitude = sc.scalar(2.8, unit='kg*m')
    loc = sc.scalar(0.4, unit='m')
    scale = sc.scalar(0.1, unit='m')
    a0 = sc.scalar(-0.4, unit='kg')
    a1 = sc.scalar(0.05, unit='kg/m')
    params = {
        'amplitude': amplitude,
        'loc': loc,
        'scale': scale,
        'a0': a0,
        'a1': a1,
    }

    left = model.PolynomialModel(degree=1)
    right = model.GaussianModel()
    m = left + right

    x = sc.linspace('xx', -4.0, 5.0, 200, unit='m')
    expected_p = a0 + a1 * x
    expected_n = (
        amplitude
        / (math.sqrt(2 * math.pi) * scale)
        * sc.exp(-((x - loc) ** 2) / (2 * scale**2))
    )
    expected = expected_p + expected_n
    actual = m(x, **params)
    sc.testing.assert_allclose(actual, expected)


def test_composite_param_name_clash():
    left = model.PolynomialModel(degree=1, prefix='p_')
    right = model.PolynomialModel(degree=2, prefix='p_')
    with pytest.raises(ValueError, match='overlap'):
        left + right


def test_composite_clash_between_param_and_prefix():
    amplitude1 = sc.scalar(2.8, unit='kg*m')
    loc1 = sc.scalar(0.4, unit='m')
    scale1 = sc.scalar(0.1, unit='m')
    amplitude2 = sc.scalar(1.7, unit='kg*m')
    loc2 = sc.scalar(1.2, unit='m')
    scale2 = sc.scalar(0.2, unit='m')
    params = {
        'amplitude': amplitude1,
        'loc': loc1,
        'scale': scale1,
        'amplitudeamplitude': amplitude2,
        'amplitudeloc': loc2,
        'amplitudescale': scale2,
    }

    left = model.GaussianModel(prefix='')  # has param 'amplitude'
    right = model.GaussianModel(prefix='amplitude')  # has param 'amplitudeamplitude'
    m = left + right

    x = sc.linspace('xx', -4.0, 5.0, 200, unit='m')
    expected1 = (
        amplitude1
        / (math.sqrt(2 * math.pi) * scale1)
        * sc.exp(-((x - loc1) ** 2) / (2 * scale1**2))
    )
    expected2 = (
        amplitude2
        / (math.sqrt(2 * math.pi) * scale2)
        * sc.exp(-((x - loc2) ** 2) / (2 * scale2**2))
    )
    expected = expected1 + expected2
    actual = m(x, **params)
    sc.testing.assert_allclose(actual, expected)


def test_polynomial_param_bounds():
    m = model.PolynomialModel(degree=1)
    assert m.param_bounds == {}


def test_gaussian_param_bounds():
    m = model.GaussianModel()
    assert m.param_bounds == {
        'scale': (0.0, np.inf),
    }


def test_gaussian_param_bounds_prefix():
    m = model.GaussianModel(prefix='g_')
    assert m.param_bounds == {
        'g_scale': (0.0, np.inf),
    }


def test_lorentzian_param_bounds():
    m = model.LorentzianModel()
    assert m.param_bounds == {
        'scale': (0.0, np.inf),
    }


def test_pseudo_voigt_param_bounds():
    m = model.PseudoVoigtModel()
    assert m.param_bounds == {
        'scale': (0.0, np.inf),
        'fraction': (0.0, 1.0),
    }
