# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import itertools
import math

import numpy as np
import pytest
import scipp as sc
import scippneutron as scn


def make_beamline_dataset():
    dset = sc.Dataset()
    dset.coords['source_position'] = sc.vector(value=[0.0, 0.0, -10.0],
                                               unit='m')
    # Test assume that the sample is in the origin.
    dset.coords['sample_position'] = sc.vector(value=[0.0, 0.0, 0.0], unit='m')
    dset.coords['position'] = sc.vectors(dims=['spectrum'],
                                         values=[[1.0, 0.0, 0.0],
                                                 [0.1, 0.0, 1.0]],
                                         unit='m')
    return dset


def make_tof_dataset():
    dset = make_beamline_dataset()
    dset['counts'] = sc.arange('x', 1, 7,
                               unit='counts').fold('x', {
                                   'spectrum': 2,
                                   'tof': 3
                               })
    dset.coords['tof'] = sc.array(dims=['tof'],
                                  values=[4000.0, 5000.0, 6100.0, 7300.0],
                                  unit='us')
    return dset


def make_count_density_variable(unit):
    return sc.arange('x', 1.0, 7.0,
                     unit=sc.units.counts / unit).fold('x', {
                         'spectrum': 2,
                         'tof': 3
                     })


def test_convert_input_unchanged():
    inputs = make_tof_dataset()
    original = inputs.copy(deep=True)
    result = scn.convert(inputs,
                         origin='tof',
                         target='wavelength',
                         scatter=True)
    assert not sc.identical(result, original)
    assert sc.identical(inputs, original)


TOF_TARGET_DIMS = ('dspacing', 'wavelength', 'energy')


@pytest.mark.parametrize('target', TOF_TARGET_DIMS)
def test_convert_slice(target):
    tof = make_tof_dataset()
    expected = scn.convert(tof['counts'],
                           origin='tof',
                           target=target,
                           scatter=True)['spectrum', 0].copy()
    # A side-effect of `convert` is that it turns relevant meta data into
    # coords or attrs, depending on the target unit. Slicing (without range)
    # turns coords into attrs, but applying `convert` effectively reverses
    # this, which is why we have this slightly unusual behavior here:
    if target != 'dspacing':
        expected.coords['position'] = expected.attrs.pop('position')
    assert sc.identical(
        scn.convert(tof['counts']['spectrum', 0].copy(),
                    origin='tof',
                    target=target,
                    scatter=True), expected)
    # Converting slice of item is same as item of converted slice
    assert sc.identical(
        scn.convert(tof['counts']['spectrum', 0].copy(),
                    origin='tof',
                    target=target,
                    scatter=True).data,
        scn.convert(tof['spectrum', 0].copy(),
                    origin='tof',
                    target=target,
                    scatter=True)['counts'].data)


@pytest.mark.parametrize('target', TOF_TARGET_DIMS)
def test_convert_fail_count_density(target):
    tof = make_tof_dataset()
    # conversion with plain counts works
    converted = scn.convert(tof, origin='tof', target=target, scatter=True)
    scn.convert(converted, origin=target, target='tof', scatter=True)

    tof[''] = make_count_density_variable(tof.coords['tof'].unit)
    converted[''] = make_count_density_variable(converted.coords[target].unit)
    # conversion with count densities fails
    with pytest.raises(sc.UnitError):
        scn.convert(tof, origin='tof', target=target, scatter=True)
    with pytest.raises(sc.UnitError):
        scn.convert(converted, origin=target, target='tof', scatter=True)


def test_convert_scattering_conversion_fails_with_noscatter_mode():
    tof = make_tof_dataset()
    scn.convert(tof, origin='tof', target='dspacing',
                scatter=True)  # no exception
    with pytest.raises(RuntimeError):
        scn.convert(tof, origin='tof', target='dspacing', scatter=False)

    wavelength = scn.convert(tof,
                             origin='tof',
                             target='wavelength',
                             scatter=True)
    scn.convert(wavelength, origin='wavelength', target='Q', scatter=True)
    with pytest.raises(RuntimeError):
        scn.convert(wavelength, origin='wavelength', target='Q', scatter=False)


def test_convert_coords_vs_attributes():
    with_coords = make_tof_dataset()
    with_attrs = make_tof_dataset()
    for key in ('sample_position', 'source_position', 'position'):
        with_attrs['counts'].attrs[key] = with_attrs.coords.pop(key)

    from_coords = scn.convert(with_coords,
                              origin='tof',
                              target='wavelength',
                              scatter=True)
    from_attrs = scn.convert(with_attrs,
                             origin='tof',
                             target='wavelength',
                             scatter=True)
    assert sc.identical(from_coords, from_attrs)


def test_convert_tof_to_dspacing():
    tof = make_tof_dataset()
    dspacing = scn.convert(tof, origin='tof', target='dspacing', scatter=True)

    for key in ('tof', 'position', 'source_position', 'sample_position'):
        assert key not in dspacing.coords
    assert 'dspacing' in dspacing.coords
    assert 'counts' in dspacing
    assert dspacing['counts'].dims == ['spectrum', 'dspacing']
    assert dspacing['counts'].shape == [2, 3]
    assert dspacing['counts'].unit == sc.units.counts
    np.testing.assert_array_equal(dspacing['counts'].values.flat,
                                  np.arange(1, 7))
    assert sc.identical(dspacing['counts'].attrs['position'],
                        tof.coords['position'])

    coord = dspacing.coords['dspacing']
    # Due to conversion, the coordinate now also depends on 'spectrum'.
    assert coord.dims == ['spectrum', 'dspacing']
    assert coord.shape == [2, 4]
    assert coord.unit == sc.units.angstrom

    # Rule of thumb (https://www.psi.ch/niag/neutron-physics):
    # v [m/s] = 3956 / \lambda [ Angstrom ]
    tof_in_seconds = tof.coords['tof'] * 1e-6
    tofs = tof_in_seconds.values

    values = coord.values.flat

    # Spectrum 0 is 11 m from source
    # 2d sin(theta) = n \lambda
    # theta = 45 deg => d = lambda / (2 * 1 / sqrt(2))
    for val, t in zip(values, tofs):
        np.testing.assert_almost_equal(val,
                                       3956.0 / (11.0 / t) / math.sqrt(2.0),
                                       val * 1e-3)

    # Spectrum 1
    # sin(2 theta) = 0.1/(L-10)
    L = 10.0 + math.sqrt(1.0 * 1.0 + 0.1 * 0.1)
    lambda_to_d = 1.0 / (2.0 * math.sin(0.5 * math.asin(0.1 / (L - 10.0))))
    for val, t in zip(values[4:], tofs):
        np.testing.assert_almost_equal(val, 3956.0 / (L / t) * lambda_to_d,
                                       val * 1e-3)


def test_convert_dspacing_to_Tof():
    """Assuming the tof_to_dspacing test is correct and passing we can test the
    inverse conversion by simply comparing a round trip conversion with the
    original data."""

    tof_original = make_tof_dataset()
    dspacing = scn.convert(tof_original,
                           origin='tof',
                           target='dspacing',
                           scatter=True)
    tof = scn.convert(dspacing, origin='dspacing', target='tof', scatter=True)

    assert 'counts' in tof
    # Broadcasting is needed as conversion introduces the
    # dependance on 'spectrum'.
    expected_tofs = sc.broadcast(tof_original.coords['tof'],
                                 tof.coords['tof'].dims,
                                 tof.coords['tof'].shape)
    np.testing.assert_allclose(tof.coords['tof'].values,
                               expected_tofs.values,
                               atol=1e-12)

    for key in ('position', 'source_position', 'sample_position'):
        assert sc.identical(tof.coords[key], tof_original.coords[key])


def make_dataset_in(dim):
    if dim == 'tof':
        return make_tof_dataset()  # TODO triggers segfault otherwise
    return scn.convert(make_tof_dataset(),
                       origin='tof',
                       target=dim,
                       scatter=True)


@pytest.mark.parametrize(('origin', 'target'),
                         itertools.product(
                             ('tof', 'dspacing', 'wavelength', 'energy'),
                             repeat=2))
def test_convert_dataset_vs_dataarray(origin, target):
    if target == 'tof' and origin == 'tof':
        return  # TODO triggers segfault otherwise
    inputs = make_dataset_in(origin)
    expected = scn.convert(inputs, origin=origin, target=target, scatter=True)
    result = sc.Dataset(
        data={
            name: scn.convert(
                data.copy(), origin=origin, target=target, scatter=True)
            for name, data in inputs.items()
        })
    for name, data in result.items():
        assert sc.identical(data, expected[name])
