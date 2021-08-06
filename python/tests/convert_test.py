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
    dset.coords['source_position'] = sc.vector(value=[0.0, 0.0, -10.0], unit='m')
    # Test assume that the sample is in the origin.
    dset.coords['sample_position'] = sc.vector(value=[0.0, 0.0, 0.0], unit='m')
    dset.coords['position'] = sc.vectors(dims=['spectrum'],
                                         values=[[1.0, 0.0, 0.0], [0.1, 0.0, 1.0]],
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


def make_tof_binned_events():
    buffer = sc.DataArray(
        sc.zeros(dims=['event'], shape=[7], dtype=float),
        coords={
            'tof':
            sc.array(dims=['event'],
                     values=[1000.0, 3000.0, 2000.0, 4000.0, 5000.0, 6000.0, 3000.0],
                     unit='us')
        })
    return sc.bins(data=buffer,
                   dim='event',
                   begin=sc.array(dims=['spectrum'], values=[0, 4]),
                   end=sc.array(dims=['spectrum'], values=[4, 7]))


def make_count_density_variable(unit):
    return sc.arange('x', 1.0, 7.0,
                     unit=sc.units.counts / unit).fold('x', {
                         'spectrum': 2,
                         'tof': 3
                     })


def check_tof_conversion_metadata(converted, target, coord_unit):
    assert 'tof' not in converted.coords
    assert target in converted.coords
    assert 'counts' in converted
    assert converted['counts'].dims == ['spectrum', target]
    assert converted['counts'].shape == [2, 3]
    assert converted['counts'].unit == sc.units.counts
    np.testing.assert_array_equal(converted['counts'].values.flat, np.arange(1, 7))

    coord = converted.coords[target]
    # Due to conversion, the coordinate now also depends on 'spectrum'.
    assert coord.dims == ['spectrum', target]
    assert coord.shape == [2, 4]
    assert coord.unit == coord_unit


def check_tof_round_trip(via):
    tof_original = make_tof_dataset()
    converted = scn.convert(tof_original, origin='tof', target=via, scatter=True)
    tof = scn.convert(converted, origin=via, target='tof', scatter=True)

    assert 'counts' in tof
    assert sc.all(
        sc.isclose(tof.coords['tof'],
                   tof_original.coords['tof'],
                   rtol=sc.scalar(0.0),
                   atol=sc.scalar(1e-12, unit=tof.coords['tof'].unit))).value
    for key in ('position', 'source_position', 'sample_position'):
        assert sc.identical(tof.coords[key], tof_original.coords[key])


def make_dataset_in(dim):
    if dim == 'tof':
        return make_tof_dataset()  # TODO triggers segfault otherwise
    return scn.convert(make_tof_dataset(), origin='tof', target=dim, scatter=True)


@pytest.mark.parametrize(('origin', 'target'),
                         itertools.product(('tof', 'dspacing', 'wavelength', 'energy'),
                                           repeat=2))
def test_convert_dataset_vs_dataarray(origin, target):
    if target == 'tof' and origin == 'tof':
        return  # TODO triggers segfault otherwise
    inputs = make_dataset_in(origin)
    expected = scn.convert(inputs, origin=origin, target=target, scatter=True)
    result = sc.Dataset(
        data={
            name: scn.convert(data.copy(), origin=origin, target=target, scatter=True)
            for name, data in inputs.items()
        })
    for name, data in result.items():
        assert sc.identical(data, expected[name])


def test_convert_input_unchanged():
    inputs = make_tof_dataset()
    original = inputs.copy(deep=True)
    result = scn.convert(inputs, origin='tof', target='wavelength', scatter=True)
    assert not sc.identical(result, original)
    assert sc.identical(inputs, original)


TOF_TARGET_DIMS = ('dspacing', 'wavelength', 'energy')


@pytest.mark.parametrize('target', TOF_TARGET_DIMS)
def test_convert_slice(target):
    tof = make_tof_dataset()
    expected = scn.convert(tof['counts'], origin='tof', target=target,
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
    scn.convert(tof, origin='tof', target='dspacing', scatter=True)  # no exception
    with pytest.raises(RuntimeError):
        scn.convert(tof, origin='tof', target='dspacing', scatter=False)

    wavelength = scn.convert(tof, origin='tof', target='wavelength', scatter=True)
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
    check_tof_conversion_metadata(dspacing, 'dspacing', sc.units.angstrom)
    for key in ('position', 'source_position', 'sample_position'):
        assert sc.identical(dspacing['counts'].attrs[key], tof.coords[key])

    # Rule of thumb (https://www.psi.ch/niag/neutron-physics):
    # v [m/s] = 3956 / \lambda [ Angstrom ]
    tof_in_seconds = tof.coords['tof'] * 1e-6

    # Spectrum 0 is 11 m from source
    # 2d sin(theta) = n \lambda
    # theta = 45 deg => d = lambda / (2 * 1 / sqrt(2))
    for val, t in zip(dspacing.coords['dspacing']['spectrum', 0].values,
                      tof_in_seconds.values):
        np.testing.assert_almost_equal(val, 3956.0 / (11.0 / t) / math.sqrt(2.0),
                                       val * 1e-3)

    # Spectrum 1
    # sin(2 theta) = 0.1/(L-10)
    L = 10.0 + math.sqrt(1.0 * 1.0 + 0.1 * 0.1)
    lambda_to_d = 1.0 / (2.0 * math.sin(0.5 * math.asin(0.1 / (L - 10.0))))
    for val, t in zip(dspacing.coords['dspacing']['spectrum', 1].values,
                      tof_in_seconds.values):
        np.testing.assert_almost_equal(val, 3956.0 / (L / t) * lambda_to_d, val * 1e-3)


def test_convert_dspacing_to_tof():
    """Assuming the tof_to_dspacing test is correct and passing we can test the
    inverse conversion by simply comparing a round trip conversion with the
    original data."""
    check_tof_round_trip(via='dspacing')


def test_convert_tof_to_wavelength():
    tof = make_tof_dataset()
    wavelength = scn.convert(tof, origin='tof', target='wavelength', scatter=True)
    check_tof_conversion_metadata(wavelength, 'wavelength', sc.units.angstrom)
    for key in ('position', 'source_position', 'sample_position'):
        assert sc.identical(wavelength['counts'].attrs[key], tof.coords[key])

    # Rule of thumb (https://www.psi.ch/niag/neutron-physics):
    # v [m/s] = 3956 / \lambda [ Angstrom ]
    tof_in_seconds = tof.coords['tof'] * 1e-6

    # Spectrum 0 is 11 m from source
    for val, t in zip(wavelength.coords['wavelength']['spectrum', 0].values,
                      tof_in_seconds.values):
        np.testing.assert_almost_equal(val, 3956.0 / (11.0 / t), val * 1e-3)
    # Spectrum 1
    L = 10.0 + math.sqrt(1.0 * 1.0 + 0.1 * 0.1)
    for val, t in zip(wavelength.coords['wavelength']['spectrum', 1].values,
                      tof_in_seconds.values):
        np.testing.assert_almost_equal(val, 3956.0 / (L / t), val * 1e-3)


def test_convert_wavelength_to_tof():
    """Assuming the tof_to_wavelength test is correct and passing we can test the
    inverse conversion by simply comparing a round trip conversion with the
    original data."""
    check_tof_round_trip(via='wavelength')


def test_convert_tof_to_energy_elastic():
    tof = make_tof_dataset()
    energy = scn.convert(tof, origin='tof', target='energy', scatter=True)
    check_tof_conversion_metadata(energy, 'energy', sc.units.meV)
    for key in ('position', 'source_position', 'sample_position'):
        assert sc.identical(energy['counts'].attrs[key], tof.coords[key])

    # Rule of thumb (https://www.psi.ch/niag/neutron-physics):
    # v [m/s] = 3956 / \lambda [ Angstrom ]
    tof_in_seconds = tof.coords['tof'] * 1e-6
    # e [J] = 1/2 m(n) [kg] (l [m] / tof [s])^2
    joule_to_mev = 6.241509125883257e21
    neutron_mass = 1.674927471e-27

    # Spectrum 0 is 11 m from source
    for val, t in zip(energy.coords['energy']['spectrum', 0].values,
                      tof_in_seconds.values):
        np.testing.assert_almost_equal(val,
                                       joule_to_mev * 0.5 * neutron_mass * (11 / t)**2,
                                       val * 1e-3)
    # Spectrum 1
    L = 10.0 + math.sqrt(1.0 * 1.0 + 0.1 * 0.1)
    for val, t in zip(energy.coords['energy']['spectrum', 1].values,
                      tof_in_seconds.values):
        np.testing.assert_almost_equal(val,
                                       joule_to_mev * 0.5 * neutron_mass * (L / t)**2,
                                       val * 1e-3)


def test_convert_tof_to_energy_elastic_fails_if_inelastic_params_present():
    # Note these conversion fail only because they are not implemented.
    # It should definitely be possible to support this.
    tof = make_tof_dataset()
    scn.convert(tof, origin='tof', target='energy', scatter=True)
    tof.coords['incident_energy'] = 2.1 * sc.units.meV
    with pytest.raises(RuntimeError):
        scn.convert(tof, origin='tof', target='energy', scatter=True)

    del tof.coords['incident_energy']
    scn.convert(tof, origin='tof', target='energy', scatter=True)
    tof.coords['final_energy'] = 2.1 * sc.units.meV
    with pytest.raises(RuntimeError):
        scn.convert(tof, origin='tof', target='energy', scatter=True)


def test_convert_energy_to_tof_elastic():
    """Assuming the tof_to_energy_elastic test is correct and passing we can test
    the inverse conversion by simply comparing a round trip conversion with the
    original data."""
    check_tof_round_trip(via='energy')


def test_convert_tof_to_energy_transfer_direct():
    tof = make_tof_dataset()
    with pytest.raises(RuntimeError):
        scn.convert(tof, origin='tof', target='energy_transfer', scatter=True)
    tof.coords['incident_energy'] = 35.0 * sc.units.meV
    direct = scn.convert(tof, origin='tof', target='energy_transfer', scatter=True)
    assert 'energy_transfer' in direct.coords
    assert 'tof' not in direct.coords
    tof_direct = scn.convert(direct,
                             origin='energy_transfer',
                             target='tof',
                             scatter=True)
    assert sc.all(
        sc.isclose(tof_direct.coords['tof'],
                   tof.coords['tof'],
                   rtol=0.0 * sc.units.one,
                   atol=1e-11 * sc.units.us)).value
    tof_direct.coords['tof'] = tof.coords['tof']
    assert sc.identical(tof_direct, tof)


def test_convert_tof_to_energy_transfer_indirect():
    tof = make_tof_dataset()
    with pytest.raises(RuntimeError):
        scn.convert(tof, origin='tof', target='energy_transfer', scatter=True)
    tof.coords['incident_energy'] = 25.0 * sc.units.meV
    tof.coords['final_energy'] = 35.0 * sc.units.meV
    with pytest.raises(RuntimeError):
        scn.convert(tof, origin='tof', target='energy_transfer', scatter=True)
    del tof.coords['incident_energy']
    indirect = scn.convert(tof, origin='tof', target='energy_transfer', scatter=True)
    assert 'energy_transfer' in indirect.coords
    assert 'tof' not in indirect.coords
    tof_indirect = scn.convert(indirect,
                               origin='energy_transfer',
                               target='tof',
                               scatter=True)
    assert sc.all(
        sc.isclose(tof_indirect.coords['tof'],
                   tof.coords['tof'],
                   rtol=0.0 * sc.units.one,
                   atol=1e-11 * sc.units.us)).value
    tof_indirect.coords['tof'] = tof.coords['tof']
    assert sc.identical(tof_indirect, tof)


def test_convert_tof_to_energy_transfer_direct_indirect_are_distinct():
    tof_direct = make_tof_dataset()
    tof_direct.coords['incident_energy'] = 22.0 * sc.units.meV
    direct = scn.convert(tof_direct,
                         origin='tof',
                         target='energy_transfer',
                         scatter=True)

    tof_indirect = make_tof_dataset()
    tof_indirect.coords['final_energy'] = 22.0 * sc.units.meV
    indirect = scn.convert(tof_indirect,
                           origin='tof',
                           target='energy_transfer',
                           scatter=True)
    assert not sc.all(
        sc.isclose(direct.coords['energy_transfer'],
                   indirect.coords['energy_transfer'],
                   rtol=0.0 * sc.units.one,
                   atol=1e-11 * sc.units.meV)).value


@pytest.mark.parametrize('target', TOF_TARGET_DIMS)
def test_convert_with_factor_type_promotion(target):
    tof = make_tof_dataset()
    tof.coords['tof'] = sc.array(dims=['tof'],
                                 values=[4000, 5000, 6100, 7300],
                                 unit='us',
                                 dtype='float32')
    res = scn.convert(tof, origin='tof', target=target, scatter=True)
    assert res.coords[target].dtype == sc.dtype.float32
    res = scn.convert(res, origin=target, target='tof', scatter=True)
    assert res.coords['tof'].dtype == sc.dtype.float32


@pytest.mark.parametrize('target', TOF_TARGET_DIMS)
def test_convert_binned_events_converted(target):
    tof = make_tof_dataset()
    # Standard dense coord for comparison purposes. The final 0 is a dummy.
    tof.coords['tof'] = sc.array(dims=['spectrum', 'tof'],
                                 values=[[1000.0, 3000.0, 2000.0, 4000.0],
                                         [5000.0, 6000.0, 3000.0, 0.0]],
                                 unit='us')
    tof['events'] = make_tof_binned_events()
    original = tof.copy(deep=True)
    assert sc.identical(tof, original)

    res = scn.convert(tof, origin='tof', target=target, scatter=True)
    values = res['events'].values
    for bin_index in range(1):
        expected = res.coords[target]['spectrum',
                                      bin_index].rename_dims({target: 'event'})
        assert 'tof' not in values[bin_index].coords
        assert target in values[bin_index].coords
        assert sc.identical(values[bin_index].coords[target], expected)


@pytest.mark.parametrize('target', TOF_TARGET_DIMS)
def test_convert_binned_convert_slice(target):
    tof = make_tof_dataset()['counts']['tof', 0].copy()
    tof.data = make_tof_binned_events()
    original = tof.copy()
    full = scn.convert(tof, origin='tof', target=target, scatter=True)
    sliced = scn.convert(tof['spectrum', 1:2],
                         origin='tof',
                         target=target,
                         scatter=True)
    assert sc.identical(sliced, full['spectrum', 1:2])
    assert sc.identical(tof, original)
