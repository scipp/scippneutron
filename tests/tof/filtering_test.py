# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
import numpy as np
import scipp as sc

from scippneutron.tof import filtering

# TODO proton charge time monotonic?

# TODO pass charge as string


def make_data_with_pulse_time(rng, n_event):
    start_time = sc.scalar(np.datetime64('2022-03-14T14:42:37.12345', 'ns'))
    pulse_time = (
        start_time +
        sc.array(dims=['event'], values=rng.integers(0, 10**6, n_event), unit='ns'))
    events = sc.DataArray(sc.array(dims=['event'],
                                   values=rng.uniform(0.0, 2.0, n_event),
                                   unit='counts'),
                          coords={
                              'tof':
                              sc.array(dims=['event'],
                                       values=rng.integers(10, 1000, n_event),
                                       unit='us'),
                              'pulse_time':
                              pulse_time,
                              'spectrum':
                              sc.array(dims=['event'],
                                       values=rng.integers(0, 10, n_event),
                                       unit=None),
                          })
    return sc.bin(events,
                  edges=[sc.array(dims=['tof'], values=[10, 500, 1000], unit='us')],
                  groups=[sc.arange('spectrum', 0, 10, unit=None)])


def test_make_data_with_pulse_time():
    rng = np.random.default_rng(9461)
    data = make_data_with_pulse_time(rng, 100)
    assert 'pulse_time' in data.bins.coords


def make_data_with_pulse_time_and_proton_charge(rng, n_event, n_proton_charge,
                                                bad_charge, bad_charge_indices):
    data = make_data_with_pulse_time(rng, n_event)

    start_time = data.bins.coords['pulse_time'].min()
    pulse_time = (start_time + sc.linspace(
        'pulse_time', 0, 10**6, n_proton_charge, unit='ns').to(dtype='int64'))
    good_charge_value = bad_charge.value * 100
    proton_charge = sc.DataArray(sc.array(dims=['pulse_time'],
                                          values=rng.uniform(good_charge_value,
                                                             good_charge_value * 1.2,
                                                             n_proton_charge),
                                          unit=bad_charge.unit),
                                 coords={'pulse_time': pulse_time})

    for i in bad_charge_indices:
        proton_charge[i] = bad_charge

    data.attrs['proton_charge'] = sc.scalar(proton_charge)
    return data


def test_make_data_with_pulse_time_and_proton_charge():
    rng = np.random.default_rng(65501)
    bad_charge = sc.scalar(1.0e5, unit='pC')
    data = make_data_with_pulse_time_and_proton_charge(rng, 100, 300, bad_charge,
                                                       [0, 2, 4])
    assert 'pulse_time' in data.bins.coords
    assert sc.identical(data.attrs['proton_charge'].value.data[0], bad_charge)
    assert sc.identical(data.attrs['proton_charge'].value.data[2], bad_charge)
    assert sc.identical(data.attrs['proton_charge'].value.data[4], bad_charge)
    assert (data.attrs['proton_charge'].value.data[1] > bad_charge).value
    assert (data.attrs['proton_charge'].value.data[3] > bad_charge).value


def test_remove_bad_pulses_does_not_modify_input():
    rng = np.random.default_rng(65501)
    bad_charge = sc.scalar(1.0e5, unit='pC')
    data = make_data_with_pulse_time_and_proton_charge(
        rng, 100, 300, bad_charge, bad_charge_indices=[0, 10, 100, 150, 200])
    original = data.copy()
    _ = filtering.remove_bad_pulses(data,
                                    proton_charge=data.attrs['proton_charge'].value,
                                    threshold_factor=0.9)
    assert sc.identical(data, original)


def test_remove_bad_pulses_without_bad_pulses():
    rng = np.random.default_rng(65501)
    bad_charge = sc.scalar(1.0e5, unit='pC')
    data = make_data_with_pulse_time_and_proton_charge(rng,
                                                       100,
                                                       300,
                                                       bad_charge,
                                                       bad_charge_indices=[])
    filtered = filtering.remove_bad_pulses(
        data, proton_charge=data.attrs['proton_charge'].value, threshold_factor=0.0)
    assert sc.identical(filtered, data)


def test_remove_bad_pulses_without_good_pulses():
    rng = np.random.default_rng(65501)
    bad_charge = sc.scalar(1.0e5, unit='pC')
    data = make_data_with_pulse_time_and_proton_charge(
        rng, 100, 300, bad_charge, bad_charge_indices=np.arange(300))
    filtered = filtering.remove_bad_pulses(
        data, proton_charge=data.attrs['proton_charge'].value, threshold_factor=10.0)
    empty = data.copy()
    empty.bins.constituents['begin'][...] = sc.index(0)
    empty.bins.constituents['end'][...] = sc.index(0)
    assert sc.identical(filtered, empty)


def test_remove_bad_pulses_contiguous_section():
    rng = np.random.default_rng(65501)
    bad_charge = sc.scalar(1.0e5, unit='pC')
    bad_indices = np.arange(100, 120)
    data = make_data_with_pulse_time_and_proton_charge(rng, 100, 300, bad_charge,
                                                       bad_indices)

    proton_charge = data.attrs['proton_charge'].value
    begin_removed = proton_charge.coords['pulse_time'][100]
    end_removed = proton_charge.coords['pulse_time'][120]
    data.bins.coords['should_be_removed'] = (
        data.bins.coords['pulse_time'] >
        begin_removed) & (data.bins.coords['pulse_time'] < end_removed)

    filtered = filtering.remove_bad_pulses(data,
                                           proton_charge=proton_charge,
                                           threshold_factor=0.9)

    assert not sc.any(filtered.bins.coords['should_be_removed']).value
    n_events_filtered = filtered.bins.size().sum().data
    expected_n_events_filtered = data.bins.size().sum(
    ).data - data.bins.coords['should_be_removed'].sum()
    assert sc.identical(n_events_filtered, expected_n_events_filtered)
