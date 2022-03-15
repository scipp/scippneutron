# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""
Event filtering for time-of-flight source.
"""
from contextlib import contextmanager
import scipp as sc


@contextmanager
def _temporary_bin_coord(data, name, coord):
    data.bins.coords[name] = coord
    yield
    del data.bins.coords[name]


def _with_pulse_time_edges(da):
    pulse_time = da.coords[da.dim]
    one = sc.scalar(1, dtype='int64', unit=pulse_time.unit)
    lo = pulse_time[0] - one
    hi = pulse_time[-1] + one
    mid = sc.midpoints(pulse_time)
    da.coords[da.dim] = sc.concat([lo, mid, hi], da.dim)
    return da


def remove_bad_pulses(data, *, proton_charge, threshold_factor):
    """
    assumes that there are bad pulses
    """
    min_charge = proton_charge.data.mean() * threshold_factor
    good_pulse = _with_pulse_time_edges(proton_charge >= min_charge)
    with _temporary_bin_coord(
            data, 'good_pulse',
            sc.lookup(good_pulse, good_pulse.dim)[data.bins.coords[good_pulse.dim]]):
        filtered = sc.bin(data, groups=[sc.array(dims=['good_pulse'], values=[True])])
    filtered = filtered.squeeze('good_pulse').copy(deep=False)
    del filtered.attrs['good_pulse']
    return filtered
