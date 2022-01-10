# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
import numpy as np
import scipp as sc
from scippneutron.tof import frames


def array(*,
          npixel=3,
          nevent=1000,
          frame_length=71.0 * sc.Unit('ms'),
          time_offset=None):
    frame_length = sc.to_unit(frame_length, unit='us')
    if time_offset is None:
        time_offset = sc.array(dims=['event'],
                               values=np.random.rand(nevent) * frame_length.value,
                               unit='us')
    pixel = sc.arange(dim='event', start=0, stop=nevent) % npixel
    events = sc.DataArray(sc.ones(sizes=time_offset.sizes),
                          coords={
                              'time_offset': time_offset,
                              'pixel': pixel
                          })
    pixel = sc.arange(dim='pixel', start=0, stop=npixel)
    da = sc.bin(events, groups=[pixel])
    da.coords['L1'] = sc.scalar(value=160.0, unit='m')
    da.coords['L2'] = sc.array(dims=['pixel'], values=np.arange(npixel), unit='m')
    return da


def test_make_frames_given_tof_bins_meta_data_raises_ValueError():
    da = array()
    da.coords['tof'] = sc.scalar(1.0, unit='ms')
    with pytest.raises(ValueError):
        frames.make_frames(da,
                           frame_length=71.0 * sc.Unit('ms'),
                           frame_offset=30.1 * sc.Unit('ms'),
                           lambda_min=2.5 * sc.Unit('Angstrom'))


def test_make_frames_given_tof_event_meta_data_raises_ValueError():
    da = array()
    da.bins.coords['tof'] = da.bins.coords['time_offset']
    with pytest.raises(ValueError):
        frames.make_frames(da,
                           frame_length=71.0 * sc.Unit('ms'),
                           frame_offset=30.1 * sc.Unit('ms'),
                           lambda_min=2.5 * sc.Unit('Angstrom'))


def test_make_frames_no_shift_and_infinite_energy_yields_tof_equal_time_offset():
    da = array(frame_length=71.0 * sc.Unit('ms'))
    da = frames.make_frames(da,
                            frame_length=71.0 * sc.Unit('ms'),
                            frame_offset=0.0 * sc.Unit('ms'),
                            lambda_min=0.0 * sc.Unit('Angstrom'))
    assert sc.identical(da.bins.coords['tof'], da.bins.attrs['time_offset'])


def test_make_frames_no_shift_and_no_events_below_lambda_min_yields_tof_equal_time_offset(
):
    da = array(frame_length=71.0 * sc.Unit('ms'))
    da.bins.coords['time_offset'] += sc.to_unit(10.0 * sc.Unit('ms'),
                                                da.bins.coords['time_offset'].bins.unit)
    da = frames.make_frames(da,
                            frame_length=81.0 * sc.Unit('ms'),
                            frame_offset=0.0 * sc.Unit('ms'),
                            lambda_min=0.2 * sc.Unit('Angstrom'))
    assert sc.identical(da.bins.coords['tof'], da.bins.attrs['time_offset'])


def test_make_frame_time_offset_pivot_and_min_define_frames():
    pivot = sc.to_unit(10.0 * sc.Unit('ms'), da.bins.coords['time_offset'].bins.unit)
    # events [before, after, after, before] pivot point
    time_offset = sc.array(dims=['event'], values=[5.0, 70.0, 21.0, 6.0], unit='ms')
    da = array(frame_length=71.0 * sc.Unit('ms'),
               npixel=1,
               nevent=4,
               time_offset=time_offset)
    da.coords['time_offset_pivot'] = pivot
    da.coords['tof_min'] = 200.0 * sc.Unit('ms')
    da = frames.make_frames(da, frame_length=71.0 * sc.Unit('ms'))
    tof = da.bins.coords['tof'].values[0]
    tof_values = [
        71.0 - 10.0 + 5.0 + 200.0,
        60.0 + 200.0,
        11.0 + 200.0,
        71.0 - 10.0 + 6.0 + 200.0,
    ]
    assert sc.identical(tof, sc.array(dims=['event'], unit='ms', values=tof_values))
