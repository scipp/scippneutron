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


def test_make_frames_time_offset_pivot_and_min_define_frames():
    # events [before, after, after, before] pivot point
    time_offset = sc.array(dims=['event'], values=[5.0, 70.0, 21.0, 6.0], unit='ms')
    da = array(frame_length=71.0 * sc.Unit('ms'),
               npixel=1,
               nevent=4,
               time_offset=time_offset)
    pivot = sc.to_unit(10.0 * sc.Unit('ms'), da.bins.coords['time_offset'].bins.unit)
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


def tof_array(*,
              npixel=3,
              nevent=1000,
              frame_length=71.0 * sc.Unit('ms'),
              tof_min=234.0 * sc.Unit('ms')):
    tof = sc.array(dims=['event'], values=np.random.rand(nevent)) * sc.to_unit(
        frame_length, tof_min.unit) + tof_min
    pixel = sc.arange(dim='event', start=0, stop=nevent) % npixel
    events = sc.DataArray(sc.ones(sizes=tof.sizes), coords={'tof': tof, 'pixel': pixel})
    pixel = sc.arange(dim='pixel', start=0, stop=npixel)
    da = sc.bin(events, groups=[pixel])
    da.coords['L1'] = sc.scalar(value=160.0, unit='m')
    da.coords['L2'] = sc.array(dims=['pixel'], values=np.arange(npixel), unit='m')
    return da


def tof_to_time_offset(tof, *, frame_length, frame_offset):
    unit = tof.bins.unit
    return (sc.to_unit(frame_offset, unit) + tof) % sc.to_unit(frame_length, unit)


@pytest.mark.parametrize(
    "tof_min", [234.0 * sc.Unit('ms'), 37000.0 * sc.Unit('us'), 337.0 * sc.Unit('ms')])
@pytest.mark.parametrize(
    "frame_offset", [0.0 * sc.Unit('ms'), 11.0 * sc.Unit('ms'), 9999.0 * sc.Unit('us')])
def test_make_frames_reproduces_true_pulses(tof_min, frame_offset):
    from scippneutron.core.conversions import _wavelength_from_tof
    frame_length = 71.0 * sc.Unit('ms')
    # Setup data with known 'tof' coord, which will serve as a reference
    da = tof_array(frame_length=frame_length, tof_min=tof_min)
    reference = da.bins.coords['tof'].copy()
    # Compute backwards to "raw" input with 'time_offset'. 'tof' coord is removed
    da.bins.coords['time_offset'] = tof_to_time_offset(da.bins.coords.pop('tof'),
                                                       frame_length=frame_length,
                                                       frame_offset=frame_offset)
    lambda_min = _wavelength_from_tof(tof_min, Ltotal=da.coords['L1'] + da.coords['L2'])

    da = frames.make_frames(da,
                            frame_length=frame_length,
                            frame_offset=frame_offset,
                            lambda_min=lambda_min)

    # Should reproduce reference 'tof' within rounding errors
    assert sc.allclose(da.bins.coords['tof'],
                       reference,
                       atol=sc.scalar(1e-12, unit=reference.bins.unit),
                       rtol=sc.scalar(1e-12))


def time_zero_to_detection_frame_index(*, frame_offset, tof_min, frame_length,
                                       frame_stride, time_zero):
    """
    Return 0-based source frame index of detection frame.

    The source frames containing the time marked by tof_min receive index 0.
    The frame after that index 1, and so on, until frame_stride-1.
    """
    return ((time_zero - tof_min) // (frame_length)).value % frame_stride


class Test_time_zero_to_detection_frame_index:
    tof_min = [234.0 * sc.Unit('ms'), 37.0 * sc.Unit('ms'), 337.0 * sc.Unit('ms')]
    frame_offset = [0.0 * sc.Unit('ms'), 11.0 * sc.Unit('ms'), 9.999 * sc.Unit('ms')]
    frame_length = [71.0 * sc.Unit('ms')]
    time_zero = [
        t0 * sc.Unit('ms') for t0 in [0.0, 11.0, 70.0, 71.0, 71.1, 300.0, 345.6]
    ]

    @pytest.mark.parametrize("tof_min", tof_min)
    @pytest.mark.parametrize("frame_offset", frame_offset)
    @pytest.mark.parametrize("frame_length", frame_length)
    @pytest.mark.parametrize("time_zero", time_zero)
    def test_frame_stride_1_yields_index_0(self, frame_length, frame_offset, tof_min,
                                           time_zero):
        assert time_zero_to_detection_frame_index(frame_length=frame_length,
                                                  frame_stride=1,
                                                  frame_offset=frame_offset,
                                                  tof_min=tof_min,
                                                  time_zero=time_zero) == 0

    def test_frame_stride_2_with_zero_params_yields_source_frame_index(self):
        frame_length = 71.0 * sc.Unit('ms')
        frame_offset = 0.0 * sc.Unit('ms')
        tof_min = 0.0 * sc.Unit('ms')
        params = dict(frame_length=frame_length,
                      frame_stride=2,
                      frame_offset=frame_offset,
                      tof_min=tof_min)
        assert time_zero_to_detection_frame_index(**params,
                                                  time_zero=1.0 * sc.Unit('ms')) == 0
        assert time_zero_to_detection_frame_index(**params,
                                                  time_zero=71.1 * sc.Unit('ms')) == 1
        assert time_zero_to_detection_frame_index(**params,
                                                  time_zero=150.0 * sc.Unit('ms')) == 0
        assert time_zero_to_detection_frame_index(**params,
                                                  time_zero=221.0 * sc.Unit('ms')) == 1

    def test_frame_stride_2_with_tof_min_yields_detection_frame_index(self):
        frame_length = 100.0 * sc.Unit('ms')
        frame_offset = 0.0 * sc.Unit('ms')
        tof_min = 217 * sc.Unit('ms')
        params = dict(frame_length=frame_length,
                      frame_stride=2,
                      frame_offset=frame_offset,
                      tof_min=tof_min)
        assert time_zero_to_detection_frame_index(**params,
                                                  time_zero=218.0 * sc.Unit('ms')) == 0
        assert time_zero_to_detection_frame_index(**params,
                                                  time_zero=316.0 * sc.Unit('ms')) == 0
        assert time_zero_to_detection_frame_index(**params,
                                                  time_zero=318.0 * sc.Unit('ms')) == 1
        assert time_zero_to_detection_frame_index(**params,
                                                  time_zero=416.0 * sc.Unit('ms')) == 1
