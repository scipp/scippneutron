# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import numpy as np
import pytest
import scipp as sc
from hypothesis import given
from hypothesis import strategies as st

from scippneutron.tof import frames, unwrap_frames


def make_array(*, npixel=3, nevent=1000, pulse_period=None, time_offset=None):
    pulse_period = (
        71.0e3 * sc.Unit('us') if pulse_period is None else pulse_period.to(unit='us')
    )
    if time_offset is None:
        time_offset = sc.array(
            dims=['event'],
            values=np.random.rand(nevent) * pulse_period.value,
            unit='us',
        )
    start = sc.datetime('now', unit='ns')
    npulse = 1234
    time_zero = start + (pulse_period * sc.linspace('event', 0, npulse, num=nevent)).to(
        unit='ns', dtype='int64'
    )
    pixel = sc.arange(dim='event', start=0, stop=nevent) % npixel
    events = sc.DataArray(
        sc.ones(sizes=time_offset.sizes),
        coords={
            'event_time_offset': time_offset,
            'event_time_zero': time_zero,
            'pixel': pixel,
        },
    )
    da = events.group(sc.arange(dim='pixel', start=0, stop=npixel, dtype=pixel.dtype))
    da.coords['L1'] = sc.scalar(value=160.0, unit='m')
    da.coords['L2'] = sc.array(dims=['pixel'], values=np.arange(npixel), unit='m')
    return da


def test_unwrap_frames_given_tof_bins_meta_data_raises_ValueError():
    da = make_array()
    da.coords['tof'] = sc.scalar(1.0, unit='ms')
    with pytest.raises(ValueError):
        unwrap_frames(
            da,
            scatter=True,
            pulse_period=71.0 * sc.Unit('ms'),
            frame_offset=30.1 * sc.Unit('ms'),
            lambda_min=2.5 * sc.Unit('Angstrom'),
        )


def test_unwrap_frames_given_tof_event_meta_data_raises_ValueError():
    da = make_array()
    da.bins.coords['tof'] = da.bins.coords['event_time_offset']
    with pytest.raises(ValueError):
        unwrap_frames(
            da,
            scatter=True,
            pulse_period=71.0 * sc.Unit('ms'),
            frame_offset=30.1 * sc.Unit('ms'),
            lambda_min=2.5 * sc.Unit('Angstrom'),
        )


def test_unwrap_frames_no_shift_and_infinite_energy_yields_tof_equal_time_offset():
    da = make_array(pulse_period=71.0 * sc.Unit('ms'))
    da = unwrap_frames(
        da,
        scatter=True,
        pulse_period=71.0 * sc.Unit('ms'),
        frame_offset=0.0 * sc.Unit('ms'),
        lambda_min=0.0 * sc.Unit('Angstrom'),
    )
    assert sc.identical(da.bins.coords['tof'], da.bins.coords['event_time_offset'])


def test_unwrap_frames_no_shift_and_no_events_below_lambda_min_yields_tof_equal_time_offset():  # noqa #501
    da = make_array(pulse_period=71.0 * sc.Unit('ms'))
    da.bins.coords['event_time_offset'] += sc.to_unit(
        10.0 * sc.Unit('ms'), da.bins.coords['event_time_offset'].bins.unit
    )
    da = unwrap_frames(
        da,
        scatter=True,
        pulse_period=81.0 * sc.Unit('ms'),
        frame_offset=0.0 * sc.Unit('ms'),
        lambda_min=0.2 * sc.Unit('Angstrom'),
    )
    assert sc.identical(da.bins.coords['tof'], da.bins.coords['event_time_offset'])


def test_unwrap_frames_time_offset_pivot_and_min_define_frames():
    # events [before, after, after, before] pivot point
    time_offset = sc.array(dims=['event'], values=[5.0, 70.0, 21.0, 6.0], unit='ms')
    da = make_array(
        pulse_period=71.0 * sc.Unit('ms'), npixel=1, nevent=4, time_offset=time_offset
    )
    pivot = sc.to_unit(
        10.0 * sc.Unit('ms'), da.bins.coords['event_time_offset'].bins.unit
    )
    da.coords['time_offset_pivot'] = pivot
    da.coords['tof_min'] = 200.0 * sc.Unit('ms')
    da.coords['pulse_period'] = 71.0 * sc.Unit('ms')
    da = da.transform_coords('tof', graph=frames.to_tof())
    tof = da.bins.coords['tof'].values[0]
    tof_values = [
        71.0 - 10.0 + 5.0 + 200.0,
        60.0 + 200.0,
        11.0 + 200.0,
        71.0 - 10.0 + 6.0 + 200.0,
    ]
    assert sc.identical(tof, sc.array(dims=['event'], unit='ms', values=tof_values))


def tof_array(*, npixel=3, nevent=1000, pulse_period=None, tof_min=None):
    pulse_period = 71.0 * sc.Unit('ms') if pulse_period is None else pulse_period
    tof_min = 234.0 * sc.Unit('ms') if tof_min is None else tof_min
    tof = (
        sc.array(dims=['event'], values=np.random.rand(nevent))
        * pulse_period.to(unit=tof_min.unit)
        + tof_min
    )
    pixel = sc.arange(dim='event', start=0, stop=nevent) % npixel
    events = sc.DataArray(sc.ones(sizes=tof.sizes), coords={'tof': tof, 'pixel': pixel})
    da = events.group(sc.arange(dim='pixel', start=0, stop=npixel, dtype=pixel.dtype))
    da.coords['L1'] = sc.scalar(value=160.0, unit='m')
    da.coords['L2'] = sc.array(dims=['pixel'], values=np.arange(npixel), unit='m')
    return da


def tof_to_time_offset(tof, *, pulse_period, frame_offset):
    unit = tof.bins.unit
    return (frame_offset.to(unit=unit) + tof) % pulse_period.to(unit=unit)


@pytest.mark.parametrize(
    "tof_min", [234.0 * sc.Unit('ms'), 37000.0 * sc.Unit('us'), 337.0 * sc.Unit('ms')]
)
@pytest.mark.parametrize(
    "frame_offset", [0.0 * sc.Unit('ms'), 11.0 * sc.Unit('ms'), 9999.0 * sc.Unit('us')]
)
def test_unwrap_frames_reproduces_true_pulses(tof_min, frame_offset):
    from scippneutron.conversion.tof import wavelength_from_tof

    pulse_period = 71.0 * sc.Unit('ms')
    # Setup data with known 'tof' coord, which will serve as a reference
    da = tof_array(pulse_period=pulse_period, tof_min=tof_min)
    reference = da.bins.coords['tof'].copy()
    # Compute backwards to "raw" input with 'event_time_offset'. 'tof' coord is removed
    da.bins.coords['event_time_offset'] = tof_to_time_offset(
        da.bins.coords.pop('tof'), pulse_period=pulse_period, frame_offset=frame_offset
    )
    lambda_min = wavelength_from_tof(
        tof=tof_min, Ltotal=da.coords['L1'] + da.coords['L2']
    )

    da = unwrap_frames(
        da,
        scatter=True,
        pulse_period=pulse_period,
        frame_offset=frame_offset,
        lambda_min=lambda_min,
    )

    # Should reproduce reference 'tof' within rounding errors
    assert sc.allclose(
        da.bins.coords['tof'],
        reference,
        atol=sc.scalar(1e-12, unit=reference.bins.unit),
        rtol=sc.scalar(1e-12),
    )


def fake_pulse_skipping_data(
    *,
    npixel=3,
    nevent,
    nframe,
    pulse_period,
    pulse_stride: int = 2,
    frame_offset,
    tof_min,
):
    from scippneutron.conversion.tof import wavelength_from_tof

    rng = np.random.default_rng(1234)

    # Setup data with known 'tof' coord, which will serve as a reference
    frame_period = (pulse_period * pulse_stride).to(unit=tof_min.unit)
    tof = sc.array(dims=['event'], values=rng.random(nevent)) * frame_period + tof_min
    start = sc.datetime('now', unit='ns')
    time_zero = start + (
        frame_period * sc.linspace('event', 0, nframe, num=nevent, dtype='int64')
    ).to(unit='ns', dtype='int64')

    pixel = sc.arange(dim='event', start=0, stop=nevent) % npixel
    events = sc.DataArray(sc.ones(sizes=tof.sizes), coords={'tof': tof, 'pixel': pixel})
    events.coords['time_zero'] = time_zero
    da = events.group(sc.arange(dim='pixel', start=0, stop=npixel, dtype=pixel.dtype))
    da.coords['L1'] = sc.scalar(value=160.0, unit='m')
    da.coords['L2'] = sc.array(dims=['pixel'], values=np.arange(npixel), unit='m')
    reference = da.copy()

    # Compute backwards to "raw" input with 'event_time_offset'. 'tof' coord is removed
    time_offset = tof_to_time_offset(
        da.bins.coords.pop('tof'), pulse_period=frame_period, frame_offset=frame_offset
    )
    event_time_offset = time_offset % pulse_period.to(unit=time_offset.bins.unit)
    da.bins.coords['event_time_offset'] = event_time_offset
    da.bins.coords['event_time_zero'] = da.bins.coords.pop('time_zero') + (
        time_offset - event_time_offset
    ).to(unit='ns', dtype='int64')
    lambda_min = wavelength_from_tof(
        tof=tof_min, Ltotal=da.coords['L1'] + da.coords['L2']
    )
    return reference, da, start, lambda_min


@given(
    nevent=st.integers(min_value=0, max_value=10000),
    nframe=st.integers(min_value=1, max_value=1000000),
    frame_offset=st.floats(min_value=0.0, max_value=10000.0),
    tof_min=st.floats(min_value=0.1, max_value=100000.0),
)
@pytest.mark.parametrize("pulse_stride", [1, 2, 3, 4, 5])
def test_unwrap_frames_with_pulse_stride_reproduces_true_pulses(
    nevent, nframe, tof_min, frame_offset, pulse_stride
):
    frame_offset = frame_offset * sc.Unit('ms')
    tof_min = tof_min * sc.Unit('us')
    pulse_period = 71.0 * sc.Unit('ms')
    # Setup data with known 'tof' coord, which will serve as a reference
    reference, da, first_pulse_time, lambda_min = fake_pulse_skipping_data(
        pulse_period=pulse_period,
        pulse_stride=pulse_stride,
        frame_offset=frame_offset,
        nevent=nevent,
        nframe=nframe,
        tof_min=tof_min,
    )

    da = unwrap_frames(
        da,
        scatter=True,
        pulse_period=pulse_period,
        pulse_stride=pulse_stride,
        first_pulse_time=first_pulse_time,
        frame_offset=frame_offset,
        lambda_min=lambda_min,
    )

    # Should reproduce reference 'tof' within rounding errors
    expected = reference.bins.coords['tof']
    assert sc.allclose(
        da.bins.coords['tof'],
        expected,
        atol=sc.scalar(1e-9, unit=expected.bins.unit),
        rtol=sc.scalar(1e-9),
    )
