# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
"""
from typing import Dict, NewType, Optional, Tuple

import scipp as sc

from . import chopper_cascade, frames

Choppers = NewType('Choppers', Dict[str, chopper_cascade.Chopper])
FirstPulseTime = NewType('FirstPulseTime', sc.Variable)
FrameAtSample = NewType('FrameAtSample', chopper_cascade.Frame)
FrameBounds = NewType('FrameBounds', sc.Variable)
FramePeriod = NewType('FramePeriod', sc.Variable)
FrameWrappedTimeOffset = NewType('FrameWrappedTimeOffset', sc.Variable)
L1 = NewType('L1', sc.Variable)
L2 = NewType('L2', sc.Variable)
PulseOffset = NewType('PulseOffset', sc.Variable)
PulsePeriod = NewType('PulsePeriod', sc.Variable)
PulseStride = NewType('PulseStride', int)
PulseWrappedTimeOffset = NewType('PulseWrappedTimeOffset', sc.Variable)
RawData = NewType('RawData', sc.DataArray)
RawSubframeData = NewType('RawSubframeData', sc.DataArray)
SourceChopperName = NewType('SourceChopperName', str)
SourceChopper = NewType('SourceChopper', chopper_cascade.Chopper)
SourceTimeRange = NewType('SourceTimeRange', Tuple[sc.Variable, sc.Variable])
SourceWavelengthRange = NewType(
    'SourceWavelengthRange', Tuple[sc.Variable, sc.Variable]
)
SubframeBounds = NewType('SubframeBounds', sc.Variable)
TimeOfFlight = NewType('TimeOfFlight', sc.Variable)
TimeOffset = NewType('TimeOffset', sc.Variable)
TimeZero = NewType('TimeZero', sc.Variable)
TofData = NewType('TofData', sc.DataArray)
UnwrappedData = NewType('UnwrappedData', sc.DataArray)


def frame_period(
    pulse_period: PulsePeriod, pulse_stride: Optional[PulseStride]
) -> FramePeriod:
    if pulse_stride is None:
        return pulse_period
    return FramePeriod(pulse_period * pulse_stride)


def frame_at_sample(
    source_wavelength_range: SourceWavelengthRange,
    source_time_range: SourceTimeRange,
    choppers: Choppers,
    l1: L1,
) -> FrameAtSample:
    frames = chopper_cascade.FrameSequence.from_source_pulse(
        time_min=source_time_range[0],
        time_max=source_time_range[1],
        wavelength_min=source_wavelength_range[0],
        wavelength_max=source_wavelength_range[1],
    )
    frames.chop(choppers.values())
    return frames[-1].propagate_to(l1)


def frame_bounds(frame_at_sample: FrameAtSample, l2: L2) -> FrameBounds:
    bounds = frame_at_sample.bounds()
    return chopper_cascade.propagate_times(**bounds, distance=l2)


def subframe_bounds(frame_at_sample: FrameAtSample, l2: L2) -> SubframeBounds:
    """Used for WFM."""
    bounds = frame_at_sample.subbounds()
    return chopper_cascade.propagate_times(**bounds, distance=l2)


def frame_wrapped_time_offset(offset: PulseWrappedTimeOffset) -> FrameWrappedTimeOffset:
    """Without pulse-skipping, this is an identity function."""
    return FrameWrappedTimeOffset(offset)


def pulse_offset(
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride,
    event_time_zero: TimeZero,
    first_pulse_time: FirstPulseTime,
) -> PulseOffset:
    return frames.pulse_offset_from_time_zero(
        pulse_period=pulse_period,
        pulse_stride=pulse_stride,
        event_time_zero=event_time_zero,
        first_pulse_time=first_pulse_time,
    )


def frame_wrapped_time_offset_pulse_skipping(
    offset: PulseWrappedTimeOffset, pulse_offset: PulseOffset
) -> FrameWrappedTimeOffset:
    """"""
    return frames.update_time_offset_for_pulse_skipping(
        event_time_offset=offset,
        pulse_offset=pulse_offset,
    )


def pulse_wrapped_time_offset(da: RawData) -> PulseWrappedTimeOffset:
    """
    In NXevent_data, event_time_offset is the time offset since event_time_zero,
    which is the start of the most recent pulse. This is not the same as the start
    of the pulse that emitted the neutron, so event_time_offset is "wrapped" around
    at the pulse period.
    """
    return da.bins.coords['event_time_offset']


def time_zero(da: RawData) -> TimeZero:
    """
    In NXevent_data, event_time_zero is the start of the most recent pulse.
    """
    return da.bins.coords['event_time_zero']


def time_offset(
    wrapped_time_offset: FrameWrappedTimeOffset,
    frame_bounds: FrameBounds,
    frame_period: FramePeriod,
) -> TimeOffset:
    """
    Time offset from the start of the frame emitting the neutron.

    This is not identical to the time-of-flight, since the time-of-flight is measured,
    e.g., from the center of the pulse, to the center of a pulse-shaping chopper slit.

    Note that this takes the frame-period, which can be a multiple of a pulse-period.
    The input time stamps must be relative the the start of the frame, i.e., in
    pulse-skipping mode a pulse offset must have been applied before calling this
    function.

    Parameters
    ----------
    wrapped_time_offset :
        Time offset from the time-zero as recorded by the data acquisition system.
    time_offset_min :
        Minimum arrival time offset of neutrons that can pass through the chopper
        cascade. Typically pixel-dependent.
    frame_period :
        Time between the start of two consecutive frames, i.e., the period of the
        time-zero used by the data acquisition system.
    """
    time_offset_min = frame_bounds['bound', 0]
    wrapped_time_min = time_offset_min % frame_period
    delta = frame_period if wrapped_time_offset < wrapped_time_min else 0
    offset_frames = time_offset_min - wrapped_time_min + delta
    return offset_frames + wrapped_time_offset


def source_chopper(
    choppers: Choppers, source_chopper_name: SourceChopperName
) -> SourceChopper:
    return choppers[source_chopper_name]


def time_of_flight(
    time_offset: TimeOffset,
    source_chopper: SourceChopper,
) -> TimeOfFlight:
    """
    Time-of-flight of neutrons passing through a chopper cascade.

    A chopper is used to define (1) the "source" location and (2) the time-of-flight
    time origin. The time-of-flight is then the time difference between the time of
    arrival of the neutron at the detector, and the time of arrival of the neutron at
    the chopper. L1 needs to be redefined to be the distance between the chopper and
    the sample.

    If there is no pulse-shaping chopper, then the source-pulse begin and end time
    should be set as the source_time_open and source_time_close, respectively.

    For WFM, the source_time_open and source_time_close will be different for each
    subframe. In this case, all input parameters should be given as variables with
    subframe as dimension.

    Parameters
    ----------
    time_offset :
        Time offset from the start of the frame emitting the neutron.
    source_chopper :
        Chopper defining the source location and time-of-flight time origin.
    """
    source_time_open = source_chopper.time_open
    source_time_close = source_chopper.time_close
    # TODO Need to handle choppers with multiple openings, where we need to select one
    time_zero = 0.5 * (source_time_open + source_time_close)
    return time_offset - time_zero


def time_of_flight_wfm(
    time_offset: TimeOffset,
    source_chopper: SourceChopper,
    subframe_bounds: SubframeBounds,
) -> TimeOfFlight:
    # time_zero will have multiple subframes
    times = subframe_bounds.flatten(dims=['subframe', 'bound'], to='subframe')
    neg_shift = sc.zeros(dims=['subframe'], shape=[len(times) - 1], unit='s')
    neg_shift[::2] -= 0.5 * (source_chopper.time_open + source_chopper.time_close)
    lut = sc.DataArray(neg_shift, coords={'subframe': times})
    # Will raise if subframes overlap, since coord for lookup table must be sorted
    out = sc.lookup(lut, dim='subframe')[time_offset]
    out += time_offset
    return out


def tof_data(da: RawData, tof: TimeOfFlight) -> TofData:
    da = da.copy(deep=False)  # todo copy bins
    da.bins.coords['tof'] = tof
    return TofData(da)


_common_providers = [
    frame_at_sample,
    frame_bounds,
    frame_period,
    pulse_wrapped_time_offset,
    source_chopper,
    time_offset,
    tof_data,
]

_non_skipping = [frame_wrapped_time_offset]
_skipping = [
    frame_wrapped_time_offset_pulse_skipping,
    pulse_offset,
    time_zero,
]

_wfm = [subframe_bounds, time_of_flight_wfm]
_non_wfm = [time_of_flight]


def providers(pulse_skipping: bool = False, wfm: bool = False):
    """
    Return the list of providers for unwrapping frames.

    Parameters
    ----------
    pulse_skipping :
        If True, the pulse-skipping mode is assumed.
    wfm :
        If True, the data is assumed to be from a WFM instrument.
    """
    skipping = _skipping if pulse_skipping else _non_skipping
    wfm = _wfm if wfm else _non_wfm
    return _common_providers + skipping + wfm
