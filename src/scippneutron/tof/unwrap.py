# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
"""
import math
from typing import Mapping, NewType, Optional, Tuple

import scipp as sc

from . import chopper_cascade, frames

Choppers = NewType('Choppers', Mapping[str, chopper_cascade.Chopper])
"""
Choppers used to define the frame parameters.
"""

FirstPulseTime = NewType('FirstPulseTime', sc.Variable)
"""
In pulse-skipping mode this defines the (or a) first pulse.

This identifies which pulses are used and which are skipped.
"""

FrameAtSample = NewType('FrameAtSample', chopper_cascade.Frame)
"""
Result of passing the source pulse through the chopper cascade.

After propagating the frame to the detector distances this will then be used to
compute the frame bounds.
"""

FrameAtDetector = NewType('FrameAtDetector', chopper_cascade.Frame)
"""
Result of passing the source pulse through the chopper cascade to the detector.

The detector may be a monitor or a detector after scattering off the sample. The frame
bounds are then computed from this.
"""

FrameBounds = NewType('FrameBounds', sc.Variable)
"""
The computed frame boundaries, used to unwrap the raw timestamps.
"""

FramePeriod = NewType('FramePeriod', sc.Variable)
"""
The period of a frame, a (small) integer multiple of the source period.
"""

FrameWrappedTimeOffset = NewType('FrameWrappedTimeOffset', sc.Variable)
"""
Time offsets wrapped around at the frame period.

In normal mode, this is identical to the NXevent_data/event_time_offset recorded by
the data acquisition system, i.e., the same as :py:class:`PulseWrappedTimeOffset`.

In pulse-skipping mode, this is the time offset since the start of the frame.
"""

Ltotal = NewType('Ltotal', sc.Variable)
"""
Total distance between the source and the detector(s).

This is used to propagate the frame to the detector position. This will then yield
detector-dependent frame bounds. This is typically the sum of L1 and L2, except for
monitors.
"""

PulseOffset = NewType('PulseOffset', sc.Variable)
"""
Offset from the start of the frame in pulse-skipping mode, multiple of pulse period.
"""

PulsePeriod = NewType('PulsePeriod', sc.Variable)
"""
Period of the source pulses, i.e., time between consecutive pulse starts.
"""

PulseStride = NewType('PulseStride', int)
"""
Stride of used pulses. Usually 1, but may be a small integer when pulse-skipping.
"""

PulseWrappedTimeOffset = NewType('PulseWrappedTimeOffset', sc.Variable)
"""
Time offsets wrapped at the pulse period, typically NXevent_data/event_time_offset.
"""

RawData = NewType('RawData', sc.DataArray)
"""
Raw detector data loaded from a NeXus file, e.g., NXdetector containing NXevent_data.
"""

SourceChopperName = NewType('SourceChopperName', str)
"""
Name of the chopper defining the source location and time-of-flight time origin.
"""

SourceChopper = NewType('SourceChopper', chopper_cascade.Chopper)
"""
Chopper defining the source location and time-of-flight time origin.

If there is no pulse-shaping chopper, a fake chopper with time_open and time_close
set to the source pulse begin and end time should be used.
"""

SourceTimeRange = NewType('SourceTimeRange', Tuple[sc.Variable, sc.Variable])
"""
Time range of the source pulse, used for computing frame bounds.
"""

SourceWavelengthRange = NewType(
    'SourceWavelengthRange', Tuple[sc.Variable, sc.Variable]
)
"""
Wavelength range of the source pulse, used for computing frame bounds.
"""

SubframeBounds = NewType('SubframeBounds', sc.Variable)
"""
The computed subframe boundaries, used to offset the raw timestamps for WFM.
"""

TimeZero = NewType('TimeZero', sc.Variable)
"""
Time of the start of the most recent pulse, typically NXevent_data/event_time_zero.
"""

TofData = NewType('TofData', sc.DataArray)
"""
Detector data with resulting 'tof' coordinate.
"""


OffsetFromTimeOfFlight = NewType('OffsetFromTimeOfFlight', sc.Variable)
OffsetFromWrapped = NewType('OffsetFromWrapped', sc.Variable)


def frame_period(
    pulse_period: PulsePeriod, pulse_stride: Optional[PulseStride]
) -> FramePeriod:
    if pulse_stride is None:
        return pulse_period
    return FramePeriod(pulse_period * pulse_stride)


def frame_at_detector(
    source_wavelength_range: SourceWavelengthRange,
    source_time_range: SourceTimeRange,
    choppers: Choppers,
    ltotal: Ltotal,
) -> FrameAtDetector:
    frames = chopper_cascade.FrameSequence.from_source_pulse(
        time_min=source_time_range[0],
        time_max=source_time_range[1],
        wavelength_min=source_wavelength_range[0],
        wavelength_max=source_wavelength_range[1],
    )
    frames = frames.chop(choppers.values())
    return FrameAtDetector(frames[ltotal])


def frame_bounds(frame: FrameAtDetector) -> FrameBounds:
    return FrameBounds(frame.bounds())


def subframe_bounds(frame: FrameAtDetector) -> SubframeBounds:
    """Used for WFM."""
    return SubframeBounds(frame.subbounds())


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
    if da.bins is None:
        # Canonical name in NXmonitor
        return PulseWrappedTimeOffset(da.coords['time_of_flight'])
    return da.bins.coords['event_time_offset']


def time_zero(da: RawData) -> TimeZero:
    """
    In NXevent_data, event_time_zero is the start of the most recent pulse.
    """
    # This is not available for histogram-mode monitors. We probably need to look it
    # up elsewhere in the file, e.g., NXsource. Should we have a separate provider for
    # monitors?
    return da.bins.coords['event_time_zero']


def offset_from_wrapped(
    wrapped_time_offset: FrameWrappedTimeOffset,
    frame_bounds: FrameBounds,
    frame_period: FramePeriod,
) -> OffsetFromWrapped:
    """
    Offset of the input time offsets from the start of the frame emitting the neutron.

    In other words, this is the part that is "lost" by the conceptual wrapping of the
    time offsets at the frame period. This is the offset that needs to be added to the
    NXevent_data/event_time_offset to obtain the offset from the start of the emitting
    frame.

    This is not identical to the offset to time-of-flight, since the time-of-flight is
    measured, e.g., from the center of the pulse, to the center of a pulse-shaping
    chopper slit.

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
    begin = sc.zeros_like(wrapped_time_min)
    end = sc.ones_like(wrapped_time_min)
    dim = 'section'
    time = sc.concat([begin, wrapped_time_min, end], dim).transpose().copy()
    offset = sc.DataArray(
        time_offset_min
        - wrapped_time_min
        + sc.concat([frame_period, sc.zeros_like(frame_period)], dim),
        coords={dim: time},
    )
    return OffsetFromWrapped(sc.lookup(offset, dim=dim)[wrapped_time_offset])


def source_chopper(
    choppers: Choppers,
    source_time_range: SourceTimeRange,
    source_chopper_name: Optional[SourceChopperName],
) -> SourceChopper:
    """
    Return the chopper defining the source location and time-of-flight time origin.

    If there is no pulse-shaping chopper, then the source-pulse begin and end time
    are used to define a fake chopper.
    """
    if source_chopper_name is not None:
        return choppers[source_chopper_name]
    return chopper_cascade.Chopper(
        distance=sc.scalar(0.0, 'm'),
        time_open=source_time_range[0],
        time_close=source_time_range[1],
    )


def offset_to_time_of_flight(
    time_offset: OffsetFromWrapped,
    source_chopper: SourceChopper,
) -> OffsetFromTimeOfFlight:
    """
    Offset from the time-of-flight of neutrons passing through a chopper cascade.

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
    return OffsetFromTimeOfFlight(time_offset - time_zero)


def offset_to_time_of_flight_wfm(
    time_offset: OffsetFromWrapped,
    source_chopper: SourceChopper,
    subframe_bounds: SubframeBounds,
) -> OffsetFromTimeOfFlight:
    times = subframe_bounds.flatten(dims=['subframe', 'bound'], to='subframe')
    neg_shift = sc.zeros(dims=['subframe'], shape=[len(times) + 2], unit='s')
    neg_shift[1::2] -= 0.5 * (source_chopper.time_open + source_chopper.time_close)
    # Set offsets before, between, and after subframes to NaN
    neg_shift[::2] = sc.scalar(math.nan, unit='s')
    lut = sc.DataArray(neg_shift, coords={'subframe': times})
    # Will raise if subframes overlap, since coord for lookup table must be sorted
    out = sc.lookup(lut, dim='subframe')[time_offset]
    out += time_offset
    return OffsetFromTimeOfFlight(out)


def tof_data(da: RawData, offset: OffsetFromTimeOfFlight) -> TofData:
    """
    Return the input data with 'tof' and 'time_zero' coordinates.

    The 'tof' coordinate is the time-of-flight of the neutron, i.e., the time
    difference between the time of arrival of the neutron at the detector, and the
    time of the neutron at the source or source-chopper. The 'time_zero' coordinate is
    the time of the neutron at the source or source-chopper.

    It is critical to compute the 'time_zero' coordinate precisely by applying the
    reverse of the offset to time-of-flight, since this may be used to compute a
    precise time at the sample position. This is important for sample environments
    with highly time-dependent properties, since precise event-filtering based on
    sample environment data may be required.
    """
    da = da.copy(deep=False)
    if da.bins is not None:
        da.data = sc.bins(**da.bins.constituents)
        da.bins.coords['tof'] = da.bins.coords['event_time_offset'] + offset
        da.bins.coords['time_zero'] = da.bins.coords['event_time_zero'] - offset
    else:
        # 'time_of_flight' is the name in, e.g., Nxmonitor
        da.coords['tof'] = da.coords['time_of_flight'] + offset
        # Generally the coord is now not ordered, might want to cut and concat, but it
        # is unclear what to do with the split bin in the middle.
    return TofData(da)


_common_providers = [
    frame_at_detector,
    frame_bounds,
    frame_period,
    pulse_wrapped_time_offset,
    source_chopper,
    offset_from_wrapped,
    tof_data,
]

_non_skipping = [frame_wrapped_time_offset]
_skipping = [
    frame_wrapped_time_offset_pulse_skipping,
    pulse_offset,
    time_zero,
]

_wfm = [subframe_bounds, offset_to_time_of_flight_wfm]
_non_wfm = [offset_to_time_of_flight]


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
