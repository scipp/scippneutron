# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
This module provides functionality for unwrapping raw frames of neutron time-of-flight
data.

The module handles standard unwrapping, unwrapping in pulse-skipping mode, and
unwrapping for WFM instruments, as well as combinations of the latter two. The
functions defined here are meant to be used as providers for a Sciline pipeline. See
https://scipp.github.io/sciline/ on how to use Sciline.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NewType

import numpy as np
import scipp as sc

from .._utils import elem_unit
from . import chopper_cascade

ChopperCascadeFrames = NewType('ChopperCascadeFrames', chopper_cascade.FrameSequence)
"""
Result of passing the source pulse through the chopper cascade.
"""


Choppers = NewType('Choppers', Mapping[str, chopper_cascade.Chopper])
"""
Choppers used to define the frame parameters.
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

CleanFrameAtDetector = NewType('CleanFrameAtDetector', chopper_cascade.Frame)
"""
Version of the frame at the detector with subframes that do not overlap.
"""

FrameBounds = NewType('FrameBounds', sc.DataGroup)
"""
The computed frame boundaries, used to unwrap the raw timestamps.
"""

FrameForTimeOfFlightOrigin = NewType(
    'FrameForTimeOfFlightOrigin', chopper_cascade.Frame
)
"""
Frame used to compute the time-of-flight origin. Used in WFM.
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

SampleDistance = NewType('SampleDistance', sc.Variable)
"""
Location of the sample along the incident beam. Origin must be consistent with chopper
distance origin.
"""

SourceTimeRange = NewType('SourceTimeRange', tuple[sc.Variable, sc.Variable])
"""
Time range of the source pulse, used for computing frame bounds.
"""

SourceWavelengthRange = NewType(
    'SourceWavelengthRange', tuple[sc.Variable, sc.Variable]
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

UnwrappedData = NewType('UnwrappedData', sc.DataArray)
"""
Detector data with unwrapped time offset and pulse time coordinates.
"""

TimeOfFlightOriginDistance = NewType('TimeOfFlightOriginDistance', sc.Variable)
"""
Distance from the source to the position the time-of-flight origin.
"""

TimeOfFlightOriginTime = NewType('TimeOfFlightOriginTime', sc.Variable)
"""
Time of the time-of-flight origin for each subframe.
"""

TofData = NewType('TofData', sc.DataArray)
"""
Detector data with time-of-flight and time zero coordinates.
"""


TimeOffset = NewType('TimeOffset', sc.Variable)
"""Unwrapped time offset relative to the pulse time."""

DeltaFromWrapped = NewType('DeltaFromWrapped', sc.Variable)
"""Positive delta to be added to the input time offsets to unwrap them."""


@dataclass
class TimeOfFlightOrigin:
    """
    The origin of the time-of-flight, time since pulse time and distance from source.
    """

    time: sc.Variable | sc.DataArray
    distance: sc.Variable


def frame_period(
    pulse_period: PulsePeriod, pulse_stride: PulseStride | None
) -> FramePeriod:
    if pulse_stride is None:
        return pulse_period
    return FramePeriod(pulse_period * pulse_stride)


def chopper_cascade_frames(
    source_wavelength_range: SourceWavelengthRange,
    source_time_range: SourceTimeRange,
    choppers: Choppers,
) -> ChopperCascadeFrames:
    """
    Return all the chopper frames.

    This is the result of propagating the source pulse through the chopper cascade.

    It is assumed that the opening and closing times of the input choppers have been
    setup correctly. This includes a correct definition of the offsets in
    pulse-skipping mode, i.e., the caller must know which pulses are in use.
    """
    frames = chopper_cascade.FrameSequence.from_source_pulse(
        time_min=source_time_range[0],
        time_max=source_time_range[-1],
        wavelength_min=source_wavelength_range[0],
        wavelength_max=source_wavelength_range[-1],
    )
    return ChopperCascadeFrames(frames.chop(choppers.values()))


def frame_at_detector(
    frames: ChopperCascadeFrames,
    ltotal: Ltotal,
) -> FrameAtDetector:
    """
    Return the frame at the detector.

    This is the result of propagating the source pulse through the chopper cascade to
    the detector. The detector may be a monitor or a detector after scattering off the
    sample. The frame bounds are then computed from this.
    """
    return FrameAtDetector(frames[-1].propagate_to(ltotal))
    # return FrameAtDetector(frames[ltotal])


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
    pulse_stride: PulseStride | None,
    event_time_zero: TimeZero,
) -> PulseOffset:
    """
    Return the offset of a pulse within a frame.

    This has no special meaning, since on its own it does not define which pulse
    is used for the frame. Instead of taking this into account here, the SourceChopper
    used to compute OffsetFromTimeOfFlight will take care of this. This assumes that
    the choppers have been defined correctly.
    """
    if pulse_stride is None:
        return PulseOffset(sc.scalar(0, unit=elem_unit(pulse_period)))
    first_pulse_time = (
        event_time_zero[0]
        if len(event_time_zero) > 0
        else sc.zeros_like(event_time_zero)
    )
    # This is roughly equivalent to
    #   (event_time_zero - first_pulse_time) % frame_period
    # but should avoid some issues with precision and drift
    pulse_index = (event_time_zero - first_pulse_time) // pulse_period.to(
        unit=elem_unit(event_time_zero)
    )
    return PulseOffset(pulse_period * (pulse_index % pulse_stride))


def frame_wrapped_time_offset_pulse_skipping(
    offset: PulseWrappedTimeOffset, pulse_offset: PulseOffset
) -> FrameWrappedTimeOffset:
    """Return the time offset wrapped around at the frame period."""
    return FrameWrappedTimeOffset(offset + pulse_offset.to(unit=elem_unit(offset)))


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
    return PulseWrappedTimeOffset(da.bins.coords['event_time_offset'])


def time_zero(da: RawData) -> TimeZero:
    """
    In NXevent_data, event_time_zero is the start of the most recent pulse.
    """
    # This is not available for histogram-mode monitors. We probably need to look it
    # up elsewhere in the file, e.g., NXsource. Should we have a separate provider for
    # monitors?
    if da.bins is None:
        raise NotImplementedError(
            "NXevent_data/event_time_zero not available for histogram-mode monitors."
        )
    return da.coords['event_time_zero']


def offset_from_wrapped(
    wrapped_time_offset: FrameWrappedTimeOffset,
    frame_bounds: FrameBounds,
    frame_period: FramePeriod,
) -> DeltaFromWrapped:
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
    frame_bounds :
        The computed frame boundaries, used to unwrap the raw timestamps.
        Typically pixel-dependent when unwrapping detectors.
    frame_period :
        Time between the start of two consecutive frames, i.e., the period of the
        time-zero used by the data acquisition system.
    """
    time_bounds = frame_bounds['time']
    frame_period = frame_period.to(unit=elem_unit(time_bounds))
    diff = (time_bounds['bound', -1] - time_bounds['bound', 0]) - frame_period
    if any(diff.flatten(to='x') > sc.scalar(0.0, unit=frame_period.unit)):
        raise ValueError(
            "Frames are overlapping: Computed frame bounds "
            f"{frame_bounds} are larger than frame period {frame_period}."
        )
    time_offset_min = time_bounds['bound', 0]
    wrapped_time_min = time_offset_min % frame_period
    # We simply cut, without special handling of times that fall outside the frame
    # period. Everything below and above is allowed. The alternative would be to, e.g.,
    # replace invalid inputs with NaN, but this would probably cause more trouble down
    # the line.
    begin = sc.full_like(wrapped_time_min, value=-math.inf)
    end = sc.full_like(wrapped_time_min, value=math.inf)
    dim = 'section'
    time = sc.concat([begin, wrapped_time_min, end], dim).transpose().copy()
    offset = sc.DataArray(
        time_offset_min
        - wrapped_time_min
        + sc.concat([frame_period, sc.zeros_like(frame_period)], dim),
        coords={dim: time.to(unit=elem_unit(wrapped_time_offset))},
    )
    return DeltaFromWrapped(sc.lookup(offset, dim=dim)[wrapped_time_offset])


def source_chopper(
    choppers: Choppers,
    source_time_range: SourceTimeRange,
    source_chopper_name: SourceChopperName | None,
) -> SourceChopper:
    """
    Return the chopper defining the source location and time-of-flight time origin.

    If there is no pulse-shaping chopper, then the source-pulse begin and end time
    are used to define a fake chopper.
    """
    if source_chopper_name is not None:
        return choppers[source_chopper_name]
    return chopper_cascade.Chopper(
        distance=sc.scalar(0.0, unit='m'),
        time_open=sc.concat([source_time_range[0]], 'cutout').to(unit='s'),
        time_close=sc.concat([source_time_range[1]], 'cutout').to(unit='s'),
    )


def time_of_flight_origin_from_chopper(
    source_chopper: SourceChopper,
) -> TimeOfFlightOrigin:
    """
    Compute the time-of-flight origin from a source chopper.

    This is a naive approach for defining the time of flight. We are not sure this
    is going to be used in practice. An possible alternative might be to calibrate
    the time-of-flight using a sample with a known Bragg edge. This is not implemented
    currently.

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
    source_chopper :
        Chopper defining the source location and time-of-flight time origin (as the
        center of the slit opening). The chopper distance is assumed to be the distance
        from the source position.
    """
    if len(source_chopper.time_open) != 1:
        raise NotImplementedError(
            "Source chopper with multiple openings not supported yet."
        )
    source_time_open = source_chopper.time_open[0]
    source_time_close = source_chopper.time_close[0]
    time_zero = 0.5 * (source_time_open + source_time_close)
    return TimeOfFlightOrigin(time=time_zero, distance=source_chopper.distance)


def time_offset(unwrapped: UnwrappedData) -> TimeOffset:
    """
    Extract the time offset coord of the unwrapped input data.
    """
    if unwrapped.bins is not None:
        return TimeOffset(unwrapped.bins.coords['time_offset'])
    return TimeOffset(unwrapped.coords['time_offset'])


def maybe_clip_detector_subframes(
    frame: FrameAtDetector, ltotal: Ltotal, chopper_cascade_frames: ChopperCascadeFrames
) -> CleanFrameAtDetector:
    """
    Check for time overlap between subframes.
    If overlap is found, we clip away the regions where there is overlap.
    This is done by adding a fake chopper which is closed during the overlapping times.

    Examples:

        1. partial overlap:
        |-------------|            ->   |--------|
                 |-------------|   ->                 |--------|

        2. total overlap:
        |----------------------|   ->   |--------|       |-----|
                 |-------|         ->

    In the case of multiple detector pixels, we find the pixel closest to the source
    and use that as the reference for the clipping.

    Parameters
    ----------
    frame:
        The frame at the detector, with subframes that may overlap.
    ltotal:
        The total distance between the source and the detector.
    chopper_cascade_frames:
        All the frames in the chopper cascade, before the detector.
    """
    # Need subframe dim at the end because sorting requires contiguous data
    dims = (*ltotal.dims, 'subframe')
    starts = sc.concat(
        [sf.start_time for sf in frame.subframes], dim='subframe'
    ).transpose(dims)
    ends = sc.concat([sf.end_time for sf in frame.subframes], dim='subframe').transpose(
        dims
    )

    sizes = starts.sizes
    sizes['subframe'] *= 2
    bounds = sc.empty(sizes=sizes, unit=starts.unit)
    bounds['subframe', ::2] = sc.sort(starts, 'subframe')
    bounds['subframe', 1::2] = sc.sort(ends, 'subframe')
    if all(sc.issorted(bounds, 'subframe').flatten(to='x')):
        return CleanFrameAtDetector(frame)

    sorted_times = sc.sort(bounds, 'subframe')
    if ltotal.dims:
        # Find the pixel closest to the source
        closest_pixel = np.argmin(ltotal.flatten(to='pixel').values)
        sorted_times = sorted_times.flatten(dims=ltotal.dims, to='pixel')[
            'pixel', closest_pixel
        ]

    # Make a fake chopper that is closed during the overlapping times, placed
    # immediately before the closest detector pixel.
    fake_chopper = chopper_cascade.Chopper(
        distance=frame.distance.min(),
        time_open=sorted_times['subframe', ::2],
        time_close=sorted_times['subframe', 1::2],
    )
    # We cannot chop frames at multiple distances at once, so instead of chopping the
    # frame at the detector, we chop the last frame in the chopper cascade before the
    # detector.
    last_frame = chopper_cascade_frames[-1]

    # Chop the subframes one by one
    # TODO: We currently need to chop them one by one as the `chop` method seems
    # to not work correctly with overlapping subframes.
    subframes = []
    for subframe in last_frame.subframes:
        f = chopper_cascade.Frame(distance=last_frame.distance, subframes=[subframe])
        chopped = f.chop(fake_chopper)
        subframes.extend(
            [
                sf
                for sf in chopped.subframes
                if not sc.allclose(sf.start_time, sf.end_time)
            ]
        )

    return CleanFrameAtDetector(
        chopper_cascade.Frame(
            distance=fake_chopper.distance,
            subframes=subframes,
        ).propagate_to(frame.distance)
    )


def time_of_flight_origin_wfm(
    frames: ChopperCascadeFrames,
    clean_detector_frame: CleanFrameAtDetector,
) -> TimeOfFlightOrigin:
    """
    Compute the time-of-flight origin in the WFM case.
    For WFM there is not a single time-of-flight "origin", but one for each subframe.

    Parameters
    ----------
    frames:
        All the frames in the chopper cascade.
    clean_detector_frame:
        The frame at the detector, with subframes that do not overlap.

    Notes
    -----
    To find the time-of-flight origin, we ray-trace the fastest and slowest neutrons of
    the subframes back to the first chopper to determine the time-of-flight origin.
    The assumption here is that the first chopper is one of the two wfm choppers.
    We also assume that the first chopper will be inundated with neutrons from the
    source, and propagating the boundaries of the frame backwards should thus give
    us a good estimate of the opening and closing times of the first chopper. There is
    also the general rule that the longer the instrument, the better the wavelength
    resolution, so going back to the first chopper also makes sense.

    We use this method instead of reading the opening and closing times of the choppers
    because it is not possible to know in a trivial way which opening and closing times
    correspond to which subframe at the detector. There is not always a 1-1
    mapping between the subframes at the detector and the chopper cutouts (for instance,
    the DREAM pulse shaping choppers have 8 cutouts, but typically create 2 subframes
    at the detector). In addition, if the chopper is rotating rapidly, there may be
    multiple opening and closing times, where the extra subframes get blocked by other
    choppers further down the beamline.

    While developing this method, we attempted several other implementations, which all
    gave worse results than the current implementation. These are summarized here for
    reference:

    1. We tried to ray-trace the fastest and slowest neutrons of the subframes back to
       the point in time and space where they intersected, to get the converging point
       for all neutrons. This gives a different distance for each subframe, which is
       not really supported in the current implementation of transform_coords.

    2. A slightly modified version of idea 1. was to ray-trace back to the intersection
       point for each frame, and then compute a mean distance that would apply to all
       frames. We then re-traced all subframes back to this mean distance. This seems to
       give a worse estimate of the time-of-flight origin, probably because in a
       slightly faulty chopper setup (such as V20), we are not always guaranteed that
       the neutrons went through the openings they were meant to go through. We are
       however relatively sure which openings of the first chopper they went through.
    """
    sorted_frame = chopper_cascade.Frame(
        distance=clean_detector_frame.distance,
        subframes=sorted(
            clean_detector_frame.subframes, key=lambda x: x.start_time.min()
        ),
    )

    times = sorted_frame.subbounds()['time'].flatten(
        dims=['subframe', 'bound'], to='subframe'
    )
    # All times before the first subframe and after the last subframe should be
    # replaced by NaN. We add a large padding to make sure all events are covered.
    padding = sc.scalar(1e9, unit='s').to(unit=times.unit)
    low = times['subframe', 0] - padding
    high = times['subframe', -1] + padding
    times = sc.concat([low, times, high], 'subframe')

    # Propagate the subframes from the detector back to the position of the first
    # chopper (note that frame 0 is the pulse itself).
    at_first_chopper = sorted_frame.propagate_to(frames[1].distance)

    starts = sc.concat(
        [subframe.start_time for subframe in at_first_chopper.subframes], dim='subframe'
    )
    ends = sc.concat(
        [subframe.end_time for subframe in at_first_chopper.subframes], dim='subframe'
    )
    time_origins = 0.5 * (starts + ends)

    # We need to add nans between each crossing_time offsets for the bins before,
    # after, and between the subframes.
    sizes = time_origins.sizes
    sizes['subframe'] = 2 * sizes['subframe'] + 1
    shift = sc.full(sizes=sizes, value=math.nan, unit='s')
    shift['subframe', 1::2] = time_origins

    # TODO: what dimension order is the best here? Where should `detector_number` be?
    origin_time = sc.DataArray(shift.transpose(times.dims), coords={'subframe': times})

    return TimeOfFlightOrigin(time=origin_time, distance=at_first_chopper.distance)


def unwrap_data(da: RawData, delta: DeltaFromWrapped) -> UnwrappedData:
    """
    Return the input data with unwrapped time offset and pulse time.

    The time offset is the time since the start of the frame emitting the neutron.
    The pulse time is the time of the pulse that emitted the neutron.

    Parameters
    ----------
    da :
        The input data.
    delta :
        The positive delta that needs to be added to the input time offsets to unwrap
        them. The same delta is also subtracted from the input time zero to obtain the
        pulse time.
    """
    if da.bins is not None:
        da = da.copy(deep=False)
        da.data = sc.bins(**da.bins.constituents)
        da.bins.coords['time_offset'] = da.bins.coords['event_time_offset'] + delta
        if 'event_time_zero' in da.bins.coords:
            coord = da.bins.coords['event_time_zero']
        else:
            # Bin edges are now invalid so we pop them
            coord = da.coords.pop('event_time_zero')
        da.bins.coords['pulse_time'] = coord - delta.to(
            unit=elem_unit(coord), dtype='int64'
        )
    else:
        # 'time_of_flight' is the name in, e.g., NXmonitor
        da = da.transform_coords(
            time_offset=lambda time_of_flight: time_of_flight + delta,
            keep_inputs=False,
        )
        # Generally the coord is now not ordered, might want to cut, swap, and concat,
        # but it is unclear what to do with the split bin in the middle and how to
        # join the two halves. The end of the last bin generally does not match the
        # begin of the first bin, there may be a significant gap (which we could fill
        # with NaNs or zeros), but there could also be a small overlap.
    return UnwrappedData(da)


def to_time_of_flight(
    da: UnwrappedData, origin: TimeOfFlightOrigin, ltotal: Ltotal
) -> TofData:
    """
    Return the input data with 'tof', 'time_zero', and corrected 'Ltotal' coordinates.

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
    time_offset = (
        da.coords['time_offset'] if da.bins is None else da.bins.coords['time_offset']
    )
    delta = origin.time
    if isinstance(delta, sc.DataArray):
        # Will raise if subframes overlap, since coord for lookup table must be sorted
        # TODO: Can we do without making a copy of delta?
        delta = sc.lookup(delta.copy(deep=True), dim='subframe')[time_offset]
    if da.bins is not None:
        da = da.copy(deep=False)
        da.data = sc.bins(**da.bins.constituents)
        da.bins.coords['tof'] = time_offset - delta
        da.bins.coords['time_zero'] = da.bins.coords['pulse_time'] + delta.to(
            unit=elem_unit(da.bins.coords['pulse_time']), dtype='int64'
        )
    else:
        da = da.transform_coords(
            tof=lambda time_offset: time_offset - delta, keep_inputs=False
        )
    if (existing := da.coords.get('Ltotal')) is not None:
        if not sc.identical(existing, ltotal):
            raise ValueError(
                f"Ltotal {existing} in data does not match Ltotal {ltotal} "
                "used for calculating time-of-flight."
            )

    da.coords['Ltotal'] = ltotal - origin.distance
    return TofData(da)


_common_providers = (
    chopper_cascade_frames,
    frame_at_detector,
    frame_bounds,
    frame_period,
    pulse_wrapped_time_offset,
    source_chopper,
    offset_from_wrapped,
    unwrap_data,
)

_non_skipping = (frame_wrapped_time_offset,)
_skipping = (frame_wrapped_time_offset_pulse_skipping, pulse_offset, time_zero)

_wfm = (
    maybe_clip_detector_subframes,
    subframe_bounds,
    time_offset,
    time_of_flight_origin_wfm,
)
_non_wfm = (time_of_flight_origin_from_chopper,)


def time_of_flight_providers() -> tuple[Callable]:
    """
    Return the providers computing the time-of-flight and time-zero coordinates.
    """
    return (to_time_of_flight,)


def time_of_flight_origin_from_choppers_providers(wfm: bool = False):
    """
    Return the providers for computing the time-of-flight origin via the chopper
    cascade.

    Parameters
    ----------
    wfm :
        If True, the data is assumed to be from a WFM instrument.
    """
    wfm = _wfm if wfm else _non_wfm
    _common = (source_chopper, frame_at_detector, subframe_bounds)
    return _common + wfm


def unwrap_providers(pulse_skipping: bool = False):
    """
    Return the list of providers for unwrapping frames.

    Parameters
    ----------
    pulse_skipping :
        If True, the pulse-skipping mode is assumed.
    """
    skipping = _skipping if pulse_skipping else _non_skipping
    return _common_providers + skipping
