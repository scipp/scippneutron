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
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NewType

import numpy as np
import scipp as sc
from scipp.core.bins import Lookup

from .._utils import elem_unit
from . import chopper_cascade

Choppers = NewType('Choppers', Mapping[str, chopper_cascade.Chopper])
"""
Choppers used to define the frame parameters.
"""

# FrameAtSample = NewType('FrameAtSample', chopper_cascade.Frame)
# """
# Result of passing the source pulse through the chopper cascade.

# After propagating the frame to the detector distances this will then be used to
# compute the frame bounds.
# """

ChopperCascadeFrames = NewType('ChopperCascadeFrames', chopper_cascade.FrameSequence)
"""
Frames of the chopper cascade.
"""

FrameAtDetector = NewType('FrameAtDetector', chopper_cascade.Frame)
"""
Result of passing the source pulse through the chopper cascade to the detector.

The detector may be a monitor or a detector after scattering off the sample. The frame
bounds are then computed from this.
"""

# FrameBounds = NewType('FrameBounds', sc.DataGroup)
# """
# The computed frame boundaries, used to unwrap the raw timestamps.
# """

FramePeriod = NewType('FramePeriod', sc.Variable)
"""
The period of a frame, a (small) integer multiple of the source period.
"""

# FrameWrappedTimeOffset = NewType('FrameWrappedTimeOffset', sc.Variable)
# """
# Time offsets wrapped around at the frame period.

# In normal mode, this is identical to the NXevent_data/event_time_offset recorded by
# the data acquisition system, i.e., the same as :py:class:`PulseWrappedTimeOffset`.

# In pulse-skipping mode, this is the time offset since the start of the frame.
# """

UnwrappedTimeOfArrival = NewType('UnwrappedTimeOfArrival', sc.Variable)
"""
Time of arrival of the neutron at the detector, unwrapped at the pulse period.
"""

FrameAtDetectorStartTime = NewType('FrameAtDetectorStartTime', sc.Variable)
"""
Time of the start of the frame at the detector.
"""

UnwrappedTimeOfArrivalMinusStartTime = NewType(
    'UnwrappedTimeOfArrivalMinusStartTime', sc.Variable
)
"""
Time of arrival of the neutron at the detector, unwrapped at the pulse period, minus
the start time of the frame.
"""

TimeOfArrivalModuloPeriod = NewType('TimeOfArrivalModuloPeriod', sc.Variable)
"""
Time of arrival of the neutron at the detector modulo the frame period.
"""

TimeOfArrivalMinusStartTimeModuloPeriod = NewType(
    'TimeOfArrivalMinusStartTimeModuloPeriod', sc.Variable
)
"""
Time of arrival of the neutron at the detector minus the start time of the frame,
modulo the frame period.
"""


@dataclass
class SlopeAndInterceptLookup:
    """ """

    slope: Lookup
    intercept: Lookup


TofCoord = NewType('TofCoord', sc.Variable)
"""
Tof coordinate computed by the workflow.
"""

Ltotal = NewType('Ltotal', sc.Variable)
"""
Total distance between the source and the detector(s).

This is used to propagate the frame to the detector position. This will then yield
detector-dependent frame bounds. This is typically the sum of L1 and L2, except for
monitors.
"""

# PulseOffset = NewType('PulseOffset', sc.Variable)
# """
# Offset from the start of the frame in pulse-skipping mode, multiple of pulse period.
# """

PulsePeriod = NewType('PulsePeriod', sc.Variable)
"""
Period of the source pulses, i.e., time between consecutive pulse starts.
"""

PulseStride = NewType('PulseStride', int)
"""
Stride of used pulses. Usually 1, but may be a small integer when pulse-skipping.
"""

# PulseWrappedTimeOffset = NewType('PulseWrappedTimeOffset', sc.Variable)
# """
# Time offsets wrapped at the pulse period, typically NXevent_data/event_time_offset.
# """

RawData = NewType('RawData', sc.DataArray)
"""
Raw detector data loaded from a NeXus file, e.g., NXdetector containing NXevent_data.
"""

# SourceChopperName = NewType('SourceChopperName', str)
# """
# Name of the chopper defining the source location and time-of-flight time origin.
# """

# SourceChopper = NewType('SourceChopper', chopper_cascade.Chopper)
# """
# Chopper defining the source location and time-of-flight time origin.

# If there is no pulse-shaping chopper, a fake chopper with time_open and time_close
# set to the source pulse begin and end time should be used.
# """

# SampleDistance = NewType('SampleDistance', sc.Variable)
# """
# Location of the sample along the incident beam. Origin must be consistent with chopper
# distance origin.
# """

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

# SubframeBounds = NewType('SubframeBounds', sc.Variable)
# """
# The computed subframe boundaries, used to offset the raw timestamps for WFM.
# """

# TimeZero = NewType('TimeZero', sc.Variable)
# """
# Time of the start of the most recent pulse, typically NXevent_data/event_time_zero.
# """

# UnwrappedData = NewType('UnwrappedData', sc.DataArray)
# """
# Detector data with unwrapped time offset and pulse time coordinates.
# """

# TimeOfFlightOriginDistance = NewType('TimeOfFlightOriginDistance', sc.Variable)
# """
# Distance from the source to the position the time-of-flight origin.
# """

# TimeOfFlightOriginTime = NewType('TimeOfFlightOriginTime', sc.Variable)
# """
# Time of the time-of-flight origin for each subframe.
# """

TofData = NewType('TofData', sc.DataArray)
"""
Detector data with time-of-flight and time zero coordinates.
"""


# # TimeOffset = NewType('TimeOffset', sc.Variable)
# # """Unwrapped time offset relative to the pulse time."""

# # DeltaFromWrapped = NewType('DeltaFromWrapped', sc.Variable)
# # """Positive delta to be added to the input time offsets to unwrap them."""


# @dataclass
# class TimeOfFlightOrigin:
#     """
#     The origin of the time-of-flight, time since pulse time and distance from source.
#     """

#     time: sc.Variable | sc.DataArray
#     distance: sc.Variable


# WFMChopperNames = NewType('WFMChopperNames', tuple[str, ...])
# """
# Names of the WFM choppers in the beamline.
# """

# WFMChoppers = NewType('WFMChoppers', tuple[chopper_cascade.Chopper, ...])
# """
# The WFM choppers in the beamline.
# """


def frame_period(pulse_period: PulsePeriod, pulse_stride: PulseStride) -> FramePeriod:
    # if pulse_stride is None:
    #     return pulse_period
    return FramePeriod(pulse_period * pulse_stride)


def chopper_cascade_frames(
    source_wavelength_range: SourceWavelengthRange,
    source_time_range: SourceTimeRange,
    choppers: Choppers,
) -> ChopperCascadeFrames:
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
    period: FramePeriod,
) -> FrameAtDetector:
    """
    Return the frame at the detector.

    This is the result of propagating the source pulse through the chopper cascade to
    the detector. The detector may be a monitor or a detector after scattering off the
    sample. The frame bounds are then computed from this.

    It is assumed that the opening and closing times of the input choppers have been
    setup correctly. This includes a correct definition of the offsets in
    pulse-skipping mode, i.e., the caller must know which pulses are in use.
    """
    # frames = chopper_cascade.FrameSequence.from_source_pulse(
    #     time_min=source_time_range[0],
    #     time_max=source_time_range[-1],
    #     wavelength_min=source_wavelength_range[0],
    #     wavelength_max=source_wavelength_range[-1],
    # )
    # frames = frames.chop(choppers.values())
    at_detector = frames[-1].propagate_to(ltotal)

    # Check that the frame bounds do not span a range larger than the frame period.
    # This would indicate that the chopper phases are not set correctly.
    bounds = at_detector.bounds()['time']
    if bounds.max() - bounds.min() > period:
        raise ValueError(
            "Frames are overlapping: Computed frame bounds "
            f"{bounds} are larger than frame period {period}."
        )
    return FrameAtDetector(at_detector)


# def frame_bounds(frame: FrameAtDetector) -> FrameBounds:
#     return FrameBounds(frame.bounds())


# def subframe_bounds(frame: FrameAtDetector) -> SubframeBounds:
#     """Used for WFM."""
#     return SubframeBounds(frame.subbounds())


# def frame_wrapped_time_offset(offset: PulseWrappedTimeOffset) -> FrameWrappedTimeOffset:
#     """Without pulse-skipping, this is an identity function."""
#     return FrameWrappedTimeOffset(offset)


# def pulse_offset(
#     pulse_period: PulsePeriod,
#     pulse_stride: PulseStride | None,
#     event_time_zero: TimeZero,
# ) -> PulseOffset:
#     """
#     Return the offset of a pulse within a frame.

#     This has no special meaning, since on its own it does not define which pulse
#     is used for the frame. Instead of taking this into account here, the SourceChopper
#     used to compute OffsetFromTimeOfFlight will take care of this. This assumes that
#     the choppers have been defined correctly.
#     """
#     if pulse_stride is None:
#         return PulseOffset(sc.scalar(0, unit=elem_unit(pulse_period)))
#     first_pulse_time = (
#         event_time_zero[0]
#         if len(event_time_zero) > 0
#         else sc.zeros_like(event_time_zero)
#     )
#     # This is roughly equivalent to
#     #   (event_time_zero - first_pulse_time) % frame_period
#     # but should avoid some issues with precision and drift
#     pulse_index = (event_time_zero - first_pulse_time) // pulse_period.to(
#         unit=elem_unit(event_time_zero)
#     )
#     return PulseOffset(pulse_period * (pulse_index % pulse_stride))


# def frame_wrapped_time_offset_pulse_skipping(
#     offset: PulseWrappedTimeOffset, pulse_offset: PulseOffset
# ) -> FrameWrappedTimeOffset:
#     """Return the time offset wrapped around at the frame period."""
#     return FrameWrappedTimeOffset(offset + pulse_offset.to(unit=elem_unit(offset)))


# def pulse_wrapped_time_offset(da: RawData) -> PulseWrappedTimeOffset:
#     """
#     In NXevent_data, event_time_offset is the time offset since event_time_zero,
#     which is the start of the most recent pulse. This is not the same as the start
#     of the pulse that emitted the neutron, so event_time_offset is "wrapped" around
#     at the pulse period.
#     """
#     if da.bins is None:
#         # Canonical name in NXmonitor
#         return PulseWrappedTimeOffset(da.coords['time_of_flight'])
#     return PulseWrappedTimeOffset(da.bins.coords['event_time_offset'])


def unwrapped_time_of_arrival(
    da: RawData, period: PulsePeriod
) -> UnwrappedTimeOfArrival:
    """ """
    if da.bins is None:
        # Canonical name in NXmonitor
        toa = da.coords['time_of_flight']
    else:
        coord = da.bins.coords['event_time_offset']
        toa = coord + (sc.arange('pulse', da.sizes['pulse']) * period).to(
            unit=elem_unit(coord), copy=False
        )
    return UnwrappedTimeOfArrival(toa)


def frame_at_detector_start_time(frame: FrameAtDetector) -> FrameAtDetectorStartTime:
    return FrameAtDetectorStartTime(frame.bounds()['time']['bound', 0] * 0.9)


def unwrapped_time_of_arrival_minus_frame_start_time(
    toa: UnwrappedTimeOfArrival, start_time: FrameAtDetectorStartTime
) -> UnwrappedTimeOfArrivalMinusStartTime:
    return UnwrappedTimeOfArrivalMinusStartTime(
        toa - start_time.to(unit=elem_unit(toa), copy=False)
    )


def time_of_arrival_minus_start_time_modulo_period(
    toa_minus_start_time: UnwrappedTimeOfArrivalMinusStartTime,
    frame_period: FramePeriod,
) -> TimeOfArrivalMinusStartTimeModuloPeriod:
    return TimeOfArrivalMinusStartTimeModuloPeriod(
        toa_minus_start_time
        % frame_period.to(unit=elem_unit(toa_minus_start_time), copy=False)
    )


def time_of_arrival_modulo_period(
    toa: UnwrappedTimeOfArrival, frame_period: FramePeriod
) -> TimeOfArrivalModuloPeriod:
    return TimeOfArrivalModuloPeriod(
        toa % frame_period.to(unit=elem_unit(toa), copy=False)
    )


def slope_and_intercept_lookups(
    frame: FrameAtDetector, frame_start: FrameAtDetectorStartTime, ltotal: Ltotal
) -> SlopeAndInterceptLookup:
    """
    Compute the slope and intercept lookups which can be used to compute the
    time-of-flight from the time-of-arrival.

    We take the polygons that define the subframes, given by the chopper cascade, and
    approximate them by straight lines.
    To compute the slopes of these lines, we compute the slopes of all the lines that
    connect any two vertices of the polygons (all-to-all).
    We then compute the weighted mean of these slopes, where the weight is the distance
    between the vertices.
    The intercepts are computed as the mean of the intercepts of the lines that pass
    through all the polygon's vertices.

    The slopes and intercepts are stored in lookup tables, which are used further down
    the pipeline to compute the time-of-flight from the time-of-arrival.
    """
    slopes = []
    intercepts = []
    subframes = sorted(frame.subframes, key=lambda x: x.start_time.min())
    edges = []
    for sf in subframes:
        edges.extend([sf.start_time, sf.end_time])
        x = sf.time - frame_start
        y = (ltotal * chopper_cascade.wavelength_to_inverse_velocity(sf.wavelength)).to(
            unit=x.unit, copy=False
        )
        # Compute the slopes for all pairs of points
        xdiff = x - x.rename_dims(vertex='vertex2')
        ydiff = y - y.rename_dims(vertex='vertex2')
        mm = ydiff / xdiff
        # Compute the weighted mean of the slope, where the weight is the distance
        # between the vertices
        dims = set(mm.dims) - set(frame_start.dims)
        weight = sc.sqrt(xdiff**2 + ydiff**2)
        a = weight * mm
        for dim in dims:
            a = a.nansum(dim)
        a /= weight.sum(dims)
        # Compute the intercept as the mean of the intercepts of the curves that pass
        # through the vertices
        b = (y - a * x).mean('vertex')
        slopes.append(a)
        intercepts.append(b)

    # It is sometimes possible that there is time overlap between subframes.
    # This is not desired in a chopper cascade but can sometimes happen if the phases
    # are not set correctly. Overlap would mean that the start of the next subframe is
    # before the end of the previous subframe.
    # We sort the edges to make sure that the lookup table is sorted. This creates a
    # gap between the overlapping subframes, and discards any neutrons (gives them a
    # NaN tof) that fall into the gap, which is the desired behaviour because we
    # cannot determine the correct tof for them.
    edges = (
        sc.sort(sc.concat(edges, 'subframe').transpose().copy(), 'subframe')
        - frame_start
    )
    sizes = frame_start.sizes | {'subframe': 2 * len(subframes) - 1}
    keys = list(sizes.keys())

    data = sc.full(sizes=sizes, value=np.nan)
    data['subframe', ::2] = sc.concat(slopes, 'subframe').transpose(keys)
    a_lookup = sc.DataArray(data=data, coords={'subframe': edges})

    data = sc.full(sizes=sizes, value=np.nan, unit=sf.time.unit)
    data['subframe', ::2] = sc.concat(intercepts, 'subframe').transpose(keys)
    b_lookup = sc.DataArray(data=data, coords={'subframe': edges})

    return SlopeAndInterceptLookup(
        # slope=sc.concat(slopes, 'subframe').transpose(keys).mean('subframe'),
        slope=a_lookup,
        intercept=b_lookup,
    )


# def time_of_flight_from_lookup(
#     toa: TimeOfArrivalMinusStartTimeModuloPeriod,
#     lookup: SlopeAndInterceptLookup,
# ) -> TofCoord:
#     """
#     Compute the wavelength from the time of arrival and the slope and intercept lookups.
#     """
#     # Ensure unit consistency
#     # subframe_edges = lookup.slope.coords['subframe'].to(unit=elem_unit(toa), copy=False)
#     # Both slope and intercepts should have the same subframe edges
#     # lookup.slope.coords['subframe'] = subframe_edges
#     lookup.intercept.coords['subframe'] = lookup.intercept.coords['subframe'].to(
#         unit=elem_unit(toa), copy=False
#     )
#     lookup.intercept.data = lookup.intercept.data.to(unit=elem_unit(toa), copy=False)

#     if toa.bins is not None:
#         # slope = sc.lookup(lookup.slope, dim='subframe')[toa]
#         intercept = sc.lookup(lookup.intercept, dim='subframe')[toa]
#         out = lookup.slope * toa + intercept
#     else:
#         # In the case of histogrammed data, we exclude the NaNs that were introduced
#         # further up, and simply broadcast the toas over the subframe dimension.
#         intercept = intercept.data[::2]
#         # Next, we add the subframe edges to the toas and then sort them: this will be
#         # our new time_of_flight coordinate.
#         dim = 'time_of_flight'
#         edges = sc.sort(
#             sc.concat(
#                 [
#                     toa,
#                     lookup.intercept.coords['subframe'].rename_dims(subframe=dim),
#                 ],
#                 dim=dim,
#             ),
#             dim,
#         )
#         out = lookup.slope * edges + intercept
#     return TofCoord(out)


def time_of_flight_no_choppers(da: RawData, toa: TimeOfArrivalModuloPeriod) -> TofData:
    """
    Compute the time-of-flight without choppers.
    """

    out = da.copy(deep=False)
    if toa.bins is not None:
        out.data = sc.bins(**out.bins.constituents)
        # da.bins.coords['tof']
        out.bins.coords['tof'] = toa
    else:
        out.coords['tof'] = toa
    return TofData(out)


def time_of_flight_data(
    da: RawData,
    toa: TimeOfArrivalMinusStartTimeModuloPeriod,
    lookup: SlopeAndInterceptLookup,
) -> TofData:
    # out = da.copy(deep=False)

    # Ensure unit consistency
    subframe_edges = lookup.slope.coords['subframe'].to(unit=elem_unit(toa), copy=False)
    # Both slope and intercepts should have the same subframe edges
    lookup.slope.coords['subframe'] = subframe_edges
    lookup.intercept.coords['subframe'] = lookup.intercept.coords['subframe'].to(
        unit=elem_unit(toa), copy=False
    )
    lookup.intercept.data = lookup.intercept.data.to(unit=elem_unit(toa), copy=False)

    # if toa.bins is not None:
    #     # slope = sc.lookup(lookup.slope, dim='subframe')[toa]
    #     intercept = sc.lookup(lookup.intercept, dim='subframe')[toa]
    #     out = lookup.slope * toa + intercept

    # lookup.intercept.coords['subframe'] = lookup.intercept.coords['subframe'].to(
    #     unit=elem_unit(toa), copy=False
    # )
    # lookup.intercept.data = lookup.intercept.data.to(unit=elem_unit(toa), copy=False)

    if toa.bins is not None:
        slope = sc.lookup(lookup.slope, dim='subframe')[toa]
        intercept = sc.lookup(lookup.intercept, dim='subframe')[toa]
        # tof = lookup.slope * toa + intercept
        tof = slope * toa + intercept
        out = da.copy(deep=False)
        out.data = sc.bins(**out.bins.constituents)
        out.bins.coords['tof'] = tof
    else:
        # In the case of histogrammed data, we exclude the NaNs that were introduced
        # further up, and simply broadcast the toas over the subframe dimension.
        slope = lookup.slope.data['subframe', ::2]
        intercept = lookup.intercept.data['subframe', ::2]
        # Next, we add the subframe edges to the toas and then sort them: this will be
        # our new time_of_flight coordinate.
        dim = 'time_of_flight'
        edges = sc.sort(
            sc.concat(
                [
                    toa,
                    lookup.intercept.coords['subframe'].rename_dims(subframe=dim),
                ],
                dim=dim,
            ),
            dim,
        )
        print(edges)
        tof = slope * edges + intercept

        with_toa = da.copy(deep=False)
        with_toa.coords['time_of_flight'] = toa
        with_toa = sc.broadcast(
            with_toa, sizes=with_toa.sizes | {'subframe': slope.sizes['subframe']}
        )

        rebinned = with_toa.rebin(time_of_flight=edges)

        # Create a 2d mask to exlude invalid data in each subframe
        mask = sc.DataArray(
            data=sc.ones(sizes=rebinned.data.sizes, dtype=bool),
            coords={'time_of_flight': rebinned.coords['time_of_flight']},
        )
        # for i in range((lookup.intercept.sizes['subframe'] - 1) // 2):
        for i in range(slope.sizes['subframe']):
            bounds = lookup.intercept[i * 2].coords['subframe']
            tmin = bounds.min()
            tmax = bounds.max()
            # Exclude the upper edge to avoid masking one bin too many
            tmax = sc.scalar(np.nextafter(tmax.value, -np.inf), unit=tmax.unit)
            print(tmin, tmax)
            mask['subframe', i]['time_of_flight', tmin:tmax].values[:] = False

        rebinned.masks[uuid.uuid4().hex] = mask.data
        rebinned.coords['tof'] = tof

        # We now switch to the new time-of-flight coordinate and rebin on a common axis.
        # We use a reasonable number of to try and preserve the resolution of the data.
        old = rebinned.coords['time_of_flight']
        new = rebinned.coords['tof']
        nbins = int(len(old) * (new.max() - new.min()) / (old.max() - old.min()))

        out = (
            rebinned.rename_dims(time_of_flight='tof').rebin(tof=nbins).sum('subframe')
        )

    #         c = reb.coords['time_of_flight'] * a.mean() + b
    # tofdata = reb.copy(deep=False)
    # tofdata.coords['tof'] = c

    return TofData(out)

    # if tof.bins is not None:
    #     out.data = sc.bins(**out.bins.constituents)
    #     # da.bins.coords['tof']
    #     out.bins.coords['tof'] = tof
    # else:
    #     da.coords['time_of_flight'] = workflow.compute(
    #         unwrap.TimeOfArrivalMinusStartTimeModuloPeriod
    #     )

    #     out.coords['tof'] = tof
    # return TofData(out)


# def time_zero(da: RawData) -> TimeZero:
#     """
#     In NXevent_data, event_time_zero is the start of the most recent pulse.
#     """
#     # This is not available for histogram-mode monitors. We probably need to look it
#     # up elsewhere in the file, e.g., NXsource. Should we have a separate provider for
#     # monitors?
#     if da.bins is None:
#         raise NotImplementedError(
#             "NXevent_data/event_time_zero not available for histogram-mode monitors."
#         )
#     return da.coords['event_time_zero']


# def offset_from_wrapped(
#     wrapped_time_offset: FrameWrappedTimeOffset,
#     frame_bounds: FrameBounds,
#     frame_period: FramePeriod,
# ) -> DeltaFromWrapped:
#     """
#     Offset of the input time offsets from the start of the frame emitting the neutron.

#     In other words, this is the part that is "lost" by the conceptual wrapping of the
#     time offsets at the frame period. This is the offset that needs to be added to the
#     NXevent_data/event_time_offset to obtain the offset from the start of the emitting
#     frame.

#     This is not identical to the offset to time-of-flight, since the time-of-flight is
#     measured, e.g., from the center of the pulse, to the center of a pulse-shaping
#     chopper slit.

#     Note that this takes the frame-period, which can be a multiple of a pulse-period.
#     The input time stamps must be relative the the start of the frame, i.e., in
#     pulse-skipping mode a pulse offset must have been applied before calling this
#     function.

#     Parameters
#     ----------
#     wrapped_time_offset :
#         Time offset from the time-zero as recorded by the data acquisition system.
#     frame_bounds :
#         The computed frame boundaries, used to unwrap the raw timestamps.
#         Typically pixel-dependent when unwrapping detectors.
#     frame_period :
#         Time between the start of two consecutive frames, i.e., the period of the
#         time-zero used by the data acquisition system.
#     """
#     time_bounds = frame_bounds['time']
#     frame_period = frame_period.to(unit=elem_unit(time_bounds))
#     if time_bounds['bound', -1] - time_bounds['bound', 0] > frame_period:
#         raise ValueError(
#             "Frames are overlapping: Computed frame bounds "
#             f"{frame_bounds} are larger than frame period {frame_period}."
#         )
#     time_offset_min = time_bounds['bound', 0]
#     wrapped_time_min = time_offset_min % frame_period
#     # We simply cut, without special handling of times that fall outside the frame
#     # period. Everything below and above is allowed. The alternative would be to, e.g.,
#     # replace invalid inputs with NaN, but this would probable cause more trouble down
#     # the line.
#     begin = sc.full_like(wrapped_time_min, value=-math.inf)
#     end = sc.full_like(wrapped_time_min, value=math.inf)
#     dim = 'section'
#     time = sc.concat([begin, wrapped_time_min, end], dim).transpose().copy()
#     offset = sc.DataArray(
#         time_offset_min
#         - wrapped_time_min
#         + sc.concat([frame_period, sc.zeros_like(frame_period)], dim),
#         coords={dim: time.to(unit=elem_unit(wrapped_time_offset))},
#     )
#     return DeltaFromWrapped(sc.lookup(offset, dim=dim)[wrapped_time_offset])


# def source_chopper(
#     choppers: Choppers,
#     source_time_range: SourceTimeRange,
#     source_chopper_name: SourceChopperName | None,
# ) -> SourceChopper:
#     """
#     Return the chopper defining the source location and time-of-flight time origin.

#     If there is no pulse-shaping chopper, then the source-pulse begin and end time
#     are used to define a fake chopper.
#     """
#     if source_chopper_name is not None:
#         return choppers[source_chopper_name]
#     return chopper_cascade.Chopper(
#         distance=sc.scalar(0.0, unit='m'),
#         time_open=sc.concat([source_time_range[0]], 'cutout').to(unit='s'),
#         time_close=sc.concat([source_time_range[1]], 'cutout').to(unit='s'),
#     )


# def time_of_flight_origin_from_chopper(
#     source_chopper: SourceChopper,
# ) -> TimeOfFlightOrigin:
#     """
#     Compute the time-of-flight origin from a source chopper.

#     This is a naive approach for defining the time of flight. We are not sure this
#     is going to be used in practice. An possible alternative might be to calibrate
#     the time-of-flight using a sample with a known Bragg edge. This is not implemented
#     currently.

#     A chopper is used to define (1) the "source" location and (2) the time-of-flight
#     time origin. The time-of-flight is then the time difference between the time of
#     arrival of the neutron at the detector, and the time of arrival of the neutron at
#     the chopper. L1 needs to be redefined to be the distance between the chopper and
#     the sample.

#     If there is no pulse-shaping chopper, then the source-pulse begin and end time
#     should be set as the source_time_open and source_time_close, respectively.

#     For WFM, the source_time_open and source_time_close will be different for each
#     subframe. In this case, all input parameters should be given as variables with
#     subframe as dimension.

#     Parameters
#     ----------
#     source_chopper :
#         Chopper defining the source location and time-of-flight time origin (as the
#         center of the slit opening). The chopper distance is assumed to be the distance
#         from the source position.
#     """
#     if len(source_chopper.time_open) != 1:
#         raise NotImplementedError(
#             "Source chopper with multiple openings not supported yet."
#         )
#     source_time_open = source_chopper.time_open[0]
#     source_time_close = source_chopper.time_close[0]
#     time_zero = 0.5 * (source_time_open + source_time_close)
#     return TimeOfFlightOrigin(time=time_zero, distance=source_chopper.distance)


# def time_offset(unwrapped: UnwrappedData) -> TimeOffset:
#     """
#     Extract the time offset coord of the unwrapped input data.
#     """
#     if unwrapped.bins is not None:
#         return TimeOffset(unwrapped.bins.coords['time_offset'])
#     return TimeOffset(unwrapped.coords['time_offset'])


# def time_of_flight_origin_wfm(
#     distance: TimeOfFlightOriginDistance,
#     subframe_origin_time: TimeOfFlightOriginTime,
#     subframe_bounds: SubframeBounds,
# ) -> TimeOfFlightOrigin:
#     """
#     Compute the time-of-flight origin in the WFM case.
#     For WFM there is not a single time-of-flight "origin", but one for each subframe.
#     In the case of a pair of WFM choppers, the distance to the time-of-flight origin
#     would typically be the average of the distances of the choppers, while the time
#     of the time-of-flight origin would be the average of the opening and closing times
#     of the WFM choppers.

#     Parameters
#     ----------
#     distance:
#         The distance to the time-of-flight origin. In the case of a pair of WFM
#         choppers, this would for example be the mid-point between the choppers.
#     subframe_origin_time:
#         The time of the time-of-flight origin for each subframe. In the case of a pair
#         of WFM choppers, this would for example be the mid-point between the opening
#         and closing times of the choppers.
#     subframe_bounds:
#         The computed subframe boundaries, used to offset the raw timestamps for WFM.
#     """
#     times = subframe_bounds['time'].flatten(dims=['subframe', 'bound'], to='subframe')
#     # All times before the first subframe and after the last subframe should be
#     # replaced by NaN. We add a large padding to make sure all events are covered.
#     padding = sc.scalar(1e9, unit='s').to(unit=times.unit)
#     low = times[0] - padding
#     high = times[-1] + padding
#     times = sc.concat([low, times, high], 'subframe')

#     # Select n time origins to match subframe bounds.
#     # We compute the time the fastest neutron would take to reach the chopper and take
#     # the first set of time origins that are greater than that time.
#     t_fastest_neutron = (
#         chopper_cascade.wavelength_to_inverse_velocity(
#             subframe_bounds['wavelength'].min()
#         )
#         * distance
#     )
#     subframe_origin_time = subframe_origin_time[
#         subframe_origin_time > t_fastest_neutron
#     ][: subframe_bounds.sizes['subframe']]

#     # We need to add nans between each subframe_origin_time offsets for the bins before,
#     # after, and between the subframes.
#     shift = sc.full(
#         sizes={'subframe': len(subframe_origin_time) * 2 + 1},
#         value=math.nan,
#         unit='s',
#     )
#     shift[1::2] = subframe_origin_time.rename_dims(cutout='subframe')
#     return TimeOfFlightOrigin(
#         time=sc.DataArray(shift, coords={'subframe': times}),
#         distance=distance,
#     )


# def wfm_choppers(
#     choppers: Choppers,
#     wfm_chopper_names: WFMChopperNames,
# ) -> WFMChoppers:
#     """
#     Get the WFM choppers from the chopper cascade.
#     The choppers are identified by their names.

#     Parameters
#     ----------
#     choppers:
#         All the choppers in the beamline.
#     wfm_chopper_names:
#         Names of the WFM choppers.
#     """
#     return WFMChoppers(tuple(choppers[name] for name in wfm_chopper_names))


# def time_of_flight_origin_distance_wfm_from_choppers(
#     wfm_choppers: WFMChoppers,
# ) -> TimeOfFlightOriginDistance:
#     """
#     Compute the distance to the time of flight origin, in the case of a set
#     (usually a pair) of WFM choppers. The distance is the average of the distances
#     of the choppers.

#     Parameters
#     ----------
#     wfm_choppers:
#         The WFM choppers.
#     """
#     return TimeOfFlightOriginDistance(
#         sc.reduce([ch.distance for ch in wfm_choppers]).mean()
#     )


# def time_of_flight_origin_time_wfm_from_choppers(
#     wfm_choppers: WFMChoppers,
# ) -> TimeOfFlightOriginTime:
#     """
#     Compute the time of the time of flight origin, in the case of a set
#     (usually a pair) of WFM choppers. The time is the mean of the opening and closing
#     times of the WFM choppers.

#     Parameters
#     ----------
#     wfm_choppers:
#         The WFM choppers.
#     """
#     times = []
#     for ch in wfm_choppers:
#         times.append(ch.time_open)
#         times.append(ch.time_close)
#     return TimeOfFlightOriginTime(sc.reduce(times).mean())


# def unwrap_data(da: RawData, delta: DeltaFromWrapped) -> UnwrappedData:
#     """
#     Return the input data with unwrapped time offset and pulse time.

#     The time offset is the time since the start of the frame emitting the neutron.
#     The pulse time is the time of the pulse that emitted the neutron.

#     Parameters
#     ----------
#     da :
#         The input data.
#     delta :
#         The positive delta that needs to be added to the input time offsets to unwrap
#         them. The same delta is also subtracted from the input time zero to obtain the
#         pulse time.
#     """
#     if da.bins is not None:
#         da = da.copy(deep=False)
#         da.data = sc.bins(**da.bins.constituents)
#         da.bins.coords['time_offset'] = da.bins.coords['event_time_offset'] + delta
#         if 'event_time_zero' in da.bins.coords:
#             coord = da.bins.coords['event_time_zero']
#         else:
#             # Bin edges are now invalid so we pop them
#             coord = da.coords.pop('event_time_zero')
#         da.bins.coords['pulse_time'] = coord - delta.to(
#             unit=elem_unit(coord), dtype='int64'
#         )
#     else:
#         # 'time_of_flight' is the name in, e.g., NXmonitor
#         da = da.transform_coords(
#             time_offset=lambda time_of_flight: time_of_flight + delta,
#             keep_inputs=False,
#         )
#         # Generally the coord is now not ordered, might want to cut, swap, and concat,
#         # but it is unclear what to do with the split bin in the middle and how to
#         # join the two halves. The end of the last bin generally does not match the
#         # begin of the first bin, there may be a significant gap (which we could fill
#         # with NaNs or zeros), but there could also be a small overlap.
#     return UnwrappedData(da)


# def to_time_of_flight(
#     da: UnwrappedData, origin: TimeOfFlightOrigin, ltotal: Ltotal
# ) -> TofData:
#     """
#     Return the input data with 'tof', 'time_zero', and corrected 'Ltotal' coordinates.

#     The 'tof' coordinate is the time-of-flight of the neutron, i.e., the time
#     difference between the time of arrival of the neutron at the detector, and the
#     time of the neutron at the source or source-chopper. The 'time_zero' coordinate is
#     the time of the neutron at the source or source-chopper.

#     It is critical to compute the 'time_zero' coordinate precisely by applying the
#     reverse of the offset to time-of-flight, since this may be used to compute a
#     precise time at the sample position. This is important for sample environments
#     with highly time-dependent properties, since precise event-filtering based on
#     sample environment data may be required.
#     """
#     time_offset = (
#         da.coords['time_offset'] if da.bins is None else da.bins.coords['time_offset']
#     )
#     delta = origin.time
#     if isinstance(delta, sc.DataArray):
#         # Will raise if subframes overlap, since coord for lookup table must be sorted
#         delta = sc.lookup(delta, dim='subframe')[time_offset]
#     if da.bins is not None:
#         da = da.copy(deep=False)
#         da.data = sc.bins(**da.bins.constituents)
#         da.bins.coords['tof'] = time_offset - delta
#         da.bins.coords['time_zero'] = da.bins.coords['pulse_time'] + delta.to(
#             unit=elem_unit(da.bins.coords['pulse_time']), dtype='int64'
#         )
#     else:
#         da = da.transform_coords(
#             tof=lambda time_offset: time_offset - delta, keep_inputs=False
#         )
#     if (existing := da.coords.get('Ltotal')) is not None:
#         if not sc.identical(existing, ltotal):
#             raise ValueError(
#                 f"Ltotal {existing} in data does not match Ltotal {ltotal} "
#                 "used for calculating time-of-flight."
#             )

#     da.coords['Ltotal'] = ltotal - origin.distance
#     return TofData(da)


def providers() -> tuple[Callable]:
    """
    Return the providers for the time-of-flight workflow.
    """
    return (
        chopper_cascade_frames,
        frame_at_detector,
        frame_period,
        unwrapped_time_of_arrival,
        frame_at_detector_start_time,
        unwrapped_time_of_arrival_minus_frame_start_time,
        # time_of_arrival_modulo_period,
        time_of_arrival_minus_start_time_modulo_period,
        slope_and_intercept_lookups,
        # time_of_flight_from_lookup,
        # wavelength_from_lookup,
        # time_of_flight_from_wavelength,
        time_of_flight_data,
    )


def providers_no_choppers() -> tuple[Callable]:
    """
    Return the providers for the time-of-flight workflow without choppers.
    """
    return (
        frame_period,
        time_of_arrival_modulo_period,
        unwrapped_time_of_arrival,
        time_of_flight_no_choppers,
        # time_of_flight_data,
    )


# _common_providers = (
#     frame_at_detector,
#     frame_bounds,
#     frame_period,
#     pulse_wrapped_time_offset,
#     source_chopper,
#     offset_from_wrapped,
#     unwrap_data,
# )

# _non_skipping = (frame_wrapped_time_offset,)
# _skipping = (
#     frame_wrapped_time_offset_pulse_skipping,
#     pulse_offset,
#     time_zero,
# )

# _wfm = (
#     subframe_bounds,
#     time_offset,
#     time_of_flight_origin_time_wfm_from_choppers,
#     time_of_flight_origin_distance_wfm_from_choppers,
#     wfm_choppers,
#     time_of_flight_origin_wfm,
# )
# _non_wfm = (time_of_flight_origin_from_chopper,)


# def time_of_flight_providers() -> tuple[Callable]:
#     """
#     Return the providers computing the time-of-flight and time-zero coordinates.
#     """
#     return (to_time_of_flight,)


# def time_of_flight_origin_from_choppers_providers(wfm: bool = False):
#     """
#     Return the providers for computing the time-of-flight origin via the chopper
#     cascade.

#     Parameters
#     ----------
#     wfm :
#         If True, the data is assumed to be from a WFM instrument.
#     """
#     wfm = _wfm if wfm else _non_wfm
#     _common = (source_chopper, frame_at_detector, subframe_bounds)
#     return _common + wfm


# def unwrap_providers(pulse_skipping: bool = False):
#     """
#     Return the list of providers for unwrapping frames.

#     Parameters
#     ----------
#     pulse_skipping :
#         If True, the pulse-skipping mode is assumed.
#     """
#     skipping = _skipping if pulse_skipping else _non_skipping
#     return _common_providers + skipping
