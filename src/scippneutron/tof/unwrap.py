# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
This module provides functionality for unwrapping raw frames of neutron time-of-flight
data.

The module handles standard unwrapping, unwrapping in pulse-skipping mode, and
unwrapping for WFM instruments, as well as combinations of the latter two. The
functions defined here are meant to be used as providers for a Sciline pipeline. See
https://scipp.github.io/sciline/ on how to use Sciline.
"""

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

ChopperCascadeFrames = NewType(
    'ChopperCascadeFrames', list[chopper_cascade.FrameSequence]
)
"""
Frames of the chopper cascade.
"""

FrameAtDetector = NewType('FrameAtDetector', chopper_cascade.Frame)
"""
Result of passing the source pulse through the chopper cascade to the detector.

The detector may be a monitor or a detector after scattering off the sample. The frame
bounds are then computed from this.
"""

FramePeriod = NewType('FramePeriod', sc.Variable)
"""
The period of a frame, a (small) integer multiple of the source period.
"""

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
class TimeOfArrivalToTimeOfFlight:
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

PulsePeriod = NewType('PulsePeriod', sc.Variable)
"""
Period of the source pulses, i.e., time between consecutive pulse starts.
"""

PulseStride = NewType('PulseStride', int)
"""
Stride of used pulses. Usually 1, but may be a small integer when pulse-skipping.
"""

PulseStrideOffset = NewType('PulseStrideOffset', int)
"""
When pulse-skipping, the offset of the first pulse in the stride.
"""

RawData = NewType('RawData', sc.DataArray)
"""
Raw detector data loaded from a NeXus file, e.g., NXdetector containing NXevent_data.
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

TofData = NewType('TofData', sc.DataArray)
"""
Detector data with time-of-flight coordinate.
"""

ReHistogrammedTofData = NewType('ReHistogrammedTofData', sc.DataArray)
"""
Detector data with time-of-flight coordinate, re-histogrammed.
"""


def frame_period(pulse_period: PulsePeriod, pulse_stride: PulseStride) -> FramePeriod:
    """
    Return the period of a frame, which is defined by the pulse period times the pulse
    stride.

    Parameters
    ----------
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when
        pulse-skipping.
    """
    return FramePeriod(pulse_period * pulse_stride)


def chopper_cascade_frames(
    source_wavelength_range: SourceWavelengthRange,
    source_time_range: SourceTimeRange,
    choppers: Choppers,
    pulse_stride: PulseStride,
    pulse_period: PulsePeriod,
) -> ChopperCascadeFrames:
    """
    Return the frames of the chopper cascade.
    This is the result of propagating the source pulse through the chopper cascade.

    In the case of pulse-skipping, the frames are computed for each pulse in the stride,
    to make sure that we include cases where e.g. the first pulse in the stride is
    skipped, but the second is not.

    Parameters
    ----------
    source_wavelength_range:
        Wavelength range of the source pulse.
    source_time_range:
        Time range of the source pulse.
    choppers:
        Choppers used to define the frame parameters.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when
        pulse-skipping.
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    """
    out = []
    for i in range(pulse_stride):
        offset = (pulse_period * i).to(unit=source_time_range[0].unit, copy=False)
        frames = chopper_cascade.FrameSequence.from_source_pulse(
            time_min=source_time_range[0] + offset,
            time_max=source_time_range[-1] + offset,
            wavelength_min=source_wavelength_range[0],
            wavelength_max=source_wavelength_range[-1],
        )
        chopped = frames.chop(choppers.values())
        for f in chopped:
            for sf in f.subframes:
                sf.time -= offset.to(unit=sf.time.unit, copy=False)
        out.append(chopped)
    return ChopperCascadeFrames(out)


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
    setup correctly.

    Parameters
    ----------
    frames:
        Frames of the chopper cascade.
    ltotal:
        Total distance between the source and the detector(s).
    period:
        Period of the frame, i.e., time between the start of two consecutive frames.
    """

    # In the case of pulse-skipping, only one of the frames should have subframes (the
    # others should be empty).
    at_detector = []
    for f in frames:
        propagated = f[-1].propagate_to(ltotal)
        if len(propagated.subframes) > 0:
            at_detector.append(propagated)
    if len(at_detector) == 0:
        raise ValueError("FrameAtDetector: No frames with subframes found.")
    if len(at_detector) > 1:
        raise ValueError("FrameAtDetector: Multiple frames with subframes found.")
    at_detector = at_detector[0]

    # Check that the frame bounds do not span a range larger than the frame period.
    # This would indicate that the chopper phases are not set correctly.
    bounds = at_detector.bounds()['time']
    diff = (bounds.max('bound') - bounds.min('bound')).flatten(to='x')
    if any(diff > period.to(unit=diff.unit, copy=False)):
        raise ValueError(
            "Frames are overlapping: Computed frame bounds "
            f"{bounds} = {diff.max()} are larger than frame period {period}."
        )
    return FrameAtDetector(at_detector)


def unwrapped_time_of_arrival(
    da: RawData, offset: PulseStrideOffset, period: PulsePeriod
) -> UnwrappedTimeOfArrival:
    """
    Compute the unwrapped time of arrival of the neutron at the detector.
    For event data, this is essentially ``event_time_offset + event_time_zero``.

    Parameters
    ----------
    da:
        Raw detector data loaded from a NeXus file, e.g., NXdetector containing
        NXevent_data.
    offset:
        Integer offset of the first pulse in the stride (typically zero unless we are
        using pulse-skipping and the events do not begin with the first pulse in the
        stride).
    period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    """
    if da.bins is None:
        # Canonical name in NXmonitor
        toa = da.coords['time_of_flight']
    else:
        # To unwrap the time of arrival, we want to add the event_time_zero to the
        # event_time_offset. However, we do not really care about the exact datetimes,
        # we just want to know the offsets with respect to the start of the run.
        # Hence we use the smallest event_time_zero as the time origin.
        time_zero = da.coords['event_time_zero'] - da.coords['event_time_zero'].min()
        coord = da.bins.coords['event_time_offset']
        unit = elem_unit(coord)
        toa = (
            coord
            + time_zero.to(dtype=float, unit=unit, copy=False)
            - (offset * period).to(unit=unit, copy=False)
        )
    return UnwrappedTimeOfArrival(toa)


def frame_at_detector_start_time(frame: FrameAtDetector) -> FrameAtDetectorStartTime:
    """
    Compute the start time of the frame at the detector.

    Parameters
    ----------
    frame:
        Frame at the detector
    """
    return FrameAtDetectorStartTime(frame.bounds()['time']['bound', 0])


def unwrapped_time_of_arrival_minus_frame_start_time(
    toa: UnwrappedTimeOfArrival, start_time: FrameAtDetectorStartTime
) -> UnwrappedTimeOfArrivalMinusStartTime:
    """
    Compute the time of arrival of the neutron at the detector, unwrapped at the pulse
    period, minus the start time of the frame.
    We subtract the start time of the frame so that we can use a modulo operation to
    wrap the time of arrival at the frame period in the case of pulse-skipping.

    Parameters
    ----------
    toa:
        Time of arrival of the neutron at the detector, unwrapped at the pulse period.
    start_time:
        Time of the start of the frame at the detector.
    """
    # Order of operation to preserve dimension order
    return UnwrappedTimeOfArrivalMinusStartTime(
        -start_time.to(unit=elem_unit(toa), copy=False) + toa
    )


def time_of_arrival_minus_start_time_modulo_period(
    toa_minus_start_time: UnwrappedTimeOfArrivalMinusStartTime,
    frame_period: FramePeriod,
) -> TimeOfArrivalMinusStartTimeModuloPeriod:
    """
    Compute the time of arrival of the neutron at the detector, unwrapped at the pulse
    period, minus the start time of the frame, modulo the frame period.

    Parameters
    ----------
    toa_minus_start_time:
        Time of arrival of the neutron at the detector, unwrapped at the pulse period,
        minus the start time of the frame.
    frame_period:
        Period of the frame, i.e., time between the start of two consecutive frames.
    """
    return TimeOfArrivalMinusStartTimeModuloPeriod(
        toa_minus_start_time
        % frame_period.to(unit=elem_unit(toa_minus_start_time), copy=False)
    )


# def time_of_arrival_modulo_period(
#     toa: UnwrappedTimeOfArrival, frame_period: FramePeriod
# ) -> TimeOfArrivalModuloPeriod:
#     """
#     Compute the time of arrival of the neutron at the detector, unwrapped at the pulse
#     period, modulo the frame period.
#     This is used when there are no choppers in the beamline.

#     Parameters
#     ----------
#     toa:
#         Time of arrival of the neutron at the detector, unwrapped at the pulse period.
#     frame_period:
#         Period of the frame, i.e., time between the start of two consecutive frames.
#     """
#     return TimeOfArrivalModuloPeriod(
#         toa % frame_period.to(unit=elem_unit(toa), copy=False)
#     )


def _approximate_polygon_with_line(
    x0: sc.Variable, y0: sc.Variable, dim: str
) -> tuple[sc.Variable, sc.Variable]:
    """
    Approximate a polygon defined by the vertices of the subframe with a straight line.
    Compute the slope and intercept of the line that minimizes the integrated squared
    error over the polygon (i.e. taking the area of the polygon into account, as opposed
    to just computing a least-squares fit of the vertices).
    The method is described at
    https://mathproblems123.wordpress.com/2022/09/13/integrating-polynomials-on-polygons/

    Parameters
    ----------
    x0:
        x coordinates of the polygon vertices.
    y0:
        y coordinates of the polygon vertices.
    dim:
        Dimension along which the vertices are defined.
    """
    iv = x0.dims.index(dim)
    x1 = sc.array(dims=x0.dims, values=np.roll(x0.values, 1, axis=iv), unit=x0.unit)
    y1 = sc.array(dims=y0.dims, values=np.roll(y0.values, 1, axis=iv), unit=x0.unit)

    x0y1 = x0 * y1
    x1y0 = x1 * y0
    x0y1_x1y0 = x0y1 - x1y0

    A = ((x0y1_x1y0) / 2).sum(dim)
    x = ((x0 + x1) * (x0y1_x1y0) / 6).sum(dim)
    y = ((y0 + y1) * (x0y1_x1y0) / 6).sum(dim)
    xy = ((x0y1_x1y0) * (2 * x0 * y0 + x0y1 + x1y0 + 2 * x1 * y1) / 24).sum(dim)
    xx = ((x0y1_x1y0) * (x0**2 + x0 * x1 + x1**2) / 12).sum(dim)

    a = (xy - x * y / A) / (xx - x**2 / A)
    b = (y / A) - a * (x / A)
    return a, b


def relation_between_time_of_arrival_and_tof(
    frame: FrameAtDetector, frame_start: FrameAtDetectorStartTime, ltotal: Ltotal
) -> TimeOfArrivalToTimeOfFlight:
    """
    Compute the slope and intercept of a linear relationship between time-of-arrival
    and tof, which can be used to create lookup tables which can give the
    time-of-flight from the time-of-arrival.

    We take the polygons that define the subframes, given by the chopper cascade, and
    approximate them by straight lines.

    Parameters
    ----------
    frame:
        Frame at the detector.
    frame_start:
        Time of the start of the frame at the detector.
    ltotal:
        Total distance between the source and the detector(s).
    """
    slopes = []
    intercepts = []
    subframes = sorted(frame.subframes, key=lambda x: x.start_time.min())
    edges = []

    for sf in subframes:
        edges.extend([sf.start_time, sf.end_time])
        a, b = _approximate_polygon_with_line(
            x0=sf.time - frame_start,  # Horizontal axis is time-of-arrival
            y0=(
                ltotal * chopper_cascade.wavelength_to_inverse_velocity(sf.wavelength)
            ).to(unit=sf.time.unit, copy=False),  # Vertical axis is time-of-flight
            dim='vertex',
        )
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
    da_slope = sc.DataArray(data=data, coords={'subframe': edges})

    data = sc.full(sizes=sizes, value=np.nan, unit=sf.time.unit)
    data['subframe', ::2] = sc.concat(intercepts, 'subframe').transpose(keys)
    da_intercept = sc.DataArray(data=data, coords={'subframe': edges})

    return TimeOfArrivalToTimeOfFlight(slope=da_slope, intercept=da_intercept)


def time_of_flight_from_lookup(
    toa: TimeOfArrivalMinusStartTimeModuloPeriod,
    toa_to_tof: TimeOfArrivalToTimeOfFlight,
) -> TofCoord:
    """
    Compute the time-of-flight from the time-of-arrival.
    Lookup tables to convert time-of-arrival to time-of-flight are created internally.

    Parameters
    ----------
    toa:
        Time of arrival of the neutron at the detector, unwrapped at the pulse period,
        minus the start time of the frame, modulo the frame period.
    toa_to_tof:
        Conversion from-time-of arrival to time-of-flight.
    """
    # Ensure unit consistency
    subframe_edges = toa_to_tof.slope.coords['subframe'].to(
        unit=elem_unit(toa), copy=False
    )
    # Both slope and intercepts should have the same subframe edges
    toa_to_tof.slope.coords['subframe'] = subframe_edges
    toa_to_tof.intercept.coords['subframe'] = subframe_edges
    toa_to_tof.intercept.data = toa_to_tof.intercept.data.to(
        unit=elem_unit(toa), copy=False
    )

    slope = sc.lookup(toa_to_tof.slope, dim='subframe')[toa]
    intercept = sc.lookup(toa_to_tof.intercept, dim='subframe')[toa]
    return TofCoord(slope * toa + intercept)


def time_of_flight_data(da: RawData, tof: TofCoord) -> TofData:
    """
    Add the time-of-flight coordinate to the data.

    Parameters
    ----------
    da:
        Raw detector data loaded from a NeXus file, e.g., NXdetector containing
        NXevent_data.
    tof:
        Time-of-flight coordinate.
    """
    out = da.copy(deep=False)
    if tof.bins is not None:
        out.data = sc.bins(**out.bins.constituents)
        out.bins.coords['tof'] = tof
    else:
        out.coords['tof'] = tof
    return TofData(out)


def re_histogram_tof_data(da: TofData) -> ReHistogrammedTofData:
    """
    Histogrammed data that has been converted to `tof` will typically have
    unsorted bin edges (due to either wrapping of `time_of_flight` or wavelength
    overlap between subframes).
    This function re-histograms the data to ensure that the bin edges are sorted.
    It generates a number of events in each bin with a uniform distribution.
    The new events are then histogrammed using a set of sorted bin edges.

    WARNING:
    This function is highly experimental, has limitations and should be used with
    caution. It is a workaround to the issue that rebinning data with unsorted bin
    edges is not supported in scipp.
    We also do not support variances on the data.
    As such, this function is not part of the default set of providers, and needs to be
    inserted manually into the workflow.

    Parameters
    ----------
    da:
        TofData with the time-of-flight coordinate.
    """
    tof = da.coords['tof']
    events_per_bin = 500
    dim = uuid.uuid4().hex
    low = tof['time_of_flight', :-1].values
    high = tof['time_of_flight', 1:].values

    # The numpy.random.uniform function below does not support NaNs, so we need to
    # replace them with zeros, and then replace them back after the random numbers
    # have been generated.
    nans = np.isnan(low) | np.isnan(high)
    low = np.where(nans, 0.0, low)
    high = np.where(nans, 0.0, high)

    # In each bin, we generate a number of events with a uniform distribution.
    rng = np.random.default_rng()
    events = rng.uniform(low, high, size=(events_per_bin, *da.shape))
    events[:, nans] = np.nan
    events = sc.array(dims=[dim, *da.dims], values=events, unit=tof.unit)

    val = sc.broadcast(sc.values(da.data) / float(events_per_bin), sizes=events.sizes)
    kwargs = {'dims': events.dims, 'values': val.values, 'unit': da.data.unit}
    if da.data.variances is not None:
        kwargs['variances'] = sc.broadcast(
            sc.variances(da.data) / float(events_per_bin), sizes=events.sizes
        ).values
    data = sc.array(**kwargs)

    # Sizes of the other dimensions
    sizes = da.sizes
    del sizes['time_of_flight']

    new = sc.DataArray(
        data=data,
        coords={dim: events}
        | {
            key: sc.broadcast(sc.arange(key, size, unit=None), sizes=events.sizes)
            for key, size in sizes.items()
        },
    )

    # Define a new bin width, close to the original bin width.
    # TODO: this could be a workflow parameter
    coord = da.coords['tof']
    bin_width = (coord['time_of_flight', 1:] - coord['time_of_flight', :-1]).nanmedian()
    flat = new.flatten(to=dim)
    if sizes:
        flat = flat.group(*list(sizes.keys()))
    rehist = flat.hist({dim: bin_width}).rename({dim: 'tof'})
    for key, var in da.coords.items():
        if 'time_of_flight' not in var.dims:
            rehist.coords[key] = var
    return ReHistogrammedTofData(rehist)


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
        time_of_arrival_minus_start_time_modulo_period,
        relation_between_time_of_arrival_and_tof,
        time_of_flight_from_lookup,
        time_of_flight_data,
    )


def default_parameters():
    return {
        PulseStride: 1,
        PulseStrideOffset: 0,
    }
