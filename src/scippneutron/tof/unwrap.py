# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Time-of-flight workflow for unwrapping the time of arrival of the neutron at the
detector.
This workflow is used to convert raw detector data with event_time_zero and
event_time_offset coordinates to data with a time-of-flight coordinate.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NewType

import numpy as np
import scipp as sc
import tof
from scipp._scipp.core import _bins_no_validate

from .._utils import elem_unit
from ..chopper import DiskChopper
from . import chopper_cascade
from .to_events import to_events

Facility = NewType('Facility', str)
"""
Facility where the experiment is performed.
"""

Choppers = NewType('Choppers', Mapping[str, DiskChopper])
"""
Choppers used to define the frame parameters.
"""

Ltotal = NewType('Ltotal', sc.Variable)
"""
Total length of the flight path from the source to the detector.
"""


@dataclass
class SimulationResults:
    """
    Results of a time-of-flight simulation used to create a lookup table.
    """

    time_of_arrival: sc.Variable
    speed: sc.Variable
    wavelength: sc.Variable
    weight: sc.Variable
    distance: sc.Variable


DistanceResolution = NewType('DistanceResolution', sc.Variable)
"""
Resolution of the distance axis in the lookup table.
"""

TimeOfFlightLookupTable = NewType('TimeOfFlightLookupTable', sc.DataArray)
"""
Lookup table giving time-of-flight as a function of distance and time of arrival.
"""

LookupTableVarianceThreshold = NewType('LookupTableVarianceThreshold', float)

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

TimeOfArrivalMinusStartTimeModuloPeriod = NewType(
    'TimeOfArrivalMinusStartTimeModuloPeriod', sc.Variable
)
"""
Time of arrival of the neutron at the detector minus the start time of the frame,
modulo the frame period.
"""

FrameFoldedTimeOfArrival = NewType('FrameFoldedTimeOfArrival', sc.Variable)


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

TofData = NewType('TofData', sc.DataArray)
"""
Detector data with time-of-flight coordinate.
"""

ReHistogrammedTofData = NewType('ReHistogrammedTofData', sc.DataArray)
"""
Detector data with time-of-flight coordinate, re-histogrammed.
"""


def pulse_period_from_source(facility: Facility) -> PulsePeriod:
    """
    Return the period of the source pulses, i.e., time between consecutive pulse starts.

    Parameters
    ----------
    facility:
        Facility where the experiment is performed (used to determine the source pulse
        parameters).
    """
    return PulsePeriod(1.0 / tof.facilities[facility].frequency)


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


def run_tof_model(
    facility: Facility,
    choppers: Choppers,
) -> SimulationResults:
    tof_choppers = [
        tof.Chopper(
            frequency=abs(ch.frequency),
            direction=tof.AntiClockwise
            if (ch.frequency.value > 0.0)
            else tof.Clockwise,
            open=ch.slit_begin,
            close=ch.slit_end,
            phase=abs(ch.phase),
            distance=ch.axle_position.fields.z,
            name=name,
        )
        for name, ch in choppers.items()
    ]
    source = tof.Source(facility=facility, neutrons=1_000_000)
    if not tof_choppers:
        events = source.data
        return SimulationResults(
            time_of_arrival=events.coords['time'],
            speed=events.coords['speed'],
            wavelength=events.coords['wavelength'],
            weight=events.data,
            distance=0.0 * sc.units.m,
        )
    model = tof.Model(source=source, choppers=tof_choppers)
    results = model.run()
    # Find name of the furthest chopper in tof_choppers
    furthest_chopper = max(tof_choppers, key=lambda c: c.distance)
    events = results[furthest_chopper.name].data.squeeze()
    events = events[~events.masks['blocked_by_others']]
    return SimulationResults(
        time_of_arrival=events.coords['toa'],
        speed=events.coords['speed'],
        wavelength=events.coords['wavelength'],
        weight=events.data,
        distance=furthest_chopper.distance,
    )


def frame_at_detector_start_time(
    facility: Facility,
    disk_choppers: Choppers,
    pulse_stride: PulseStride,
    pulse_period: PulsePeriod,
    ltotal: Ltotal,
) -> FrameAtDetectorStartTime:
    """
    Compute the start time of the frame at the detector.

    This is the result of propagating the source pulse through the chopper cascade to
    the detector. The detector may be a single-pixel monitor or a multi-pixel detector
    bank after scattering off the sample.
    The frame bounds are then computed from this.

    Parameters
    ----------
    facility:
        Facility where the experiment is performed (used to determine the source pulse
        parameters).
    disk_choppers:
        Disk choppers used to chop the pulse and define the frame parameters.
    pulse_stride:
        Stride of used pulses.
        Usually 1, but may be a small integer when pulse-skipping.
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    ltotal:
        Total length of the flight path from the source to the detector.
    """
    source_pulse_params = tof.facilities[facility]
    time = source_pulse_params.time.coords['time']
    source_time_range = time.min(), time.max()
    wavelength = source_pulse_params.wavelength.coords['wavelength']
    source_wavelength_range = wavelength.min(), wavelength.max()

    # Convert DiskChoppers to chopper_cascade.Chopper
    choppers = {
        key: chopper_cascade.Chopper.from_disk_chopper(
            chop,
            pulse_frequency=source_pulse_params.frequency,
            npulses=1,
        )
        for key, chop in disk_choppers.items()
    }

    chopper_cascade_frames = []
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
        chopper_cascade_frames.append(chopped)

    # In the case of pulse-skipping, only one of the frames should have subframes (the
    # others should be empty).
    at_detector = []
    for f in chopper_cascade_frames:
        propagated = f[-1].propagate_to(ltotal)
        if len(propagated.subframes) > 0:
            at_detector.append(propagated)
    if len(at_detector) == 0:
        raise ValueError("FrameAtDetector: No frames with subframes found.")
    if len(at_detector) > 1:
        raise ValueError("FrameAtDetector: Multiple frames with subframes found.")
    at_detector = at_detector[0]

    return FrameAtDetectorStartTime(at_detector.bounds()['time']['bound', 0])


def tof_lookup(
    simulation: SimulationResults,
    ltotal: Ltotal,
    distance_resolution: DistanceResolution,
    variance_threshold: LookupTableVarianceThreshold,
) -> TimeOfFlightLookupTable:
    simulation_distance = simulation.distance.to(unit=ltotal.unit)
    dist = ltotal - simulation_distance
    res = distance_resolution.to(unit=dist.unit)
    # Add padding to ensure that the lookup table covers the full range of distances
    # because the midpoints of the table edges are used in the 2d grid interpolator.
    min_dist, max_dist = dist.min() - res, dist.max() + res
    ndist = int(((max_dist - min_dist) / res).value) + 1
    distances = sc.linspace(
        'distance', min_dist.value, max_dist.value, ndist, unit=dist.unit
    )

    time_unit = simulation.time_of_arrival.unit
    toas = simulation.time_of_arrival + (distances / simulation.speed).to(
        unit=time_unit, copy=False
    )

    data = sc.DataArray(
        data=sc.broadcast(simulation.weight, sizes=toas.sizes).flatten(to='event'),
        coords={
            'toa': toas.flatten(to='event'),
            'wavelength': sc.broadcast(simulation.wavelength, sizes=toas.sizes).flatten(
                to='event'
            ),
            'distance': sc.broadcast(distances, sizes=toas.sizes).flatten(to='event'),
        },
    )

    # TODO: move toa resolution to workflow parameter
    binned = data.bin(distance=ndist, toa=500)
    # Weighted mean of wavelength inside each bin
    wavelength = (
        binned.bins.data * binned.bins.coords['wavelength']
    ).bins.sum() / binned.bins.sum()
    # Compute the variance of the wavelength to mask out regions with large uncertainty
    variance = (
        binned.bins.data * (binned.bins.coords['wavelength'] - wavelength) ** 2
    ).bins.sum() / binned.bins.sum()
    # wavelength.masks["uncertain"] = binned.data > sc.scalar(
    #     variance_threshold, unit=variance.data.unit
    # )

    # return wavelength, variance

    binned.coords['distance'] += simulation_distance

    # Convert wavelengths to time-of-flight
    h = sc.constants.h
    m_n = sc.constants.m_n
    velocity = (h / (wavelength * m_n)).to(unit='m/s')
    timeofflight = (sc.midpoints(binned.coords['distance'])) / velocity
    out = timeofflight.to(unit=time_unit, copy=False)
    # wavelength.masks["uncertain"] = binned.data > sc.scalar(
    #     variance_threshold, unit=variance.data.unit
    # )

    # lookup_values = lookup.data.to(unit=elem_unit(toas), copy=False).values
    mask = (
        variance.data > sc.scalar(variance_threshold, unit=variance.data.unit)
    ).values
    print(mask.sum())
    # var.masks['m'].values
    out.values[mask] = np.nan
    return TimeOfFlightLookupTable(out)


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


def time_of_arrival_folded_by_frame(
    toa: TimeOfArrivalMinusStartTimeModuloPeriod,
    start_time: FrameAtDetectorStartTime,
) -> FrameFoldedTimeOfArrival:
    """
    The time of arrival of the neutron at the detector, folded by the frame period.

    Parameters
    ----------
    toa:
        Time of arrival of the neutron at the detector, unwrapped at the pulse period,
        minus the start time of the frame, modulo the frame period.
    start_time:
        Time of the start of the frame at the detector.
    """
    return FrameFoldedTimeOfArrival(
        toa + start_time.to(unit=elem_unit(toa), copy=False)
    )


def time_of_flight_data(
    da: RawData,
    lookup: TimeOfFlightLookupTable,
    ltotal: Ltotal,
    toas: FrameFoldedTimeOfArrival,
) -> TofData:
    from scipy.interpolate import RegularGridInterpolator

    f = RegularGridInterpolator(
        (
            sc.midpoints(
                lookup.coords['toa'].to(unit=elem_unit(toas), copy=False)
            ).values,
            sc.midpoints(lookup.coords['distance']).values,
        ),
        lookup.data.to(unit=elem_unit(toas), copy=False).values.T,
        method='linear',
        bounds_error=False,
    )

    if da.bins is not None:
        ltotal = sc.bins_like(toas, ltotal).bins.constituents['data']
        toas = toas.bins.constituents['data']

    tofs = sc.array(
        dims=toas.dims, values=f((toas.values, ltotal.values)), unit=elem_unit(toas)
    )

    out = da.copy(deep=False)
    if out.bins is not None:
        parts = out.bins.constituents
        out.data = sc.bins(**parts)
        parts['data'] = tofs
        out.bins.coords['tof'] = _bins_no_validate(**parts)
    else:
        out.coords['tof'] = tofs
    return TofData(out)


def re_histogram_tof_data(da: TofData) -> ReHistogrammedTofData:
    """
    Histogrammed data that has been converted to `tof` will typically have
    unsorted bin edges (due to either wrapping of `time_of_flight` or wavelength
    overlap between subframes).
    This function re-histograms the data to ensure that the bin edges are sorted.
    It makes use of the ``to_events`` helper which generates a number of events in each
    bin with a uniform distribution. The new events are then histogrammed using a set of
    sorted bin edges.

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
    events = to_events(da.rename_dims(time_of_flight='tof'), 'event')

    # Define a new bin width, close to the original bin width.
    # TODO: this could be a workflow parameter
    coord = da.coords['tof']
    bin_width = (coord['time_of_flight', 1:] - coord['time_of_flight', :-1]).nanmedian()
    rehist = events.hist(tof=bin_width)
    for key, var in da.coords.items():
        if 'time_of_flight' not in var.dims:
            rehist.coords[key] = var
    return ReHistogrammedTofData(rehist)


def providers():
    """
    Providers of the time-of-flight workflow.
    """
    return (
        pulse_period_from_source,
        frame_period,
        run_tof_model,
        frame_at_detector_start_time,
        tof_lookup,
        unwrapped_time_of_arrival,
        unwrapped_time_of_arrival_minus_frame_start_time,
        time_of_arrival_minus_start_time_modulo_period,
        time_of_arrival_folded_by_frame,
        time_of_flight_data,
    )


def params() -> dict:
    """
    Default parameters of the time-of-flight workflow.
    """
    return {
        PulseStride: 1,
        PulseStrideOffset: 0,
        DistanceResolution: sc.scalar(1.0, unit='cm'),
        LookupTableVarianceThreshold: 1.0e-3,
    }
