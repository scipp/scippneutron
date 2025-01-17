# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
# @author Neil Vaytet
"""
Time-of-flight workflow for unwrapping the time of arrival of the neutron at the
detector.
This workflow is used to convert raw detector data with event_time_zero and
event_time_offset coordinates to data with a time-of-flight coordinate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from typing import Any, NewType

import numpy as np
import scipp as sc
from scipp._scipp.core import _bins_no_validate

from .._utils import elem_unit
from ..chopper import DiskChopper
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

SimulationSeed = NewType('SimulationSeed', int)
"""
Seed for the random number generator used in the simulation.
"""


NumberOfNeutrons = NewType('NumberOfNeutrons', int)
"""
Number of neutrons to use in the simulation.
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


@dataclass
class FastestNeutron:
    """
    Properties of the fastest neutron in the simulation results.
    """

    time_of_arrival: sc.Variable
    speed: sc.Variable
    distance: sc.Variable


LtotalRange = NewType('LtotalRange', tuple[sc.Variable, sc.Variable])
"""
Range (min, max) of the total length of the flight path from the source to the detector.
"""


DistanceResolution = NewType('DistanceResolution', sc.Variable)
"""
Resolution of the distance axis in the lookup table.
"""

TimeOfArrivalResolution = NewType('TimeOfArrivalResolution', int | sc.Variable)
"""
Resolution of the time of arrival axis in the lookup table.
Can be an integer (number of bins) or a sc.Variable (bin width).
"""

TimeOfFlightLookupTable = NewType('TimeOfFlightLookupTable', sc.DataArray)
"""
Lookup table giving time-of-flight as a function of distance and time of arrival.
"""

MaskedTimeOfFlightLookupTable = NewType('MaskedTimeOfFlightLookupTable', sc.DataArray)
"""
Lookup table giving time-of-flight as a function of distance and time of arrival, with
regions of large uncertainty masked out.
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

PivotTimeAtDetector = NewType('PivotTimeAtDetector', sc.Variable)
"""
Pivot time at the detector, i.e., the time of the start of the frame at the detector.
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
    facilities = {"ess": sc.scalar(14.0, unit='Hz')}
    return PulsePeriod(1.0 / facilities[facility])


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


def extract_ltotal(da: RawData) -> Ltotal:
    """
    Extract the total length of the flight path from the source to the detector from the
    detector data.

    Parameters
    ----------
    da:
        Raw detector data loaded from a NeXus file, e.g., NXdetector containing
        NXevent_data.
    """
    return Ltotal(da.coords["Ltotal"])


def compute_tof_lookup_table(
    simulation: SimulationResults,
    ltotal_range: LtotalRange,
    distance_resolution: DistanceResolution,
    toa_resolution: TimeOfArrivalResolution,
) -> TimeOfFlightLookupTable:
    distance_unit = 'm'
    res = distance_resolution.to(unit=distance_unit)
    simulation_distance = simulation.distance.to(unit=distance_unit)

    # We need to bin the data below, to compute the weighted mean of the wavelength.
    # This results in data with bin edges.
    # However, the 2d interpolator expects bin centers.
    # We want to give the 2d interpolator a table that covers the requested range,
    # hence we need to extend the range by half a resolution in each direction.
    min_dist, max_dist = [
        x.to(unit=distance_unit) - simulation_distance for x in ltotal_range
    ]
    min_dist, max_dist = min_dist - 0.5 * res, max_dist + 0.5 * res

    dist_edges = sc.array(
        dims=['distance'],
        values=np.arange(
            min_dist.value, np.nextafter(max_dist.value, np.inf), res.value
        ),
        unit=distance_unit,
    )
    distances = sc.midpoints(dist_edges)

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

    binned = data.bin(distance=dist_edges, toa=toa_resolution)
    # Weighted mean of wavelength inside each bin
    wavelength = (
        binned.bins.data * binned.bins.coords['wavelength']
    ).bins.sum() / binned.bins.sum()
    # Compute the variance of the wavelength to track regions with large uncertainty
    variance = (
        binned.bins.data * (binned.bins.coords['wavelength'] - wavelength) ** 2
    ).bins.sum() / binned.bins.sum()

    # Need to add the simulation distance to the distance coordinate
    wavelength.coords['distance'] = wavelength.coords['distance'] + simulation_distance
    h = sc.constants.h
    m_n = sc.constants.m_n
    velocity = (h / (wavelength * m_n)).to(unit='m/s')
    timeofflight = (sc.midpoints(wavelength.coords['distance'])) / velocity
    out = timeofflight.to(unit=time_unit, copy=False)
    # Include the variances computed above
    out.variances = variance.values

    # Convert coordinates to midpoints
    out.coords['toa'] = sc.midpoints(out.coords['toa'])
    out.coords['distance'] = sc.midpoints(out.coords['distance'])

    return TimeOfFlightLookupTable(out)


def masked_tof_lookup_table(
    tof_lookup: TimeOfFlightLookupTable,
    variance_threshold: LookupTableVarianceThreshold,
) -> MaskedTimeOfFlightLookupTable:
    """
    Mask regions of the lookup table where the variance of the projected time-of-flight
    is larger than a given threshold.

    Parameters
    ----------
    tof_lookup:
        Lookup table giving time-of-flight as a function of distance and
        time-of-arrival.
    variance_threshold:
        Threshold for the variance of the projected time-of-flight above which regions
        are masked.
    """
    variances = sc.variances(tof_lookup.data)
    mask = variances > sc.scalar(variance_threshold, unit=variances.unit)
    out = tof_lookup.copy(deep=False)
    if mask.any():
        out.masks["uncertain"] = mask
    return MaskedTimeOfFlightLookupTable(out)


def find_fastest_neutron(simulation: SimulationResults) -> FastestNeutron:
    """
    Find the fastest neutron in the simulation results.
    """
    ind = np.argmax(simulation.speed.values)
    return FastestNeutron(
        time_of_arrival=simulation.time_of_arrival[ind],
        speed=simulation.speed[ind],
        distance=simulation.distance,
    )


def pivot_time_at_detector(
    fastest_neutron: FastestNeutron, ltotal: Ltotal
) -> PivotTimeAtDetector:
    """
    Compute the pivot time at the detector, i.e., the time of the start of the frame at
    the detector.
    The assumption here is that the fastest neutron in the simulation results is the one
    that arrives at the detector first.
    One could have an edge case where a slightly slower neutron which is born earlier
    could arrive at the detector first, but this edge case is most probably uncommon,
    and the difference in arrival times is likely to be small.

    Parameters
    ----------
    fastest_neutron:
        Properties of the fastest neutron in the simulation results.
    ltotal:
        Total length of the flight path from the source to the detector.
    """
    dist = ltotal - fastest_neutron.distance.to(unit=ltotal.unit)
    toa = fastest_neutron.time_of_arrival + (dist / fastest_neutron.speed).to(
        unit=fastest_neutron.time_of_arrival.unit, copy=False
    )
    return PivotTimeAtDetector(toa)


def unwrapped_time_of_arrival(
    da: RawData, offset: PulseStrideOffset, pulse_period: PulsePeriod
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
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    """
    if da.bins is None:
        # 'time_of_flight' is the canonical name in NXmonitor, but in some files, it
        # may be called 'tof'.
        key = next(iter(set(da.coords.keys()) & {'time_of_flight', 'tof'}))
        toa = da.coords[key]
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
            - (offset * pulse_period).to(unit=unit, copy=False)
        )
    return UnwrappedTimeOfArrival(toa)


def unwrapped_time_of_arrival_minus_frame_start_time(
    toa: UnwrappedTimeOfArrival, pivot_time: PivotTimeAtDetector
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
    pivot_time:
        Pivot time at the detector, i.e., the time of the start of the frame at the
        detector.
    """
    # Order of operation to preserve dimension order
    return UnwrappedTimeOfArrivalMinusStartTime(
        -pivot_time.to(unit=elem_unit(toa), copy=False) + toa
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
    pivot_time: PivotTimeAtDetector,
) -> FrameFoldedTimeOfArrival:
    """
    The time of arrival of the neutron at the detector, folded by the frame period.

    Parameters
    ----------
    toa:
        Time of arrival of the neutron at the detector, unwrapped at the pulse period,
        minus the start time of the frame, modulo the frame period.
    pivot_time:
        Pivot time at the detector, i.e., the time of the start of the frame at the
        detector.
    """
    return FrameFoldedTimeOfArrival(
        toa + pivot_time.to(unit=elem_unit(toa), copy=False)
    )


def time_of_flight_data(
    da: RawData,
    lookup: MaskedTimeOfFlightLookupTable,
    ltotal: Ltotal,
    toas: FrameFoldedTimeOfArrival,
) -> TofData:
    from scipy.interpolate import RegularGridInterpolator

    lookup_values = lookup.data.to(unit=elem_unit(toas), copy=False).values
    # Merge all masks into a single mask
    if lookup.masks:
        one_mask = reduce(lambda a, b: a | b, lookup.masks.values()).values
        # Set masked values to NaN
        lookup_values[one_mask] = np.nan

    f = RegularGridInterpolator(
        (
            lookup.coords['toa'].to(unit=elem_unit(toas), copy=False).values,
            lookup.coords['distance'].to(unit=ltotal.unit, copy=False).values,
        ),
        lookup_values.T,
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


def default_parameters() -> dict:
    """
    Default parameters of the time-of-flight workflow.
    """
    return {
        PulseStride: 1,
        PulseStrideOffset: 0,
        DistanceResolution: sc.scalar(1.0, unit='cm'),
        TimeOfArrivalResolution: 500,
        LookupTableVarianceThreshold: 1.0e-2,
        SimulationSeed: 1234,
        NumberOfNeutrons: 1_000_000,
    }


def _providers() -> tuple[Callable]:
    """
    Base providers of the time-of-flight workflow.
    """
    return (
        compute_tof_lookup_table,
        extract_ltotal,
        find_fastest_neutron,
        frame_period,
        masked_tof_lookup_table,
        pivot_time_at_detector,
        pulse_period_from_source,
        time_of_arrival_folded_by_frame,
        time_of_arrival_minus_start_time_modulo_period,
        time_of_flight_data,
        unwrapped_time_of_arrival,
        unwrapped_time_of_arrival_minus_frame_start_time,
    )


def standard_providers() -> tuple[Callable]:
    """
    Standard providers of the time-of-flight workflow, using the ``tof`` library to
    build the time-of-arrival to time-of-flight lookup table.
    """
    from .tof_simulation import run_tof_simulation

    return (*_providers(), run_tof_simulation)


class TofWorkflow:
    """
    Helper class to build a time-of-flight workflow and cache the expensive part of
    the computation: running the simulation and building the lookup table.
    """

    def __init__(
        self,
        choppers,
        facility,
        ltotal_range,
        pulse_stride=None,
        pulse_stride_offset=None,
        distance_resolution=None,
        toa_resolution=None,
        variance_threshold=None,
        seed=None,
        number_of_neutrons=None,
    ):
        import sciline as sl

        self.pipeline = sl.Pipeline(standard_providers())
        self.pipeline[Facility] = facility
        self.pipeline[Choppers] = choppers
        self.pipeline[LtotalRange] = ltotal_range

        params = default_parameters()
        self.pipeline[PulseStride] = pulse_stride or params[PulseStride]
        self.pipeline[PulseStrideOffset] = (
            pulse_stride_offset or params[PulseStrideOffset]
        )
        self.pipeline[DistanceResolution] = (
            distance_resolution or params[DistanceResolution]
        )
        self.pipeline[TimeOfArrivalResolution] = (
            toa_resolution or params[TimeOfArrivalResolution]
        )
        self.pipeline[LookupTableVarianceThreshold] = (
            variance_threshold or params[LookupTableVarianceThreshold]
        )
        self.pipeline[SimulationSeed] = seed or params[SimulationSeed]
        self.pipeline[NumberOfNeutrons] = number_of_neutrons or params[NumberOfNeutrons]

    def __getitem__(self, key):
        return self.pipeline[key]

    def __setitem__(self, key, value):
        self.pipeline[key] = value

    def persist(self) -> None:
        for t in (SimulationResults, MaskedTimeOfFlightLookupTable, FastestNeutron):
            self.pipeline[t] = self.pipeline.compute(t)

    def compute(self, *args, **kwargs) -> Any:
        return self.pipeline.compute(*args, **kwargs)

    def visualize(self, *args, **kwargs) -> Any:
        return self.pipeline.visualize(*args, **kwargs)

    def copy(self) -> TofWorkflow:
        out = self.__class__(choppers=None, facility=None, ltotal_range=None)
        out.pipeline = self.pipeline.copy()
        return out
