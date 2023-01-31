# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Frame transformations for time-of-flight neutron scattering data.
"""
import scipp as sc
from scipp.constants import h, m_n

from .._utils import as_float_type, elem_unit
from ..conversion.graph.beamline import Ltotal


def _tof_from_wavelength(*, wavelength, Ltotal):
    scale = (m_n / h).to(unit=sc.units.us / elem_unit(Ltotal) / elem_unit(wavelength))
    return as_float_type(Ltotal * scale, wavelength) * wavelength


def _tof_to_time_offset(*, tof, frame_period, frame_offset):
    unit = tof.unit
    frame_period = frame_period.to(unit=unit)
    arrival_time_offset = frame_offset.to(unit=unit) + tof
    time_offset = arrival_time_offset % frame_period
    return time_offset


def _time_offset_to_tof(*, time_offset, time_offset_pivot, tof_min, frame_period):
    frame_period = frame_period.to(unit=elem_unit(time_offset))
    time_offset_pivot = time_offset_pivot.to(unit=elem_unit(time_offset))
    tof_min = tof_min.to(unit=elem_unit(time_offset))
    shift = tof_min - time_offset_pivot
    tof = sc.where(time_offset >= time_offset_pivot, shift, shift + frame_period)
    tof += time_offset
    return tof


def time_zero_to_pulse_offset(*, pulse_period, pulse_stride, event_time_zero,
                              first_pulse_time):
    """
    Return 0-based source frame index of detection frame.

    The source frames containing the time marked by tof_min receive index 0.
    The frame after that index 1, and so on, until frame_stride-1.

    Example
    -------
    event_time_zero = 12:05:00
    first_pulse_time = 12:00:00  # time of first (or any) pulse that passes through
                                   choppers
    """
    # This is roughly equivalent to
    #   (event_time_zero - first_pulse_time) % frame_period
    # but should avoid some issues with precision and drift
    pulse_index = (event_time_zero - first_pulse_time) // pulse_period
    return pulse_period * (pulse_index % pulse_stride)


def update_time_offset_for_pulse_skipping(event_time_offset, pulse_offset):
    return event_time_offset + pulse_offset


def pulse_to_frame(pulse_period: sc.Variable, pulse_stride: int) -> sc.Variable:
    return pulse_period * pulse_stride


def to_tof(*, pulse_skipping: bool = False):
    """
    Return a graph suitable for :py:func:`scipp.transform_coords` to convert frames
    to time-of-flight.

    The graph requires the following input nodes:

    - event_time_zero and event_time_offset as read from an NXevent_data group.
    - lambda_min, used as a proxy for defining where to split and unwrap frames.
    - pulse_period
    - frame_length and frame_offset defining the frame structure.

    This assumes elastic scattering, or at least that the gaps between frames arriving
    at detectors are sufficiently large such that a common lambda_min definition is
    applicable.
    """

    def _tof_min(*, Ltotal, lambda_min):
        return _tof_from_wavelength(Ltotal=Ltotal, wavelength=lambda_min)

    def _time_offset_pivot(*, tof_min, frame_period, frame_offset):
        return _tof_to_time_offset(tof=tof_min,
                                   frame_period=frame_period,
                                   frame_offset=frame_offset)

    def _tof(*, event_time_offset, time_offset_pivot, tof_min, frame_period):
        return _time_offset_to_tof(time_offset=event_time_offset,
                                   time_offset_pivot=time_offset_pivot,
                                   tof_min=tof_min,
                                   frame_period=frame_period)

    graph = {}
    if pulse_skipping:
        graph['frame_period'] = pulse_to_frame
        graph['time_offset'] = update_time_offset_for_pulse_skipping
        graph['pulse_offset'] = time_zero_to_pulse_offset
    else:
        graph['frame_period'] = 'pulse_period'
        graph['time_offset'] = 'event_time_offset'

    graph['tof_min'] = _tof_min
    graph['time_offset_pivot'] = _time_offset_pivot
    graph['tof'] = _time_offset_to_tof
    return graph


def make_frames(da, *, frame_length, frame_offset=None, lambda_min=None):
    da = da.copy(deep=False)
    # TODO Should check if any of these exist, raise if they do
    da.attrs['frame_length'] = frame_length
    if frame_offset is not None:
        da.attrs['frame_offset'] = frame_offset
    if lambda_min is not None:
        da.attrs['lambda_min'] = lambda_min
    graph = Ltotal(scatter=True)
    graph.update(to_tof())
    if 'tof' in da.bins.meta or 'tof' in da.meta:
        raise ValueError("Coordinate 'tof' already defined in input data array. "
                         "Expected input with 'event_time_offset' coordinate.")
    return da.transform_coords('tof', graph=graph)
