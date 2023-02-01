# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Frame transformations for time-of-flight neutron scattering data.
"""
from typing import Optional

import scipp as sc
from scipp.constants import h, m_n

from .._utils import as_float_type, elem_unit
from ..conversion.graph.beamline import Ltotal


def _tof_from_wavelength(*, wavelength: sc.Variable,
                         Ltotal: sc.Variable) -> sc.Variable:
    scale = (m_n / h).to(unit=sc.units.us / elem_unit(Ltotal) / elem_unit(wavelength))
    return as_float_type(Ltotal * scale, wavelength) * wavelength


def _tof_to_time_offset(*, tof: sc.Variable, frame_period: sc.Variable,
                        frame_offset: sc.Variable) -> sc.Variable:
    unit = elem_unit(tof)
    frame_period = frame_period.to(unit=unit)
    arrival_time_offset = frame_offset.to(unit=unit) + tof
    time_offset = arrival_time_offset % frame_period
    return time_offset


def _time_offset_to_tof(*, time_offset: sc.Variable, time_offset_pivot: sc.Variable,
                        tof_min: sc.Variable, frame_period: sc.Variable) -> sc.Variable:
    frame_period = frame_period.to(unit=elem_unit(time_offset))
    time_offset_pivot = time_offset_pivot.to(unit=elem_unit(time_offset))
    tof_min = tof_min.to(unit=elem_unit(time_offset))
    shift = tof_min - time_offset_pivot
    tof = sc.where(time_offset >= time_offset_pivot, shift, shift + frame_period)
    tof += time_offset
    return tof


def time_zero_to_pulse_offset(*, pulse_period: sc.Variable, pulse_stride: sc.Variable,
                              event_time_zero: sc.Variable,
                              first_pulse_time: sc.Variable) -> sc.Variable:
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
    pulse_index = (event_time_zero -
                   first_pulse_time) // pulse_period.to(unit=elem_unit(event_time_zero))
    return pulse_period * (pulse_index % pulse_stride)


def update_time_offset_for_pulse_skipping(*, event_time_offset: sc.Variable,
                                          pulse_offset: sc.Variable) -> sc.Variable:
    return event_time_offset + pulse_offset.to(unit=elem_unit(event_time_offset))


def pulse_to_frame(*, pulse_period: sc.Variable,
                   pulse_stride: sc.Variable) -> sc.Variable:
    return pulse_period * pulse_stride


def to_tof(*, pulse_skipping: Optional[bool] = False) -> dict:
    """
    Return a graph suitable for :py:func:`scipp.transform_coords` to convert frames
    to time-of-flight.

    The graph requires the following input nodes:

    - event_time_zero and event_time_offset as read from an NXevent_data group.
    - lambda_min, used as a proxy for defining where to split and unwrap frames.
    - pulse_period
    - pulse_period and frame_offset defining the frame structure.

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


def make_frames(da: sc.DataArray,
                *,
                pulse_period: sc.Variable,
                pulse_stride: int = 1,
                frame_offset: Optional[sc.Variable] = None,
                lambda_min: Optional[sc.Variable] = None,
                first_pulse_time: Optional[sc.Variable] = None) -> sc.DataArray:
    """
    Unwrap raw timestamps from ``NXevent_data`` into time-of-flight.

    The input data must provide the following coordinates:

    - ``event_time_offset``, as read from ``NXevent_data``
    - ``event_time_zero``, as read from ``NXevent_data``
      (only for ``pulse_stride > 1``)
    - ``Ltotal`` or coordinates that allow for computation of thereof. By default
      ``Ltotal`` may be defined including the full distance between source and sample.
      If resolution choppers are used to shape the raw pulse then this should be
      redefined to set the position of this chopper as the source, and the opening
      time offset of this chopper as the ``frame_offset``. This can be achieved, e.g.,
      by setting ``L1``, which will then be used to compute ``Ltotal``.


    Parameters
    ----------
    da:
        Input data without 'tof' coordinate.
    pulse_period:
        Pulse period, i.e., time between consecutive pulse starts. This corresponds
        to the pulses as given by ``event_time_zero`` from ``NXevent_data``.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when choppers are
        used to skip pulses.
    frame_offset:
        Offset of the frame, i.e., time the neutron are considered to be emitted,
        w.r.t., the corresponding ``event_time_zero``. This may be a small offset,
        e.g., to reference the neutron pulse shortly after the proton pulse. When a
        resolution chopper is used, the frame offset can be defined as the opening of
        that chopper, if L1 is redefined accordingly. The chopper will then act as a
        "virtual" source.
    lambda_min:
        Minimal wavelength that can pass through the chopper cascade. This is used as
        a reference point for determining where to "cut" data when assigning to frames.
    first_pulse_time:
        Time of the "first" pulse. This is required only when pulse-skipping is
        performed, i.e., with ``pulse_stride`` unequal 1. This determines which pulse
        is the first one that passes through the choppers.

    Returns
    -------
    :
        Data with 'tof' coordinate.
    """
    if 'tof' in da.bins.meta or 'tof' in da.meta:
        raise ValueError("Coordinate 'tof' already defined in input data array. "
                         "Expected input with 'event_time_offset' coordinate.")
    da = da.copy(deep=False)
    # TODO Should check if any of these exist, raise if they do
    da.attrs['pulse_period'] = pulse_period
    if pulse_stride != 1:
        da.attrs['pulse_stride'] = sc.scalar(pulse_stride)
        da.attrs['first_pulse_time'] = first_pulse_time
    if frame_offset is not None:
        da.attrs['frame_offset'] = frame_offset
    if lambda_min is not None:
        da.attrs['lambda_min'] = lambda_min
    graph = Ltotal(scatter=True)
    graph.update(to_tof(pulse_skipping=pulse_stride != 1))
    return da.transform_coords('tof', graph=graph)
