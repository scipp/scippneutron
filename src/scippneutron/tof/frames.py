# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from functools import partial
r"""
Coordinate transformation graphs for time-of-flight neutron scattering data.

"""
import scipp as sc
from scipp.constants import h, m_n
from .conversions import Ltotal
from ..core.conversions import _as_float_type, _elem_unit


def _tof_from_wavelength(*, wavelength, Ltotal):
    scale = sc.to_unit(m_n / h,
                       sc.units.us / _elem_unit(Ltotal) / _elem_unit(wavelength))
    return _as_float_type(Ltotal * scale, wavelength) * wavelength


def _tof_to_time_offset(*, tof, frame_length, frame_offset):
    unit = tof.unit
    frame_length = sc.to_unit(frame_length, unit)
    arrival_time_offset = sc.to_unit(frame_offset, unit) + tof
    time_offset = arrival_time_offset % frame_length
    return time_offset


def _time_offset_to_tof(*, time_offset, time_offset_pivot, tof_min, frame_length):
    """
    """
    frame_length = sc.to_unit(frame_length, _elem_unit(time_offset))
    time_offset_pivot = sc.to_unit(time_offset_pivot, _elem_unit(time_offset))
    tof_min = sc.to_unit(tof_min, _elem_unit(time_offset))
    shift = tof_min - time_offset_pivot
    tof = sc.where(time_offset >= time_offset_pivot, shift, shift + frame_length)
    tof += time_offset
    return tof


def make_frames(da, *, frame_length, frame_offset=None, lambda_min=None):
    """
    This assumes that there is a fixed frame_length, but in practice this is
    likely not the case.

    Note: This assumes elastic scattering.
    """
    if 'tof' in da.bins.meta or 'tof' in da.meta:
        raise ValueError("Coordinate 'tof' already define in input data array. "
                         "Expected input with 'time_offset' coordinate.")

    def _tof_min(*, Ltotal):
        return _tof_from_wavelength(Ltotal=Ltotal, wavelength=lambda_min)

    def _time_offset_pivot(*, Ltotal, tof_min):
        return _tof_to_time_offset(tof=tof_min,
                                   frame_length=frame_length,
                                   frame_offset=frame_offset)

    def _tof(*, time_offset, time_offset_pivot, tof_min):
        return _time_offset_to_tof(time_offset=time_offset,
                                   time_offset_pivot=time_offset_pivot,
                                   tof_min=tof_min,
                                   frame_length=frame_length)

    graph = Ltotal(scatter=True)
    graph['tof_min'] = _tof_min
    graph['time_offset_pivot'] = _time_offset_pivot
    graph['tof'] = _tof
    return da.transform_coords('tof', graph=graph)
