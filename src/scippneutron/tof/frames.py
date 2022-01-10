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
    unit = _elem_unit(tof)
    arrival_time_offset = sc.to_unit(frame_offset, unit) + tof
    time_offset = arrival_time_offset % sc.to_unit(frame_length, unit)
    return time_offset


def _time_offset_to_tof(frame_length):
    def func(*, time_offset, time_offset_split, tof_min):
        frame_length_ = sc.to_unit(frame_length, tof_min.unit)
        shift = tof_min - time_offset_split
        tof = sc.where(time_offset >= time_offset_split, shift, shift + frame_length_)
        tof += time_offset
        return tof
    return func


def make_frames(da, *, frame_length, frame_offset, lambda_min):
    """
    This assumes that there is a fixed frame_length, but in practice this is
    likely not the case.
    """
    def _tof_min(Ltotal):
        return _tof_from_wavelength(Ltotal=Ltotal, wavelength=lambda_min)

    def _t_min(tof_min):
        return _tof_to_time_offset(tof=tof_min,
                                   frame_length=frame_length,
                                   frame_offset=frame_offset)

    graph = Ltotal(scatter=True)
    graph['tof_min'] = _tof_min
    graph['time_offset_split'] = _t_min
    graph['tof'] = _time_offset_to_tof(frame_length)
    da = da.transform_coords('tof', graph=graph)
    return da
    unit = da.meta['tof_min'].unit
    frame_length = sc.to_unit(frame_length, unit)

    time_offset_split = da.meta['time_offset_min']
    time_offset = da.bins.coords['time_offset']
    shift = da.meta['tof_min'] - time_offset_split
    tof = sc.where(time_offset >= time_offset_split, shift, shift + frame_length)
    tof += time_offset
    da.bins.coords['tof'] = tof
    return da

    #subframe = sc.concat(
    #    [sc.zeros_like(time_offset_split), time_offset_split, frame_length],
    #    'time_offset').transpose().copy()
    #    da = sc.bin(da, edges=[subframe])
    #    tof_min = da.meta['tof_min']
    #    tof_max = da.meta['tof_max']
    #    time_offset_min = da.meta['time_offset_min']
    #    time_offset_max = da.meta['time_offset_max']
    #    shift = sc.concat([
    #        tof_max - time_offset_max,
    #        sc.scalar(float('nan'), unit=tof_max.unit),
    #        tof_min - time_offset_min,
    #    ], 'time_offset')
    #    da.bins.coords['tof'] = da.bins.coords['time_offset'] + shift
    #    del da.bins.coords['time_offset']
    #    # Order does not really matter, but swapping the two contributions might lead to
    #    # better memory access patterns in follow-up operations.
    #    return da['time_offset', 2].bins.concatenate(da['time_offset', 0])
    #    #da.masks['outside_frame'] = sc.array(dims=['time_offset'], values=[False, True, False])
    #    #return sc.bin(da, edges=[sc.concat([tof_min.min(), tof_max.max()], 'tof')], erase=['time_offset'])
