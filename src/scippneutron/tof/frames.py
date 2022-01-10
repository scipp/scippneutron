# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import math
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


def make_frames(da, *, frame_length, frame_offset, lambda_min, lambda_max):
    """
    This assumes that there is a fixed frame_length, but in practice this is
    likely not the case.
    """
    def _tof_min(Ltotal):
        return _tof_from_wavelength(Ltotal=Ltotal, wavelength=lambda_min)

    def _tof_max(Ltotal):
        return _tof_from_wavelength(Ltotal=Ltotal, wavelength=lambda_max)

    def _t(tof):
        return _tof_to_time_offset(tof=tof,
                                   frame_length=frame_length,
                                   frame_offset=frame_offset)

    def _t_min(tof_min):
        return _t(tof_min)

    def _t_max(tof_max):
        return _t(tof_max)

    graph = Ltotal(scatter=True)
    graph['tof_min'] = _tof_min
    graph['tof_max'] = _tof_max
    graph['time_offset_min'] = _t_min
    graph['time_offset_max'] = _t_max
    da = da.transform_coords(['time_offset_min', 'time_offset_max'], graph=graph)
    unit = da.meta['tof_max'].unit
    frame_length = sc.to_unit(frame_length, unit)
    if sc.any(da.meta['tof_max'] - da.meta['tof_min'] > frame_length).value:
        raise ValueError('frame too short')
    # Two cases:
    # 1. Both min and max fall into same frame => bin to extract
    # 2.max is in next frame, need to merge

    offset_min = da.meta['time_offset_min']
    offset_max = da.meta['time_offset_max']
    #delta = offset_max - offset_min
    # TODO sort in case max<min, which happens when target frame overlaps input frame bounary
    unwrap = False
    if (offset_max < offset_min).any().value:
        if (offset_min < offset_max).any().value:
            raise ValueError("Some but not all frames cross the source frame boundary")
        #offset_min, offset_max = offset_max, offset_min
        subframe = sc.concat(
            [sc.zeros_like(offset_min), offset_max, offset_min, frame_length],
            'time_offset').transpose().copy()
        da = sc.bin(da, edges=[subframe])
        tof_min = da.meta['tof_min']
        tof_max = da.meta['tof_max']
        time_offset_min = da.meta['time_offset_min']
        time_offset_max = da.meta['time_offset_max']
        shift = sc.concat([
            tof_max - time_offset_max,
            sc.scalar(float('nan'), unit=tof_max.unit),
            tof_min - time_offset_min,
        ], 'time_offset')
        da.bins.coords['tof'] = da.bins.coords['time_offset'] + shift
        del da.bins.coords['time_offset']
        # Order does not really matter, but swapping the two contributions might lead to
        # better memory access patterns in follow-up operations.
        return da['time_offset', 2].bins.concatenate(da['time_offset', 0])
        #da.masks['outside_frame'] = sc.array(dims=['time_offset'], values=[False, True, False])
        #return sc.bin(da, edges=[sc.concat([tof_min.min(), tof_max.max()], 'tof')], erase=['time_offset'])
