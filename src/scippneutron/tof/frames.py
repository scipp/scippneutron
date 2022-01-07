# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
r"""
Coordinate transformation graphs for time-of-flight neutron scattering data.

"""
import scipp as sc
from scipp.constants import h, m_n
from .conversions import Ltotal
from ..core.conversions import _as_float_type, _elem_unit


def shift_frames(pulse_time_offset):
    # wrong!
    def pulse_time(time_zero):
        time_zero += sc.to_unit(pulse_time_offset,
                                _elem_unit(time_zero)).astype('int64')
        return time_zero

    def tof(time_offset):
        time_offset -= sc.to_unit(pulse_time_offset, _elem_unit(time_offset))
        return time_offset

    return {'pulse_time': pulse_time, 'tof': tof}


def merge_frames(da, stride):
    # wrong&incomplete
    da = da.copy(deep=False)
    time_zero = da.coords['time_zero'].copy()
    for offset in range(1, stride):
        frames = time_zero.values[offset::stride]
        base_frames = time_zero.values[0::stride][:len(frames)]
        frames[:] = base_frames
    time_offset_shift = da.coords['time_zero'] - time_zero
    time_offset = da.bins.coords['time_offset']
    time_offset += sc.to_unit(time_offset_shift, time_offset.bins.unit)
    da.coords['time_zero'] = time_zero
    return da


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
        return _tof_from_wavelength(Ltotal=Ltotal, wavelength=lambda_max)

    def _tof_max(Ltotal):
        return _tof_from_wavelength(Ltotal=Ltotal, wavelength=lambda_min)

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
    return da
