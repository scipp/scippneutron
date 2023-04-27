# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Wavelength-frame multiplication (WFM) transformations and stitching
"""
import uuid

import scipp as sc

from ..conversion.graph import beamline
from .frames import _tof_from_wavelength


def _subframe_time_bounds_from_wavelengths(
    *,
    Lopen: sc.Variable,
    Lclose: sc.Variable,
    wavelength_min: sc.Variable,
    wavelength_max: sc.Variable,
    subframe_offset: sc.Variable,
) -> sc.Variable:
    """Compute bounds of subframes for cutting data."""
    dummy = uuid.uuid4().hex
    time_open = _tof_from_wavelength(wavelength=wavelength_min, Ltotal=Lopen)
    time_close = _tof_from_wavelength(wavelength=wavelength_max, Ltotal=Lclose)
    dims = time_open.dims + (dummy,)
    time_bounds = subframe_offset + sc.concat([time_open, time_close], dummy).transpose(
        dims
    )
    dims = [subframe_offset.dim, dummy]

    return time_bounds.flatten(dims=dims, to='tof')


def _incident_beam_from_subframe_source_position(
    subframe_source_position: sc.Variable, sample_position: sc.Variable
) -> sc.Variable:
    """Vector from subframe source (such as WFM chopper) to sample"""
    return sample_position - subframe_source_position


def _Ltotal(da, position):
    graph = beamline.beamline(scatter=True)
    graph['incident_beam'] = _incident_beam_from_subframe_source_position
    da = da.copy(deep=False)
    da.coords['subframe_source_position'] = position
    da = da.transform_coords('Ltotal', graph=graph)
    return da.coords['Ltotal']


def cut_and_offset_subframes(
    da: sc.DataArray, subframe_bounds: sc.Variable, subframe_offset: sc.Variable
) -> sc.DataArray:
    """Cut and offset subframes of event data based on known bounds and offsets."""
    dim = 'tof'
    binned = da.bin({dim: subframe_bounds})
    subframe_offset = subframe_offset.rename({subframe_offset.dim: dim})
    binned.bins.coords[dim][dim, ::2] -= subframe_offset
    del binned.coords[dim]
    return binned[dim, ::2].bins.concat(dim)


def stitch_elastic(
    da: sc.DataArray,
    *,
    wavelength_min: sc.Variable,
    wavelength_max: sc.Variable,
    subframe_begin_source_position: sc.Variable,
    subframe_end_source_position: sc.Variable,
    subframe_offset: sc.Variable,
) -> sc.DataArray:
    """
    Stitch WFM subframes of unstitched event data from elastic scattering.

    We refer to the stitched components as "subframes", as opposed to the "frames" that
    originate from different pulses.

    Parameters
    ----------
    da:
        Input event data with raw 'tof'.
    wavelength_min:
        Minimum wavelengths for each subframe.
    wavelength_max:
        Maximum wavelengths for each subframe.
    subframe_begin_source_position:
        Position of the subframe "bin" source. Typically the position of a WFM chopper,
        often the second one. Precise location depends on the concrete design of the
        chopper cascade. This is the position of the "virtual source" at which the
        *first* neutrons with ``wavelength_min`` would originate after stitching the
        frames.
    subframe_end_source_position:
        Position of the subframe "bin" source. Typically the position of a WFM chopper,
        often the first one. Precise location depends on the concrete design of the
        chopper cascade. This is the position of the "virtual source" at which the
        *last* neutrons with ``wavelength_max`` would originate after stitching the
        frames.
    subframe_offset:
        Time offset for each subframe. This is the time difference between the base time
        of the input's 'tof' coordinate and the base time for each subframe. This is
        essentially the time offset of the WFM chopper openings.

    Returns
    -------
    :
        Data with corrected 'tof' coordinate. Furthermore, the 'incident_beam'
        coordinate is set based on `subframe_source_position` such that subsequent
        calculations of flight-path-length-dependent quantities will be correct
        when using the standard conversion graphs.
    """
    Lopen = _Ltotal(da, subframe_begin_source_position)
    Lclose = _Ltotal(da, subframe_end_source_position)
    da = da.copy(deep=False)
    da.coords['Ltotal'] = 0.5 * (Lopen + Lclose)

    subframe_bounds = _subframe_time_bounds_from_wavelengths(
        Lopen=Lopen,
        Lclose=Lclose,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        subframe_offset=subframe_offset,
    )
    return cut_and_offset_subframes(
        da, subframe_bounds=subframe_bounds, subframe_offset=subframe_offset
    )
