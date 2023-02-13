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


def subframe_time_bounds_from_wavelengths(Ltotal: sc.Variable,
                                          wavelength_min: sc.Variable,
                                          wavelength_max: sc.Variable,
                                          subframe_offset: sc.Variable) -> sc.Variable:
    dummy = uuid.uuid4().hex
    dims = [subframe_offset.dim, dummy]
    wavelength = sc.concat([wavelength_min, wavelength_max], dummy).transpose(dims)
    time_bounds = subframe_offset + _tof_from_wavelength(wavelength=wavelength,
                                                         Ltotal=Ltotal)
    return time_bounds.flatten(dims=dims, to='tof')


def incident_beam_from_subframe_source_position(
        subframe_source_position: sc.Variable,
        sample_position: sc.Variable) -> sc.Variable:
    """Vector from subframe source (such as WFM chopper) to sample"""
    return sample_position - subframe_source_position


def cut_and_offset_subframes(da: sc.DataArray, subframe_bounds: sc.Variable,
                             subframe_offset: sc.Variable) -> sc.DataArray:
    dim = 'tof'
    binned = da.bin({dim: subframe_bounds})
    subframe_offset = subframe_offset.rename({subframe_offset.dim: dim})
    binned.bins.coords[dim][dim, ::2] -= subframe_offset
    del binned.coords[dim]
    return binned[dim, ::2].bins.concat(dim)


def stitch(da: sc.DataArray, wavelength_min: sc.Variable, wavelength_max: sc.Variable,
           subframe_source_position: sc.Variable,
           subframe_offset: sc.Variable) -> sc.DataArray:
    da = da.copy(deep=False)
    da.coords['subframe_source_position'] = subframe_source_position
    graph = beamline.beamline(scatter=True)
    graph['incident_beam'] = incident_beam_from_subframe_source_position
    da = da.transform_coords('Ltotal', graph=graph)

    subframe_bounds = subframe_time_bounds_from_wavelengths(
        Ltotal=da.coords['Ltotal'],
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        subframe_offset=subframe_offset)
    return cut_and_offset_subframes(da,
                                    subframe_bounds=subframe_bounds,
                                    subframe_offset=subframe_offset)
