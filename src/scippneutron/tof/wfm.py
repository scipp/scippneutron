# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Wavelength-frame multiplication (WFM) transformations and stitching
"""
import uuid

import scipp as sc

from .frames import _tof_from_wavelength


def subframe_time_bounds_from_wavelengths(
        wavelength_min: sc.Variable, wavelength_max: sc.Variable,
        sample_position: sc.Variable, L2: sc.Variable, subframe_offset: sc.Variable,
        subframe_source_position: sc.Variable) -> sc.Variable:
    L1 = sc.norm(sample_position - subframe_source_position)
    Ltotal = L1 + L2
    dummy = uuid.uuid4().hex
    dims = [subframe_offset.dim, dummy]
    wavelength = sc.concat([wavelength_min, wavelength_max], dummy).transpose(dims)
    time_bounds = subframe_offset + _tof_from_wavelength(wavelength=wavelength,
                                                         Ltotal=Ltotal)
    return time_bounds.flatten(dims=dims, to='tof')


def stitch(da: sc.DataArray, subframe_bounds: sc.Variable,
           subframe_offset: sc.Variable) -> sc.DataArray:
    dim = 'tof'
    binned = da.bin({dim: subframe_bounds})
    binned.bins.coords[dim][dim, ::2] -= subframe_offset.rename(subframe=dim)
    del binned.coords[dim]
    return binned[dim, ::2].bins.concat(dim)
