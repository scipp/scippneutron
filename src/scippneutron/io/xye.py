# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""File writer and reader for XYE files."""

import io
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipp as sc


class GenerateHeaderType:

    def __repr__(self) -> str:
        return 'GenerateHeader'


GenerateHeader = GenerateHeaderType()


def save_xye(fname: Union[str, Path, io.TextIOBase],
             da: sc.DataArray,
             *,
             coord: Optional[str] = None,
             header: Union[str, GenerateHeaderType] = GenerateHeader) -> None:
    """Write a data array to an XYE file.

    The input must be 1-dimensional, have variances, and at least one coordinate.
    It is possible to select which coordinate gets written with the ``coord`` argument.

    The data are written as an ASCII table with columns X, Y, E, where

    - X: coordinate of the data array,
    - Y: data values of the data array,
    - E: standard deviations corresponding to Y.

    This format is lossy, coordinates other than X, attributes,
    and masks are not written and the coordinate name, dimension name, and
    units of the input are lost.
    To improve the situation slightly, ``save_xye`` writes a basic header by default.
    All lines in the header are prefixed with ``#``.

    Parameters
    ----------
    fname:
        Name or file handle of the output file.
    da:
        1-dimensional data to write.
    coord:
        Coordinate name of ``da`` to write.
        Can be omitted if ``da`` has only one coordinate.
    header:
        String to write at the beginning of the file.
        A simple table header gets generated automatically by default.
        Set ``header=''`` to prevent this.

    See Also
    --------
    scippneutron.io.xye.load_xye:
        Function to load XYE files.
    """
    if da.variances is None:
        raise sc.VariancesError(
            'Cannot save data to XYE file because it has no variances.')
    if da.ndim != 1:
        raise sc.DimensionError(
            'Cannot save data to XYE file because it is not one-dimensional. '
            f'It has dimensions {da.dims}')
    if len(da.coords) == 0:
        raise ValueError('Cannot save data to XYE file because it has no coordinates.')
    if coord is None:
        if len(da.coords) > 1:
            raise ValueError(
                'Cannot deduce which coordinate to save because the data has more '
                f'than one: {list(da.coords.keys())}')
        coord = next(iter(da.coords))
    if da.coords.is_edges(coord):
        raise sc.CoordError('Cannot save data with bin-edges to XYE file.'
                            'Compute bin-centers before calling save_xye.')
    to_save = np.c_[da.coords[coord].values, da.values, np.sqrt(da.variances)]
    if header is GenerateHeader:
        header = _generate_xye_header(da, coord)
    np.savetxt(fname, to_save, delimiter=' ', header=header)


def load_xye(fname: Union[str, Path, io.TextIOBase],
             *,
             dim: str = 'dim_0',
             coord: Optional[str] = None,
             unit: Optional[sc.Unit] = None,
             coord_unit: Optional[sc.Unit] = None) -> sc.DataArray:
    """Read a data array from an XYE file.

    See :func:`scippneutron.io.xye.save_xye` for a description of the file format.

    Since XYE files are lossy, some metadata must be provided manually when calling
    this function.

    Parameters
    ----------
    fname:
        Name or file handle of the input file.
    dim:
        Dimension of the returned data.
    coord:
        Coordinate name of the returned data.
        Defaults to the value of ``dim``.
    unit:
        Unit of the returned data array.
    coord_unit:
        Unit of the coordinate of the returned data array.

    Returns
    -------
    da:
        Data array read from the file.

    See Also
    --------
    scippneutron.io.xye.save_xye:
        Function to write XYE files.
    """
    coord = dim if coord is None else coord
    loaded = np.loadtxt(fname, delimiter=' ', unpack=True)
    if loaded.ndim == 1:
        loaded = loaded[:, np.newaxis]
    return sc.DataArray(
        sc.array(dims=[dim], values=loaded[1], variances=loaded[2]**2, unit=unit),
        coords={coord: sc.array(dims=[dim], values=loaded[0], unit=coord_unit)})


def _generate_xye_header(da: sc.DataArray, coord: str) -> str:

    def format_unit(unit):
        return f'[{unit}]' if unit is not None else ''

    c = f'{coord} {format_unit(da.coords[coord].unit)}'
    y = f'Y {format_unit(da.unit)}'
    e = f'E {format_unit(da.unit)}'
    # Widths are for the default format of `0.0`
    return f'{c:22} {y:24} {e:24}'
