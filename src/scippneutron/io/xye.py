# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""File writer and reader for XYE files."""

import io
from pathlib import Path

import numpy as np
import scipp as sc

from ..logging import get_logger


class GenerateHeaderType:
    def __repr__(self) -> str:
        return 'GenerateHeader'


GenerateHeader = GenerateHeaderType()


def save_xye(
    fname: str | Path | io.TextIOBase,
    da: sc.DataArray,
    *,
    coord: str | None = None,
    header: str | GenerateHeaderType = GenerateHeader,
) -> None:
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
        If omitted and ``da`` has only one coordinate, that coordinate is used.
        If omitted and ``da`` has multiple coordinates, attempts to use a coordinate
        with the same name as ``da.dim``.
        If that does not exist, raise an error.
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
            'Cannot save data to XYE file because it has no variances.'
        )
    if da.ndim != 1:
        raise sc.DimensionError(
            'Cannot save data to XYE file because it is not one-dimensional. '
            f'It has dimensions {da.dims}'
        )
    if da.masks:
        raise ValueError(
            'Cannot save data to XYE file because it has masks: '
            f'{list(da.masks.keys())}'
        )
    if len(da.coords) == 0:
        raise ValueError('Cannot save data to XYE file because it has no coordinates.')
    coord = _deduce_coord(da) if coord is None else coord
    if da.coords.is_edges(coord):
        raise sc.CoordError(
            'Cannot save data with bin-edges to XYE file. '
            'Compute bin-centers before calling save_xye. '
            'Use, e.g., scipp.midpoints for linearly spaced bins.'
        )
    to_save = np.c_[da.coords[coord].values, da.values, np.sqrt(da.variances)]
    if header is GenerateHeader:
        header = _generate_xye_header(da, coord)

    get_logger().info(
        "Saving data with unit %s and coordinate '%s' to XYE file %s",
        da.unit,
        coord,
        fname,
    )
    np.savetxt(fname, to_save, delimiter=' ', header=header)


def load_xye(
    fname: str | Path | io.TextIOBase,
    *,
    dim: str,
    unit: sc.Unit | str | None,
    coord_unit: sc.Unit | str | None,
    coord: str | None = None,
) -> sc.DataArray:
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
    unit:
        Unit of the returned data array.
    coord_unit:
        Unit of the coordinate of the returned data array.
    coord:
        Coordinate name of the returned data.
        Defaults to the value of ``dim``.

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
        sc.array(dims=[dim], values=loaded[1], variances=loaded[2] ** 2, unit=unit),
        coords={coord: sc.array(dims=[dim], values=loaded[0], unit=coord_unit)},
    )


def _generate_xye_header(da: sc.DataArray, coord: str) -> str:
    def format_unit(unit):
        return f'[{unit}]' if unit is not None else ''

    c = f'{coord} {format_unit(da.coords[coord].unit)}'
    y = f'Y {format_unit(da.unit)}'
    e = f'E {format_unit(da.unit)}'
    # Widths are for the default format of `0.0`
    return f'{c:22} {y:24} {e:24}'


def _deduce_coord(da: sc.DataArray) -> str:
    if len(da.coords) == 1:
        return next(iter(da.coords))
    if len(da.coords) > 1 and da.dim not in da.coords:
        raise ValueError(
            'Cannot deduce which coordinate to save because the data has more '
            f'than one and no dimension-coordinate (dim={da.dim}): '
            f'{list(da.coords.keys())}'
        )
    return da.dim
