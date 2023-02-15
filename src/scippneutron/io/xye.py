# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import io
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipp as sc


def save_xye(fname: Union[str, Path, io.TextIOBase],
             da: sc.DataArray,
             *,
             coord: Optional[str] = None,
             header: str = '') -> None:
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
    to_save = np.c_[da.coords[coord].values, da.values, np.sqrt(da.variances)]
    np.savetxt(fname, to_save, delimiter=' ', header=header)


def load_xye(fname: Union[str, Path, io.TextIOBase],
             *,
             dim: str = 'dim_0',
             coord: Optional[str] = None,
             unit: Optional[sc.Unit] = None,
             coord_unit: Optional[sc.Unit] = None) -> sc.DataArray:
    coord = dim if coord is None else coord
    loaded = np.loadtxt(fname, delimiter=' ', unpack=True)
    if loaded.ndim == 1:
        loaded = loaded[:, np.newaxis]
    return sc.DataArray(
        sc.array(dims=[dim], values=loaded[1], variances=loaded[2]**2, unit=unit),
        coords={coord: sc.array(dims=[dim], values=loaded[0], unit=coord_unit)})
