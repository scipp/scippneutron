# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from functools import partial
from itertools import chain, starmap
from typing import Generator, List, Tuple, Union

import numpy as np
import scipp as sc


def serialize_to_table(
    data: Union[sc.DataArray, sc.Dataset], *, units: bool = True
) -> Generator[Tuple[str, ...], None, None]:
    headers, columns = _select_columns(data, units=units)
    yield headers
    # Objects without ndim are scalar (e.g. str)
    if getattr(columns[0], 'ndim', 0) == 0:
        yield tuple(map(_scalar_to_str, columns))
    else:
        for row in zip(*columns):
            yield tuple(map(str, row))


def _scalar_to_str(x: Union[str, np.ndarray]) -> str:
    if isinstance(x, np.ndarray):
        return str(x[()])
    return str(x)


def _select_columns(
    data: Union[sc.DataArray, sc.Dataset], *, units: bool
) -> Tuple[Tuple[str, ...], Tuple[np.ndarray, ...]]:
    coords = data.coords
    data_map = {data.name or "data": data} if isinstance(data, sc.DataArray) else data

    return tuple(
        zip(
            *(
                col
                for cols in starmap(
                    partial(_columns_from_array, units=units),
                    chain(coords.items(), data_map.items()),
                )
                for col in cols
            )
        )
    )  # type: ignore [return-value]


def _columns_from_array(
    name: str, array: Union[sc.DataArray, sc.Variable], *, units: bool
) -> List[Tuple[str, np.ndarray]]:
    columns = []
    unit = f' [{array.unit}]' if units and array.unit is not None else ''
    columns.append((f'{name}{unit}', array.values))
    if array.variances is not None:
        columns.append((f'{name}_sigma{unit}', sc.stddevs(array).values))
    return columns
