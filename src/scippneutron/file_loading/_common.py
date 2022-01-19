# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from typing import Union, Optional, Any
import h5py
from scipp.spatial import affine_transform, linear_transform, \
    rotation, translation
import scipp as sc
import numpy as np
from typing import Dict


class BadSource(Exception):
    """
    Raise if something is wrong with data source which
    prevents it being used. Warn the user.
    """
    pass


class SkipSource(Exception):
    """
    Raise to abort using the data source, do not
    warn the user.
    """
    pass


class MissingDataset(Exception):
    pass


class MissingAttribute(Exception):
    pass


class JSONGroup(dict):
    def __init__(self, parent: dict, name: str, file: dict, group: dict):
        super().__init__(**group)
        self.parent = parent
        self.name = name
        self.file = file


Dataset = Union[h5py.Dataset, Dict]
Group = Union[h5py.Group, JSONGroup]


def _add_attr_to_loaded_data(attr_name: str,
                             data: sc.Variable,
                             value: np.ndarray,
                             unit: sc.Unit,
                             dtype: Optional[Any] = None):
    try:
        data = data.attrs
    except AttributeError:
        pass

    try:
        if dtype is not None:
            if dtype == sc.DType.vector3:
                data[attr_name] = sc.vector(value=value, unit=unit)
            elif dtype == sc.DType.affine_transform3:
                data[attr_name] = affine_transform(value=value, unit=unit)
            elif dtype == sc.DType.linear_transform3:
                data[attr_name] = linear_transform(value=value, unit=unit)
            elif dtype == sc.DType.rotation3:
                if unit != sc.units.one:
                    raise sc.UnitError(
                        f'Rotations must be dimensionless, got unit {unit}')
                data[attr_name] = rotation(value=value)
            elif dtype == sc.DType.translation3:
                data[attr_name] = translation(value=value, unit=unit)
            else:
                data[attr_name] = sc.scalar(value=value, dtype=dtype, unit=unit)
        else:
            data[attr_name] = sc.scalar(value=value, unit=unit)
    except KeyError:
        pass


def to_plain_index(dims, select, ignore_missing=False):
    """
    Given a valid "scipp" index 'select', return an equivalent plain numpy-style index.
    """
    def check_1d():
        if len(dims) != 1:
            raise ValueError(f"Dataset has multiple dimensions {dims}, "
                             "specify the dimension to index.")

    if select is Ellipsis:
        return select
    if isinstance(select, tuple) and len(select) == 0:
        return select
    if isinstance(select, tuple) and isinstance(select[0], str):
        key, sel = select
        select = {key: sel}

    if isinstance(select, tuple):
        check_1d()
        return select
    elif isinstance(select, int) or isinstance(select, slice):
        check_1d()
        return select
    elif isinstance(select, dict):
        index = [slice(None)] * len(dims)
        for key, sel in select.items():
            if not ignore_missing and key not in dims:
                raise ValueError(
                    f"'{key}' used for indexing not found in dataset dims {dims}.")
            index[dims.index(key)] = sel
        return tuple(index)
    raise ValueError("Cannot process index {select}.")
