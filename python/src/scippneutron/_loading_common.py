# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from typing import Union, Any, List, Optional

import h5py
import numpy as np
import scipp as sc


def ensure_str(str_or_bytes: Union[str, bytes]) -> str:
    try:
        str_or_bytes = str(str_or_bytes, encoding="utf8")  # type: ignore
    except TypeError:
        pass
    return str_or_bytes


class BadSource(Exception):
    pass


unsigned_to_signed = {
    np.uint32: np.int32,
    np.uint64: np.int64,
}


def ensure_not_unsigned(dataset_type: Any):
    try:
        return unsigned_to_signed[dataset_type]
    except KeyError:
        return dataset_type


def load_dataset(dataset: h5py.Dataset,
                 dimensions: List[str],
                 dtype: Optional[Any] = None) -> sc.Variable:
    """
    Load an HDF5 dataset into a Scipp Variable
    :param dataset: The dataset to load
    :param dimensions: Dimensions for the output Variable
    :param dtype: Cast to this dtype during load,
      otherwise retain dataset dtype
    """
    if dtype is None:
        dtype = ensure_not_unsigned(dataset.dtype.type)
    variable = sc.empty(dims=dimensions,
                        shape=dataset.shape,
                        dtype=dtype,
                        unit=get_units(dataset))
    dataset.read_direct(variable.values)
    return variable


def get_units(dataset: h5py.Dataset) -> str:
    try:
        units = dataset.attrs["units"]
    except (AttributeError, KeyError):
        return ""
    return ensure_str(units)
