# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from typing import Union, Any, List, Optional, Tuple, Dict

import h5py
import numpy as np
import scipp as sc


def _get_attr_as_str(h5_object, attribute_name: str):
    try:
        return h5_object.attrs[attribute_name].decode("utf8")
    except AttributeError:
        return h5_object.attrs[attribute_name]


def ensure_str(str_or_bytes: Union[str, bytes]) -> str:
    try:
        str_or_bytes = str(str_or_bytes, encoding="utf8")  # type: ignore
    except TypeError:
        pass
    return str_or_bytes


class BadSource(Exception):
    pass


unsigned_to_signed = {
    np.uint8: np.int8,
    np.uint16: np.int16,
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


def find_by_nx_class(
        nx_class_names: Tuple[str, ...],
        root: Union[h5py.File, h5py.Group]) -> Dict[str, List[h5py.Group]]:
    """
    Finds groups with requested NX_class in the subtree of root

    Returns a dictionary with NX_class name as the key and list of matching
    groups as the value
    """
    groups_with_requested_nx_class: Dict[str, List[h5py.Group]] = {
        class_name: []
        for class_name in nx_class_names
    }

    def _match_nx_class(_, h5_object):
        if isinstance(h5_object, h5py.Group):
            try:
                nx_class = _get_attr_as_str(h5_object, "NX_class")
                if nx_class in nx_class_names:
                    groups_with_requested_nx_class[nx_class].append(h5_object)
            except KeyError:
                pass

    root.visititems(_match_nx_class)
    # Also check if root itself is an NX_class
    _match_nx_class(None, root)
    return groups_with_requested_nx_class
