# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from typing import Union, Any, List, Optional, Tuple, Dict

import h5py
import numpy as np
import scipp as sc
from ._loading_common import Group, MissingDataset, MissingAttribute


def _get_attr_as_str(h5_object, attribute_name: str):
    try:
        return h5_object.attrs[attribute_name].decode("utf8")
    except AttributeError:
        return h5_object.attrs[attribute_name]


def _ensure_str(str_or_bytes: Union[str, bytes]) -> str:
    try:
        str_or_bytes = str(str_or_bytes, encoding="utf8")  # type: ignore
    except TypeError:
        pass
    return str_or_bytes


_map_to_supported_type = {
    np.int8: np.int32,
    np.int16: np.int32,
    np.uint8: np.int32,
    np.uint16: np.int32,
    np.uint32: np.int32,
    np.uint64: np.int64,
}


def _ensure_supported_int_type(dataset_type: Any):
    try:
        return _map_to_supported_type[dataset_type]
    except KeyError:
        return dataset_type


class LoadFromHdf5:
    @staticmethod
    def find_by_nx_class(
            nx_class_names: Tuple[str, ...],
            root: Union[h5py.File, h5py.Group]) -> \
            Dict[str, List[Group]]:
        """
        Finds groups with requested NX_class in the subtree of root

        Returns a dictionary with NX_class name as the key and
        list of matching groups as the value
        """
        found_groups: Dict[str, List[Group]] = {
            class_name: []
            for class_name in nx_class_names
        }

        def _match_nx_class(_, h5_object):
            if isinstance(h5_object, h5py.Group):
                try:
                    nx_class = _get_attr_as_str(h5_object, "NX_class")
                    if nx_class in nx_class_names:
                        found_groups[nx_class].append(
                            Group(h5_object, h5_object.parent, h5_object.name))
                except KeyError:
                    pass

        root.visititems(_match_nx_class)
        # Also check if root itself is an NX_class
        _match_nx_class(None, root)
        return found_groups

    @staticmethod
    def dataset_in_group(group: h5py.Group,
                         dataset_name: str) -> Tuple[bool, str]:
        if dataset_name not in group:
            return False, (f"Unable to load data from NXevent_data "
                           f"at '{group.name}' due to missing '{dataset_name}'"
                           f" field\n")
        return True, ""

    def load_dataset(self,
                     group: h5py.Group,
                     dataset_name: str,
                     dimensions: List[str],
                     dtype: Optional[Any] = None) -> sc.Variable:
        """
        Load an HDF5 dataset into a Scipp Variable
        :param group: Group containing dataset to load
        :param dataset_name: Name of the dataset to load
        :param dimensions: Dimensions for the output Variable
        :param dtype: Cast to this dtype during load,
          otherwise retain dataset dtype
        """
        try:
            dataset = group[dataset_name]
        except KeyError:
            raise MissingDataset()
        if dtype is None:
            dtype = _ensure_supported_int_type(dataset.dtype.type)
        variable = sc.empty(dims=dimensions,
                            shape=dataset.shape,
                            dtype=dtype,
                            unit=self.get_unit(dataset))
        dataset.read_direct(variable.values)
        return variable

    def load_dataset_from_group_as_numpy_array(self, group: h5py.Group,
                                               dataset_name: str):
        """
        Load a dataset into a numpy array
        Prefer use of load_dataset to load directly to a scipp variable,
        this function should only be used in rare cases that a
        numpy array is required.
        :param group: Group containing dataset to load
        :param dataset_name: Name of the dataset to load
        """
        try:
            dataset = group[dataset_name]
        except KeyError:
            raise MissingDataset()
        return self.load_dataset_as_numpy_array(dataset)

    @staticmethod
    def load_dataset_as_numpy_array(dataset: h5py.Dataset):
        """
        Load a dataset into a numpy array
        Prefer use of load_dataset to load directly to a scipp variable,
        this function should only be used in rare cases that a
        numpy array is required.
        :param dataset: The dataset to load values from
        """
        return dataset[...].astype(
            _ensure_supported_int_type(dataset.dtype.type))

    @staticmethod
    def get_dataset_numpy_dtype(group: h5py.Group, dataset_name: str) -> Any:
        return _ensure_supported_int_type(group[dataset_name].dtype.type)

    @staticmethod
    def get_name(group: Union[h5py.Group, h5py.Dataset]) -> str:
        """
        Just the name of this group, not the full path
        """
        return group.name.split("/")[-1]

    @staticmethod
    def get_unit(node: Union[h5py.Dataset, h5py.Group]) -> Union[str, sc.Unit]:
        try:
            units = node.attrs["units"]
        except (AttributeError, KeyError):
            return sc.units.dimensionless
        return _ensure_str(units)

    @staticmethod
    def get_child_from_group(
            group: Dict,
            child_name: str) -> Union[h5py.Dataset, h5py.Group, None]:
        try:
            return group[child_name]
        except KeyError:
            return None

    def get_dataset_from_group(self, group: h5py.Group,
                               dataset_name: str) -> Optional[h5py.Dataset]:
        dataset = self.get_child_from_group(group, dataset_name)
        if isinstance(dataset, h5py.Dataset):
            return dataset
        return None

    @staticmethod
    def load_scalar_string(group: h5py.Group, dataset_name: str) -> str:
        try:
            return _ensure_str(group[dataset_name][...].item())
        except KeyError:
            raise MissingDataset

    @staticmethod
    def get_object_by_path(group: Union[h5py.Group, h5py.File],
                           path: str) -> h5py.Dataset:
        try:
            return group[path]
        except KeyError:
            raise MissingDataset

    @staticmethod
    def get_attribute_as_numpy_array(node: Union[h5py.Group, h5py.Dataset],
                                     attribute_name: str) -> np.ndarray:
        try:
            return node.attrs[attribute_name]
        except KeyError:
            raise MissingAttribute

    @staticmethod
    def get_string_attribute(node: Union[h5py.Group, h5py.Dataset],
                             attribute_name: str) -> str:
        try:
            return _ensure_str(node.attrs[attribute_name])
        except KeyError:
            raise MissingAttribute

    @staticmethod
    def is_group(node: Any):
        return isinstance(node, h5py.Group)
