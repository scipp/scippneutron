# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from typing import Tuple, Dict, List, Optional, Any, Union
import scipp as sc
import numpy as np
from ._loading_common import Group, MissingDataset, MissingAttribute

_nexus_class = "NX_class"
_nexus_units = "units"
_nexus_name = "name"
_nexus_values = "values"
_nexus_dataset = "dataset"
_nexus_group = "group"
_nexus_children = "children"
_nexus_link = "link"

_filewriter_to_supported_numpy_dtype = {
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int32,
    "int16": np.int32,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.int32,
    "uint16": np.int32,
    "uint32": np.int32,
    "uint64": np.int64
}


def _get_attribute_value(
        element: Dict,
        attribute_name: str) -> Union[str, float, int, List, None]:
    """
    attributes can be a dictionary of key-value pairs, or an array
    of dictionaries with key, value, type, etc
    """
    try:
        attributes = element["attributes"]
        try:
            return attributes[attribute_name]
        except TypeError:
            for attribute in attributes:
                if attribute[_nexus_name] == attribute_name:
                    return attribute[_nexus_values]
    except KeyError:
        pass


def _visit_nodes(root: Dict, nx_class_names: Tuple[str, ...],
                 groups_with_requested_nx_class: Dict[str, List[Group]],
                 path: List[str]):
    try:
        for child in root[_nexus_children]:
            try:
                path.append(child[_nexus_name])
            except KeyError:
                continue
            nx_class = _get_attribute_value(child, _nexus_class)
            if nx_class is not None:
                if nx_class in nx_class_names:
                    groups_with_requested_nx_class[nx_class].append(
                        Group(child, root, "/".join(path)))
            _visit_nodes(child, nx_class_names, groups_with_requested_nx_class,
                         path)
            path.pop(-1)
    except KeyError:
        pass


class LoadFromJson:
    def __init__(self, root: Dict):
        self._root = root

    def _get_child_from_group(
            self,
            group: Dict,
            name: str,
            allowed_nexus_classes: Optional[Tuple[str]] = None
    ) -> Optional[Dict]:
        """
        Returns dictionary for dataset or None if not found
        """
        if allowed_nexus_classes is None:
            allowed_nexus_classes = (_nexus_dataset, _nexus_group)
        for child in group[_nexus_children]:
            try:
                if child[_nexus_name] == name:
                    if child["type"] == _nexus_link:
                        child = self.get_object_by_path(
                            self._root, child["target"])
                    if child["type"] in allowed_nexus_classes:
                        return child
            except KeyError:
                pass

    @staticmethod
    def find_by_nx_class(nx_class_names: Tuple[str, ...],
                         root: Dict) -> Dict[str, List[Group]]:
        """
        Finds groups with requested NX_class in the subtree of root

        Returns a dictionary with NX_class name as the key and list of matching
        groups as the value
        """
        groups_with_requested_nx_class: Dict[str, List[Group]] = {
            class_name: []
            for class_name in nx_class_names
        }

        path = []
        try:
            path.append(root[_nexus_name])
        except KeyError:
            pass
        _visit_nodes(root, nx_class_names, groups_with_requested_nx_class, [])

        return groups_with_requested_nx_class

    def get_child_from_group(self, group: Dict,
                             child_name: str) -> Optional[Dict]:
        return self._get_child_from_group(group, child_name)

    def get_dataset_from_group(self, group: Dict,
                               dataset_name: str) -> Optional[Dict]:
        """
        Returns dictionary for dataset or None if not found
        """
        return self._get_child_from_group(group, dataset_name,
                                          (_nexus_dataset, ))

    def dataset_in_group(self, group: Dict,
                         dataset_name: str) -> Tuple[bool, str]:
        if self.get_dataset_from_group(group, dataset_name) is not None:
            return True, ""
        return False, (f"Unable to load data from NXevent_data "
                       f" due to missing '{dataset_name}' field\n")

    def load_dataset(self,
                     group: Dict,
                     dataset_name: str,
                     dimensions: List[str],
                     dtype: Optional[Any] = None) -> sc.Variable:
        """
        Load a dataset into a Scipp Variable
        :param group: Group containing dataset to load
        :param dataset_name: Name of the dataset to load
        :param dimensions: Dimensions for the output Variable
        :param dtype: Cast to this dtype during load,
          otherwise retain dataset dtype
        """
        dataset = self.get_dataset_from_group(group, dataset_name)
        if dataset is None:
            raise MissingDataset()

        if dtype is None:
            try:
                dtype = _filewriter_to_supported_numpy_dtype[
                    dataset[_nexus_dataset]["type"]]
            except KeyError:
                dtype = _filewriter_to_supported_numpy_dtype[
                    dataset[_nexus_dataset]["dtype"]]

        units = _get_attribute_value(dataset, _nexus_units)
        if units is None:
            units = sc.units.dimensionless

        if isinstance(dataset[_nexus_values], list):
            return sc.Variable(dims=dimensions,
                               values=dataset[_nexus_values],
                               dtype=dtype,
                               unit=units)

        raise NotImplementedError("Loading scalar datasets not implemented")

    def load_dataset_from_group_as_numpy_array(self, group: Dict,
                                               dataset_name: str):
        """
        Load a dataset into a numpy array
        :param group: Group containing dataset to load
        :param dataset_name: Name of the dataset to load
        """
        dataset = self.get_dataset_from_group(group, dataset_name)
        if dataset is None:
            raise MissingDataset()
        return self.load_dataset_as_numpy_array(dataset)

    @staticmethod
    def load_dataset_as_numpy_array(dataset: Dict):
        """
        Load a dataset into a numpy array
        :param dataset: The dataset to load values from
        """
        try:
            dtype = _filewriter_to_supported_numpy_dtype[
                dataset[_nexus_dataset]["type"]]
        except KeyError:
            dtype = _filewriter_to_supported_numpy_dtype[
                dataset[_nexus_dataset]["dtype"]]
        return np.array(dataset[_nexus_values]).astype(dtype)

    def get_dataset_numpy_dtype(self, group: Dict, dataset_name: str) -> Any:
        dataset = self.get_dataset_from_group(group, dataset_name)
        return _filewriter_to_supported_numpy_dtype[dataset[_nexus_dataset]
                                                    ["type"]]

    @staticmethod
    def get_name(group: Dict) -> str:
        return group[_nexus_name]

    @staticmethod
    def get_unit(dataset: Dict) -> Union[str, sc.Unit]:
        unit = _get_attribute_value(dataset, _nexus_units)
        if unit is None:
            return sc.units.dimensionless
        return unit

    def load_scalar_string(self, group: Dict,
                           dataset_name: str) -> sc.Variable:
        dataset = self.get_dataset_from_group(group, dataset_name)
        if dataset is None:
            raise MissingDataset()
        return dataset[_nexus_values]

    def get_object_by_path(self, group: Dict, path_str: str) -> Dict:
        for node in filter(None, path_str.split("/")):
            group = self._get_child_from_group(group, node)
            if group is None:
                raise MissingDataset()
        return group

    @staticmethod
    def get_attribute_as_numpy_array(node: Dict,
                                     attribute_name: str) -> np.ndarray:
        attribute_value = _get_attribute_value(node, attribute_name)
        if attribute_value is None:
            raise MissingAttribute
        return np.array(attribute_value)

    @staticmethod
    def get_string_attribute(node: Dict, attribute_name: str) -> str:
        attribute_value = _get_attribute_value(node, attribute_name)
        if attribute_value is None:
            raise MissingAttribute
        return attribute_value

    @staticmethod
    def is_group(node: Any):
        try:
            return node["type"] == _nexus_group
        except KeyError:
            return False
