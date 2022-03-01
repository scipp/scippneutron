# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from typing import Tuple, Dict, List, Optional, Any, Union
import scipp as sc
import numpy as np
from ._common import Group, JSONGroup, MissingDataset, MissingAttribute
from dataclasses import dataclass
from warnings import warn

_nexus_class = "NX_class"
_nexus_units = "units"
_nexus_name = "name"
_nexus_path = "path"
_nexus_values = "values"
_nexus_dataset = "dataset"
_nexus_group = "group"
_nexus_children = "children"
_nexus_link = "link"
_nexus_stream = "stream"

_filewriter_to_supported_numpy_dtype = {
    "float32": np.float32,
    "float64": np.float64,
    "float": np.float32,
    "double": np.float64,
    "int8": np.int32,
    "int16": np.int32,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.int32,
    "uint16": np.int32,
    "uint32": np.int32,
    "uint64": np.int64,
    "string": np.str_
}


def _get_attribute_value(element: Dict,
                         attribute_name: str) -> Union[str, float, int, List]:
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
    raise MissingAttribute


def _visit_nodes(root: Dict, nx_class_names: Tuple[str, ...],
                 groups_with_requested_nx_class: Dict[str,
                                                      List[Group]], path: List[str]):
    try:
        for child in root[_nexus_children]:
            try:
                path.append(child[_nexus_name])
            except KeyError:
                # If the object doesn't have a name it can't be a NeXus
                # class we are looking for, nor can it be a group
                # containing a NeXus class we are looking for, so skip to
                # next object
                continue
            try:
                nx_class = _get_attribute_value(child, _nexus_class)
                if nx_class in nx_class_names:
                    groups_with_requested_nx_class[nx_class].append(
                        JSONGroup(group=child,
                                  parent=root,
                                  name="/".join(path),
                                  file={_nexus_children: [root]}))
            except MissingAttribute:
                # It may be a group but not an NX_class,
                # that's fine, continue to its children
                pass
            _visit_nodes(child, nx_class_names, groups_with_requested_nx_class, path)
            path.pop(-1)
    except KeyError:
        pass


def contains_stream(group: Dict) -> bool:
    """
    Return True if the group contains a stream object
    """
    try:
        for child in group[_nexus_children]:
            try:
                if child["type"] == _nexus_stream:
                    return True
            except KeyError:
                # "type" field ought to exist, but if it does
                # not then assume it is not a stream
                pass
    except KeyError:
        # "children" field may be missing, that is okay
        # but means this this group cannot contain a stream
        pass
    return False


def _find_by_type(type_name: str, root: Dict) -> List[Group]:
    """
    Finds objects with the requested "type" value
    Returns a list of objects with requested type
    """
    def _visit_nodes_for_type(obj: Dict, requested_type: str,
                              objects_found: List[Group]):
        try:
            for child in obj[_nexus_children]:
                if child["type"] == requested_type:
                    objects_found.append(
                        JSONGroup(group=child,
                                  parent=obj,
                                  name="",
                                  file={_nexus_children: [obj]}))
                _visit_nodes_for_type(child, requested_type, objects_found)
        except KeyError:
            # If this object does not have "children" array then go to next
            pass

    objects_with_requested_type: List[Group] = []
    _visit_nodes_for_type(root, type_name, objects_with_requested_type)

    return objects_with_requested_type


class LoadFromJson:
    def __init__(self, root: Dict):
        self._root = root

    def keys(self, group: Dict):
        children = group[_nexus_children]
        return [child[_nexus_name] for child in children]

    def values(self, group: Dict):
        return group[_nexus_children]

    def _get_child_from_group(
            self,
            group: Dict,
            name: str,
            allowed_nexus_classes: Optional[Tuple[str]] = None) -> Optional[Dict]:
        """
        Returns dictionary for dataset or None if not found
        """
        print("_get_child_from_group", group.keys(), name, allowed_nexus_classes)
        if allowed_nexus_classes is None:
            allowed_nexus_classes = (_nexus_dataset, _nexus_group, _nexus_stream)
        print("_get_child_from_group", allowed_nexus_classes)
        print(group.keys())
        print(len(group[_nexus_children]))
        for child in group[_nexus_children]:
            print(child.keys(), _nexus_name, name)
            if 'children' in child:
                print(child['children'])
                print(len(child['children']))
                # child = child['children']
                for c in child['children']:
                    print("===================")
                    print(c)
                # return self._get_child_from_group(
                #     group=child, name=name, allowed_nexus_classes=allowed_nexus_classes)
            try:
                if child[_nexus_name] == name:
                    child[_nexus_path] = f"{self.get_path(group)}/{name}"
                    if child["type"] == _nexus_link:
                        child = self.get_object_by_path(self._root, child["target"])
                    if child["type"] in allowed_nexus_classes:
                        return child
            except KeyError:
                # if name or type are missing then it is
                # not what we are looking for
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
        _visit_nodes(root, nx_class_names, groups_with_requested_nx_class, path)

        return groups_with_requested_nx_class

    def get_child_from_group(self, group: Dict, child_name: str) -> Optional[Dict]:
        if '/' in child_name:
            child, remainder = child_name.split('/', maxsplit=1)
            if child == '':
                return self.get_child_from_group(group, remainder)
            return self.get_child_from_group(self.get_child_from_group(group, child),
                                             remainder)
        name = self.get_path(group).rstrip('/')
        child = self._get_child_from_group(group, child_name)
        if child is None:
            return child
        return JSONGroup(group=child,
                         parent=group,
                         name=f'{name}/{child_name}',
                         file={_nexus_children: [group]})

    def get_dataset_from_group(self, group: Dict, dataset_name: str) -> Optional[Dict]:
        """
        Returns dictionary for dataset or None if not found
        """
        return self._get_child_from_group(group, dataset_name, (_nexus_dataset, ))

    def dataset_in_group(self, group: Dict, dataset_name: str) -> Tuple[bool, str]:
        if self.get_dataset_from_group(group, dataset_name) is not None:
            return True, ""
        return False, (f"Unable to load data from NXevent_data "
                       f" due to missing '{dataset_name}' field\n")

    @staticmethod
    def supported_int_type(dataset):
        return _filewriter_to_supported_numpy_dtype[LoadFromJson.get_dtype(dataset)]

    def load_dataset(self,
                     group: Dict,
                     dataset_name: str,
                     dimensions: Optional[List[str]] = [],
                     dtype: Optional[Any] = None,
                     index=tuple()) -> sc.Variable:
        """
        Load a dataset into a Scipp Variable (array or scalar)
        :param group: Group containing dataset to load
        :param dataset_name: Name of the dataset to load
        :param dimensions: Dimensions for the output Variable. If empty, yields scalar.
        :param dtype: Cast to this dtype during load,
          otherwise retain dataset dtype
        """
        dataset = self.get_dataset_from_group(group, dataset_name)
        if dataset is None:
            raise MissingDataset()
        return self.load_dataset_direct(dataset,
                                        dimensions=dimensions,
                                        dtype=dtype,
                                        index=index)

    def load_dataset_direct(self,
                            dataset: Dict,
                            dimensions: Optional[List[str]] = [],
                            dtype: Optional[Any] = None,
                            index=tuple()) -> sc.Variable:
        """
        Same as `load_dataset` but dataset given directly instead of by group and name.
        """
        if dtype is None:
            dtype = self.supported_int_type(dataset)

        try:
            units = _get_attribute_value(dataset, _nexus_units)
            try:
                units = sc.Unit(units)
            except sc.UnitError:
                warn(f"Unrecognized unit '{units}' for value dataset "
                     f"in '{self.get_name(dataset)}'; setting unit as 'dimensionless'")
                units = sc.units.dimensionless
        except MissingAttribute:
            units = None

        return sc.array(dims=dimensions,
                        values=np.asarray(dataset[_nexus_values])[index],
                        dtype=dtype,
                        unit=units)

    def load_dataset_from_group_as_numpy_array(self,
                                               group: Dict,
                                               dataset_name: str,
                                               index=tuple()):
        """
        Load a dataset into a numpy array
        Prefer use of load_dataset to load directly to a scipp variable,
        this function should only be used in rare cases that a
        numpy array is required.
        :param group: Group containing dataset to load
        :param dataset_name: Name of the dataset to load
        """
        dataset = self.get_dataset_from_group(group, dataset_name)
        if dataset is None:
            raise MissingDataset()
        return self.load_dataset_as_numpy_array(dataset, index=index)

    @staticmethod
    def load_dataset_as_numpy_array(dataset: Dict, index=tuple()):
        """
        Load a dataset into a numpy array
        Prefer use of load_dataset to load directly to a scipp variable,
        this function should only be used in rare cases that a
        numpy array is required.
        :param dataset: The dataset to load values from
        """
        dtype = LoadFromJson.supported_int_type(dataset)
        return np.asarray(dataset[_nexus_values])[index].astype(dtype)

    def get_dataset_numpy_dtype(self, dataset: Dict) -> Any:
        return _filewriter_to_supported_numpy_dtype[dataset[_nexus_dataset]["type"]]

    @staticmethod
    def get_name(group: Dict) -> str:
        return group[_nexus_name]

    @staticmethod
    def get_path(group: Dict) -> str:
        if isinstance(group, JSONGroup):
            return group.name
        else:
            return group.get(_nexus_path, '/')

    @staticmethod
    def get_dtype(dataset: Dict) -> str:
        try:
            return dataset[_nexus_dataset]["type"]
        except KeyError:
            return dataset[_nexus_dataset]["dtype"]

    @staticmethod
    def get_shape(dataset: Dict) -> List:
        """
        The shape of the dataset
        """
        return np.asarray(dataset[_nexus_values]).shape

    @staticmethod
    def get_unit(dataset: Dict) -> str:
        try:
            unit = _get_attribute_value(dataset, _nexus_units)
        except MissingAttribute:
            unit = None
        return unit

    def load_scalar_string(self, group: Dict, dataset_name: str) -> sc.Variable:
        dataset = self.get_dataset_from_group(group, dataset_name)
        if dataset is None:
            raise MissingDataset()
        return dataset[_nexus_values]

    def get_object_by_path(self, group: Dict, path_str: str) -> Dict:
        print("get_object_by_path", group.keys(), path_str)
        for node in filter(None, path_str.split("/")):
            print("get_object_by_path", node)
            group = self._get_child_from_group(group, node)
            if group is None:
                raise MissingDataset()
        return group

    @staticmethod
    def get_attribute_as_numpy_array(node: Dict, attribute_name: str) -> np.ndarray:
        attribute_value = _get_attribute_value(node, attribute_name)
        return np.array(attribute_value)

    @staticmethod
    def get_attribute(node: Dict, attribute_name: str) -> Any:
        return _get_attribute_value(node, attribute_name)

    @staticmethod
    def get_string_attribute(node: Dict, attribute_name: str) -> str:
        return _get_attribute_value(node, attribute_name)

    @staticmethod
    def is_group(node: Any):
        try:
            return node["type"] == _nexus_group
        except KeyError:
            return False

    @staticmethod
    def contains_stream(group: Dict):
        return contains_stream(group)


@dataclass
class StreamInfo:
    topic: str
    flatbuffer_id: str
    source_name: str
    dtype: Any
    unit: str


def get_streams_info(root: Dict) -> List[StreamInfo]:
    found_streams = _find_by_type(_nexus_stream, root)
    streams = []
    for stream in found_streams:
        try:
            dtype = _filewriter_to_supported_numpy_dtype[stream["stream"]["dtype"]]
        except KeyError:
            try:
                dtype = _filewriter_to_supported_numpy_dtype[stream["stream"]["type"]]
            except KeyError:
                dtype = None

        units = "dimensionless"
        try:
            units = _get_attribute_value(stream.parent, _nexus_units)
        except MissingAttribute:
            pass

        streams.append(
            StreamInfo(stream["stream"]["topic"], stream["stream"]["writer_module"],
                       stream["stream"]["source"], dtype, units))
    return streams
