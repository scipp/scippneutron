# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from typing import Tuple, Dict, List, Optional, Any, Union
import scipp as sc
import numpy as np
from ._common import Group, JSONGroup, MissingDataset, MissingAttribute
from dataclasses import dataclass

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


def _visit_nodes(root: Dict, group: Dict, nx_class_names: Tuple[str, ...],
                 groups_with_requested_nx_class: Dict[str,
                                                      List[Group]], path: List[str]):
    try:
        for child in group[_nexus_children]:
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
                                  parent=group,
                                  name="/".join(path),
                                  file=root))
            except MissingAttribute:
                # It may be a group but not an NX_class,
                # that's fine, continue to its children
                pass
            _visit_nodes(root, child, nx_class_names, groups_with_requested_nx_class,
                         path)
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
        if allowed_nexus_classes is None:
            allowed_nexus_classes = (_nexus_dataset, _nexus_group, _nexus_stream)
        for child in group[_nexus_children]:
            try:
                if child[_nexus_name] == name:
                    path = self.get_path(group)
                    if path == '/':
                        path = ''
                    child[_nexus_path] = f"{path}/{name}"
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
        _visit_nodes(root, root, nx_class_names, groups_with_requested_nx_class, path)

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
                         file=self._root)

    def dataset_in_group(self, group: Dict, dataset_name: str) -> bool:
        return self._get_child_from_group(group, dataset_name,
                                          (_nexus_dataset, )) is not None

    def load_dataset_direct(self,
                            dataset: Dict,
                            unit: Optional[sc.Unit] = None,
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
        if dtype is None:
            dtype = _filewriter_to_supported_numpy_dtype[LoadFromJson.get_dtype(
                dataset)]

        return sc.array(dims=dimensions,
                        values=np.asarray(dataset[_nexus_values])[index],
                        dtype=dtype,
                        unit=unit)

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

    def get_object_by_path(self, group: Dict, path_str: str) -> Dict:
        for node in filter(None, path_str.split("/")):
            group = self._get_child_from_group(group, node)
            if group is None:
                raise MissingDataset()
        return group

    @staticmethod
    def is_group(node: Any):
        try:
            return node["type"] == _nexus_group
        except KeyError:
            return False

    @staticmethod
    def contains_stream(group: Dict):
        return contains_stream(group)


class JSONAttributeManager:
    def __init__(self, node: dict):
        self._node = node

    def __contains__(self, name):
        try:
            self[name]
        except MissingAttribute:
            return False
        return True

    def __getitem__(self, name):
        return _get_attribute_value(self._node, name)


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
