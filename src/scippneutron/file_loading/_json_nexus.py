# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
from __future__ import annotations
from typing import Tuple, Dict, List, Any, Union
from dataclasses import dataclass
import numpy as np
from scippnexus.typing import H5Group

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

numpy_to_filewriter_type = {
    np.float32: "float32",
    np.float64: "float64",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.str_: "string",
    np.object_: "string"
}


class MissingAttribute(Exception):
    pass


def make_json_attr(name: str, value) -> dict:
    if isinstance(value, (str, bytes)):
        attr_info = {"string_size": len(value), "type": "string"}
    elif isinstance(value, float):
        attr_info = {"size": 1, "type": "float64"}
    elif isinstance(value, int):
        attr_info = {"size": 1, "type": "int64"}
    elif isinstance(value, list):
        attr_info = {"size": len(value), "type": "string"}
    else:
        attr_info = {
            "size": value.shape,
            "type": numpy_to_filewriter_type[value.dtype.type]
        }
    name_and_value = {"name": name, "values": value}
    return {**attr_info, **name_and_value}


def make_json_dataset(name: str, data) -> dict:
    if isinstance(data, (str, bytes)):
        dataset_info = {"string_size": len(data), "type": "string"}
    elif isinstance(data, float):
        dataset_info = {"size": 1, "type": "float64"}
    elif isinstance(data, int):
        dataset_info = {"size": 1, "type": "int32"}
    else:
        dataset_info = {
            "size": data.shape,
            "type": numpy_to_filewriter_type[data.dtype.type]
        }

    return {
        "type": "dataset",
        "name": name,
        "values": data,
        "dataset": dataset_info,
        "attributes": []
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


class _Node(dict):

    def __init__(self, parent: dict, name: str, file: dict, group: dict):
        super().__init__(**group)
        self.parent = parent
        self.name = name
        self.file = file


def _visit_nodes(root: Dict, group: Dict, nx_class_names: Tuple[str, ...],
                 groups_with_requested_nx_class: Dict[str,
                                                      List[H5Group]], path: List[str]):
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
                        _Node(group=child, parent=group, name="/".join(path),
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
    if not isinstance(group, JSONGroup):
        return False
    try:
        for child in group._node[_nexus_children]:
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


def _find_by_type(type_name: str, root: Dict) -> List[H5Group]:
    """
    Finds objects with the requested "type" value
    Returns a list of objects with requested type
    """

    def _visit_nodes_for_type(obj: Dict, requested_type: str,
                              objects_found: List[H5Group]):
        try:
            for child in obj[_nexus_children]:
                if child["type"] == requested_type:
                    objects_found.append(
                        _Node(group=child,
                              parent=obj,
                              name="",
                              file={_nexus_children: [obj]}))
                _visit_nodes_for_type(child, requested_type, objects_found)
        except KeyError:
            # If this object does not have "children" array then go to next
            pass

    objects_with_requested_type: List[H5Group] = []
    _visit_nodes_for_type(root, type_name, objects_with_requested_type)

    return objects_with_requested_type


class JSONTypeStringID():

    def get_cset(self):
        import h5py
        return h5py.h5t.CSET_UTF8


class JSONAttrID:

    def __init__(self):
        pass

    def get_type(self):
        return JSONTypeStringID()


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

    def __setitem__(self, name, value):
        if name in self:
            raise NotImplementedError("Replacing existing item not implemented yet.")
        attr = make_json_attr(name, value)
        self._node['attributes'].append(attr)
        return self[name]

    def __iter__(self):
        if (attrs := self._node.get('attributes')) is not None:
            if isinstance(attrs, dict):
                yield from attrs
            else:
                for item in attrs:
                    yield item[_nexus_name]

    def get(self, name: str, default=None):
        return self[name] if name in self else default

    def get_id(self, name) -> JSONAttrID:
        # TODO This is a hack that only works since this is used only for a single
        # purpose by scippnexus.NXobject
        return JSONAttrID()


class JSONNode:

    def __init__(self, node: dict, *, parent=None):
        self._file = parent.file if parent is not None else self
        self._parent = self if parent is None else parent
        self._node = node
        if parent is None or parent.name == '/':
            self._name = f'/{self._node.get(_nexus_name, "")}'
        else:
            self._name = f'{parent.name}/{self._node[_nexus_name]}'

    @property
    def attrs(self) -> JSONAttributeManager:
        return JSONAttributeManager(self._node)

    @property
    def name(self) -> str:
        return self._name

    @property
    def file(self):
        return self._file

    @property
    def parent(self):
        return self._parent


class JSONDataset(JSONNode):

    @property
    def dtype(self) -> str:
        try:
            dtype = self._node[_nexus_dataset]["type"]
        except KeyError:
            dtype = self._node[_nexus_dataset]["dtype"]
        if dtype == 'string':
            return np.dtype(str)
        return np.dtype(dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self):
        return np.asarray(self._node[_nexus_values]).shape

    def __getitem__(self, index):
        return np.asarray(self._node[_nexus_values])[index]

    def read_direct(self, buf, source_sel):
        buf[...] = self[source_sel]

    def asstr(self, **ignored):
        return self


class JSONGroup(JSONNode):

    def __contains__(self, name: str) -> bool:
        try:
            self[name]
            return True
        except KeyError:
            return False

    def keys(self) -> List[str]:
        children = self._node[_nexus_children]
        return [child[_nexus_name] for child in children if not contains_stream(child)]

    def _as_group_or_dataset(self, item, parent):
        if item['type'] == _nexus_group:
            return JSONGroup(item, parent=parent)
        else:
            return JSONDataset(item, parent=parent)

    def __getitem__(self, name: str) -> Union[JSONDataset, JSONGroup]:
        if name.startswith('/') and name.count('/') == 1:
            parent = self.file
        elif '/' in name:
            parent = self['/'.join(name.split('/')[:-1])]
        else:
            parent = self

        for child in parent._node[_nexus_children]:
            if child.get(_nexus_name) != name.split('/')[-1]:
                continue
            if child.get('type') == _nexus_link:
                return self[child["target"]]
            if child.get('type') in (_nexus_dataset, _nexus_group):
                return self._as_group_or_dataset(child, parent)

        raise KeyError(f"Unable to open object (object '{name}' doesn't exist)")

    def __iter__(self):
        yield from self.keys()

    def visititems(self, callable):

        def skip(node):
            return node['type'] == _nexus_link or contains_stream(self)

        children = [
            child[_nexus_name] for child in self._node[_nexus_children]
            if not skip(child)
        ]
        for key in children:
            item = self[key]
            callable(key, item)
            if isinstance(item, JSONGroup):
                item.visititems(callable)

    def create_dataset(self, name: str, data) -> JSONDataset:
        if name in self:
            raise NotImplementedError("Replacing existing item not implemented yet.")
        dataset = make_json_dataset(name, data)
        self._node[_nexus_children].append(dataset)
        return self[name]

    def create_group(self, name: str) -> JSONGroup:
        if name in self:
            raise NotImplementedError("Replacing existing item not implemented yet.")
        group = {"type": "group", "name": name, "children": [], "attributes": []}
        self._node[_nexus_children].append(group)
        return self[name]


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
