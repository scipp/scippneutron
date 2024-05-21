# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

_nexus_class = "NX_class"
_nexus_units = "units"
_nexus_name = "name"
_nexus_path = "path"
_nexus_values = "values"
_nexus_dataset = "dataset"
_nexus_config = "config"
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
    "string": np.str_,
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
    np.object_: "string",
}


class MissingAttribute(Exception):
    pass


def make_json_attr(name: str, value) -> dict:
    if isinstance(value, str | bytes):
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
            "type": numpy_to_filewriter_type[value.dtype.type],
        }
    name_and_value = {"name": name, "values": value}
    return {**attr_info, **name_and_value}


def make_json_dataset(name: str, data) -> dict:
    if isinstance(data, str | bytes):
        dataset_info = {"string_size": len(data), "type": "string"}
    elif isinstance(data, float):
        dataset_info = {"size": 1, "type": "float64"}
    elif isinstance(data, int):
        dataset_info = {"size": 1, "type": "int32"}
    else:
        dataset_info = {
            "size": data.shape,
            "type": numpy_to_filewriter_type[data.dtype.type],
        }
    return {
        'module': _nexus_dataset,
        _nexus_config: {
            **dataset_info,
            _nexus_name: name,
            _nexus_values: data,
        },
        "attributes": [],
    }


def _get_attribute_value(
    element: dict, attribute_name: str
) -> str | float | int | list:
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


def _visitnodes(root: dict):
    for child in root.get(_nexus_children, ()):
        yield child
        yield from _visitnodes(child)


def _name(node: dict):
    if _nexus_name in node:
        return node[_nexus_name]
    if _nexus_config in node:
        return node[_nexus_config][_nexus_name]
    return ''


def _is_group(node: dict):
    return _nexus_children in node


def _is_dataset(node: dict):
    return node.get('module') == _nexus_dataset


def _is_link(node: dict):
    return node.get('module') == _nexus_link


def _is_stream(node: dict):
    return 'module' in node and not (_is_dataset(node) or _is_link(node))


def contains_stream(group: JSONGroup) -> bool:
    """Return True if the group contains a stream object"""
    return (
        isinstance(group, JSONGroup)
        and _nexus_children in group._node
        and any(map(_is_stream, group._node[_nexus_children]))
    )


class JSONTypeStringID:
    def get_cset(self):
        import h5py

        return h5py.h5t.CSET_UTF8


class JSONAttrID:
    def __init__(self):
        pass

    def get_type(self):
        return JSONTypeStringID()


class JSONAttributeManager(Mapping[str, Any]):
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

    def __len__(self):
        return sum(1 for _ in self)

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
        name = _name(self._node)
        if parent is None or parent.name == '/':
            self._name = f'/{name}'
        else:
            self._name = f'{parent.name}/{name}'

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
            dtype = self._node[_nexus_config]["type"]
        except KeyError:
            if "dtype" not in self._node[_nexus_config] and isinstance(
                self._node[_nexus_config][_nexus_values], str
            ):
                dtype = 'string'
            else:
                dtype = self._node[_nexus_config]["dtype"]
        if dtype == 'string':
            return np.dtype(str)
        return np.dtype(dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self):
        return np.asarray(self._node[_nexus_config][_nexus_values]).shape

    def __getitem__(self, index):
        return np.asarray(self._node[_nexus_config][_nexus_values])[index]

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

    def keys(self) -> list[str]:
        if contains_stream(self):
            return []
        children = self._node[_nexus_children]
        return [_name(child) for child in children if not contains_stream(child)]

    def items(self) -> list[tuple[str, JSONNode]]:
        return [(key, self[key]) for key in self.keys()]

    def _as_group_or_dataset(self, item, parent):
        if _is_group(item):
            return JSONGroup(item, parent=parent)
        return JSONDataset(item, parent=parent)

    def __getitem__(self, name: str) -> JSONDataset | JSONGroup:
        if name.startswith('/') and name.count('/') == 1:
            parent = self.file
        elif '/' in name:
            parent = self['/'.join(name.split('/')[:-1])]
        else:
            parent = self

        for child in parent._node[_nexus_children]:
            if _name(child) != name.split('/')[-1]:
                continue
            if _is_link(child):
                return self[child[_nexus_config]["target"]]
            if _is_group(child) or _is_dataset(child):
                return self._as_group_or_dataset(child, parent)

        raise KeyError(f"Unable to open object (object '{name}' doesn't exist)")

    def __iter__(self):
        yield from self.keys()

    def visititems(self, callable):
        def skip(node):
            return _is_link(node) or contains_stream(self)

        children = [
            _name(child) for child in self._node[_nexus_children] if not skip(child)
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


def get_streams_info(root: dict) -> list[StreamInfo]:
    found_streams = [node for node in _visitnodes(root) if _is_stream(node)]
    streams = []
    for stream in found_streams:
        try:
            dtype = _filewriter_to_supported_numpy_dtype[stream[_nexus_config]["dtype"]]
        except KeyError:
            try:
                dtype = _filewriter_to_supported_numpy_dtype[
                    stream[_nexus_config]["type"]
                ]
            except KeyError:
                dtype = None

        units = "dimensionless"
        try:
            units = _get_attribute_value(stream, _nexus_units)
        except MissingAttribute:
            pass

        streams.append(
            StreamInfo(
                stream[_nexus_config]["topic"],
                stream["module"],
                stream[_nexus_config]["source"],
                dtype,
                units,
            )
        )
    return streams
