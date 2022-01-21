# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc
from enum import Enum, auto
import functools
from typing import Union

from ._nexus import LoadFromNexus
from ._hdf5_nexus import LoadFromHdf5
from ._common import Group, Dataset, MissingAttribute


class NX_class(Enum):
    NXdata = auto()
    NXdetector = auto()
    NXentry = auto()
    NXevent_data = auto()
    NXlog = auto()
    NXmonitor = auto()
    NXroot = auto()


class Attrs:
    """HDF5 attributes.
    """
    def __init__(self,
                 node: Union[Dataset, Group],
                 loader: LoadFromNexus = LoadFromHdf5()):
        self._node = node
        self._loader = loader

    def __contains__(self, name):
        try:
            _ = self[name]
            return True
        except MissingAttribute:
            return False

    def __getitem__(self, name):
        attr = self._loader.get_attribute(self._node, name)
        # Is this check for string attributes sufficient? Is there a better way?
        if isinstance(attr, str) or isinstance(attr, bytes):
            return self._loader.get_string_attribute(self._node, name)
        return attr

    def get(self, name, default=None):
        return self[name] if name in self else default


class Field:
    """NeXus field.

    In HDF5 fields are represented as dataset.
    """
    def __init__(self, dataset: Dataset, loader: LoadFromNexus = LoadFromHdf5()):
        self._dataset = dataset
        self._loader = loader

    def __getitem__(self, index):
        return self._loader.load_dataset_as_numpy_array(self._dataset, index)

    def __repr__(self):
        return f'<Nexus field "{self._dataset.name}">'

    @property
    def attrs(self):
        return Attrs(self._dataset, self._loader)

    @property
    def dtype(self):
        return self._loader.get_dtype(self._dataset)

    @property
    def name(self) -> str:
        return self._loader.get_path(self._dataset)

    @property
    def shape(self):
        return self._loader.get_shape(self._dataset)

    @property
    def unit(self) -> Union[sc.Unit, None]:
        if 'units' in self.attrs:
            return sc.Unit(self._loader.get_unit(self._dataset))
        return None


class NXobject:
    """Base class for all NeXus groups.
    """
    def __init__(self, group: Group, loader: LoadFromNexus = LoadFromHdf5()):
        self._group = group
        self._loader = loader

    def _make(self, group):
        nx_class = self._loader.get_string_attribute(group, 'NX_class')
        return _nx_class_registry().get(nx_class, NXobject)(group, self._loader)

    def __getitem__(self, name):
        if name is None:
            raise KeyError("None is not a valid index")
        if isinstance(name, str):
            item = self._loader.get_child_from_group(self._group, name)
            if item is None:
                raise KeyError(f"Unable to open object (object '{name}' doesn't exist")
            if self._loader.is_group(item):
                return self._make(item)
            else:
                return Field(item, self._loader)
        return self._getitem(name)

    def _getitem(self, index):
        raise NotImplementedError(f'Loading {self.nx_class} is not supported.')

    def __contains__(self, name) -> bool:
        return self._loader.dataset_in_group(self._group, name)[0]

    @property
    def attrs(self):
        return Attrs(self._group, self._loader)

    @property
    def name(self):
        return self._loader.get_path(self._group)

    def keys(self):
        return self._loader.keys(self._group)

    def values(self):
        return [
            self._make(v) if self._loader.is_group(v) else Field(v, self._loader)
            for v in self._loader.values(self._group)
        ]

    @functools.lru_cache()
    def by_nx_class(self):
        classes = self._loader.find_by_nx_class(tuple(_nx_class_registry()),
                                                self._group)
        out = {}
        for nx_class, groups in classes.items():
            names = [self._loader.get_name(group) for group in groups]
            if len(names) != len(set(names)):  # fall back to full path if duplicate
                names = [group.name for group in groups]
            out[NX_class[nx_class]] = {n: self._make(g) for n, g in zip(names, groups)}
        return out

    @property
    def nx_class(self) -> NX_class:
        """The value of the NX_class attribute of the group.

        In case of the subclass NXroot this returns 'NXroot' even if the attribute
        is not actually set. This is support the majority of all legacy files, which
        do not have this attribute.
        """
        return NX_class[self.attrs['NX_class']]

    def __repr__(self):
        return f'<{type(self).__name__} "{self._group.name}">'


class NXroot(NXobject):
    @property
    def nx_class(self):
        # As an oversight in the NeXus standard and the reference implementation,
        # the NX_class was never set to NXroot. This applies to essentially all
        # files in existence before 2016, and files written by other implementations
        # that were inspired by the reference implementation. We thus hardcode NXroot:
        return NX_class['NXroot']


class NXentry(NXobject):
    pass


@functools.lru_cache()
def _nx_class_registry():
    from ..file_loading._monitor_data import NXmonitor
    from ..file_loading._detector_data import NXevent_data
    from ..file_loading._log_data import NXlog
    from ..file_loading.nxdata import NXdata
    from ..file_loading.nxdetector import NXdetector
    return {
        cls.__name__: cls
        for cls in
        [NXroot, NXentry, NXevent_data, NXlog, NXmonitor, NXdata, NXdetector]
    }
