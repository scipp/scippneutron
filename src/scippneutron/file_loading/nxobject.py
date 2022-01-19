# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from enum import Enum, auto
import functools

from ._nexus import LoadFromNexus
from ..file_loading._hdf5_nexus import LoadFromHdf5
from ._common import Group, Dataset


class NX_class(Enum):
    NXroot = auto()
    NXentry = auto()
    NXlog = auto()
    NXmonitor = auto()
    NXevent_data = auto()


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
    def name(self):
        return self._loader.get_path(self._dataset)

    @property
    def shape(self):
        return self._loader.get_shape(self._dataset)

    @property
    def unit(self):
        # TODO Prefer to return None if no such attr, provided that scipp supports this
        return self._loader.get_unit(self._dataset)


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
        if isinstance(name, str):
            item = self._loader.get_child_from_group(self._group, name)
            if self._loader.is_group(item):
                return self._make(item)
            else:
                return Field(item, self._loader)
        return self._getitem(name)

    def _getitem(self, index):
        raise NotImplementedError(f'Loading {self.nx_class} is not supported.')

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
    def nx_class(self):
        key = self._loader.get_string_attribute(self._group, 'NX_class')
        return NX_class[key]

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
    return {
        cls.__name__: cls
        for cls in [NXroot, NXentry, NXevent_data, NXlog, NXmonitor]
    }
