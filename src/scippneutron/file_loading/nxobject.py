# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from enum import Enum, auto
import functools
import h5py

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
    def __init__(self, dataset: Dataset, loader=LoadFromHdf5()):
        self._dataset = dataset
        self._loader = loader

    def __getitem__(self, index):
        return self._loader.load_dataset_as_numpy_array(self._dataset, index)

    def __repr__(self):
        return f'<Nexus field "{self._dataset.name}">'


class NXobject:
    def __init__(self, group: h5py.Group, loader=LoadFromHdf5()):
        self._group = group
        self._loader = loader

    def make(self, group):
        nx_class = self._loader.get_string_attribute(group, 'NX_class')
        return _nx_class_registry().get(nx_class, NXobject)(group, self._loader)

    # TODO Should probably remove this and forward desired method explictly
    def __getattr__(self, name):
        return getattr(self._group, name)

    def __getitem__(self, index):
        if isinstance(index, str):
            item = self._group[index]
            if hasattr(item, 'visititems'):
                return self.make(item)
            else:
                return item
        return self._getitem(index)

    def _getitem(self, index):
        # TODO Is it better to fall back to returning h5py.Group?
        # distinguish classes not implementing _getitem, vs missing classes!
        print(f'Loading {self.NX_class} is not supported.')

    def keys(self):
        return self._group.keys()

    def values(self):
        return [
            self.make(v) if self._loader.is_group(v) else Field(v, self._loader)
            for v in self._group.values()
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
            out[NX_class[nx_class]] = {n: self.make(g) for n, g in zip(names, groups)}
        return out

    @property
    def NX_class(self):
        nx_class = self._loader.get_string_attribute(self._group, 'NX_class')
        return NX_class[nx_class]

    def __repr__(self):
        return f'<{type(self).__name__} "{self._group.name}">'


class NXroot(NXobject):
    @property
    def NX_class(self):
        # Most files violate the standard and do not define NX_class on file root
        return 'NXroot'


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
