# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from enum import Enum, auto
import functools
import h5py

from ..file_loading._hdf5_nexus import LoadFromHdf5


class NX_class(Enum):
    NXentry = auto()
    NXlog = auto()
    NXmonitor = auto()
    NXevent_data = auto()


class NXobject:
    _registry = {}

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls._registry[cls.__name__] = cls

    def __init__(self, group: h5py.Group, loader=LoadFromHdf5()):
        self._group = group
        self._loader = loader

    @staticmethod
    def make(group, loader=LoadFromHdf5()):
        nx_class = loader.get_string_attribute(group, 'NX_class')
        return NXobject._registry.get(nx_class, NXobject)(group, loader)

    # TODO Should probably remove this and forward desired method explictly
    def __getattr__(self, name):
        return getattr(self._group, name)

    def __getitem__(self, index):
        if isinstance(index, str):
            item = self._group[index]
            if hasattr(item, 'visititems'):
                return NXobject.make(item)
            else:
                return item
        return self._getitem(index)

    def _getitem(self, index):
        # TODO Is it better to fall back to returning h5py.Group?
        # distinguish classes not implementing _getitem, vs missing classes!
        print(f'Cannot load unsupported class {self.NX_class}')

    @functools.lru_cache()
    def by_nx_class(self):
        keys = [c.name for c in NX_class]
        classes = self._loader.find_by_nx_class(tuple(keys), self._group)
        out = {}
        for nx_class, groups in classes.items():
            names = [self._loader.get_name(group) for group in groups]
            if len(names) != len(set(names)):  # fall back to full path if duplicate
                names = [group.name for group in groups]
            out[NX_class[nx_class]] = {
                n: NXobject.make(g)
                for n, g in zip(names, groups)
            }
        return out

    @property
    def NX_class(self):
        nx_class = self._loader.get_string_attribute(self._group, 'NX_class')
        return NX_class[nx_class]
