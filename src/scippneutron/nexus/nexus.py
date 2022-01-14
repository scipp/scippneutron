# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

from contextlib import contextmanager, AbstractContextManager
from enum import Enum, auto
import functools
import h5py

from ..file_loading._log_data import _load_log_data_from_group
from ..file_loading._monitor_data import load_monitor
from ..file_loading._hdf5_nexus import LoadFromHdf5
from ..file_loading._detector_data import _load_event_group, DetectorData


class NX_class(Enum):
    NXentry = auto()
    NXlog = auto()
    NXmonitor = auto()
    NXevent_data = auto()


class Group():
    def __init__(self, group: h5py.Group):
        self._group = group
        self._loader = LoadFromHdf5()

    def __getitem__(self, index):
        if isinstance(index, str):
            item = self._group[index]
            if hasattr(item, 'visititems'):
                return Group(item)
            else:
                return item
        if self.NX_class == NX_class.NXlog:
            name, var = _load_log_data_from_group(self._group,
                                                  self._loader,
                                                  select=index)
            da = var.value
            da.name = name
            return da
        if self.NX_class == NX_class.NXmonitor:
            return load_monitor(self._group, self._loader, select=index)
        if self.NX_class == NX_class.NXevent_data:
            detector_data = _load_event_group(self._group,
                                              self._loader,
                                              DetectorData(),
                                              quiet=False,
                                              select=index)
            data = detector_data.event_data
            data.bins.coords['id'] = data.bins.coords.pop('detector_id')
            data.bins.coords['time_offset'] = data.bins.coords.pop('tof')
            return data
        print(f'Cannot load unsupported class {self.NX_class}')

    def __getattr__(self, name):
        return getattr(self._group, name)

    @property
    def NX_class(self):
        return NX_class[self._group.attrs['NX_class'].decode('UTF-8')]

    def keys(self):
        return self._group.keys()

    @functools.lru_cache
    def by_nx_class(self):
        keys = [c.name for c in NX_class]
        classes = self._loader.find_by_nx_class(tuple(keys), self._group)
        out = {}
        for nx_class, groups in classes.items():
            names = [self._loader.get_name(group) for group in groups]
            if len(names) != len(set(names)):  # fall back to full path if duplicate
                names = [group.name for group in groups]
            out[NX_class[nx_class]] = {
                name: Group(group)
                for name, group in zip(names, groups)
            }
        return out


class File(AbstractContextManager, Group):
    def __init__(self, *args, **kwargs):
        self._file = h5py.File(*args, **kwargs)
        Group.__init__(self, self._file)

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
