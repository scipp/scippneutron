# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

from contextlib import contextmanager, AbstractContextManager
import h5py

from ..file_loading._log_data import _load_log_data_from_group
from ..file_loading._hdf5_nexus import LoadFromHdf5
from ..file_loading._detector_data import _load_event_group, DetectorData


class Group():
    def __init__(self, group: h5py.Group):
        self._group = group

    def __getitem__(self, index):
        if isinstance(index, str):
            item = self._group[index]
            if hasattr(item, 'visititems'):
                return Group(item)
            else:
                return item
        if self.NX_class == 'NXlog':
            name, var = _load_log_data_from_group(self._group,
                                                  LoadFromHdf5(),
                                                  select=index)
            da = var.value
            da.name = name
            return da
        if self.NX_class == 'NXevent_data':
            detector_data = _load_event_group(self._group,
                                              LoadFromHdf5(),
                                              DetectorData(),
                                              quiet=False,
                                              select=index)
            data = detector_data.event_data
            data.bins.coords['id'] = data.bins.coords.pop('detector_id')
            data.bins.coords['time_offset'] = data.bins.coords.pop('tof')
            return data
        print(f'Cannot load unsupported class {self.NX_class}')

    @property
    def NX_class(self):
        return self._group.attrs['NX_class'].decode('UTF-8')

    def keys(self):
        return self._group.keys()


class File(AbstractContextManager, Group):
    def __init__(self, *args, **kwargs):
        self._file = h5py.File(*args, **kwargs)
        Group.__init__(self, self._file)

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
