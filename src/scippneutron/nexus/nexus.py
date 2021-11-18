# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

from contextlib import contextmanager, AbstractContextManager
import h5py

from ..file_loading._log_data import _load_log_data_from_group
from ..file_loading._hdf5_nexus import LoadFromHdf5


class Dataset():
    def __init__(self, dataset: h5py.Dataset):
        self._dataset = dataset

    def __getitem__(self, index):
        if index is Ellipsis:
            return self._dataset[index]
        print(index)


class Group():
    def __init__(self, group: h5py.Group):
        self._group = group

    def __getitem__(self, index):
        if isinstance(index, str):
            item = self._group[index]
            if hasattr(item, 'visititems'):
                return Group(item)
            else:
                return Dataset(item)
        name, var = _load_log_data_from_group(self._group, LoadFromHdf5(), index)
        da = var.value
        da.name = name
        return da

    @property
    def NX_class(self):
        return self._group.attrs['NX_class']

    def keys(self):
        return self._group.keys()


class File(AbstractContextManager, Group):
    def __init__(self, *args, **kwargs):
        self._file = h5py.File(*args, **kwargs)
        Group.__init__(self, self._file)

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
