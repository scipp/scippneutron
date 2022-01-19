# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

from contextlib import contextmanager, AbstractContextManager
import h5py

from ..file_loading.nxobject import NX_class, NXroot


class File(AbstractContextManager, NXroot):
    def __init__(self, *args, **kwargs):
        self._file = h5py.File(*args, **kwargs)
        NXroot.__init__(self, self._file)

    def __exit__(self, exc_type, exc_value, traceback):
        self._group.close()
