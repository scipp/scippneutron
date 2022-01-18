# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

from contextlib import contextmanager, AbstractContextManager
import h5py

from ..file_loading.nxobject import NX_class, NXobject


class File(AbstractContextManager, NXobject):
    def __init__(self, *args, **kwargs):
        # TODO how can we make this an instance of the correct subclass?
        self._file = h5py.File(*args, **kwargs)
        NXobject.__init__(self, self._file)

    def __exit__(self, exc_type, exc_value, traceback):
        self._file.close()
