# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc
from ._common import to_plain_index
from .nxobject import NXobject, ScippIndex


class NXdisk_chopper(NXobject):
    @property
    def shape(self):
        return []

    @property
    def dims(self):
        return []

    def _getitem(self, select: ScippIndex) -> sc.Dataset:
        index = to_plain_index([], select)
        if index != tuple():
            raise ValueError("Cannot select slice when loading NXdisk_chopper")
        ds = sc.Dataset()
        for name in ['distance', 'rotation_speed']:
            if (field := self.get(name)) is not None:
                ds[name] = field[()]
        return ds
