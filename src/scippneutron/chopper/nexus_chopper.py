# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import dataclasses
from typing import Optional, Union

import scipp as sc
import scippnexus as snx

from .disk_chopper import DiskChopper, DiskChopperType


class NXDiskChopper(snx.NXdisk_chopper):
    """NeXus definition to load a DiskChopper using ScippNeXus.

    Examples
    --------
    Use as

    .. code-block:: python

        defs = {
           **snx.base_definitions(),
           'NXdisk_chopper': NXDiskChopper,
        }
        with snx.File(path, definitions=defs) as f:
            ...
    """

    _SPECIAL_FIELDS = {
        'typ',
        'position',
        'rotation_speed',
        'top_dead_center',
        '_clockwise',
    }

    def assemble(self, dg: sc.DataGroup) -> DiskChopper:
        dg = snx.compute_positions(dg)

        field_names = {
            field.name for field in dataclasses.fields(DiskChopper)
        } - self._SPECIAL_FIELDS

        return DiskChopper(
            typ=dg.get('type', DiskChopperType.single),
            position=dg['position'],
            rotation_speed=_parse_rotation_speed(dg['rotation_speed']),
            top_dead_center=_parse_tdc(dg.get('top_dead_center')),
            **{name: _parse_maybe_log(dg.get(name)) for name in field_names},
        )


def _parse_rotation_speed(x: Union[sc.DataArray, sc.DataGroup]) -> sc.DataArray:
    x = _parse_maybe_log(x)
    if x.unit is None:
        x.unit = 'Hz'
    return x


def _parse_tdc(
    x: Optional[Union[sc.Variable, sc.DataArray, sc.DataGroup]]
) -> Optional[Union[sc.Variable, sc.DataArray]]:
    if x is None:
        return x
    if isinstance(x, sc.DataGroup):
        return x['time']
    return x


def _parse_maybe_log(
    x: Optional[Union[sc.DataArray, sc.DataGroup]]
) -> Optional[sc.DataArray]:
    if x is None:
        return x
    if isinstance(x, sc.DataGroup):
        return x['value'].squeeze()
    return x
