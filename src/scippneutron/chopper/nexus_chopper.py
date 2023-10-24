# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Optional, Union

import scipp as sc
import scippnexus as snx

from .disk_chopper import DiskChopper


class NXDiskChopper(snx.NXdisk_chopper):
    """NeXus definition to load a DiskChopper using ScippNeXus.

    Examples
    --------
    Use as

        >>> defs = {
        ...    **snx.base_definitions(),
        ...    'NXdisk_chopper': NXDiskChopper,
        ... }
        >>> with snx.File(path, definitions=defs) as f:
        ...     ...
    """

    def assemble(self, dg: sc.DataGroup) -> DiskChopper:
        # TODO needs depends_on which is not in the old file we have
        position = sc.vector([0, 0, 0], unit='m')

        return DiskChopper(
            typ=dg.get('type', 'single'),
            position=position,
            rotation_speed=_parse_rotation_speed(dg['rotation_speed']),
            delay=_parse_maybe_log(dg.get('delay')),
            **{key: dg.get(key) for key in ('radius', 'slit_height', 'slit_edges')},
        )


def _parse_rotation_speed(x: Union[sc.DataArray, sc.DataGroup]) -> sc.DataArray:
    x = _parse_maybe_log(x)
    if x.unit is None:
        x.unit = 'Hz'
    return x


def _parse_maybe_log(
    x: Optional[Union[sc.DataArray, sc.DataGroup]]
) -> Optional[sc.DataArray]:
    if x is None:
        return x
    if isinstance(x, sc.DataGroup):
        return x['value']
    return x
