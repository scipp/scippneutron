# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Mapping, Optional, Union

import scipp as sc

from .disk_chopper import DiskChopperType

_CHOPPER_FIELD_NAMES = (
    'beam_position',
    'delay',
    'phase',
    'position',
    'radius',
    'slits',
    'slit_height',
)


def post_process_disk_chopper(
    chopper: Mapping[str, Union[sc.Variable, sc.DataArray, sc.DataGroup]]
) -> sc.DataGroup:
    """Convert loaded NeXus disk chopper data to the layout used by ScipNeutron.

    This function extracts relevant time series from ``NXlog``.
    The output may, however, still contain time-dependent fields which need to be
    processed further.
    See :ref:`disk_chopper-time_dependent_parameters`.

    Parameters
    ----------
    chopper:
        The loaded NeXus disk chopper data.

    Returns
    -------
    :
        A new data group with processed fields in the layout expected by
        :meth:`DiskChopper.from_nexus
        <scippneutron.chopper.disk_chopper.DiskChopper.from_nexus>`.
    """
    return sc.DataGroup(
        {
            'type': DiskChopperType(chopper.get('type', DiskChopperType.single)),
            'rotation_speed': _parse_rotation_speed(chopper['rotation_speed']),
            'slit_edges': chopper.get('slit_edges'),
            'top_dead_center': _parse_maybe_log(chopper.get('top_dead_center')),
            **{
                name: _parse_maybe_log(chopper.get(name))
                for name in _CHOPPER_FIELD_NAMES
            },
        }
    )


def _parse_rotation_speed(
    rotation_speed: Union[sc.DataArray, sc.DataGroup]
) -> sc.DataArray:
    return _parse_maybe_log(rotation_speed)


def _parse_maybe_log(
    x: Optional[Union[sc.Variable, sc.DataArray, sc.DataGroup]]
) -> Optional[Union[sc.Variable, sc.DataArray]]:
    if x is None:
        return x
    if isinstance(x, sc.DataGroup):
        # An NXlog
        return x['value'].squeeze()
    return x
