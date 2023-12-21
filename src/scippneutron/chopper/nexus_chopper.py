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

    This function

    - extracts relevant time series from ``NXlog``,
    - converts slit edges to the 2d layout required by
      :class:`~scippneutron.chopper.disk_chopper.DiskChopper`

    The output may still contain time-dependent fields which need to be
    further processed.
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
            'slit_edges': _parse_slit_edges(chopper['slit_edges']),
            'top_dead_center': _parse_tdc(chopper['top_dead_center']),
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


# TODO check 2d edge order -> also in DiskChopper
def _parse_slit_edges(edges: Optional[sc.Variable]) -> Optional[sc.Variable]:
    if edges is None:
        return None
    if edges.ndim == 1:
        edge_dim = 'edge' if edges.dim != 'edge' else '__edge_dim'
        folded = edges.fold(edges.dim, sizes={edges.dim: -1, edge_dim: 2})
        if sc.any(folded[edge_dim, 0] > folded[edge_dim, 1]):
            raise ValueError(
                "Invalid slit edges, must be given as "
                "[begin_0, end_0, begin_1, end_1, ...] where begin_n < end_n"
            )
        return folded
    if edges.ndim == 2:
        if edges.shape[1] != 2:
            raise sc.DimensionError(
                "The second dim of the slit edges must be length 2."
            )
        return edges
    else:
        raise sc.DimensionError("The slit edges must be 1- or 2-dimensional")


def _parse_tdc(
    tdc: Optional[Union[sc.Variable, sc.DataArray, sc.DataGroup]]
) -> Optional[Union[sc.Variable, sc.DataArray]]:
    if tdc is None:
        return tdc
    if isinstance(tdc, sc.DataGroup):
        # An NXlog
        return tdc['time']
    return tdc


def _parse_maybe_log(
    x: Optional[Union[sc.Variable, sc.DataArray, sc.DataGroup]]
) -> Optional[Union[sc.Variable, sc.DataArray]]:
    if x is None:
        return x
    if isinstance(x, sc.DataGroup):
        # An NXlog
        return x['value'].squeeze()
    return x
