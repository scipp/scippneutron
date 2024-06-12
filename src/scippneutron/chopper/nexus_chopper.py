# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from collections.abc import Mapping
from typing import Any

import scipp as sc

from .disk_chopper import DiskChopperType


def extract_chopper_from_nexus(
    chopper: Mapping[str, sc.Variable | sc.DataArray | sc.DataGroup],
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
            "type": DiskChopperType(chopper.get("type", DiskChopperType.single)),
            **{key: _parse_field(key, val) for key, val in chopper.items()},
        }
    )


def _parse_field(key: str, value: Any) -> Any:
    if key == "top_dead_center":
        return _parse_tdc(value)
    return _parse_maybe_log(value)


def _parse_tdc(
    tdc: sc.Variable | sc.DataArray | sc.DataGroup,
) -> sc.Variable | sc.DataArray:
    if isinstance(tdc, sc.DataGroup):
        # An NXlog without 'value'
        return tdc["time"]
    return tdc


def _parse_maybe_log(
    x: sc.Variable | sc.DataArray | sc.DataGroup,
) -> sc.Variable | sc.DataArray:
    if isinstance(x, sc.DataGroup) and "value" in x:
        # An NXlog
        return x["value"].squeeze()
    return x
