# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from _warnings import warn
from typing import List, Optional, Tuple, Any
import numpy as np
import scipp as sc
from ._common import Group
from ._transformations import (get_position_from_transformations, TransformationError)
from ._nexus import LoadFromNexus


class PositionError(Exception):
    pass


def load_position_of_unique_component(groups: List[Group],
                                      data: sc.Variable,
                                      name: str,
                                      nx_class: str,
                                      nexus: LoadFromNexus,
                                      default_position: Optional[np.ndarray] = None):
    if len(groups) > 1:
        warn(f"More than one {nx_class} found in file, "
             f"skipping loading {name} position")
        return
    try:
        position, units = _get_position_of_component(groups[0], name, nx_class, nexus,
                                                     default_position)
    except PositionError:
        return
    _add_coord_to_loaded_data(f"{name}_position",
                              data,
                              position,
                              unit=units,
                              dtype=sc.dtype.vector3)


def load_positions_of_components(groups: List[Group],
                                 data: sc.Variable,
                                 name: str,
                                 nx_class: str,
                                 nexus: LoadFromNexus,
                                 default_position: Optional[np.ndarray] = None):
    for group in groups:
        try:
            position, units = _get_position_of_component(group, name, nx_class, nexus,
                                                         default_position)
        except PositionError:
            continue
        if len(groups) == 1:
            _add_coord_to_loaded_data(f"{name}_position",
                                      data,
                                      position,
                                      unit=units,
                                      dtype=sc.dtype.vector3)
        else:
            _add_coord_to_loaded_data(f"{nexus.get_name(group)}_position",
                                      data,
                                      position,
                                      unit=units,
                                      dtype=sc.dtype.vector3)


def _get_position_of_component(
        group: Group,
        name: str,
        nx_class: str,
        nexus: LoadFromNexus,
        default_position: Optional[np.ndarray] = None) -> Tuple[np.ndarray, sc.Unit]:
    depends_on_found, _ = nexus.dataset_in_group(group, "depends_on")
    distance_found, _ = nexus.dataset_in_group(group, "distance")
    if depends_on_found:
        try:
            position = get_position_from_transformations(group, nexus)
        except TransformationError as e:
            warn(f"Skipping loading {name} position due to error: {e}")
            raise PositionError
        units = sc.units.m
    elif distance_found:

        position = np.array(
            [0, 0,
             nexus.load_dataset_from_group_as_numpy_array(group, "distance")])
        units = nexus.get_unit(nexus.get_dataset_from_group(group, "distance"))
        if units == sc.units.dimensionless:
            warn(f"'distance' dataset in {nx_class} is missing "
                 f"units attribute, skipping loading {name} position")
            raise PositionError
    elif default_position is None:
        warn(f"No position given for {name} in file")
        raise PositionError
    else:
        position = np.array([0, 0, 0])
        units = sc.units.m

    return position, units


def _add_coord_to_loaded_data(attr_name: str,
                              data: sc.Variable,
                              value: np.ndarray,
                              unit: sc.Unit,
                              dtype: Optional[Any] = None):

    if isinstance(data, sc.DataArray):
        data = data.coords

    try:
        if dtype is not None:
            if dtype == sc.dtype.vector3:
                data[attr_name] = sc.vector(value=value, unit=unit)
            else:
                data[attr_name] = sc.scalar(value, dtype=dtype, unit=unit)
        else:
            data[attr_name] = sc.scalar(value, unit=unit)
    except KeyError:
        pass
