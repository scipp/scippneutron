# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from _warnings import warn
from typing import List, Optional, Tuple, Union
import numpy as np
import scipp as sc
import scipp.spatial
from ._common import Group
from ._transformations import TransformationError, get_full_transformation_matrix
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
        position, transformations = _get_base_pos_and_transforms_of_component(
            groups[0], name, nx_class, nexus, default_position)
    except PositionError:
        return

    _add_position_to_data(name, data, position, transformations)


def load_positions_of_components(groups: List[Group],
                                 data: sc.Variable,
                                 name: str,
                                 nx_class: str,
                                 nexus: LoadFromNexus,
                                 default_position: Optional[np.ndarray] = None):
    for group in groups:
        try:
            position, transformation = _get_base_pos_and_transforms_of_component(
                group, name, nx_class, nexus, default_position)
        except PositionError:
            continue

        if len(groups) != 1:
            name = nexus.get_name(group)

        _add_position_to_data(name, data, position, transformation)


def _add_position_to_data(name: str, data: sc.Variable, position: sc.Variable,
                          transformation: Optional[Union[sc.Variable, sc.DataArray]]):

    _add_coord_to_loaded_data(attr_name=f"{name}_base_position",
                              data=data,
                              coord=position)

    if transformation is None:
        _add_coord_to_loaded_data(attr_name=f"{name}_position",
                                  data=data,
                                  coord=position)
    else:
        if isinstance(transformation, sc.Variable):
            # The transform will be a (scalar) variable if it is not time-dependent.
            # This means that we can calculate the actual position easily.
            _add_coord_to_loaded_data(attr_name=f"{name}_position",
                                      data=data,
                                      coord=transformation * position)

        _add_coord_to_loaded_data(attr_name=f"{name}_transform",
                                  data=data,
                                  coord=sc.scalar(value=transformation))


def _get_base_pos_and_transforms_of_component(
        group: Group,
        name: str,
        nx_class: str,
        nexus: LoadFromNexus,
        default_position: Optional[np.ndarray] = None
) -> Tuple[sc.Variable, sc.DataArray]:
    depends_on_found, _ = nexus.dataset_in_group(group, "depends_on")
    distance_found, _ = nexus.dataset_in_group(group, "distance")

    transformations = sc.spatial.translation(unit=sc.units.m, value=[0, 0, 0])

    if depends_on_found:
        try:
            transformations = get_full_transformation_matrix(group, nexus)
            base_position = np.array([0, 0, 0])
        except TransformationError as e:
            warn(f"Skipping loading {name} position due to error: {e}")
            raise PositionError
        units = sc.units.m
    elif distance_found:
        base_position = np.array(
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
        base_position = np.array([0, 0, 0])
        units = sc.units.m

    return sc.vector(value=base_position, unit=units), transformations


def _add_coord_to_loaded_data(attr_name: str, data: sc.Variable, coord: sc.Variable):

    if isinstance(data, sc.DataArray):
        data = data.coords

    try:
        data[attr_name] = coord
    except KeyError:
        pass
