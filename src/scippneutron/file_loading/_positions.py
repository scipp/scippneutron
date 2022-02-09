# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from _warnings import warn
from typing import List, Optional, Tuple
import numpy as np
import scipp as sc
import scipp.spatial
from ._common import Group, add_position_and_transforms_to_data
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

    add_position_and_transforms_to_data(
        data=data,
        transform_name=f"{name}_transform",
        position_name=f"{name}_position",
        base_position_name=f"{name}_base_position",
        positions=position,
        transforms=transformations)


def load_positions_of_components(groups: List[Group],
                                 data: sc.Variable,
                                 name: str,
                                 nx_class: str,
                                 nexus: LoadFromNexus,
                                 default_position: Optional[np.ndarray] = None):
    for group in groups:
        try:
            position, transformations = _get_base_pos_and_transforms_of_component(
                group, name, nx_class, nexus, default_position)
        except PositionError:
            continue

        if len(groups) != 1:
            name = nexus.get_name(group)

        add_position_and_transforms_to_data(
            data=data,
            transform_name=f"{name}_transform",
            position_name=f"{name}_position",
            base_position_name=f"{name}_base_position",
            positions=position,
            transforms=transformations)


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
