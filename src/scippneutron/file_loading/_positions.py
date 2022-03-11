# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from _warnings import warn
from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
import scipp as sc
import scipp.spatial
from ._common import Group, add_position_and_transforms_to_data
from .nxtransformations import TransformationError, get_full_transformation_matrix
from ._nexus import LoadFromNexus


class PositionError(Exception):
    pass


def _make_prefixed_names(name, prefix):
    if prefix is None:
        prefix = f"{name}_"
    return prefix + "transform", prefix + "position", prefix + "base_position"


def load_positions_of_components(groups: List[Group],
                                 data: sc.Variable,
                                 name: str,
                                 nx_class: str,
                                 nexus: LoadFromNexus,
                                 default_position: Optional[ArrayLike] = None,
                                 name_prefix: Optional[str] = None):
    for group in groups:
        try:
            position, transformations = _get_base_pos_and_transforms_of_component(
                group=group,
                name=name,
                nx_class=nx_class,
                nexus=nexus,
                default_position=default_position)
        except PositionError:
            continue

        if len(groups) != 1:
            name = nexus.get_name(group)
        transform_name, position_name, base_position_name = _make_prefixed_names(
            name=name, prefix=name_prefix)

        add_position_and_transforms_to_data(data=data,
                                            transform_name=transform_name,
                                            position_name=position_name,
                                            base_position_name=base_position_name,
                                            positions=position,
                                            transforms=transformations)


def _get_base_pos_and_transforms_of_component(
        group: Group, name: str, nx_class: str, nexus: LoadFromNexus,
        default_position: ArrayLike) -> Tuple[sc.Variable, sc.DataArray]:
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
        if units is None:
            warn(f"'distance' dataset in {nx_class} is missing "
                 f"units attribute, skipping loading {name} position")
            raise PositionError
    elif default_position is None:
        warn(f"No position given for {name} in file")
        raise PositionError
    else:
        base_position = np.asarray(default_position)
        units = sc.units.m

    return sc.vector(value=base_position, unit=units), transformations
