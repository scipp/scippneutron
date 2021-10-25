# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
import warnings

import numpy as np
from ._common import MissingDataset, MissingAttribute, Group
from typing import Union, List, Tuple
import scipp as sc
import h5py
from cmath import isclose
from ._nexus import LoadFromNexus, GroupObject
from ._json_nexus import contains_stream


class TransformationError(Exception):
    pass


def _rotation_matrix_from_axis_and_angle(axis: np.ndarray,
                                         angle_radians: float) -> np.ndarray:
    # Following convention for passive transformation
    # Variable naming follows that used in
    # https://doi.org/10.1061/(ASCE)SU.1943-5428.0000247
    i = np.identity(3)
    l1, l2, l3 = tuple(axis)
    ll = np.array([[0, -l3, l2], [l3, 0, -l1], [-l2, l1, 0]])
    l1l2 = l1 * l2
    l1l3 = l1 * l3
    l2l3 = l2 * l3
    l1_squared = l1 * l1
    l2_squared = l2 * l2
    l3_squared = l3 * l3
    ll_2 = np.array([[-(l2_squared + l3_squared), l1l2, l1l3],
                     [l1l2, -(l1_squared + l3_squared), l2l3],
                     [l1l3, l2l3, -(l1_squared + l2_squared)]])

    return i - np.sin(angle_radians) * ll + (1 - np.cos(angle_radians)) * ll_2


def get_position_from_transformations(group: Group, nexus: LoadFromNexus) -> np.ndarray:
    """
    Get position of a component which has a "depends_on" dataset

    :param group: The HDF5 group of the component, containing depends_on
    :param nexus: wrap data access to hdf file or objects from json
    :return: Position of the component as a three-element numpy array
    """
    total_transform_matrix = get_full_transformation_matrix(group, nexus)
    return np.matmul(total_transform_matrix, np.array([0, 0, 0, 1], dtype=float))[0:3]


def get_full_transformation_matrix(group: Group, nexus: LoadFromNexus) -> np.ndarray:
    """
    Get the 4x4 transformation matrix for a component, resulting
    from the full chain of transformations linked by "depends_on"
    attributes

    :param group: The HDF5 group of the component, containing depends_on
    :param nexus: wrap data access to hdf file or objects from json
    :return: 4x4 passive transformation matrix as a numpy array
    """
    transformations = []
    try:
        depends_on = nexus.load_scalar_string(group.group, "depends_on")
    except MissingDataset:
        depends_on = '.'
    _get_transformations(depends_on, transformations, group,
                         nexus.get_name(group.group), nexus)
    total_transform_matrix = np.identity(4)
    for transformation in transformations:
        total_transform_matrix = np.matmul(transformation, total_transform_matrix)
    return total_transform_matrix


def _get_transformations(transform_path: str, transformations: List[np.ndarray],
                         group: Group, group_name: str, nexus: LoadFromNexus):
    """
    Get all transformations in the depends_on chain.

    :param transform_path: The first depends_on path string
    :param transformations: List of transformations to populate
    :param root: root of the file, depends_on paths assumed to be
      relative to this
    """

    # TODO: this list of transformation should probably be cached in the future
    # to deal with changing beamline components (e.g. pixel positions) during a
    # live data stream (see https://github.com/scipp/scippneutron/issues/76).

    if transform_path != '.':
        try:
            transform = nexus.get_object_by_path(group.file, transform_path)
        except KeyError:
            raise TransformationError(
                f"Non-existent depends_on path '{transform_path}' found "
                f"in transformations chain for {group_name}")
        next_depends_on = _append_transformation(transform, transformations, group_name,
                                                 nexus)
        _get_transformations(next_depends_on, transformations, group, group_name, nexus)


def _transformation_is_nx_log_stream(transform: Union[h5py.Dataset, GroupObject],
                                     nexus: LoadFromNexus):
    # Stream objects are only in the dict loaded from json
    if isinstance(transform, dict):
        # If transform is a group and contains a stream but not a value dataset
        # then assume it is a streamed NXlog transformation
        try:
            if nexus.is_group(transform):
                found_value_dataset, _ = nexus.dataset_in_group(transform, "value")
                if not found_value_dataset and contains_stream(transform):
                    return True
        except KeyError:
            pass
    return False


def _append_transformation(transform: Union[h5py.Dataset, GroupObject],
                           transformations: List[np.ndarray], group_name: str,
                           nexus: LoadFromNexus) -> str:
    if _transformation_is_nx_log_stream(transform, nexus):
        warnings.warn("Streamed NXlog found in transformation "
                      "chain, getting its value from stream is "
                      "not yet implemented and instead it will be "
                      "treated as a 0-distance translation")
        matrix = np.eye(4, dtype=float)
        transformations.append(matrix)
    else:
        try:
            vector = nexus.get_attribute_as_numpy_array(transform,
                                                        "vector").astype(float)
            vector = _normalise(vector, nexus.get_name(transform))
        except MissingAttribute:
            raise TransformationError(f"Missing 'vector' attribute in transformation "
                                      f"at {nexus.get_name(transform)}")

        try:
            offset = nexus.get_attribute_as_numpy_array(transform,
                                                        "offset").astype(float)
        except MissingAttribute:
            offset = np.array([0., 0., 0.], dtype=float)

        transform_type = nexus.get_string_attribute(transform, "transformation_type")
        if transform_type == 'translation':
            _append_translation(offset, transform, transformations, vector, group_name,
                                nexus)
        elif transform_type == 'rotation':
            _append_rotation(offset, transform, transformations, vector, group_name,
                             nexus)
        else:
            raise TransformationError(f"Unknown transformation type "
                                      f"'{transform_type}'"
                                      f" at {nexus.get_name(transform)}")
    try:
        depends_on = nexus.get_string_attribute(transform, "depends_on")
    except MissingAttribute:
        depends_on = "."
    return depends_on


def _normalise(vector: np.ndarray, transform_name: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if isclose(norm, 0.):
        raise TransformationError(
            f"Magnitude of 'vector' attribute in transformation at "
            f"{transform_name} is too close to zero")
    return vector / norm


def _append_translation(offset: np.ndarray, transform: GroupObject,
                        transformations: List[np.ndarray],
                        direction_unit_vector: np.ndarray, group_name: str,
                        nexus: LoadFromNexus):
    magnitude, unit = _get_transformation_magnitude_and_unit(group_name, transform,
                                                             nexus)

    if unit != sc.units.m:
        magnitude_var = magnitude * unit
        magnitude_var = sc.to_unit(magnitude_var, sc.units.m)
        magnitude = magnitude_var.value
    # -1 as describes passive transformation
    vector = direction_unit_vector * -1. * magnitude
    offset_vector = vector + offset
    matrix = np.block([[np.eye(3), offset_vector[np.newaxis].T], [0., 0., 0., 1.]])
    transformations.append(matrix)


def _get_unit(attributes: h5py.AttributeManager, transform_name: str) -> sc.Unit:
    try:
        unit_str = attributes["units"]
    except KeyError:
        raise TransformationError(
            f"Missing units for transformation at {transform_name}")
    try:
        unit = sc.Unit(unit_str)
    except RuntimeError:
        raise TransformationError(f"Unrecognised units '{unit_str}' for "
                                  f"transformation at {transform_name}")
    return unit


def _get_transformation_magnitude_and_unit(
        group_name: str, transform: Union[h5py.Dataset, GroupObject],
        nexus: LoadFromNexus) -> Tuple[float, sc.Unit]:
    if nexus.is_group(transform):
        value = nexus.load_dataset_from_group_as_numpy_array(transform, "value")
        try:
            if value.size > 1:
                raise TransformationError(f"Found multivalued NXlog as a "
                                          f"transformation for {group_name}, "
                                          f"this is not yet supported")
            if value.size == 0:
                raise TransformationError(f"Found empty NXlog as a "
                                          f"transformation for {group_name}")
            magnitude = value.astype(float).item()
        except KeyError:
            raise TransformationError(
                f"Encountered {nexus.get_name(transform)} in transformation "
                f"chain for {group_name} but it is a group without a value "
                "dataset; not a valid transformation")
        unit = nexus.get_unit(nexus.get_dataset_from_group(transform, "value"))
        if unit == sc.units.dimensionless:
            # See if the value unit is on the NXLog itself instead
            unit = nexus.get_unit(transform)
            if unit == sc.units.dimensionless:
                raise TransformationError(f"Missing units for transformation at "
                                          f"{nexus.get_name(transform)}")
    else:
        magnitude = nexus.load_dataset_as_numpy_array(transform).astype(float).item()
        unit = nexus.get_unit(transform)
        if unit == sc.units.dimensionless:
            raise TransformationError(f"Missing units for transformation at "
                                      f"{nexus.get_name(transform)}")
    return magnitude, sc.Unit(unit)


def _append_rotation(offset: np.ndarray, transform: GroupObject,
                     transformations: List[np.ndarray], rotation_axis: np.ndarray,
                     group_name: str, nexus: LoadFromNexus):
    angle, unit = _get_transformation_magnitude_and_unit(group_name, transform, nexus)
    if unit == sc.units.deg:
        angle = np.deg2rad(angle)
    elif unit != sc.units.rad:
        raise TransformationError(f"Unit for rotation transformation must be radians "
                                  f"or degrees, problem in {transform.name}")
    rotation_matrix = _rotation_matrix_from_axis_and_angle(rotation_axis, angle)
    # Make 4x4 matrix from our 3x3 rotation matrix to include
    # possible "offset"
    matrix = np.block([[rotation_matrix, offset[np.newaxis].T], [0., 0., 0., 1.]])
    transformations.append(matrix)
