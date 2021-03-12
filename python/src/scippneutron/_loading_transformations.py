# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

import numpy as np
from ._loading_common import ensure_str
from typing import Union, List, Tuple
import scipp as sc
import h5py
from cmath import isclose


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


def get_position_from_transformations(
        group: h5py.Group, root: [h5py.File, h5py.Group]) -> np.ndarray:
    """
    Get position of a component which has a "depends_on" dataset

    :param group: The HDF5 group of the component, containing depends_on
    :param root: The root of the NeXus file, transformation paths are
      assumed to be relative to this
    :return: Position of the component as a three-element numpy array
    """
    total_transform_matrix = get_full_transformation_matrix(group, root)
    return np.matmul(total_transform_matrix, np.array([0, 0, 0, 1],
                                                      dtype=float))[0:3]


def get_full_transformation_matrix(group: h5py.Group,
                                   root: h5py.File) -> np.ndarray:
    """
    Get the 4x4 transformation matrix for a component, resulting
    from the full chain of transformations linked by "depends_on"
    attributes

    :param group: The HDF5 group of the component, containing depends_on
    :param root: The root of the NeXus file, transformation paths are
      assumed to be relative to this
    :return: 4x4 passive transformation matrix as a numpy array
    """
    transformations = []
    try:
        depends_on = group["depends_on"][...].item()
    except KeyError:
        depends_on = '.'
    _get_transformations(depends_on, transformations, root, group.name)
    total_transform_matrix = np.identity(4)
    for transformation in transformations:
        total_transform_matrix = np.matmul(transformation,
                                           total_transform_matrix)
    return total_transform_matrix


def _get_transformations(depends_on: Union[str, bytes],
                         transformations: List[np.ndarray], root: h5py.File,
                         group_name: str):
    """
    Get all transformations in the depends_on chain

    :param depends_on: The first depends_on path string
    :param transformations: List of transformations to populate
    :param root: root of the file, depends_on paths assumed to be
      relative to this
    """
    transform_path = ensure_str(depends_on)
    if transform_path != '.':
        try:
            transform = root[transform_path]
        except KeyError:
            raise TransformationError(
                f"Non-existent depends_on path '{transform_path}' found "
                f"in transformations chain for {group_name}")
        next_depends_on = _append_transformation(transform, transformations,
                                                 group_name)
        _get_transformations(next_depends_on, transformations, root,
                             group_name)


def _append_transformation(transform: Union[h5py.Dataset, h5py.Group],
                           transformations: List[np.ndarray],
                           group_name: str) -> str:
    attributes = transform.attrs
    try:
        vector = attributes['vector'].astype(float)
        vector = _normalise(vector, transform.name)
    except KeyError:
        raise TransformationError(
            f"Missing 'vector' attribute in transformation at {transform.name}"
        )
    offset = [0., 0., 0.]
    if 'offset' in attributes:
        offset = attributes['offset'].astype(float)
    if ensure_str(attributes['transformation_type']) == 'translation':
        _append_translation(offset, transform, transformations, vector,
                            group_name)
    elif ensure_str(attributes['transformation_type']) == 'rotation':
        _append_rotation(offset, transform, transformations, vector,
                         group_name)
    else:
        raise TransformationError(
            f"Unknown transformation type "
            f"'{attributes['transformation_type'].astype(str)}'"
            f" at {transform.name}")
    return attributes['depends_on']


def _normalise(vector: np.ndarray, transform_name: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if isclose(norm, 0.):
        raise TransformationError(
            f"Magnitude of 'vector' attribute in transformation at "
            f"{transform_name} is too close to zero")
    return vector / norm


def _append_translation(offset: List[float], transform: Union[h5py.Dataset,
                                                              h5py.Group],
                        transformations: List[np.ndarray],
                        direction_unit_vector: np.ndarray, group_name: str):
    magnitude, unit = _get_transformation_magnitude_and_unit(
        group_name, transform)

    if unit != sc.units.m:
        magnitude_var = magnitude * unit
        magnitude_var = sc.to_unit(magnitude_var, sc.units.m)
        magnitude = magnitude_var.value
    # -1 as describes passive transformation
    vector = direction_unit_vector * -1. * magnitude
    matrix = np.array([[1., 0., 0., vector[0] + offset[0]],
                       [0., 1., 0., vector[1] + offset[1]],
                       [0., 0., 1., vector[2] + offset[2]], [0., 0., 0., 1.]])
    transformations.append(matrix)


def _get_unit(attributes: h5py.AttributeManager,
              transform_name: str) -> sc.Unit:
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
        group_name: str,
        transform: Union[h5py.Dataset, h5py.Group]) -> Tuple[float, sc.Unit]:
    if isinstance(transform, h5py.Group):
        try:
            if transform["value"].size > 1:
                raise TransformationError(f"Found multivalued NXlog as a "
                                          f"transformation for {group_name}, "
                                          f"this is not yet supported")
            if transform["value"].size == 0:
                raise TransformationError(f"Found empty NXlog as a "
                                          f"transformation for {group_name}")
            magnitude = transform["value"][...].astype(float).item()
        except KeyError:
            raise TransformationError(
                f"Encountered {transform.name} in transformation chain "
                f"for {group_name} but it is a group without a value "
                "dataset; not a valid transformation")
        try:
            unit = _get_unit(transform["value"].attrs, transform.name)
        except TransformationError:
            # See if the value unit is on the NXLog itself instead
            unit = _get_unit(transform.attrs, transform.name)
    else:
        magnitude = transform[...].astype(float).item()
        unit = _get_unit(transform.attrs, transform.name)
    return magnitude, unit


def _append_rotation(offset: List[float], transform: Union[h5py.Dataset,
                                                           h5py.Group],
                     transformations: List[np.ndarray],
                     rotation_axis: np.ndarray, group_name: str):
    angle, unit = _get_transformation_magnitude_and_unit(group_name, transform)
    if unit == sc.units.deg:
        angle = np.deg2rad(angle)
    elif unit != sc.units.rad:
        raise TransformationError(
            f"Unit for rotation transformation must be radians "
            f"or degrees, problem in {transform.name}")
    rotation_matrix = _rotation_matrix_from_axis_and_angle(
        rotation_axis, angle)
    # Make 4x4 matrix from our 3x3 rotation matrix to include
    # possible "offset"
    matrix = np.array([[
        rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
        offset[0]
    ],
                       [
                           rotation_matrix[1, 0], rotation_matrix[1, 1],
                           rotation_matrix[1, 2], offset[1]
                       ],
                       [
                           rotation_matrix[2, 0], rotation_matrix[2, 1],
                           rotation_matrix[2, 2], offset[2]
                       ], [0., 0., 0., 1.]])
    transformations.append(matrix)
