# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

import numpy as np
from ._loading_common import ensure_str
from typing import Union, List
import scipp as sc
import h5py
from cmath import isclose


class TransformationError(Exception):
    pass


def _alt_rotation_matrix_from_axis_and_angle(axis: np.ndarray,
                                             angle_radians: float):
    axis_x = axis[0]
    axis_y = axis[1]
    axis_z = axis[2]
    cos_t = np.cos(angle_radians)
    sin_t = np.sin(angle_radians)
    one_minus_cos_t = 1 - cos_t
    x_sin_t = axis_x * sin_t
    y_sint_t = axis_y * sin_t
    z_sint_t = axis_z * sin_t
    return np.array([[
        cos_t + axis_x**2.0 * one_minus_cos_t,
        axis_x * axis_y * one_minus_cos_t - z_sint_t,
        axis_x * axis_z * one_minus_cos_t + y_sint_t
    ],
                     [
                         axis_y * axis_x * one_minus_cos_t + z_sint_t,
                         cos_t + axis_y**2.0 * one_minus_cos_t,
                         axis_y * axis_z * one_minus_cos_t - x_sin_t
                     ],
                     [
                         axis_z * axis_x * one_minus_cos_t - y_sint_t,
                         axis_z * axis_y * one_minus_cos_t + x_sin_t,
                         cos_t + axis_z**2.0 * one_minus_cos_t
                     ]])


def _rotation_matrix_from_axis_and_angle(axis: np.ndarray,
                                         angle_radians: float):
    # Following convention for passive transformation
    # Variable naming follows that used in
    # https://doi.org/10.1061/(ASCE)SU.1943-5428.0000247
    i = np.identity(3)
    l1, l2, l3 = tuple(axis)
    ll = np.array([[0, -l3, l2], [l3, 0, -l1], [-l2, l1, 0]])
    l1l2 = np.matmul(l1, l2)
    l1l3 = np.matmul(l1, l3)
    l2l3 = np.matmul(l2, l3)
    l1_squared = np.matmul(l1, l1)
    l2_squared = np.matmul(l2, l2)
    l3_squared = np.matmul(l3, l3)
    ll_2 = np.array([[-(l2_squared + l3_squared), l1l2, l1l3],
                     [l1l2, -(l1_squared + l3_squared), l2l3],
                     [l1l3, l2l3, -(l1_squared + l2_squared)]])

    return i - np.sin(angle_radians) * ll + (1 - np.cos(angle_radians)) * ll_2


def get_position_from_transformations(group: h5py.Group,
                                      root: [h5py.File, h5py.Group]):
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

    return np.matmul(total_transform_matrix, np.array([0, 0, 0, 1],
                                                      dtype=float))[0:3]

    # vertices = vertices.T
    # Add fourth element of 1 to each vertex, indicating these are
    # positions not direction vectors
    # vertices = np.vstack((vertices, np.ones(vertices.shape[1])))
    # vertices = do_transformations(transformations, vertices)
    # Now the transformations are done we do not need the 4th element
    # return vertices[:3, :].T


def _get_transformations(depends_on: Union[str, bytes],
                         transformations: List[np.ndarray],
                         root: [h5py.File, h5py.Group], group_name: str):
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
        next_depends_on = _append_transformation(transform, transformations)
        _get_transformations(next_depends_on, transformations, root,
                             group_name)


def _append_transformation(transform: h5py.Dataset,
                           transformations: List[np.ndarray]):
    attributes = transform.attrs
    offset = [0., 0., 0.]
    if 'offset' in attributes:
        offset = attributes['offset'].astype(float)
    if attributes['transformation_type'] == 'translation':
        _append_translation(attributes, offset, transform, transformations)
    elif attributes['transformation_type'] == 'rotation':
        _append_rotation(attributes, offset, transform, transformations)
    else:
        raise TransformationError(
            f"Unknown transformation type "
            f"'{attributes['transformation_type'].astype(str)}'"
            f" at {transform.name}")
    return attributes['depends_on']


def normalise(vector: np.ndarray, transform: h5py.Dataset):
    norm = np.linalg.norm(vector)
    if isclose(norm, 0.):
        raise TransformationError(
            f"Magnitude of 'vector' attribute in transformation at "
            f"{transform.name} has magnitude too close to zero")
    return vector / norm


def _append_translation(attributes: h5py.AttributeManager, offset: List[float],
                        transform: h5py.Dataset,
                        transformations: List[np.ndarray]):
    # TODO no assumptions about units (convert everything to metres?)
    try:
        direction = attributes['vector']
    except KeyError:
        raise TransformationError(
            f"Missing 'vector' attribute in transformation at {transform.name}"
        )
    vector = normalise(direction,
                       transform) * transform[...].astype(float).item()
    matrix = np.array([[1., 0., 0., vector[0] + offset[0]],
                       [0., 1., 0., vector[1] + offset[1]],
                       [0., 0., 1., vector[2] + offset[2]], [0., 0., 0., 1.]])
    transformations.append(matrix)


def _append_rotation(attributes, offset, transform, transformations):
    axis = attributes['vector']
    unit_str = ensure_str(transform.attrs['units'])
    unit_error = TransformationError(
        f"Unit for rotation transformation must be radians "
        f"or degrees but found '{unit_str}' at {transform.name}")
    try:
        units = sc.Unit(unit_str)
    except RuntimeError:
        raise unit_error
    if units == sc.units.deg:
        angle = np.deg2rad(transform[...])
    elif units == sc.units.rad:
        angle = transform[...]
    else:
        raise unit_error
    rotation_matrix = _rotation_matrix_from_axis_and_angle(axis, angle)
    # Make 4x4 matrix from our 3x3 rotation matrix to include
    # possible "offset"
    # TODO try using rotation_matrix.resize((4, 4))
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
