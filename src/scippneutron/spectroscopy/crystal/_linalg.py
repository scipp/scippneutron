# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Linear algebra functions on top of scipp.spatial."""

import warnings

import numpy as np
import scipp as sc
from scipy.spatial.transform import Rotation


def transpose_matrix(mat: sc.Variable) -> sc.Variable:
    """Transpose a matrix given as a linear transform."""
    if mat.dtype != sc.DType.linear_transform3:
        raise sc.DTypeError(f"Expected dtype 'linear_transform3' but got {mat.dtype}.")
    return sc.spatial.linear_transform(value=mat.value.T, unit=mat.unit)


def invert_transform(mat: sc.Variable) -> sc.Variable:
    """Invert a spatial transformation.

    This uses :func:`scipp.spatial.inv` but raises :class:`ValueError`
    if the input cannot be inverted.
    """
    inv = sc.spatial.inv(mat)
    if np.any(np.isnan(inv.values)):
        raise ValueError(
            "Cannot invert this transformation. Check whether the matrix is singular."
        )
    return inv


def align_vectors(
    a: sc.Variable, b: sc.Variable, *, a_name: str, b_name: str, error_threshold: float
) -> sc.Variable:
    """Compute a rotation that maps b onto a.

    This function solves

    .. math::

        a = R b

    for :math:`R`.

    This function requires that ``a`` and ``b`` are orthonormal and have the same
    handedness. Otherwise, it may fail to compute a rotation and may raise a
    :class:`RuntimeWarning`. If any vectors in ``a`` or ``b`` are linearly-dependent,
    the operation will fail with a :class:`ValueError`.

    Parameters
    ----------
    a:
        Three orthonormal vectors.
    b:
        Three orthonormal vectors.
    a_name:
        Name of the variable ``a`` in error messages.
    b_name:
        Name of the variable ``b`` in error messages.
    error_threshold:
        Warn when the resulting error exceeds this threshold.

    Returns
    -------
    :
        The rotation as a variable with :class:`scipp.DType.rotation3`.
    """
    try:
        rotation, error = Rotation.align_vectors(a.values, b.values)
    except ValueError as error:
        error.add_note(f"When computing a rotation from '{b_name}' to '{a_name}'")
        raise
    if error > error_threshold:
        warnings.warn(
            f"Computing a rotation from '{b_name}' to '{a_name}' resulted in an error"
            f"of {error} (threshold is {error_threshold}).",
            RuntimeWarning,
            stacklevel=2,
        )
    return sc.spatial.rotation(value=rotation.as_quat())


def orthonormal_system_from_xy(v1: sc.Variable, v2: sc.Variable) -> sc.Variable:
    """Compute an orthonormal system from two vectors.

    Given two vectors v1 and v2, compute a set of vectors
    where z ⟂ (v1, v2), y ⟂ (v1, z), x || v1
    """
    x = v1
    z = sc.cross(x, v2)
    y = sc.cross(z, x)
    return normed_coord_basis(x, y, z)


def normed_coord_basis(a: sc.Variable, b: sc.Variable, c: sc.Variable) -> sc.Variable:
    """Construct a basis of 3 normalized vectors."""
    return sc.concat(
        [
            sc.to_unit(a / sc.norm(a), 'one'),
            sc.to_unit(b / sc.norm(b), 'one'),
            sc.to_unit(c / sc.norm(c), 'one'),
        ],
        dim='vector',
    )
