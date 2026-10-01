# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Linear algebra functions on top of scipp.spatial."""

import numpy as np
import scipp as sc


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
