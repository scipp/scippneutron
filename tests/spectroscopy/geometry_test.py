# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc
import scipp.testing
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from scippneutron.spectroscopy.geometry import Plane


def _vectors() -> st.SearchStrategy[sc.Variable]:
    return (
        arrays(
            shape=(3,),
            dtype=np.float64,
            elements=st.floats(
                min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False
            ),
        )
        .filter(lambda v: np.all(abs(v) > 1e-10))  # all 0 does not work
        .map(lambda v: sc.vector(v))
    )


def _assert_orthonormal_system(
    vector1: sc.Variable, vector2: sc.Variable, normal: sc.Variable
) -> None:
    # All vectors are normalized
    one = sc.scalar(1.0)
    sc.testing.assert_allclose(sc.norm(vector1), one)
    sc.testing.assert_allclose(sc.norm(vector2), one)
    sc.testing.assert_allclose(sc.norm(normal), one)

    # The vectors from a right-handed system
    sc.testing.assert_allclose(
        sc.cross(vector1, vector2), normal, atol=sc.scalar(1e-10)
    )
    sc.testing.assert_allclose(
        sc.cross(vector2, normal), vector1, atol=sc.scalar(1e-10)
    )
    sc.testing.assert_allclose(
        sc.cross(normal, vector1), vector2, atol=sc.scalar(1e-10)
    )


@settings(max_examples=100)
@given(normal=_vectors(), reference=_vectors())
def test_plane_from_normal_and_reference(
    normal: sc.Variable, reference: sc.Variable
) -> None:
    assume(sc.norm(sc.cross(normal, reference)) > sc.scalar(1e-10))

    plane = Plane.from_normal_and_reference(normal=normal, reference=reference)

    # The normal is parallel to the input
    sc.testing.assert_allclose(
        sc.dot(plane.normal, normal), sc.norm(normal), atol=sc.scalar(1e-10)
    )

    _assert_orthonormal_system(
        vector1=plane.vector1, vector2=plane.vector2, normal=plane.normal
    )


@settings(max_examples=100)
@given(reference1=_vectors(), reference2=_vectors())
def test_plane_from_in_plane_vectors(
    reference1: sc.Variable, reference2: sc.Variable
) -> None:
    assume(sc.norm(sc.cross(reference1, reference2)) > sc.scalar(1e-10))

    plane = Plane.from_in_plane_vectors(reference1=reference1, reference2=reference2)

    # vector1 is parallel to the input reference1
    sc.testing.assert_allclose(
        sc.dot(plane.vector1, reference1), sc.norm(reference1), atol=sc.scalar(1e-10)
    )

    _assert_orthonormal_system(
        vector1=plane.vector1, vector2=plane.vector2, normal=plane.normal
    )
