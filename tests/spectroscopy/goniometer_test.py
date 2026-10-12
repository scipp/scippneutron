# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import warnings
from itertools import permutations, product

import numpy as np
import numpy.typing as npt
import pytest
import scipp as sc
import scipp.testing
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from scippneutron.spectroscopy.goniometer import Axis, Goniometer, GoniometerAxis


def test_goniometer_rotation_matrix_3_axes() -> None:
    inf = sc.scalar(np.inf, unit="rad")
    goniometer = Goniometer(
        axes=[
            GoniometerAxis(name="ax1", axis="+z", min=-inf, max=inf),
            GoniometerAxis(name="ax2", axis="-x", min=-inf, max=inf),
            GoniometerAxis(name="ax3", axis="+y", min=-inf, max=inf),
        ]
    )

    angles = [
        sc.scalar(0.2, unit='rad'),
        sc.scalar(40, unit='deg'),
        sc.scalar(-1.6, unit='rad'),
    ]
    rotations = [
        sc.spatial.rotations_from_rotvecs(sc.vector([0, 0, 1]) * angles[0]),
        sc.spatial.rotations_from_rotvecs(sc.vector([-1, 0, 0]) * angles[1]),
        sc.spatial.rotations_from_rotvecs(sc.vector([0, 1, 0]) * angles[2]),
    ]
    expected = rotations[2] * rotations[1] * rotations[0]

    actual = goniometer.rotation_for(angles)
    sc.testing.assert_allclose(actual, expected)


def test_goniometer_rotation_matrix_2_axes() -> None:
    inf = sc.scalar(np.inf, unit="rad")
    goniometer = Goniometer(
        axes=[
            GoniometerAxis(name="ax1", axis="-z", min=-inf, max=inf),
            GoniometerAxis(name="ax3", axis="+y", min=-inf, max=inf),
        ]
    )

    angles = [
        sc.scalar(-15, unit='deg'),
        sc.scalar(3.1, unit='rad'),
    ]
    rotations = [
        sc.spatial.rotations_from_rotvecs(sc.vector([0, 0, -1]) * angles[0]),
        sc.spatial.rotations_from_rotvecs(sc.vector([0, 1, 0]) * angles[1]),
    ]
    expected = rotations[1] * rotations[0]

    actual = goniometer.rotation_for(angles)
    sc.testing.assert_allclose(actual, expected)


def test_goniometer_rotation_matrix_1_axis() -> None:
    inf = sc.scalar(np.inf, unit="rad")
    goniometer = Goniometer(
        axes=[
            GoniometerAxis(name="ax3", axis="+y", min=-inf, max=inf),
        ]
    )

    angles = [
        sc.scalar(2.0, unit='rad'),
    ]
    expected = sc.spatial.rotations_from_rotvecs(sc.vector([0, 1, 0]) * angles[0])

    actual = goniometer.rotation_for(angles)
    sc.testing.assert_allclose(actual, expected)


# Each element is a tuple of the form `(+x, -z, +y)`.
# The list contains through all possible combinations of axes and signs.
_AXIS_SEQUENCES = [
    tuple(s + x for x, s in zip(xyz, signs, strict=True))
    for xyz, signs in product(permutations("xyz"), product("+-", "+-", "+-"))
]


def _axis_sequences(n_axes: int) -> st.SearchStrategy[tuple[Axis, ...]]:
    """Strategy that returns sequences of axes of length ``n_axes``."""
    return st.sampled_from(_AXIS_SEQUENCES).map(lambda x: x[:n_axes])


def _unit_quaternions():
    def normalize(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        norm = np.linalg.norm(q)
        if norm == 0:
            return np.full((4,), 1 / 2)
        return q / norm

    return arrays(
        dtype=np.float64,
        shape=(4,),
        elements=st.floats(
            min_value=-100,  # Restricted to avoid overflow in `np.linalg.norm`
            max_value=100,
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
        ),
    ).map(normalize)


@settings(max_examples=100)
@given(quat=_unit_quaternions(), sequence=_axis_sequences(3))
def test_goniometer_decompose_rotation_roundtrip_3_axes(
    quat: npt.NDArray[np.float64],
    sequence: tuple[Axis, Axis, Axis],
) -> None:
    inf = sc.scalar(np.inf, unit="rad")
    goniometer = Goniometer(
        axes=[
            GoniometerAxis(name="ax1", axis=sequence[0], min=-inf, max=inf),
            GoniometerAxis(name="ax2", axis=sequence[1], min=-inf, max=inf),
            GoniometerAxis(name="ax3", axis=sequence[2], min=-inf, max=inf),
        ]
    )
    initial = sc.spatial.rotation(value=quat)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Gimbal lock", UserWarning)
        decomposed = goniometer.decompose_rotation(initial)
    reconstructed = goniometer.rotation_for(decomposed)

    # Compare rotation matrices, not the dtype=rotation3 variables because
    # of https://github.com/scipp/scipp/issues/3975
    one = sc.spatial.linear_transform(value=np.eye(3))
    sc.testing.assert_allclose(
        reconstructed * one,
        initial * one,
        atol=sc.scalar(1e-8, unit='one'),
    )


@settings(max_examples=100)
@given(
    angle0=st.floats(min_value=-np.pi, max_value=np.pi),
    angle1=st.floats(min_value=-np.pi, max_value=np.pi),
    sequence=_axis_sequences(2),
)
def test_goniometer_decompose_rotation_roundtrip_2_axes(
    angle0: float, angle1: float, sequence: tuple[Axis, Axis]
) -> None:
    inf = sc.scalar(np.inf, unit="rad")
    goniometer = Goniometer(
        axes=[
            GoniometerAxis(name="ax1", axis=sequence[0], min=-inf, max=inf),
            GoniometerAxis(name="ax2", axis=sequence[1], min=-inf, max=inf),
        ]
    )
    r0 = sc.spatial.rotations_from_rotvecs(
        goniometer.axes[0].vector() * sc.scalar(angle0, unit='rad')
    )
    r1 = sc.spatial.rotations_from_rotvecs(
        goniometer.axes[1].vector() * sc.scalar(angle1, unit='rad')
    )
    initial = r1 * r0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Gimbal lock", UserWarning)
        decomposed = goniometer.decompose_rotation(initial)

    sc.testing.assert_allclose(
        decomposed[0],
        sc.scalar(angle0, unit='rad'),
        atol=sc.scalar(1e-8, unit='rad'),
    )
    sc.testing.assert_allclose(
        decomposed[1],
        sc.scalar(angle1, unit='rad'),
        atol=sc.scalar(1e-8, unit='rad'),
    )


def test_goniometer_decompose_rotation_roundtrip_2_detects_unreachable() -> None:
    inf = sc.scalar(np.inf, unit="rad")
    goniometer = Goniometer(
        axes=[
            GoniometerAxis(name="ax1", axis="+x", min=-inf, max=inf),
            GoniometerAxis(name="ax2", axis="+y", min=-inf, max=inf),
        ]
    )
    r0 = sc.spatial.rotations_from_rotvecs(sc.vector([0.1, 0.0, 0.0], unit='rad'))
    r1 = sc.spatial.rotations_from_rotvecs(sc.vector([0.0, -2.9, 0.0], unit='rad'))
    r2 = sc.spatial.rotations_from_rotvecs(sc.vector([0.0, 0.0, 0.7], unit='rad'))
    initial = r2 * r1 * r0

    with pytest.raises(RuntimeError, match="rotation around axis 'z'"):
        _ = goniometer.decompose_rotation(initial)


def test_goniometer_decompose_rotation_detects_gimbal_lock() -> None:
    inf = sc.scalar(np.inf, unit="rad")
    goniometer = Goniometer(
        axes=[
            GoniometerAxis(name="ax1", axis="+x", min=-inf, max=inf),
            GoniometerAxis(name="ax2", axis="+y", min=-inf, max=inf),
            GoniometerAxis(name="ax3", axis="+z", min=-inf, max=inf),
        ]
    )
    # Rotating around y by pi/2 triggers a gimbal lock
    rx = sc.spatial.rotations_from_rotvecs(sc.vector([0.2, 0.0, 0.0], unit='rad'))
    ry = sc.spatial.rotations_from_rotvecs(sc.vector([0.0, np.pi / 2, 0.0], unit='rad'))
    rz = sc.spatial.rotations_from_rotvecs(sc.vector([0.0, 0.0, 1.7], unit='rad'))
    initial = rz * ry * rx

    with pytest.warns(UserWarning, match="Gimbal lock"):
        decomposed = goniometer.decompose_rotation(initial)

    # The angles are usable anyway:
    reconstructed = goniometer.rotation_for(decomposed)

    # Compare rotation matrices, not the dtype=rotation3 variables because
    # of https://github.com/scipp/scipp/issues/3975
    one = sc.spatial.linear_transform(value=np.eye(3))
    sc.testing.assert_allclose(
        reconstructed * one,
        initial * one,
        atol=sc.scalar(1e-8, unit='one'),
    )


def test_goniometer_decompose_rotation_detects_out_of_bounds_angle() -> None:
    inf = sc.scalar(np.inf, unit="rad")
    rad = sc.Unit("rad")
    goniometer = Goniometer(
        axes=[
            GoniometerAxis(name="ax1", axis="+z", min=-0.1 * rad, max=0.1 * rad),
            GoniometerAxis(name="ax2", axis="+y", min=-inf, max=inf),
        ]
    )
    rz = sc.spatial.rotations_from_rotvecs(sc.vector([0.0, 0.0, 0.2], unit='rad'))
    ry = sc.spatial.rotations_from_rotvecs(sc.vector([0.0, 3.1, 0.0], unit='rad'))
    initial = ry * rz

    with pytest.warns(RuntimeWarning, match="outside its limits"):
        decomposed = goniometer.decompose_rotation(initial)

    # The angles are computed regardless:
    sc.testing.assert_allclose(decomposed[0], sc.scalar(0.2, unit='rad'))
    sc.testing.assert_allclose(decomposed[1], sc.scalar(3.1, unit='rad'))
