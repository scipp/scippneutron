# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
import scipp.testing

from scippneutron.conversion.tof import momentum_from_energy
from scippneutron.spectroscopy.crystal import alignment


def test_ub_from_3_peaks() -> None:
    peaks = alignment.BraggPeaks(
        hkl=sc.vectors(
            dims=['p'], values=[[1, 0, 0], [1, 1, 0], [0, 1, 1]], unit='1/angstrom'
        ),
        q=sc.vectors(
            dims=['p'], values=[[1, 2, 3], [4, 5, 6], [-1, -2, -3]], unit='1/angstrom'
        ),
        r=sc.spatial.rotations_from_rotvecs(
            sc.vectors(
                dims=['p'],
                values=[[0, 0.1, 0], [0, 1.1, 0], [0, 0.8, 0]],
                unit='rad',
            )
        ),
    )

    ub = alignment.ub_from_3_peaks(peaks)
    assert ub.sizes == {}
    assert ub.dtype == sc.DType.linear_transform3
    assert ub.unit == 'one'

    for hkl, q, r in zip(peaks.hkl, peaks.q, peaks.r, strict=True):
        q_calc = r * ub * hkl * (2 * np.pi)
        sc.testing.assert_allclose(q_calc, q)


def test_u_from_b_and_ub() -> None:
    quat = np.array([5, 2, 6, 3])
    original_u = sc.spatial.rotation(value=quat / np.linalg.norm(quat))
    b = sc.spatial.linear_transform(
        value=[[7, 2, 3], [7, 1, 2], [-5, -3, 9]], unit='one'
    )
    ub = original_u * b

    reconstructed_u = alignment.u_from_b_and_ub(b=b, ub=ub)

    original_u_matrix = original_u * sc.spatial.linear_transform(value=np.eye(3))
    sc.testing.assert_allclose(reconstructed_u, original_u_matrix)


@pytest.mark.parametrize('two_theta_value', [13, 45, 90, 180, -45])
def test_target_rotation_for_reflection(two_theta_value: int) -> None:
    hkl = sc.vector([1, 0, 1])
    two_theta = sc.scalar(float(two_theta_value), unit='deg')
    ei = sc.scalar(1.2, unit='meV')
    ef = sc.scalar(1.5, unit='meV')
    ub = sc.spatial.linear_transform(
        value=[[1.1, 0.5, 0.4], [0.0, 1.0, 1.8], [0, 0, 0.02]], unit='1/Å'
    )
    scattering_plane_normal = sc.vector([0, 1, 0])

    rotation = alignment.target_rotation_for_reflection(
        hkl=hkl,
        two_theta=two_theta,
        ei=ei,
        ef=ef,
        ub=ub,
        scattering_plane_normal=scattering_plane_normal,
        scattering_plane_reference=None,
    )
    q_lab = rotation * ub * hkl

    ki = momentum_from_energy(ei)
    kf = momentum_from_energy(ef)
    expected_q_lab = sc.spatial.as_vectors(
        -kf * sc.sin(two_theta), sc.scalar(0, unit='1/Å'), ki - kf * sc.cos(two_theta)
    )
    # Check that the calculated q_lab is parallel to the expected q_lab.
    # `expected_q_lab` only gives us a direction.
    sc.testing.assert_allclose(
        sc.norm(sc.cross(q_lab, expected_q_lab)),
        sc.scalar(0.0, unit='1/Å**2'),
        atol=sc.scalar(1e-12, unit='1/Å**2'),
    )
