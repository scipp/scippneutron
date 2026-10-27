# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
import scipp.testing
from hypothesis import given, settings
from hypothesis import strategies as st

from scippneutron.conversion.tof import momentum_from_energy
from scippneutron.spectroscopy.crystal import alignment, lattice
from scippneutron.spectroscopy.geometry import (
    ScatteringPlaneMiller,
    ScatteringPlaneSampleHolder,
)
from scippneutron.spectroscopy.goniometer import Goniometer, GoniometerAxis


@settings(max_examples=50)
@given(
    angles=st.lists(st.floats(min_value=-180, max_value=180), min_size=3, max_size=3)
)
@pytest.mark.parametrize('hkl', [[4, 0, 0], [0, 0, 6]])
def test_u_from_1_peak_and_scattering_plane(
    hkl: list[float], angles: list[float]
) -> None:
    inf = sc.scalar(np.inf, unit='rad')
    goniometer = Goniometer(
        axes=[
            GoniometerAxis(name='sgu', axis="+x", min=-inf, max=inf),
            GoniometerAxis(name='sgl', axis="-z", min=-inf, max=inf),
            GoniometerAxis(name='omega', axis="+y", min=-inf, max=inf),
        ]
    )
    peaks = alignment.BraggPeaks.from_q_lab(
        q_lab=sc.vector([0.0007, 0.0004, 1.5335], unit='1/Å'),
        hkl=sc.vector(hkl),
        r=goniometer.rotation_for(sc.array(dims=['angles'], values=angles, unit='deg')),
    )
    plane = ScatteringPlaneMiller.from_in_plane_vectors(
        sc.vector([0, 0, 1]),  # order s.t. normal points along +y
        sc.vector([1, 0, 0]),
    )
    b = lattice.b_matrix_from_lattice_parameters(
        lattice.LatticeParameters(
            a=sc.scalar(5.0556, unit='Å'),
            b=sc.scalar(5.0556, unit='Å'),
            c=sc.scalar(24.5980, unit='Å'),
            alpha=sc.scalar(90.0, unit='deg'),
            beta=sc.scalar(90.0, unit='deg'),
            gamma=sc.scalar(120.0, unit='deg'),
        )
    )

    u = alignment.u_from_1_peak_and_scattering_plane(
        peaks=peaks,
        scattering_plane=plane,
        b=b,
    )

    computed = peaks.r * u * b * peaks.hkl
    actual = peaks.q_lab
    sc.testing.assert_allclose(computed / sc.norm(computed), actual / sc.norm(actual))


def test_u_from_2_peaks() -> None:
    peaks = alignment.BraggPeaks.from_q_sample(
        q_sample=sc.vectors(
            dims=['peak'],
            values=[
                [2.0000e-04, -3.0000e-04, 1.5328e00],
                [1.3008e00, -6.0700e-01, 2.5520e-01],
            ],
            unit='1/Å',
        ),
        hkl=sc.vectors(dims=['peak'], values=[[0, 0, 6], [1, 0, 1]]),
    )
    b = lattice.b_matrix_from_lattice_parameters(
        lattice.LatticeParameters(
            a=sc.scalar(5.0556, unit='Å'),
            b=sc.scalar(5.0556, unit='Å'),
            c=sc.scalar(24.5980, unit='Å'),
            alpha=sc.scalar(90.0, unit='deg'),
            beta=sc.scalar(90.0, unit='deg'),
            gamma=sc.scalar(120.0, unit='deg'),
        )
    )

    u = alignment.u_from_2_peaks(peaks=peaks, b=b)

    computed = u * b * peaks.hkl
    actual = peaks.q_sample
    sc.testing.assert_allclose(
        computed / sc.norm(computed), actual / sc.norm(actual), rtol=sc.scalar(2e-5)
    )


def test_ub_from_3_peaks() -> None:
    peaks = alignment.BraggPeaks.from_q_sample(
        hkl=sc.vectors(dims=['p'], values=[[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
        q_sample=sc.vectors(
            dims=['p'], values=[[1, 2, 3], [4, 5, 6], [-1, -2, -3]], unit='1/Å'
        ),
    )

    ub = alignment.ub_from_3_peaks(peaks)
    assert ub.dtype == sc.DType.linear_transform3

    for hkl, q_sample in zip(peaks.hkl, peaks.q_sample, strict=True):
        q_calc = ub * hkl * (2 * np.pi)
        sc.testing.assert_allclose(q_calc, q_sample)


def test_ub_from_many_peaks_with_three_peaks() -> None:
    # The result is best with 3 linearly-independent vectors.
    peaks = alignment.BraggPeaks.from_q_sample(
        hkl=sc.vectors(dims=['p'], values=[[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
        q_sample=sc.vectors(
            dims=['p'], values=[[1, 2, 3], [4, 5, 6], [-1, -2, -3]], unit='1/Å'
        ),
    )

    ub = alignment.ub_from_many_peaks(peaks)
    assert ub.dtype == sc.DType.linear_transform3

    for hkl, q_sample in zip(peaks.hkl, peaks.q_sample, strict=True):
        q_calc = ub * hkl * (2 * np.pi)
        sc.testing.assert_allclose(q_calc, q_sample)


def test_ub_from_many_peaks_with_many_peaks() -> None:
    # Not all peaks can be reproduced exactly when they are linearly-dependent.
    # => need a high tolerance
    peaks = alignment.BraggPeaks.from_q_sample(
        hkl=sc.vectors(
            dims=['p'],
            values=[
                [1, -0, 1],
                [-0, -0, 6],
                [2, 1, -0],
                [-1, -0, 11],
                [2, 1, 6],
            ],
        ),
        q_sample=sc.vectors(
            dims=['p'],
            values=[
                [1.3008, -6.0700e-01, 2.5520e-01],
                [2e-04, -3e-04, 1.5328],
                [2.4763, 2.1720e-01, 1e-04],
                [-1.3014, 6.0670e-01, 2.8098],
                [2.4765, 2.1720e-01, 1.5329],
            ],
            unit='1/Å',
        ),
    )

    ub = alignment.ub_from_many_peaks(peaks)
    assert ub.dtype == sc.DType.linear_transform3

    for hkl, q_sample in zip(peaks.hkl, peaks.q_sample, strict=True):
        q_calc = ub * hkl * (2 * np.pi)
        sc.testing.assert_allclose(q_calc, q_sample, atol=sc.scalar(1e-3, unit='1/Å'))


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
    scattering_plane = ScatteringPlaneSampleHolder.from_normal_and_reference(
        normal=sc.vector([0, 1, 0]), reference=sc.vector([0, 0, 1])
    )

    rotation = alignment.target_rotation_for_reflection(
        hkl=hkl,
        two_theta=two_theta,
        ei=ei,
        ef=ef,
        ub=ub,
        scattering_plane=scattering_plane,
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
