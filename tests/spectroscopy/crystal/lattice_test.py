# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import dataclasses

import numpy as np
import scipp as sc
import scipp.testing

from scippneutron.spectroscopy.crystal import lattice
from scippneutron.spectroscopy.crystal._linalg import transpose_matrix


def test_lattice_parameters_to_reciprocal() -> None:
    lattice_parameters = lattice.LatticeParameters(
        a=sc.scalar(1.0, unit='angstrom'),
        b=sc.scalar(0.02, unit='nm'),
        c=sc.scalar(34, unit='pm'),
        alpha=sc.scalar(90.0, unit='degree'),
        beta=sc.scalar(80.0, unit='degree'),
        gamma=sc.scalar(1.2, unit='radian'),
    )
    reciprocal = lattice_parameters.to_reciprocal()
    assert reciprocal.a_star.unit == '1/angstrom'
    assert reciprocal.b_star.unit == '1/angstrom'
    assert reciprocal.b_star.unit == '1/angstrom'
    assert reciprocal.alpha_star.unit == 'radian'
    assert reciprocal.beta_star.unit == 'radian'
    assert reciprocal.gamma_star.unit == 'radian'


def test_b_matrix_from_lattice_parameters() -> None:
    lattice_parameters = lattice.LatticeParameters(
        a=sc.scalar(1.0, unit='angstrom'),
        b=sc.scalar(0.02, unit='nm'),
        c=sc.scalar(34, unit='pm'),
        alpha=sc.scalar(90.0, unit='degree'),
        beta=sc.scalar(80.0, unit='degree'),
        gamma=sc.scalar(1.2, unit='radian'),
    )
    b = lattice.b_matrix_from_lattice_parameters(lattice_parameters)
    # There is little we can test here, so just check that the function returns
    # and that the unit is correct.
    assert b.unit == '1/angstrom'


def test_uv_vectors_roundtrip() -> None:
    """Test that conversions between uv, vectors and UB matrix are consistent.

    This test starts with a UB matrix, computes u,v,w vectors, and recomputes a
    U matrix. It would not work the other way round (starting with u,v) because
    the vector lengths and precise direction of v are lost when computing U.
    """
    lattice_parameters = lattice.LatticeParameters(
        a=sc.scalar(1.0, unit='angstrom'),
        b=sc.scalar(0.02, unit='nm'),
        c=sc.scalar(34, unit='pm'),
        alpha=sc.scalar(90.0, unit='degree'),
        beta=sc.scalar(80.0, unit='degree'),
        gamma=sc.scalar(1.2, unit='radian'),
    )
    b = lattice.b_matrix_from_lattice_parameters(lattice_parameters)
    quat = np.array([1.2, 0.4, -0.7, 2.3])
    u_matrix = sc.spatial.rotation(value=quat / np.linalg.norm(quat))

    uvw = lattice.uvw_vectors_from_ub_matrix(u_matrix * b)
    reconstructed_u = lattice.u_matrix_from_uv_vectors(uvw[0], uvw[1], b)

    assert uvw.unit == 'Å'
    sc.testing.assert_allclose(u_matrix, reconstructed_u)


def test_lattice_params_from_g_star() -> None:
    original = lattice.LatticeParameters(
        a=sc.scalar(1.0, unit='angstrom'),
        b=sc.scalar(1.4, unit='angstrom'),
        c=sc.scalar(6.4, unit='angstrom'),
        alpha=sc.scalar(1.05, unit='rad'),
        beta=sc.scalar(0.7, unit='rad'),
        gamma=sc.scalar(1.3, unit='rad'),
    )
    b = lattice.b_matrix_from_lattice_parameters(original)
    g_star = transpose_matrix(b) * b

    reconstructed = lattice.lattice_params_from_g_star(g_star)

    for field in dataclasses.fields(original):
        sc.testing.assert_allclose(
            getattr(reconstructed, field.name),
            getattr(original, field.name),
        )
