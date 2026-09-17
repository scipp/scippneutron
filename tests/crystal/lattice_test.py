# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import dataclasses

import scipp as sc
import scipp.testing

from scippneutron.crystal import lattice
from scippneutron.crystal._linalg import transpose_matrix


def test_unit_cell_to_reciprocal() -> None:
    unit_cell = lattice.UnitCell(
        a=sc.scalar(1.0, unit='angstrom'),
        b=sc.scalar(0.02, unit='nm'),
        c=sc.scalar(34, unit='pm'),
        alpha=sc.scalar(90.0, unit='degree'),
        beta=sc.scalar(80.0, unit='degree'),
        gamma=sc.scalar(1.2, unit='radian'),
    )
    reciprocal = unit_cell.to_reciprocal()
    assert reciprocal.a_star.unit == '1/angstrom'
    assert reciprocal.b_star.unit == '1/angstrom'
    assert reciprocal.b_star.unit == '1/angstrom'
    assert reciprocal.alpha_star.unit == 'radian'
    assert reciprocal.beta_star.unit == 'radian'
    assert reciprocal.gamma_star.unit == 'radian'


def test_b_matrix_from_unit_cell() -> None:
    unit_cell = lattice.UnitCell(
        a=sc.scalar(1.0, unit='angstrom'),
        b=sc.scalar(0.02, unit='nm'),
        c=sc.scalar(34, unit='pm'),
        alpha=sc.scalar(90.0, unit='degree'),
        beta=sc.scalar(80.0, unit='degree'),
        gamma=sc.scalar(1.2, unit='radian'),
    )
    b = lattice.build_b_matrix(unit_cell)
    # There is little we can test here, so just check that the function returns
    # and that the unit is correct.
    assert b.unit == '1/angstrom'


def test_lattice_params_from_g_star() -> None:
    original = lattice.UnitCell(
        a=sc.scalar(1.0, unit='angstrom'),
        b=sc.scalar(1.4, unit='angstrom'),
        c=sc.scalar(6.4, unit='angstrom'),
        alpha=sc.scalar(1.05, unit='rad'),
        beta=sc.scalar(0.7, unit='rad'),
        gamma=sc.scalar(1.3, unit='rad'),
    )
    b = lattice.build_b_matrix(original)
    g_star = transpose_matrix(b) * b

    reconstructed = lattice.lattice_params_from_g_star(g_star)

    for field in dataclasses.fields(original):
        sc.testing.assert_allclose(
            getattr(reconstructed, field.name),
            getattr(original, field.name),
        )
