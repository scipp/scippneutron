# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Lattice parameters and conversions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipp as sc

from ._linalg import invert_transform


# TODO name?
@dataclass(frozen=True, slots=True)
class UnitCell:
    """A unit cell of a crystal.

    Encodes the lattice lengths a, b, c and
    the angles alpha, beta, gamma.
    """

    a: sc.Variable
    b: sc.Variable
    c: sc.Variable
    alpha: sc.Variable
    beta: sc.Variable
    gamma: sc.Variable

    def to_reciprocal(self) -> ReciprocalUnitCell:
        """Convert to reciprocal lattice parameters."""
        sin_alpha = sc.sin(self.alpha)
        cos_alpha = sc.cos(self.alpha)
        sin_beta = sc.sin(self.beta)
        cos_beta = sc.cos(self.beta)
        sin_gamma = sc.sin(self.gamma)
        cos_gamma = sc.cos(self.gamma)

        v = _v_alpha_beta_gamma(cos_alpha, cos_beta, cos_gamma)

        # TODO the PDF does not have 2pi here (eq. 39)
        a_star = sc.to_unit(sin_alpha / self.a / v * np.pi * 2, unit='1/angstrom')
        b_star = sc.to_unit(sin_beta / self.b / v * np.pi * 2, unit='1/angstrom')
        c_star = sc.to_unit(sin_gamma / self.c / v * np.pi * 2, unit='1/angstrom')
        alpha_star = sc.acos((cos_beta * cos_gamma - cos_alpha) / sin_beta / sin_gamma)
        beta_star = sc.acos((cos_gamma * cos_alpha - cos_beta) / sin_alpha / sin_gamma)
        gamma_star = sc.acos((cos_alpha * cos_beta - cos_gamma) / sin_beta / sin_alpha)

        return ReciprocalUnitCell(
            a_star, b_star, c_star, alpha_star, beta_star, gamma_star
        )


@dataclass(frozen=True, slots=True)
class ReciprocalUnitCell:
    """A reciprocal unit cell of a crystal.

    Encodes the reciprocal lattice lengths a*, b*, c* and
    the angles alpha*, beta*, gamma*.
    """

    a_star: sc.Variable
    b_star: sc.Variable
    c_star: sc.Variable
    alpha_star: sc.Variable
    beta_star: sc.Variable
    gamma_star: sc.Variable


def build_b_matrix(
    unit_cell: UnitCell, reciprocal_unit_cell: ReciprocalUnitCell | None = None
) -> sc.Variable:
    r"""Construct the B matrix from (reciprocal) lattice parameters.

    This function computes the B matrix defined via the conversion between Miller
    indices and lab-frame momentum transfer :math:`Q_l`:

    .. math::
        Q_l = 2\pi R U B \begin{pmatrix}
            h \\
            k \\
            l
        \end{pmatrix}

    The matrix is

    .. math::
        \mathbf{B} = \begin{pmatrix}
            a^* & b^* \cos \gamma^* & c^* \cos \beta^* \\
            0 & b^* \sin \gamma^* & -c^* \sin \beta^* \cos \alpha \\
            0 & 0 & 1 / c
        \end{pmatrix}

    Parameters
    ----------
    unit_cell:
        The lattice parameters of the crystal.
    reciprocal_unit_cell:
        The reciprocal lattice parameters of the crystal.
        If not provided, they will be computed from the unit cell.

    Returns
    -------
    :
        The B matrix.
    """
    uc = unit_cell
    ruc = reciprocal_unit_cell or unit_cell.to_reciprocal()
    zero = sc.scalar(0, unit='1/angstrom')
    raw = [
        [
            ruc.a_star,
            ruc.b_star * sc.cos(ruc.gamma_star),
            ruc.c_star * sc.cos(ruc.beta_star),
        ],
        [
            zero,
            ruc.b_star * sc.sin(ruc.gamma_star),
            -ruc.c_star * sc.sin(ruc.beta_star) * sc.cos(uc.alpha),
        ],
        [zero, zero, 2 * np.pi / uc.c],  # TODO no 2pi in PDF
    ]
    # TODO no 2pi in PDF
    return sc.spatial.linear_transform(
        unit='1/angstrom',
        value=[
            [x.to(unit='1/angstrom').value / (2 * np.pi) for x in row] for row in raw
        ],
    )


def lattice_params_from_g_star(g_star: sc.Variable) -> UnitCell:
    r"""Compute lattice parameters from a G* matrix.

    This function extracts the lattice parameters from the metric tensor
    of a reciprocal lattice defined by

    .. math::

        G^* = \begin{pmatrix}
            (a^*)^2 & a^* b^* \cos(\gamma^*) & c^* \cos(\beta^*) \\
            a^* b^* \cos(\gamma^*) & (b^*)^2 & b^* c^* \cos(\alpha^*) \\
            a^* c^* \cos(\beta^*) & b^* c^* \cos(\alpha^*) & (c^*)^2
        \end{pmatrix}

    This matrix is inverted to get the metric tensor of the direct lattice

    .. math::

        G = (G^*)^{-1} = \begin{pmatrix}
            a^2 & ab\cos(\gamma) & ac\cos(\beta) \\
            ab\cos(\gamma) & b^2 & bc\cos(\alpha) \\
            ac\cos(\beta) & bc\cos(\alpha) & c^2
        \end{pmatrix}

    The metric tensor must be symmetric, but that is not guaranteed
    when it is constructed from experimental data.
    So the matrix is symmetrized:

    .. math::

        \tilde{G} = \frac{1}{2} (G + G^T)

    Finally, the lattice parameters are extracted from :math:`\tilde{G}`.

    Parameters
    ----------
    g_star:
        A :math:`G^*` matrix.

    Returns
    -------
    :
        The lattice parameters described by ``g_star``.

    Raises
    ------
    ValueError
        If ``g_star`` is not invertible.

    See Also
    --------
    .alignment.g_star_from_ub:
        Compute the metric tensor :math:`G^*` from a :math:`UB` matrix.
    """
    try:
        g_matrix = invert_transform(g_star)
    except ValueError as error:
        error.add_note("When inverting the G* matrix to obtain G")
        raise

    g = g_matrix.value

    a = np.sqrt(g[0, 0])
    b = np.sqrt(g[1, 1])
    c = np.sqrt(g[2, 2])
    alpha = np.arccos((g[1, 2] + g[2, 1]) / (2 * b * c))
    beta = np.arccos((g[0, 2] + g[2, 0]) / (2 * a * c))
    gamma = np.arccos((g[0, 1] + g[1, 0]) / (2 * a * b))

    return UnitCell(
        a=sc.scalar(a, unit=sc.sqrt(g_matrix.unit)).to(unit='angstrom'),
        b=sc.scalar(b, unit=sc.sqrt(g_matrix.unit)).to(unit='angstrom'),
        c=sc.scalar(c, unit=sc.sqrt(g_matrix.unit)).to(unit='angstrom'),
        alpha=sc.scalar(alpha, unit='rad'),
        beta=sc.scalar(beta, unit='rad'),
        gamma=sc.scalar(gamma, unit='rad'),
    )


def _v_alpha_beta_gamma(
    cos_alpha: sc.Variable, cos_beta: sc.Variable, cos_gamma: sc.Variable
) -> sc.Variable:
    """Calculate the normalized volume of a unit cell.

    This computes V_{alpha beta gamma} = Volume / (abc)
    where Volume = a * (b x c)
    """
    return sc.sqrt(
        1
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2 * cos_alpha * cos_beta * cos_gamma
    )
