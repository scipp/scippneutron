# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Lattice parameters and conversions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipp as sc
from scipy.linalg import solve

from ._linalg import align_vectors, invert_transform, orthonormal_system_from_xy


@dataclass(frozen=True, slots=True)
class LatticeParameters:
    """Parameters for a forward crystal lattice.

    Encodes the lattice lengths a, b, c and
    the angles alpha, beta, gamma.
    """

    a: sc.Variable
    b: sc.Variable
    c: sc.Variable
    alpha: sc.Variable
    beta: sc.Variable
    gamma: sc.Variable

    def to_reciprocal(self) -> ReciprocalLatticeParameters:
        """Convert to reciprocal lattice parameters."""
        sin_alpha = sc.sin(self.alpha)
        cos_alpha = sc.cos(self.alpha)
        sin_beta = sc.sin(self.beta)
        cos_beta = sc.cos(self.beta)
        sin_gamma = sc.sin(self.gamma)
        cos_gamma = sc.cos(self.gamma)

        v = _v_alpha_beta_gamma(cos_alpha, cos_beta, cos_gamma)

        a_star = sc.to_unit(sin_alpha / self.a / v, unit='1/angstrom')
        b_star = sc.to_unit(sin_beta / self.b / v, unit='1/angstrom')
        c_star = sc.to_unit(sin_gamma / self.c / v, unit='1/angstrom')
        alpha_star = sc.acos((cos_beta * cos_gamma - cos_alpha) / sin_beta / sin_gamma)
        beta_star = sc.acos((cos_gamma * cos_alpha - cos_beta) / sin_alpha / sin_gamma)
        gamma_star = sc.acos((cos_alpha * cos_beta - cos_gamma) / sin_beta / sin_alpha)

        return ReciprocalLatticeParameters(
            a_star, b_star, c_star, alpha_star, beta_star, gamma_star
        )


@dataclass(frozen=True, slots=True)
class ReciprocalLatticeParameters:
    """Parameters of a reciprocal crystal lattice.

    Encodes the reciprocal lattice lengths a*, b*, c* and
    the angles alpha*, beta*, gamma*.
    """

    a_star: sc.Variable
    b_star: sc.Variable
    c_star: sc.Variable
    alpha_star: sc.Variable
    beta_star: sc.Variable
    gamma_star: sc.Variable


def b_matrix_from_lattice_parameters(
    lattice_parameters: LatticeParameters,
    reciprocal_lattice_parameters: ReciprocalLatticeParameters | None = None,
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
    lattice_parameters:
        The lattice parameters of the crystal.
    reciprocal_lattice_parameters:
        The reciprocal lattice parameters of the crystal.
        If not provided, they will be computed from the unit cell.

    Returns
    -------
    :
        The B matrix.
    """
    p = lattice_parameters
    rp = reciprocal_lattice_parameters or lattice_parameters.to_reciprocal()
    zero = sc.scalar(0, unit='1/angstrom')
    raw = [
        [
            rp.a_star,
            rp.b_star * sc.cos(rp.gamma_star),
            rp.c_star * sc.cos(rp.beta_star),
        ],
        [
            zero,
            rp.b_star * sc.sin(rp.gamma_star),
            -rp.c_star * sc.sin(rp.beta_star) * sc.cos(p.alpha),
        ],
        [zero, zero, 1 / p.c],
    ]
    return sc.spatial.linear_transform(
        unit='1/angstrom',
        value=[[x.to(unit='1/angstrom').value for x in row] for row in raw],
    )


def u_matrix_from_uv_vectors(
    u: sc.Variable, v: sc.Variable, b: sc.Variable, *, error_threshold: float = 1e-5
) -> sc.Variable:
    r"""Calculate a U matrix from u and v vectors and a B matrix.

    The :math:`\vec{u}` and :math:`\vec{v}` vectors define the z and x axes of
    the sample coordinate system. That is, given

    .. math::

        \vec{t}_1 = B \vec{u},\quad \vec{t}_2 = B \vec{v}

    we get

    .. math::

        U \hat{t}_1 &= {\begin{pmatrix}0 & 0 & 1\end{pmatrix}}^T \\
        U \hat{t}_2 &= {\begin{pmatrix}1 & 0 & 0\end{pmatrix}}^T

    Extending this with a vector :math:`\vec{w} = \vec{u} \times \vec{v}` that
    defines :math:`\vec{t}_3`, we have

    .. math::

        U T = O,\quad T = \begin{pmatrix}\hat{t}_1 & \hat{t}_2 & \hat{t}_3\end{pmatrix}

    with

    .. math::

        O = \begin{pmatrix}
            0 & 1 & 0 \\
            0 & 0 & 1 \\
            1 & 0 & 0
        \end{pmatrix}

    This function computes :math:`U` by finding a rotation that maps
    :math:`T` onto :math:`O`.

    Parameters
    ----------
    u:
        A vector in Miller indices along the incoming beam (for R = 1).
    v:
        A vector in Miller indices in the scattering plane.
        Ideally, :math:`\vec{u} \cdot \vec{v}` should point up.
    b:
        The B matrix of the sample.
    error_threshold:
        Raise a warning when the error (squared residuals) of computing the target
        rotation is larger than this threshold.

    Returns
    -------
    :
        The U 'matrix' as a :class:`scipp.DType.rotation3`.

    See Also
    --------
    .uvw_vectors_from_ub_matrix:
        For computing u and v vectors from a UB matrix.
    """

    t = orthonormal_system_from_xy(b * u, b * v)
    o = sc.vectors(dims=['vector'], values=[[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    return align_vectors(
        o,
        t,
        a_name="O",
        b_name="T",
        error_threshold=error_threshold,
    )


def uvw_vectors_from_ub_matrix(ub: sc.Variable) -> sc.Variable:
    r"""Compute u, v, and w vectors from a UB matrix.

    This functions solves

    .. math::

        UB \begin{pmatrix}\vec{u} & \vec{v} & \vec{w}\end{pmatrix} = O

    for the vectors :math:`\vec{u}`, :math:`\vec{v}`, and :math:`\vec{w}` given

    with

    .. math::

        O = \begin{pmatrix}
            0 & 1 & 0 \\
            0 & 0 & 1 \\
            1 & 0 & 0
        \end{pmatrix}

    Parameters
    ----------
    ub:
        A UB matrix.

    Returns
    -------
    :
        Three vectors u, v, w.

    See Also
    --------
    .u_matrix_from_uv_vectors:
        For computing a U matrix from u and v vectors.
    """
    o = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T
    return sc.vectors(
        dims=['vector'], values=solve(ub.values, o).T, unit=sc.reciprocal(ub.unit)
    )


def lattice_params_from_g_star(g_star: sc.Variable) -> LatticeParameters:
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

    return LatticeParameters(
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
