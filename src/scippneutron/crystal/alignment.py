# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Alignment of single-crystal samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipp as sc

from ._linalg import invert_transform, transpose_matrix


@dataclass(frozen=True, slots=True)
class BraggPeaks:
    """Lattice coordinates and instrumental parameters for one or more Bragg peaks.

    All fields must have matching shapes.
    ``hkl`` and ``q`` must have dtype 'vector3' and ``r`` must have dtype 'rotation3'.
    """

    hkl: sc.Variable
    """The Miller indices of the peak."""
    q: sc.Variable
    """The lab-frame momentum transfer where the peak is observed."""
    r: sc.Variable
    """The sample rotation matrix corresponding the the observed momentum transfer."""

    def __post_init__(self) -> None:
        if self.r.dtype != sc.DType.rotation3:
            raise sc.DTypeError(
                f"Expected dtype 'rotation3' for 'r' but got {self.r.dtype}."
            )

        sizes = self.r.sizes

        for key in ('hkl', 'q'):
            item = getattr(self, key)
            if item.sizes != sizes:
                raise sc.DimensionError(
                    f"Mismatch between sizes: '{key}' has sizes {item.sizes} "
                    f"but expected {sizes}."
                )
            if item.dtype != sc.DType.vector3:
                raise sc.DTypeError(
                    f"Expected dtype 'vector3' for '{key}' but got {item.dtype}."
                )

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the peak coordinates."""
        return self.hkl.shape


def ub_from_3_peaks(peaks: BraggPeaks) -> sc.Variable:
    r"""Compute the UB matrix from three Bragg peaks.

    Given three Bragg peaks with known

    .. math::

        \vec{v}_i = \begin{pmatrix} h_i \\ k_i \\ l_i \end{pmatrix}

    that where observed at rotations :math:`R_i` and momentum transfers

    .. math::

        \vec{Q}_i = \begin{pmatrix} q_{ix} \\ q_{iy} \\ q_{iz} \end{pmatrix}

    define matrices

    .. math::

        Q_\nu &= \begin{pmatrix}
                    \vec{Q}_{\nu 1} & \vec{Q}_{\nu 2} & \vec{Q}_{\nu 3}
                 \end{pmatrix}, \quad
        \vec{Q}_{\nu i} = \frac{1}{2\pi} R_i^{-1} \vec{Q}_i, \\
        V &= \begin{pmatrix} \vec{v}_1 & \vec{v}_2 & \vec{v}_3 \end{pmatrix}

    With this we get

    .. math::

        \vec{Q}_i &= 2 \pi R_i U B \begin{pmatrix} h_i \\ k_i \\ l_i \end{pmatrix} \\
        \Rightarrow U B &= Q_\nu V^{-1}

    Parameters
    ----------
    peaks:
        The three Bragg peaks to use for the UB matrix calculation.

    Returns
    -------
    :
        The combined :math:`UB` matrix as a linear transform with unit 'one'.

    Raises
    ------
    ValueError:
        If the rotation matrices or :math:`V` cannot be inverted.
    """
    if peaks.shape != (3,):
        raise ValueError(f"Expected exactly 3 peaks, got {peaks.shape}.")

    try:
        r_inv_times_q = [
            invert_transform(r) * q for q, r in zip(peaks.q, peaks.r, strict=True)
        ]
    except ValueError as error:
        error.add_note("When inverting the sample rotation matrix.")
        raise
    q_mat = sc.spatial.linear_transform(
        value=np.array([x.value for x in r_inv_times_q]).T / (2 * np.pi),
        unit=peaks.q.unit / peaks.r.unit,
    )

    v_mat = sc.spatial.linear_transform(
        value=np.array([hkl.value for hkl in peaks.hkl]).T, unit=peaks.hkl.unit
    )
    try:
        v_mat_inv = invert_transform(v_mat)
    except ValueError as error:
        error.add_note("When inverting the V matrix (combination of hkl vectors).")
        raise

    return sc.to_unit(q_mat * v_mat_inv, "one")


def g_star_from_ub(ub: sc.Variable) -> sc.Variable:
    """Compute the reciprocal metric tensor G* from a UB matrix.

    The result is calculated via

    .. math::

        {(UB)}^T (UB) = B^T U^T U B = B^T B = G^*

    Using the fact that :math:`U` is a rotation matrix and thus :math:`U^T = U^{-1}`.

    Parameters
    ----------
    ub:
        The combined UB matrix as a linear transform.

    Returns
    -------
    :
        :math:`G^*`, the metric tensor of the reciprocal lattice.

    See Also
    --------
    ub_from_3_peaks:
        Compute the UB matrix from three known Bragg peaks.
    """
    return transpose_matrix(ub) * ub


def u_from_b_and_ub(b: sc.Variable, ub: sc.Variable) -> sc.Variable:
    """Compute the U matrix from B and the combined UB matrix.

    This function computes

    .. math::

        U = (UB) B^{-1}

    Parameters
    ----------
    b:
        The B matrix as a linear transform.
    ub:
        The combined UB matrix as a linear transform.

    Returns
    -------
    :
        The U matrix.

    Raises
    ------
    ValueError
        If B cannot be inverted.

    See Also
    --------
    ub_from_3_peaks:
        Compute the UB matrix from three known Bragg peaks.
    .lattice.build_b_matrix:
        Construct a B matrix from lattice parameters.
    """
    try:
        b_inv = invert_transform(b)
    except ValueError as error:
        error.add_note("When inverting a B matrix")
        raise
    # TODO do we want to convert to a rotation?
    #   that would require computing quaternions, see
    #   https://www.iri.upc.edu/files/scidoc/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf
    return ub * b_inv
