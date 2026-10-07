# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Alignment of single-crystal samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipp as sc
from scipy.linalg import solve

from ...conversion.tof import momentum_from_energy
from ._linalg import invert_transform, transpose_matrix

# TODO
# u: along beam
# v: perp s.t. u x v points up
# only direction matters
# in terms of miller indices
#
#
# - Needs motor angles from peak
# - indices from peak position


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
    """The sample rotation (matrix) corresponding the the observed momentum transfer."""

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


@dataclass(frozen=True, slots=True)
class UAndB:
    """A U and a B matrix."""

    u: sc.Variable
    """A U matrix transforming crystal coordinates into lab frame coordinates."""
    b: sc.Variable
    """A B matrix transforming miller indices into crystal coordinates."""


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
        q_nu = invert_transform(peaks.r) * peaks.q / (2 * np.pi)
    except ValueError as error:
        error.add_note("When inverting the sample rotation matrix.")
        raise

    try:
        # Determine UB from
        #   Q_nu = UB * V
        # by solving the linear equations
        #   Q_nu^T = V^T * (UB)^T
        # for (UB)^T and then transposing the result.
        # We use this instead of inverting V because it is more numerically stable.
        ub_array = solve(peaks.hkl.values, q_nu.values)
    except ValueError as error:
        error.add_note(
            "When solving for the UB matrix. Check for collinear hkl vectors."
        )
        raise

    return sc.to_unit(
        sc.spatial.linear_transform(value=ub_array.T, unit=q_nu.unit / peaks.hkl.unit),  # type: ignore[operator]
        "one",
    )


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

        Note that this is a :attr:`sc.DType.linear_transform` and not a clean rotation
        because the inputs are typically inaccurate measurements. So the result may
        have a scaling component in addition to the rotation.

    Raises
    ------
    ValueError
        If B cannot be inverted.

    See Also
    --------
    ub_from_3_peaks:
        Compute the UB matrix from three known Bragg peaks.
    .lattice.b_matrix_from_lattice_parameters:
        Construct a B matrix from lattice parameters.
    """
    try:
        b_inv = invert_transform(b)
    except ValueError as error:
        error.add_note("When inverting a B matrix")
        raise
    return ub * b_inv


def target_rotation_from_ub(
    *,
    hkl: sc.Variable,
    two_theta: sc.Variable,
    ei: sc.Variable,
    ef: sc.Variable,
    ub: sc.Variable,
) -> sc.Variable:
    ki = momentum_from_energy(ei)
    kf = momentum_from_energy(ef)

    q = sc.sqrt(ki**2 + kf**2 - 2 * ki * kf * sc.cos(two_theta))
    t = _construct_orthogonal_target_system(hkl=hkl, ub=ub)


def _construct_orthogonal_target_system(
    *, hkl: sc.Variable, ub: sc.Variable
) -> sc.Variable:
    eps = sc.scalar(1e-5)

    t1 = ub * hkl
    # TODO handle in-plane vectors
    # if np.abs(np.dot(t1, plane_normal)) < eps:
    #     # t1 in plane
    #     t3 = plane_normal
    #     t2 = np.cross(t3, t1)
    # elif np.linalg.norm(np.cross(plane_normal, t1)) < ZERO:
    #     # oops, t1 along plane_normal
    #     if in_plane_ref is None:
    #         raise ValueError(
    #             "Peak ({:.3g}, {:.3g}, {:.3g}) is perpendicular to the horizaontal scattering plane. in_plane_ref is required to determine R matrix.".format(
    #                 *hkl
    #             )
    #         )
    #     t2 = in_plane_ref
    #     t3 = np.cross(t1, t2)
    # else:
    # t1 not in plane, need to change tilts
    # t2p = np.cross(plane_normal, t1)
    # t3 = np.cross(t1, t2p)
    # t2 = np.cross(t3, t1)
    t2 = sc.cross(plane_normal, t1)
    t3 = np.cross(t1, t2)

    return sc.spatial.linear_transform(
        value=np.c_[
            _normed_vector_values(t1),
            _normed_vector_values(t2),
            _normed_vector_values(t3),
        ]
    )


def _normed_vector_values(vector: sc.Variable) -> npt.NDArray[np.float64]:
    return sc.to_unit(vector / sc.norm(vector), 'one').values
