# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Alignment of single-crystal samples."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipp as sc
from scipy.linalg import solve
from scipy.spatial.transform import Rotation

from ...conversion.tof import momentum_from_energy
from ..geometry import ScatteringPlaneMiller, ScatteringPlaneSampleHolder
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

# TODO make symbols internally consistent
#   - maybe avoid using v for miller indices to avoid confusion if u-v


@dataclass(frozen=True, slots=True)
class BraggPeaks:
    """Lattice coordinates and instrumental parameters for one or more Bragg peaks.

    All fields must have matching shapes.
    ``hkl`` and ``q_lab`` must have dtype 'vector3'
    and ``r`` must have dtype 'rotation3'.
    """

    hkl: sc.Variable
    """The Miller indices of the peak."""
    q_lab: sc.Variable
    """The lab-frame momentum transfer where the peak is observed."""
    r: sc.Variable
    """The sample rotation (matrix) corresponding the the observed momentum transfer."""

    def __post_init__(self) -> None:
        if self.r.dtype != sc.DType.rotation3:
            raise sc.DTypeError(
                f"Expected dtype 'rotation3' for 'r' but got {self.r.dtype}."
            )

        sizes = self.r.sizes

        for key in ('hkl', 'q_lab'):
            item = getattr(self, key)
            if item.sizes != sizes:
                raise sc.DimensionError(
                    f"Mismatch between sizes: `{key}` has sizes {item.sizes} "
                    f"but expected {sizes} based on `r`."
                )
            if item.dtype != sc.DType.vector3:
                raise sc.DTypeError(
                    f"Expected dtype 'vector3' for `{key}` but got {item.dtype}."
                )

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the peak coordinates."""
        return self.hkl.shape


def u_from_1_peak_and_scattering_plane(
    peaks: BraggPeaks,
    b: sc.Variable,
    scattering_plane: ScatteringPlaneMiller,
    *,
    error_threshold: float = 1e-5,
) -> sc.Variable:
    r"""Compute U from one Bragg peak, a scattering plane, and a B matrix.

    This function solves

    .. math::

        \vec{Q}_{\text{lab}} = 2 \pi R U B \begin{pmatrix} h \\ k \\ l \end{pmatrix}

    for the rotation :math:`U`.
    All other quantities are provides as function arguments.

    The above equation does not uniquely determine :math:`U`. This is resolved by
    placing the x-axis in the sample holder system :math:`R^{-1} \vec{Q}_{\text{lab}}`
    in the given scattering plane. The y-axis is then as close to the lab-frame
    y-axis as possible.

    Attention
    ---------
    The given peak (hkl) must lie in the given scattering plane.
    Otherwise, :math:`U` would bot be a pure rotation.
    But since this function only returns a rotation, any other contribution
    to :math:`U` would be lost.

    Parameters
    ----------
    peaks:
        The Bragg peak to use for the U matrix calculation.
        Must contain exactly one peak.
    b:
        The B matrix as a linear transform.
    scattering_plane:
        The scattering plane **in Miller indices**.
        Must contain the peak.
    error_threshold:
        Raise a warning when the error (squared residuals) of computing the target
        rotation is larger than this threshold.

    Returns
    -------
    :
        The best estimate for the rotation :math:`U`.

    Warns
    -----
    RuntimeWarning
        The target rotation is determined using a minimization algorithm which returns
        approximate results. When the error of the result exceeds ``error_threshold``,
        a :class:`RuntimeWarning` is raised.
        See the underlying
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html>`_.
    """
    if peaks.shape not in ((), (1,)):
        raise ValueError(f"Expected exactly 1 peak, got {peaks.shape}.")
    if not isinstance(scattering_plane, ScatteringPlaneMiller):
        raise TypeError(
            f"Expected a ScatteringPlaneMiller, got {type(scattering_plane)}."
        )

    hkl = peaks.hkl.squeeze()
    q_lab = peaks.q_lab.squeeze()
    r = peaks.r.squeeze()

    # Construct a basis in the cartesian sample space (B * hkl)
    # Order: (given_hkl, in_plane, out_of_plane)
    c1, c2 = (
        sc.dot(scattering_plane.vector1, hkl),
        sc.dot(scattering_plane.vector2, hkl),
    )
    if abs(c1) > abs(c2):
        q_hkl2p = b * scattering_plane.vector2 * np.sign(c1.value)
    else:
        q_hkl2p = b * scattering_plane.vector1 * np.sign(c2.value)
    q_hkl1 = b * hkl
    q_hkl3 = sc.cross(q_hkl1, q_hkl2p)
    q_hkl2 = sc.cross(q_hkl3, q_hkl1)
    q_hkl = _normed_coord_basis(q_hkl1, q_hkl2, q_hkl3)

    # Construct a basis in the sample holder space (R^-1 * q_lab)
    # Order: (given_q_lab, close_to_xz_plane, y)
    r_inv = invert_transform(r)
    q_holder1 = r_inv * q_lab
    q_holder3 = r_inv * sc.vector([0, 1, 0])
    q_holder2 = sc.cross(q_holder3, q_holder1)
    q_holder3 = sc.cross(q_holder1, q_holder2)
    q_holder = _normed_coord_basis(q_holder1, q_holder2, q_holder3)

    rotation, error = Rotation.align_vectors(q_holder, q_hkl)
    if error > error_threshold:
        warnings.warn(
            f"The target rotation to align a sample with hkl={hkl:c} could only be "
            f"computed up to an error of {error} (threshold is {error_threshold}).",
            RuntimeWarning,
            stacklevel=2,
        )
    return sc.spatial.rotation(value=rotation.as_quat())


def ub_from_3_peaks(peaks: BraggPeaks) -> sc.Variable:
    r"""Compute the UB matrix from three Bragg peaks.

    Given three Bragg peaks with known

    .. math::

        \vec{v}_i = \begin{pmatrix} h_i \\ k_i \\ l_i \end{pmatrix}

    that were observed at rotations :math:`R_i` and momentum transfers

    .. math::

        \vec{Q}_{\text{lab}\,i} = \begin{pmatrix}
            q_{ix} \\ q_{iy} \\ q_{iz}
        \end{pmatrix}

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
        q_nu = invert_transform(peaks.r) * peaks.q_lab / (2 * np.pi)
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
        ub_array = solve(peaks.hkl.values, q_nu.values).T
    except ValueError as error:
        error.add_note(
            "When solving for the UB matrix. Check for collinear hkl vectors."
        )
        raise

    return sc.to_unit(
        sc.spatial.linear_transform(value=ub_array, unit=q_nu.unit / peaks.hkl.unit),  # type: ignore[operator]
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


def target_rotation_for_reflection(
    *,
    hkl: sc.Variable,
    two_theta: sc.Variable,
    ei: sc.Variable,
    ef: sc.Variable,
    ub: sc.Variable,
    scattering_plane: ScatteringPlaneSampleHolder,
    epsilon: float = 1e-5,
    error_threshold: float = 1e-5,
) -> sc.Variable:
    r"""Compute the rotation that places the given reflection at the given $2 \theta$.

    Given a known sample UB matrix and scattering plane information, this function
    computes the sample rotation that places a reflection with Miller indices ``hkl``
    at a specific scattering angle $2 \theta$.

    This problem is under determined. This function resolves this issue by computing
    the rotation with the minimal tilt of the sample while placing the target
    in or close to the horizontal plane ($\phi \approx 0$ or $\phi \approx \pi$).

    Parameters
    ----------
    hkl:
        Miller indices of the target reflection.
    two_theta
        Scattering angle $2 \theta$ where the target reflection should be visible.
        **Assumes that the incident beam is along z.**
    ei:
        Incident energy.
    ef:
        Final energy.
    ub:
        UB matrix of the sample.
    scattering_plane:
        The scattering plane.
    epsilon:
        Threshold for checking whether vectors align with the scattering plane
        or scattering plane normal.
    error_threshold:
        Raise a warning when the error (squared residuals) of computing the target
        rotation is larger than this threshold.

    Returns
    -------
    :
        The best estimate for the rotation that places the target peak at the given
        $2 \theta$.

    Warns
    -----
    RuntimeWarning
        The target rotation is determined using a minimization algorithm which returns
        approximate results. When the error of the result exceeds ``error_threshold``,
        a :class:`RuntimeWarning` is raised.
        See the underlying
        `SciPy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html>`_.
    """
    if not isinstance(scattering_plane, ScatteringPlaneSampleHolder):
        raise TypeError(
            f"Expected a ScatteringPlaneSampleHolder, got {type(scattering_plane)}."
        )

    ki = momentum_from_energy(ei)
    kf = momentum_from_energy(ef)
    two_theta = two_theta.to(dtype=float)  # to support integer inputs

    q = sc.sqrt(ki**2 + kf**2 - 2 * ki * kf * sc.cos(two_theta))
    if q.value == 0:
        # This should not happen in practice because there are no detectors two_theta=0.
        raise ValueError(
            "Invalid scattering condition: The momentum transfer Q is zero."
        )

    zero = sc.scalar(0.0, unit=ki.unit)
    q_list = [
        [-kf / q * sc.sin(two_theta), zero, ki / q - kf / q * sc.cos(two_theta)],
        [ki / q - kf / q * sc.cos(two_theta), zero, kf / q * sc.sin(two_theta)],
        [zero, sc.scalar(1.0, unit=ki.unit), zero],
    ]
    # All elements have unit 'one' because ki, kf, and q have the same unit.
    q_lab = np.array([[x.value for x in row] for row in q_list])

    t = _construct_orthogonal_target_system(
        hkl=hkl,
        ub=ub,
        scattering_plane=scattering_plane,
        epsilon=epsilon,
    )

    rotation, error = Rotation.align_vectors(q_lab, t)
    if error > error_threshold:
        warnings.warn(
            f"The target rotation to align a sample with hkl={hkl:c} could only be "
            f"computed up to an error of {error} (threshold is {error_threshold}).",
            RuntimeWarning,
            stacklevel=2,
        )
    return sc.spatial.rotation(value=rotation.as_quat())


def _construct_orthogonal_target_system(
    *,
    hkl: sc.Variable,
    ub: sc.Variable,
    scattering_plane: ScatteringPlaneSampleHolder,
    epsilon: float = 1e-5,
) -> npt.NDArray[np.float64]:
    """Construct an orthogonal coordinate system for a target rotation.

    The implementation is based on ``rot_matrix_with_minimal_tilt`` from TAVI:
    https://github.com/neutrons/TAVI/blob/905c5f5dcd372f9eef2c9d88a37323fd0edb6276/src/tavi/library/geometry/oriented_lattice.py#L399

    Parameters
    ----------
    hkl:
        Target peak.
    ub:
        Sample UB matrix.
    scattering_plane:
        The horizontal scattering plane.
    epsilon:
        Tolerance for considering a vector to be perpendicular or parallel to another.

    Returns
    -------
    :
        A numpy array encoding the three basis vectors as rows.
    """
    eps = sc.scalar(epsilon)
    t1 = ub * hkl

    if abs(sc.dot(t1, scattering_plane.normal).value) < eps:
        # t1 in plane
        t3 = scattering_plane.normal
        t2 = sc.cross(t3, t1)
    elif sc.norm(sc.cross(t1, scattering_plane.normal)).value < eps:
        # Calculate R when t1 is along scattering_plane_normal.
        # The easiest way to construct three perp axes is to use t2 as in-plane axis.
        # This already considers a 90 degree rotation to bring t1 in-plane.
        t2 = scattering_plane.vector1
        t3 = sc.cross(t1, t2)  # t3 along y
    else:
        # t1 not in plane.
        # cross(scattering_plane_normal, t1) incorporates a minimal rotation along
        # t2 to bring t1 in-plane.
        t2 = sc.cross(scattering_plane.normal, t1)
        t3 = sc.cross(t1, t2)

    return _normed_coord_basis(t1, t2, t3)


def _normed_coord_basis(
    a: sc.Variable, b: sc.Variable, c: sc.Variable
) -> npt.NDArray[np.float64]:
    """Construct a basis of 3 normalized vectors.

    The vectors are returned as *rows* of a matrix.
    """
    return np.stack(
        [_normed_vector_values(a), _normed_vector_values(b), _normed_vector_values(c)]
    )


def _normed_vector_values(vector: sc.Variable) -> npt.NDArray[np.float64]:
    return sc.to_unit(vector / sc.norm(vector), 'one').values
