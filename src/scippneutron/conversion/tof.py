# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""Functions for computing coordinates in time-of-flight neutron scattering.

Some functions in this module use the neutron mass :math:`m_n`
and Planck constant :math:`h`.
Their values are taken from :mod:`scipp.constants`.
"""

import numpy as np
import scipp as sc
import scipp.constants as const
from scipp.typing import Variable, VariableLike

from .._utils import as_float_type, elem_dtype, elem_unit


def _common_dtype(a, b):
    """Very limited type promotion.
    Only useful to check if the combination of a and b results in
    single or double precision float.
    """
    if elem_dtype(a) == sc.DType.float32 and elem_dtype(b) == sc.DType.float32:
        return sc.DType.float32
    return sc.DType.float64


def wavelength_from_tof(*, tof: Variable, Ltotal: Variable) -> Variable:
    r"""Compute the wavelength from time-of-flight.

    The result is the de Broglie wavelength

    .. math::

        \lambda = \frac{h t}{m_n L_\mathsf{total}}

    Where :math:`m_n` is the neutron mass and :math:`h` the Planck constant.

    Parameters
    ----------
    tof:
        Time-of-flight :math:`t`.
    Ltotal:
        Total beam length.

    Returns
    -------
    :
        Wavelength :math:`\lambda`.
        Has unit ångström.
    """
    c = sc.to_unit(
        const.h / const.m_n,
        sc.units.angstrom * elem_unit(Ltotal) / elem_unit(tof),
        copy=False,
    )
    return as_float_type(c / Ltotal, tof) * tof


def dspacing_from_tof(
    *, tof: Variable, Ltotal: Variable, two_theta: Variable
) -> Variable:
    r"""Compute the d-spacing from time-of-flight.

    The result is the inter-planar lattice spacing

    .. math::

        d = \frac{h t}{m_n L_\mathsf{total}\; 2 \sin \theta}

    Where :math:`m_n` is the neutron mass and :math:`h` the Planck constant.

    Parameters
    ----------
    tof:
        Time-of-flight :math:`t`.
    Ltotal:
        Total beam length.
    two_theta:
        Scattering angle :math:`2 \theta`.

    Returns
    -------
    :
        Inter-planar lattice spacing :math:`d`.
        Has unit ångström.

    See Also
    --------
    scippneutron.conversions.beamline:
        Definitions of ``two_theta`` and ``Ltotal``.
    """
    c = sc.to_unit(
        2 * const.m_n / const.h,
        elem_unit(tof) / sc.units.angstrom / elem_unit(Ltotal),
        copy=False,
    )
    return 1 / as_float_type(c * Ltotal * sc.sin(two_theta / 2), tof) * tof


def _energy_constant(energy_unit: sc.Unit, tof: Variable, length: Variable):
    return sc.to_unit(
        const.m_n / 2,
        energy_unit * (elem_unit(tof) / elem_unit(length)) ** 2,
        copy=False,
    )


def energy_from_tof(*, tof: Variable, Ltotal: Variable) -> Variable:
    r"""Compute the neutron energy from time-of-flight.

    The result is

    .. math::

        E = \frac{m_n L_\mathsf{total}^2}{2 t^2}

    Where :math:`m_n` is the neutron mass.

    Parameters
    ----------
    tof:
        Time-of-flight :math:`t`.
    Ltotal:
        Total beam length.
    Returns
    -------
    :
        Neutron energy :math:`E`.
        Has unit meV.
    """
    c = _energy_constant(sc.units.meV, tof, Ltotal)
    return as_float_type(c * Ltotal**2, tof) / tof ** sc.scalar(
        2, dtype=elem_dtype(tof)
    )


def _energy_transfer_t0(energy, tof, length):
    dtype = _common_dtype(energy, tof)
    c = as_float_type(_energy_constant(elem_unit(energy), tof, length), energy)
    return length.astype(dtype, copy=False) * sc.sqrt(c / energy)


def energy_transfer_direct_from_tof(
    *, tof: Variable, L1: Variable, L2: Variable, incident_energy: Variable
) -> Variable:
    r"""Compute the energy transfer in direct inelastic scattering.

    The result is

    .. math::

        \Delta E = E_i - \frac{m_n L_2^2}{2 {(t - t_0)}^2}

    With

    .. math::

        t_0 = \sqrt{m_n L_1^2 / (2 E_i)}

    and :math:`m_n` the neutron mass.

    The result is ``NaN`` for unphysical points, that is, where :math:`t < t_0`.

    Parameters
    ----------
    tof:
        Time-of-flight :math:`t`.
    L1:
        Primary beam length.
    L2:
        Secondary beam length.
    incident_energy:
        Energy before scattering :math:`E_i`.

    Returns
    -------
    :
        Energy transfer :math:`\Delta E`.
        Has the same unit as incident_energy.

    See Also
    --------
    scippneutron.conversions.tof.energy_transfer_indirect_from_tof
    """
    t0 = _energy_transfer_t0(incident_energy, tof, L1)
    c = _energy_constant(elem_unit(incident_energy), tof, L2)
    dtype = _common_dtype(incident_energy, tof)
    scale = (c * L2**2).astype(dtype, copy=False)
    delta_tof = tof - t0
    return sc.where(
        delta_tof <= sc.scalar(0, unit=elem_unit(delta_tof)),
        sc.scalar(np.nan, dtype=dtype, unit=elem_unit(incident_energy)),
        incident_energy - scale / delta_tof**2,
    )


def energy_transfer_indirect_from_tof(
    *, tof: Variable, L1: Variable, L2: Variable, final_energy: Variable
) -> Variable:
    r"""Compute the energy transfer in indirect inelastic scattering.

    The result is

    .. math::

        \Delta E = \frac{m_n L_1^2}{2 {(t - t_0)}^2} - E_f

    With

    .. math::

        t_0 = \sqrt{m_n L_2^2 / (2 E_f)}

    and :math:`m_n` the neutron mass.

    The result is ``NaN`` for unphysical points, that is, where :math:`t < t_0`.

    Parameters
    ----------
    tof:
        Time-of-flight :math:`t`.
    L1:
        Primary beam length.
    L2:
        Secondary beam length.
    final_energy:
        Energy after scattering :math:`E_f`.

    Returns
    -------
    :
        Energy transfer :math:`\Delta E`.
        Has the same unit as final_energy.

    See Also
    --------
    scippneutron.conversions.tof.energy_transfer_direct_from_tof
    """
    t0 = _energy_transfer_t0(final_energy, tof, L2)
    c = _energy_constant(elem_unit(final_energy), tof, L1)
    dtype = _common_dtype(final_energy, tof)
    scale = (c * L1**2).astype(dtype, copy=False)
    delta_tof = -t0 + tof  # Order chosen such that output.dims = ['spectrum', 'tof']
    return sc.where(
        delta_tof <= sc.scalar(0, unit=elem_unit(delta_tof)),
        sc.scalar(np.nan, dtype=dtype, unit=elem_unit(final_energy)),
        scale / delta_tof**2 - final_energy,
    )


def energy_from_wavelength(*, wavelength: Variable) -> Variable:
    r"""Compute the neutron energy from wavelength.

    The result is

    .. math::

        E = \frac{h^2}{2 m_n \lambda^2}

    Where :math:`m_n` is the neutron mass and :math:`h` the Planck constant.

    Parameters
    ----------
    wavelength:
        De Broglie wavelength :math:`\lambda`.
        Has unit meV.

    Returns
    -------
    :
        Neutron energy :math:`E`.
    """
    c = as_float_type(
        sc.to_unit(
            const.h**2 / 2 / const.m_n, sc.units.meV * elem_unit(wavelength) ** 2
        ),
        wavelength,
    )
    return c / wavelength**2


def wavelength_from_energy(*, energy: Variable) -> Variable:
    r"""Compute the wavelength from the neutron energy.

    The result is the de Broglie wavelength

    .. math::

        \lambda = \frac{h}{\sqrt{2 m_n E}}

    Where :math:`m_n` is the neutron mass and :math:`h` the Planck constant.

    Parameters
    ----------
    energy:
        Neutron energy :math:`E`.

    Returns
    -------
    :
        Wavelength :math:`\lambda`.
        Has unit ångström.
    """
    c = as_float_type(
        sc.to_unit(
            const.h**2 / 2 / const.m_n, sc.units.angstrom**2 * elem_unit(energy)
        ),
        energy,
    )
    return sc.sqrt(c / energy)


def _wavelength_Q_conversions(x: Variable, two_theta: Variable) -> Variable:
    """Convert either from Q to wavelength or vice-versa."""
    c = as_float_type(4 * const.pi, x)
    return c * sc.sin(as_float_type(two_theta, x) / 2) / x


def Q_from_wavelength(*, wavelength: Variable, two_theta: Variable) -> Variable:
    r"""Compute the absolute value of the momentum transfer from wavelength.

    The result is

    .. math::

        Q = \frac{4 \pi \sin \theta}{\lambda}

    Parameters
    ----------
    wavelength:
        De Broglie wavelength :math:`\lambda`.
    two_theta:
        Scattering angle :math:`2 \theta`.

    Returns
    -------
    :
        Momentum transfer :math:`Q`.

    See Also
    --------
    scippneutron.conversions.beamline:
        Definition of ``two_theta``.
    """
    return _wavelength_Q_conversions(wavelength, two_theta)


def wavelength_from_Q(*, Q: Variable, two_theta: Variable) -> Variable:
    r"""Compute the wavelength from momentum transfer.

    The result is the de Broglie wavelength

    .. math::

        \lambda = \frac{4 \pi \sin \theta}{Q}

    Parameters
    ----------
    Q:
        Momentum transfer.
    two_theta:
        Scattering angle :math:`2 \theta`.

    Returns
    -------
    :
        Wavelength :math:`\lambda`
        Has unit ångström.

    See Also
    --------
    scippneutron.conversions.beamline:
        Definition of ``two_theta``.
    """
    return sc.to_unit(
        _wavelength_Q_conversions(Q, two_theta), unit='angstrom', copy=False
    )


def Q_elements_from_wavelength(
    *, wavelength: Variable, incident_beam: Variable, scattered_beam: Variable
) -> tuple[Variable, Variable, Variable]:
    r"""Compute them momentum transfer vector from wavelength.

    Computes the three components of the Q-vector :math:`Q_x, Q_y, Q_z`
    separately using

    .. math::

        \vec{Q} &= (Q_x, Q_y, Q_z) \\
        \vec{Q} &= \vec{k}_i - \vec{k}_f
                 = \frac{2\pi}{\lambda} \left(\hat{e}_i - \hat{e}_f\right),

    where the unit vectors for incident momentum and final momentum

    .. math::

        \hat{e}_i &= \vec{k_i} / | \vec{k_i} | \\
        \hat{e}_f &= \vec{k_f} / | \vec{k_f} |

    are defined as the directions of ``incident_beam`` and ``scattered_beam``,
    respectively.

    Parameters
    ----------
    wavelength:
        De Broglie wavelength :math:`\lambda`.
    incident_beam:
        Beam from source to sample. Expects ``dtype=vector3``.
    scattered_beam:
        Beam from sample to detector. Expects ``dtype=vector3``.

    Returns
    -------
    Qx: scipp.Variable
        x-component of the momentum transfer :math:`\vec{Q}`.
    Qy: scipp.Variable
        y-component of the momentum transfer :math:`\vec{Q}`.
    Qz: scipp.Variable
        z-component of the momentum transfer :math:`\vec{Q}`.
    """
    e_i = incident_beam / sc.norm(incident_beam)
    e_f = scattered_beam / sc.norm(scattered_beam)
    e = e_i - e_f
    k = 2 * np.pi / wavelength
    return k * e.fields.x, k * e.fields.y, k * e.fields.z


def dspacing_from_wavelength(*, wavelength: Variable, two_theta: Variable) -> Variable:
    r"""Compute the d-spacing from wavelength.

    The result is the inter-planar lattice spacing

    .. math::

        d = \frac{\lambda}{2 \sin \theta}

    Parameters
    ----------
    wavelength:
        De Broglie wavelength :math:`\lambda`.
    two_theta:
        Scattering angle :math:`2 \theta`.

    Returns
    -------
    :
        Inter-planar lattice spacing :math:`d`.
        Has unit ångström.

    See Also
    --------
    scippneutron.conversions.beamline:
        Definition of ``two_theta``.
    """
    c = as_float_type(
        sc.scalar(0.5).to(unit=sc.units.angstrom / elem_unit(wavelength)), wavelength
    )
    return c * wavelength / sc.sin(as_float_type(two_theta, wavelength) / 2)


def dspacing_from_energy(*, energy: Variable, two_theta: Variable) -> Variable:
    r"""Compute the d-spacing from the neutron energy.

    The result is the inter-planar lattice spacing

    .. math::

        d = \frac{h}{\sqrt{8 m_n E} \sin \theta}

    Where :math:`m_n` is the neutron mass and :math:`h` the Planck constant.

    Parameters
    ----------
    energy:
        Neutron energy :math:`E`.
    two_theta:
        Scattering angle :math:`2 \theta`.

    Returns
    -------
    :
        Inter-planar lattice spacing :math:`d`.
        Has unit ångström.

    See Also
    --------
    scippneutron.conversions.beamline:
        Definition of ``two_theta``.
    """
    c = as_float_type(
        sc.to_unit(
            const.h**2 / 8 / const.m_n, sc.units.angstrom**2 * elem_unit(energy)
        ),
        energy,
    )
    return sc.sqrt(c / energy) / sc.sin(as_float_type(two_theta, energy) / 2)


def Q_vec_from_Q_elements(*, Qx: Variable, Qy: Variable, Qz: Variable) -> Variable:
    """Combine elements of Q into a single vector variable.

    Parameters
    ----------
    Qx:
        x-elements of the momentum transfer.
    Qy:
        y-elements of the momentum transfer.
    Qz:
        z-elements of the momentum transfer.

    Returns
    -------
    :
        ``Qx``, ``Qy``, ``Qz`` combined into a single variable of dtype ``vector3``.
    """
    if Qx.sizes != Qy.sizes or Qx.sizes != Qz.sizes:
        raise sc.DimensionError(
            "Qx, Qy, Qz must have the same sizes. "
            f"Got {Qx.sizes=}, {Qy.sizes=}, {Qz.sizes=}."
        )
    return sc.spatial.as_vectors(Qx, Qy, Qz)


def ub_matrix_from_u_and_b(*, u_matrix: Variable, b_matrix: Variable) -> Variable:
    r"""Compute the UB matrix from U and B matrices.

    .. math::

        \mathsf{UB} = U \cdot B

    where :math:`U` and :math:`B` are defined as in
    :cite:`busing:1967,mantid-lattice:2023`.

    Parameters
    ----------
    u_matrix:
        :math:`U`.
    b_matrix:
        :math:`B`.

    Returns
    -------
    :
        :math:`\mathsf{UB}`.
    """
    return u_matrix * b_matrix


def hkl_vec_from_Q_vec(
    *, Q_vec: Variable, ub_matrix: Variable, sample_rotation: Variable
) -> Variable:
    r"""Compute hkl indices from momentum transfer.

    The hkl indices define the components of the momentum transfer in the
    sample coordinate system

    .. math::

        \vec{Q} = \begin{pmatrix} h \\ k \\ l \end{pmatrix}.

    In the lab frame, the momentum transfer as computed by
    :func:`scippneutron.conversion.tof.Q_elements_from_wavelength`
    is defined as

    .. math::

        \vec{Q}_l = \vec{k}_i - \vec{k}_f .

    This quantity is called :math:`Q` elsewhere in ScippNeutron.

    Those two :math:`Q`'s are related by via

    .. math::

        \vec{Q}_l = 2 \pi R U B \vec{Q},

    where :math:`U` and :math:`B` transform from sample space to the lab frame.
    :math:`R` encodes the sample rotation, e.g., as given by a goniometer.
    See, e.g., Refs. :cite:`busing:1967,mantid-lattice:2023,savici:2011`
    for a definition.

    This function computes the elements of :math:`\vec{Q}`, :math:`h, k, l` by inverting
    the above equation.

    Parameters
    ----------
    Q_vec:
        Momentum transfer :math:`\vec{Q}_l` as a vector variable.
    ub_matrix:
        Matrix :math:`\mathsf{UB}`.
    sample_rotation:
        Sample rotation matrix :math:`R`.

    Returns
    -------
    :
        :math:`h, k, l` as a vector variable.

    See also
    --------
    scippneutron.conversion.tof.Q_elements_from_wavelength:
        Computes ``Q_l``.
    scippneutron.conversion.tof.Q_vec_from_Q_elements:
        Packs elements ``Qx``, ``Qy``, ``Qz`` into a single vector.
    scippneutron.conversion.tof.ub_matrix_from_u_and_b:
        Compute :math:`\mathsf{UB}` from :math:`B` and :math:`B` matrices.
    scippneutron.conversion.tof.hkl_elements_from_hkl_vec:
        Unpack the returned hkl vector.
    """
    # There are different ways to implement this with different performance and
    # accuracy characteristics.
    # Matrix-matrix products typically have worse accuracy than matrix-vector
    # products due to repeated floating point cutoffs.
    # Matrix inversions are often unstable; however, all matrices used here are small.
    # Potential implementations are
    #
    # (sc.spatial.inv(B) * (sc.spatial.inv(U) * (sc.spatial.inv(R) * Q))) / (2*np.pi)
    #
    # (((sc.spatial.inv(B) * sc.spatial.inv(U)) * sc.spatial.inv(R)) * Q) / (2*np.pi)
    #
    # (sc.spatial.inv((R * U) * B) * Q) / (2*np.pi)
    #
    # All 3 were tested with random matrices and vectors and the first was found
    # to have the best overall accuracy, being an order of magnitude better than the
    # others in particularly bad cases and equal in the majority of cases.
    #
    # Concerning performance, R, U, B are scalar variables or short array variables
    # in typical use cases while Q is typically a long, potentially multi-dim array.
    # So implementation 3 will likely perform the best.
    #
    # This function uses implementation 3 as the performance gain
    # is expected to be significant over 1 and 2.
    return (sc.spatial.inv(sample_rotation * ub_matrix) * Q_vec) / (2 * np.pi)


def hkl_elements_from_hkl_vec(
    *, hkl_vec: Variable
) -> tuple[Variable, Variable, Variable]:
    """Unpack vector of hkl indices into separate variables.

    Parameters
    ----------
    hkl_vec:
        Vector of hkl indices.

    Returns
    -------
    h: scipp.Variable
        1st component of the hkl vector.
    k: scipp.Variable
        2nd component of the hkl vector.
    l: scipp.Variable
        3rd component of the hkl vector.
    """
    return hkl_vec.fields.x, hkl_vec.fields.y, hkl_vec.fields.z


def time_at_sample_from_tof(
    *,
    pulse_time: VariableLike,
    tof: VariableLike,
    L2: VariableLike,
    wavelength: VariableLike,
) -> VariableLike:
    """Compute the absolute time when the neutron passed through the sample.

    The result is

    .. math::

        t_{sample} = t_{pulse} + t_{of} - L_2 / v

    where

    .. math::

        v = \\frac{h}{m_n \\lambda}

    where :math:`v` is the estimated velocity of the neutron over its path from
    sample to detector and :math:`\\lambda` is the wavelength of the neutron.

    Parameters
    ----------
    pulse_time:
        absolute time when time of flight is 0
    tof:
        the time of fligth of the neutron
    L2:
        path length from sample to detector
    wavelength:
        wavelength of the neutron (at the detector).
        Assuming this did not change during the neutrons
        travel from sample to detector this can be used to compute the
        velocity the neutron had between the sample and the detector.

    Returns
    -------
    :
        :math:`t_{sample}`
    """
    c = sc.to_unit(
        const.h / const.m_n,
        sc.units.angstrom * elem_unit(L2) / elem_unit(tof),
        copy=False,
    )
    return pulse_time + tof - L2 * wavelength / c
