# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""Functions for computing coordinates in time-of-flight neutron scattering.

Some functions in this module use the neutron mass :math:`m_n`
and Planck constant :math:`h`.
Their values are taken from :mod:`scipp.constants`.
"""

import numpy as np
import scipp.constants as const
from scipp.typing import VariableLike
import scipp as sc

from .._utils import as_float_type, elem_dtype, elem_unit


def _common_dtype(a, b):
    """Very limited type promotion.
    Only useful to check if the combination of a and b results in
    single or double precision float.
    """
    if elem_dtype(a) == sc.DType.float32 and elem_dtype(b) == sc.DType.float32:
        return sc.DType.float32
    return sc.DType.float64


def wavelength_from_tof(*, tof: VariableLike, Ltotal: VariableLike) -> VariableLike:
    """Compute the wavelength from time-of-flight.

    The result is the de Broglie wavelength

    .. math::

        \\lambda = \\frac{h t}{m_n L_\\mathsf{total}}

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
        Wavelength :math:`\\lambda`.
        Has unit ångström.
    """
    c = sc.to_unit(const.h / const.m_n,
                   sc.units.angstrom * elem_unit(Ltotal) / elem_unit(tof),
                   copy=False)
    return as_float_type(c / Ltotal, tof) * tof


def dspacing_from_tof(*, tof: VariableLike, Ltotal: VariableLike,
                      two_theta: VariableLike) -> VariableLike:
    """Compute the d-spacing from time-of-flight.

    The result is the inter-planar lattice spacing

    .. math::

        d = \\frac{h t}{m_n L_\\mathsf{total}\\; 2 \\sin \\theta}

    Where :math:`m_n` is the neutron mass and :math:`h` the Planck constant.

    Parameters
    ----------
    tof:
        Time-of-flight :math:`t`.
    Ltotal:
        Total beam length.
    two_theta:
        Scattering angle :math:`2 \\theta`.

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
    c = sc.to_unit(2 * const.m_n / const.h,
                   elem_unit(tof) / sc.units.angstrom / elem_unit(Ltotal),
                   copy=False)
    return 1 / as_float_type(c * Ltotal * sc.sin(two_theta / 2), tof) * tof


def _energy_constant(energy_unit: sc.Unit, tof: VariableLike, length: VariableLike):
    return sc.to_unit(const.m_n / 2,
                      energy_unit * (elem_unit(tof) / elem_unit(length))**2,
                      copy=False)


def energy_from_tof(*, tof: VariableLike, Ltotal: VariableLike) -> VariableLike:
    """Compute the neutron energy from time-of-flight.

    The result is

    .. math::

        E = \\frac{m_n L_\\mathsf{total}^2}{2 t^2}

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
    return as_float_type(c * Ltotal**2, tof) / tof**sc.scalar(2, dtype=elem_dtype(tof))


def _energy_transfer_t0(energy, tof, length):
    dtype = _common_dtype(energy, tof)
    c = as_float_type(_energy_constant(elem_unit(energy), tof, length), energy)
    return length.astype(dtype, copy=False) * sc.sqrt(c / energy)


def energy_transfer_direct_from_tof(*, tof: VariableLike, L1: VariableLike,
                                    L2: VariableLike,
                                    incident_energy: VariableLike) -> VariableLike:
    """Compute the energy transfer in direct inelastic scattering.

    The result is

    .. math::

        \\Delta E = E_i - \\frac{m_n L_2^2}{2 {(t - t_0)}^2}

    With

    .. math::

        t_0 = \\sqrt{m_n L_1^2 / (2 E_i)}

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
        Energy transfer :math:`\\Delta E`.
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
    return sc.where(delta_tof <= sc.scalar(0, unit=elem_unit(delta_tof)),
                    sc.scalar(np.nan, dtype=dtype, unit=elem_unit(incident_energy)),
                    incident_energy - scale / delta_tof**2)


def energy_transfer_indirect_from_tof(*, tof: VariableLike, L1: VariableLike,
                                      L2: VariableLike,
                                      final_energy: VariableLike) -> VariableLike:
    """Compute the energy transfer in indirect inelastic scattering.

    The result is

    .. math::

        \\Delta E = \\frac{m_n L_1^2}{2 {(t - t_0)}^2} - E_f

    With

    .. math::

        t_0 = \\sqrt{m_n L_2^2 / (2 E_f)}

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
        Energy transfer :math:`\\Delta E`.
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
    return sc.where(delta_tof <= sc.scalar(0, unit=elem_unit(delta_tof)),
                    sc.scalar(np.nan, dtype=dtype, unit=elem_unit(final_energy)),
                    scale / delta_tof**2 - final_energy)


def energy_from_wavelength(*, wavelength: VariableLike) -> VariableLike:
    """Compute the neutron energy from wavelength.

    The result is

    .. math::

        E = \\frac{h^2}{2 m_n \\lambda^2}

    Where :math:`m_n` is the neutron mass and :math:`h` the Planck constant.

    Parameters
    ----------
    wavelength:
        De Broglie wavelength :math:`\\lambda`.
        Has unit meV.

    Returns
    -------
    :
        Neutron energy :math:`E`.
    """
    c = as_float_type(
        sc.to_unit(const.h**2 / 2 / const.m_n,
                   sc.units.meV * elem_unit(wavelength)**2), wavelength)
    return c / wavelength**2


def wavelength_from_energy(*, energy: VariableLike) -> VariableLike:
    """Compute the wavelength from the neutron energy.

    The result is the de Broglie wavelength

    .. math::

        \\lambda = \\frac{h}{\\sqrt{2 m_n E}}

    Where :math:`m_n` is the neutron mass and :math:`h` the Planck constant.

    Parameters
    ----------
    energy:
        Neutron energy :math:`E`.

    Returns
    -------
    :
        Wavelength :math:`\\lambda`.
        Has unit ångström.
    """
    c = as_float_type(
        sc.to_unit(const.h**2 / 2 / const.m_n,
                   sc.units.angstrom**2 * elem_unit(energy)), energy)
    return sc.sqrt(c / energy)


def _wavelength_Q_conversions(x: VariableLike, two_theta: VariableLike) -> VariableLike:
    """Convert either from Q to wavelength or vice-versa."""
    c = as_float_type(4 * const.pi, x)
    return c * sc.sin(as_float_type(two_theta, x) / 2) / x


def Q_from_wavelength(*, wavelength: VariableLike,
                      two_theta: VariableLike) -> VariableLike:
    """Compute the momentum transfer from wavelength.

    The result is

    .. math::

        Q = \\frac{4 \\pi \\sin \\theta}{\\lambda}

    Parameters
    ----------
    wavelength:
        De Broglie wavelength :math:`\\lambda`.
    two_theta:
        Scattering angle :math:`2 \\theta`.

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


def wavelength_from_Q(*, Q: VariableLike, two_theta: VariableLike) -> VariableLike:
    """Compute the wavelength from momentum transfer.

    The result is the de Broglie wavelength

    .. math::

        \\lambda = \\frac{4 \\pi \\sin \\theta}{Q}

    Parameters
    ----------
    Q:
        Momentum transfer.
    two_theta:
        Scattering angle :math:`2 \\theta`.

    Returns
    -------
    :
        Wavelength :math:`\\lambda`
        Has unit ångström.

    See Also
    --------
    scippneutron.conversions.beamline:
        Definition of ``two_theta``.
    """
    return sc.to_unit(_wavelength_Q_conversions(Q, two_theta),
                      unit='angstrom',
                      copy=False)


def dspacing_from_wavelength(*, wavelength: VariableLike,
                             two_theta: VariableLike) -> VariableLike:
    """Compute the d-spacing from wavelength.

    The result is the inter-planar lattice spacing

    .. math::

        d = \\frac{\\lambda}{2 \\sin \\theta}

    Parameters
    ----------
    wavelength:
        De Broglie wavelength :math:`\\lambda`.
    two_theta:
        Scattering angle :math:`2 \\theta`.

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
        sc.scalar(0.5).to(unit=sc.units.angstrom / elem_unit(wavelength)), wavelength)
    return c * wavelength / sc.sin(as_float_type(two_theta, wavelength) / 2)


def dspacing_from_energy(*, energy: VariableLike,
                         two_theta: VariableLike) -> VariableLike:
    """Compute the d-spacing from the neutron energy.

    The result is the inter-planar lattice spacing

    .. math::

        d = \\frac{h}{\\sqrt{8 m_n E} \\sin \\theta}

    Where :math:`m_n` is the neutron mass and :math:`h` the Planck constant.

    Parameters
    ----------
    energy:
        Neutron energy :math:`E`.
    two_theta:
        Scattering angle :math:`2 \\theta`.

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
        sc.to_unit(const.h**2 / 8 / const.m_n,
                   sc.units.angstrom**2 * elem_unit(energy)), energy)
    return sc.sqrt(c / energy) / sc.sin(as_float_type(two_theta, energy) / 2)
