# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

from typing import Callable, Dict, Tuple, Union

import numpy as np
import scipp as sc
import scipp.constants as const


def _elem_unit(var):
    if var.bins is not None:
        return var.events.unit
    return var.unit


def _elem_dtype(var):
    if var.bins is not None:
        return var.events.dtype
    return var.dtype


def _total_beam_length_no_scatter(source_position, position):
    return sc.norm(position - source_position)


def _incident_beam(source_position, sample_position):
    return sample_position - source_position


def _scattered_beam(position, sample_position):
    return position - sample_position


def _L1(incident_beam):
    return sc.norm(incident_beam)


def _L2(scattered_beam):
    return sc.norm(scattered_beam)


def _two_theta(incident_beam, scattered_beam):
    return sc.acos(
        sc.dot(incident_beam / _L1(incident_beam),
               scattered_beam / _L2(scattered_beam)))


def _total_beam_length_scatter(L1, L2):
    return L1 + L2


def _wavelength_from_tof(tof, Ltotal):
    c = sc.to_unit(const.h / const.m_n,
                   sc.units.angstrom * _elem_unit(Ltotal) / _elem_unit(tof),
                   copy=False)
    return (c / Ltotal).astype(_elem_dtype(tof), copy=False) * tof


def _dspacing_from_tof(tof, Ltotal, two_theta):
    c = sc.to_unit(2 * const.m_n / const.h,
                   _elem_unit(tof) / sc.units.angstrom / _elem_unit(Ltotal),
                   copy=False)
    return 1 / (c * Ltotal * sc.sin(two_theta / 2)).astype(_elem_dtype(tof),
                                                           copy=False) * tof


def _energy_constant(energy_unit, tof, length):
    return sc.to_unit(const.m_n / 2,
                      energy_unit * (_elem_unit(tof) / _elem_unit(length))**2,
                      copy=False)


def _common_dtype(a, b):
    """
    Very limited type promotion.
    Only useful to check if the combination of a and b results in
    single or double precision float.
    """
    if _elem_dtype(a) == sc.dtype.float32 and _elem_dtype(b) == sc.dtype.float32:
        return sc.dtype.float32
    return sc.dtype.float64


def _energy_transfer_t0(energy, tof, length):
    dtype = _common_dtype(energy, tof)
    c = _energy_constant(_elem_unit(energy), tof, length).astype(energy.dtype,
                                                                 copy=False)
    return length.astype(dtype, copy=False) * sc.sqrt(c / energy)


def _energy_from_tof(tof, Ltotal):
    c = _energy_constant(sc.units.meV, tof, Ltotal)
    return (c * Ltotal**2).astype(_elem_dtype(tof), copy=False) / tof**sc.scalar(
        2, dtype='float32')


def _energy_transfer_direct_from_tof(tof, L1, L2, incident_energy):
    t0 = _energy_transfer_t0(incident_energy, tof, L1)
    c = _energy_constant(_elem_unit(incident_energy), tof, L2)
    dtype = _common_dtype(incident_energy, tof)
    scale = (c * L2**2).astype(dtype, copy=False)
    delta_tof = tof - t0
    return sc.where(delta_tof <= sc.scalar(0, unit=_elem_unit(delta_tof)),
                    sc.scalar(np.nan, dtype=dtype, unit=_elem_unit(incident_energy)),
                    incident_energy - scale / delta_tof**2)


def _energy_transfer_indirect_from_tof(tof, L1, L2, final_energy):
    t0 = _energy_transfer_t0(final_energy, tof, L2)
    c = _energy_constant(_elem_unit(final_energy), tof, L1)
    dtype = _common_dtype(final_energy, tof)
    scale = (c * L1**2).astype(dtype, copy=False)
    delta_tof = -t0 + tof  # Order chosen such that output.dims = ['spectrum', 'tof']
    return sc.where(delta_tof <= sc.scalar(0, unit=_elem_unit(delta_tof)),
                    sc.scalar(np.nan, dtype=dtype, unit=_elem_unit(final_energy)),
                    scale / delta_tof**2 - final_energy)


def _energy_from_wavelength(wavelength):
    c = sc.to_unit(const.h**2 / 2 / const.m_n,
                   sc.units.meV * _elem_unit(wavelength)**2).astype(
                       _elem_dtype(wavelength), copy=False)
    return c / wavelength**2


def _wavelength_from_energy(energy):
    c = sc.to_unit(const.h**2 / 2 / const.m_n, sc.units.angstrom**2 *
                   _elem_unit(energy)).astype(_elem_dtype(energy), copy=False)
    return sc.sqrt(c / energy)


def _wavelength_Q_conversions(x, two_theta):
    """
    Convert either from Q to wavelength or vice-versa.
    """
    dtype = _elem_dtype(x)
    c = (4 * const.pi).astype(dtype)
    return c * sc.sin(two_theta.astype(dtype, copy=False) / 2) / x


def _Q_from_wavelength(wavelength, two_theta):
    return _wavelength_Q_conversions(wavelength, two_theta)


def _wavelength_from_Q(Q, two_theta):
    return _wavelength_Q_conversions(Q, two_theta)


def _dspacing_from_wavelength(wavelength, two_theta):
    dtype = _elem_dtype(wavelength)
    c = sc.scalar(0.5, unit=sc.units.angstrom / _elem_unit(wavelength)).astype(dtype)
    return c * wavelength / sc.sin(two_theta.astype(dtype, copy=False) / 2)


def _dspacing_from_energy(energy, two_theta):
    dtype = _elem_dtype(energy)
    c = sc.to_unit(const.h**2 / 8 / const.m_n,
                   sc.units.angstrom**2 * _elem_unit(energy)).astype(dtype)
    return sc.sqrt(c / energy) / sc.sin(two_theta.astype(dtype, copy=False) / 2)


_NO_SCATTER_GRAPH_KINEMATICS = {
    'Ltotal': _total_beam_length_no_scatter,
}

_NO_SCATTER_GRAPH = {
    **_NO_SCATTER_GRAPH_KINEMATICS,
    'wavelength': _wavelength_from_tof,
    'energy': _energy_from_tof,
}

_SCATTER_GRAPH_KINEMATICS = {
    'incident_beam': _incident_beam,
    'scattered_beam': _scattered_beam,
    'L1': _L1,
    'L2': _L2,
    'two_theta': _two_theta,
    'Ltotal': _total_beam_length_scatter,
}

_SCATTER_GRAPH_DYNAMICS_BY_ORIGIN = {
    'energy': {
        'dspacing': _dspacing_from_energy,
        'wavelength': _wavelength_from_energy,
    },
    'tof': {
        'dspacing': _dspacing_from_tof,
        'energy': _energy_from_tof,
        'Q': _Q_from_wavelength,
        'wavelength': _wavelength_from_tof,
    },
    'Q': {
        'wavelength': _wavelength_from_Q,
    },
    'wavelength': {
        'dspacing': _dspacing_from_wavelength,
        'energy': _energy_from_wavelength,
        'Q': _Q_from_wavelength,
    },
}


def _inelastic_scatter_graph(energy_mode):
    inelastic_step = {
        'direct_inelastic': {
            'energy_transfer': _energy_transfer_direct_from_tof
        },
        'indirect_inelastic': {
            'energy_transfer': _energy_transfer_indirect_from_tof
        }
    }[energy_mode]
    return {**_SCATTER_GRAPH_KINEMATICS, **inelastic_step}


def _reachable_by(target, graph):
    return any(target == targets if isinstance(targets, str) else target in targets
               for targets in graph.keys())


def _elastic_scatter_graph(origin, target):
    if _reachable_by(target, _SCATTER_GRAPH_KINEMATICS):
        return dict(_SCATTER_GRAPH_KINEMATICS)
    return {**_SCATTER_GRAPH_KINEMATICS, **_SCATTER_GRAPH_DYNAMICS_BY_ORIGIN[origin]}


def _scatter_graph(origin, target, energy_mode):
    graph = (_elastic_scatter_graph(origin, target)
             if energy_mode == 'elastic' else _inelastic_scatter_graph(energy_mode))
    return graph


def conversion_graph(origin: str, target: str, scatter: bool,
                     energy_mode: str) -> Dict[Union[str, Tuple[str]], Callable]:
    """
    Get a conversion graph for given parameters.

    The graph can be used with `scipp.transform_coords`.

    :param origin: Name of the input coordinate.
    :param target: Name of the output coordinate.
    :param scatter: Choose whether to use scattering or non-scattering conversions.
    :param energy_mode: Select if energy is conserved. One of `'elastic'`,
                        `'direct_inelastic'`, `'indirect_inelastic'`.
    :return: Conversion graph.
    :seealso: :py:func:`scippneutron.convert`,
              :py:func:`scippneutron.deduce_conversion_graph`.
    """

    # Results are copied to ensure users do not modify the global dictionaries.
    if scatter:
        return dict(_scatter_graph(origin, target, energy_mode))
    else:
        return dict(_NO_SCATTER_GRAPH)


def _find_inelastic_inputs(data):
    return [name for name in ('incident_energy', 'final_energy') if name in data.coords]


def _deduce_energy_mode(data, origin, target):
    inelastic_inputs = _find_inelastic_inputs(data)
    if target == 'energy_transfer':
        if len(inelastic_inputs) > 1:
            raise RuntimeError(
                "Data contains coords for incident *and* final energy, cannot have "
                "both for inelastic scattering.")
        if len(inelastic_inputs) == 0:
            raise RuntimeError(
                "Data contains neither coords for incident nor for final energy, this "
                "does not appear to be inelastic-scattering data, cannot convert to "
                "energy transfer.")
        return {
            'incident_energy': 'direct_inelastic',
            'final_energy': 'indirect_inelastic'
        }[inelastic_inputs[0]]

    if 'energy' in (origin, target):
        if inelastic_inputs:
            raise RuntimeError(
                f"Data contains coords for inelastic scattering "
                f"({inelastic_inputs}) but conversion with elastic energy requested. "
                f"This is not implemented.")
    return 'elastic'


def deduce_conversion_graph(data: Union[sc.DataArray,
                                        sc.Dataset], origin: str, target: str,
                            scatter: bool) -> Dict[Union[str, Tuple[str]], Callable]:
    """
    Get the conversion graph used by :py:func:`scippneutron.convert`
    when called with identical arguments.

    :param data: Input data.
    :param origin: Name of the input coordinate.
    :param target: Name of the output coordinate.
    :param scatter: Choose whether to use scattering or non-scattering conversions.
    :return: Conversion graph.
    :seealso: :py:func:`scippneutron.convert`, :py:func:`scippneutron.conversion_graph`.
    """
    return conversion_graph(origin, target, scatter,
                            _deduce_energy_mode(data, origin, target))


def convert(data: Union[sc.DataArray, sc.Dataset], origin: str, target: str,
            scatter: bool) -> Union[sc.DataArray, sc.Dataset]:
    """
    Perform a unit conversion from the given origin unit to target.
    See the the documentation page on "Unit conversions"
    (https://scipp.github.io/scippneutron/user-guide/unit-conversions.html)
    for more details.

    :param data: Input data.
    :param origin: Name of the input coordinate.
    :param target: Name of the output coordinate.
    :param scatter: Choose whether to use scattering or non-scattering conversions.
    :return: A new scipp.DataArray or scipp.Dataset with the new coordinate.
    :seealso: :py:func:`scippneutron.deduce_conversion_graph` and
              :py:func:`scippneutron.conversion_graph` to inspect
              the possible conversions.
    """

    graph = deduce_conversion_graph(data, origin, target, scatter)

    try:
        converted = data.transform_coords(target, graph=graph)
    except KeyError as err:
        if err.args[0] == target:
            raise RuntimeError(f"No viable conversion from '{origin}' to '{target}' "
                               f"with scatter={scatter}.")
        raise RuntimeError(f"Missing coordinate '{err.args[0]}' for conversion "
                           f"from '{origin}' to '{target}'") from None

    return converted
