# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

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


def _scattering_beams(source_position, sample_position, position):
    return {
        'incident_beam': sample_position - source_position,
        'scattered_beam': position - sample_position
    }


def _two_theta(incident_beam, scattered_beam, L1, L2):
    return sc.acos(sc.dot(incident_beam / L1, scattered_beam / L2))


def _beam_lengths_and_angle(incident_beam, scattered_beam):
    L1 = sc.norm(incident_beam)
    L2 = sc.norm(scattered_beam)
    return {
        'L1': L1,
        'L2': L2,
        'two_theta': _two_theta(incident_beam, scattered_beam, L1, L2)
    }


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
    return tof / (c * Ltotal * sc.sin(two_theta / 2)).astype(_elem_dtype(tof),
                                                             copy=False)


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
    c = sc.to_unit(const.m_n,
                   _elem_unit(tof)**2 * _elem_unit(energy) /
                   _elem_unit(length)**2).astype(dtype, copy=False)
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
    delta_tof = tof - t0
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


NO_SCATTER_GRAPH_KINEMATICS = {
    'Ltotal': _total_beam_length_no_scatter,
}

NO_SCATTER_GRAPH = {
    **NO_SCATTER_GRAPH_KINEMATICS,
    'wavelength': _wavelength_from_tof,
    'energy': _energy_from_tof,
}

SCATTER_GRAPH_KINEMATICS = {
    ('incident_beam', 'scattered_beam'): _scattering_beams,
    ('L1', 'L2', 'two_theta'): _beam_lengths_and_angle,
    'Ltotal': _total_beam_length_scatter,
}

SCATTER_GRAPHS_DYNAMICS = {
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


def _find_inelastic_inputs(data):
    return [name for name in ('incident_energy', 'final_energy') if name in data.coords]


def _inelastic_scatter_graph(data, origin, target):
    inputs = _find_inelastic_inputs(data)
    if len(inputs) > 1:
        raise RuntimeError(
            "Data contains coords for incident *and* final energy, cannot have "
            "both for inelastic scattering.")
    if len(inputs) == 0:
        raise RuntimeError(
            "Data contains neither coords for incident nor for final energy, this "
            "does not appear to be inelastic-scattering data, cannot convert to "
            "energy transfer.")

    inelastic_step = {
        'incident_energy': {
            'energy_transfer': _energy_transfer_direct_from_tof
        },
        'final_energy': {
            'energy_transfer': _energy_transfer_indirect_from_tof
        }
    }[inputs[0]]
    return {**_elastic_scatter_graph(data, origin, target), **inelastic_step}


def _reachable_by(target, graph):
    return any(target == targets if isinstance(targets, str) else target in targets
               for targets in graph.keys())


def _elastic_scatter_base_graph(origin, target):
    if _reachable_by(target, SCATTER_GRAPH_KINEMATICS):
        return dict(SCATTER_GRAPH_KINEMATICS)
    return {**SCATTER_GRAPH_KINEMATICS, **SCATTER_GRAPHS_DYNAMICS[origin]}


def _elastic_scatter_graph(data, origin, target):
    if target == 'energy':
        inelastic_inputs = _find_inelastic_inputs(data)
        if inelastic_inputs:
            raise RuntimeError(
                f"Data contains coords for inelastic scattering "
                f"({inelastic_inputs}) but conversion to elastic energy requested. "
                f"This is not implemented.")
    return _elastic_scatter_base_graph(origin, target)


def _scatter_graph(data, origin, target):
    graph = (_inelastic_scatter_graph(data, origin, target) if target
             == 'energy_transfer' else _elastic_scatter_graph(data, origin, target))
    if not graph:
        raise RuntimeError(f"No viable conversion from '{origin}' to '{target}'.")
    return graph


def conversion_graph(data, origin, target, scatter):
    if scatter:
        return _scatter_graph(data, origin, target)
    else:
        return NO_SCATTER_GRAPH


def convert(data, origin, target, scatter):
    try:
        converted = data.transform_coords(target,
                                          graph=conversion_graph(
                                              data, origin, target, scatter))
    except KeyError as err:
        if err.args[0] == target:
            raise RuntimeError(f"No viable conversion from '{origin}' to '{target}' "
                               f"with scatter={scatter}.")
        raise RuntimeError(f"Missing coordinate '{err.args[0]}' for conversion "
                           f"from '{origin}' to '{target}'") from None

    return converted
