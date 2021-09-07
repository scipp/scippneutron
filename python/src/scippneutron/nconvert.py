# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import numpy as np
import scipp as sc
import scipp.constants as const

# TODO check old convert for disallowed combinations of coords, e.g. inelastic energy


def _elem_unit(var):
    if var.bins is not None:
        return var.events.unit
    return var.unit


def _total_beam_length_no_scatter(source_position, position):
    return sc.norm(position - source_position)


def _scattering_beams(source_position, sample_position, position):
    return {
        'incident_beam': sample_position - source_position,
        'scattered_beam': position - sample_position
    }


def two_theta(incident_beam, scattered_beam, L1, L2):
    return sc.acos(sc.dot(incident_beam / L1, scattered_beam / L2))


def _beam_lengths_and_angle(incident_beam, scattered_beam):
    L1 = sc.norm(incident_beam)
    L2 = sc.norm(scattered_beam)
    return {
        'L1': L1,
        'L2': L2,
        'two_theta': two_theta(incident_beam, scattered_beam, L1, L2)
    }


def _total_beam_length_scatter(L1, L2):
    return L1 + L2


def _wavelength_from_tof(tof, Ltotal):
    c = sc.to_unit(const.h / const.m_n,
                   sc.units.angstrom * _elem_unit(Ltotal) / _elem_unit(tof),
                   copy=False)
    return (c / Ltotal).astype(tof.dtype, copy=False) * tof


def _Q_from_wavelength(wavelength, two_theta):
    return (4 * const.pi) * sc.sin(two_theta / 2) / wavelength


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
    if a.dtype == sc.dtype.float32 and b.dtype == sc.dtype.float32:
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
    return (c * Ltotal**2).astype(tof.dtype, copy=False) / tof**sc.scalar(
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


def _dspacing_from_tof(tof, Ltotal, two_theta):
    c = sc.to_unit(2 * const.m_n / const.h,
                   _elem_unit(tof) / sc.units.angstrom / _elem_unit(Ltotal),
                   copy=False)
    return tof / (c * Ltotal * sc.sin(two_theta / 2)).astype(tof.dtype, copy=False)


NO_SCATTER_GRAPH = {
    'Ltotal': _total_beam_length_no_scatter,
    'wavelength': _wavelength_from_tof
}

SCATTER_GRAPH_DETECTOR_TO_PHYS = {
    ('incident_beam', 'scattered_beam'): _scattering_beams,
    ('L1', 'L2', 'two_theta'): _beam_lengths_and_angle,
    'Ltotal': _total_beam_length_scatter,
    'wavelength': _wavelength_from_tof,
    'Q': _Q_from_wavelength,
    'energy': _energy_from_tof,
    'dspacing': _dspacing_from_tof,
}


def incoming(edge):
    if isinstance(edge, str):
        return edge
    return sc.coords._argnames(edge)


def _path_exists_impl(origin, target, graph):
    try:
        target_node = graph[target]
    except KeyError:
        return False
    for node in incoming(target_node):
        if node == target:
            continue  # Node that produces its input as output.
        if node == origin:
            return True
        if _path_exists_impl(origin, node, graph):
            return True
    return False


# TODO move into sc.coords?
def path_exists(origin, target, graph):
    if origin == target:
        return True
    return _path_exists_impl(origin, target, sc.coords.Graph(graph))


def _find_inelastic_inputs(data):
    return [name for name in ('incident_energy', 'final_energy') if name in data.coords]


def _inelastic_scatter_graph(data):
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
    return {**SCATTER_GRAPH_DETECTOR_TO_PHYS, **inelastic_step}


def _elastic_scatter_graph(data, origin, target):
    if target == 'energy':
        inelastic_inputs = _find_inelastic_inputs(data)
        if inelastic_inputs:
            raise RuntimeError(
                f"Data contains coords for inelastic scattering "
                f"({inelastic_inputs}) but conversion to elastic energy requested. "
                f"This is not implemented.")
    for graph in (SCATTER_GRAPH_DETECTOR_TO_PHYS, ):
        if path_exists(origin, target, graph):
            return graph
    return None


def _scatter_graph(data, origin, target):
    graph = _inelastic_scatter_graph(
        data) if target == 'energy_transfer' else _elastic_scatter_graph(
            data, origin, target)
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
        raise RuntimeError(f"Missing coordinate '{err.args[0]}' for conversion "
                           f"from '{origin}' to '{target}'") from None

    return converted
