# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import scipp as sc
import scipp.constants as const

from ._scippneutron import conversions

# TODO check old convert for disallowed combinations of coords, e.g. inelastic energy


def _total_beam_length_no_scatter(source_position, position):
    return sc.norm(position - source_position)


def _scattering_beams(source_position, sample_position, position):
    return {
        'incident_beam': sample_position - source_position,
        'scattered_beam': position - sample_position
    }


def _beam_lengths_and_angle(incident_beam, scattered_beam):
    def normalized(v):
        return v / sc.norm(v)

    return {
        'L1': sc.norm(incident_beam),
        'L2': sc.norm(scattered_beam),
        'two_theta':
        sc.acos(sc.dot(normalized(incident_beam), normalized(scattered_beam)))
    }


def _total_beam_length_scatter(L1, L2):
    return L1 + L2


def _wavelength_from_tof(tof, Ltotal):
    c = sc.to_unit(const.h / const.m_n,
                   sc.units.angstrom * Ltotal.unit / tof.unit,
                   copy=False)
    return c / Ltotal * tof


def _Q_from_wavelength(wavelength, two_theta):
    return (4 * const.pi) * sc.sin(two_theta / 2) / wavelength


def _energy_from_tof(tof, Ltotal):
    c = sc.to_unit(const.m_n / 2,
                   sc.units.meV * (tof.unit / Ltotal.unit)**2,
                   copy=False)
    return c * sc.pow(Ltotal / tof, sc.scalar(2))


def _energy_transfer_direct_from_tof(tof, L1, L2, incident_energy):
    return conversions.energy_transfer_direct_from_tof(tof, L1, L2, incident_energy)


def _energy_transfer_indirect_from_tof(tof, L1, L2, final_energy):
    return conversions.energy_transfer_indirect_from_tof(tof, L1, L2, final_energy)


def _dspacing_from_tof(tof, Ltotal, two_theta):
    c = sc.to_unit(2 * const.m_n / const.h,
                   tof.unit / sc.units.angstrom / Ltotal.unit,
                   copy=False)
    return tof / (c * Ltotal * sc.sin(two_theta / 2))


# TODO consumes position, do we want that in scatter=False?
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


def _inelastic_scatter_graph(data):
    if 'incident_energy' in data.coords:
        if 'final_energy' in data.coords:
            raise RuntimeError(
                "Data contains coords for incident *and* final energy, cannot have "
                "both for inelastic scattering.")
        return {
            **SCATTER_GRAPH_DETECTOR_TO_PHYS, 'energy_transfer':
            _energy_transfer_direct_from_tof
        }
    elif 'final_energy' in data.coords:
        return {
            **SCATTER_GRAPH_DETECTOR_TO_PHYS, 'energy_transfer':
            _energy_transfer_indirect_from_tof
        }
    raise RuntimeError(
        "Data contains neither coords for incident nor for final energy, this "
        "does not appear to be inelastic-scattering data, cannot convert to "
        "energy transfer.")


def _scatter_graph(data, origin, target):
    if target == 'energy_transfer':
        return _inelastic_scatter_graph(data)
    # else: elastic
    for graph in (SCATTER_GRAPH_DETECTOR_TO_PHYS, ):
        if path_exists(origin, target, graph):
            return graph

    raise RuntimeError(f"No viable conversion from '{origin}' to '{target}'.")


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
