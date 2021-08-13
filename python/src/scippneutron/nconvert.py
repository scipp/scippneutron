# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen

import scipp as sc
from ._scippneutron import conversions

# TODO check old convert for disallowed combinations of coords, e.g. inelastic energy


def _Ltotal_no_scatter(source_position, position):
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
    return conversions.wavelength_from_tof(tof, Ltotal)


def _Q_from_wavelength(wavelength, two_theta):
    return conversions.Q_from_wavelength(wavelength, two_theta)


def _energy_from_tof(tof, Ltotal):
    return conversions.energy_from_tof(tof, Ltotal)


def _energy_transfer_direct_from_tof(tof, L1, L2, incident_energy):
    return conversions.energy_transfer_direct_from_tof(tof, L1, L2, incident_energy)


def _energy_transfer_indirect_from_tof(tof, L1, L2, final_energy):
    return conversions.energy_transfer_indirect_from_tof(tof, L1, L2, final_energy)


def _dspacing_from_tof(tof, Ltotal, two_theta):
    return conversions.dspacing_from_tof(tof, Ltotal, two_theta)


# TODO consumes position, do we want that in scatter=False?
NO_SCATTER_GRAPH = {'Ltotal': _Ltotal_no_scatter, 'wavelength': _wavelength_from_tof}

SCATTER_GRAPH_DETECTOR_TO_PHYS = {
    ('incident_beam', 'scattered_beam'): _scattering_beams,
    ('L1', 'L2', 'two_theta'): _beam_lengths_and_angle,
    'Ltotal': _total_beam_length_scatter,
    'wavelength': _wavelength_from_tof,
    'Q': _Q_from_wavelength,
    'energy': _energy_from_tof,
    'dspacing': _dspacing_from_tof,
}

SCATTER_GRAPH_PHYS_TO_DETECTOR = {}


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
    for graph in (SCATTER_GRAPH_DETECTOR_TO_PHYS, SCATTER_GRAPH_PHYS_TO_DETECTOR):
        if path_exists(origin, target, graph):
            return graph

    raise RuntimeError(f"No viable conversion from '{origin}' to '{target}'.")


def conversion_graph(data, origin, target, scatter):
    if scatter:
        return _scatter_graph(data, origin, target)
    else:
        return NO_SCATTER_GRAPH


def _remove_attr(data, name):
    if isinstance(data, sc.DataArray):
        del data.attrs[name]
    else:
        for array in data.values():
            _remove_attr(array, name)


def convert(data, origin, target, scatter, keep_origin=False):
    try:
        converted = data.transform_coords(target,
                                          graph=conversion_graph(
                                              data, origin, target, scatter))
    except KeyError as err:
        raise RuntimeError(
            f"Missing coordinate '{err.args[0]}' for conversion "
            f"from '{origin}' to '{target}'"
        ) from None

    if not keep_origin:
        _remove_attr(converted, origin)
    return converted
