# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Coordinate transformation graphs for time-of-flight neutron scattering data.

Each graph is a :py:class:`dict` and intended for use with
:py:func:`sc.transform_coords`.
Typically, multiple graphs need to be combined for a full transformation, e.g., a
"beamline" graph with an "elastic" graph.
"""

from ..core.conversions import _SCATTER_GRAPH_KINEMATICS, \
    _NO_SCATTER_GRAPH_KINEMATICS, \
    _SCATTER_GRAPH_DYNAMICS_BY_ORIGIN
from ..conversions.beamline import straight_incident_beam, straight_scattered_beam
from ..conversions.tof import energy_transfer_direct_from_tof, \
    energy_transfer_indirect_from_tof, wavelength_from_tof, \
    energy_from_tof


def incident_beam():
    """
    Graph for computing 'incident_beam'.
    """
    return {'incident_beam': straight_incident_beam}


def scattered_beam():
    """
    Graph for computing 'scattered_beam'.
    """
    return {'scattered_beam': straight_scattered_beam}


def two_theta():
    """
    Graph for computing 'two_theta'.
    """
    graph = beamline(scatter=True)
    for node in ['L1', 'L2', 'Ltotal']:
        del graph[node]
    return graph


def L1():
    """
    Graph for computing 'L1'.
    """
    graph = beamline(scatter=True)
    for node in ['scattered_beam', 'two_theta', 'L2', 'Ltotal']:
        del graph[node]
    return graph


def L2():
    """
    Graph for computing 'L2'.
    """
    graph = beamline(scatter=True)
    for node in ['incident_beam', 'two_theta', 'L1', 'Ltotal']:
        del graph[node]
    return graph


def Ltotal(scatter: bool):
    """
    Graph for computing 'Ltotal'.
    """
    graph = beamline(scatter=scatter)
    if not scatter:
        return graph
    del graph['two_theta']
    return graph


def beamline(scatter: bool):
    """
    Graph defining a simple beamline geometry.

    This can be used as part of transformation graphs that require, e.g., scattering
    angles (``two_theta``) or flight path lengths (``L1`` and ``L2``).

    :param scatter: If True, a graph for scattering from ``sample_position`` is
                    returned, else a graph without scattering.
    """
    if scatter:
        return dict(_SCATTER_GRAPH_KINEMATICS)
    else:
        return dict(_NO_SCATTER_GRAPH_KINEMATICS)


def kinematic(start: str):
    """
    Graph with pure kinematics without scattering.

    :param start: Input coordinate. Currently only 'tof' is supported.
    """
    return {
        'tof': {
            'wavelength': wavelength_from_tof,
            'energy': energy_from_tof
        }
    }[start]


def elastic(start: str):
    """
    Graph for elastic scattering transformations.

    :param start: Input coordinate. One of 'dspacing', 'energy', 'tof', 'Q',
                  or 'wavelength'.
    """
    return dict(_SCATTER_GRAPH_DYNAMICS_BY_ORIGIN[start])


def _strip_elastic(start: str, keep: list):
    graph = elastic(start)
    for node in ['dspacing', 'energy', 'Q', 'wavelength']:
        if node not in keep and node in graph:
            del graph[node]
    return graph


def elastic_dspacing(start: str):
    """
    Graph for elastic scattering transformation to dspacing.

    :param start: Input coordinate. One of 'energy', or 'tof', or 'wavelength'.
    """
    return _strip_elastic(start, keep=['dspacing'])


def elastic_energy(start: str):
    """
    Graph for elastic scattering transformation to energy.

    :param start: Input coordinate. One of 'tof' or 'wavelength'.
    """
    return _strip_elastic(start, keep=['energy'])


def elastic_Q(start: str):
    """
    Graph for elastic scattering transformation to Q.

    :param start: Input coordinate. One of 'tof' or 'wavelength'.
    """
    return _strip_elastic(start, keep=['Q', 'wavelength'])


def elastic_wavelength(start: str):
    """
    Graph for elastic scattering transformation to wavelength.

    :param start: Input coordinate. One of 'energy', 'tof', or 'Q'.
    """
    return _strip_elastic(start, keep=['wavelength'])


def direct_inelastic(start: str):
    """
    Graph for direct-inelastic scattering transformations.

    :param start: Input coordinate. Currently only 'tof' is supported.
    """
    return {'tof': {'energy_transfer': energy_transfer_direct_from_tof}}[start]


def indirect_inelastic(start: str):
    """
    Graph for indirect-inelastic scattering transformations.

    :param start: Input coordinate. Currently only 'tof' is supported.
    """
    return {'tof': {'energy_transfer': energy_transfer_indirect_from_tof}}[start]
