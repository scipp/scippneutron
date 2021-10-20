# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
r"""
Coordinate transformation graphs for time-of-flight neutron scattering data.

Each graph is a ``dict`` and intended for use with ``sc.transform_coords``.
Typically multiple graphs need to be combined for a full transformation, e.g., a
"beamline" graph with an "elastic" graph.
"""

from ..core.conversion import _SCATTER_GRAPH_KINEMATICS, _NO_SCATTER_GRAPH_KINEMATICS, \
        _SCATTER_GRAPH_DYNAMICS_BY_ORIGIN, _energy_transfer_direct_from_tof, \
        _energy_transfer_indirect_from_tof, _wavelength_from_tof, \
        _energy_from_tof, _incident_beam


def incident_beam():
    """
    Graph for computing 'incident_beam'.
    """
    return {'incident_beam': _incident_beam}


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
            'wavelength': _wavelength_from_tof,
            'energy': _energy_from_tof
        }
    }[start]


def elastic(start: str):
    """
    Graph for elastic scattering transformations.

    :param start: Input coordinate. One of 'energy', 'tof', 'Q', or 'wavelength'.
    """
    return dict(_SCATTER_GRAPH_DYNAMICS_BY_ORIGIN[start])


def direct_inelastic(start: str):
    """
    Graph for direct-inelastic scattering transformations.

    :param start: Input coordinate. Currently only 'tof' is supported.
    """
    return {'tof': {'energy_transfer': _energy_transfer_direct_from_tof}}[start]


def indirect_inelastic(start: str):
    """
    Graph for indirect-inelastic scattering transformations.

    :param start: Input coordinate. Currently only 'tof' is supported.
    """
    return {'tof': {'energy_transfer': _energy_transfer_indirect_from_tof}}[start]
