# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""Graphs for computing beamline parameters from positions.

See :mod:`scippneutron.conversion.beamline` for definitions
of the quantities used here.
"""

from collections.abc import Callable

from .. import beamline as _kernels

Graph = dict[str, Callable]


def incident_beam() -> Graph:
    """Graph for computing 'incident_beam'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return {'incident_beam': _kernels.straight_incident_beam}


def scattered_beam() -> Graph:
    """Graph for computing 'scattered_beam'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return {'scattered_beam': _kernels.straight_scattered_beam}


def two_theta() -> Graph:
    """Graph for computing the scattering angle 'two_theta'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    graph = beamline(scatter=True)
    for node in ['L1', 'L2', 'Ltotal']:
        del graph[node]
    return graph


def L1() -> Graph:
    """Graph for computing the primary path length 'L1'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return {'incident_beam': _kernels.straight_incident_beam, 'L1': _kernels.L1}


def L2() -> Graph:
    """Graph for computing the secondary path length 'L2'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return {'scattered_beam': _kernels.straight_scattered_beam, 'L2': _kernels.L2}


def Ltotal(scatter: bool) -> Graph:
    """Graph for computing 'Ltotal'.

    Parameters
    ----------
    scatter:
        If True, a graph for scattering from ``sample_position`` is
        returned, else a graph without scattering.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    graph = beamline(scatter=scatter)
    if not scatter:
        return graph
    del graph['two_theta']
    return graph


_NO_SCATTER_GRAPH_BEAMLINE = {
    'Ltotal': _kernels.total_straight_beam_length_no_scatter,
}

_SCATTER_GRAPH_BEAMLINE = {
    'incident_beam': _kernels.straight_incident_beam,
    'scattered_beam': _kernels.straight_scattered_beam,
    'L1': _kernels.L1,
    'L2': _kernels.L2,
    'two_theta': _kernels.two_theta,
    'Ltotal': _kernels.total_beam_length,
}


def beamline(scatter: bool) -> Graph:
    """Graph defining a straight beamline geometry.

    This can be used as part of transformation graphs that require, e.g., scattering
    angles (``two_theta``) or flight path lengths (``L1`` and ``L2``).

    Parameters
    ----------
    scatter:
        If True, a graph for scattering from ``sample_position`` is
        returned, else a graph without scattering.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    if scatter:
        return dict(_SCATTER_GRAPH_BEAMLINE)
    else:
        return dict(_NO_SCATTER_GRAPH_BEAMLINE)
