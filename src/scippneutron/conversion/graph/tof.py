# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Jan-Lukas Wynen
"""Graphs for computing coordinates in time-of-flight neutron scattering.

See :mod:`scippneutron.conversion.tof` for definitions
of the quantities used here.
And the `user guide <../../user-guide/coordinate-transformations.rst>`_
for examples.

Functions in this module come in two categories and return graphs that

- can be used to compute a specific coordinate as identified by the function name
  (e.g. ``elastic_energy``) from a given coordinate as given by the ``start`` argument.
  These graphs may contain more than one node if necessary.
- can be used to compute multiple coordinates, (``elastic`` and ``kinematic``).
  Their ``start`` argument works as in the other functions.
"""

from collections.abc import Callable

from .. import tof as _kernels

Graph = dict[str, Callable]

_GRAPH_DYNAMICS_BY_ORIGIN = {
    'energy': {
        'dspacing': _kernels.dspacing_from_energy,
        'wavelength': _kernels.wavelength_from_energy,
    },
    'tof': {
        'dspacing': _kernels.dspacing_from_tof,
        'energy': _kernels.energy_from_tof,
        'hkl_vec': _kernels.hkl_vec_from_Q_vec,
        ('h', 'k', 'l'): _kernels.hkl_elements_from_hkl_vec,
        'ub_matrix': _kernels.ub_matrix_from_u_and_b,
        'Q': _kernels.Q_from_wavelength,
        'Q_vec': _kernels.Q_vec_from_Q_elements,
        ('Qx', 'Qy', 'Qz'): _kernels.Q_elements_from_wavelength,
        'wavelength': _kernels.wavelength_from_tof,
        'time_at_sample': _kernels.time_at_sample_from_tof,
    },
    'Q': {
        'wavelength': _kernels.wavelength_from_Q,
    },
    'wavelength': {
        'dspacing': _kernels.dspacing_from_wavelength,
        'energy': _kernels.energy_from_wavelength,
        'hkl_vec': _kernels.hkl_vec_from_Q_vec,
        ('h', 'k', 'l'): _kernels.hkl_elements_from_hkl_vec,
        'ub_matrix': _kernels.ub_matrix_from_u_and_b,
        'Q': _kernels.Q_from_wavelength,
        'Q_vec': _kernels.Q_vec_from_Q_elements,
        ('Qx', 'Qy', 'Qz'): _kernels.Q_elements_from_wavelength,
    },
}


def _strip_elastic(start: str, keep: list) -> Graph:
    full_graph = elastic(start)
    return {key: full_graph[key] for key in keep if key != start}


def elastic(start: str) -> Graph:
    """Graph for elastic scattering transformations.

    Parameters
    ----------
    start:
        Input coordinate. One of 'dspacing', 'energy', 'tof', 'Q',
        or 'wavelength'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return dict(_GRAPH_DYNAMICS_BY_ORIGIN[start])


def kinematic(start: str) -> Graph:
    """Graph with pure kinematics.

    The returned graph can be used to compute scattering-independent quantities.

    Parameters
    ----------
    start:
        Input coordinate. Currently, only 'tof' is supported.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return _strip_elastic(start, keep=['wavelength', 'energy'])


def elastic_dspacing(start: str) -> Graph:
    """
    Graph for elastic scattering transformation to dspacing.

    Parameters
    ----------
    start:
     Input coordinate. One of 'energy', 'tof', or 'wavelength'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return _strip_elastic(start, keep=['dspacing'])


def elastic_energy(start: str) -> Graph:
    """
    Graph for elastic scattering transformation to energy.

    Parameters
    ----------
    start:
        Input coordinate. One of 'tof' or 'wavelength'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return _strip_elastic(start, keep=['energy'])


def elastic_Q(start: str) -> Graph:
    """
    Graph for elastic scattering transformation to Q.

    Parameters
    ----------
    start:
        Input coordinate. One of 'tof' or 'wavelength'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return _strip_elastic(start, keep=['Q', 'wavelength'])


def elastic_Q_vec(start: str) -> Graph:
    """
    Graph for elastic scattering transformation to Q vector.

    Parameters
    ----------
    start:
        Input coordinate. One of 'tof' or 'wavelength'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return _strip_elastic(start, keep=[('Qx', 'Qy', 'Qz'), 'Q_vec', 'wavelength'])


def elastic_hkl(start: str) -> Graph:
    """
    Graph for elastic scattering transformation to Q vector.

    Parameters
    ----------
    start:
        Input coordinate. One of 'tof' or 'wavelength'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return _strip_elastic(
        start,
        keep=[
            ('Qx', 'Qy', 'Qz'),
            'Q_vec',
            ('h', 'k', 'l'),
            'hkl_vec',
            'ub_matrix',
            'wavelength',
        ],
    )


def elastic_wavelength(start: str) -> Graph:
    """
    Graph for elastic scattering transformation to wavelength.

    Parameters
    ----------
    start:
        Input coordinate. One of 'energy', 'tof', or 'Q'.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return _strip_elastic(start, keep=['wavelength'])


def direct_inelastic(start: str) -> Graph:
    """
    Graph for direct-inelastic scattering transformations.

    Parameters
    ----------
    start:
        Input coordinate. Currently, only 'tof' is supported.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return {'tof': {'energy_transfer': _kernels.energy_transfer_direct_from_tof}}[start]


def indirect_inelastic(start: str) -> Graph:
    """
    Graph for indirect-inelastic scattering transformations.

    Parameters
    ----------
    start:
        Input coordinate. Currently, only 'tof' is supported.

    Returns
    -------
    :
        A dict defining a coordinate transformation graph.
    """
    return {'tof': {'energy_transfer': _kernels.energy_transfer_indirect_from_tof}}[
        start
    ]
