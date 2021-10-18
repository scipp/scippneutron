# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from ..core.conversion import _SCATTER_GRAPH_KINEMATICS, _NO_SCATTER_GRAPH_KINEMATICS, \
        _SCATTER_GRAPH_DYNAMICS_BY_ORIGIN, _energy_transfer_direct_from_tof, \
        _energy_transfer_indirect_from_tof


def beamline(scatter: bool):
    if scatter:
        return dict(_SCATTER_GRAPH_KINEMATICS)
    else:
        return dict(_NO_SCATTER_GRAPH_KINEMATICS)


def elastic(start: str):
    return dict(_SCATTER_GRAPH_DYNAMICS_BY_ORIGIN[start])


def direct_inelastic(start: str):
    return {'tof': {'energy_transfer': _energy_transfer_direct_from_tof}}[start]


def indirect_inelastic(start: str):
    return {'tof': {'energy_transfer': _energy_transfer_indirect_from_tof}}[start]
