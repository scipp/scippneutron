# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E741  # we use `l` here

import numpy as np
import pytest
import scipp as sc
import scipp.constants
import scipp.testing

from scippneutron.normalization import compute_q_de_norm


class TrajectoryHelper:
    def __init__(self, incident_energy: sc.Variable) -> None:
        self.incident_energy = incident_energy

    def kf_to_de(self, kf: sc.Variable) -> sc.Variable:
        final_energy = sc.constants.h**2 / (8 * np.pi**2 * sc.constants.m_n) * kf
        return sc.to_unit(self.incident_energy - final_energy, 'meV')


def to_grid_variables(
    h: float, k: float, l: float, mom: float
) -> tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable]:
    return (
        sc.scalar(h),
        sc.scalar(k),
        sc.scalar(l),
        sc.scalar(mom, unit='1/Å'),
    )


@pytest.mark.parametrize('helper', [TrajectoryHelper(sc.scalar(1.3, unit='meV'))])
def test_single_crystal_norm_ins_det_traj_within_grid_2d(
    helper: TrajectoryHelper,
) -> None:
    """Case A1 from tools/detector_test_trajectories.py"""
    trajectory_start = to_grid_variables(0.1, 0.0, 0.0, 1.0)
    trajectory_stop = to_grid_variables(0.9, 0.0, 0.0, 1.5)

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.0, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('l', 0.1, 1.3, 5)
    mom_edges = sc.array(dims=['mom'], values=[0.5, 0.9, 1.3, 1.6], unit='1/Å')
    energy_transfer_edges = helper.kf_to_de(mom_edges)
    edges = (h_edges, k_edges, l_edges, energy_transfer_edges)

    norm = compute_q_de_norm(
        trajectory_start=trajectory_start,
        trajectory_stop=trajectory_stop,
        grid=edges,
    )

    expected = sc.zeros(
        dims=['h', 'k', 'l', 'energy_transfer'], shape=[3, 3, 4, 3], unit='meV'
    )
    expected['k', 1]['l', 1].values = [
        [0, 0.17142857142857149, 0],
        [0, 0.12857142857142856, 0.12857142857142856],
        [0, 0, 0.17142857142857149],
        [0, 0, 0],
    ]

    sc.testing.assert_allclose(norm, expected)
