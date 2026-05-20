# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E741  # we use `l` here

import pytest
import scipp as sc
import scipp.constants
import scipp.testing

from scippneutron.normalization import compute_q_de_norm


class TrajectoryHelper:
    def __init__(self, incident_energy: sc.Variable) -> None:
        self.incident_energy = incident_energy

    def kf_to_de(self, kf: sc.Variable) -> sc.Variable:
        final_energy = sc.constants.hbar**2 / (2 * sc.constants.m_n) * kf**2
        return sc.to_unit(
            self.incident_energy - final_energy.to(unit=self.incident_energy.unit),
            'meV',
        )


def to_grid_variables(
    h: float, k: float, l: float, mom: float
) -> tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable]:
    return (
        sc.scalar(h),
        sc.scalar(k),
        sc.scalar(l),
        sc.scalar(mom, unit='1/Å'),
    )


# TODO test invariants:
#    - rotate hkl in 2d trajectory
#    - flip traj ends
#    - multi traj: swap trajectories


# TODO permutations of hkl
# TODO ranges of other hkl (test with single bin)
@pytest.mark.parametrize(
    'helper',
    [
        TrajectoryHelper(sc.scalar(1.3, unit='meV')),
        TrajectoryHelper(sc.scalar(0.04, unit='meV')),
    ],
)
def test_single_crystal_norm_ins_det_traj_within_grid_2d(
    helper: TrajectoryHelper,
) -> None:
    """Case A1 from tools/detector_test_trajectories.py"""
    trajectory_start = to_grid_variables(0.1, 0.0, 0.0, 1.0)
    trajectory_stop = to_grid_variables(0.9, 0.0, 0.0, 1.5)

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.0, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('k', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.5, 0.9, 1.3, 1.6], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=trajectory_start,
        trajectory_stop=trajectory_stop,
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    expected = sc.zeros(
        dims=['h', 'k', 'l', 'energy_transfer'], shape=[4, 3, 3, 3], unit='meV'
    )
    expected['k', 1]['l', 1].values[:] = [
        [0, 0.125, 0],
        [0, 0.175, 0.075],
        [0, 0, 0.125],
        [0, 0, 0],
    ]

    sc.testing.assert_allclose(norm, expected)


# TODO permutations of hkl
@pytest.mark.parametrize(
    'helper',
    [
        TrajectoryHelper(sc.scalar(1.3, unit='meV')),
        TrajectoryHelper(sc.scalar(0.04, unit='meV')),
    ],
)
def test_single_crystal_norm_ins_det_traj_ends_outside_grid_2d(
    helper: TrajectoryHelper,
) -> None:
    """Case B from tools/detector_test_trajectories.py"""
    trajectory_start = to_grid_variables(1.22, 0.0, 0.0, 1.1)
    trajectory_stop = to_grid_variables(1.35, 0.0, 0.0, 0.1)

    h_edges = sc.array(dims=['h'], values=[0.9, 1.0, 1.2, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('k', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.0, 0.2, 0.7, 1.0], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=trajectory_start,
        trajectory_stop=trajectory_stop,
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    expected = sc.zeros(
        dims=['h', 'k', 'l', 'energy_transfer'], shape=[3, 3, 3, 3], unit='meV'
    )
    expected['k', 1]['l', 1].values[:] = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0.21538461538461542, 0.3],
    ]

    sc.testing.assert_allclose(norm, expected)


# TODO permutations of hkl
@pytest.mark.parametrize(
    'helper',
    [
        TrajectoryHelper(sc.scalar(1.8, unit='meV')),
        TrajectoryHelper(sc.scalar(0.01, unit='meV')),
    ],
)
def test_single_crystal_norm_ins_det_traj_start_inside_end_outside_grid_2d(
    helper: TrajectoryHelper,
) -> None:
    """Case C from tools/detector_test_trajectories.py"""
    trajectory_start = to_grid_variables(1.0, 0.0, 0.0, 0.9)
    trajectory_stop = to_grid_variables(0.6, 0.0, 0.0, 0.3)

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.1, 1.5, 1.9])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('k', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.4, 0.5, 1.0, 1.1], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=trajectory_start,
        trajectory_stop=trajectory_stop,
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    expected = sc.zeros(
        dims=['h', 'k', 'l', 'energy_transfer'], shape=[5, 3, 3, 3], unit='meV'
    )
    expected['k', 1]['l', 1].values[:] = [
        [0, 0, 0],
        [0.05, 0, 0],
        [0.05, 0.4, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    sc.testing.assert_allclose(norm, expected)


# TODO permutations of hkl
@pytest.mark.parametrize(
    'helper',
    [
        TrajectoryHelper(sc.scalar(1.8, unit='meV')),
        TrajectoryHelper(sc.scalar(0.01, unit='meV')),
    ],
)
def test_single_crystal_norm_ins_det_traj_single_cell_grid_2d(
    helper: TrajectoryHelper,
) -> None:
    """Case D from tools/detector_test_trajectories.py"""
    trajectory_start = to_grid_variables(0.6, 0.0, 0.0, 1.0)
    trajectory_stop = to_grid_variables(0.4, 0.0, 0.0, 1.2)

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.0, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('k', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.5, 0.9, 1.3, 1.6], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=trajectory_start,
        trajectory_stop=trajectory_stop,
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    expected = sc.zeros(
        dims=['h', 'k', 'l', 'energy_transfer'], shape=[4, 3, 3, 3], unit='meV'
    )
    expected['k', 1]['l', 1].values[:] = [
        [0, 0, 0],
        [0, 0.2, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    sc.testing.assert_allclose(norm, expected)


# TODO permutations of hkl
@pytest.mark.parametrize(
    'helper',
    [
        TrajectoryHelper(sc.scalar(1.8, unit='meV')),
        TrajectoryHelper(sc.scalar(0.01, unit='meV')),
    ],
)
def test_single_crystal_norm_ins_det_traj_vertical_grid_2d(
    helper: TrajectoryHelper,
) -> None:
    """Case E from tools/detector_test_trajectories.py"""
    trajectory_start = to_grid_variables(0.4, 0.0, 0.0, 0.6)
    trajectory_stop = to_grid_variables(0.4, 0.0, 0.0, 1.4)

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.0, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('k', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.5, 0.9, 1.3, 1.6], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=trajectory_start,
        trajectory_stop=trajectory_stop,
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    expected = sc.zeros(
        dims=['h', 'k', 'l', 'energy_transfer'], shape=[4, 3, 3, 3], unit='meV'
    )
    expected['k', 1]['l', 1].values[:] = [
        [0, 0, 0],
        [0.3, 0.4, 0.1],
        [0, 0, 0],
        [0, 0, 0],
    ]

    sc.testing.assert_allclose(norm, expected)


# TODO permutations of hkl
@pytest.mark.parametrize(
    'helper',
    [
        TrajectoryHelper(sc.scalar(1.8, unit='meV')),
        TrajectoryHelper(sc.scalar(0.01, unit='meV')),
    ],
)
def test_single_crystal_norm_ins_det_traj_at_grid_lines_grid_2d(
    helper: TrajectoryHelper,
) -> None:
    """Case F from tools/detector_test_trajectories.py"""
    trajectory_start = to_grid_variables(0.3, 0.0, 0.0, 0.7)
    trajectory_stop = to_grid_variables(0.8, 0.0, 0.0, 1.3)

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.0, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('k', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.5, 0.9, 1.3, 1.6], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=trajectory_start,
        trajectory_stop=trajectory_stop,
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    expected = sc.zeros(
        dims=['h', 'k', 'l', 'energy_transfer'], shape=[4, 3, 3, 3], unit='meV'
    )
    expected['k', 1]['l', 1].values[:] = [
        [0, 0, 0],
        [0.2, 0.28, 0],
        [0, 0.12, 0],
        [0, 0, 0],
    ]

    sc.testing.assert_allclose(norm, expected)
