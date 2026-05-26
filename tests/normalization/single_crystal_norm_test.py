# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E741  # we use `l` here

from collections.abc import Callable
from typing import Any, TypeAlias

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

    def norm_grid(
        self,
        shape: tuple[int, int, int, int],
        cells: list[tuple[int, int, int, int]],
        segments: list[float],
    ) -> sc.Variable:
        """Construct a normalization factor on a hkl-dE grid.

        ``cells`` is a list of indices of grid cells to fill.
        ``segments`` is a list of dE edges for each cell.
        ```len(segments) == len(cells) + 1```, like bin-edges.
        """
        de_segments = [
            self.kf_to_de(sc.scalar(segment, unit='1/Å')) for segment in segments
        ]
        expected = sc.zeros(
            dims=['h', 'k', 'l', 'energy_transfer'], shape=shape, unit='meV'
        )
        for i, cell in enumerate(cells):
            de = abs(de_segments[i + 1] - de_segments[i])
            expected['h', cell[0]]['k', cell[1]]['l', cell[2]][
                'energy_transfer', cell[3]
            ].value = de.value
        return expected


@pytest.fixture(params=[1.3, 0.04], ids=["Ei0", "Ei1"])
def helper(request: Any) -> TrajectoryHelper:
    return TrajectoryHelper(sc.scalar(request.param, unit='meV'))


def to_grid_variables(
    h: float, k: float, l: float, mom: float
) -> tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable]:
    return (
        sc.scalar(h),
        sc.scalar(k),
        sc.scalar(l),
        sc.scalar(mom, unit='1/Å'),
    )


Trajectory: TypeAlias = tuple[
    tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable],
    tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable],
]


def traj_as_given(
    start: tuple[float, float, float, float], stop: tuple[float, float, float, float]
) -> Trajectory:
    return to_grid_variables(*start), to_grid_variables(*stop)


def traj_flipped(
    start: tuple[float, float, float, float], stop: tuple[float, float, float, float]
) -> Trajectory:
    return to_grid_variables(*stop), to_grid_variables(*start)


@pytest.fixture(params=[traj_as_given, traj_flipped], ids=['normal', 'flipped'])
def make_trajectory(
    request: Any,
) -> Callable[
    [tuple[float, float, float, float], tuple[float, float, float, float]], Trajectory
]:
    return request.param


# TODO test invariants:
#    - rotate hkl in traj and grid
#    - multi traj: swap trajectories
#    - flip ends
#    - shift in grid by multiple of cell length -> norm shifts the same


# TODO ranges of other hkl (test with single bin)
def test_single_crystal_norm_ins_det_traj_within_grid_2d(
    helper: TrajectoryHelper, make_trajectory: Callable[..., Trajectory]
) -> None:
    """Case A1 from tools/detector_test_trajectories.py

    Only the blue trajectory.
    """
    trajectory_start, trajectory_stop = make_trajectory(
        (0.1, 0.0, 0.0, 1.0), (0.9, 0.0, 0.0, 1.5)
    )

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.0, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('l', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.5, 0.9, 1.3, 1.6], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=[trajectory_start],
        trajectory_stop=[trajectory_stop],
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    a1 = 0.125
    c1 = 0.075
    cells = [(0, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 2), (2, 1, 1, 2)]
    segments = [1.0, 1.0 + a1, 1.3, 1.3 + c1, 1.5]
    expected = helper.norm_grid(
        shape=(4, 3, 3, 3),
        cells=cells,
        segments=segments,
    )

    sc.testing.assert_allclose(norm, expected)


def test_single_crystal_norm_ins_det_traj_within_grid_2d_multi_traj(
    helper: TrajectoryHelper, make_trajectory: Callable[..., Trajectory]
) -> None:
    """Case A2 from tools/detector_test_trajectories.py"""
    trajectory_start1, trajectory_stop1 = make_trajectory(
        (0.1, 0.0, 0.0, 1.0), (0.9, 0.0, 0.0, 1.5)
    )
    trajectory_start2, trajectory_stop2 = make_trajectory(
        (0.5, 0.0, 0.0, 0.9), (0.8, 0.0, 0.0, 1.4)
    )

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.0, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('l', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.5, 0.9, 1.3, 1.6], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=[trajectory_start1, trajectory_start2],
        trajectory_stop=[trajectory_stop1, trajectory_stop2],
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    a1 = 0.125
    c1 = 0.075

    a2 = 1 / 3
    c2 = 0.1

    expected1 = helper.norm_grid(
        shape=(4, 3, 3, 3),
        cells=[(0, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 2), (2, 1, 1, 2)],
        segments=[1.0, 1.0 + a1, 1.3, 1.3 + c1, 1.5],
    )
    expected2 = helper.norm_grid(
        shape=(4, 3, 3, 3),
        cells=[(1, 1, 1, 1), (2, 1, 1, 1), (2, 1, 1, 2)],
        segments=[0.9, 0.9 + a2, 1.3, 1.3 + c2],
    )
    expected = expected1 + expected2

    sc.testing.assert_allclose(norm, expected)


def test_single_crystal_norm_ins_det_traj_ends_outside_grid_2d(
    helper: TrajectoryHelper, make_trajectory: Callable[..., Trajectory]
) -> None:
    """Case B from tools/detector_test_trajectories.py"""
    trajectory_start, trajectory_stop = make_trajectory(
        (1.22, 0.0, 0.0, 1.1), (1.35, 0.0, 0.0, 0.1)
    )

    h_edges = sc.array(dims=['h'], values=[0.9, 1.0, 1.2, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('l', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.0, 0.2, 0.7, 1.0], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=[trajectory_start],
        trajectory_stop=[trajectory_stop],
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    b = 0.21538461538461542
    expected = helper.norm_grid(
        shape=(3, 3, 3, 3),
        cells=[(2, 1, 1, 2), (2, 1, 1, 1)],
        segments=[1.0, 0.7, 0.7 - b],
    )

    sc.testing.assert_allclose(norm, expected)


def test_single_crystal_norm_ins_det_traj_start_inside_end_outside_grid_2d(
    helper: TrajectoryHelper, make_trajectory: Callable[..., Trajectory]
) -> None:
    """Case C from tools/detector_test_trajectories.py"""
    trajectory_start, trajectory_stop = make_trajectory(
        (1.0, 0.0, 0.0, 0.9), (0.6, 0.0, 0.0, 0.3)
    )

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.1, 1.5, 1.9])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('l', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.4, 0.5, 1.0, 1.1], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=[trajectory_start],
        trajectory_stop=[trajectory_stop],
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    b = 0.05
    expected = helper.norm_grid(
        shape=(5, 3, 3, 3),
        cells=[(2, 1, 1, 1), (2, 1, 1, 0), (1, 1, 1, 0)],
        segments=[0.9, 0.5, 0.5 - b, 0.4],
    )

    sc.testing.assert_allclose(norm, expected)


def test_single_crystal_norm_ins_det_traj_single_cell_grid_2d(
    helper: TrajectoryHelper, make_trajectory: Callable[..., Trajectory]
) -> None:
    """Case D from tools/detector_test_trajectories.py"""
    trajectory_start, trajectory_stop = make_trajectory(
        (0.6, 0.0, 0.0, 1.0), (0.4, 0.0, 0.0, 1.2)
    )

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.0, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('l', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.5, 0.9, 1.3, 1.6], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=[trajectory_start],
        trajectory_stop=[trajectory_stop],
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    expected = helper.norm_grid(
        shape=(4, 3, 3, 3),
        cells=[(1, 1, 1, 1)],
        segments=[1.0, 1.2],
    )

    sc.testing.assert_allclose(norm, expected)


def test_single_crystal_norm_ins_det_traj_vertical_grid_2d(
    helper: TrajectoryHelper, make_trajectory: Callable[..., Trajectory]
) -> None:
    """Case E from tools/detector_test_trajectories.py"""
    trajectory_start, trajectory_stop = make_trajectory(
        (0.4, 0.0, 0.0, 0.6), (0.4, 0.0, 0.0, 1.4)
    )

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.0, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('l', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.5, 0.9, 1.3, 1.6], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=[trajectory_start],
        trajectory_stop=[trajectory_stop],
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    expected = helper.norm_grid(
        shape=(4, 3, 3, 3),
        cells=[(1, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 2)],
        segments=[0.6, 0.9, 1.3, 1.4],
    )

    sc.testing.assert_allclose(norm, expected)


def test_single_crystal_norm_ins_det_traj_at_grid_lines_grid_2d(
    helper: TrajectoryHelper, make_trajectory: Callable[..., Trajectory]
) -> None:
    """Case F from tools/detector_test_trajectories.py"""
    trajectory_start, trajectory_stop = make_trajectory(
        (0.3, 0.0, 0.0, 0.7), (0.8, 0.0, 0.0, 1.3)
    )

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.0, 1.3])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('l', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.5, 0.9, 1.3, 1.6], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=[trajectory_start],
        trajectory_stop=[trajectory_stop],
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    b = 0.28
    expected = helper.norm_grid(
        shape=(4, 3, 3, 3),
        cells=[(1, 1, 1, 0), (1, 1, 1, 1), (2, 1, 1, 1)],
        segments=[0.7, 0.9, 0.9 + b, 1.3],
    )

    sc.testing.assert_allclose(norm, expected)


def test_single_crystal_norm_ins_det_traj_outside_grid_single_cell(
    helper: TrajectoryHelper, make_trajectory: Callable[..., Trajectory]
) -> None:
    """Case G from tools/detector_test_trajectories.py"""
    trajectory_start, trajectory_stop = make_trajectory(
        (2.0, 0.0, 0.0, 0.6), (2.0, 0.0, 0.0, 0.9)
    )

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.1, 1.5, 1.9])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('l', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.4, 0.5, 1.0, 1.1], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=[trajectory_start],
        trajectory_stop=[trajectory_stop],
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    expected = helper.norm_grid(shape=(5, 3, 3, 3), cells=[], segments=[])

    sc.testing.assert_allclose(norm, expected)


def test_single_crystal_norm_ins_det_traj_outside_grid_multi_cell(
    helper: TrajectoryHelper, make_trajectory: Callable[..., Trajectory]
) -> None:
    """Case H from tools/detector_test_trajectories.py"""
    trajectory_start, trajectory_stop = make_trajectory(
        (0.1, 0.0, 0.0, 1.2), (1.2, 0.0, 0.0, 1.2)
    )

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.1, 1.5, 1.9])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('l', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.4, 0.5, 1.0, 1.1], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=[trajectory_start],
        trajectory_stop=[trajectory_stop],
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    expected = helper.norm_grid(shape=(5, 3, 3, 3), cells=[], segments=[])

    sc.testing.assert_allclose(norm, expected)


def test_single_crystal_norm_ins_det_traj_outside_grid_diagonal(
    helper: TrajectoryHelper, make_trajectory: Callable[..., Trajectory]
) -> None:
    """Case I from tools/detector_test_trajectories.py"""
    trajectory_start, trajectory_stop = make_trajectory(
        (-0.4, 0.0, 0.0, 0.6), (0.3, 0.0, 0.0, 0.1)
    )

    h_edges = sc.array(dims=['h'], values=[-0.1, 0.3, 0.7, 1.1, 1.5, 1.9])
    k_edges = sc.linspace('k', -0.5, 0.5, 4)
    l_edges = sc.linspace('l', -0.5, 0.5, 4)
    mom_edges = sc.array(dims=['mom'], values=[0.4, 0.5, 1.0, 1.1], unit='1/Å')
    edges = (h_edges, k_edges, l_edges, helper.kf_to_de(mom_edges))

    norm = compute_q_de_norm(
        trajectory_start=[trajectory_start],
        trajectory_stop=[trajectory_stop],
        grid=edges,
        incident_energy=helper.incident_energy,
    )

    expected = helper.norm_grid(shape=(5, 3, 3, 3), cells=[], segments=[])

    sc.testing.assert_allclose(norm, expected)
