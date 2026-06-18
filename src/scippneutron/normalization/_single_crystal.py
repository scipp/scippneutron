# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Normalization routines for single-crystal experiments (SXD and INS)."""

import numpy as np
import numpy.typing as npt
import scipp as sc
import scipp.constants


def compute_q_de_norm(
    *,
    trajectory_start: sc.Variable,  # shape: [*other, pixel, hkl_kf]
    trajectory_stop: sc.Variable,
    solid_angle: sc.Variable,
    grid: tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable],
    incident_energy: sc.Variable,
) -> sc.DataArray:
    """TODO

    The grid is specified in (h, k, l, dE),
    gets converted to (h, k, l, kf)
    The trajectory is specified in (h, k, l, kf)
    """
    for edges in grid:
        if not sc.issorted(edges, edges.dim):
            raise sc.CoordError(f"The input bin-edges must be sorted, got {edges}")
    # TODO convert units to match between traj and grid

    grid_energy_transfer = grid[3]
    grid = (
        *grid[:3],
        _flip_array(
            _energy_to_final_momentum(
                energy_transfer=grid[3], incident_energy=incident_energy
            ).rename(energy_transfer='kf')
        ),
    )

    # Per traj and dimension smallest and larges values,
    # cannot use left=start, right=stop because trajectories are not sorted like that.
    traj_left = np.minimum(trajectory_start.values, trajectory_stop.values)
    traj_right = np.maximum(trajectory_start.values, trajectory_stop.values)

    interior_grid = _filter_grid(grid, traj_left, traj_right)

    intersections = _compute_trajectory_grid_intersections(
        start=trajectory_start,
        stop=trajectory_stop,
        traj_left=traj_left,
        traj_right=traj_right,
        grid=interior_grid,
    )
    coverage = _compute_detector_coverage(
        segment_ends=intersections, solid_angle=solid_angle
    )
    coverage.coords['energy_transfer'] = incident_energy - _momentum_to_energy(
        coverage.coords.pop('kf')
    )
    norm = coverage.hist(
        {
            'h': grid[0],
            'k': grid[1],
            'l': grid[2],
            'energy_transfer': grid_energy_transfer,
        }
    )

    return norm


def _filter_grid(
    grid: tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable],
    traj_left: npt.NDArray[np.float64],
    traj_right: npt.NDArray[np.float64],
) -> tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable]:
    """Filter out all edges that are outside the range of the trajectories.

    For computing intersections, we only need grid lines that are between the
    trajectory endpoints.
    """
    total_left = np.min(traj_left, axis=0)
    total_right = np.max(traj_right, axis=0)

    def select_in_range(axis: int, edges: sc.Variable) -> sc.Variable:
        sel = (edges.values >= total_left[axis]) & (edges.values < total_right[axis])
        return sc.array(dims=edges.dims, values=edges.values[sel], unit=edges.unit)

    return tuple(select_in_range(axis, edges) for axis, edges in enumerate(grid))


# TODO move to coord transforms (and use in essspectroscopy)
def _energy_to_final_momentum(
    *, incident_energy: sc.Variable, energy_transfer: sc.Variable
) -> sc.Variable:
    final_energy = incident_energy - energy_transfer
    return sc.to_unit(
        sc.sqrt(2 * sc.constants.m_n / sc.constants.hbar**2 * final_energy),
        '1/Å',
        copy=False,
    )


def _flip_array(x: sc.Variable) -> sc.Variable:
    return sc.array(dims=x.dims, values=x.values[::-1], unit=x.unit)


def _momentum_to_energy(mom: sc.Variable) -> sc.Variable:
    return sc.to_unit(
        sc.constants.hbar**2 / (2 * sc.constants.m_n) * mom**2, 'meV', copy=False
    )


def _compute_trajectory_grid_intersections(
    start: sc.Variable,
    stop: sc.Variable,
    traj_left: npt.NDArray[np.float64],
    traj_right: npt.NDArray[np.float64],
    grid: tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable],
) -> sc.Variable:
    intersections = [np.stack([start.values, stop.values])]

    for dim in range(4):
        slope = (stop - start) / (stop['q-e', dim] - start['q-e', dim])
        pos = slope * (grid[dim] - start['q-e', dim]) + start

        # The condition here is right-inclusive even though binning is right-exclusive.
        # This is required to handle trajectories that are constant in one dimension.
        # Doing this is ok because this will lead to an intersection with distance 0 to
        # a trajectory endpoint and so does not contribute to the result.
        pos.values[:] = np.where(
            (pos.values < traj_left) | (pos.values > traj_right),
            np.nan,
            pos.values,
        )

        intersections.append(pos.values)

    inter = np.concat(intersections, axis=0)
    idx = np.argsort(inter[:, :, 3], axis=0)
    sorted = np.stack(
        [np.take_along_axis(inter[:, :, i], idx, axis=0) for i in range(4)], axis=2
    )
    return sc.array(dims=['intersection', *start.dims], values=sorted, unit='1/Å')


def _compute_detector_coverage(
    segment_ends: sc.Variable,
    solid_angle: sc.Variable,
) -> sc.DataArray:
    centers = sc.midpoints(segment_ends, dim='intersection')
    energy = _momentum_to_energy(segment_ends)
    energy_delta = (
        energy['intersection', 1:]['q-e', 3] - energy['intersection', :-1]['q-e', 3]
    )
    return (
        sc.DataArray(
            energy_delta * solid_angle,
            coords={
                'h': centers['q-e', 0],
                'k': centers['q-e', 1],
                'l': centers['q-e', 2],
                'kf': centers['q-e', 3],
                'pixel': sc.arange('pixel', segment_ends.sizes['pixel'], unit=None),
            },
        )
        .flatten(to='observation')
        .copy()
    )
