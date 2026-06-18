# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Normalization routines for single-crystal experiments (SXD and INS)."""

import numpy as np
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

    grid_energy_transfer = grid[3]
    grid = (
        *grid[:3],
        _flip_array(
            _energy_to_final_momentum(
                energy_transfer=grid[3], incident_energy=incident_energy
            ).rename(energy_transfer='kf')
        ),
    )

    # TODO prefilter

    intersections = _compute_trajectory_grid_intersections(
        trajectory_start, trajectory_stop, grid
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


def _trim_nan(array: np.ndarray) -> tuple[np.ndarray, int]:
    is_nan = np.isnan(array)
    trim_start = 0
    for i, x in enumerate(is_nan):
        if not x:
            trim_start = i
            break
    trim_end = 0
    for i, x in enumerate(is_nan[::-1]):
        if not x:
            trim_end = i
            break

    trimmed = array[trim_start : len(array) - trim_end]
    if np.isnan(trimmed).any():
        raise ValueError("Array contains interior NaN")
    return trimmed, trim_start


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
    grid: tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable],
) -> sc.Variable:
    intersections = [np.stack([start.values, stop.values])]

    traj_left = np.minimum(start.values, stop.values)
    traj_right = np.maximum(start.values, stop.values)

    for dim in range(4):
        slope = (stop - start) / (stop['q-e', dim] - start['q-e', dim])
        pos = slope * (grid[dim] - start['q-e', dim]) + start

        # The condition here is right-inclusive even though binning is right-exclusive.
        # This is ok because this will lead to an intersection with distance 0 to
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


# TODO optimise (needs to handle NaN)
def _index_of(point: float, array: np.ndarray) -> int:
    """Assumes that `array` is sorted."""
    for i, val in enumerate(array):
        if val > point:
            return i - 1
    raise ValueError("Element not in array")  # should never happen (? maybe with NaNs)


def _is_in_grid(
    point: tuple[float, float, float, float],
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    return all(edges[0] <= p < edges[-1] for p, edges in zip(point, grid, strict=True))


def _midpoints(a):
    return (a[0:-1] + a[1:]) / 2
