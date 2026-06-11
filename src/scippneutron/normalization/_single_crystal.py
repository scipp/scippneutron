# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E741  # we use `l` here
"""Normalization routines for single-crystal experiments (SXD and INS)."""

import numpy as np
import scipp as sc
import scipp.constants


def compute_q_de_norm(
    *,
    trajectory_start: list[tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable]],
    trajectory_stop: list[tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable]],
    solid_angle: sc.Variable,
    grid: tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable],
    incident_energy: sc.Variable,
) -> sc.DataArray:
    """TODO

    The grid is specified in (h, k, l, dE),
    gets converted to (h, k, l, kf)
    The trajectory is specified in (h, k, l, kf)
    """
    orig_grid = tuple(x.copy() for x in grid)
    grid = (
        *grid[:3],
        _energy_to_final_momentum(
            energy_transfer=grid[3], incident_energy=incident_energy
        ),
    )

    # TODO sort is bad when we have NaN from OOB delta E (?)
    #  dE -> kf reverses order, if inputs are ordered, then just flip the array
    grid = (*(x.values for x in grid[:3]), grid[3].values[::-1])

    norm = sc.zeros(
        dims=[edge.dim for edge in orig_grid],
        shape=[len(edge) - 1 for edge in grid],
        unit='meV',
    )

    trimmed, n_trimmed = _trim_nan(grid[3])
    grid = (*grid[:3], trimmed)

    for start, stop, omega in zip(
        trajectory_start, trajectory_stop, solid_angle, strict=True
    ):
        # TODO for now, convert to raw numbers:
        start = tuple(x.value for x in start)
        stop = tuple(x.value for x in stop)

        if abs(stop[3] - start[3]) < 1e-10:
            continue  # no energy transfer in this trajectory

        intersections = _compute_trajectory_grid_intersections(start, stop, grid)

        indices, segment_lengths = _compute_trajectory_segment_lengths(
            start=start,
            stop=stop,
            grid=grid,
            intersections=intersections,
        )

        for i, l in zip(indices, segment_lengths, strict=True):
            norm.values[(*i[:3], i[3] + n_trimmed)] += l * omega.value

    # TODO handle sorting better?
    norm.values[:] = norm.values[:, :, :, ::-1]
    return sc.DataArray(norm, coords={edge.dim: edge for edge in orig_grid})


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


def _momentum_to_energy(mom: sc.Variable) -> sc.Variable:
    return sc.to_unit(
        sc.constants.hbar**2 / (2 * sc.constants.m_n) * mom**2, 'meV', copy=False
    )


def _compute_trajectory_grid_intersections(
    start: tuple[float, float, float, float],
    stop: tuple[float, float, float, float],
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    intersections = []
    eps = 1e-10

    # intersections w/ gridlines in h
    dim = 0
    if abs(start[dim] - stop[dim]) > eps:
        fk = (stop[1] - start[1]) / (stop[dim] - start[dim])
        fl = (stop[2] - start[2]) / (stop[dim] - start[dim])
        fmom = (stop[3] - start[3]) / (stop[dim] - start[dim])
        for h in grid[dim]:
            if (start[dim] <= h < stop[dim]) or (stop[dim] < h <= start[dim]):
                k = fk * (h - start[dim]) + start[1]
                l = fl * (h - start[dim]) + start[2]
                if (
                    (k >= grid[1][0])
                    and (k < grid[1][-1])
                    and (l >= grid[2][0])
                    and (l < grid[2][-1])
                ):
                    mom = fmom * (h - start[dim]) + start[3]
                    if mom >= grid[3][0] and mom < grid[3][-1]:
                        intersections.append((h, k, l, mom))

    # intersections w/ gridlines in k
    dim = 1
    if abs(start[dim] - stop[dim]) > eps:
        fh = (stop[0] - start[0]) / (stop[dim] - start[dim])
        fl = (stop[2] - start[2]) / (stop[dim] - start[dim])
        fmom = (stop[3] - start[3]) / (stop[dim] - start[dim])
        for k in grid[dim]:
            if (start[dim] <= k < stop[dim]) or (stop[dim] < k <= start[dim]):
                h = fh * (k - start[dim]) + start[0]
                l = fl * (k - start[dim]) + start[2]
                if (
                    (h >= grid[0][0])
                    and (h < grid[0][-1])
                    and (l >= grid[2][0])
                    and (l < grid[2][-1])
                ):
                    mom = fmom * (k - start[dim]) + start[3]
                    if mom >= grid[3][0] and mom < grid[3][-1]:
                        intersections.append((h, k, l, mom))

    # intersections w/ gridlines in l
    dim = 2
    if abs(start[dim] - stop[dim]) > eps:
        fh = (stop[0] - start[0]) / (stop[dim] - start[dim])
        fk = (stop[1] - start[1]) / (stop[dim] - start[dim])
        fmom = (stop[3] - start[3]) / (stop[dim] - start[dim])
        for l in grid[dim]:
            if (start[dim] <= l < stop[dim]) or (stop[dim] < l <= start[dim]):
                h = fh * (l - start[dim]) + start[0]
                k = fk * (l - start[dim]) + start[1]
                if (
                    (h >= grid[0][0])
                    and (h < grid[0][-1])
                    and (k >= grid[1][0])
                    and (k < grid[1][-1])
                ):
                    mom = fmom * (l - start[dim]) + start[3]
                    if mom >= grid[3][0] and mom < grid[3][-1]:
                        intersections.append((h, k, l, mom))

    # intersections w/ gridlines in final momentum
    dim = 3
    fh = (stop[0] - start[0]) / (stop[dim] - start[dim])
    fk = (stop[1] - start[1]) / (stop[dim] - start[dim])
    fl = (stop[2] - start[2]) / (stop[dim] - start[dim])
    for mom in grid[dim]:
        if (start[dim] <= mom < stop[dim]) or (stop[dim] <= mom < start[dim]):
            h = fh * (mom - start[dim]) + start[0]
            k = fk * (mom - start[dim]) + start[1]
            l = fl * (mom - start[dim]) + start[2]
            # TODO do we need these checks? why do we not check ei in the above cases?
            if (
                (h >= grid[0][0])
                and (h < grid[0][-1])
                and (k >= grid[1][0])
                and (k < grid[1][-1])
                and (l >= grid[2][0])
                and (l < grid[2][-1])
            ):
                intersections.append((h, k, l, mom))

    return np.array(sorted(intersections, key=lambda t: t[3]))


def _compute_trajectory_segment_lengths(
    start: tuple[float, float, float, float],
    stop: tuple[float, float, float, float],
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    intersections: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p0, p1 = (start, stop) if stop[3] >= start[3] else (stop, start)
    segment_ends = intersections
    if segment_ends.size == 0:
        segment_ends = np.zeros((0, 4))
    if _is_in_grid(p0, grid):
        segment_ends = np.concat([[p0], segment_ends])
    if _is_in_grid(p1, grid):
        segment_ends = np.concat([segment_ends, [p1]])
    if segment_ends.size == 0:
        return np.array([]), np.array([])

    centers = _midpoints(segment_ends)
    indices = np.stack(
        [
            [_index_of(center, grid[dim]) for center in centers[:, dim]]
            for dim in range(len(grid))
        ]
    ).T

    # delta_e = abs(diff(energy_transfer))
    #         = abs(-diff(Ef))   # because Ei is constant
    #         = diff(Ef)         # because kf (and thus Ef) is sorted in ascending order
    delta_e = np.diff(
        _momentum_to_energy(
            sc.array(dims=['kf'], values=segment_ends[:, 3], unit='1/Å')
        ).values
    )

    return indices, delta_e


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
