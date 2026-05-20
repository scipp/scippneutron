# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E741  # we use `l` here
"""Normalization routines for single-crystal experiments (SXD and INS)."""

import numpy as np
import scipp as sc
import scipp.constants


def compute_q_de_norm(
    *,
    trajectory_start: tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable],
    trajectory_stop: tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable],
    grid: tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable],
    incident_energy: sc.Variable,
) -> sc.Variable:
    """TODO

    The grid is specified in (h, k, l, dE),
    gets converted to (h, k, l, kf)
    """
    grid = (
        *grid[:3],
        _energy_to_final_momentum(
            energy_transfer=grid[3], incident_energy=incident_energy
        ),
    )

    intersections = _compute_trajectory_grid_intersections(
        trajectory_start, trajectory_stop, all_edges
    )

    indices, segment_lengths = _compute_trajectory_segment_lengths(
        start=trajectory_start,
        stop=trajectory_stop,
        edges=grid,
        intersections=intersections,
    )

    norm = sc.zeros(
        dims=['h', 'k', 'l', 'energy_transfer'],
        shape=[len(edge) - 1 for edge in grid],
        unit='meV',
    )
    # for i, l in zip(indices, segment_lengths, strict=True):
    #     norm[tuple(i)] += l

    return norm


# TODO move to coord transforms (and use in essspectroscopy)
def _energy_to_final_momentum(
    *, incident_energy: sc.Variable, energy_transfer: sc.Variable
) -> sc.Variable:
    final_energy = incident_energy - energy_transfer
    return sc.to_unit(
        8 * np.pi**2 * sc.constants.m_n / sc.constants.h**2 * final_energy, 'meV'
    )


def _compute_trajectory_grid_intersections(
    start: list[float], stop: list[float], edges: list[np.ndarray]
) -> np.ndarray:
    intersections = []
    eps = 1e-10

    # intersections w/ gridlines in h
    dim = 0
    if abs(start[dim] - stop[dim]) > eps:
        fk = (stop[1] - start[1]) / (stop[dim] - start[dim])
        fl = (stop[2] - start[2]) / (stop[dim] - start[dim])
        fmom = (stop[3] - start[3]) / (stop[dim] - start[dim])
        for h in edges[dim]:
            if start[dim] < h < stop[dim]:
                k = fk * (h - start[dim]) + start[1]
                l = fl * (h - start[dim]) + start[2]
                if (
                    (k >= edges[1][0])
                    and (k <= edges[1][-1])
                    and (l >= edges[2][0])
                    and (l <= edges[2][-1])
                ):
                    momi = fmom * (h - start[dim]) + start[3]
                    intersections.append((h, k, l, momi))

    # intersections w/ gridlines in k
    dim = 1
    if abs(start[dim] - stop[dim]) > eps:
        fh = (stop[0] - start[0]) / (stop[dim] - start[dim])
        fl = (stop[2] - start[2]) / (stop[dim] - start[dim])
        fmom = (stop[3] - start[3]) / (stop[dim] - start[dim])
        for k in edges[dim]:
            if start[dim] < k < stop[dim]:
                h = fh * (k - start[dim]) + start[0]
                l = fl * (k - start[dim]) + start[2]
                if (
                    (h >= edges[0][0])
                    and (h <= edges[0][-1])
                    and (l >= edges[2][0])
                    and (l <= edges[2][-1])
                ):
                    mom = fmom * (k - start[dim]) + start[3]
                    intersections.append((h, k, l, mom))

    # intersections w/ gridlines in l
    dim = 2
    if abs(start[dim] - stop[dim]) > eps:
        fh = (stop[0] - start[0]) / (stop[dim] - start[dim])
        fk = (stop[1] - start[1]) / (stop[dim] - start[dim])
        fmom = (stop[3] - start[3]) / (stop[dim] - start[dim])
        for l in edges[dim]:
            if start[dim] < l < stop[dim]:
                h = fh * (l - start[dim]) + start[0]
                k = fk * (l - start[dim]) + start[1]
                if (
                    (h >= edges[0][0])
                    and (h <= edges[0][-1])
                    and (k >= edges[1][0])
                    and (k <= edges[1][-1])
                ):
                    mom = fmom * (l - start[dim]) + start[3]
                    intersections.append((h, k, l, mom))

    # intersections w/ gridlines in final momentum
    dim = 3
    if abs(start[dim] - stop[dim]) < eps:
        raise ValueError("not enough delta energy")

    fh = (stop[0] - start[0]) / (stop[dim] - start[dim])
    fk = (stop[1] - start[1]) / (stop[dim] - start[dim])
    fl = (stop[2] - start[2]) / (stop[dim] - start[dim])
    for mom in edges[dim]:
        if start[dim] < mom < stop[dim]:
            h = fh * (mom - start[dim]) + start[0]
            k = fk * (mom - start[dim]) + start[1]
            l = fl * (mom - start[dim]) + start[2]
            # TODO do we need these checks? why do we not check ei in the above cases?
            if (
                (h >= edges[0][0])
                and (h <= edges[0][-1])
                and (k >= edges[1][0])
                and (k <= edges[1][-1])
                and (l >= edges[2][0])
                and (l <= edges[2][-1])
            ):
                intersections.append((h, k, l, mom))

    return np.array(sorted(intersections, key=lambda t: t[3]))


def _compute_trajectory_segment_lengths(
    start: list[float],
    stop: list[float],
    edges: list[np.ndarray],
    intersections: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p0, p1 = (start, stop) if stop[3] >= start[3] else (stop, start)
    segment_ends = intersections
    if _is_in_grid(p0, edges):
        segment_ends = np.concat([[p0], segment_ends])
    if _is_in_grid(p1, edges):
        segment_ends = np.concat([segment_ends, [p1]])
    if segment_ends.size == 0:
        return np.array([]), np.array([])

    delta_kf = np.diff(segment_ends[:, 3])

    centers = _midpoints(segment_ends)
    indices = np.stack(
        [
            np.searchsorted(edges[dim], centers[:, dim], side="right") - 1
            for dim in range(len(edges))
        ]
    ).T

    return indices, delta_kf


def _is_in_grid(point: list[float], all_edges: list[np.ndarray]) -> bool:
    return all(
        edges[0] <= p < edges[-1] for p, edges in zip(point, all_edges, strict=True)
    )


def _midpoints(a):
    return (a[0:-1] + a[1:]) / 2
