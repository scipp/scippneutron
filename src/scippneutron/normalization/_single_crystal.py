"""Normalization for single crystal experiments."""

import scipp as sc


def compute_single_crystal_norm(
    *,
    trajectory_start: sc.Variable,  # shape: [*other, pixel, q-e]
    trajectory_stop: sc.Variable,
    solid_angle: sc.Variable,
    grid: tuple[sc.Variable, sc.Variable, sc.Variable, sc.Variable],
    incident_energy: sc.Variable,
    n_threads: int | None = None,
    block_size: int | None = None,
) -> sc.DataArray:
    """Compute a normalization factor for single crystal data.

    Attention
    ---------
        This function is a work in progress and will likely undergo breaking changes.

        This function does not currently normalize by proton charge.

        This function does not currently support elastic experiments.


    This function implements a normalization algorithm that is equivalent to the one
    described in :cite:`Savici:fs5205` and available in Mantid as
    `MDNorm v1 <https://docs.mantidproject.org/nightly/algorithms/MDNorm-v1.html>`_

    The grid is defined as arrays of edges in :math:`(h, k, l, \\Delta E)`.
    Trajectories are defined as start end stop positions in :math:`(h, k, l, k_f)`
    and should have unit :math:`Å^{-1}`.
    (Careful! The grid uses :math:`\\Delta E` and trajectories use :math:`k_f`.
    The reason is that the grid is a user input and the trajectories are
    expected to be determined automatically.)

    Parameters
    ----------
    trajectory_start:
        Start positions of each detector trajectory.
    trajectory_stop:
        Stop positions of each detector trajectory.
    solid_angle:
        The solid angle for each detector pixel as seen from the sample.
    grid:
        The :math:`Q - \\Delta E` grid to compute the norm on.
    incident_energy:
        The incident energy :math:`E_i`.
    n_threads:
        Number of CPU threads to use.
        Both *runtime* and *memory usage* scale nearly linearly
        with the number of threads.

        Defaults to the number of available CPU cores.
    block_size:
        Number of trajectories to process together.
        When using multi-threading, trajectories are  divided dynamically among threads
        in chunks of size ``block_size``.

        Defaults to a reasonable, experimentally determined number.


    Returns
    -------
    :
        The normalization denominator as a 4D data array
        containing a histogram on ``grid``.


    Implementation
    --------------
    The implementation of this function is in a separate Python package
    called ``scippneutron_algorithms``.
    The algorithm computes the same physical quantity as :cite:`Savici:fs5205`
    but uses a different approach which does not require sorting intersections
    or frequent synchronization between threads.
    It has not been benchmarked against Mantid.
    """
    from scippneutron_algorithms.normalization import (
        compute_single_crystal_norm as impl,
    )

    return impl(
        trajectory_start=trajectory_start,
        trajectory_stop=trajectory_stop,
        solid_angle=solid_angle,
        grid=grid,
        incident_energy=incident_energy,
        n_threads=n_threads,
        block_size=block_size,
    )
