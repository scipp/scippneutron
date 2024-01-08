# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import uuid
from typing import Union

import numpy as np
import scipp as sc


def find_plateaus(
    data: sc.DataArray,
    *,
    atol: sc.Variable,
    min_n_points: Union[int, sc.Variable],
    plateau_dim: str = 'plateau',
) -> sc.DataArray:
    """Find regions where the input data is approximately constant.

    Plateaus are found by collecting streaks of points that differ by less than
    the given absolute tolerance.
    Variances are ignored for this comparison.

    Parameters
    ----------
    data:
        Data to search for plateaus in.
        Must be 1-dimensional.
    atol:
        Absolute tolerance.
    min_n_points:
        Only return plateaus that have at least this many points.
    plateau_dim:
        Name for the plateau-dimension in the output.

    Returns
    -------
    :
        A 1d binned data array where each bin holds the data points for one plateau.

    See Also
    --------
    scippneutron.chopper.filtering.collapse_plateaus:
        Combine plateau bins into dense data.
    """
    if data.ndim != 1:
        raise NotImplementedError(
            'find_plateaus only supports 1-dimensional data, ' f'got {data.ndim} dims'
        )

    min_n_points = (
        min_n_points
        if isinstance(min_n_points, sc.Variable)
        else sc.index(min_n_points)
    )
    diff = abs(data.data[1:] - data.data[:-1])
    group_id = sc.cumsum((diff > atol).to(dtype='int64'))
    # Prepend a 0 to align the groups with the data points (diff reduces length by 1).
    group_id = sc.concat([sc.index(0, dtype='int64'), group_id], dim=diff.dim)

    group_label = str(uuid.uuid4())
    to_group = data.copy(deep=False)
    to_group.coords[group_label] = group_id
    groups = to_group.group(group_label)
    del groups.coords[group_label]

    plateaus = groups[groups.bins.size().data >= min_n_points].rename_dims(
        {group_label: plateau_dim}
    )
    plateaus.coords[plateau_dim] = sc.arange(plateau_dim, len(plateaus), unit=None)
    return plateaus


def _next_highest(x: sc.Variable) -> sc.Variable:
    if x.dtype in ('float64', 'float32'):
        return sc.array(
            dims=x.dims,
            variances=x.variances,
            unit=x.unit,
            dtype=x.dtype,
            values=np.nextafter(x.values, np.inf),
        )
    if x.dtype == 'datetime64':
        return x + sc.scalar(1, dtype='int64', unit=x.unit)
    return x + sc.scalar(1, dtype=x.dtype, unit=x.unit)


def collapse_plateaus(plateaus: sc.DataArray, *, coord: str = 'time') -> sc.DataArray:
    """Merge plateaus bins into dense data.

    Useful for post-processing the result of
    :func:`scippneutron.chopper.find_plateaus
    <scippneutron.chopper.filtering.find_plateaus>`.
    Averages the bins and turns the given event coordinate into a bin-edge coordinate.

    Parameters
    ----------
    plateaus:
        Data that has been binned into plateaus.
    coord:
        Name of the constructed bin-edge coordinate.

    Returns
    -------
    :
        Dense data with one element per plateau.

    See Also
    --------
    scippneutron.chopper.filtering.find_plateaus:
        Function to construct suitable input data.
    """
    collapsed = plateaus.bins.mean()
    low = plateaus.bins.coords[coord].bins.min()
    high = _next_highest(plateaus.bins.coords[coord].bins.max())
    collapsed.coords[coord] = sc.concat([low, high], dim=coord)
    return collapsed


def _is_approximate_multiple(
    x: sc.Variable, *, ref: sc.Variable, rtol: sc.Variable
) -> sc.Variable:
    # If x = n * ref
    quot = x / ref
    a = abs(sc.round(quot) - quot) < rtol
    # If n = x * ref
    quot = sc.reciprocal(quot)
    b = abs(sc.round(quot) - quot) < rtol
    return a | b


def _is_in_phase(
    frequency: sc.DataArray, *, reference: sc.Variable, rtol: sc.Variable
) -> sc.Variable:
    return _is_approximate_multiple(frequency.data, ref=reference, rtol=rtol)


def filter_in_phase(
    frequency: sc.DataArray, *, reference: sc.Variable, rtol: sc.Variable
) -> sc.DataArray:
    """Remove all elements where a given frequency is not in phase with a reference.

    Frequencies are considered to be in-phase if they are an integer multiple
    of the reference or vice versa.

    Parameters
    ----------
    frequency:
        Frequency data to filter.
    reference:
        Reference frequency, e.g., the neutron source frequency.
    rtol:
        Relative tolerance.

    Returns
    -------
    :
        ``frequency`` with all out-of-phase elements removed.
    """
    in_phase = _is_in_phase(frequency, reference=reference, rtol=rtol)
    return frequency[in_phase]
