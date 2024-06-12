# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import uuid

import numpy as np
import scipp as sc


def find_plateaus(
    data: sc.DataArray,
    *,
    atol: sc.Variable,
    min_n_points: int | sc.Variable,
    plateau_dim: str = "plateau",
) -> sc.DataArray:
    """Find regions where the input data is approximately constant.

    Plateaus are found by collecting streaks of points where the derivative of the input
    is less than the tolerance.
    Variances are ignored for this comparison.

    Warning
    -------
    This function technically does not search for plateaus,
    but regions with small derivative.
    This means that if there is a slope smaller than the noise in the input data,
    that sloped region may be falsely identified as a plateau.
    ``find_plateaus`` attempts to catch such a case and raise a ``RuntimeError``,
    but you should always inspect the result!

    Parameters
    ----------
    data:
        Data to search for plateaus in.
        Must be 1-dimensional.
    atol:
        Absolute tolerance for the derivative.
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
            "find_plateaus only supports 1-dimensional data, " f"got {data.ndim} dims"
        )
    if not sc.issorted(data.coords[data.dim], data.dim, order="ascending"):
        raise sc.CoordError(
            "The coord used by find_plateaus must be sorted in ascending order"
        )

    min_n_points = (
        min_n_points
        if isinstance(min_n_points, sc.Variable)
        else sc.index(min_n_points)
    )
    derivative = _derive(data)
    group_id = sc.cumsum(
        (abs(derivative) > atol.to(unit=derivative.unit)).to(dtype="int64")
    )
    # Prepend a 0 to align the groups with the data points (diff reduces length by 1).
    group_id = sc.concat([sc.index(0, dtype="int64"), group_id], dim=derivative.dim)

    group_label = str(uuid.uuid4())
    to_group = data.copy(deep=False)
    to_group.coords[group_label] = group_id
    groups = to_group.group(group_label)
    del groups.coords[group_label]

    plateaus = groups[groups.bins.size().data >= min_n_points].rename_dims(
        {group_label: plateau_dim}
    )
    plateaus.coords[plateau_dim] = sc.arange(plateau_dim, len(plateaus), unit=None)
    if exceeds_tolerance := _check_total_tolerance(plateaus, atol=atol):
        raise RuntimeError(
            f"The following plateaus exceed the tolerance: {exceeds_tolerance}"
        )
    return plateaus


def _derive(da: sc.DataArray) -> sc.Variable:
    x = da.coords[da.dim]
    y = da.data
    return (y[1:] - y[:-1]) / (x[1:] - x[:-1])


def _check_total_tolerance(plateaus: sc.DataArray, *, atol: sc.Variable) -> list[int]:
    # We assume that the noise within a plateau is random.
    # So if the points within a plateau were reordered arbitrarily, the slopes between
    # all neighbors must still be within tolerance.
    # So this function takes the extreme points of each plateau, pretends they
    # are next to each other and computes the corresponding slope and compares
    # it to the tolerance.
    #
    # There is a fudge factor of 2 in the comparison to allow for some larger
    # deviations, especially around the ends of a plateau.
    # Without it, the check would almost always fail.
    exceeds_tolerance = []
    for plateau_bin in plateaus:
        plateau = plateau_bin.value
        max_diff = plateau.data.max() - plateau.data.min()
        coord = plateau.coords[plateau.dim]
        average_step = sc.mean(coord[1:] - coord[:-1])
        slope = max_diff / average_step
        if slope > 2 * atol.to(unit=slope.unit):
            exceeds_tolerance.append(plateau_bin.coords[plateaus.dim].value)
    return exceeds_tolerance


def _next_highest(x: sc.Variable) -> sc.Variable:
    if x.dtype in ("float64", "float32"):
        return sc.array(
            dims=x.dims,
            variances=x.variances,
            unit=x.unit,
            dtype=x.dtype,
            values=np.nextafter(x.values, np.inf),
        )
    if x.dtype == "datetime64":
        return x + sc.scalar(1, dtype="int64", unit=x.unit)
    return x + sc.scalar(1, dtype=x.dtype, unit=x.unit)


def collapse_plateaus(plateaus: sc.DataArray, *, coord: str = "time") -> sc.DataArray:
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
    collapsed.coords[coord] = sc.concat([low, high], dim=coord).transpose()
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
