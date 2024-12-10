# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc


def to_events(
    da: sc.DataArray, event_dim: str, events_per_bin: int = 500
) -> sc.DataArray:
    """
    Convert a histogrammed data array to an event list.
    Each dimension with a bin-edge coordinate is converted to an event coordinate.
    The contract is that if we re-histogram the event list with the same bin edges,
    we should get the original counts back.

    Parameters
    ----------
    da:
        DataArray to convert to events.
    event_dim:
        Name of the new event dimension.
    events_per_bin:
        Number of events to generate per bin.
    """
    rng = np.random.default_rng()
    event_coords = {}
    edge_dims = []
    midp_dims = []
    # Separate bin-edge and midpoints coords
    for dim in da.dims:
        if da.coords.is_edges(dim):
            edge_dims.append(dim)
        else:
            midp_dims.append(dim)

    edge_sizes = {dim: da.sizes[dim] for dim in edge_dims}
    for dim in edge_dims:
        coord = da.coords[dim]
        low = sc.broadcast(coord[dim, :-1], sizes=edge_sizes).values
        high = sc.broadcast(coord[dim, 1:], sizes=edge_sizes).values

        # The numpy.random.uniform function below does not support NaNs, so we need to
        # replace them with zeros, and then replace them back after the random numbers
        # have been generated.
        nans = np.isnan(low) | np.isnan(high)
        low = np.where(nans, 0.0, low)
        high = np.where(nans, 0.0, high)

        # In each bin, we generate a number of events with a uniform distribution.
        events = rng.uniform(
            low, high, size=(events_per_bin, *list(edge_sizes.values()))
        )
        events[..., nans] = np.nan
        event_coords[dim] = sc.array(
            dims=[event_dim, *edge_dims], values=events, unit=coord.unit
        )

    # Create the data counts, which are the original counts divided by the number of
    # events per bin
    sizes = {event_dim: events_per_bin} | da.sizes
    val = sc.broadcast(sc.values(da.data) / float(events_per_bin), sizes=sizes)
    kwargs = {'dims': sizes.keys(), 'values': val.values, 'unit': da.data.unit}
    if da.data.variances is not None:
        kwargs['variances'] = sc.broadcast(
            sc.variances(da.data) / float(events_per_bin), sizes=sizes
        ).values
    data = sc.array(**kwargs)

    new = sc.DataArray(data=data, coords=event_coords)
    new = new.transpose((*midp_dims, *edge_dims, event_dim)).flatten(
        dims=[*edge_dims, event_dim], to=event_dim
    )
    for dim in midp_dims:
        new.coords[dim] = da.coords[dim].copy()
    return new
