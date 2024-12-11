# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc

from scippneutron.tof.to_events import to_events


def test_to_events_1d():
    table = sc.data.table_xyz(1000)
    hist = table.hist(x=20)
    events = to_events(hist, 'event')
    assert 'x' not in events.dims
    result = events.hist(x=hist.coords['x'])
    assert sc.identical(hist.coords['x'], result.coords['x'])
    assert sc.allclose(hist.data, result.data)


def test_to_events_1d_with_group_coord():
    table = sc.data.table_xyz(1000, coord_max=10)
    table.coords['l'] = table.coords['x'].to(dtype=int)
    hist = table.group('l').hist(x=20)
    events = to_events(hist, 'event')
    assert 'x' not in events.dims
    assert 'l' in events.dims
    result = events.hist(x=hist.coords['x'])
    assert sc.identical(hist.coords['x'], result.coords['x'])
    assert sc.identical(hist.coords['l'], result.coords['l'])
    assert sc.allclose(hist.data, result.data)


def test_to_events_2d():
    table = sc.data.table_xyz(1000)
    hist = table.hist(y=20, x=10)
    events = to_events(hist, 'event')
    assert 'x' not in events.dims
    assert 'y' not in events.dims
    result = events.hist(y=hist.coords['y'], x=hist.coords['x'])
    assert sc.identical(hist.coords['x'], result.coords['x'])
    assert sc.identical(hist.coords['y'], result.coords['y'])
    assert sc.allclose(hist.data, result.data)


def test_to_events_2d_with_group_coord():
    table = sc.data.table_xyz(1000, coord_max=10)
    table.coords['l'] = table.coords['x'].to(dtype=int)
    hist = table.group('l').hist(y=20, x=10)
    events = to_events(hist, 'event')
    assert 'x' not in events.dims
    assert 'y' not in events.dims
    assert 'l' in events.dims
    result = events.hist(y=hist.coords['y'], x=hist.coords['x'])
    assert sc.identical(hist.coords['x'], result.coords['x'])
    assert sc.identical(hist.coords['y'], result.coords['y'])
    assert sc.identical(hist.coords['l'], result.coords['l'])
    assert sc.allclose(hist.data, result.data)


def test_to_events_binned_data_input_raises():
    table = sc.data.table_xyz(1000)
    binned = table.bin(x=20)
    with pytest.raises(ValueError, match="Cannot convert a binned DataArray to events"):
        _ = to_events(binned, 'event')


def test_to_events_mask_on_midpoints_dim():
    table = sc.data.table_xyz(1000, coord_max=10)
    table.coords['l'] = table.coords['y'].to(dtype=int)

    hist = table.group('l').hist(x=6)
    hist.masks['m'] = hist.coords['l'] == sc.scalar(3.0, unit='m')
    events = to_events(hist, 'event')
    result = events.hist(x=hist.coords['x'])
    assert sc.identical(hist.coords['x'], result.coords['x'])
    assert sc.allclose(hist.data, result.data)
    assert sc.identical(hist.masks['m'], result.masks['m'])


def test_to_events_mask_on_binedge_dim():
    table = sc.data.table_xyz(1000, coord_max=10)
    table.coords['l'] = table.coords['x'].to(dtype=int)

    hist = table.group('l').hist(x=6)
    hist.masks['m'] = sc.array(
        dims=['x'], values=[False, False, True, True, False, False]
    )
    events = to_events(hist, 'event')
    result = events.hist(x=hist.coords['x'])
    assert sc.identical(hist.coords['x'], result.coords['x'])
    assert sc.isclose(hist.sum().data, result.sum().data)
    assert 'm' not in result.masks
    assert result['x', 2:4].data.sum() == sc.scalar(0.0, unit=table.unit)


def test_to_events_two_masks():
    table = sc.data.table_xyz(1000, coord_max=10)
    table.coords['l'] = table.coords['x'].to(dtype=int)

    hist = table.group('l').hist(x=6)
    hist.masks['m1'] = sc.array(
        dims=['x'], values=[False, False, True, True, False, False]
    )
    hist.masks['m2'] = hist.coords['l'] == sc.scalar(3.0, unit='m')
    events = to_events(hist, 'event')
    result = events.hist(x=hist.coords['x'])
    assert sc.identical(hist.coords['x'], result.coords['x'])
    assert sc.isclose(hist.sum().data, result.sum().data)
    assert 'm1' not in result.masks
    assert sc.identical(hist.masks['m2'], result.masks['m2'])
    assert result['x', 2:4].data.sum() == sc.scalar(0.0, unit=table.unit)
