# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

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
