# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
import scipp.testing

from scippneutron.chopper import collapse_plateaus, filter_in_phase, find_plateaus


def test_find_plateaus_only_plateau_exact_float():
    da = sc.DataArray(
        sc.full(value=4.2, sizes={'x': 12}), coords={'x': sc.arange('x', 12)}
    )
    plateaus = find_plateaus(da, atol=sc.scalar(1e-7), min_n_points=3)
    assert plateaus.sizes == {'plateau': 1}
    sc.testing.assert_identical(
        plateaus.coords['plateau'], sc.array(dims=['plateau'], values=[0], unit=None)
    )
    sc.testing.assert_identical(plateaus[0].value, da)


def test_find_plateaus_only_plateau_exact_int():
    da = sc.DataArray(
        sc.full(value=6, sizes={'x': 12}, dtype='int64'),
        coords={'x': sc.arange('x', 12)},
    )
    plateaus = find_plateaus(da, atol=sc.scalar(1e-7), min_n_points=3)
    assert plateaus.sizes == {'plateau': 1}
    sc.testing.assert_identical(
        plateaus.coords['plateau'], sc.array(dims=['plateau'], values=[0], unit=None)
    )
    sc.testing.assert_identical(plateaus[0].value, da)


def test_find_plateaus_only_plateau_approx():
    da = sc.DataArray(
        sc.array(dims=['y'], values=[4.02, 4.05, 3.97, 4.0], unit='m'),
        coords={'y': sc.arange('y', 4, unit='s')},
    )
    plateaus = find_plateaus(da, atol=sc.scalar(0.1, unit='m/s'), min_n_points=3)
    assert plateaus.sizes == {'plateau': 1}
    sc.testing.assert_identical(
        plateaus.coords['plateau'], sc.array(dims=['plateau'], values=[0], unit=None)
    )
    sc.testing.assert_identical(plateaus[0].value, da)


def test_find_plateaus_only_plateau_select_output_dim():
    da = sc.DataArray(
        sc.full(value=4.2, sizes={'x': 12}), coords={'x': sc.arange('x', 12)}
    )
    plateaus = find_plateaus(
        da, atol=sc.scalar(1e-7), min_n_points=3, plateau_dim='custom'
    )
    assert plateaus.sizes == {'custom': 1}
    sc.testing.assert_identical(
        plateaus.coords['custom'], sc.array(dims=['custom'], values=[0], unit=None)
    )
    sc.testing.assert_identical(plateaus[0].value, da)


def test_find_plateaus_no_plateau():
    da = sc.DataArray(
        sc.array(dims=['z'], values=[3, 6, 1, 2, 3, 2]), coords={'z': sc.arange('z', 6)}
    )
    plateaus = find_plateaus(da, atol=sc.scalar(0.1), min_n_points=2)
    assert plateaus.sizes == {'plateau': 0}
    sc.testing.assert_identical(
        plateaus.coords['plateau'],
        sc.array(dims=['plateau'], values=[], dtype=da.dtype, unit=None),
    )


def test_find_plateaus_plateaus_at_ends():
    """
    Plateau numbers:
    data ^
         |       1 1 1
         |     x
         | 0 0
         +------------>
                     t
    """
    da = sc.DataArray(
        sc.array(dims=['t'], values=[0, 0, 1, 2, 2, 2]), coords={'t': sc.arange('t', 6)}
    )
    plateaus = find_plateaus(da, atol=sc.scalar(0.1), min_n_points=2)
    assert plateaus.sizes == {'plateau': 2}
    sc.testing.assert_identical(
        plateaus.coords['plateau'], sc.array(dims=['plateau'], values=[0, 1], unit=None)
    )

    plateau0 = sc.DataArray(
        sc.array(dims=['t'], values=[0, 0]),
        coords={'t': sc.array(dims=['t'], values=[0, 1])},
    )
    sc.testing.assert_identical(plateaus[0].value, plateau0)
    plateau1 = sc.DataArray(
        sc.array(dims=['t'], values=[2, 2, 2]),
        coords={'t': sc.array(dims=['t'], values=[3, 4, 5])},
    )
    sc.testing.assert_identical(plateaus[1].value, plateau1)


def test_find_plateaus_slow_start_and_end():
    """
    Plateau numbers:
    data ^
         |         x
         |   1 1 1
         | x
         +------------>
                     t
    """
    da = sc.DataArray(
        sc.array(dims=['t'], values=[-3, -2, -2, -2, -1]),
        coords={'t': sc.arange('t', 5)},
    )
    plateaus = find_plateaus(da, atol=sc.scalar(0.1), min_n_points=2)
    assert plateaus.sizes == {'plateau': 1}
    sc.testing.assert_identical(
        plateaus.coords['plateau'], sc.array(dims=['plateau'], values=[0], unit=None)
    )

    plateau0 = sc.DataArray(
        sc.array(dims=['t'], values=[-2, -2, -2]),
        coords={'t': sc.array(dims=['t'], values=[1, 2, 3])},
    )
    sc.testing.assert_identical(plateaus[0].value, plateau0)


def test_find_plateaus_steep_start_and_end():
    """
    Plateau numbers:
    data ^
         | x
         |   x
         |     1 1
         |         x
         |           x
         +------------>
                     t
    """
    da = sc.DataArray(
        sc.array(dims=['t'], values=[1, 0, -1, -1, -1.5, -2]),
        coords={'t': sc.arange('t', 6)},
    )
    plateaus = find_plateaus(da, atol=sc.scalar(0.1), min_n_points=2)
    assert plateaus.sizes == {'plateau': 1}
    sc.testing.assert_identical(
        plateaus.coords['plateau'], sc.array(dims=['plateau'], values=[0], unit=None)
    )

    plateau0 = sc.DataArray(
        sc.array(dims=['t'], values=[-1.0, -1.0]),
        coords={'t': sc.array(dims=['t'], values=[2, 3])},
    )
    sc.testing.assert_identical(plateaus[0].value, plateau0)


def test_find_plateaus_peak():
    """
    Plateau numbers:
    data ^
         |       x
         |     x   x
         | 0 0
         +------------>
                     t
    """
    da = sc.DataArray(
        sc.array(dims=['t'], values=[10, 10, 20, 25, 20]),
        coords={'t': sc.arange('t', 5)},
    )
    plateaus = find_plateaus(da, atol=sc.scalar(0.1), min_n_points=2)
    assert plateaus.sizes == {'plateau': 1}
    sc.testing.assert_identical(
        plateaus.coords['plateau'], sc.array(dims=['plateau'], values=[0], unit=None)
    )

    plateau0 = sc.DataArray(
        sc.array(dims=['t'], values=[10, 10]),
        coords={'t': sc.array(dims=['t'], values=[0, 1])},
    )
    sc.testing.assert_identical(plateaus[0].value, plateau0)


def test_find_plateaus_adjacent_plateaus():
    """
    Plateau numbers:
    data ^
         |     1 1 1
         | 0 0
         +------------>
                     t
    """
    da = sc.DataArray(
        sc.array(dims=['t'], values=[0, 0, 1, 1, 1]), coords={'t': sc.arange('t', 5)}
    )
    plateaus = find_plateaus(da, atol=sc.scalar(0.1), min_n_points=2)
    assert plateaus.sizes == {'plateau': 2}
    sc.testing.assert_identical(
        plateaus.coords['plateau'], sc.array(dims=['plateau'], values=[0, 1], unit=None)
    )

    plateau0 = sc.DataArray(
        sc.array(dims=['t'], values=[0, 0]),
        coords={'t': sc.array(dims=['t'], values=[0, 1])},
    )
    sc.testing.assert_identical(plateaus[0].value, plateau0)
    plateau1 = sc.DataArray(
        sc.array(dims=['t'], values=[1, 1, 1]),
        coords={'t': sc.array(dims=['t'], values=[2, 3, 4])},
    )
    sc.testing.assert_identical(plateaus[1].value, plateau1)


def test_find_plateaus_adjacent_plateaus_select_long():
    """
    Plateau numbers:
    data ^
         |     1 1 1
         | 0 0
         +------------>
                     t
    """
    da = sc.DataArray(
        sc.array(dims=['t'], values=[0, 0, 1, 1, 1]), coords={'t': sc.arange('t', 5)}
    )
    plateaus = find_plateaus(da, atol=sc.scalar(0.1), min_n_points=3)
    assert plateaus.sizes == {'plateau': 1}
    sc.testing.assert_identical(
        plateaus.coords['plateau'], sc.array(dims=['plateau'], values=[0], unit=None)
    )

    plateau0 = sc.DataArray(
        sc.array(dims=['t'], values=[1, 1, 1]),
        coords={'t': sc.array(dims=['t'], values=[2, 3, 4])},
    )
    sc.testing.assert_identical(plateaus[0].value, plateau0)


def test_find_plateaus_adjacent_plateaus_select_none():
    """
    Plateau numbers:
    data ^
         |     1 1 1
         | 0 0
         +------------>
                     t
    """
    da = sc.DataArray(
        sc.array(dims=['t'], values=[0, 0, 1, 1, 1]), coords={'t': sc.arange('t', 5)}
    )
    plateaus = find_plateaus(da, atol=sc.scalar(0.1), min_n_points=4)
    assert plateaus.sizes == {'plateau': 0}
    sc.testing.assert_identical(
        plateaus.coords['plateau'],
        sc.array(dims=['plateau'], values=[], dtype=da.dtype, unit=None),
    )


def test_find_plateaus_slow_slope():
    da = sc.DataArray(
        sc.concat(
            [
                sc.linspace('s', 0.0, 1.0, 100, endpoint=False),
                sc.full(value=1.0, sizes={'s': 10}),
            ],
            dim='s',
        ),
        coords={'s': sc.arange('s', 110)},
    )
    plateaus = find_plateaus(da, atol=sc.scalar(1e-6), min_n_points=2)
    assert plateaus.sizes == {'plateau': 1}
    plateau0 = sc.DataArray(
        sc.full(value=1.0, sizes={'s': 10}), coords={'s': sc.arange('s', 100, 110)}
    )
    sc.testing.assert_identical(plateaus[0].value, plateau0)


def test_find_plateaus_slow_slope_false_positive_caught():
    da = sc.DataArray(sc.arange('s', 0.0, 1.0, 0.01), coords={'s': sc.arange('s', 100)})
    with pytest.raises(RuntimeError, match='tolerance'):
        find_plateaus(da, atol=sc.scalar(0.02), min_n_points=2)


def test_find_plateaus_plateaus_negative():
    """
    Plateau numbers:
                     t
         +------------>
         |           x
         |     1 1 1
         | 0 0
    data v
    """
    da = sc.DataArray(
        sc.array(dims=['t'], values=[-5, -5, -3, -3, -3, -2]),
        coords={'t': sc.arange('t', 6)},
    )
    plateaus = find_plateaus(da, atol=sc.scalar(0.1), min_n_points=2)
    assert plateaus.sizes == {'plateau': 2}
    sc.testing.assert_identical(
        plateaus.coords['plateau'], sc.array(dims=['plateau'], values=[0, 1], unit=None)
    )

    plateau0 = sc.DataArray(
        sc.array(dims=['t'], values=[-5, -5]),
        coords={'t': sc.array(dims=['t'], values=[0, 1])},
    )
    sc.testing.assert_identical(plateaus[0].value, plateau0)
    plateau1 = sc.DataArray(
        sc.array(dims=['t'], values=[-3, -3, -3]),
        coords={'t': sc.array(dims=['t'], values=[2, 3, 4])},
    )
    sc.testing.assert_identical(plateaus[1].value, plateau1)


def test_collapse_plateaus():
    """
    Plateau numbers:
    data ^
         |       1 1 1
         |     x       x
         | 0 0
         +--------------->
                        t
    """
    da = sc.DataArray(
        sc.array(dims=['t'], values=[0, 0, 1, 2, 2, 2, 1]),
        coords={
            't': sc.array(dims=['t'], values=[12, 13, 14, 15, 16, 17, 18], unit='s')
        },
    )
    plateaus = find_plateaus(da, atol=sc.scalar(0.1, unit='Hz'), min_n_points=2)
    collapsed = collapse_plateaus(plateaus, coord='t')

    t = sc.array(dims=['plateau', 't'], values=[[12, 14], [15, 18]], unit='s')
    t = t.transpose(collapsed.coords['t'].dims)
    expected = sc.DataArray(
        sc.array(dims=['plateau'], values=[0.0, 2.0]),
        coords={
            'plateau': sc.array(dims=['plateau'], values=[0, 1], unit=None),
            't': t,
        },
    )
    sc.testing.assert_identical(collapsed, expected)


def test_filter_in_phase_integers_positive():
    da = sc.DataArray(sc.array(dims=['t'], values=[7, 8, 8, 8, 3, 2, 4]))
    filtered = filter_in_phase(da, reference=sc.scalar(4), rtol=sc.scalar(0.1))
    expected = sc.DataArray(sc.array(dims=['t'], values=[8, 8, 8, 2, 4]))
    sc.testing.assert_identical(filtered, expected)


def test_filter_in_phase_integers_negative():
    da = sc.DataArray(sc.array(dims=['t'], values=[-7, -8, -8, -8, -3, -2, -4]))
    filtered = filter_in_phase(da, reference=sc.scalar(-4), rtol=sc.scalar(0.1))
    expected = sc.DataArray(sc.array(dims=['t'], values=[-8, -8, -8, -2, -4]))
    sc.testing.assert_identical(filtered, expected)


def test_filter_in_phase_floats():
    da = sc.DataArray(sc.array(dims=['t'], values=[0.23, 0.6, 1.2, 3.3, 2.4]))
    filtered = filter_in_phase(da, reference=sc.scalar(1.2), rtol=sc.scalar(0.1))
    expected = sc.DataArray(sc.array(dims=['t'], values=[0.6, 1.2, 2.4]))
    sc.testing.assert_identical(filtered, expected)
