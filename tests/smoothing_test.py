# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
from scipy.special import ndtr
from scipy.stats import uniform

import scippneutron as scn
from scippneutron.smoothing import smooth, smooth_relative


def _normal_pdf(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def _quadratic(x):
    return 1.0 + 0.3 * x + 0.7 * x**2


def _exact_smoothed_quadratic_with_sigma(x, sigma, lower, upper):
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    z_lower = (lower - x) / sigma
    z_upper = (upper - x) / sigma
    normalization = ndtr(z_upper) - ndtr(z_lower)

    mean_z = (_normal_pdf(z_lower) - _normal_pdf(z_upper)) / normalization
    mean_z_squared = (
        1.0
        + (z_lower * _normal_pdf(z_lower) - z_upper * _normal_pdf(z_upper))
        / normalization
    )

    mean_x = x + sigma * mean_z
    mean_x_squared = x**2 + 2.0 * x * sigma * mean_z + sigma**2 * mean_z_squared
    return 1.0 + 0.3 * mean_x + 0.7 * mean_x_squared


def _exact_smoothed_quadratic(x, alpha, lower, upper):
    """Exactly smooth ``_quadratic`` with a relative finite-domain Gaussian."""
    return _exact_smoothed_quadratic_with_sigma(x, alpha * np.asarray(x), lower, upper)


def _geometric_cell_centers(lower, upper, size):
    log_step = np.log(upper / lower) / size
    return lower * np.exp((np.arange(size) + 0.5) * log_step)


def _linear_cell_centers(lower, upper, size):
    step = (upper - lower) / size
    return lower + (np.arange(size) + 0.5) * step


def _variables(x, y):
    return (
        sc.array(dims=['x'], values=np.asarray(x), unit='m'),
        sc.array(
            dims=['x'],
            values=np.asarray(y),
            unit='counts',
        ),
    )


def _smooth_values(function, x, y, **kwargs):
    x, y = _variables(x, y)
    return function(x, y, **kwargs).values


def _quadratic_smoothing_error(*, size, alpha, tail=1e-9):
    lower = 0.1
    upper = 0.9
    x = _geometric_cell_centers(lower, upper, size)
    actual = _smooth_values(
        smooth_relative,
        x,
        _quadratic(x),
        scale=alpha,
        tail=tail,
    )
    expected = _exact_smoothed_quadratic(x, alpha, lower, upper)
    return actual - expected


def _fixed_width_quadratic_smoothing_error(*, size, width, tail=1e-9):
    lower = -0.4
    upper = 0.9
    x = _linear_cell_centers(lower, upper, size)
    x_var, y_var = _variables(x, _quadratic(x))
    actual = smooth(
        x_var,
        y_var,
        scale=sc.scalar(width, unit='m'),
        tail=tail,
    ).values
    expected = _exact_smoothed_quadratic_with_sigma(x, width, lower, upper)
    return actual - expected


def test_fixed_width_gaussian_matches_exact_quadratic_on_regular_grid():
    error = _fixed_width_quadratic_smoothing_error(size=4000, width=0.1)

    assert np.max(np.abs(error)) < 5e-7
    assert np.sqrt(np.mean(error**2)) < 1e-7


def test_fixed_width_gaussian_error_is_second_order_in_grid_spacing():
    sizes = np.array([500, 1000, 2000, 4000])
    errors = np.array(
        [
            np.max(np.abs(_fixed_width_quadratic_smoothing_error(size=size, width=0.1)))
            for size in sizes
        ]
    )

    observed_orders = np.log2(errors[:-1] / errors[1:])
    np.testing.assert_allclose(observed_orders, 2.0, atol=0.06)


def test_fixed_width_gaussian_error_scales_with_tail_until_grid_error_dominates():
    tails = np.array([1e-2, 1e-3, 1e-4, 1e-5])
    errors = np.array(
        [
            np.max(
                np.abs(
                    _fixed_width_quadratic_smoothing_error(
                        size=4000, width=0.1, tail=tail
                    )
                )
            )
            for tail in tails
        ]
    )

    reduction_per_decade = errors[:-1] / errors[1:]
    assert np.all((5.0 < reduction_per_decade) & (reduction_per_decade < 15.0))

    grid_limited_errors = np.array(
        [
            np.max(
                np.abs(
                    _fixed_width_quadratic_smoothing_error(
                        size=4000, width=0.1, tail=tail
                    )
                )
            )
            for tail in (1e-9, 1e-12)
        ]
    )
    np.testing.assert_allclose(
        grid_limited_errors[0], grid_limited_errors[1], rtol=0.05
    )


@pytest.mark.parametrize(
    ("kernel", "kernel_variance", "max_error"),
    [
        ("boxcar", 1.0 / 3.0, 3e-7),
        ("triangular", 1.0 / 6.0, 5e-8),
    ],
)
def test_fixed_width_compact_kernel_matches_interior_quadratic_moments(
    kernel, kernel_variance, max_error
):
    lower = -0.4
    upper = 0.9
    size = 4000
    width = 0.1
    x = _linear_cell_centers(lower, upper, size)
    x_var, y_var = _variables(x, _quadratic(x))

    actual = smooth(
        x_var,
        y_var,
        scale=sc.scalar(width, unit='m'),
        kernel=kernel,
    ).values
    expected = 1.0 + 0.3 * x + 0.7 * (x**2 + width**2 * kernel_variance)
    interior = (x - width >= lower) & (x + width <= upper)

    assert np.max(np.abs(actual[interior] - expected[interior])) < max_error


def test_fixed_width_smoothing_converts_scale_to_coordinate_unit():
    x = sc.linspace('x', -0.4, 0.9, 100, unit='m')
    y = sc.array(dims=['x'], values=_quadratic(x.values), unit='counts')

    in_meters = smooth(x, y, scale=sc.scalar(0.1, unit='m'))
    in_centimeters = smooth(x, y, scale=sc.scalar(10.0, unit='cm'))

    assert sc.identical(in_meters, in_centimeters)


def test_fixed_width_smoothing_accepts_distribution():
    x = sc.linspace('x', -0.4, 0.9, 100, unit='m')
    y = sc.array(dims=['x'], values=_quadratic(x.values), unit='counts')
    scale = sc.scalar(0.1, unit='m')

    actual = smooth(
        x,
        y,
        scale=scale,
        kernel=uniform(loc=-1.0, scale=2.0),
    )
    expected = smooth(x, y, scale=scale, kernel="boxcar")

    assert sc.identical(actual, expected)


def test_fixed_width_asymmetric_kernel_has_correct_direction():
    x = sc.arange('x', 0.0, 2.0, 0.1, unit='m')
    y = sc.array(dims=['x'], values=x.values, unit='counts')

    actual = smooth(
        x,
        y,
        scale=sc.scalar(0.2, unit='m'),
        kernel=uniform(loc=0.5, scale=1.0),
    )

    np.testing.assert_allclose(actual.values[3:-3], y.values[3:-3] + 0.2)


def test_nonfinite_value_only_affects_overlapping_kernel_windows():
    size = 10_000
    x = sc.arange('x', float(size))
    values = np.ones(size)
    values[size // 2] = np.nan
    y = sc.array(dims=['x'], values=values)

    actual = smooth(x, y, scale=sc.scalar(150.0))

    assert np.isfinite(actual.values[0])
    assert np.isnan(actual.values[size // 2])
    assert np.isfinite(actual.values[-1])


def test_fixed_width_smoothing_accepts_data_array():
    x = sc.linspace('x', -0.4, 0.9, 100, unit='m')
    data = sc.DataArray(
        sc.array(dims=['x'], values=_quadratic(x.values), unit='counts'),
        coords={'x': x, 'aux': sc.arange('x', 100)},
    )

    actual = smooth(data, scale=sc.scalar(0.1, unit='m'))
    expected = smooth(x, data.data, scale=sc.scalar(0.1, unit='m'))

    assert sc.identical(actual.data, expected)
    assert sc.identical(actual.coords['x'], data.coords['x'])
    assert sc.identical(actual.coords['aux'], data.coords['aux'])


def test_fixed_width_smoothing_resamples_nonuniform_grid():
    x_values = np.array([0.0, 0.25, 0.75, 1.0])
    y_values = _quadratic(x_values)
    scale = sc.scalar(0.2, unit='m')
    x, y = _variables(x_values, y_values)

    actual = smooth(x, y, scale=scale)

    xp_values = np.linspace(0.0, 1.0, 5)
    xp, yp = _variables(xp_values, np.interp(xp_values, x_values, y_values))
    smoothed_yp = smooth(xp, yp, scale=scale)
    expected = np.interp(x_values, xp_values, smoothed_yp.values)
    np.testing.assert_allclose(actual.values, expected)


def test_fixed_width_gaussian_matches_exact_quadratic_on_nonuniform_grid():
    lower = -0.4
    upper = 0.9
    size = 2000
    width = 0.1
    spacing = (upper - lower) / size
    x = _linear_cell_centers(lower, upper, size)
    x += 0.2 * spacing * np.sin(np.linspace(0.0, 8.0 * np.pi, size))
    x_var, y_var = _variables(x, _quadratic(x))

    actual = smooth(
        x_var,
        y_var,
        scale=sc.scalar(width, unit='m'),
        tail=1e-9,
    ).values

    intervals = int(np.ceil((x[-1] - x[0]) / np.min(np.diff(x))))
    resampled_spacing = (x[-1] - x[0]) / intervals
    expected = _exact_smoothed_quadratic_with_sigma(
        x,
        width,
        x[0] - 0.5 * resampled_spacing,
        x[-1] + 0.5 * resampled_spacing,
    )

    assert np.max(np.abs(actual - expected)) < 5e-7


def test_fixed_width_smoothing_rejects_resampled_grid_larger_than_limit():
    x, y = _variables([0.0, 0.125, 1.0], [1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match=r"uniform resampling would require too many points.*max_grid_points=4",
    ):
        smooth(x, y, scale=sc.scalar(0.1, unit='m'), max_grid_points=4)


def test_fixed_width_smoothing_accepts_resampled_grid_equal_to_limit():
    x, y = _variables([0.0, 0.25, 1.0], [1.0, 1.0, 1.0])

    actual = smooth(
        x,
        y,
        scale=sc.scalar(0.1, unit='m'),
        max_grid_points=5,
    )

    np.testing.assert_allclose(actual.values, y.values)


@pytest.mark.parametrize(
    ("function", "x", "kwargs"),
    [
        (
            smooth,
            np.linspace(0.0, 1.0, 11),
            {"scale": sc.scalar(0.1, unit='m')},
        ),
        (smooth_relative, np.geomspace(1.0, 2.0, 11), {"scale": 0.1}),
    ],
)
def test_input_grid_larger_than_limit_is_accepted(function, x, kwargs):
    y = np.ones_like(x)

    actual = _smooth_values(function, x, y, max_grid_points=10, **kwargs)

    np.testing.assert_allclose(actual, y)


@pytest.mark.parametrize(
    ("function", "x", "kwargs"),
    [
        (smooth, np.array([0.0, 0.26, 1.0]), {"scale": sc.scalar(0.1, unit='m')}),
        (smooth_relative, 2.0 ** np.array([0.0, 0.26, 1.0]), {"scale": 0.1}),
    ],
)
def test_modest_resampling_larger_than_limit_is_accepted(function, x, kwargs):
    y = np.ones_like(x)

    actual = _smooth_values(function, x, y, max_grid_points=4, **kwargs)

    np.testing.assert_allclose(actual, y)


@pytest.mark.parametrize(
    "coordinates",
    [
        [1.0, np.nextafter(1.0, 2.0), 2.0],
        # The spacing ratio overflows to infinity rather than merely being large.
        [0.0, 5e-324, 1.0],
        [1.0, np.nextafter(1.0, 2.0), 1e300],
    ],
)
def test_fixed_width_pathologically_close_coordinates_fail_before_allocation(
    coordinates,
):
    x, y = _variables(coordinates, [1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="exceeding max_grid_points=1,000,000"):
        smooth(x, y, scale=sc.scalar(0.1, unit='m'))


@pytest.mark.parametrize("scale", [np.nan, np.inf, -np.inf, -1.0])
def test_fixed_width_smoothing_rejects_invalid_scale(scale):
    x, y = _variables([], [])

    with pytest.raises(ValueError, match="scale must be non-negative"):
        smooth(x, y, scale=sc.scalar(scale, unit='m'))


def test_fixed_width_smoothing_rejects_non_scalar_scale():
    x, y = _variables([0.0, 1.0], [1.0, 2.0])

    with pytest.raises(sc.DimensionError, match="scale must be a scalar"):
        smooth(x, y, scale=sc.array(dims=['scale'], values=[0.1], unit='m'))


def test_fixed_width_smoothing_rejects_non_variable_scale():
    x, y = _variables([0.0, 1.0], [1.0, 2.0])

    with pytest.raises(TypeError, match=r"scale must be a scipp\.Variable"):
        smooth(x, y, scale=0.1)  # type: ignore[arg-type]


def test_fixed_width_smoothing_rejects_incompatible_scale_unit():
    x, y = _variables([0.0, 1.0], [1.0, 2.0])

    with pytest.raises(sc.UnitError):
        smooth(x, y, scale=sc.scalar(0.1, unit='s'))


def test_fixed_width_smoothing_rejects_scale_with_variance():
    x, y = _variables([0.0, 1.0], [1.0, 2.0])

    with pytest.raises(sc.VariancesError, match="scales with variances"):
        smooth(
            x,
            y,
            scale=sc.scalar(0.1, variance=0.01, unit='m'),
        )


def test_relative_width_gaussian_matches_exact_quadratic_on_geometric_grid():
    # The outer cell edges of this grid coincide with the finite integration
    # domain used by _quadratic_smoothing_error.
    error = _quadratic_smoothing_error(size=4000, alpha=0.1)

    assert np.max(np.abs(error)) < 5e-7
    assert np.sqrt(np.mean(error**2)) < 1e-7


def test_relative_width_gaussian_error_is_second_order_in_log_grid_spacing():
    sizes = np.array([500, 1000, 2000, 4000])
    errors = np.array(
        [
            np.max(np.abs(_quadratic_smoothing_error(size=size, alpha=0.1)))
            for size in sizes
        ]
    )

    # Halving the log-grid spacing should reduce midpoint quadrature error by
    # four, corresponding to second-order convergence.
    observed_orders = np.log2(errors[:-1] / errors[1:])
    np.testing.assert_allclose(observed_orders, 2.0, atol=0.06)


def test_gaussian_smoothing_error_scales_with_inverse_kernel_width():
    size = 4000
    alphas = np.array([0.025, 0.05, 0.1])
    errors = np.array(
        [
            np.max(np.abs(_quadratic_smoothing_error(size=size, alpha=alpha)))
            for alpha in alphas
        ]
    )

    # For a well-resolved narrow Gaussian, the largest error is at a truncated
    # boundary and scales as h**2 / alpha.
    log_step = np.log(0.9 / 0.1) / size
    scaled_errors = errors * alphas / log_step**2
    assert np.all(np.diff(errors) < 0.0)
    np.testing.assert_allclose(
        scaled_errors,
        np.mean(scaled_errors),
        rtol=0.08,
    )


def test_relative_width_gaussian_error_scales_with_tail_until_grid_error_dominates():
    tails = np.array([1e-2, 1e-3, 1e-4, 1e-5])
    truncation_errors = np.array(
        [
            np.max(np.abs(_quadratic_smoothing_error(size=4000, alpha=0.1, tail=tail)))
            for tail in tails
        ]
    )

    # Gaussian moments in the omitted tails add logarithmic factors, so each
    # decade should improve the result by approximately, but not exactly, ten.
    reduction_per_decade = truncation_errors[:-1] / truncation_errors[1:]
    assert np.all((5.0 < reduction_per_decade) & (reduction_per_decade < 15.0))

    # Once tail truncation is negligible, reducing it further cannot improve
    # the fixed-grid quadrature error.
    grid_limited_errors = np.array(
        [
            np.max(np.abs(_quadratic_smoothing_error(size=4000, alpha=0.1, tail=tail)))
            for tail in (1e-9, 1e-12)
        ]
    )
    np.testing.assert_allclose(
        grid_limited_errors[0], grid_limited_errors[1], rtol=0.05
    )


@pytest.mark.parametrize(
    ("kernel", "relative_variance", "max_error"),
    [
        ("boxcar", 1.0 / 3.0, 3e-7),
        ("triangular", 1.0 / 6.0, 5e-8),
    ],
)
def test_relative_width_compact_kernel_matches_interior_quadratic_moments(
    kernel, relative_variance, max_error
):
    lower = 0.1
    upper = 0.9
    size = 4000
    alpha = 0.1
    x = _geometric_cell_centers(lower, upper, size)

    actual = _smooth_values(
        smooth_relative,
        x,
        _quadratic(x),
        scale=alpha,
        kernel=kernel,
    )
    expected = 1.0 + 0.3 * x + 0.7 * x**2 * (1.0 + alpha**2 * relative_variance)
    interior = (x * (1.0 - alpha) >= lower) & (x * (1.0 + alpha) <= upper)

    assert np.max(np.abs(actual[interior] - expected[interior])) < max_error


def test_smoothing_module_is_exposed_by_package():
    assert scn.smoothing.smooth_relative is smooth_relative


def test_accepts_data_array_and_preserves_metadata():
    x = sc.geomspace('x', 0.1, 0.9, 100, unit='m')
    data = sc.DataArray(
        sc.array(dims=['x'], values=_quadratic(x.values), unit='counts'),
        coords={
            'x': x,
            'aux': sc.arange('x', 100, unit='s'),
            'scalar': sc.scalar(1.2, unit='K'),
        },
    )
    original = data.copy()

    actual = smooth_relative(data, scale=0.1)
    expected = smooth_relative(x, data.data, scale=0.1)

    assert sc.identical(data, original)
    assert isinstance(actual, sc.DataArray)
    assert sc.identical(actual.data, expected)
    assert sc.identical(actual.coords['x'], data.coords['x'])
    assert sc.identical(actual.coords['aux'], data.coords['aux'])
    assert sc.identical(actual.coords['scalar'], data.coords['scalar'])

    actual.values[0] = -1.0
    assert sc.identical(data, original)


def test_relative_smoothing_accepts_dimensionless_variable_scale():
    x, y = _variables([1.0, 2.0, 3.0], [1.0, 4.0, 9.0])

    actual = smooth_relative(x, y, scale=sc.scalar(0.1))
    expected = smooth_relative(x, y, scale=0.1)

    assert sc.identical(actual, expected)


def test_relative_smoothing_rejects_non_scalar_scale():
    x, y = _variables([1.0, 2.0], [1.0, 2.0])

    with pytest.raises(sc.DimensionError, match="scale must be a scalar"):
        smooth_relative(x, y, scale=sc.array(dims=['scale'], values=[0.1]))


def test_relative_smoothing_rejects_scale_with_unit():
    x, y = _variables([1.0, 2.0], [1.0, 2.0])

    with pytest.raises(sc.UnitError):
        smooth_relative(x, y, scale=sc.scalar(0.1, unit='m'))


def test_relative_smoothing_rejects_scale_with_variance():
    x, y = _variables([1.0, 2.0], [1.0, 2.0])

    with pytest.raises(sc.VariancesError, match="scales with variances"):
        smooth_relative(x, y, scale=sc.scalar(0.1, variance=0.01))


def test_rejects_variable_with_variances():
    x = sc.geomspace('x', 0.1, 0.9, 100, unit='m')
    values = _quadratic(x.values)
    y = sc.array(
        dims=['x'],
        values=values,
        variances=2.0 + x.values,
        unit='counts',
    )

    with pytest.raises(sc.VariancesError, match="signals with variances"):
        smooth_relative(x, y, scale=0.1)


def test_rejects_data_array_with_variances():
    x = sc.geomspace('x', 0.1, 0.9, 100, unit='m')
    data = sc.DataArray(
        sc.array(
            dims=['x'],
            values=_quadratic(x.values),
            variances=2.0 + x.values,
            unit='counts',
        ),
        coords={'x': x},
    )

    with pytest.raises(sc.VariancesError, match="signals with variances"):
        smooth_relative(data, scale=0.1)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (np.arange(1.0, 4.0), sc.ones(dims=['x'], shape=[3])),
        (sc.arange('x', 1.0, 4.0), np.ones(3)),
    ],
)
def test_rejects_non_scipp_pair_member(x, y):
    with pytest.raises(TypeError, match="DataArray or a pair of Variables"):
        smooth_relative(x, y, scale=0.1)


def test_rejects_variables_with_different_dimensions():
    x = sc.arange('x', 1.0, 4.0)
    y = sc.ones(dims=['y'], shape=[3])

    with pytest.raises(sc.DimensionError, match="same dimension"):
        smooth_relative(x, y, scale=0.1)


def test_rejects_data_array_without_dimension_coordinate():
    data = sc.DataArray(sc.ones(dims=['x'], shape=[3]))

    with pytest.raises(sc.CoordError, match="dimension coordinate"):
        smooth_relative(data, scale=0.1)


def test_rejects_data_array_with_bin_edge_coordinate():
    data = sc.DataArray(
        sc.ones(dims=['x'], shape=[3]),
        coords={'x': sc.arange('x', 1.0, 5.0)},
    )

    with pytest.raises(sc.CoordError, match="bin edges"):
        smooth_relative(data, scale=0.1)


def test_rejects_data_array_with_masks():
    data = sc.DataArray(
        sc.ones(dims=['x'], shape=[3]),
        coords={'x': sc.arange('x', 1.0, 4.0)},
        masks={'bad': sc.array(dims=['x'], values=[False, True, False])},
    )

    with pytest.raises(ValueError, match="data with masks"):
        smooth_relative(data, scale=0.1)


def test_rejects_y_with_data_array():
    x, y = _variables([1.0, 2.0], [3.0, 4.0])
    data = sc.DataArray(y, coords={'x': x})

    with pytest.raises(TypeError, match="y must be omitted"):
        smooth_relative(data, y, scale=0.1)


def test_rejects_geometric_grid_larger_than_limit():
    x = np.array([1.0, 1.01, 2.0])
    y = np.ones_like(x)

    with pytest.raises(
        ValueError,
        match=r"geometric resampling would require too many points.*max_grid_points=70",
    ):
        _smooth_values(smooth_relative, x, y, scale=0.1, max_grid_points=70)


def test_accepts_geometric_grid_equal_to_limit():
    x = np.array([1.0, 1.01, 2.0])
    y = np.ones_like(x)

    actual = _smooth_values(smooth_relative, x, y, scale=0.1, max_grid_points=71)

    np.testing.assert_allclose(actual, y)


def test_geometric_input_does_not_gain_a_point_from_roundoff():
    size = 100
    x = _geometric_cell_centers(0.1, 0.9, size)
    y = _quadratic(x)

    actual = _smooth_values(
        smooth_relative,
        x,
        y,
        scale=0.1,
        max_grid_points=size,
    )

    assert actual.shape == y.shape


@pytest.mark.filterwarnings("ignore:divide by zero encountered in scalar divide")
@pytest.mark.parametrize("x0", [1.0, 3.0, 10.0, 0.1])
def test_pathologically_close_coordinates_fail_before_allocation(x0):
    x = np.array([x0, np.nextafter(x0, 2.0 * x0), 2.0 * x0])
    y = np.ones_like(x)

    with pytest.raises(ValueError, match="exceeding max_grid_points=1,000,000"):
        _smooth_values(smooth_relative, x, y, scale=0.1)


def test_wide_coordinate_range_does_not_overflow_grid_construction():
    x = np.array([1e-300, 1.0, 1e300])
    y = np.ones_like(x)

    actual = _smooth_values(smooth_relative, x, y, scale=0.1)

    np.testing.assert_allclose(actual, y)


def test_kernel_stencil_is_bounded_before_allocation():
    # A gaussian truncated at tail=1e-12 spans roughly 14 sigma, which is about
    # 1e7 offsets on this grid. The stencil must be clamped to offsets that can
    # reach the input rather than allocated at its nominal width.
    size = 21
    x = np.geomspace(1.0, np.exp((size - 1) * 1e-6), size)
    y = np.arange(size, dtype=float)

    actual = _smooth_values(smooth_relative, x, y, scale=1.0)

    # The kernel is flat to within 1e-5 over the whole input, so every point
    # averages the entire array.
    np.testing.assert_allclose(actual, np.full(size, y.mean()), rtol=1e-4)


def test_asymmetric_kernel_convolution_matches_direct_weighted_sum():
    size = 20
    log_spacing = 0.03
    alpha = 0.2
    kernel = uniform(loc=0.5, scale=1.0)
    y = np.arange(size, dtype=float) ** 2
    x = np.geomspace(1.0, np.exp((size - 1) * log_spacing), size)

    offsets = np.arange(-size + 1, size)
    lower = np.expm1((offsets - 0.5) * log_spacing) / alpha
    upper = np.expm1((offsets + 0.5) * log_spacing) / alpha
    weights = kernel.cdf(upper) - kernel.cdf(lower)
    expected = np.empty_like(y)
    for i in range(size):
        valid = (0 <= i + offsets) & (i + offsets < size)
        denominator = np.sum(weights[valid])
        expected[i] = (
            np.dot(weights[valid], y[i + offsets[valid]]) / denominator
            if denominator > 0.0
            else np.nan
        )

    actual = _smooth_values(
        smooth_relative,
        x,
        y,
        scale=alpha,
        kernel=kernel,
        tail=1e-12,
    )

    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_kernel_with_no_reachable_mass_returns_nan():
    y = np.arange(5.0)
    x = np.geomspace(1.0, np.exp((y.size - 1) * 0.1), y.size)
    actual = _smooth_values(
        smooth_relative,
        x,
        y,
        scale=0.1,
        kernel=uniform(loc=100.0, scale=1.0),
        tail=1e-12,
    )

    assert np.all(np.isnan(actual))


@pytest.mark.parametrize("scale", [np.nan, np.inf, -np.inf, -1.0])
def test_rejects_invalid_relative_scale_before_noop_return(scale):
    x, y = _variables([], [])
    with pytest.raises(ValueError, match="scale must be non-negative"):
        smooth_relative(x, y, scale=scale)


@pytest.mark.parametrize("tail", [np.nan, np.inf, -np.inf, 0.0, 1.0])
def test_rejects_invalid_tail_before_noop_return(tail):
    x, y = _variables([], [])
    with pytest.raises(ValueError, match="tail must be between 0 and 1"):
        smooth_relative(x, y, scale=0.1, tail=tail)


@pytest.mark.parametrize(
    ("x", "message"),
    [
        ([-1.0], "x must be positive"),
        ([np.nan], "x must contain only finite values"),
        ([np.inf], "x must contain only finite values"),
    ],
)
def test_rejects_invalid_single_coordinate_before_noop_return(x, message):
    x, y = _variables(x, [1.0])
    with pytest.raises(ValueError, match=message):
        smooth_relative(x, y, scale=0.1)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (smooth_relative, {"scale": 0.1}),
        (smooth, {"scale": sc.scalar(0.1, unit='m')}),
    ],
)
@pytest.mark.parametrize("max_grid_points", [True, 2.5])
def test_rejects_non_integer_max_grid_points(function, kwargs, max_grid_points):
    x, y = _variables([], [])
    with pytest.raises(TypeError, match="max_grid_points must be an integer"):
        function(x, y, max_grid_points=max_grid_points, **kwargs)


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (smooth_relative, {"scale": 0.1}),
        (smooth, {"scale": sc.scalar(0.1, unit='m')}),
    ],
)
@pytest.mark.parametrize("max_grid_points", [-1, 0, 1])
def test_rejects_too_small_max_grid_points(function, kwargs, max_grid_points):
    x, y = _variables([], [])
    with pytest.raises(ValueError, match="max_grid_points must be at least 2"):
        function(x, y, max_grid_points=max_grid_points, **kwargs)
