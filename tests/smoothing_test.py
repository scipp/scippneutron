# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
from scipy.special import ndtr
from scipy.stats import uniform

import scippneutron as scn
from scippneutron.smoothing import (
    _relative_kernel_weights,
    _smooth_relative_kernel_on_geomgrid,
    smooth_relative_gaussian,
    smooth_relative_kernel,
    smooth_relative_rectangle,
    smooth_relative_triangle,
)


def _normal_pdf(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def _quadratic(x):
    return 1.0 + 0.3 * x + 0.7 * x**2


def _exact_smoothed_quadratic(x, alpha, lower, upper):
    """Exactly smooth ``_quadratic`` with a finite-domain Gaussian."""
    x = np.asarray(x, dtype=float)
    sigma = alpha * x

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


def _geometric_cell_centers(lower, upper, size):
    log_step = np.log(upper / lower) / size
    return lower * np.exp((np.arange(size) + 0.5) * log_step)


def _variables(x, y, *, variances=None):
    return (
        sc.array(dims=['x'], values=np.asarray(x), unit='m'),
        sc.array(
            dims=['x'],
            values=np.asarray(y),
            variances=None if variances is None else np.asarray(variances),
            unit='counts',
        ),
    )


def _smooth_values(smooth, x, y, **kwargs):
    x, y = _variables(x, y)
    return smooth(x, y, **kwargs).values


def _quadratic_smoothing_error(*, size, alpha, tail=1e-9):
    lower = 0.1
    upper = 0.9
    x = _geometric_cell_centers(lower, upper, size)
    actual = _smooth_values(
        smooth_relative_gaussian,
        x,
        _quadratic(x),
        alpha=alpha,
        tail=tail,
    )
    expected = _exact_smoothed_quadratic(x, alpha, lower, upper)
    return actual - expected


def test_gaussian_smoothing_matches_exact_quadratic_on_geometric_grid():
    # The outer cell edges of this grid coincide with the finite integration
    # domain used by _quadratic_smoothing_error.
    error = _quadratic_smoothing_error(size=4000, alpha=0.1)

    assert np.max(np.abs(error)) < 5e-7
    assert np.sqrt(np.mean(error**2)) < 1e-7


def test_gaussian_smoothing_error_is_second_order_in_log_grid_spacing():
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


def test_gaussian_smoothing_error_scales_with_tail_until_grid_error_dominates():
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
    ("smooth", "relative_variance", "max_error"),
    [
        (smooth_relative_rectangle, 1.0 / 3.0, 3e-7),
        (smooth_relative_triangle, 1.0 / 6.0, 5e-8),
    ],
)
def test_compact_symmetric_kernel_matches_exact_interior_quadratic_moments(
    smooth, relative_variance, max_error
):
    lower = 0.1
    upper = 0.9
    size = 4000
    alpha = 0.1
    x = _geometric_cell_centers(lower, upper, size)

    actual = _smooth_values(smooth, x, _quadratic(x), alpha=alpha)
    expected = 1.0 + 0.3 * x + 0.7 * x**2 * (1.0 + alpha**2 * relative_variance)
    interior = (x * (1.0 - alpha) >= lower) & (x * (1.0 + alpha) <= upper)

    assert np.max(np.abs(actual[interior] - expected[interior])) < max_error


def test_smoothing_module_is_exposed_by_package():
    assert scn.smoothing.smooth_relative_gaussian is smooth_relative_gaussian


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

    actual = smooth_relative_gaussian(data, alpha=0.1)
    expected = smooth_relative_gaussian(x, data.data, alpha=0.1)

    assert isinstance(actual, sc.DataArray)
    assert sc.identical(actual.data, expected)
    assert sc.identical(actual.coords['x'], data.coords['x'])
    assert sc.identical(actual.coords['aux'], data.coords['aux'])
    assert sc.identical(actual.coords['scalar'], data.coords['scalar'])


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
        smooth_relative_gaussian(x, y, alpha=0.1)


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
        smooth_relative_gaussian(data, alpha=0.1)


def test_rejects_numpy_arrays():
    with pytest.raises(TypeError, match="DataArray or a pair of Variables"):
        smooth_relative_gaussian(np.arange(1.0, 4.0), np.ones(3))


def test_rejects_variables_with_different_dimensions():
    x = sc.arange('x', 1.0, 4.0)
    y = sc.ones(dims=['y'], shape=[3])

    with pytest.raises(sc.DimensionError, match="same dimension"):
        smooth_relative_gaussian(x, y)


def test_rejects_data_array_without_dimension_coordinate():
    data = sc.DataArray(sc.ones(dims=['x'], shape=[3]))

    with pytest.raises(sc.CoordError, match="dimension coordinate"):
        smooth_relative_gaussian(data)


def test_rejects_data_array_with_bin_edge_coordinate():
    data = sc.DataArray(
        sc.ones(dims=['x'], shape=[3]),
        coords={'x': sc.arange('x', 1.0, 5.0)},
    )

    with pytest.raises(sc.CoordError, match="bin edges"):
        smooth_relative_gaussian(data)


def test_rejects_data_array_with_masks():
    data = sc.DataArray(
        sc.ones(dims=['x'], shape=[3]),
        coords={'x': sc.arange('x', 1.0, 4.0)},
        masks={'bad': sc.array(dims=['x'], values=[False, True, False])},
    )

    with pytest.raises(ValueError, match="data with masks"):
        smooth_relative_gaussian(data)


def test_rejects_y_with_data_array():
    x, y = _variables([1.0, 2.0], [3.0, 4.0])
    data = sc.DataArray(y, coords={'x': x})

    with pytest.raises(TypeError, match="y must be omitted"):
        smooth_relative_gaussian(data, y)


def test_rejects_geometric_grid_larger_than_limit():
    x = np.array([1.0, 1.01, 2.0])
    y = np.ones_like(x)

    with pytest.raises(
        ValueError,
        match=r"geometric resampling would require too many points.*max_grid_points=70",
    ):
        _smooth_values(smooth_relative_kernel, x, y, max_grid_points=70)


def test_accepts_geometric_grid_equal_to_limit():
    x = np.array([1.0, 1.01, 2.0])
    y = np.ones_like(x)

    actual = _smooth_values(smooth_relative_kernel, x, y, max_grid_points=71)

    np.testing.assert_allclose(actual, y)


def test_geometric_input_does_not_gain_a_point_from_roundoff():
    size = 100
    x = _geometric_cell_centers(0.1, 0.9, size)
    y = _quadratic(x)

    actual = _smooth_values(smooth_relative_gaussian, x, y, max_grid_points=size)

    assert actual.shape == y.shape


@pytest.mark.parametrize(
    "smooth",
    [
        smooth_relative_gaussian,
        smooth_relative_rectangle,
        smooth_relative_triangle,
    ],
)
def test_convenience_functions_forward_max_grid_points(smooth):
    x = np.array([1.0, 1.01, 2.0])
    y = np.ones_like(x)

    with pytest.raises(ValueError, match="max_grid_points=2"):
        _smooth_values(smooth, x, y, max_grid_points=2)


def test_pathologically_close_coordinates_fail_before_allocation():
    x = np.array([1.0, np.nextafter(1.0, 2.0), 2.0])
    y = np.ones_like(x)

    with pytest.raises(ValueError, match="exceeding max_grid_points=1,000,000"):
        _smooth_values(smooth_relative_kernel, x, y)


def test_wide_coordinate_range_does_not_overflow_grid_construction():
    x = np.array([1e-300, 1.0, 1e300])
    y = np.ones_like(x)

    actual = _smooth_values(smooth_relative_kernel, x, y)

    np.testing.assert_allclose(actual, y)


def test_kernel_stencil_is_bounded_before_allocation():
    max_offset = 1000

    offsets, weights = _relative_kernel_weights(
        log_spacing=1e-6,
        alpha=1.0,
        kernel="gaussian",
        tail=1e-12,
        max_offset=max_offset,
    )

    assert offsets[0] >= -max_offset
    assert offsets[-1] <= max_offset
    assert offsets.size <= 2 * max_offset + 1
    assert offsets.size == weights.size


def test_asymmetric_kernel_convolution_matches_direct_weighted_sum():
    size = 20
    log_spacing = 0.03
    alpha = 0.2
    kernel = uniform(loc=0.5, scale=1.0)
    y = np.arange(size, dtype=float) ** 2

    offsets, weights = _relative_kernel_weights(
        log_spacing=log_spacing,
        alpha=alpha,
        kernel=kernel,
        tail=1e-12,
        max_offset=size - 1,
    )
    expected = np.empty_like(y)
    for i in range(size):
        valid = (0 <= i + offsets) & (i + offsets < size)
        denominator = np.sum(weights[valid])
        expected[i] = (
            np.dot(weights[valid], y[i + offsets[valid]]) / denominator
            if denominator > 0.0
            else np.nan
        )

    actual = _smooth_relative_kernel_on_geomgrid(
        y,
        log_spacing=log_spacing,
        alpha=alpha,
        kernel=kernel,
        tail=1e-12,
    )

    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_kernel_with_no_reachable_mass_returns_nan():
    actual = _smooth_relative_kernel_on_geomgrid(
        np.arange(5.0),
        log_spacing=0.1,
        alpha=0.1,
        kernel=uniform(loc=100.0, scale=1.0),
        tail=1e-12,
    )

    assert np.all(np.isnan(actual))


@pytest.mark.parametrize("alpha", [np.nan, np.inf, -np.inf, -1.0])
def test_rejects_invalid_alpha_before_noop_return(alpha):
    x, y = _variables([], [])
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        smooth_relative_kernel(x, y, alpha=alpha)


@pytest.mark.parametrize("tail", [np.nan, np.inf, -np.inf, 0.0, 1.0])
def test_rejects_invalid_tail_before_noop_return(tail):
    x, y = _variables([], [])
    with pytest.raises(ValueError, match="tail must be between 0 and 1"):
        smooth_relative_kernel(x, y, tail=tail)


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
        smooth_relative_kernel(x, y)


@pytest.mark.parametrize("max_grid_points", [True, 2.5])
def test_rejects_non_integer_max_grid_points(max_grid_points):
    x, y = _variables([], [])
    with pytest.raises(TypeError, match="max_grid_points must be an integer"):
        smooth_relative_kernel(x, y, max_grid_points=max_grid_points)


@pytest.mark.parametrize("max_grid_points", [-1, 0, 1])
def test_rejects_too_small_max_grid_points(max_grid_points):
    x, y = _variables([], [])
    with pytest.raises(ValueError, match="max_grid_points must be at least 2"):
        smooth_relative_kernel(x, y, max_grid_points=max_grid_points)
