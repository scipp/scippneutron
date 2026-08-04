# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import scipp as sc
from numpy.typing import ArrayLike, NDArray
from scipy.signal import convolve
from scipy.stats import norm, triang, uniform

__all__ = [
    "smooth_gaussian",
    "smooth_kernel",
    "smooth_rectangle",
    "smooth_relative_gaussian",
    "smooth_relative_kernel",
    "smooth_relative_rectangle",
    "smooth_relative_triangle",
    "smooth_triangle",
]


class _Distribution(Protocol):
    def cdf(self, x: ArrayLike) -> Any: ...

    def ppf(self, probability: ArrayLike) -> Any: ...

    def support(self) -> tuple[Any, Any]: ...


_Kernel: TypeAlias = str | _Distribution
_FloatArray: TypeAlias = NDArray[np.float64]
_IntArray: TypeAlias = NDArray[np.int64]
_ScippArray: TypeAlias = sc.DataArray | sc.Variable

_BUILTIN_KERNELS: dict[str, _Distribution] = {
    # Standard Gaussian.
    "gaussian": norm(),
    "normal": norm(),
    # Centered rectangle on [-1, 1].
    "rectangle": uniform(loc=-1.0, scale=2.0),
    "rect": uniform(loc=-1.0, scale=2.0),
    "box": uniform(loc=-1.0, scale=2.0),
    "uniform": uniform(loc=-1.0, scale=2.0),
    # Centered triangle on [-1, 1], peak at 0.
    # Users can pass triang(c=...) themselves for asymmetric triangles.
    "triangle": triang(c=0.5, loc=-1.0, scale=2.0),
    "triangular": triang(c=0.5, loc=-1.0, scale=2.0),
}


def _as_kernel_distribution(kernel: _Kernel) -> _Distribution:
    if isinstance(kernel, str):
        try:
            return _BUILTIN_KERNELS[kernel.lower()]
        except KeyError:
            valid = ", ".join(sorted(_BUILTIN_KERNELS))
            raise ValueError(
                f"unknown kernel {kernel!r}; expected one of: {valid}"
            ) from None

    required = ("cdf", "ppf", "support")
    missing = [name for name in required if not hasattr(kernel, name)]
    if missing:
        raise TypeError(
            "kernel must be a scipy.stats distribution-like object with methods "
            f"{', '.join(required)}; missing {', '.join(missing)}"
        )

    return kernel


def _kernel_support(dist: _Distribution) -> tuple[float, float]:
    try:
        support_min, support_max = dist.support()
    except TypeError as e:
        raise TypeError(
            "kernel must be a fully specified distribution. For distributions "
            "with shape parameters, pass a frozen distribution such as "
            "triang(c=0.5), not triang."
        ) from e
    return float(support_min), float(support_max)


def _trim_kernel_weights(
    offsets: _IntArray, weights: _FloatArray
) -> tuple[_IntArray, _FloatArray]:
    nonzero = np.flatnonzero(weights > 0.0)
    if nonzero.size == 0:
        return offsets, weights

    # Trim zero-only ends, but preserve offset zero for convolution alignment.
    zero = -offsets[0]
    first = min(nonzero[0], zero)
    last = max(nonzero[-1], zero) + 1
    weights = weights[first:last]
    return offsets[first:last], weights / weights.sum()


def _relative_kernel_weights(
    log_spacing: float,
    alpha: float,
    kernel: _Kernel,
    tail: float,
    max_offset: int,
) -> tuple[_IntArray, _FloatArray]:
    """
    Weights for smoothing on a geometric grid q_i = q0 * exp(i * log_spacing).

    The kernel distribution describes the relative displacement Z:

        q' = q * (1 + alpha * Z)

    Equivalently,

        K(q, q') = 1 / (alpha * q) * f((q' - q) / (alpha * q))

    where f is the PDF of the supplied distribution.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive")
    if not np.isfinite(log_spacing) or log_spacing <= 0:
        raise ValueError("log_spacing must be positive")
    if not np.isfinite(tail) or not (0.0 < tail < 1.0):
        raise ValueError("tail must be between 0 and 1")
    if max_offset < 0:
        raise ValueError("max_offset must be non-negative")

    dist = _as_kernel_distribution(kernel)

    # Positive physical domain:
    #
    #   q' > 0
    #   q * (1 + alpha * z) > 0
    #   z > -1 / alpha
    z_domain_min = -1.0 / alpha

    p_domain_min = float(dist.cdf(z_domain_min))
    norm_mass = 1.0 - p_domain_min

    if not np.isfinite(norm_mass) or norm_mass <= 0.0:
        raise ValueError("kernel has no positive-domain mass for this alpha")

    support_min, support_max = _kernel_support(dist)

    # Exact z-support after clipping to q' > 0.
    z_left_exact = max(z_domain_min, support_min)
    z_right_exact = support_max

    # If the clipped support maps to finite log-space, use it exactly.
    # If it touches q' = 0, the log lower bound is -inf, so use a tail cutoff.
    has_finite_log_support = (
        np.isfinite(z_left_exact)
        and np.isfinite(z_right_exact)
        and (1.0 + alpha * z_left_exact > 0.0)
    )

    if has_finite_log_support:
        u_left = np.log1p(alpha * z_left_exact)
        u_right = np.log1p(alpha * z_right_exact)
    else:
        probabilities = (
            p_domain_min + np.array([0.5 * tail, 1.0 - 0.5 * tail]) * norm_mass
        )
        z_left, z_right = (float(value) for value in dist.ppf(probabilities))

        u_left = np.log1p(alpha * z_left)
        u_right = np.log1p(alpha * z_right)

    if not np.isfinite(u_right):
        raise ValueError("right kernel bound is not finite; increase tail")

    # Cells are centered at m*h and span [(m-1/2)h, (m+1/2)h].
    # Include offset zero even for one-sided kernels and clamp the stencil to
    # offsets that can contribute to the finite input.
    m_min = int(np.clip(np.floor(u_left / log_spacing + 0.5), -max_offset, 0))
    m_max = int(np.clip(np.ceil(u_right / log_spacing - 0.5), 0, max_offset))

    m = np.arange(m_min, m_max + 1, dtype=np.int64)

    L = (m - 0.5) * log_spacing
    U = (m + 0.5) * log_spacing

    # z = (q' - q) / (alpha q)
    #   = (exp(u) - 1) / alpha
    zL = np.expm1(L) / alpha
    zU = np.expm1(U) / alpha

    # Exact cell-integrated weights in log-space.
    w = (dist.cdf(zU) - dist.cdf(zL)) / norm_mass
    w = np.maximum(w, 0.0)

    return _trim_kernel_weights(m, w)


def _translation_invariant_kernel_weights(
    spacing: float,
    width: float,
    kernel: _Kernel,
    tail: float,
    max_offset: int,
) -> tuple[_IntArray, _FloatArray]:
    if not np.isfinite(width) or width <= 0:
        raise ValueError("width must be positive")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive")
    if not np.isfinite(tail) or not (0.0 < tail < 1.0):
        raise ValueError("tail must be between 0 and 1")
    if max_offset < 0:
        raise ValueError("max_offset must be non-negative")

    dist = _as_kernel_distribution(kernel)
    z_left, z_right = _kernel_support(dist)
    if not np.all(np.isfinite([z_left, z_right])):
        z_left, z_right = (
            float(value) for value in dist.ppf([0.5 * tail, 1.0 - 0.5 * tail])
        )

    bounds = width * np.array([z_left, z_right])
    if not np.all(np.isfinite(bounds)):
        raise ValueError("kernel bounds are not finite; increase tail")

    m_min = int(np.clip(np.floor(bounds[0] / spacing + 0.5), -max_offset, 0))
    m_max = int(np.clip(np.ceil(bounds[1] / spacing - 0.5), 0, max_offset))
    m = np.arange(m_min, m_max + 1, dtype=np.int64)

    lower = (m - 0.5) * spacing / width
    upper = (m + 0.5) * spacing / width
    weights = np.maximum(dist.cdf(upper) - dist.cdf(lower), 0.0)
    return _trim_kernel_weights(m, weights)


def _valid_weight_sums(n: int, m: _IntArray, w: _FloatArray) -> _FloatArray:
    """
    Boundary normalization.

    Equivalent to

        np.convolve(np.ones(n), w[::-1], mode="full")[start:start+n]

    but O(n), not another full convolution.
    """
    i = np.arange(n, dtype=np.int64)

    m_min = int(m[0])
    m_max = int(m[-1])

    lower = np.maximum(m_min, -i)
    upper = np.minimum(m_max, n - 1 - i)

    cumsum = np.empty(w.size + 1, dtype=float)
    cumsum[0] = 0.0
    np.cumsum(w, out=cumsum[1:])

    return cast(
        _FloatArray,
        cumsum[upper - m_min + 1] - cumsum[lower - m_min],
    )


def _smooth_with_weights(
    y: _FloatArray, offsets: _IntArray, weights: _FloatArray
) -> _FloatArray:
    # Desired operation:
    #
    #   out[i] = sum_m weights[m] * y[i + m]
    #
    # scipy convolution reverses the second argument, hence weights[::-1].
    # FFT convolution would spread a single NaN or infinity over the entire
    # output. SciPy recommends the direct method for non-finite inputs.
    method = "auto" if np.all(np.isfinite(y)) else "direct"
    full = cast(_FloatArray, convolve(y, weights[::-1], mode="full", method=method))
    start = int(offsets[-1])
    numerator = full[start : start + y.size]
    denominator = _valid_weight_sums(y.size, offsets, weights)

    out = np.full_like(numerator, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0.0)
    return out


def _smooth_relative_kernel_on_geomgrid(
    y: ArrayLike,
    log_spacing: float,
    alpha: float,
    kernel: _Kernel,
    tail: float,
) -> _FloatArray:
    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not np.isfinite(log_spacing) or log_spacing <= 0:
        raise ValueError("log_spacing must be positive")
    if not np.isfinite(tail) or not (0.0 < tail < 1.0):
        raise ValueError("tail must be between 0 and 1")
    if y.size == 0 or alpha == 0:
        return y.copy()

    offsets, weights = _relative_kernel_weights(
        log_spacing=log_spacing,
        alpha=alpha,
        kernel=kernel,
        tail=tail,
        max_offset=y.size - 1,
    )
    return _smooth_with_weights(y, offsets, weights)


def _smooth_relative_kernel(
    x: ArrayLike,
    y: ArrayLike,
    alpha: float = 1.0,
    kernel: _Kernel = "gaussian",
    tail: float = 1e-12,
    max_grid_points: int = 1_000_000,
) -> _FloatArray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not np.isfinite(tail) or not (0.0 < tail < 1.0):
        raise ValueError("tail must be between 0 and 1")
    if isinstance(max_grid_points, bool | np.bool_) or not isinstance(
        max_grid_points, int | np.integer
    ):
        raise TypeError("max_grid_points must be an integer")
    if max_grid_points < 2:
        raise ValueError("max_grid_points must be at least 2")
    if np.any(~np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if np.any(x <= 0):
        raise ValueError("x must be positive")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    if x.size == 0 or alpha == 0 or x.size == 1:
        return y.copy()

    logx = np.log(x)
    dlog = np.diff(logx)
    log_range = float(logx[-1] - logx[0])

    # Preserve an existing geometric grid. Otherwise choose a geometric grid
    # at least as dense as the smallest input spacing in log-space.
    if np.allclose(dlog, dlog[0], rtol=1e-7, atol=0.0):
        k = x.size
    else:
        k = int(np.ceil(log_range / np.min(dlog))) + 1

    if k > max_grid_points:
        raise ValueError(
            "geometric resampling would require too many points, exceeding "
            f"max_grid_points={max_grid_points:,}. Increase max_grid_points to "
            "allow a larger grid."
        )

    xp = np.geomspace(x[0], x[-1], k)
    yg = np.interp(xp, x, y)
    zg = _smooth_relative_kernel_on_geomgrid(
        yg,
        alpha=alpha,
        log_spacing=log_range / (k - 1),
        kernel=kernel,
        tail=tail,
    )
    return cast(_FloatArray, np.interp(x, xp, zg))


def _smooth_kernel_values(
    x: ArrayLike,
    y: ArrayLike,
    width: float,
    kernel: _Kernel = "gaussian",
    tail: float = 1e-12,
) -> _FloatArray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if not np.isfinite(width) or width < 0:
        raise ValueError("width must be non-negative")
    if not np.isfinite(tail) or not (0.0 < tail < 1.0):
        raise ValueError("tail must be between 0 and 1")
    if np.any(~np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    if x.size == 0 or width == 0 or x.size == 1:
        return y.copy()

    spacing = np.diff(x)
    if not np.allclose(spacing, spacing[0], rtol=1e-7, atol=0.0):
        raise ValueError("x must be regularly spaced")

    offsets, weights = _translation_invariant_kernel_weights(
        spacing=float(spacing[0]),
        width=width,
        kernel=kernel,
        tail=tail,
        max_offset=y.size - 1,
    )
    return _smooth_with_weights(y, offsets, weights)


def _scipp_input(
    x: _ScippArray, y: sc.Variable | None
) -> tuple[sc.Variable, sc.Variable, sc.DataArray | None]:
    template: sc.DataArray | None = None
    if isinstance(x, sc.DataArray):
        if y is not None:
            raise TypeError("y must be omitted when x is a DataArray")
        if x.ndim != 1:
            raise sc.DimensionError("data must be one-dimensional")
        if x.dim not in x.coords:
            raise sc.CoordError("data must have a dimension coordinate")
        if x.coords.is_edges(x.dim):
            raise sc.CoordError("the dimension coordinate must not contain bin edges")
        if x.masks:
            raise ValueError("smoothing data with masks is not supported")
        template = x
        y = x.data
        x = x.coords[x.dim]
    elif not isinstance(y, sc.Variable):
        raise TypeError("expected a DataArray or a pair of Variables")

    if x.ndim != 1 or y.ndim != 1:
        raise sc.DimensionError("x and y must be one-dimensional")
    if x.dims != y.dims:
        raise sc.DimensionError("x and y must have the same dimension")
    if x.is_binned or y.is_binned:
        raise sc.DTypeError("x and y must not be binned")
    if y.variances is not None:
        raise sc.VariancesError(
            "Smoothing signals with variances is not supported because it would "
            "introduce correlations between data points."
        )
    return x, y, template


def _scipp_output(
    template: sc.DataArray | None, y: sc.Variable, values: _FloatArray
) -> _ScippArray:
    data = sc.array(dims=y.dims, values=values, unit=y.unit)
    if template is None:
        return data

    # The new container shares unchanged coordinates, but its data is independent.
    out = template.copy(deep=False)
    out.data = data
    return out


def _width_in_coordinate_unit(width: sc.Variable, x: sc.Variable) -> float:
    if not isinstance(width, sc.Variable):
        raise TypeError("width must be a scipp.Variable")
    if width.ndim != 0:
        raise sc.DimensionError("width must be a scalar")
    if width.variances is not None:
        raise sc.VariancesError("kernel widths with variances are not supported")
    return float(width.to(unit=x.unit).value)


def smooth_kernel(
    x: _ScippArray,
    y: sc.Variable | None = None,
    *,
    width: sc.Variable,
    kernel: _Kernel = "gaussian",
    tail: float = 1e-12,
) -> _ScippArray:
    """Smooth regularly sampled data with a translation-invariant kernel.

    The kernel describes a distribution of displacements ``Z``, with displaced
    coordinates given by ``x' = x + width * Z``. At the boundaries, the kernel
    is renormalized over the available finite input domain.

    Parameters
    ----------
    x:
        One-dimensional data to smooth, or strictly increasing, regularly
        spaced sample coordinates. A data array must have a dimension
        coordinate.
    y:
        Values to smooth when ``x`` contains the sample coordinates. Must be a
        one-dimensional variable with the same dimension as ``x``. Must be
        omitted when ``x`` is a data array.
    width:
        Scale factor for the displacement distribution. Must be a scalar with a
        unit compatible with the coordinate. Set to zero to return a copy of the
        input without smoothing.
    kernel:
        Kernel distribution. Supported names are ``'gaussian'``, ``'rectangle'``,
        and ``'triangle'``, including their aliases. Alternatively, provide a
        fully specified distribution with ``cdf``, ``ppf``, and ``support``
        methods.
    tail:
        Total probability omitted when truncating a kernel with unbounded
        support.

    Returns
    -------
    :
        Smoothed data of the same type as the input. Coordinates and units are
        preserved.

    Raises
    ------
    ValueError
        If the coordinate is not regularly spaced or the inputs otherwise have
        invalid values, if a data array has masks, or if a string does not
        identify a supported kernel.
    scipp.DimensionError
        If the inputs are not one-dimensional, a pair of variables does not
        have matching dimensions, or ``width`` is not scalar.
    scipp.CoordError
        If a data array has no dimension coordinate or has a bin-edge
        coordinate.
    scipp.UnitError
        If the unit of ``width`` is incompatible with the coordinate unit.
    scipp.VariancesError
        If the signal or ``width`` has variances.
    TypeError
        If ``width`` is not a variable or ``kernel`` is not a distribution-like
        object.
    """
    x, y, template = _scipp_input(x, y)
    values = _smooth_kernel_values(
        x.values,
        y.values,
        width=_width_in_coordinate_unit(width, x),
        kernel=kernel,
        tail=tail,
    )
    return _scipp_output(template, y, values)


def smooth_gaussian(
    x: _ScippArray,
    y: sc.Variable | None = None,
    *,
    width: sc.Variable,
    tail: float = 1e-12,
) -> _ScippArray:
    """Smooth regularly sampled data with a fixed-width Gaussian kernel.

    Parameters
    ----------
    x:
        One-dimensional data to smooth, or strictly increasing, regularly
        spaced sample coordinates. A data array must have a dimension
        coordinate.
    y:
        Values to smooth when ``x`` contains the sample coordinates. Must be a
        one-dimensional variable with the same dimension as ``x``. Must be
        omitted when ``x`` is a data array.
    width:
        Standard deviation of the Gaussian. Must be a scalar with a unit
        compatible with the coordinate.
    tail:
        Total Gaussian probability omitted when truncating the kernel.

    Returns
    -------
    :
        Smoothed data of the same type as the input.

    See Also
    --------
    smooth_kernel:
        Smooth with a named or user-provided translation-invariant kernel.
    """
    return smooth_kernel(x, y, width=width, kernel="gaussian", tail=tail)


def smooth_rectangle(
    x: _ScippArray,
    y: sc.Variable | None = None,
    *,
    width: sc.Variable,
) -> _ScippArray:
    """Smooth regularly sampled data with a fixed-width rectangular kernel.

    The displacement is uniformly distributed on ``[-width, width]``.

    Parameters
    ----------
    x:
        One-dimensional data to smooth, or strictly increasing, regularly
        spaced sample coordinates. A data array must have a dimension
        coordinate.
    y:
        Values to smooth when ``x`` contains the sample coordinates. Must be a
        one-dimensional variable with the same dimension as ``x``. Must be
        omitted when ``x`` is a data array.
    width:
        Half-width of the rectangular kernel. Must be a scalar with a unit
        compatible with the coordinate.

    Returns
    -------
    :
        Smoothed data of the same type as the input.

    See Also
    --------
    smooth_kernel:
        Smooth with a named or user-provided translation-invariant kernel.
    """
    return smooth_kernel(x, y, width=width, kernel="rectangle")


def smooth_triangle(
    x: _ScippArray,
    y: sc.Variable | None = None,
    *,
    width: sc.Variable,
) -> _ScippArray:
    """Smooth regularly sampled data with a fixed-width triangular kernel.

    The displacement has symmetric triangular support on ``[-width, width]``
    and its peak at zero.

    Parameters
    ----------
    x:
        One-dimensional data to smooth, or strictly increasing, regularly
        spaced sample coordinates. A data array must have a dimension
        coordinate.
    y:
        Values to smooth when ``x`` contains the sample coordinates. Must be a
        one-dimensional variable with the same dimension as ``x``. Must be
        omitted when ``x`` is a data array.
    width:
        Half-width of the triangular kernel. Must be a scalar with a unit
        compatible with the coordinate.

    Returns
    -------
    :
        Smoothed data of the same type as the input.

    See Also
    --------
    smooth_kernel:
        Smooth with a named or user-provided translation-invariant kernel.
    """
    return smooth_kernel(x, y, width=width, kernel="triangle")


def smooth_relative_kernel(
    x: _ScippArray,
    y: sc.Variable | None = None,
    alpha: float = 1.0,
    kernel: _Kernel = "gaussian",
    tail: float = 1e-12,
    max_grid_points: int = 1_000_000,
) -> _ScippArray:
    """Smooth sampled data with a kernel of relative width.

    The kernel describes a distribution of relative displacements ``Z``, with
    displaced coordinates given by ``x' = x * (1 + alpha * Z)``. At the
    boundaries, the kernel is renormalized over the available finite input
    domain.

    Input that is not geometrically spaced is interpolated to a geometric grid,
    smoothed, and interpolated back to the original coordinates.

    Parameters
    ----------
    x:
        One-dimensional data to smooth, or positive, strictly increasing
        one-dimensional sample coordinates. A data array must have a
        dimension coordinate.
    y:
        Values to smooth when ``x`` contains the sample coordinates. Must be a
        one-dimensional variable with the same dimension as ``x``. Must be
        omitted when ``x`` is a data array.
    alpha:
        Scale factor for the relative-displacement distribution. Set to zero to
        return a copy of the input without smoothing.
    kernel:
        Kernel distribution. Supported names are ``'gaussian'``, ``'rectangle'``,
        and ``'triangle'``, including their aliases. Alternatively, provide a
        fully specified distribution with ``cdf``, ``ppf``, and ``support``
        methods.
    tail:
        Total probability omitted when truncating a kernel with unbounded support
        or support reaching the nonpositive coordinate domain.
    max_grid_points:
        Maximum permitted size of the intermediate geometric grid. Raises an
        error rather than silently reducing resolution if this limit is exceeded.

    Returns
    -------
    :
        Smoothed data of the same type as the input. Coordinates and units are
        preserved.

    Raises
    ------
    ValueError
        If the inputs have invalid values, if a data array has masks, if a
        string does not identify a supported kernel, or if the required
        intermediate grid exceeds ``max_grid_points``.
    scipp.DimensionError
        If the inputs are not one-dimensional or a pair of variables does not
        have matching dimensions.
    scipp.CoordError
        If a data array has no dimension coordinate or has a bin-edge
        coordinate.
    scipp.VariancesError
        If the signal has variances.
    TypeError
        If ``kernel`` is not a distribution-like object, or if
        ``max_grid_points`` is not an integer.
    """
    x, y, template = _scipp_input(x, y)
    values = _smooth_relative_kernel(
        x.values,
        y.values,
        alpha=alpha,
        kernel=kernel,
        tail=tail,
        max_grid_points=max_grid_points,
    )
    return _scipp_output(template, y, values)


def smooth_relative_gaussian(
    x: _ScippArray,
    y: sc.Variable | None = None,
    alpha: float = 1.0,
    tail: float = 1e-12,
    max_grid_points: int = 1_000_000,
) -> _ScippArray:
    """Smooth sampled data with a relative Gaussian kernel.

    ``alpha`` is the standard deviation of the Gaussian as a fraction of each
    coordinate value.

    Parameters
    ----------
    x:
        One-dimensional data to smooth, or positive, strictly increasing
        one-dimensional sample coordinates. A data array must have a
        dimension coordinate.
    y:
        Values to smooth when ``x`` contains the sample coordinates. Must be a
        one-dimensional variable with the same dimension as ``x``. Must be
        omitted when ``x`` is a data array.
    alpha:
        Relative standard deviation of the Gaussian kernel.
    tail:
        Total Gaussian probability omitted when truncating the kernel.
    max_grid_points:
        Maximum permitted size of the intermediate geometric grid.

    Returns
    -------
    :
        Smoothed data of the same type as the input.

    See Also
    --------
    smooth_relative_kernel:
        Smooth with a named or user-provided relative kernel.
    """
    return smooth_relative_kernel(
        x,
        y,
        alpha=alpha,
        kernel="gaussian",
        tail=tail,
        max_grid_points=max_grid_points,
    )


def smooth_relative_rectangle(
    x: _ScippArray,
    y: sc.Variable | None = None,
    alpha: float = 1.0,
    tail: float = 1e-12,
    max_grid_points: int = 1_000_000,
) -> _ScippArray:
    """Smooth sampled data with a relative rectangular kernel.

    The relative displacement is uniformly distributed on
    ``[-alpha, alpha]``.

    Parameters
    ----------
    x:
        One-dimensional data to smooth, or positive, strictly increasing
        one-dimensional sample coordinates. A data array must have a
        dimension coordinate.
    y:
        Values to smooth when ``x`` contains the sample coordinates. Must be a
        one-dimensional variable with the same dimension as ``x``. Must be
        omitted when ``x`` is a data array.
    alpha:
        Half-width of the rectangular kernel relative to each coordinate value.
    tail:
        Total probability omitted if the kernel reaches the nonpositive
        coordinate domain.
    max_grid_points:
        Maximum permitted size of the intermediate geometric grid.

    Returns
    -------
    :
        Smoothed data of the same type as the input.

    See Also
    --------
    smooth_relative_kernel:
        Smooth with a named or user-provided relative kernel.
    """
    return smooth_relative_kernel(
        x,
        y,
        alpha=alpha,
        kernel="rectangle",
        tail=tail,
        max_grid_points=max_grid_points,
    )


def smooth_relative_triangle(
    x: _ScippArray,
    y: sc.Variable | None = None,
    alpha: float = 1.0,
    tail: float = 1e-12,
    max_grid_points: int = 1_000_000,
) -> _ScippArray:
    """Smooth sampled data with a relative triangular kernel.

    The relative displacement has symmetric triangular support on
    ``[-alpha, alpha]`` and its peak at zero.

    Parameters
    ----------
    x:
        One-dimensional data to smooth, or positive, strictly increasing
        one-dimensional sample coordinates. A data array must have a
        dimension coordinate.
    y:
        Values to smooth when ``x`` contains the sample coordinates. Must be a
        one-dimensional variable with the same dimension as ``x``. Must be
        omitted when ``x`` is a data array.
    alpha:
        Half-width of the triangular kernel relative to each coordinate value.
    tail:
        Total probability omitted if the kernel reaches the nonpositive
        coordinate domain.
    max_grid_points:
        Maximum permitted size of the intermediate geometric grid.

    Returns
    -------
    :
        Smoothed data of the same type as the input.

    See Also
    --------
    smooth_relative_kernel:
        Smooth with a named or user-provided relative kernel.
    """
    return smooth_relative_kernel(
        x,
        y,
        alpha=alpha,
        kernel="triangle",
        tail=tail,
        max_grid_points=max_grid_points,
    )
