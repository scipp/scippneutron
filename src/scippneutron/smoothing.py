# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Kernel smoothing of one-dimensional sampled signals.

``smooth`` applies a translation-invariant displacement distribution, for a kernel
whose width is constant in coordinate units. ``smooth_relative`` instead applies a
distribution of fractional displacements, so its width scales with the coordinate.

Both procedures accept irregularly spaced samples. They interpolate onto a uniform
or geometric working grid that is no coarser than the tightest input spacing,
integrate the kernel probability over grid cells, renormalize the convolution at
finite boundaries, and interpolate the result back to the original coordinates.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import scipp as sc
from numpy.typing import ArrayLike, NDArray
from scipy.signal import convolve
from scipy.stats import norm, triang, uniform

__all__ = [
    "smooth",
    "smooth_relative",
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
    "boxcar": uniform(loc=-1.0, scale=2.0),
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
    scale: float,
    kernel: _Kernel,
    tail: float,
    max_offset: int,
) -> tuple[_IntArray, _FloatArray]:
    """
    Weights for smoothing on a geometric grid q_i = q0 * exp(i * log_spacing).

    The kernel distribution describes the relative displacement Z:

        q' = q * (1 + scale * Z)

    Equivalently,

        K(q, q') = 1 / (scale * q) * f((q' - q) / (scale * q))

    where f is the PDF of the supplied distribution.
    """
    dist = _as_kernel_distribution(kernel)

    # Positive physical domain:
    #
    #   q' > 0
    #   q * (1 + scale * z) > 0
    #   z > -1 / scale
    z_domain_min = -1.0 / scale

    p_domain_min = float(dist.cdf(z_domain_min))
    norm_mass = 1.0 - p_domain_min

    if not np.isfinite(norm_mass) or norm_mass <= 0.0:
        raise ValueError("kernel has no positive-domain mass for this scale")

    support_min, support_max = _kernel_support(dist)

    # Exact z-support after clipping to q' > 0.
    z_left_exact = max(z_domain_min, support_min)
    z_right_exact = support_max

    # If the clipped support maps to finite log-space, use it exactly.
    # If it touches q' = 0, the log lower bound is -inf, so use a tail cutoff.
    has_finite_log_support = (
        np.isfinite(z_left_exact)
        and np.isfinite(z_right_exact)
        and (1.0 + scale * z_left_exact > 0.0)
    )

    if has_finite_log_support:
        u_left = np.log1p(scale * z_left_exact)
        u_right = np.log1p(scale * z_right_exact)
    else:
        probabilities = (
            p_domain_min + np.array([0.5 * tail, 1.0 - 0.5 * tail]) * norm_mass
        )
        z_left, z_right = (float(value) for value in dist.ppf(probabilities))

        u_left = np.log1p(scale * z_left)
        u_right = np.log1p(scale * z_right)

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

    # z = (q' - q) / (scale q)
    #   = (exp(u) - 1) / scale
    zL = np.expm1(L) / scale
    zU = np.expm1(U) / scale

    # Exact cell-integrated weights in log-space.
    w = (dist.cdf(zU) - dist.cdf(zL)) / norm_mass
    w = np.maximum(w, 0.0)

    return _trim_kernel_weights(m, w)


def _translation_invariant_kernel_weights(
    spacing: float,
    scale: float,
    kernel: _Kernel,
    tail: float,
    max_offset: int,
) -> tuple[_IntArray, _FloatArray]:
    dist = _as_kernel_distribution(kernel)
    z_left, z_right = _kernel_support(dist)
    if not np.all(np.isfinite([z_left, z_right])):
        z_left, z_right = (
            float(value) for value in dist.ppf([0.5 * tail, 1.0 - 0.5 * tail])
        )

    bounds = scale * np.array([z_left, z_right])
    if not np.all(np.isfinite(bounds)):
        raise ValueError("kernel bounds are not finite; increase tail")

    m_min = int(np.clip(np.floor(bounds[0] / spacing + 0.5), -max_offset, 0))
    m_max = int(np.clip(np.ceil(bounds[1] / spacing - 0.5), 0, max_offset))
    m = np.arange(m_min, m_max + 1, dtype=np.int64)

    lower = (m - 0.5) * spacing / scale
    upper = (m + 0.5) * spacing / scale
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


def _validate_max_grid_points(max_grid_points: int) -> None:
    if isinstance(max_grid_points, bool | np.bool_) or not isinstance(
        max_grid_points, int | np.integer
    ):
        raise TypeError("max_grid_points must be an integer")
    if max_grid_points < 2:
        raise ValueError("max_grid_points must be at least 2")


def _smooth_relative_values(
    x: ArrayLike,
    y: ArrayLike,
    scale: float,
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
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be non-negative")
    if not np.isfinite(tail) or not (0.0 < tail < 1.0):
        raise ValueError("tail must be between 0 and 1")
    _validate_max_grid_points(max_grid_points)
    if np.any(~np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if np.any(x <= 0):
        raise ValueError("x must be positive")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    if x.size == 0 or scale == 0 or x.size == 1:
        return y.copy()

    logx = np.log(x)
    dlog = np.diff(logx)
    log_range = float(logx[-1] - logx[0])

    # Preserve an existing geometric grid. Otherwise choose a geometric grid
    # at least as dense as the smallest input spacing in log-space.
    if np.allclose(dlog, dlog[0], rtol=1e-7, atol=0.0):
        k = x.size
    else:
        k = np.ceil(log_range / np.min(dlog)) + 1.0

    if not (k <= max_grid_points) and k > 2 * x.size:
        raise ValueError(
            "geometric resampling would require too many points, exceeding "
            f"max_grid_points={max_grid_points:,}. Increase max_grid_points to "
            "allow a larger grid."
        )
    k = int(k)
    log_spacing = log_range / (k - 1)
    if not np.isfinite(log_spacing) or log_spacing <= 0:
        raise ValueError("log_spacing must be positive")

    xp = np.geomspace(x[0], x[-1], k)
    yg = np.interp(xp, x, y)
    offsets, weights = _relative_kernel_weights(
        scale=scale,
        log_spacing=log_spacing,
        kernel=kernel,
        tail=tail,
        max_offset=k - 1,
    )
    zg = _smooth_with_weights(yg, offsets, weights)
    return cast(_FloatArray, np.interp(x, xp, zg))


def _smooth_values(
    x: ArrayLike,
    y: ArrayLike,
    scale: float,
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
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be non-negative")
    if not np.isfinite(tail) or not (0.0 < tail < 1.0):
        raise ValueError("tail must be between 0 and 1")
    _validate_max_grid_points(max_grid_points)
    if np.any(~np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    if x.size == 0 or scale == 0 or x.size == 1:
        return y.copy()

    spacing = np.diff(x)
    x_range = float(x[-1] - x[0])

    # Preserve an existing uniform grid. Otherwise choose a uniform grid at
    # least as dense as the smallest input spacing.
    if np.allclose(spacing, spacing[0], rtol=1e-7, atol=0.0):
        k = x.size
    else:
        k = int(np.ceil(x_range / np.min(spacing))) + 1

    if k > max_grid_points and k > 2 * x.size:
        raise ValueError(
            "uniform resampling would require too many points, exceeding "
            f"max_grid_points={max_grid_points:,}. Increase max_grid_points to "
            "allow a larger grid."
        )

    spacing = x_range / (k - 1)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive")
    offsets, weights = _translation_invariant_kernel_weights(
        spacing=spacing,
        scale=scale,
        kernel=kernel,
        tail=tail,
        max_offset=k - 1,
    )
    xp = np.linspace(x[0], x[-1], k)
    yg = np.interp(xp, x, y)

    zg = _smooth_with_weights(yg, offsets, weights)
    return cast(_FloatArray, np.interp(x, xp, zg))


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
    elif not isinstance(x, sc.Variable) or not isinstance(y, sc.Variable):
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


def _scale_in_coordinate_unit(scale: sc.Variable, x: sc.Variable) -> float:
    if not isinstance(scale, sc.Variable):
        raise TypeError("scale must be a scipp.Variable")
    if scale.ndim != 0:
        raise sc.DimensionError("scale must be a scalar")
    if scale.variances is not None:
        raise sc.VariancesError("kernel scales with variances are not supported")
    return float(scale.to(unit=x.unit).value)


def _dimensionless_scale(scale: object) -> float:
    if not isinstance(scale, sc.Variable):
        if not isinstance(scale, Real):
            raise TypeError("scale must be a real number or a scipp.Variable")
        return float(scale)
    if scale.ndim != 0:
        raise sc.DimensionError("scale must be a scalar")
    if scale.variances is not None:
        raise sc.VariancesError("kernel scales with variances are not supported")
    return float(scale.to(unit=sc.units.dimensionless).value)


def smooth(
    x: _ScippArray,
    y: sc.Variable | None = None,
    *,
    scale: sc.Variable,
    kernel: _Kernel = "gaussian",
    tail: float = 1e-12,
    max_grid_points: int = 1_000_000,
) -> _ScippArray:
    """Smooth sampled data with a translation-invariant kernel.

    The kernel describes a distribution of displacements ``Z``, with displaced
    coordinates given by ``x' = x + scale * Z``. At the boundaries, the kernel
    is renormalized over the available finite input domain.

    Input that is not uniformly spaced is interpolated to a uniform grid,
    smoothed, and interpolated back to the original coordinates.

    Parameters
    ----------
    x:
        One-dimensional data to smooth, or strictly increasing sample
        coordinates. A data array must have a dimension coordinate.
    y:
        Values to smooth when ``x`` contains the sample coordinates. Must be a
        one-dimensional variable with the same dimension as ``x``. Must be
        omitted when ``x`` is a data array.
    scale:
        Scale factor for the displacement distribution. Must be a scalar with a
        unit compatible with the coordinate. Set to zero to return a copy of the
        input without smoothing.
    kernel:
        Kernel distribution. The canonical names are ``'gaussian'``,
        ``'boxcar'``, and ``'triangular'``. They represent a standard normal
        distribution, a uniform distribution on [-1, 1], and a symmetric
        triangular distribution on [-1, 1], respectively. Other aliases are
        accepted. Alternatively, provide a fully specified distribution with
        ``cdf``, ``ppf``, and ``support`` methods.
    tail:
        Total probability omitted when truncating a kernel with unbounded
        support.
    max_grid_points:
        Intermediate uniform grids no larger than this are always allowed. Larger
        grids may be rejected to guard against excessive resampling.

    Returns
    -------
    :
        Smoothed data of the same type as the input. Coordinates and units are
        preserved.

    Raises
    ------
    ValueError
        If the inputs have invalid values, if a data array has masks, if a string
        does not identify a supported kernel, or if the required intermediate
        grid exceeds ``max_grid_points``.
    scipp.DimensionError
        If the inputs are not one-dimensional, a pair of variables does not
        have matching dimensions, or ``scale`` is not scalar.
    scipp.CoordError
        If a data array has no dimension coordinate or has a bin-edge
        coordinate.
    scipp.UnitError
        If the unit of ``scale`` is incompatible with the coordinate unit.
    scipp.VariancesError
        If the signal or ``scale`` has variances.
    TypeError
        If ``scale`` is not a variable, ``kernel`` is not a distribution-like
        object, or ``max_grid_points`` is not an integer.
    """
    x, y, template = _scipp_input(x, y)
    values = _smooth_values(
        x.values,
        y.values,
        scale=_scale_in_coordinate_unit(scale, x),
        kernel=kernel,
        tail=tail,
        max_grid_points=max_grid_points,
    )
    return _scipp_output(template, y, values)


def smooth_relative(
    x: _ScippArray,
    y: sc.Variable | None = None,
    *,
    scale: float | sc.Variable,
    kernel: _Kernel = "gaussian",
    tail: float = 1e-12,
    max_grid_points: int = 1_000_000,
) -> _ScippArray:
    """Smooth sampled data with a kernel of relative width.

    The kernel describes a distribution of relative displacements ``Z``, with
    displaced coordinates given by ``x' = x * (1 + scale * Z)``. At the
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
    scale:
        Dimensionless scale factor for the relative-displacement distribution.
        May be a real number or a scalar, dimensionless variable. Set to zero to
        return a copy of the input without smoothing.
    kernel:
        Kernel distribution. The canonical names are ``'gaussian'``,
        ``'boxcar'``, and ``'triangular'``. They represent a standard normal
        distribution, a uniform distribution on [-1, 1], and a symmetric
        triangular distribution on [-1, 1], respectively. Other aliases are
        accepted. Alternatively, provide a fully specified distribution with
        ``cdf``, ``ppf``, and ``support`` methods.
    tail:
        Total probability omitted when truncating a kernel with unbounded support
        or support reaching the nonpositive coordinate domain.
    max_grid_points:
        Intermediate geometric grids no larger than this are always allowed. Larger
        grids may be rejected to guard against excessive resampling.

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
        have matching dimensions, or ``scale`` is not scalar.
    scipp.CoordError
        If a data array has no dimension coordinate or has a bin-edge
        coordinate.
    scipp.UnitError
        If ``scale`` is a variable with a non-dimensionless unit.
    scipp.VariancesError
        If the signal or ``scale`` has variances.
    TypeError
        If ``scale`` is neither a real number nor a variable, ``kernel`` is not
        a distribution-like object, or ``max_grid_points`` is not an integer.
    """
    x, y, template = _scipp_input(x, y)
    values = _smooth_relative_values(
        x.values,
        y.values,
        scale=_dimensionless_scale(scale),
        kernel=kernel,
        tail=tail,
        max_grid_points=max_grid_points,
    )
    return _scipp_output(template, y, values)
