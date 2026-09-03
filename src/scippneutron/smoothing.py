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

import warnings
from numbers import Real
from typing import Any, Protocol, cast

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


type _Kernel = str | _Distribution
type _FloatArray = NDArray[np.float64]
type _IntArray = NDArray[np.int64]

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
    return offsets[first:last], weights[first:last]


class _Geometry(Protocol):
    """
    A grid on which the kernel is translation invariant.

    Both smoothing procedures displace a coordinate x by a distribution Z. The
    displacement law differs, but in both cases there is a working coordinate
    ``u`` in which the displacement is an additive offset independent of x, so
    that a single set of weights applies at every grid point. ``offset`` and
    ``displacement`` convert between a displacement z and its offset in u; they
    are inverses of each other.
    """

    #: Names the grid in user-facing messages.
    name: str

    def check(self, x: _FloatArray) -> None:
        """Reject coordinates outside the domain of the displacement law."""

    def coordinate(self, x: _FloatArray) -> _FloatArray:
        """Working coordinate u(x)."""

    def points(self, start: float, stop: float, count: int) -> _FloatArray:
        """``count`` samples from ``start`` to ``stop``, uniform in u."""

    def displacement_min(self, scale: float) -> float:
        """Smallest displacement z that keeps x within the valid domain."""

    def offset(self, scale: float, z: ArrayLike) -> Any:
        """Offset in u produced by displacement z."""

    def displacement(self, scale: float, u: ArrayLike) -> Any:
        """Displacement z producing an offset u."""


class _Uniform:
    """Constant kernel width: ``x' = x + scale * Z``, additive in x itself."""

    name = "uniform"

    def check(self, x: _FloatArray) -> None:
        pass

    def coordinate(self, x: _FloatArray) -> _FloatArray:
        return x

    def points(self, start: float, stop: float, count: int) -> _FloatArray:
        return np.linspace(start, stop, count)

    def displacement_min(self, scale: float) -> float:
        return -np.inf

    def offset(self, scale: float, z: ArrayLike) -> Any:
        return scale * np.asarray(z, dtype=float)

    def displacement(self, scale: float, u: ArrayLike) -> Any:
        return np.asarray(u, dtype=float) / scale


class _Geometric:
    """
    Relative kernel width: ``x' = x * (1 + scale * Z)``, additive in ``log(x)``.

    Equivalently, the kernel is

        K(x, x') = 1 / (scale * x) * f((x' - x) / (scale * x))

    for a distribution with PDF f. Displacements are restricted to ``x' > 0``,
    that is ``1 + scale * z > 0``.
    """

    name = "geometric"

    def check(self, x: _FloatArray) -> None:
        if np.any(x <= 0):
            raise ValueError("x must be positive")

    def coordinate(self, x: _FloatArray) -> _FloatArray:
        return cast(_FloatArray, np.log(x))

    def points(self, start: float, stop: float, count: int) -> _FloatArray:
        return np.geomspace(start, stop, count)

    def displacement_min(self, scale: float) -> float:
        return -1.0 / scale

    def offset(self, scale: float, z: ArrayLike) -> Any:
        # log1p(-1) is -inf, which _kernel_weights treats as unbounded support.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log1p(scale * np.asarray(z, dtype=float))

    def displacement(self, scale: float, u: ArrayLike) -> Any:
        return np.expm1(np.asarray(u, dtype=float)) / scale


def _kernel_weights(
    geometry: _Geometry,
    spacing: float,
    scale: float,
    kernel: _Kernel,
    tail: float,
    max_offset: int,
) -> tuple[_IntArray, _FloatArray]:
    """
    Cell-integrated kernel weights on a grid of the given spacing in u.

    Weights are exact integrals of the kernel probability over grid cells, so
    the result is a proper quadrature of the smoothing integral rather than a
    point sampling of the kernel.
    """
    dist = _as_kernel_distribution(kernel)

    z_min = geometry.displacement_min(scale)
    support_min, support_max = _kernel_support(dist)

    p_min = float(dist.cdf(z_min))
    reachable_mass = 1.0 - p_min
    if not np.isfinite(reachable_mass) or reachable_mass <= 0.0:
        raise ValueError("kernel has no mass in the valid domain for this scale")
    u_left = float(geometry.offset(scale, max(z_min, support_min)))
    u_right = float(geometry.offset(scale, support_max))

    # Unbounded support, or support reaching the edge of the valid domain,
    # gives an infinite bound in u. Truncate at the requested tail instead.
    if not (np.isfinite(u_left) and np.isfinite(u_right)):
        probabilities = (
            p_min + np.array([0.5 * tail, 1.0 - 0.5 * tail]) * reachable_mass
        )
        u_left, u_right = (
            float(value) for value in geometry.offset(scale, dist.ppf(probabilities))
        )
        # The geometric domain boundary may leave u_left non-finite; m_min is
        # safely clamped to the finite input below. u_right has no such bound.
        if not np.isfinite(u_right):
            raise ValueError("right kernel bound is not finite; increase tail")

    # Cells are centered at m*h and span [(m-1/2)h, (m+1/2)h].
    # Include offset zero even for one-sided kernels and clamp the stencil to
    # offsets that can contribute to the finite input.
    m_min = int(np.clip(np.floor(u_left / spacing + 0.5), -max_offset, 0))
    m_max = int(np.clip(np.ceil(u_right / spacing - 0.5), 0, max_offset))
    m = np.arange(m_min, m_max + 1, dtype=np.int64)

    lower = geometry.displacement(scale, (m - 0.5) * spacing)
    upper = geometry.displacement(scale, (m + 0.5) * spacing)
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
    # _kernel_weights preserves offset zero, which is valid for every i, so
    # [lower, upper] is never empty.

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


def _smooth_values(
    geometry: _Geometry,
    x: ArrayLike,
    y: ArrayLike,
    scale: float,
    kernel: _Kernel,
    tail: float,
    max_grid_points: int,
) -> _FloatArray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be non-negative")
    if not np.isfinite(tail) or not (0.0 < tail < 1.0):
        raise ValueError("tail must be between 0 and 1")
    _validate_max_grid_points(max_grid_points)
    if np.any(~np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    geometry.check(x)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    if x.size < 2 or scale == 0:
        return y.copy()

    u = geometry.coordinate(x)
    du = np.diff(u)
    u_range = float(u[-1] - u[0])

    # Preserve an existing regular grid. Otherwise choose a grid at least as
    # dense as the smallest input spacing, measured in the working coordinate.
    if np.allclose(du, du[0], rtol=1e-7, atol=0.0):
        k = float(x.size)
    else:
        # Coordinates that are distinct but collide, or nearly so, in the
        # working coordinate give an infinite point count. Let it propagate to
        # the guard below rather than raising here.
        with np.errstate(divide="ignore", over="ignore"):
            k = np.ceil(u_range / np.min(du)) + 1.0

    if not (k <= max_grid_points) and k > 2 * x.size:
        raise ValueError(
            f"{geometry.name} resampling would require too many points, exceeding "
            f"max_grid_points={max_grid_points:,}. Increase max_grid_points to "
            "allow a larger grid."
        )
    k = int(k)

    spacing = u_range / (k - 1)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("grid spacing must be positive")

    offsets, weights = _kernel_weights(
        geometry,
        spacing=spacing,
        scale=scale,
        kernel=kernel,
        tail=tail,
        max_offset=k - 1,
    )
    xp = geometry.points(float(x[0]), float(x[-1]), k)
    zg = _smooth_with_weights(np.interp(xp, x, y), offsets, weights)
    return cast(_FloatArray, np.interp(x, xp, zg))


def _validate_data(data: object) -> sc.Variable:
    """Validate user input and return its dimension coordinate."""
    if not isinstance(data, sc.DataArray):
        raise TypeError("expected a DataArray")
    if data.ndim != 1:
        raise sc.DimensionError("data must be one-dimensional")
    if data.is_binned:
        raise sc.DTypeError("data must not be binned")
    if data.dim not in data.coords:
        raise sc.CoordError("data must have a dimension coordinate")
    if data.coords.is_edges(data.dim):
        raise sc.CoordError("the dimension coordinate must not contain bin edges")
    if data.masks:
        raise ValueError("smoothing data with masks is not supported")
    if data.variances is not None:
        raise sc.VariancesError(
            "Smoothing signals with variances is not supported because it would "
            "introduce correlations between data points."
        )
    if np.any(~np.isfinite(data.values)):
        warnings.warn(
            "Data contains NaNs or infinities; smoothing may fall back to a slower "
            "method.",
            UserWarning,
            stacklevel=3,
        )
    return data.coords[data.dim]


def _with_values(data: sc.DataArray, values: _FloatArray) -> sc.DataArray:
    # The new container shares unchanged coordinates, but its data is independent.
    out = data.copy(deep=False)
    out.data = sc.array(dims=data.dims, values=values, unit=data.unit)
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
    data: sc.DataArray,
    *,
    scale: sc.Variable,
    kernel: _Kernel = "gaussian",
    tail: float = 1e-12,
    max_grid_points: int = 1_000_000,
) -> sc.DataArray:
    """Smooth sampled data with a translation-invariant kernel.

    The kernel describes a distribution of displacements ``Z``, with displaced
    coordinates given by ``x' = x + scale * Z``. At the boundaries, the kernel
    is renormalized over the available finite input domain.

    Input that is not uniformly spaced is interpolated to a uniform grid,
    smoothed, and interpolated back to the original coordinates.

    Parameters
    ----------
    data:
        One-dimensional data to smooth. Must have a strictly increasing
        dimension coordinate.
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
        Smoothed data. Coordinates and units are preserved.

    Raises
    ------
    ValueError
        If the inputs have invalid values, if a data array has masks, if a string
        does not identify a supported kernel, or if the required intermediate
        grid exceeds ``max_grid_points``.
    scipp.DimensionError
        If the input is not one-dimensional or ``scale`` is not scalar.
    scipp.CoordError
        If a data array has no dimension coordinate or has a bin-edge
        coordinate.
    scipp.DTypeError
        If ``data`` is binned.
    scipp.UnitError
        If the unit of ``scale`` is incompatible with the coordinate unit.
    scipp.VariancesError
        If the signal or ``scale`` has variances.
    TypeError
        If ``data`` is not a data array, ``scale`` is not a variable, ``kernel``
        is not a distribution-like object, or ``max_grid_points`` is not an
        integer.
    """
    x = _validate_data(data)
    values = _smooth_values(
        _Uniform(),
        x.values,
        data.values,
        scale=_scale_in_coordinate_unit(scale, x),
        kernel=kernel,
        tail=tail,
        max_grid_points=max_grid_points,
    )
    return _with_values(data, values)


def smooth_relative(
    data: sc.DataArray,
    *,
    scale: float | sc.Variable,
    kernel: _Kernel = "gaussian",
    tail: float = 1e-12,
    max_grid_points: int = 1_000_000,
) -> sc.DataArray:
    """Smooth sampled data with a kernel of relative width.

    The kernel describes a distribution of relative displacements ``Z``, with
    displaced coordinates given by ``x' = x * (1 + scale * Z)``. At the
    boundaries, the kernel is renormalized over the available finite input
    domain.

    Input that is not geometrically spaced is interpolated to a geometric grid,
    smoothed, and interpolated back to the original coordinates.

    Parameters
    ----------
    data:
        One-dimensional data to smooth. Must have a positive, strictly
        increasing dimension coordinate.
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
        Smoothed data. Coordinates and units are preserved.

    Raises
    ------
    ValueError
        If the inputs have invalid values, if a data array has masks, if a
        string does not identify a supported kernel, or if the required
        intermediate grid exceeds ``max_grid_points``.
    scipp.DimensionError
        If the input is not one-dimensional or ``scale`` is not scalar.
    scipp.CoordError
        If a data array has no dimension coordinate or has a bin-edge
        coordinate.
    scipp.DTypeError
        If ``data`` is binned.
    scipp.UnitError
        If ``scale`` is a variable with a non-dimensionless unit.
    scipp.VariancesError
        If the signal or ``scale`` has variances.
    TypeError
        If ``data`` is not a data array, ``scale`` is neither a real number nor a
        variable, ``kernel`` is not a distribution-like object, or
        ``max_grid_points`` is not an integer.
    """
    x = _validate_data(data)
    values = _smooth_values(
        _Geometric(),
        x.values,
        data.values,
        scale=_dimensionless_scale(scale),
        kernel=kernel,
        tail=tail,
        max_grid_points=max_grid_points,
    )
    return _with_values(data, values)
