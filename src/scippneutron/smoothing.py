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
    "smooth_relative_gaussian",
    "smooth_relative_kernel",
    "smooth_relative_rectangle",
    "smooth_relative_triangle",
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
    # Standard Gaussian in relative-coordinate units.
    "gaussian": norm(),
    "normal": norm(),
    # Centered rectangle on [-1, 1].
    # With alpha=a, this has relative support [-a, a].
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

    try:
        support_min, support_max = dist.support()
    except TypeError as e:
        raise TypeError(
            "kernel must be a fully specified distribution. For distributions "
            "with shape parameters, pass a frozen distribution such as "
            "triang(c=0.5), not triang."
        ) from e

    support_min = float(support_min)
    support_max = float(support_max)

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

    nonzero = np.flatnonzero(w > 0.0)
    if nonzero.size == 0:
        # The distribution has no mass within reach of the finite input.
        # Returning zero weights lets the caller mark those values as NaN.
        return m, w

    # Trim zero-only ends, but preserve offset zero for convolution alignment.
    zero = -m_min
    first = min(nonzero[0], zero)
    last = max(nonzero[-1], zero) + 1

    # Renormalize after finite tail truncation.
    trimmed_weights = w[first:last]
    return m[first:last], trimmed_weights / trimmed_weights.sum()


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

    m, w = _relative_kernel_weights(
        log_spacing=log_spacing,
        alpha=alpha,
        kernel=kernel,
        tail=tail,
        max_offset=y.size - 1,
    )

    # Desired operation:
    #
    #   out[i] = sum_m w[m] * y[i + m]
    #
    # scipy convolution reverses the second argument, hence w[::-1].
    full = cast(_FloatArray, convolve(y, w[::-1], mode="full"))

    start = int(m[-1])
    numerator = full[start : start + y.size]

    denom = _valid_weight_sums(y.size, m, w)

    out = np.full_like(numerator, np.nan)
    np.divide(
        numerator,
        denom,
        out=out,
        where=denom > 0.0,
    )
    return out


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


def _smooth_variable(
    x: sc.Variable,
    y: sc.Variable,
    *,
    alpha: float,
    kernel: _Kernel,
    tail: float,
    max_grid_points: int,
) -> sc.Variable:
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

    return sc.array(
        dims=y.dims,
        values=_smooth_relative_kernel(
            x.values,
            y.values,
            alpha=alpha,
            kernel=kernel,
            tail=tail,
            max_grid_points=max_grid_points,
        ),
        unit=y.unit,
    )


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

        out = x.copy(deep=False)
        out.data = _smooth_variable(
            x.coords[x.dim],
            x.data,
            alpha=alpha,
            kernel=kernel,
            tail=tail,
            max_grid_points=max_grid_points,
        )
        return out

    if not isinstance(y, sc.Variable):
        raise TypeError("expected a DataArray or a pair of Variables")
    return _smooth_variable(
        x,
        y,
        alpha=alpha,
        kernel=kernel,
        tail=tail,
        max_grid_points=max_grid_points,
    )


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
