# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Goniometers for spectroscopy experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipp as sc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class Goniometer:
    """Description of a goniometer.

    This class encodes how a goniometer is configured in its neutral position.
    It does not encode concrete motor positions.

    Axes are defined in terms of a named x, y, z axis of the *lab frame*.
    That is, y points up (the opposite direction of gravity), z points along the
    incident beam projected onto the horizontal plane, and x is perpendicular to both,
    forming a right-handed coordinate system.
    Note that this differs slightly from the convention used in
    `Coordinate Transformations <../../user-guide/coordinate-transformations.html>`_
    because for goniometers, the z-axis is always horizontal.

    Motors rotate around their axis in a right-handed sense.
    The sign ("+" or "-") indicates the orientation of the rotation axis.

    Examples
    --------
    When formatted as below, the axes can be read as a stack of motors in the order
    they are written in code.
    For example, given

        >>> from scippneutron.spectroscopy.goniometer import Goniometer, GoniometerAxis
        >>> inf = sc.scalar(np.inf, unit="rad")
        >>> goniometer = Goniometer(
        ...     axes=[
        ...         GoniometerAxis(name="C", axis="+x", min=-inf, max=inf),
        ...         GoniometerAxis(name="B", axis="-z", min=-inf, max=inf),
        ...         GoniometerAxis(name="A", axis="+y", min=-inf, max=inf),
        ...     ]
        ... )

    Axis A is attached to the lab, B is attached to A, and C is attached to B.
    Finally, the sample is attached to C.
    So when A rotates, the orientation of B and C changes.
    When B rotates, the orientation of C changes, but A is unaffected.
    And when C rotates, only the sample is affected.
    """

    def __init__(self, axes: Iterable[GoniometerAxis]) -> None:
        self.axes = list(axes)

    @overload
    def plot(self, *, ax: Axes3D) -> plt.Figure: ...

    @overload
    def plot(self, *, ax: None = None) -> None: ...

    def plot(self, *, ax: Axes3D | None = None) -> plt.Figure | None:
        """Display the axes in a 3D plot.

        Parameters
        ----------
        ax:
            If given, plot into these axes.
            Must use ``projection="3d"``.

        Returns
        -------
        :
            A figure with the plot if ``ax`` is not given.
            Returns ``None`` if ``ax`` is given.
        """
        if ax is None:
            fig = plt.figure()
            ax: Axes3D = fig.add_subplot(projection='3d')  # type: ignore[assignment]
        else:
            fig = None

        ax.view_init(elev=None, azim=-135, vertical_axis="y")

        x, y, z = 0, 0, 0
        ax.quiver(x, y, z, 1, 0, 0, length=1, arrow_length_ratio=0.20, colors='k')
        ax.quiver(x, y, z, 0, 1, 0, length=1, arrow_length_ratio=0.20, colors='k')
        ax.quiver(x, y, z, 0, 0, 1, length=1, arrow_length_ratio=0.20, colors='k')

        ax.text(1.1, 0, 0, "x", color='k')
        ax.text(0, 1.1, 0, "y", color='k')
        ax.text(0, 0, 1.1, "z", color='k')

        for i, axis in enumerate(self.axes):
            _draw_goniometer_axis(
                ax,
                axis=axis.vector().value,
                y=-0.5 * (i + 1),
                color=f"C{len(self.axes) - i}",
                name=axis.name,
            )

        # Set limits such that the plot axes are a cube.
        ylim = -len(self.axes) / 2 - 0.5, 1.3
        ax_height = ylim[1] - ylim[0]
        ax.set_xlim(-ax_height / 2, ax_height / 2)
        ax.set_zlim(-ax_height / 2, ax_height / 2)
        ax.set_ylim(*ylim)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticks([])
            axis.line.set_visible(False)

        return fig


class GoniometerAxis:
    """A single rotation axis of a goniometer.

    Parameters
    ----------
    name:
        A name for this axis.
    axis:
        Indicates which axis of the lab coordinate system
        this goniometer axis rotates around.
    min:
        The minimum angle this axis can rotate to.
    max:
        The maximum angle this axis can rotate to.

    See Also
    --------
    Goniometer:
        For how to use this class.
    """

    __slots__ = ("axis", "limits", "name")

    def __init__(
        self, *, name: str, axis: Axis, min: sc.Variable, max: sc.Variable
    ) -> None:
        self.name = name
        self.axis = _check_axis_param(axis)
        self.limits = (min, sc.to_unit(max, min.unit))

    def vector(self) -> sc.Variable:
        """Return a unit vector pointing along this axis."""
        return sc.vector(
            [
                1 * ("x" in self.axis),
                1 * ("y" in self.axis),
                1 * ("z" in self.axis),
            ]
        ) * (-1 if "-" in self.axis else +1)

    def __str__(self) -> str:
        return (
            f"GoniometerAxis({self.name!r}, axis={self.axis!r}, "
            f"limits=({self.limits[0]:c}, {self.limits[1]:c}))"
        )

    def __repr__(self) -> str:
        return str(self)


Axis = Literal['+x', '-x', '+y', '-y', '+z', '-z']


def _check_axis_param(axis: Axis) -> Axis:
    if axis not in Axis.__args__:
        raise ValueError(
            f"Invalid axis: {axis!r}, expected one of {', '.join(Axis.__args__)}"
        )
    return axis


def _draw_goniometer_axis(
    ax: Axes3D,
    *,
    axis: npt.ArrayLike,
    y: float,
    color: str,
    name: str | None = None,
    axis_length: float = 1,
    circle_radius: float = 0.5,
    n_circle_segments: int = 20,
) -> None:
    center = np.array([0, y, 0])
    axis = np.asarray(axis) / np.linalg.norm(axis)

    tail = center - axis * axis_length / 2
    ax.quiver(
        *tail,
        *axis,
        length=axis_length,
        arrow_length_ratio=0.2,
        colors=color,
        alpha=0.7,
        linewidth=2,
    )
    if name is not None:
        ax.text(
            *tail - axis * axis_length / 4,
            name,
            color=color,
            horizontalalignment="center",
            verticalalignment="center",
        )

    _draw_arrow_circle(
        ax,
        axis=axis,
        center=center,
        radius=circle_radius,
        n_segments=n_circle_segments,
        color=color,
    )


def _draw_arrow_circle(
    ax: Axes3D,
    *,
    axis: npt.ArrayLike,
    center: npt.ArrayLike,
    radius: float,
    n_segments: int,
    color: str,
    arc_fraction: float = 3 / 4,
) -> None:
    axis = np.asarray(axis, dtype=float)
    w = axis / np.linalg.norm(axis)

    # Pick a reference vector that is not (nearly) parallel to the axis.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, w)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0])

    u = np.cross(ref, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)  # right-handed: u x v = w

    center = np.asarray(center, dtype=float)
    angles = np.linspace(0, arc_fraction * 2 * np.pi, n_segments + 1)
    points = center + radius * (
        np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    )

    lines = Line3DCollection([points[:-1]], colors=color)
    vec = points[-1] - points[-2]
    ax.quiver(
        *points[-2],
        *vec,
        length=np.linalg.norm(vec),
        arrow_length_ratio=15,
        colors=color,
    )
    ax.add_artist(lines)
