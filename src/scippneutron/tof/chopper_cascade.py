# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Compute result of applying a chopper cascade to a neutron pulse at a time-of-flight
neutron source.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import scipp as sc


@dataclass
class Frame:
    """
    Neutron pulse at a time-of-flight neutron source, described as the corners of a
    polygon (initially a rectangle) in time and inverse velocity.
    """

    time: np.ndarray
    inverse_velocity: np.ndarray

    def _repr_html_(self) -> str:
        """
        SVG representation of the frame, as a polygon.
        """
        min_time = self.time.min().value
        max_time = self.time.max().value
        min_inverse_velocity = self.inverse_velocity.min().value
        max_inverse_velocity = self.inverse_velocity.max().value
        dt = max_time - min_time
        div = max_inverse_velocity - min_inverse_velocity
        dt = 0.021
        div = 0.01
        return f"""
        <svg viewBox="{0} {0} {div} {dt}" xmlns="http://www.w3.org/2000/svg">
        {self.draw_polygon()}
        </svg>
        """

    def draw_polygon(self, color: Optional[str] = None) -> str:
        color = color or 'black'
        points = ','.join(
            f'{y},{x}' for x, y in zip(self.time.values, self.inverse_velocity.values)
        )
        return f"""
            <polygon points="{points}" fill="{color}""/>
        """


def draw_matplotlib(frames: List[Frame]) -> None:
    """Draw frames using matplotlib"""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    colors = [
        'red',
        'green',
        'blue',
        'cyan',
        'magenta',
        'orange',
        'purple',
        'brown',
        'pink',
    ]
    for frame, color in zip(frames, colors):
        polygon = patches.Polygon(
            np.stack((frame.time.values, frame.inverse_velocity.values), axis=1),
            closed=True,
            fill=True,
            color=color,
        )
        ax.add_patch(polygon)
    ax.set_xlabel(frames[0].time.unit)
    ax.set_ylabel(frames[0].inverse_velocity.unit)
    max_time = max(frame.time.max().value for frame in frames)
    max_iv = max(frame.inverse_velocity.max().value for frame in frames)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, max_iv)
    return plt


def to_svg(frames: List[Frame], tmax: sc.Variable, ivmax: sc.Variable, scale=1) -> str:
    from IPython.display import HTML, display

    colors = [
        'red',
        'green',
        'blue',
        'yellow',
        'cyan',
        'magenta',
        'orange',
        'purple',
        'brown',
        'pink',
    ]
    svg = f"""
    <svg viewBox="0 0 {tmax.value} {ivmax.value}" xmlns="http://www.w3.org/2000/svg">
    {''.join(frame.draw_polygon(color) for frame, color in zip(frames, colors))}
    </svg>
    """
    return display(
        HTML(
            f"""
        <div style="width: {scale*100}%; overflow-x: auto; overflow-y: hidden;">
            {svg}
        </div>
        """
        )
    )


@dataclass
class Chopper:
    time_open: float
    time_close: float


def propagate(frame: Frame, distance: float) -> Frame:
    """
    Propagate a neutron pulse through a distance.

    Parameters
    ----------
    frame:
        Input neutron pulse.
    distance:
        Distance to propagate, in meter.
    """
    return Frame(
        time=frame.time + distance * frame.inverse_velocity,
        inverse_velocity=frame.inverse_velocity,
    )


def chop(frame: Frame, chopper: Chopper) -> Frame:
    """
    Apply a chopper to a neutron frame.

    A frame is a polygon in time and inverse velocity. Its initial shape is distorted
    by propagation to the chopper. The chopper then cuts off the parts of the frame
    that is outside of the chopper opening. Here we apply and algorithm that
    computes a new polygon that is the intersection of the frame and the chopper
    opening.

    Parameters
    ----------
    frame:
        Input neutron pulse.
    chopper:
        Chopper to apply.
    """

    # The chopper's time_open and time_close define lines that intersect the polygon.
    # We find the intersections and keep the points that are inside the chopper.
    def inside(time: sc.Variable) -> sc.Variable:
        return time >= chopper.time_open

    frame = _chop(frame, chopper.time_open, lambda time: time >= chopper.time_open)
    return _chop(frame, chopper.time_close, lambda time: time <= chopper.time_close)


def _chop(frame: Frame, time: sc.Variable, inside: Callable) -> Frame:
    output = []
    for i in range(len(frame.time)):
        j = (i + 1) % len(frame.time)
        inside_i = inside(frame.time[i])
        inside_j = inside(frame.time[j])
        if inside_i != inside_j:
            # Intersection
            t = (time - frame.time[i]) / (frame.time[j] - frame.time[i])
            v = (1 - t) * frame.inverse_velocity[i] + t * frame.inverse_velocity[j]
            output.append((time, v))
        if inside_j:
            output.append((frame.time[j], frame.inverse_velocity[j]))
    time = sc.concat([t for t, _ in output], dim=frame.time.dim)
    inverse_velocity = sc.concat([v for _, v in output], dim=frame.inverse_velocity.dim)
    return Frame(time=time, inverse_velocity=inverse_velocity)
