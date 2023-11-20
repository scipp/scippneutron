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
class Subframe:
    """
    Neutron pulse at a time-of-flight neutron source, described as the corners of a
    polygon (initially a rectangle) in time and inverse velocity.
    """

    time: sc.Variable
    inverse_velocity: sc.Variable

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


@dataclass
class Frame:
    distance: sc.Variable
    subframes: List[Subframe]


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
    max_time = 0
    max_iv = 0
    for frame, color in zip(frames, colors):
        # All subframes have same color
        for subframe in frame.subframes:
            time_unit = subframe.time.unit
            iv_unit = subframe.inverse_velocity.unit
            max_time = max(max_time, subframe.time.max().value)
            max_iv = max(max_iv, subframe.inverse_velocity.max().value)
            polygon = patches.Polygon(
                np.stack(
                    (subframe.time.values, subframe.inverse_velocity.values), axis=1
                ),
                closed=True,
                fill=True,
                color=color,
            )
            ax.add_patch(polygon)
    ax.set_xlabel(time_unit)
    ax.set_ylabel(iv_unit)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, max_iv)
    return plt


def to_svg(
    frames: List[Subframe], tmax: sc.Variable, ivmax: sc.Variable, scale=1
) -> str:
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
    distance: sc.Variable
    time_open: sc.Variable
    time_close: sc.Variable


def propagate(frame: Frame, distance: sc.Variable) -> Frame:
    """
    Propagate a neutron pulse to a new distance.

    Parameters
    ----------
    frame:
        Input neutron pulse.
    distance:
        New distance.
    """
    delta = distance - frame.distance
    if delta.value < 0:
        raise ValueError(f'Cannot propagate backwards: {delta}')
    subframes = [
        Subframe(
            time=subframe.time + delta * subframe.inverse_velocity,
            inverse_velocity=subframe.inverse_velocity,
        )
        for subframe in frame.subframes
    ]
    return Frame(distance=distance, subframes=subframes)


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
    frame = propagate(frame, chopper.distance)

    # A chopper can have multiple openings, call _chop for each of them. The result
    # is the union of the resulting subframes.
    chopped = Frame(distance=frame.distance, subframes=[])
    for subframe in frame.subframes:
        for open, close in zip(chopper.time_open, chopper.time_close):
            tmp = _chop(subframe, open, lambda time, open=open: time >= open)
            if not tmp:
                continue
            tmp = _chop(tmp, close, lambda time, close=close: time <= close)
            if tmp:
                chopped.subframes.append(tmp)
    return chopped


def _chop(frame: Subframe, time: sc.Variable, inside: Callable) -> Optional[Subframe]:
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
    if not output:
        return None
    time = sc.concat([t for t, _ in output], dim=frame.time.dim)
    inverse_velocity = sc.concat([v for _, v in output], dim=frame.inverse_velocity.dim)
    return Subframe(time=time, inverse_velocity=inverse_velocity)
