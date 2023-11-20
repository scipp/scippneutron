# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Compute result of applying a chopper cascade to a neutron pulse at a time-of-flight
neutron source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipp as sc


def wavelength_to_inverse_velocity(wavelength):
    h = sc.constants.h
    m_n = sc.constants.m_n
    return (wavelength * m_n / h).to(unit='s/m')


@dataclass
class Subframe:
    """
    Neutron "subframe" at a time-of-flight neutron source, described as the corners of a
    polygon (initially a rectangle) in time and wavelength.
    """

    time: sc.Variable
    wavelength: sc.Variable

    def __init__(self, time: sc.Variable, wavelength: sc.Variable):
        if time.sizes != wavelength.sizes:
            raise ValueError(
                f'Inconsistent dims or shape: {time.sizes} vs {wavelength.sizes}'
            )
        self.time = time.to(unit='s', copy=False)
        self.wavelength = wavelength.to(unit='angstrom', copy=False)

    def propagate(self, distance: sc.Variable) -> Subframe:
        """
        Propagate a neutron pulse by a new distance.

        Parameters
        ----------
        distance:
            Distance to propagate.

        Returns
        -------
        :
            Propagated subframe.
        """
        inverse_velocity = wavelength_to_inverse_velocity(self.wavelength)
        return Subframe(
            time=self.time + distance * inverse_velocity,
            wavelength=self.wavelength,
        )


@dataclass
class Frame:
    """
    A frame of neutrons, created from a single neutron pulse, potentially chopped into
    subframes by choppers.
    """

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
    max_wav = 0
    for frame, color in zip(frames, colors):
        # All subframes have same color
        for subframe in frame.subframes:
            time_unit = subframe.time.unit
            wav_unit = subframe.wavelength.unit
            max_time = max(max_time, subframe.time.max().value)
            max_wav = max(max_wav, subframe.wavelength.max().value)
            polygon = patches.Polygon(
                np.stack((subframe.time.values, subframe.wavelength.values), axis=1),
                closed=True,
                fill=True,
                color=color,
            )
            ax.add_patch(polygon)
    ax.set_xlabel(time_unit)
    ax.set_ylabel(wav_unit)
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, max_wav)
    return fig, ax


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
        Input frame.
    distance:
        New distance.

    Returns
    -------
    :
        Propagated frame.
    """
    delta = distance - frame.distance
    if delta.value < 0:
        raise ValueError(f'Cannot propagate backwards: {delta}')
    subframes = [subframe.propagate(delta) for subframe in frame.subframes]
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

    Returns
    -------
    :
        Chopped frame.
    """
    frame = propagate(frame, chopper.distance)

    # A chopper can have multiple openings, call _chop for each of them. The result
    # is the union of the resulting subframes.
    chopped = Frame(distance=frame.distance, subframes=[])
    for subframe in frame.subframes:
        for open, close in zip(chopper.time_open, chopper.time_close):
            if (tmp := _chop(subframe, open, close_to_open=True)) is not None:
                if (tmp := _chop(tmp, close, close_to_open=False)) is not None:
                    chopped.subframes.append(tmp)
    return chopped


def _chop(
    frame: Subframe, time: sc.Variable, close_to_open: bool
) -> Optional[Subframe]:
    def inside(t):
        return t >= time if close_to_open else t <= time

    output = []
    for i in range(len(frame.time)):
        j = (i + 1) % len(frame.time)
        inside_i = inside(frame.time[i])
        inside_j = inside(frame.time[j])
        if inside_i != inside_j:
            # Intersection
            t = (time - frame.time[i]) / (frame.time[j] - frame.time[i])
            v = (1 - t) * frame.wavelength[i] + t * frame.wavelength[j]
            output.append((time, v))
        if inside_j:
            output.append((frame.time[j], frame.wavelength[j]))
    if not output:
        return None
    time = sc.concat([t for t, _ in output], dim=frame.time.dim)
    wavelength = sc.concat([v for _, v in output], dim=frame.wavelength.dim)
    return Subframe(time=time, wavelength=wavelength)
