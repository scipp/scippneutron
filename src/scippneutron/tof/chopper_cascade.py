# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Compute result of applying a chopper cascade to a neutron pulse at a time-of-flight
neutron source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import scipp as sc


def wavelength_to_inverse_velocity(wavelength):
    h = sc.constants.h
    m_n = sc.constants.m_n
    return (wavelength * m_n / h).to(unit='s/m')


def propagate_times(
    time: sc.Variable, wavelength: sc.Variable, distance: sc.Variable
) -> sc.Variable:
    """
    Propagate a neutron frame by a distance.

    Parameters
    ----------
    time:
        Time of the neutron frame.
    wavelength:
        Wavelength of the neutron frame.
    distance:
        Distance to propagate. Can be a range of distances.

    Returns
    -------
    :
        Propagated time.
    """
    inverse_velocity = wavelength_to_inverse_velocity(wavelength)
    return time + distance * inverse_velocity


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
            raise sc.DimensionError(
                f'Inconsistent dims or shape: {time.sizes} vs {wavelength.sizes}'
            )
        self.time = time.to(unit='s', copy=False)
        self.wavelength = wavelength.to(unit='angstrom', copy=False)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Subframe):
            return NotImplemented
        return sc.identical(self.time, other.time) and sc.identical(
            self.wavelength, other.wavelength
        )

    def is_regular(self) -> bool:
        """
        Return True if the subframe is regular, i.e., if the vertex with the minimum
        wavelength also has the minimum time, and the vertex with the maximum wavelength
        also has the maximum time.
        """
        min_time = self.time == self.time.min()
        min_wavelength = self.wavelength == self.wavelength.min()
        max_time = self.time == self.time.max()
        max_wavelength = self.wavelength == self.wavelength.max()
        coinciding_min = min_time & min_wavelength
        coinciding_max = max_time & max_wavelength
        return coinciding_min.any() and coinciding_max.any()

    def propagate_by(self, distance: sc.Variable) -> Subframe:
        """
        Propagate subframe by a distance.

        Parameters
        ----------
        distance:
            Distance to propagate. Note that this is a difference, not an absolute
            value, in contrast to the distance in :py:meth:`Frame.propagate_to`.

        Returns
        -------
        :
            Propagated subframe.
        """
        return Subframe(
            time=propagate_times(self.time, self.wavelength, distance),
            wavelength=self.wavelength,
        )

    @property
    def start_time(self) -> sc.Variable:
        """The start time of the subframe."""
        return self.time.min()

    @property
    def end_time(self) -> sc.Variable:
        """The end time of the subframe."""
        return self.time.max()

    @property
    def start_wavelength(self) -> sc.Variable:
        """The start wavelength of the subframe."""
        return self.wavelength.min()

    @property
    def end_wavelength(self) -> sc.Variable:
        """The end wavelength of the subframe."""
        return self.wavelength.max()


@dataclass
class Frame:
    """
    A frame of neutrons, created from a single neutron pulse, potentially chopped into
    subframes by choppers.
    """

    distance: sc.Variable
    subframes: List[Subframe]

    def propagate_to(self, distance: sc.Variable) -> Frame:
        """
        Compute new frame by propagating to a distance.

        Parameters
        ----------
        distance:
            New distance.

        Returns
        -------
        :
            Propagated frame.
        """
        delta = distance - self.distance
        if delta.value < 0:
            raise ValueError(f'Cannot propagate backwards: {delta}')
        subframes = [subframe.propagate_by(delta) for subframe in self.subframes]
        return Frame(distance=distance, subframes=subframes)

    def chop(self, chopper: Chopper) -> Frame:
        """
        Compute a new frame by applying a chopper.

        A frame is a polygon in time and wavelength. Its initial shape is distorted
        by propagation to the chopper. The chopper then cuts off the parts of the frame
        that is outside of the chopper opening. Here we apply and algorithm that
        computes a new polygon that is the intersection of the frame and the chopper
        opening.

        In practice a chopper may have multiple openings, so a frame may be chopped into
        a number of subframes.

        Parameters
        ----------
        chopper:
            Chopper to apply.

        Returns
        -------
        :
            Chopped frame.
        """
        frame = self.propagate_to(chopper.distance)

        # A chopper can have multiple openings, call _chop for each of them. The result
        # is the union of the resulting subframes.
        chopped = Frame(distance=frame.distance, subframes=[])
        for subframe in frame.subframes:
            for open, close in zip(chopper.time_open, chopper.time_close):
                if (tmp := _chop(subframe, open, close_to_open=True)) is not None:
                    if (tmp := _chop(tmp, close, close_to_open=False)) is not None:
                        chopped.subframes.append(tmp)
        return chopped

    def frame_bounds(self) -> sc.Dataset:
        """The bounds of the frame, i.e., the global min and max time and wavelength."""
        start = sc.reduce([sub.start_time for sub in self.subframes]).min()
        end = sc.reduce([sub.end_time for sub in self.subframes]).max()
        wav_start = sc.reduce([sub.start_wavelength for sub in self.subframes]).min()
        wav_end = sc.reduce([sub.end_wavelength for sub in self.subframes]).max()
        return sc.Dataset(
            {
                'time': sc.concat([start, end], dim='bound'),
                'wavelength': sc.concat([wav_start, wav_end], dim='bound'),
            }
        )

    def subframe_bounds(self) -> sc.Dataset:
        """
        The bounds of the subframes, defined as the union over subframes.

        This is not the same as the bounds of the individual subframes, but defined as
        the union of all subframes. Subframes that overlap in time are "merged" into a
        single subframe.
        """
        starts = [subframe.start_time for subframe in self.subframes]
        ends = [subframe.end_time for subframe in self.subframes]
        # Given how time-propagation and chopping works, the min wavelength is always
        # given by the same vertex as the min time, and the max wavelength by the same
        # vertex as the max time. Thus, this check should generally always pass.
        # Exceptions may be subframes that have been created manually.
        if not all(subframe.is_regular() for subframe in self.subframes):
            raise NotImplementedError(
                'Subframes must be regular, i.e., min/max time and wavelength must '
                'coincide.'
            )
        wav_starts = [subframe.start_wavelength for subframe in self.subframes]
        wav_ends = [subframe.end_wavelength for subframe in self.subframes]
        # sort by start
        starts, ends, wav_starts, wav_ends = zip(
            *sorted(zip(starts, ends, wav_starts, wav_ends), key=lambda x: x[0])
        )
        bounds = []
        current = (starts[0], ends[0], wav_starts[0], wav_ends[0])
        for start, end, wav_start, wav_end in zip(
            starts[1:], ends[1:], wav_starts[1:], wav_ends[1:]
        ):
            # If start is before current end, merge
            if start <= current[1]:
                current = (
                    current[0],
                    max(current[1], end),
                    current[2],
                    max(current[3], wav_end),
                )
            else:
                bounds.append(current)
                current = (start, end, wav_start, wav_end)
        bounds.append(current)
        time_bounds = [
            sc.concat([start, end], dim='bound') for start, end, _, _ in bounds
        ]
        times = sc.concat(time_bounds, dim='subframe')
        wav_bounds = [
            sc.concat([wav_start, wav_end], dim='bound')
            for _, _, wav_start, wav_end in bounds
        ]
        wavs = sc.concat(wav_bounds, dim='subframe')
        return sc.Dataset({'time': times, 'wavelength': wavs})


class FrameSequence:
    def __init__(
        self,
        time_min: sc.Variable,
        time_max: sc.Variable,
        wavelength_min: sc.Variable,
        wavelength_max: sc.Variable,
    ):
        """
        Initialize a frame sequence from min/max time and wavelength of a pulse.

        The distance is set to 0 m.
        """
        time = sc.concat([time_min, time_max, time_max, time_min], dim='vertex').to(
            unit='s'
        )
        wavelength = sc.concat(
            [wavelength_min, wavelength_min, wavelength_max, wavelength_max],
            dim='vertex',
        ).to(unit='angstrom')
        self._frames = [
            Frame(
                distance=sc.scalar(0, unit='m'),
                subframes=[Subframe(time=time, wavelength=wavelength)],
            )
        ]

    def __len__(self) -> int:
        """Number of frames."""
        return len(self._frames)

    def __getitem__(self, item: int) -> Frame:
        """Get a frame by index."""
        return self._frames[item]

    def propagate_to(self, distance: sc.Variable) -> None:
        """
        Propagate the frame sequence to a distance, adding a new frame.

        Parameters
        ----------
        distance:
            Distance to propagate.
        """
        self._frames.append(self._frames[-1].propagate_to(distance))

    def chop(self, choppers: List[Chopper]) -> None:
        """
        Chop the frame sequence by a list of choppers.

        Parameters
        ----------
        choppers:
            List of choppers.
        """
        for chopper in choppers:
            self._frames.append(self._frames[-1].chop(chopper))

    def draw(self) -> Any:
        """Draw frames using matplotlib"""
        import matplotlib.colors as mcolors
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        colors = list(mcolors.TABLEAU_COLORS.values())
        colors += colors
        colors += colors
        max_time = 0
        max_wav = 0
        for frame, color in zip(self._frames, colors):
            # Add label to legend
            ax.plot([], [], color=color, label=f'{frame.distance.value:.2f} m')
            # All subframes have same color
            for subframe in frame.subframes:
                time_unit = subframe.time.unit
                wav_unit = subframe.wavelength.unit
                max_time = max(max_time, subframe.time.max().value)
                max_wav = max(max_wav, subframe.wavelength.max().value)
                polygon = patches.Polygon(
                    np.stack(
                        (subframe.time.values, subframe.wavelength.values), axis=1
                    ),
                    closed=True,
                    fill=True,
                    color=color,
                )
                ax.add_patch(polygon)
        ax.set_xlabel(time_unit)
        ax.set_ylabel(wav_unit)
        ax.set_xlim(0, max_time)
        ax.set_ylim(0, max_wav)
        ax.legend(loc='best')
        return fig, ax


@dataclass
class Chopper:
    distance: sc.Variable
    time_open: sc.Variable
    time_close: sc.Variable


def _chop(
    frame: Subframe, time: sc.Variable, close_to_open: bool
) -> Optional[Subframe]:
    inside = frame.time >= time if close_to_open else frame.time <= time
    output = []
    for i in range(len(frame.time)):
        # Note how j wraps around to 0
        j = (i + 1) % len(frame.time)
        inside_i = inside[i]
        inside_j = inside[j]
        if inside_i:
            output.append((frame.time[i], frame.wavelength[i]))
        if inside_i != inside_j:
            # Intersection
            t = (time - frame.time[i]) / (frame.time[j] - frame.time[i])
            v = (1 - t) * frame.wavelength[i] + t * frame.wavelength[j]
            output.append((time, v))
    if not output:
        return None
    time = sc.concat([t for t, _ in output], dim=frame.time.dim)
    wavelength = sc.concat([v for _, v in output], dim=frame.wavelength.dim)
    return Subframe(time=time, wavelength=wavelength)
