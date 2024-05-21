# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
Compute result of applying a chopper cascade to a neutron pulse at a time-of-flight
neutron source.

See :py:class:`FrameSequence` for the main entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


class Subframe:
    """
    Neutron "subframe" at a time-of-flight neutron source, described as the corners of a
    polygon (initially a rectangle) in time and wavelength.
    """

    def __init__(self, time: sc.Variable, wavelength: sc.Variable):
        if time.sizes != wavelength.sizes:
            raise sc.DimensionError(
                f'Inconsistent dims or shape: {time.sizes} vs {wavelength.sizes}'
            )
        self.time = time.to(unit='s', copy=False)
        self.wavelength = wavelength.to(unit='angstrom', copy=False)

    def __eq__(self, other: object) -> bool:
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
    subframes: list[Subframe]

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
        if chopper.distance < self.distance:
            raise ValueError(
                f'Chopper distance {chopper.distance} is smaller than frame distance '
                f'{self.distance}'
            )
        frame = self.propagate_to(chopper.distance)

        # A chopper can have multiple openings, call _chop for each of them. The result
        # is the union of the resulting subframes.
        chopped = Frame(distance=frame.distance, subframes=[])
        for subframe in frame.subframes:
            for open, close in zip(chopper.time_open, chopper.time_close, strict=True):
                if (tmp := _chop(subframe, open, close_to_open=True)) is not None:
                    if (tmp := _chop(tmp, close, close_to_open=False)) is not None:
                        chopped.subframes.append(tmp)
        return chopped

    def bounds(self) -> sc.DataGroup:
        """The bounds of the frame, i.e., the global min and max time and wavelength."""
        start = sc.reduce([sub.start_time for sub in self.subframes]).min()
        end = sc.reduce([sub.end_time for sub in self.subframes]).max()
        wav_start = sc.reduce([sub.start_wavelength for sub in self.subframes]).min()
        wav_end = sc.reduce([sub.end_wavelength for sub in self.subframes]).max()
        return sc.DataGroup(
            time=sc.concat([start, end], dim='bound'),
            wavelength=sc.concat([wav_start, wav_end], dim='bound'),
        )

    def subbounds(self) -> sc.DataGroup:
        """
        The bounds of the subframes, defined as the union over subframes.

        This is not the same as the bounds of the individual subframes, but defined as
        the union of all subframes. Subframes that overlap in time are "merged" into a
        single subframe.

        This function is to some extent experimental: It is not clear if taking the
        union of overlapping subframes has any utility in practice, since this may
        simply indicate a problem with the chopper cascade. Attempts to handle this
        automatically may be misguided.
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

        @dataclass
        class Bound:
            start: sc.Variable
            end: sc.Variable
            wav_start: sc.Variable
            wav_end: sc.Variable

        bounds = [
            Bound(start, end, wav_start, wav_end)
            for start, end, wav_start, wav_end in zip(
                starts, ends, wav_starts, wav_ends, strict=True
            )
        ]
        bounds = sorted(bounds, key=lambda x: x.start)
        current = bounds[0]
        merged_bounds = []
        for bound in bounds[1:]:
            # If start is before current end, merge
            if bound.start <= current.end:
                current = Bound(
                    current.start,
                    max(current.end, bound.end),
                    current.wav_start,
                    max(current.wav_end, bound.wav_end),
                )
            else:
                merged_bounds.append(current)
                current = bound
        merged_bounds.append(current)
        time_bounds = [
            sc.concat([bound.start, bound.end], dim='bound') for bound in merged_bounds
        ]
        wav_bounds = [
            sc.concat([bound.wav_start, bound.wav_end], dim='bound')
            for bound in merged_bounds
        ]
        return sc.DataGroup(
            time=sc.concat(time_bounds, dim='subframe'),
            wavelength=sc.concat(wav_bounds, dim='subframe'),
        )


@dataclass
class FrameSequence:
    """
    A sequence of frames, created from a single neutron pulse, potentially chopped into
    subframes by choppers.

    It is recommended to use the :py:meth:`from_source_pulse` constructor to create a
    frame sequence from a source pulse. Then, a chopper cascade can be applied using
    :py:meth:`chop`.
    """

    frames: list[Frame]

    @staticmethod
    def from_source_pulse(
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
        frames = [
            Frame(
                distance=sc.scalar(0, unit='m'),
                subframes=[Subframe(time=time, wavelength=wavelength)],
            )
        ]
        return FrameSequence(frames)

    def __len__(self) -> int:
        """Number of frames."""
        return len(self.frames)

    def __getitem__(self, item: int | sc.Variable) -> Frame:
        """Get a frame by index or distance."""
        if isinstance(item, int):
            return self.frames[item]
        distance = item.to(unit='m')
        frame_before_detector = None
        for frame in self:
            if frame.distance > distance:
                break
            frame_before_detector = frame

        return frame_before_detector.propagate_to(distance)

    def propagate_to(self, distance: sc.Variable) -> FrameSequence:
        """
        Propagate the frame sequence to a distance, adding a new frame.

        Use this, e.g., to propagate to the sample position after applying choppers.

        Parameters
        ----------
        distance:
            Distance to propagate.

        Returns
        -------
        :
            New frame sequence.
        """
        return FrameSequence([*self.frames, self.frames[-1].propagate_to(distance)])

    def chop(self, choppers: list[Chopper]) -> FrameSequence:
        """
        Chop the frame sequence by a list of choppers.

        The choppers will be sorted by their distance, and applied in order.

        Parameters
        ----------
        choppers:
            List of choppers.

        Returns
        -------
        :
            New frame sequence.
        """
        choppers = sorted(choppers, key=lambda x: x.distance)
        frames = list(self.frames)
        for chopper in choppers:
            frames.append(frames[-1].chop(chopper))
        return FrameSequence(frames)

    def draw(
        self,
        linewidth: float = 0,
        fill: bool = True,
        alpha: float | None = None,
        transpose: bool = False,
        colors: list[str] | None = None,
        grid: bool = True,
        title: str = 'Frame propagation through chopper cascade',
        time_unit: str = 'ms',
        wavelength_unit: str = 'angstrom',
    ) -> Any:
        """
        Draw frames using matplotlib.

        Parameters
        ----------
        linewidth:
            Line width of frame edges.
        fill:
            Fill frame with color.
        alpha:
            Transparency of frame.
        transpose:
            Transpose axes.
        colors:
            List of colors to use for frames. If None, use default matplotlib colors.
        grid:
            Show grid.
        time_unit:
            Unit for time axis. Default is ms.
        wavelength_unit:
            Unit for wavelength axis. Default is angstrom.
        """
        import matplotlib.colors as mcolors
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x_max = 0
        y_max = 0
        for i, frame in enumerate(self.frames):
            color = colors[i] if colors is not None else f'C{i}'
            # Add label to legend
            ax.plot([], [], color=color, label=f'{frame.distance:c}')
            # All subframes have same color
            for subframe in frame.subframes:
                x = subframe.time.to(unit=time_unit, copy=False)
                y = subframe.wavelength.to(unit=wavelength_unit, copy=False)
                if transpose:
                    x, y = y, x
                x_unit = x.unit
                y_unit = y.unit
                x_max = max(x_max, x.max().value)
                y_max = max(y_max, y.max().value)
                if alpha:
                    color = mcolors.to_rgba(color, alpha=alpha)
                polygon = patches.Polygon(
                    np.stack((x.values, y.values), axis=1),
                    closed=True,
                    fill=fill,
                    color=color,
                    linewidth=linewidth,
                )
                ax.add_patch(polygon)
        ax.set_xlabel(x_unit)
        ax.set_ylabel(y_unit)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.minorticks_on()
        if grid:
            ax.grid(True, linestyle='-', linewidth='0.5', color='gray')
            ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        ax.legend(loc='best')
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax

    def acceptance_diagram(self):
        """
        Draw a chopper acceptance diagram.

        See, e.g., J.R.D. Copley, An acceptance diagram analysis of the contaminant
        pulse removal problem with direct geometry neutron chopper spectrometers,
        https://doi.org/10.1016/S0168-9002(03)01731-5 for more background.
        """
        import matplotlib.pyplot as plt

        source_distance = self.frames[0].distance
        frames = FrameSequence(
            [frame.propagate_to(source_distance) for frame in self.frames]
        )
        # Reset frame distance for plotting labels. This is a bit of a hack.
        # We should use chopper names if available.
        for i, frame in enumerate(frames.frames):
            frame.distance = self.frames[i].distance
        blue = np.linspace(0.1, 1, len(frames.frames))
        colors = plt.cm.Blues(blue)
        fig, ax = frames.draw(
            fill=True,
            linewidth=0.5,
            transpose=True,
            colors=colors,
            title='Chopper acceptance diagram',
        )
        # Put legend outside of plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            title='Distance',
            frameon=False,
        )
        return fig, ax


@dataclass
class Chopper:
    distance: sc.Variable
    time_open: sc.Variable
    time_close: sc.Variable

    def __post_init__(self):
        if self.time_open.sizes != self.time_close.sizes:
            raise sc.DimensionError(
                f'Inconsistent dims or shape: {self.time_open.sizes} vs '
                f'{self.time_close.sizes}'
            )

    def __getitem__(self, key) -> Chopper:
        return Chopper(
            distance=self.distance,
            time_open=self.time_open[key],
            time_close=self.time_close[key],
        )


def _chop(frame: Subframe, time: sc.Variable, close_to_open: bool) -> Subframe | None:
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
