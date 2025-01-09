# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
A fake time-of-flight neutron beamline for documentation and testing.

This provides data in a structure as typically provided in a NeXus file, including:

- Detector event data including event_time_offset and event_time_zero
- Monitor event data including event_time_offset and event_time_zero
- Chopper timestamps
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipp as sc
from numpy import random

from ..chopper import DiskChopper
from . import chopper_cascade


class FakeSource:
    def __init__(
        self,
        frequency: sc.Variable,
        run_length: sc.Variable,
        events_per_pulse: int = 10000,
    ):
        """
        Return a fake source.

        Parameters
        ----------
        frequency:
            Frequency of the source.
        run_length:
            Run length of the source.
        events_per_pulse:
            Number of events per pulse.
        """
        self.frequency = frequency
        self.t0 = self._make_t0(frequency=frequency, run_length=run_length)
        self.events_per_pulse = events_per_pulse
        self.rng = random.default_rng(seed=0)

    @property
    def pulse_period(self) -> sc.Variable:
        return sc.scalar(1.0) / self.frequency

    @property
    def number_of_pulses(self) -> int:
        return len(self.t0)

    def _make_t0(self, frequency: sc.Variable, run_length: sc.Variable) -> sc.Variable:
        start = sc.datetime('2019-12-25T06:00:00.0', unit='ns')
        npulse = (run_length * frequency).to(unit='').value
        pulses = sc.arange(dim='pulse', start=0, stop=npulse, dtype='int64')
        return start + (pulses * (1.0 / frequency)).to(dtype='int64', unit='ns')


class FakePulse:
    """
    Simplified model of a pulse.

    Currently this is simply a time and wavelength interval. The plan is to also model
    a tail in the future, e.g., by overlaying multiple pulses.
    """

    def __init__(
        self,
        time_min: sc.Variable,
        time_max: sc.Variable,
        wavelength_min: sc.Variable,
        wavelength_max: sc.Variable,
    ):
        self.time_min = time_min
        self.time_max = time_max
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self._center = sc.scalar(0.5) * (time_min + time_max)
        self._frames = chopper_cascade.FrameSequence.from_source_pulse(
            time_min=time_min,
            time_max=time_max,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )

    @property
    def center(self) -> sc.Variable:
        return self._center

    def chop(
        self, choppers: list[chopper_cascade.Chopper]
    ) -> chopper_cascade.FrameSequence:
        return self._frames.chop(choppers)


class FakeBeamline:
    def __init__(
        self,
        source: FakeSource,
        pulse: FakePulse,
        choppers: dict[str, chopper_cascade.Chopper],
        monitors: dict[str, sc.Variable],
        detectors: dict[str, sc.Variable],
        time_of_flight_origin: str | None = None,
    ):
        """
        Return a fake beamline.

        Parameters
        ----------
        source:
            Fake source.
        pulse:
            Fake pulse.
        choppers:
            Choppers.
        monitors:
            Distances of monitors.
        detectors:
            Distances of detectors.
        time_of_flight_origin:
            Name of the chopper to use as time-of-flight origin. If None, use the
            source pulse. The center time of the slit opening of the chopper (or the
            source pulse) is used as time-of-flight origin.
        """
        self._frames = pulse.chop(choppers.values())
        self._pulse = pulse
        self._choppers = choppers
        self._source = source
        self._monitors = {
            key: self._frames[distance] for key, distance in monitors.items()
        }
        self.detectors = {
            key: self._frames[distance] for key, distance in detectors.items()
        }
        self._time_of_flight_origin = time_of_flight_origin

    def get_monitor(self, name: str) -> sc.DataGroup:
        frame = self._monitors[name]
        return self._fake_monitor(frame)

    def _split_size(self, size, N):
        base, remainder = divmod(size, N)
        sizes = [base + 1 if i < remainder else base for i in range(N)]
        return sizes

    def _fake_monitor(
        self, frame: chopper_cascade.Frame
    ) -> tuple[sc.DataArray, sc.DataArray]:
        bounds = frame.bounds()['time']
        subbounds = frame.subbounds()['time']
        subframes = subbounds.sizes['subframe']

        sizes = sc.array(
            dims=['pulse'],
            values=self._source.rng.integers(
                0, self._source.events_per_pulse, size=self._source.number_of_pulses
            ),
            unit=None,
        )
        event_index = sc.cumsum(sizes, dim='pulse', mode='exclusive')
        size = sizes.sum().value
        subsizes = self._split_size(size, subframes)
        subframe_times = [
            sc.array(
                dims=['event'],
                values=self._source.rng.uniform(
                    subbounds['subframe', i][0].value,
                    subbounds['subframe', i][-1].value,
                    size=subsizes[i],
                ),
                unit=bounds.unit,
            )
            for i in range(subframes)
        ]

        # Offset from pulse that created the monitor event
        time_offset = sc.concat(subframe_times, 'event')
        # Ensure all pulses have events from all subframes
        self._source.rng.shuffle(time_offset.values)
        event_time_offset = time_offset % self._source.pulse_period
        time_zero_offset = time_offset - event_time_offset
        event_time_zero = self._source.t0
        wrapped_events = sc.DataArray(
            sc.ones(sizes=event_time_offset.sizes, unit='counts'),
            coords={'event_time_offset': event_time_offset},
        )
        unwrapped_events = sc.DataArray(
            sc.ones(sizes=time_offset.sizes, unit='counts'),
            coords={
                'time_offset': time_offset,
                'time_zero_offset': time_zero_offset.to(dtype='int64', unit='ns'),
            },
        )
        wrapped = sc.DataArray(
            data=sc.bins(begin=event_index, dim='event', data=wrapped_events),
            coords={'event_time_zero': event_time_zero},
        )
        unwrapped = sc.DataArray(
            data=sc.bins(begin=event_index, dim='event', data=unwrapped_events),
            coords={'event_time_zero': event_time_zero},
        )
        if self._time_of_flight_origin is None:
            offset_to_tof = self._pulse.center
        else:
            source_chopper = self._choppers[self._time_of_flight_origin]
            if len(source_chopper.time_open) != 1:
                raise NotImplementedError(
                    "Using a chopper with multiple openings as "
                    "source chopper is not implemented yet."
                )
            offset_to_tof = 0.5 * (
                source_chopper.time_open[0] + source_chopper.time_close[0]
            )
        unwrapped = unwrapped.transform_coords(
            tof=lambda time_offset: time_offset - offset_to_tof.to(unit='s'),
            time_zero=lambda event_time_zero, time_zero_offset: event_time_zero
            + time_zero_offset.to(dtype='int64', unit='ns')
            + offset_to_tof.to(dtype='int64', unit='ns'),
        )
        if self._time_of_flight_origin is None:
            unwrapped.coords['Ltotal'] = frame.distance
        else:
            unwrapped.coords['Ltotal'] = frame.distance - source_chopper.distance
        return wrapped, unwrapped


class FakeBeamlineEss:
    def __init__(
        self,
        choppers: dict[str, DiskChopper],
        monitors: dict[str, sc.Variable],
        run_length: sc.Variable,
        events_per_pulse: int = 200000,
        source: Callable | None = None,
    ):
        import math

        import tof as tof_pkg
        from tof.facilities.ess_pulse import pulse

        self.frequency = pulse.frequency
        self.npulses = math.ceil((run_length * self.frequency).to(unit='').value)
        self.events_per_pulse = events_per_pulse

        # Create a source
        if source is None:
            self.source = tof_pkg.Source(
                facility='ess', neutrons=self.events_per_pulse, pulses=self.npulses
            )
        else:
            self.source = source(pulses=self.npulses)

        # # Convert the choppers to tof.Chopper
        # def _open_close_angles(chopper, frequency):
        #     angular_speed = sc.constants.pi * (2.0 * sc.units.rad) * frequency
        #     return (
        #         chopper.time_open * angular_speed,
        #         chopper.time_close * angular_speed,
        #     )

        # self.choppers = []
        # for name, ch in choppers.items():
        #     frequency = self.frequency
        #     open_angles, close_angles = _open_close_angles(ch, frequency)
        #     # If the difference between open and close angles is larger than 2pi,
        #     # the boundaries have crossed, which means that the chopper is rotating
        #     # at a lower frequency.
        #     two_pi = np.pi * 2
        #     if any(abs(np.diff(open_angles.values) > two_pi)) or any(
        #         abs(np.diff(close_angles.values) > two_pi)
        #     ):
        #         frequency = 0.5 * frequency
        #         open_angles, close_angles = _open_close_angles(ch, frequency)
        #     self.choppers.append(
        #         tof_pkg.Chopper(
        #             frequency=frequency,
        #             open=open_angles,
        #             close=close_angles,
        #             phase=sc.scalar(0.0, unit='rad'),
        #             distance=ch.distance,
        #             name=name,
        #         )
        #     )

        self.choppers = [
            tof_pkg.Chopper(
                frequency=abs(ch.frequency),
                direction=tof_pkg.AntiClockwise
                if (ch.frequency.value > 0.0)
                else tof_pkg.Clockwise,
                open=ch.slit_begin,
                close=ch.slit_end,
                phase=abs(ch.phase),
                distance=ch.axle_position.fields.z,
                name=name,
            )
            for name, ch in choppers.items()
        ]

        # Add detectors
        self.monitors = [
            tof_pkg.Detector(distance=distance, name=key)
            for key, distance in monitors.items()
        ]

        #  Propagate the neutrons
        self.model = tof_pkg.Model(
            source=self.source, choppers=self.choppers, detectors=self.monitors
        )
        self.model_result = self.model.run()

    def get_monitor(self, name: str) -> sc.DataGroup:
        # Create some fake pulse time zero
        start = sc.datetime("2024-01-01T12:00:00.000000")
        period = sc.reciprocal(self.frequency)

        detector = self.model_result.detectors[name]
        raw_data = detector.data.flatten(to='event')
        # Select only the neutrons that make it to the detector
        raw_data = raw_data[~raw_data.masks['blocked_by_others']].copy()
        raw_data.coords['Ltotal'] = detector.distance

        # Format the data in a way that resembles data loaded from NeXus
        event_data = raw_data.copy(deep=False)
        dt = period.to(unit='us')
        event_time_zero = (dt * (event_data.coords['toa'] // dt)).to(dtype=int) + start
        raw_data.coords['event_time_zero'] = event_time_zero
        event_data.coords['event_time_zero'] = event_time_zero
        event_data.coords['event_time_offset'] = (
            event_data.coords.pop('toa').to(unit='s') % period
        )
        del event_data.coords['tof']
        del event_data.coords['speed']
        del event_data.coords['time']
        del event_data.coords['wavelength']

        return (
            event_data.group('event_time_zero').rename_dims(event_time_zero='pulse'),
            raw_data.group('event_time_zero').rename_dims(event_time_zero='pulse'),
        )


wfm1 = chopper_cascade.Chopper(
    distance=sc.scalar(6.6, unit='m'),
    time_open=sc.array(
        dims=('cutout',),
        values=[
            -0.000396,
            0.001286,
            0.005786,
            0.008039,
            0.010133,
            0.012080,
            0.013889,
            0.015571,
        ],
        unit='s',
    ),
    time_close=sc.array(
        dims=('cutout',),
        values=[
            0.000654,
            0.002464,
            0.006222,
            0.008646,
            0.010899,
            0.012993,
            0.014939,
            0.016750,
        ],
        unit='s',
    ),
)
wfm2 = chopper_cascade.Chopper(
    distance=sc.scalar(7.1, unit='m'),
    time_open=sc.array(
        dims=('cutout',),
        values=[
            0.000654,
            0.002451,
            0.006222,
            0.008645,
            0.010898,
            0.012993,
            0.014940,
            0.016737,
        ],
        unit='s',
    ),
    time_close=sc.array(
        dims=('cutout',),
        values=[
            0.001567,
            0.003641,
            0.006658,
            0.009252,
            0.011664,
            0.013759,
            0.015853,
            0.017927,
        ],
        unit='s',
    ),
)
# psc1 and psc2 are defined as a single slit of wfm1 and wfm2, respectively.
psc1 = chopper_cascade.Chopper(
    distance=sc.scalar(6.6, unit='m'),
    time_open=sc.array(dims=('cutout',), values=[0.010133], unit='s'),
    time_close=sc.array(dims=('cutout',), values=[0.010899], unit='s'),
)
psc2 = chopper_cascade.Chopper(
    distance=sc.scalar(7.1, unit='m'),
    time_open=sc.array(dims=('cutout',), values=[0.010898], unit='s'),
    time_close=sc.array(dims=('cutout',), values=[0.011664], unit='s'),
)
frame_overlap_1 = chopper_cascade.Chopper(
    distance=sc.scalar(8.8, unit='m'),
    time_open=sc.array(
        dims=('cutout',),
        values=[
            -0.000139,
            0.002460,
            0.006796,
            0.010020,
            0.012733,
            0.015263,
            0.017718,
            0.020317,
        ],
        unit='s',
    ),
    time_close=sc.array(
        dims=('cutout',),
        values=[
            0.000640,
            0.003671,
            0.007817,
            0.011171,
            0.013814,
            0.016146,
            0.018497,
            0.021528,
        ],
        unit='s',
    ),
)
frame_overlap_2 = chopper_cascade.Chopper(
    distance=sc.scalar(15.9, unit='m'),
    time_open=sc.array(
        dims=('cutout',),
        values=[
            -0.000306,
            0.010939,
            0.016495,
            0.021733,
            0.026416,
            0.030880,
            0.035409,
        ],
        unit='s',
    ),
    time_close=sc.array(
        dims=('cutout',),
        values=[
            0.002582,
            0.014570,
            0.020072,
            0.024730,
            0.029082,
            0.033316,
            0.038297,
        ],
        unit='s',
    ),
)

pulse_skipping = chopper_cascade.Chopper(
    distance=sc.scalar(15.91, unit='m'),
    time_open=sc.scalar(0.021733, unit='s')
    + sc.arange('cutout', 2, unit='s') * (2 / 14),
    time_close=sc.scalar(0.024730, unit='s')
    + sc.arange('cutout', 2, unit='s') * (2 / 14),
)

wfm_choppers = sc.DataGroup(
    wfm1=wfm1,
    wfm2=wfm2,
    frame_overlap_1=frame_overlap_1,
    frame_overlap_2=frame_overlap_2,
)
psc_choppers = sc.DataGroup(
    psc1=psc1,
    psc2=psc2,
    frame_overlap_1=frame_overlap_1,
    frame_overlap_2=frame_overlap_2,
)


ess_time_min = sc.scalar(0.0, unit='ms')
ess_time_max = sc.scalar(3.0, unit='ms')
ess_wavelength_min = sc.scalar(0.0, unit='angstrom')
ess_wavelength_max = sc.scalar(10.0, unit='angstrom')

wfm_frames = chopper_cascade.FrameSequence.from_source_pulse(
    time_min=ess_time_min,
    time_max=ess_time_max,
    wavelength_min=ess_wavelength_min,
    wavelength_max=ess_wavelength_max,
)
wfm_frames = wfm_frames.chop(wfm_choppers.values())

psc_frames = chopper_cascade.FrameSequence.from_source_pulse(
    time_min=ess_time_min,
    time_max=ess_time_max,
    wavelength_min=ess_wavelength_min,
    wavelength_max=ess_wavelength_max,
)
psc_frames = psc_frames.chop(psc_choppers.values())


wfm1_disk_chopper = DiskChopper(
    frequency=sc.scalar(-70.0, unit="Hz"),
    beam_position=sc.scalar(0.0, unit="deg"),
    phase=sc.scalar(-47.10, unit="deg"),
    axle_position=sc.vector(value=[0, 0, 6.6], unit="m"),
    slit_begin=sc.array(
        dims=["cutout"],
        values=np.array([83.71, 140.49, 193.26, 242.32, 287.91, 330.3]) + 15.0,
        unit="deg",
    ),
    slit_end=sc.array(
        dims=["cutout"],
        values=np.array([94.7, 155.79, 212.56, 265.33, 314.37, 360.0]) + 15.0,
        unit="deg",
    ),
    slit_height=sc.scalar(10.0, unit="cm"),
    radius=sc.scalar(30.0, unit="cm"),
)

wfm2_disk_chopper = DiskChopper(
    frequency=sc.scalar(-70.0, unit="Hz"),
    beam_position=sc.scalar(0.0, unit="deg"),
    phase=sc.scalar(-76.76, unit="deg"),
    axle_position=sc.vector(value=[0, 0, 7.1], unit="m"),
    slit_begin=sc.array(
        dims=["cutout"],
        values=np.array([65.04, 126.1, 182.88, 235.67, 284.73, 330.32]) + 15.0,
        unit="deg",
    ),
    slit_end=sc.array(
        dims=["cutout"],
        values=np.array([76.03, 141.4, 202.18, 254.97, 307.74, 360.0]) + 15.0,
        unit="deg",
    ),
    slit_height=sc.scalar(10.0, unit="cm"),
    radius=sc.scalar(30.0, unit="cm"),
)

foc1_disk_chopper = DiskChopper(
    frequency=sc.scalar(-56.0, unit="Hz"),
    beam_position=sc.scalar(0.0, unit="deg"),
    phase=sc.scalar(-62.40, unit="deg"),
    axle_position=sc.vector(value=[0, 0, 8.8], unit="m"),
    slit_begin=sc.array(
        dims=["cutout"],
        values=np.array([74.6, 139.6, 194.3, 245.3, 294.8, 347.2]),
        unit="deg",
    ),
    slit_end=sc.array(
        dims=["cutout"],
        values=np.array([95.2, 162.8, 216.1, 263.1, 310.5, 371.6]),
        unit="deg",
    ),
    slit_height=sc.scalar(10.0, unit="cm"),
    radius=sc.scalar(30.0, unit="cm"),
)

foc2_disk_chopper = DiskChopper(
    frequency=sc.scalar(-28.0, unit="Hz"),
    beam_position=sc.scalar(0.0, unit="deg"),
    phase=sc.scalar(-12.27, unit="deg"),
    axle_position=sc.vector(value=[0, 0, 15.9], unit="m"),
    slit_begin=sc.array(
        dims=["cutout"],
        values=np.array([98.0, 154.0, 206.8, 255.0, 299.0, 344.65]),
        unit="deg",
    ),
    slit_end=sc.array(
        dims=["cutout"],
        values=np.array([134.6, 190.06, 237.01, 280.88, 323.56, 373.76]),
        unit="deg",
    ),
    slit_height=sc.scalar(10.0, unit="cm"),
    radius=sc.scalar(30.0, unit="cm"),
)

pol_disk_chopper = DiskChopper(
    frequency=sc.scalar(-14.0, unit="Hz"),
    beam_position=sc.scalar(0.0, unit="deg"),
    phase=sc.scalar(0.0, unit="deg"),
    axle_position=sc.vector(value=[0, 0, 17.0], unit="m"),
    slit_begin=sc.array(
        dims=["cutout"],
        values=np.array([40.0]),
        unit="deg",
    ),
    slit_end=sc.array(
        dims=["cutout"],
        values=np.array([240.0]),
        unit="deg",
    ),
    slit_height=sc.scalar(10.0, unit="cm"),
    radius=sc.scalar(30.0, unit="cm"),
)

wfm_disk_choppers = {
    "wfm1": wfm1_disk_chopper,
    "wfm2": wfm2_disk_chopper,
    "foc1": foc1_disk_chopper,
    "foc2": foc2_disk_chopper,
    "pol": pol_disk_chopper,
}

psc_disk_choppers = {
    name: DiskChopper(
        frequency=ch.frequency,
        beam_position=ch.beam_position,
        phase=ch.phase,
        axle_position=ch.axle_position,
        slit_begin=ch.slit_begin[0:1],
        slit_end=ch.slit_end[0:1],
        slit_height=ch.slit_height[0:1],
        radius=ch.radius,
    )
    for name, ch in wfm_disk_choppers.items()
}
