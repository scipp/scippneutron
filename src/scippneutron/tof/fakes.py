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

import scipp as sc
from numpy import random

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
pulse_overlap = chopper_cascade.Chopper(
    distance=sc.scalar(22.0, unit='m'),
    time_open=sc.array(dims=('cutout',), values=[-0.130952, 0.011905], unit='s'),
    time_close=sc.array(dims=('cutout',), values=[-0.087302, 0.055556], unit='s'),
)

wfm_choppers = sc.DataGroup(
    wfm1=wfm1,
    wfm2=wfm2,
    frame_overlap_1=frame_overlap_1,
    frame_overlap_2=frame_overlap_2,
    pulse_overlap=pulse_overlap,
)
psc_choppers = sc.DataGroup(
    psc1=psc1,
    psc2=psc2,
    frame_overlap_1=frame_overlap_1,
    frame_overlap_2=frame_overlap_2,
    pulse_overlap=pulse_overlap,
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
