# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
"""
A fake time-of-flight neutron beamline for documentation and testing.

This provides data in a structure as typically provided in a NeXus file, including:

- Detector event data including event_time_offset and event_time_zero
- Monitor event data including event_time_offset and event_time_zero
- Monitor time_of_flight histogram data
- Chopper timestamps
"""
from __future__ import annotations
from numpy import random

import scipp as sc
from . import chopper_cascade

# TODO
# - background
# - how will tests verify they got the correct output? provide unwrapped as reference?
#   also wavelength range, ...
#   Yes: Set ranges and params when creating fake, fake will create raw source info


class FakeBackground:
    pass


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
        duration:
            Duration of the source.
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


class FakeBeamline:
    def __init__(
        self,
        source: FakeSource,
        frames: chopper_cascade.FrameSequence,
        monitors: dict[str, sc.Variable],
        detectors: dict[str, sc.Variable],
        # TODO define source chopper, so we can compute TOF?
    ):
        self.source = source
        self.monitors = {key: frames[distance] for key, distance in monitors.items()}
        self.detectors = {key: frames[distance] for key, distance in detectors.items()}
        # We propagate the source pulse, and compute time bounds at monitor and
        # detector positions.
        # Model the tail by using multiple source pulses and overlaying them?
        # Overall algorithms:
        # - define collection of source frames
        # - propagate to next component using chopper_cascade.FrameSequence
        #   - generate neutrons in time-bounds if component is a monitor or detector
        #   - map generated neutrons into frames
        # - iterate

    def get_monitor(self, name: str) -> sc.DataGroup:
        frame = self.monitors[name]
        return self._fake_monitor(frame)

    def _split_size(self, size, N):
        base, remainder = divmod(size, N)
        sizes = [base + 1 if i < remainder else base for i in range(N)]
        return sizes

    def _fake_monitor(self, frame: chopper_cascade.Frame) -> sc.DataArray:
        # TODO define TOF reference point, create expected result before wrapping
        # how to handle WFM?
        # need some TOF reference points as when unwrapping, apply reverse?
        # TODO generate with some shape?
        bounds = frame.bounds()['time']
        subbounds = frame.subbounds()['time']
        subframes = subbounds.sizes['subframe']

        sizes = sc.array(
            dims=['pulse'],
            values=self.source.rng.integers(
                0, self.source.events_per_pulse, size=self.source.number_of_pulses
            ),
            unit=None,
        )
        event_index = sc.cumsum(sizes, dim='pulse', mode='exclusive')
        size = sizes.sum().value
        subsizes = self._split_size(size, subframes)
        subframe_times = []
        for i in range(subframes):
            subframe_times.append(
                sc.array(
                    dims=['event'],
                    values=self.source.rng.uniform(
                        subbounds['subframe', i][0].value,
                        subbounds['subframe', i][-1].value,
                        size=subsizes[i],
                    ),
                    unit=bounds.unit,
                )
            )

        # Offset from pulse that created the monitor event
        time_offset = sc.concat(subframe_times, 'event')
        # Ensure all pulses have events from all subframes
        self.source.rng.shuffle(time_offset.values)
        event_time_offset = time_offset % self.source.pulse_period
        time_zero_offset = time_offset - event_time_offset
        event_time_zero = self.source.t0
        events = sc.DataArray(
            sc.ones(sizes=event_time_offset.sizes, unit='counts'),
            coords={'event_time_offset': event_time_offset},
        )
        binned = sc.DataArray(
            data=sc.bins(begin=event_index, dim='event', data=events),
            coords={'event_time_zero': event_time_zero},
        )
        return binned


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
