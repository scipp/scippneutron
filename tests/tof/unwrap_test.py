# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
import scipp as sc
import sciline as sl
from scipp.testing import assert_identical

from scippneutron.tof import chopper_cascade, fakes, unwrap


@pytest.fixture
def ess_10s_14Hz() -> fakes.FakeSource:
    return fakes.FakeSource(
        frequency=sc.scalar(14.0, unit='Hz'), run_length=sc.scalar(10.0, unit='s')
    )


def test_standard_unwrap(ess_10s_14Hz) -> None:
    time_min = sc.scalar(0.0, unit='ms')
    time_max = sc.scalar(3.0, unit='ms')
    wavelength_min = sc.scalar(0.1, unit='angstrom')
    wavelength_max = sc.scalar(10.0, unit='angstrom')
    frames = chopper_cascade.FrameSequence.from_source_pulse(
        time_min=time_min,
        time_max=time_max,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
    )
    frames = fakes.psc_frames
    distance = sc.scalar(46.0, unit='m')
    beamline = fakes.FakeBeamline(
        source=ess_10s_14Hz,
        frames=frames,
        monitors={'source_monitor': distance},
        detectors={},
    )
    mon = beamline.get_monitor('source_monitor')
    mon.hist(event_time_offset=1000).sum('pulse').plot().save("raw.png")

    pl = sl.Pipeline(unwrap.providers())
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = beamline.source.pulse_period
    pl[unwrap.SourceTimeRange] = time_min, time_max
    pl[unwrap.SourceWavelengthRange] = wavelength_min, wavelength_max
    pl[unwrap.Choppers] = fakes.psc_choppers
    pl[unwrap.SourceChopperName] = 'psc1'
    pl[unwrap.Ltotal] = distance
    result = pl.compute(unwrap.TofData)
    result.hist(tof=1000).sum('pulse').plot().save("unwrapped.png")
