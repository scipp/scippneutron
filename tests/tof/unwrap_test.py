# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
import sciline as sl
import scipp as sc
from scipp.testing import assert_identical

from scippneutron.tof import fakes, unwrap


@pytest.fixture
def ess_10s_14Hz() -> fakes.FakeSource:
    return fakes.FakeSource(
        frequency=sc.scalar(14.0, unit='Hz'), run_length=sc.scalar(10.0, unit='s')
    )


@pytest.fixture
def ess_pulse() -> fakes.FakePulse:
    return fakes.FakePulse(
        time_min=sc.scalar(0.0, unit='ms'),
        time_max=sc.scalar(3.0, unit='ms'),
        wavelength_min=sc.scalar(0.1, unit='angstrom'),
        wavelength_max=sc.scalar(10.0, unit='angstrom'),
    )


def test_unwrap_with_no_choppers(ess_10s_14Hz, ess_pulse) -> None:
    # At this small distance the frames are not overlapping (with the given wavelength
    # range), despite not using any choppers.
    distance = sc.scalar(10.0, unit='m')
    beamline = fakes.FakeBeamline(
        source=ess_10s_14Hz,
        pulse=ess_pulse,
        choppers={},  # no choppers
        monitors={'monitor': distance},
        detectors={},
    )
    mon, ref = beamline.get_monitor('monitor')

    pl = sl.Pipeline(unwrap.providers())
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = {}
    pl[unwrap.Ltotal] = distance
    result = pl.compute(unwrap.TofData)
    assert_identical(
        result.hist(tof=1000).sum('pulse'), ref.hist(tof=1000).sum('pulse')
    )


def test_unwrap_with_frame_overlap_raises(ess_10s_14Hz, ess_pulse) -> None:
    distance = sc.scalar(46.0, unit='m')
    beamline = fakes.FakeBeamline(
        source=ess_10s_14Hz,
        pulse=ess_pulse,
        choppers={},  # no choppers
        monitors={'monitor': distance},
        detectors={},
    )
    mon, _ = beamline.get_monitor('monitor')

    pl = sl.Pipeline(unwrap.providers())
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = {}
    pl[unwrap.Ltotal] = distance
    with pytest.raises(ValueError, match='Frames are overlapping'):
        pl.compute(unwrap.TofData)


def test_standard_unwrap(ess_10s_14Hz, ess_pulse) -> None:
    distance = sc.scalar(46.0, unit='m')
    beamline = fakes.FakeBeamline(
        source=ess_10s_14Hz,
        pulse=ess_pulse,
        choppers=fakes.psc_choppers,
        monitors={'monitor': distance},
        detectors={},
        time_of_flight_origin='psc1',
    )
    mon, ref = beamline.get_monitor('monitor')

    pl = sl.Pipeline(unwrap.providers())
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = fakes.psc_choppers
    pl[unwrap.SourceChopperName] = 'psc1'
    pl[unwrap.Ltotal] = distance
    result = pl.compute(unwrap.TofData)
    assert_identical(
        result.hist(tof=1000).sum('pulse'), ref.hist(tof=1000).sum('pulse')
    )


def test_standard_unwrap_histogram_mode(ess_10s_14Hz, ess_pulse) -> None:
    distance = sc.scalar(46.0, unit='m')
    beamline = fakes.FakeBeamline(
        source=ess_10s_14Hz,
        pulse=ess_pulse,
        choppers=fakes.psc_choppers,
        monitors={'monitor': distance},
        detectors={},
        time_of_flight_origin='psc1',
    )
    mon, ref = beamline.get_monitor('monitor')
    mon = (
        mon.hist(
            event_time_offset=sc.linspace(
                'event_time_offset', 0.0, 1000.0 / 14, num=1001, unit='ms'
            ).to(unit='s')
        )
        .sum('pulse')
        .rename(event_time_offset='time_of_flight')
    )

    pl = sl.Pipeline(unwrap.providers())
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = fakes.psc_choppers
    pl[unwrap.SourceChopperName] = 'psc1'
    pl[unwrap.Ltotal] = distance
    result = pl.compute(unwrap.TofData)
    assert_identical(result.sum(), ref.sum())
