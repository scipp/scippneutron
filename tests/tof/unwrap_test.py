# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import pytest
import scipp as sc
from scipp.testing import assert_identical

from scippneutron.tof import fakes, unwrap

sl = pytest.importorskip('sciline')


@pytest.fixture()
def ess_10s_14Hz() -> fakes.FakeSource:
    return fakes.FakeSource(
        frequency=sc.scalar(14.0, unit='Hz'), run_length=sc.scalar(10.0, unit='s')
    )


@pytest.fixture()
def ess_10s_7Hz() -> fakes.FakeSource:
    return fakes.FakeSource(
        frequency=sc.scalar(7.0, unit='Hz'), run_length=sc.scalar(10.0, unit='s')
    )


@pytest.fixture()
def ess_pulse() -> fakes.FakePulse:
    return fakes.FakePulse(
        time_min=sc.scalar(0.0, unit='ms'),
        time_max=sc.scalar(3.0, unit='ms'),
        wavelength_min=sc.scalar(0.1, unit='angstrom'),
        wavelength_max=sc.scalar(10.0, unit='angstrom'),
    )


def test_frame_period_is_pulse_period_if_not_pulse_skipping() -> None:
    pl = sl.Pipeline(unwrap.unwrap_providers())
    period = sc.scalar(123.0, unit='ms')
    pl[unwrap.PulsePeriod] = period
    assert_identical(pl.compute(unwrap.FramePeriod), period)


@pytest.mark.parametrize('stride', [1, 2, 3, 4])
def test_frame_period_is_multiple_pulse_period_if_pulse_skipping(stride) -> None:
    pl = sl.Pipeline(unwrap.unwrap_providers())
    period = sc.scalar(123.0, unit='ms')
    pl[unwrap.PulsePeriod] = period
    pl[unwrap.PulseStride] = stride
    assert_identical(pl.compute(unwrap.FramePeriod), stride * period)


@pytest.mark.parametrize('stride', [1, 2, 3, 4])
def test_pulse_offset(stride) -> None:
    pl = sl.Pipeline(unwrap.unwrap_providers(pulse_skipping=True))
    period = sc.scalar(123.0, unit='ms')
    pl[unwrap.PulsePeriod] = period
    pl[unwrap.PulseStride] = stride
    start = sc.datetime('2020-01-01T00:00:00.0', unit='ns')
    time_zero = start + sc.arange('pulse', 0 * period, 100 * period, period).to(
        unit='ns', dtype='int64'
    )
    pl[unwrap.TimeZero] = time_zero

    result = pl.compute(unwrap.PulseOffset)
    assert_identical(
        result, (sc.arange('pulse', 0, 100, dtype='int64') % stride) * period
    )


def test_offset_from_wrapped() -> None:
    pl = sl.Pipeline(unwrap.unwrap_providers())
    period = sc.scalar(123.0, unit='ms')
    pl[unwrap.PulsePeriod] = period
    pl[unwrap.FrameBounds] = unwrap.FrameBounds(
        sc.DataGroup(time=sc.array(dims=['bound'], values=[0.01, 0.02], unit='s'))
    )
    wrapped_offset = sc.linspace('event', 0.0, 123.0, num=1001, unit='ms')
    pl[unwrap.PulseWrappedTimeOffset] = unwrap.PulseWrappedTimeOffset(wrapped_offset)
    offset = pl.compute(unwrap.DeltaFromWrapped)
    # Times below 10 ms (we currently cut at lower bound) should be offset by period.
    da = sc.DataArray(offset, coords={'time': wrapped_offset})
    assert sc.all(da['time', : 10 * sc.Unit('ms')].data == period.to(unit='s'))
    assert sc.all(da['time', 10 * sc.Unit('ms') :].data == sc.scalar(0.0, unit='s'))


def test_offset_from_wrapped_has_no_special_handling_for_out_of_period_events() -> None:
    pl = sl.Pipeline(unwrap.unwrap_providers())
    period = sc.scalar(123.0, unit='ms')
    pl[unwrap.PulsePeriod] = period
    pl[unwrap.FrameBounds] = unwrap.FrameBounds(
        sc.DataGroup(time=sc.array(dims=['bound'], values=[0.01, 0.02], unit='s'))
    )
    wrapped_offset = sc.linspace('event', -10000.0, 10000.0, num=10001, unit='ms')
    pl[unwrap.PulseWrappedTimeOffset] = unwrap.PulseWrappedTimeOffset(wrapped_offset)
    offset = pl.compute(unwrap.DeltaFromWrapped)
    da = sc.DataArray(offset, coords={'time': wrapped_offset})
    # Negative times and times > 123 ms are technically invalid, but it does not affect
    # unwrapping, so they should be left as-is.
    assert sc.all(da['time', : 10 * sc.Unit('ms')].data == period.to(unit='s'))
    assert sc.all(da['time', 10 * sc.Unit('ms') :].data == sc.scalar(0.0, unit='s'))


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

    pl = sl.Pipeline(
        unwrap.unwrap_providers()
        + unwrap.time_of_flight_providers()
        + unwrap.time_of_flight_origin_from_choppers_providers()
    )
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = {}
    pl[unwrap.Ltotal] = distance
    unwrapped = pl.compute(unwrap.UnwrappedData)
    # No unwrap is happening, frame does not overlap next pulse.
    assert (mon.coords['event_time_zero'] == unwrapped.bins.coords['pulse_time']).all()

    origin = pl.compute(unwrap.TimeOfFlightOrigin)
    assert_identical(origin.time, sc.scalar(0.0015, unit='s'))
    assert_identical(origin.distance, sc.scalar(0.0, unit='m'))

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

    pl = sl.Pipeline(unwrap.unwrap_providers())
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
        pl.compute(unwrap.UnwrappedData)


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

    pl = sl.Pipeline(
        unwrap.unwrap_providers()
        + unwrap.time_of_flight_providers()
        + unwrap.time_of_flight_origin_from_choppers_providers()
    )
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

    pl = sl.Pipeline(
        unwrap.unwrap_providers()
        + unwrap.time_of_flight_providers()
        + unwrap.time_of_flight_origin_from_choppers_providers()
    )
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


def test_pulse_skipping_unwrap(ess_10s_7Hz, ess_pulse) -> None:
    # Pretend pulse-skipping by using source of different frequency. In reality this
    # would be done using a chopper.
    distance = sc.scalar(46.0, unit='m')
    beamline = fakes.FakeBeamline(
        source=ess_10s_7Hz,
        pulse=ess_pulse,
        choppers=fakes.psc_choppers,
        monitors={'monitor': distance},
        detectors={},
        time_of_flight_origin='psc1',
    )
    mon, ref = beamline.get_monitor('monitor')

    pl = sl.Pipeline(
        unwrap.unwrap_providers(pulse_skipping=True)
        + unwrap.time_of_flight_providers()
        + unwrap.time_of_flight_origin_from_choppers_providers()
    )
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = 0.5 * beamline._source.pulse_period
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )

    pl[unwrap.Choppers] = fakes.psc_choppers
    pl[unwrap.SourceChopperName] = 'psc1'
    pl[unwrap.Ltotal] = distance
    pl[unwrap.PulseStride] = 2

    result = pl.compute(unwrap.TofData)
    assert_identical(result.sum(), ref.sum())
    assert_identical(
        result.hist(tof=1000).sum('pulse'), ref.hist(tof=1000).sum('pulse')
    )


def test_pulse_skipping_unwrap_histogram_mode_not_implemented(
    ess_10s_7Hz, ess_pulse
) -> None:
    distance = sc.scalar(46.0, unit='m')
    beamline = fakes.FakeBeamline(
        source=ess_10s_7Hz,
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

    pl = sl.Pipeline(
        unwrap.unwrap_providers(pulse_skipping=True)
        + unwrap.time_of_flight_providers()
        + unwrap.time_of_flight_origin_from_choppers_providers()
    )
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = 0.5 * beamline._source.pulse_period
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = fakes.psc_choppers
    pl[unwrap.SourceChopperName] = 'psc1'
    pl[unwrap.Ltotal] = distance
    pl[unwrap.PulseStride] = 2
    with pytest.raises(NotImplementedError):
        pl.compute(unwrap.TofData)


def test_wfm_unwrap(ess_10s_14Hz, ess_pulse) -> None:
    distance = sc.scalar(20.0, unit='m')
    choppers = fakes.wfm_choppers.copy()
    # We currently have no way of detecting which cutouts of the "source" chopper
    # are used (not blocked by other choppers), for this test we remove those we
    # know are not used.
    choppers['wfm1'] = choppers['wfm1'][2:]
    beamline = fakes.FakeBeamline(
        source=ess_10s_14Hz,
        pulse=ess_pulse,
        choppers=choppers,
        monitors={'monitor': distance},
        detectors={},
        # time_of_flight_origin='wfm1',
    )
    mon, ref = beamline.get_monitor('monitor')

    pl = sl.Pipeline(
        unwrap.unwrap_providers()
        + unwrap.time_of_flight_providers()
        + unwrap.time_of_flight_origin_from_choppers_providers(wfm=True)
    )
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = choppers
    # Actually the source should be at the center between wfm1 and wfm2, but we
    # currently don't have a way of handling this. We could manually create a
    # "virtual" chopper.
    pl[unwrap.SourceChopperName] = 'wfm1'
    pl[unwrap.Ltotal] = distance
    bounds = pl.compute(unwrap.SubframeBounds)
    assert bounds.sizes == {'bound': 2, 'subframe': 6}
    result = pl.compute(unwrap.TofData)
    ref.coords['Ltotal'] = distance - choppers['wfm1'].distance
    # FakeBeamline does not support WFM yet, we cannot run a better check for now.
    assert_identical(result.sum(), ref.sum())
