# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

from scippneutron.conversion.graph.beamline import beamline as beamline_graph
from scippneutron.conversion.graph.tof import elastic as elastic_graph
from scippneutron.tof import fakes, unwrap

sl = pytest.importorskip('sciline')


@pytest.fixture
def ess_10s_14Hz() -> fakes.FakeSource:
    return fakes.FakeSource(
        frequency=sc.scalar(14.0, unit='Hz'), run_length=sc.scalar(10.0, unit='s')
    )


@pytest.fixture
def ess_10s_7Hz() -> fakes.FakeSource:
    return fakes.FakeSource(
        frequency=sc.scalar(7.0, unit='Hz'), run_length=sc.scalar(10.0, unit='s')
    )


@pytest.fixture
def ess_pulse() -> fakes.FakePulse:
    return fakes.FakePulse(
        time_min=sc.scalar(0.0, unit='ms'),
        time_max=sc.scalar(3.0, unit='ms'),
        wavelength_min=sc.scalar(0.1, unit='angstrom'),
        wavelength_max=sc.scalar(10.0, unit='angstrom'),
    )


def test_frame_period_is_pulse_period_if_not_pulse_skipping() -> None:
    # pl = sl.Pipeline(unwrap.unwrap_providers())
    pl = sl.Pipeline(unwrap.providers())
    period = sc.scalar(123.0, unit='ms')
    pl[unwrap.PulsePeriod] = period
    pl[unwrap.PulseStride] = 1
    assert_identical(pl.compute(unwrap.FramePeriod), period)


@pytest.mark.parametrize('stride', [1, 2, 3, 4])
def test_frame_period_is_multiple_pulse_period_if_pulse_skipping(stride) -> None:
    # pl = sl.Pipeline(unwrap.unwrap_providers())
    pl = sl.Pipeline(unwrap.providers())
    period = sc.scalar(123.0, unit='ms')
    pl[unwrap.PulsePeriod] = period
    pl[unwrap.PulseStride] = stride
    assert_identical(pl.compute(unwrap.FramePeriod), stride * period)


# @pytest.mark.parametrize('stride', [1, 2, 3, 4])
# def test_pulse_offset(stride) -> None:
#     pl = sl.Pipeline(unwrap.unwrap_providers(pulse_skipping=True))
#     period = sc.scalar(123.0, unit='ms')
#     pl[unwrap.PulsePeriod] = period
#     pl[unwrap.PulseStride] = stride
#     start = sc.datetime('2020-01-01T00:00:00.0', unit='ns')
#     time_zero = start + sc.arange('pulse', 0 * period, 100 * period, period).to(
#         unit='ns', dtype='int64'
#     )
#     pl[unwrap.TimeZero] = time_zero

#     result = pl.compute(unwrap.PulseOffset)
#     assert_identical(
#         result, (sc.arange('pulse', 0, 100, dtype='int64') % stride) * period
#     )


# def test_offset_from_wrapped() -> None:
#     pl = sl.Pipeline(unwrap.unwrap_providers())
#     period = sc.scalar(123.0, unit='ms')
#     pl[unwrap.PulsePeriod] = period
#     pl[unwrap.PulseStride] = None
#     pl[unwrap.FrameBounds] = unwrap.FrameBounds(
#         sc.DataGroup(time=sc.array(dims=['bound'], values=[0.01, 0.02], unit='s'))
#     )
#     wrapped_offset = sc.linspace('event', 0.0, 123.0, num=1001, unit='ms')
#     pl[unwrap.PulseWrappedTimeOffset] = unwrap.PulseWrappedTimeOffset(wrapped_offset)
#     offset = pl.compute(unwrap.DeltaFromWrapped)
#     # Times below 10 ms (we currently cut at lower bound) should be offset by period.
#     da = sc.DataArray(offset, coords={'time': wrapped_offset})
#     assert sc.all(da['time', : 10 * sc.Unit('ms')].data == period.to(unit='s'))
#     assert sc.all(da['time', 10 * sc.Unit('ms') :].data == sc.scalar(0.0, unit='s'))


# def test_offset_from_wrapped_has_no_special_handling_for_out_of_period_events() -> None:
#     pl = sl.Pipeline(unwrap.unwrap_providers())
#     period = sc.scalar(123.0, unit='ms')
#     pl[unwrap.PulsePeriod] = period
#     pl[unwrap.PulseStride] = None
#     pl[unwrap.FrameBounds] = unwrap.FrameBounds(
#         sc.DataGroup(time=sc.array(dims=['bound'], values=[0.01, 0.02], unit='s'))
#     )
#     wrapped_offset = sc.linspace('event', -10000.0, 10000.0, num=10001, unit='ms')
#     pl[unwrap.PulseWrappedTimeOffset] = unwrap.PulseWrappedTimeOffset(wrapped_offset)
#     offset = pl.compute(unwrap.DeltaFromWrapped)
#     da = sc.DataArray(offset, coords={'time': wrapped_offset})
#     # Negative times and times > 123 ms are technically invalid, but it does not affect
#     # unwrapping, so they should be left as-is.
#     assert sc.all(da['time', : 10 * sc.Unit('ms')].data == period.to(unit='s'))
#     assert sc.all(da['time', 10 * sc.Unit('ms') :].data == sc.scalar(0.0, unit='s'))


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
        unwrap.providers_no_choppers()
        # ()
        # + unwrap.time_of_flight_providers()
        # + unwrap.time_of_flight_origin_from_choppers_providers()
    )
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    pl[unwrap.PulseStride] = 1
    # pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    # pl[unwrap.SourceWavelengthRange] = (
    #     ess_pulse.wavelength_min,
    #     ess_pulse.wavelength_max,
    # )
    # pl[unwrap.Choppers] = {}
    # pl[unwrap.SourceChopperName | None] = None
    pl[unwrap.Ltotal] = distance

    result = pl.compute(unwrap.TofData)

    assert sc.allclose(
        result.bins.concat().value.coords['tof'],
        ref.bins.concat().value.coords['tof'],
    )

    # unwrapped_toa = pl.compute(unwrap.UnwrappedTimeOfArrival)
    # # No unwrap is happening, frame does not overlap next pulse.
    # assert (mon.coords['event_time_zero'] == unwrapped.bins.coords['pulse_time']).all()

    # origin = pl.compute(unwrap.TimeOfFlightOrigin)
    # assert_identical(origin.time, sc.scalar(0.0015, unit='s'))
    # assert_identical(origin.distance, sc.scalar(0.0, unit='m'))

    # result = pl.compute(unwrap.TofData)
    # assert_identical(
    #     result.hist(tof=1000).sum('pulse'), ref.hist(tof=1000).sum('pulse')
    # )


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
    pl[unwrap.PulseStride] = 1
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = {}
    pl[unwrap.Ltotal] = distance
    with pytest.raises(ValueError, match='Frames are overlapping'):
        pl.compute(unwrap.TofData)


# At 44m, event_time_offset does not wrap around (all events are within the same pulse).
# At 47m, event_time_offset wraps around.
@pytest.mark.parametrize('dist', [44.0, 47.0])
def test_standard_unwrap(ess_10s_14Hz, ess_pulse, dist) -> None:
    distance = sc.scalar(dist, unit='m')
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
        unwrap.providers()
        # unwrap.unwrap_providers()
        # + unwrap.time_of_flight_providers()
        # + unwrap.time_of_flight_origin_from_choppers_providers()
    )
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    pl[unwrap.PulseStride] = 1
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = fakes.psc_choppers
    # pl[unwrap.SourceChopperName | None] = 'psc1'
    pl[unwrap.Ltotal] = distance
    result = pl.compute(unwrap.TofData)
    # assert_identical(
    #     result.hist(tof=1000).sum('pulse'), ref.hist(tof=1000).sum('pulse')
    # )
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    ref_wav = ref.transform_coords('wavelength', graph=graph).bins.concat().value
    result.coords['Ltotal'] = distance
    result_wav = result.transform_coords('wavelength', graph=graph).bins.concat().value

    assert sc.allclose(
        result_wav.coords['wavelength'],
        ref_wav.coords['wavelength'],
        rtol=sc.scalar(1e-02),
    )


# def test_standard_unwrap(ess_10s_14Hz, ess_pulse) -> None:
#     distance = sc.scalar(47.0, unit='m')
#     beamline = fakes.FakeBeamline(
#         source=ess_10s_14Hz,
#         pulse=ess_pulse,
#         choppers=fakes.psc_choppers,
#         monitors={'monitor': distance},
#         detectors={},
#         time_of_flight_origin='psc1',
#     )
#     mon, ref = beamline.get_monitor('monitor')

#     pl = sl.Pipeline(
#         unwrap.providers
#         # unwrap.unwrap_providers()
#         # + unwrap.time_of_flight_providers()
#         # + unwrap.time_of_flight_origin_from_choppers_providers()
#     )
#     pl[unwrap.RawData] = mon
#     pl[unwrap.PulsePeriod] = beamline._source.pulse_period
#     pl[unwrap.PulseStride] = 1
#     pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
#     pl[unwrap.SourceWavelengthRange] = (
#         ess_pulse.wavelength_min,
#         ess_pulse.wavelength_max,
#     )
#     pl[unwrap.Choppers] = fakes.psc_choppers
#     # pl[unwrap.SourceChopperName | None] = 'psc1'
#     pl[unwrap.Ltotal] = distance
#     result = pl.compute(unwrap.TofData)
#     assert_identical(
#         result.hist(tof=1000).sum('pulse'), ref.hist(tof=1000).sum('pulse')
#     )


# At 44m, event_time_offset does not wrap around (all events are within the same pulse).
# At 47m, event_time_offset wraps around.
@pytest.mark.parametrize('dist', [44.0, 47.0])
def test_standard_unwrap_histogram_mode(ess_10s_14Hz, ess_pulse, dist) -> None:
    distance = sc.scalar(dist, unit='m')
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
        unwrap.providers()
        # + unwrap.time_of_flight_providers()
        # + unwrap.time_of_flight_origin_from_choppers_providers()
    )
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    pl[unwrap.PulseStride] = 1
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = fakes.psc_choppers
    # pl[unwrap.SourceChopperName | None] = 'psc1'
    pl[unwrap.Ltotal] = distance
    # result = pl.compute(unwrap.TofData)
    # assert_identical(result.sum(), ref.sum())
    result = pl.compute(unwrap.ReHistogrammedTofData)
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    result.coords['Ltotal'] = distance
    result_wav = result.transform_coords('wavelength', graph=graph)
    ref_wav = (
        ref.transform_coords('wavelength', graph=graph)
        .bins.concat()
        .value.hist(wavelength=result_wav.coords['wavelength'])
    )
    diff = (result_wav - ref_wav) / ref_wav
    # There are outliers in the diff because the bins don't cover the exact same range.
    # We check that 96% of the data has an error below 0.1.
    x = np.abs(diff.data.values)
    assert np.percentile(x[np.isfinite(x)], 96.0) < 0.1


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
    pl[unwrap.PulseStride] = None
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )

    pl[unwrap.Choppers] = fakes.psc_choppers
    pl[unwrap.SourceChopperName | None] = 'psc1'
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
    pl[unwrap.PulseStride] = None
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = fakes.psc_choppers
    pl[unwrap.SourceChopperName | None] = 'psc1'
    pl[unwrap.Ltotal] = distance
    pl[unwrap.PulseStride] = 2
    with pytest.raises(NotImplementedError):
        pl.compute(unwrap.TofData)


def test_wfm_unwrap(ess_10s_14Hz, ess_pulse) -> None:
    distance = sc.scalar(20.0, unit='m')
    choppers = fakes.wfm_choppers
    beamline = fakes.FakeBeamline(
        source=ess_10s_14Hz,
        pulse=ess_pulse,
        choppers=choppers,
        monitors={'monitor': distance},
        detectors={},
    )
    mon, _ = beamline.get_monitor('monitor')

    pl = sl.Pipeline(
        unwrap.unwrap_providers()
        + unwrap.time_of_flight_providers()
        + unwrap.time_of_flight_origin_from_choppers_providers(wfm=True)
    )
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    pl[unwrap.PulseStride] = None
    pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    pl[unwrap.SourceWavelengthRange] = (
        ess_pulse.wavelength_min,
        ess_pulse.wavelength_max,
    )
    pl[unwrap.Choppers] = choppers
    pl[unwrap.Ltotal] = distance
    bounds = pl.compute(unwrap.FrameAtDetector).subbounds()
    assert bounds.sizes == {'bound': 2, 'subframe': 6}
    result = pl.compute(unwrap.TofData)
    assert_identical(result.coords['Ltotal'], distance - choppers['wfm1'].distance)
