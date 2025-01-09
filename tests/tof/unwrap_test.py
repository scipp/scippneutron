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
    pl = sl.Pipeline(unwrap.providers())
    period = sc.scalar(123.0, unit='ms')
    pl[unwrap.PulsePeriod] = period
    pl[unwrap.PulseStride] = 1
    assert_identical(pl.compute(unwrap.FramePeriod), period)


@pytest.mark.parametrize('stride', [1, 2, 3, 4])
def test_frame_period_is_multiple_pulse_period_if_pulse_skipping(stride) -> None:
    pl = sl.Pipeline(unwrap.providers())
    period = sc.scalar(123.0, unit='ms')
    pl[unwrap.PulsePeriod] = period
    pl[unwrap.PulseStride] = stride
    assert_identical(pl.compute(unwrap.FramePeriod), stride * period)


def test_unwrap_with_no_choppers() -> None:
    # At this small distance the frames are not overlapping (with the given wavelength
    # range), despite not using any choppers.
    distance = sc.scalar(10.0, unit='m')

    beamline = fakes.FakeBeamlineEss(
        # choppers=choppers,
        choppers={},
        monitors={"detector": distance},
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=100_000,
    )

    # beamline = fakes.FakeBeamline(
    #     source=ess_10s_14Hz,
    #     pulse=ess_pulse,
    #     choppers={},  # no choppers
    #     monitors={'monitor': distance},
    #     detectors={},
    # )
    mon, ref = beamline.get_monitor('detector')

    #  # Set up the workflow
    # workflow = sl.Pipeline(unwrap.providers(), params=unwrap.params())
    # workflow[unwrap.Facility] = 'ess'
    # workflow[unwrap.RawData] = raw_data
    # workflow[unwrap.Choppers] = disk_choppers
    # workflow[unwrap.Ltotal] = raw_data.coords['Ltotal']

    pl = sl.Pipeline(unwrap.providers(), params=unwrap.params())
    pl[unwrap.Facility] = 'ess'
    pl[unwrap.RawData] = mon
    # pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    # pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    # pl[unwrap.SourceWavelengthRange] = (
    #     ess_pulse.wavelength_min,
    #     ess_pulse.wavelength_max,
    # )
    pl[unwrap.Choppers] = {}
    pl[unwrap.Ltotal] = distance

    tofs = pl.compute(unwrap.TofData)

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph).bins.concat().value
    ref = ref.bins.concat().value

    diff = abs(
        (wavs.coords['wavelength'] - ref.coords['wavelength'])
        / ref.coords['wavelength']
    )
    # Most errors should be small
    assert np.nanpercentile(diff.values, 96) < 1.0

    # # Ensure that the bounds are close
    # res_tof = result.coords['tof']
    # ref_tof = ref.coords['tof']
    # delta = ref_tof.max() - ref_tof.min()
    # assert sc.abs((res_tof.min() - ref_tof.min()) / delta) < sc.scalar(1e-02)
    # assert sc.abs((res_tof.max() - ref_tof.max()) / delta) < sc.scalar(1e-02)

    # # Because the bounds are not the same, using the same bins for bot results would
    # # lead to large differences at the edges. So we pick the most narrow range to
    # # histogram.
    # bins = sc.linspace(
    #     'tof',
    #     max(res_tof.min(), ref_tof.min()),
    #     min(res_tof.max(), ref_tof.max()),
    #     num=501,
    # )

    # ref_hist = ref.hist(tof=bins)
    # res_hist = result.hist(tof=bins)
    # diff = ((res_hist - ref_hist) / ref_hist.max()).data
    # assert sc.abs(diff).max() < sc.scalar(1.0e-1)


# def test_unwrap_with_frame_overlap_raises(ess_10s_14Hz, ess_pulse) -> None:
#     distance = sc.scalar(46.0, unit='m')
#     beamline = fakes.FakeBeamline(
#         source=ess_10s_14Hz,
#         pulse=ess_pulse,
#         choppers={},  # no choppers
#         monitors={'monitor': distance},
#         detectors={},
#     )
#     mon, _ = beamline.get_monitor('monitor')

#     pl = sl.Pipeline(unwrap.providers(), params=unwrap.params())
#     pl[unwrap.RawData] = mon
#     pl[unwrap.PulsePeriod] = beamline._source.pulse_period
#     pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
#     pl[unwrap.SourceWavelengthRange] = (
#         ess_pulse.wavelength_min,
#         ess_pulse.wavelength_max,
#     )
#     pl[unwrap.Choppers] = {}
#     pl[unwrap.Ltotal] = distance
#     with pytest.raises(ValueError, match='Frames are overlapping'):
#         pl.compute(unwrap.TofData)


# At 80m, event_time_offset does not wrap around (all events are within the same pulse).
# At 85m, event_time_offset wraps around.
@pytest.mark.parametrize('dist', [80.0, 85.0])
def test_standard_unwrap(dist) -> None:
    distance = sc.scalar(dist, unit='m')
    beamline = fakes.FakeBeamlineEss(
        choppers=fakes.psc_disk_choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=100_000,
    )
    # beamline = fakes.FakeBeamline(
    #     source=ess_10s_14Hz,
    #     pulse=ess_pulse,
    #     choppers=fakes.psc_disk_choppers,
    #     monitors={'monitor': distance},
    #     detectors={},
    #     time_of_flight_origin='psc1',
    # )
    mon, ref = beamline.get_monitor('detector')

    pl = sl.Pipeline(unwrap.providers(), params=unwrap.params())
    pl[unwrap.Facility] = 'ess'
    pl[unwrap.RawData] = mon
    pl[unwrap.Choppers] = fakes.psc_disk_choppers
    pl[unwrap.Ltotal] = distance

    # pl = sl.Pipeline(unwrap.providers(), params=unwrap.params())
    # pl[unwrap.RawData] = mon
    # pl[unwrap.PulsePeriod] = beamline._source.pulse_period
    # pl[unwrap.SourceTimeRange] = ess_pulse.time_min, ess_pulse.time_max
    # pl[unwrap.SourceWavelengthRange] = (
    #     ess_pulse.wavelength_min,
    #     ess_pulse.wavelength_max,
    # )
    # pl[unwrap.Choppers] = fakes.psc_choppers
    # pl[unwrap.Ltotal] = distance

    tofs = pl.compute(unwrap.TofData)

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph).bins.concat().value
    ref = ref.bins.concat().value

    diff = abs(
        (wavs.coords['wavelength'] - ref.coords['wavelength'])
        / ref.coords['wavelength']
    )
    # All errors should be small
    assert np.nanpercentile(diff.values, 100) < 0.01

    # assert sc.allclose(
    #     wavs.coords['wavelength'], ref.coords['wavelength'], rtol=sc.scalar(1e-02)
    # )


# At 80m, event_time_offset does not wrap around (all events are within the same pulse).
# At 85m, event_time_offset wraps around.
@pytest.mark.parametrize('dist', [80.0, 85.0])
def test_standard_unwrap_histogram_mode(dist) -> None:
    distance = sc.scalar(dist, unit='m')
    beamline = fakes.FakeBeamlineEss(
        choppers=fakes.psc_disk_choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=100_000,
    )
    mon, ref = beamline.get_monitor('detector')
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
        (*unwrap.providers(), unwrap.re_histogram_tof_data), params=unwrap.params()
    )
    pl[unwrap.Facility] = 'ess'
    pl[unwrap.RawData] = mon
    pl[unwrap.Choppers] = fakes.psc_disk_choppers
    pl[unwrap.Ltotal] = distance
    tofs = pl.compute(unwrap.ReHistogrammedTofData)
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords('wavelength', graph=graph)
    ref = ref.bins.concat().value.hist(wavelength=wavs.coords['wavelength'])
    # We divide by the maximum to avoid large relative differences at the edges of the
    # frames where the counts are low.
    diff = (wavs - ref) / ref.max()
    assert np.nanpercentile(diff.values, 96.0) < 0.3


@pytest.mark.parametrize('dist', [150.0, 180.0])
def test_pulse_skipping_unwrap(dist) -> None:
    distance = sc.scalar(dist, unit='m')
    choppers = fakes.psc_disk_choppers.copy()
    choppers['pulse_skipping'] = fakes.pulse_skipping

    beamline = fakes.FakeBeamlineEss(
        choppers=choppers,
        monitors={'detector': distance},
        run_length=sc.scalar(1.0, unit='s'),
        events_per_pulse=100_000,
    )
    mon, ref = beamline.get_monitor('detector')

    pl = sl.Pipeline(unwrap.providers(), params=unwrap.params())
    pl[unwrap.Facility] = 'ess'
    pl[unwrap.RawData] = mon
    pl[unwrap.Choppers] = choppers
    pl[unwrap.Ltotal] = distance
    pl[unwrap.PulseStride] = 2

    tofs = pl.compute(unwrap.TofData)

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph).bins.concat().value
    ref = ref.bins.concat().value

    diff = abs(
        (wavs.coords['wavelength'] - ref.coords['wavelength'])
        / ref.coords['wavelength']
    )
    # All errors should be small
    assert np.nanpercentile(diff.values, 100) < 0.01


@pytest.mark.parametrize('dist', [44.0, 47.0])
def test_pulse_skipping_with_180deg_phase_unwrap(dist) -> None:
    from copy import copy

    distance = sc.scalar(dist, unit='m')

    # We will add 180 deg to the phase of the pulse-skipping chopper. This means that
    # the first pulse will be blocked and the second one will be transmitted.
    # When finding the FrameAtDetector, we need to propagate the second pulse through
    # the cascade as well. For that, we need to spin the choppers by an additional
    # rotation.
    period = 1.0 / sc.scalar(14.0, unit='Hz')
    choppers = sc.DataGroup()
    for key, value in fakes.psc_choppers.items():
        ch = copy(value)
        ch.time_open = sc.concat(
            [ch.time_open, ch.time_open + period], ch.time_open.dim
        )
        ch.time_close = sc.concat(
            [ch.time_close, ch.time_close + period], ch.time_close.dim
        )
        choppers[key] = ch

    choppers['pulse_skipping'] = copy(fakes.pulse_skipping)
    # Add 180 deg to the phase of the pulse-skipping chopper (same as offsetting the
    # time by one period).
    choppers['pulse_skipping'].time_open = choppers['pulse_skipping'].time_open + period
    choppers['pulse_skipping'].time_close = (
        choppers['pulse_skipping'].time_close + period
    )

    # We use the ESS fake here because the fake beamline does not support choppers
    # rotating at 7 Hz.
    beamline = fakes.FakeBeamlineEss(
        choppers=choppers,
        monitors={'monitor': distance},
        run_length=sc.scalar(1.0, unit='s'),
        events_per_pulse=100_000,
    )
    mon, ref = beamline.get_monitor('monitor')

    pl = sl.Pipeline(unwrap.providers(), params=unwrap.params())
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = 1.0 / beamline.source.frequency
    pl[unwrap.PulseStride] = 2

    one_pulse = beamline.source.data['pulse', 0]
    pl[unwrap.SourceTimeRange] = (
        one_pulse.coords['time'].min(),
        one_pulse.coords['time'].max(),
    )
    pl[unwrap.SourceWavelengthRange] = (
        one_pulse.coords['wavelength'].min(),
        one_pulse.coords['wavelength'].max(),
    )

    pl[unwrap.Choppers] = choppers
    pl[unwrap.Ltotal] = distance
    result = pl.compute(unwrap.TofData)
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    ref_wav = ref.transform_coords('wavelength', graph=graph).bins.concat().value
    result.coords['Ltotal'] = distance
    result_wav = result.transform_coords('wavelength', graph=graph).bins.concat().value

    assert sc.allclose(
        result_wav.coords['wavelength'],
        ref_wav.coords['wavelength'],
        rtol=sc.scalar(1e-02),
    )


def test_pulse_skipping_unwrap_with_half_of_first_frame_missing() -> None:
    distance = sc.scalar(50.0, unit='m')

    choppers = fakes.psc_disk_choppers.copy()
    choppers['pulse_skipping'] = fakes.pulse_skipping

    # We use the ESS fake here because the fake beamline does not support choppers
    # rotating at 7 Hz.
    beamline = fakes.FakeBeamlineEss(
        choppers=choppers,
        monitors={'detector': distance},
        run_length=sc.scalar(1.0, unit='s'),
        events_per_pulse=100_000,
    )
    mon, ref = beamline.get_monitor('detector')

    pl = sl.Pipeline(unwrap.providers(), params=unwrap.params())
    pl[unwrap.Facility] = 'ess'
    pl[unwrap.RawData] = mon[1:].copy()  # Skip first pulse = half of the first frame
    pl[unwrap.Choppers] = choppers
    pl[unwrap.Ltotal] = distance
    pl[unwrap.PulseStride] = 2

    tofs = pl.compute(unwrap.TofData)

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph).bins.concat().value
    ref = ref.bins.concat().value

    diff = abs(
        (wavs.coords['wavelength'] - ref.coords['wavelength'])
        / ref.coords['wavelength']
    )
    # All errors should be small
    assert np.nanpercentile(diff.values, 100) < 0.01

    # choppers = fakes.psc_choppers.copy()
    # choppers['pulse_skipping'] = fakes.pulse_skipping

    # # We use the ESS fake here because the fake beamline does not support choppers
    # # rotating at 7 Hz.
    # beamline = fakes.FakeBeamlineEss(
    #     choppers=choppers,
    #     monitors={'monitor': distance},
    #     run_length=sc.scalar(1.0, unit='s'),
    #     events_per_pulse=100_000,
    # )
    # mon, ref = beamline.get_monitor('monitor')

    # pl = sl.Pipeline(unwrap.providers(), params=unwrap.params())
    # pl[unwrap.RawData] = mon[1:].copy()  # Skip first pulse = half of the first frame
    # pl[unwrap.PulsePeriod] = 1.0 / beamline.source.frequency
    # pl[unwrap.PulseStride] = 2
    # pl[unwrap.PulseStrideOffset] = 1  # Start the stride at the second pulse

    # one_pulse = beamline.source.data['pulse', 0]
    # pl[unwrap.SourceTimeRange] = (
    #     one_pulse.coords['time'].min(),
    #     one_pulse.coords['time'].max(),
    # )
    # pl[unwrap.SourceWavelengthRange] = (
    #     one_pulse.coords['wavelength'].min(),
    #     one_pulse.coords['wavelength'].max(),
    # )

    # pl[unwrap.Choppers] = choppers
    # pl[unwrap.Ltotal] = distance
    # result = pl.compute(unwrap.TofData)
    # graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    # ref_wav = (
    #     ref[1:].copy().transform_coords('wavelength', graph=graph).bins.concat().value
    # )
    # result.coords['Ltotal'] = distance
    # result_wav = result.transform_coords('wavelength', graph=graph).bins.concat().value

    # assert sc.allclose(
    #     result_wav.coords['wavelength'],
    #     ref_wav.coords['wavelength'],
    #     rtol=sc.scalar(1e-02),
    # )


@pytest.mark.parametrize('dist', [44.0, 47.0])
def test_pulse_skipping_unwrap_histogram_mode(dist) -> None:
    distance = sc.scalar(dist, unit='m')
    choppers = fakes.psc_choppers.copy()
    choppers['pulse_skipping'] = fakes.pulse_skipping

    # We use the ESS fake here because the fake beamline does not support choppers
    # rotating at 7 Hz.
    beamline = fakes.FakeBeamlineEss(
        choppers=choppers,
        monitors={'monitor': distance},
        run_length=sc.scalar(1.0, unit='s'),
        events_per_pulse=100_000,
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
        (*unwrap.providers(), unwrap.re_histogram_tof_data), params=unwrap.params()
    )
    pl[unwrap.RawData] = mon
    pl[unwrap.PulsePeriod] = 1.0 / beamline.source.frequency
    pl[unwrap.PulseStride] = 2

    one_pulse = beamline.source.data['pulse', 0]
    pl[unwrap.SourceTimeRange] = (
        one_pulse.coords['time'].min(),
        one_pulse.coords['time'].max(),
    )
    pl[unwrap.SourceWavelengthRange] = (
        one_pulse.coords['wavelength'].min(),
        one_pulse.coords['wavelength'].max(),
    )

    pl[unwrap.Choppers] = choppers
    pl[unwrap.Ltotal] = distance

    result = pl.compute(unwrap.ReHistogrammedTofData)
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    result.coords['Ltotal'] = distance
    result_wav = result.transform_coords('wavelength', graph=graph)
    ref_wav = (
        ref.transform_coords('wavelength', graph=graph)
        .bins.concat()
        .value.hist(wavelength=result_wav.coords['wavelength'])
    )

    # In this case, we used the ESS pulse. The counts on the edges of the frame are low,
    # so relative differences can be large. Instead of a plain relative difference, we
    # use the maximum of the reference data as the denominator.
    diff = (result_wav - ref_wav) / ref_wav.max()
    # Note: very conservative threshold to avoid making the test flaky.
    assert sc.abs(diff).data.max() < sc.scalar(0.5)
