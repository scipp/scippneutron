# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from functools import partial

import pytest
import scipp as sc
import tof

from scippneutron.chopper import DiskChopper
from scippneutron.conversion.graph.beamline import beamline
from scippneutron.conversion.graph.tof import elastic
from scippneutron.tof import chopper_cascade, fakes, unwrap

sl = pytest.importorskip('sciline')


@pytest.fixture
def disk_choppers():
    psc1 = DiskChopper(
        frequency=sc.scalar(14.0, unit="Hz"),
        beam_angle=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(286 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 6.145], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=[-1.23, 70.49, 84.765, 113.565, 170.29, 271.635, 286.035, 301.17],
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=[1.23, 73.51, 88.035, 116.835, 175.31, 275.565, 289.965, 303.63],
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    psc2 = DiskChopper(
        frequency=sc.scalar(-14.0, unit="Hz"),
        beam_angle=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-236, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 6.155], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=[-1.23, 27.0, 55.8, 142.385, 156.765, 214.115, 257.23, 315.49],
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=[1.23, 30.6, 59.4, 145.615, 160.035, 217.885, 261.17, 318.11],
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    oc = DiskChopper(
        frequency=sc.scalar(14.0, unit="Hz"),
        beam_angle=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(297 - 180 - 90, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 6.174], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-27.6 * 0.5], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[27.6 * 0.5], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    bcc = DiskChopper(
        frequency=sc.scalar(112.0, unit="Hz"),
        beam_angle=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(240 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 9.78], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-36.875, 143.125], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[36.875, 216.875], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    t0 = DiskChopper(
        frequency=sc.scalar(28.0, unit="Hz"),
        beam_angle=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(280 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 13.05], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-314.9 * 0.5], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[314.9 * 0.5], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    return {"psc1": psc1, "psc2": psc2, "oc": oc, "bcc": bcc, "t0": t0}


@pytest.fixture
def overlap_choppers(disk_choppers):
    out = disk_choppers.copy()
    out['bcc'] = DiskChopper(
        frequency=sc.scalar(112.0, unit="Hz"),
        beam_angle=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(240 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 9.78], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-36.875, 143.125], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[46.875, 216.875], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )
    return out


@pytest.mark.parametrize("npulses", [1, 2])
@pytest.mark.parametrize(
    "ltotal",
    [
        sc.array(dims=['detector_number'], values=[77.675], unit='m'),
        sc.array(dims=['detector_number'], values=[77.675, 76.5], unit='m'),
        sc.array(
            dims=['y', 'x'],
            values=[[77.675, 76.1, 78.05], [77.15, 77.3, 77.675]],
            unit='m',
        ),
    ],
)
def test_dream_wfm(disk_choppers, npulses, ltotal):
    choppers = {
        key: chopper_cascade.Chopper.from_disk_chopper(
            chop, pulse_frequency=sc.scalar(14.0, unit="Hz"), npulses=npulses
        )
        for key, chop in disk_choppers.items()
    }

    monitors = {
        f"detector{i}": ltot for i, ltot in enumerate(ltotal.flatten(to='detector'))
    }

    # Create some neutron events
    wavelengths = sc.array(
        dims=['event'], values=[1.5, 1.6, 1.7, 3.3, 3.4, 3.5], unit='angstrom'
    )
    birth_times = sc.full(sizes=wavelengths.sizes, value=1.5, unit='ms')
    ess_beamline = fakes.FakeBeamlineEss(
        choppers=choppers,
        monitors=monitors,
        run_length=sc.scalar(1 / 14, unit="s") * npulses,
        events_per_pulse=len(wavelengths),
        source=partial(
            tof.Source.from_neutrons,
            birth_times=birth_times,
            wavelengths=wavelengths,
            frequency=sc.scalar(14.0, unit="Hz"),
        ),
    )

    # Save the true wavelengths for later
    true_wavelengths = ess_beamline.source.data.coords["wavelength"]

    raw_data = sc.concat(
        [ess_beamline.get_monitor(key) for key in monitors.keys()],
        dim='detector',
    ).fold(dim='detector', sizes=ltotal.sizes)

    # Verify that all 6 neutrons made it through the chopper cascade
    assert sc.identical(
        raw_data.bins.concat('pulse').hist().data,
        sc.array(
            dims=['detector'],
            values=[len(wavelengths) * npulses] * len(monitors),
            unit="counts",
            dtype='float64',
        ).fold(dim='detector', sizes=ltotal.sizes),
    )

    # Set up the workflow
    workflow = sl.Pipeline(
        unwrap.unwrap_providers()
        + unwrap.time_of_flight_providers()
        + unwrap.time_of_flight_origin_from_choppers_providers(wfm=True)
    )
    workflow[unwrap.PulsePeriod] = sc.reciprocal(ess_beamline.source.frequency)
    workflow[unwrap.PulseStride | None] = None

    # Define the extent of the pulse that contains the 6 neutrons in time and wavelength
    # Note that we make a larger encompassing pulse to ensure that the frame bounds are
    # computed correctly
    workflow[unwrap.SourceTimeRange] = (
        sc.scalar(0.0, unit='ms'),
        sc.scalar(4.9, unit='ms'),
    )
    workflow[unwrap.SourceWavelengthRange] = (
        sc.scalar(0.2, unit='angstrom'),
        sc.scalar(16.0, unit='angstrom'),
    )

    workflow[unwrap.Choppers] = choppers
    workflow[unwrap.Ltotal] = ltotal
    workflow[unwrap.RawData] = raw_data

    # Compute time-of-flight
    tofs = workflow.compute(unwrap.TofData)
    assert {dim: tofs.sizes[dim] for dim in ltotal.sizes} == ltotal.sizes

    # Convert to wavelength
    graph = {**beamline(scatter=False), **elastic("tof")}
    wav_wfm = tofs.transform_coords("wavelength", graph=graph)

    # Compare the computed wavelengths to the true wavelengths
    for i in range(npulses):
        result = wav_wfm['pulse', i].flatten(to='detector')
        for j in range(len(result)):
            computed_wavelengths = result[j].values.coords["wavelength"]
            assert sc.allclose(
                computed_wavelengths,
                true_wavelengths['pulse', i],
                rtol=sc.scalar(1e-02),
            )


@pytest.mark.parametrize("npulses", [1, 2])
@pytest.mark.parametrize(
    "ltotal",
    [
        sc.array(dims=['detector_number'], values=[77.675], unit='m'),
        sc.array(dims=['detector_number'], values=[77.675, 76.5], unit='m'),
        sc.array(
            dims=['y', 'x'],
            values=[[77.675, 76.1, 78.05], [77.15, 77.3, 77.675]],
            unit='m',
        ),
    ],
)
def test_dream_wfm_with_subframe_time_overlap(overlap_choppers, npulses, ltotal):
    choppers = {
        key: chopper_cascade.Chopper.from_disk_chopper(
            chop, pulse_frequency=sc.scalar(14.0, unit="Hz"), npulses=npulses
        )
        for key, chop in overlap_choppers.items()
    }

    monitors = {
        f"detector{i}": ltot for i, ltot in enumerate(ltotal.flatten(to='detector'))
    }

    # Create some neutron events
    wavelengths = [1.5, 1.6, 1.7, 3.3, 3.4, 3.5]
    birth_times = [1.5] * len(wavelengths)

    # Add overlap neutrons
    birth_times.extend([0.0, 3.1])
    wavelengths.extend([2.7, 2.5])

    wavelengths = sc.array(dims=['event'], values=wavelengths, unit='angstrom')
    birth_times = sc.array(dims=['event'], values=birth_times, unit='ms')

    ess_beamline = fakes.FakeBeamlineEss(
        choppers=choppers,
        monitors=monitors,
        run_length=sc.scalar(1 / 14, unit="s") * npulses,
        events_per_pulse=len(wavelengths),
        source=partial(
            tof.Source.from_neutrons,
            birth_times=birth_times,
            wavelengths=wavelengths,
            frequency=sc.scalar(14.0, unit="Hz"),
        ),
    )

    # Save the true wavelengths for later
    true_wavelengths = ess_beamline.source.data.coords["wavelength"]

    raw_data = sc.concat(
        [ess_beamline.get_monitor(key) for key in monitors.keys()],
        dim='detector',
    ).fold(dim='detector', sizes=ltotal.sizes)

    # Verify that all 6 neutrons made it through the chopper cascade
    assert sc.identical(
        raw_data.bins.concat('pulse').hist().data,
        sc.array(
            dims=['detector'],
            values=[len(wavelengths) * npulses] * len(monitors),
            unit="counts",
            dtype='float64',
        ).fold(dim='detector', sizes=ltotal.sizes),
    )

    # Set up the workflow
    workflow = sl.Pipeline(
        unwrap.unwrap_providers()
        + unwrap.time_of_flight_providers()
        + unwrap.time_of_flight_origin_from_choppers_providers(wfm=True)
    )
    workflow[unwrap.PulsePeriod] = sc.reciprocal(ess_beamline.source.frequency)
    workflow[unwrap.PulseStride | None] = None

    # Define the extent of the pulse that contains the 6 neutrons in time and wavelength
    # Note that we make a larger encompassing pulse to ensure that the frame bounds are
    # computed correctly
    workflow[unwrap.SourceTimeRange] = (
        sc.scalar(0.0, unit='ms'),
        sc.scalar(4.9, unit='ms'),
    )
    workflow[unwrap.SourceWavelengthRange] = (
        sc.scalar(0.2, unit='angstrom'),
        sc.scalar(16.0, unit='angstrom'),
    )

    workflow[unwrap.Choppers] = choppers
    workflow[unwrap.Ltotal] = ltotal
    workflow[unwrap.RawData] = raw_data

    # Compute time-of-flight
    tofs = workflow.compute(unwrap.TofData)
    assert {dim: tofs.sizes[dim] for dim in ltotal.sizes} == ltotal.sizes

    # Convert to wavelength
    graph = {**beamline(scatter=False), **elastic("tof")}
    wav_wfm = tofs.transform_coords("wavelength", graph=graph)

    # Compare the computed wavelengths to the true wavelengths
    for i in range(npulses):
        result_wav = wav_wfm['pulse', i].flatten(to='detector')
        result_tof = tofs['pulse', i].flatten(to='detector')
        for j in range(len(result_wav)):
            computed_tofs = result_tof[j].values.coords["tof"]
            # The two neutrons in the overlap region should have NaN tofs
            assert sc.isnan(computed_tofs[-2])
            assert sc.isnan(computed_tofs[-1])

            computed_wavelengths = result_wav[j].values.coords["wavelength"]
            assert sc.allclose(
                computed_wavelengths[:-2],
                true_wavelengths['pulse', i][:-2],
                rtol=sc.scalar(1e-02),
            )
            # The two neutrons in the overlap region should have NaN wavelengths
            assert sc.isnan(computed_wavelengths[-2])
            assert sc.isnan(computed_wavelengths[-1])
