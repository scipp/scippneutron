# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from functools import partial

import numpy as np
import pytest
import scipp as sc
import tof as tof_pkg

from scippneutron.chopper import DiskChopper
from scippneutron.conversion.graph.beamline import beamline
from scippneutron.conversion.graph.tof import elastic
from scippneutron.tof import chopper_cascade, fakes, unwrap

sl = pytest.importorskip('sciline')


@pytest.fixture
def disk_choppers():
    wfm1 = DiskChopper(
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

    wfm2 = DiskChopper(
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

    foc1 = DiskChopper(
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

    foc2 = DiskChopper(
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

    return {"wfm1": wfm1, "wfm2": wfm2, "foc1": foc1, "foc2": foc2}


@pytest.mark.parametrize("npulses", [1, 2])
@pytest.mark.parametrize(
    "ltotal",
    [
        sc.array(dims=['detector_number'], values=[26.0], unit='m'),
        sc.array(dims=['detector_number'], values=[26.0, 25.5], unit='m'),
        sc.array(
            dims=['y', 'x'], values=[[26.0, 25.1, 26.33], [25.9, 26.0, 25.7]], unit='m'
        ),
    ],
)
def test_v20_compute_wavelengths_from_wfm(disk_choppers, npulses, ltotal):
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
        dims=['event'], values=[2.75, 4.2, 5.4, 6.5, 7.6, 8.75], unit='angstrom'
    )
    birth_times = sc.full(sizes=wavelengths.sizes, value=1.5, unit='ms')
    ess_beamline = fakes.FakeBeamlineEss(
        choppers=choppers,
        monitors=monitors,
        run_length=sc.scalar(1 / 14, unit="s") * npulses,
        events_per_pulse=len(wavelengths),
        source=partial(
            tof_pkg.Source.from_neutrons,
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
        sc.scalar(3.4, unit='ms'),
    )
    workflow[unwrap.SourceWavelengthRange] = (
        sc.scalar(0.2, unit='angstrom'),
        sc.scalar(10.0, unit='angstrom'),
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
