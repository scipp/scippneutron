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
    chopper = DiskChopper(
        frequency=sc.scalar(14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(297 - 180 - 90 + 30, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 6.174], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[15.0 * 0.5], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[27.6 * 0.5], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    return {"chopper1": chopper}


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
@pytest.mark.parametrize("time_offset_unit", ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize("distance_unit", ['m', 'mm'])
def test_compute_tofs(disk_choppers, npulses, ltotal, time_offset_unit, distance_unit):
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
        dims=['event'], values=[6.2, 5.6, 4.6, 4.0, 3.5, 3.0], unit='angstrom'
    )
    birth_times = sc.array(
        dims=['event'], values=[0.0, 0.0, 2.5, 2.5, 4.0, 4.0], unit='ms'
    )
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

    print("NPULSES", npulses)

    raw_data = sc.concat(
        [ess_beamline.get_monitor(key) for key in monitors.keys()],
        dim='detector',
    ).fold(dim='detector', sizes=ltotal.sizes)

    print("RAW DATA 1", raw_data.sizes)

    # Save the true tofs for later
    # model_result = []
    # for key in monitors.keys():
    #     data = ess_beamline.model_result[key].data
    #     model_result.append(data[~data.masks["blocked_by_others"]])
    model_result = sc.concat(
        [ess_beamline.model_result[key].data for key in monitors.keys()], dim='detector'
    ).fold(dim='detector', sizes=ltotal.sizes)
    true_tofs = model_result.coords["toa"] - model_result.coords["time"]
    print("TRUE TOFS", true_tofs)

    # Convert the time offset to the unit requested by the test
    raw_data.bins.coords["event_time_offset"] = raw_data.bins.coords[
        "event_time_offset"
    ].to(unit=time_offset_unit, copy=False)

    raw_data.coords['Ltotal'] = ltotal.to(unit=distance_unit, copy=False)

    print("RAW DATA", raw_data.sizes)

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
        unwrap.providers
        # unwrap.unwrap_providers()
        # + unwrap.time_of_flight_providers()
        # + unwrap.time_of_flight_origin_from_choppers_providers(wfm=True)
    )
    workflow[unwrap.PulsePeriod] = sc.reciprocal(ess_beamline.source.frequency)
    workflow[unwrap.PulseStride] = 1

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
    workflow[unwrap.Ltotal] = raw_data.coords['Ltotal']
    workflow[unwrap.RawData] = raw_data

    # Compute time-of-flight
    tofs = workflow.compute(unwrap.TofData)
    assert {dim: tofs.sizes[dim] for dim in ltotal.sizes} == ltotal.sizes

    # print(tofs)
    # print(true_tofs)

    assert sc.allclose(
        tofs,
        true_tofs,
        rtol=sc.scalar(1e-02),
    )
    # print("TOFS", tofs)

    # # # Convert to wavelength
    # # graph = {**beamline(scatter=False), **elastic("tof")}
    # # wav_wfm = tofs.transform_coords("wavelength", graph=graph)

    # # Compare the computed wavelengths to the true wavelengths
    # for i in range(npulses):
    #     result = tofs['pulse', i].flatten(to='detector')
    #     # print("RESULT", result, len(result))
    #     # print("result[j]", result[0])
    #     # print("res.values", result[0].values)
    #     for j in range(len(result)):
    #         computed_tofs = result[j].values.coords["tof"]
    #         # print(computed_wavelengths)
    #         # print(true_wavelengths['pulse', i])
    #         assert sc.allclose(
    #             computed_tofs,
    #             true_tofs['pulse', i],
    #             rtol=sc.scalar(1e-02),
    #         )


# import pytest
# import scipp as sc
# from scipp.testing import assert_identical

# from scippneutron.tof import unwrap


# def test_to_time_of_flight_raises_if_Ltotal_incompatible() -> None:
#     unwrapped = sc.DataArray(
#         data=sc.scalar(1.0, unit='counts'),
#         coords={
#             'time_offset': sc.scalar(0.3, unit='s'),
#             'Ltotal': sc.scalar(3.0, unit='m'),
#         },
#     )
#     origin = unwrap.TimeOfFlightOrigin(
#         time=sc.scalar(0.1, unit='s'), distance=sc.scalar(1.0, unit='m')
#     )
#     ltotal = sc.scalar(2.0, unit='m')
#     with pytest.raises(ValueError, match='Ltotal'):
#         unwrap.to_time_of_flight(unwrapped, origin=origin, ltotal=ltotal)


# def test_to_time_of_flight_subtracts_from_time_offset() -> None:
#     unwrapped = sc.DataArray(
#         data=sc.scalar(1.0, unit='counts'),
#         coords={
#             'time_offset': sc.scalar(3.0, unit='s'),
#             'Ltotal': sc.scalar(3.0, unit='m'),
#         },
#     )
#     origin = unwrap.TimeOfFlightOrigin(
#         time=sc.scalar(1.0, unit='s'), distance=sc.scalar(1.0, unit='m')
#     )
#     ltotal = sc.scalar(3.0, unit='m')
#     result = unwrap.to_time_of_flight(unwrapped, origin=origin, ltotal=ltotal)
#     assert_identical(result.coords['tof'], sc.scalar(2.0, unit='s'))


# def test_to_time_of_flight_event_mode() -> None:
#     content = sc.DataArray(
#         data=sc.array(dims=['event'], values=[1.0, 2.0], unit='counts'),
#         coords={
#             'time_offset': sc.array(dims=['event'], values=[3.0, 4.0], unit='s'),
#             'pulse_time': sc.array(dims=['event'], values=[10.0, 20.0], unit='s'),
#         },
#     )
#     unwrapped = sc.DataArray(data=sc.bins(begin=sc.index(0), dim='event', data=content))
#     origin = unwrap.TimeOfFlightOrigin(
#         time=sc.scalar(1.0, unit='s'), distance=sc.scalar(1.0, unit='m')
#     )
#     ltotal = sc.scalar(3.0, unit='m')
#     result = unwrap.to_time_of_flight(unwrapped, origin=origin, ltotal=ltotal)

#     assert_identical(result.coords['Ltotal'], sc.scalar(3.0 - 1.0, unit='m'))
#     assert_identical(
#         result.value.coords['tof'],
#         sc.array(dims=['event'], values=[2.0, 3.0], unit='s'),
#     )
#     assert_identical(
#         result.value.coords['time_zero'],
#         sc.array(dims=['event'], values=[11.0, 21.0], unit='s'),
#     )
