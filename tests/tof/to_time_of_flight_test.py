import pytest
import scipp as sc
from scipp.testing import assert_identical

from scippneutron.tof import unwrap


def test_to_time_of_flight_raises_if_Ltotal_incompatible() -> None:
    unwrapped = sc.DataArray(
        data=sc.scalar(1.0, unit='counts'),
        coords={
            'time_offset': sc.scalar(0.3, unit='s'),
            'Ltotal': sc.scalar(3.0, unit='m'),
        },
    )
    origin = unwrap.TimeOfFlightOrigin(
        time=sc.scalar(0.1, unit='s'), distance=sc.scalar(1.0, unit='m')
    )
    ltotal = sc.scalar(2.0, unit='m')
    with pytest.raises(ValueError, match='Ltotal'):
        unwrap.to_time_of_flight(unwrapped, origin=origin, ltotal=ltotal)


def test_to_time_of_flight_subtracts_from_time_offset() -> None:
    unwrapped = sc.DataArray(
        data=sc.scalar(1.0, unit='counts'),
        coords={
            'time_offset': sc.scalar(3.0, unit='s'),
            'Ltotal': sc.scalar(3.0, unit='m'),
        },
    )
    origin = unwrap.TimeOfFlightOrigin(
        time=sc.scalar(1.0, unit='s'), distance=sc.scalar(1.0, unit='m')
    )
    ltotal = sc.scalar(3.0, unit='m')
    result = unwrap.to_time_of_flight(unwrapped, origin=origin, ltotal=ltotal)
    assert_identical(result.coords['tof'], sc.scalar(2.0, unit='s'))


def test_to_time_of_flight_event_mode() -> None:
    content = sc.DataArray(
        data=sc.array(dims=['event'], values=[1.0, 2.0], unit='counts'),
        coords={
            'time_offset': sc.array(dims=['event'], values=[3.0, 4.0], unit='s'),
            'pulse_time': sc.array(dims=['event'], values=[10.0, 20.0], unit='s'),
        },
    )
    unwrapped = sc.DataArray(data=sc.bins(begin=sc.index(0), dim='event', data=content))
    origin = unwrap.TimeOfFlightOrigin(
        time=sc.scalar(1.0, unit='s'), distance=sc.scalar(1.0, unit='m')
    )
    ltotal = sc.scalar(3.0, unit='m')
    result = unwrap.to_time_of_flight(unwrapped, origin=origin, ltotal=ltotal)

    assert_identical(result.coords['Ltotal'], sc.scalar(3.0 - 1.0, unit='m'))
    assert_identical(
        result.value.coords['tof'],
        sc.array(dims=['event'], values=[2.0, 3.0], unit='s'),
    )
    assert_identical(
        result.value.coords['time_zero'],
        sc.array(dims=['event'], values=[11.0, 21.0], unit='s'),
    )
