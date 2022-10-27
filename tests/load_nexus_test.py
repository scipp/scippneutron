# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import pathlib
from typing import List, Type, Union, Callable
import warnings

from dateutil.parser import parse as parse_date
import numpy as np
import pytest
import scipp as sc

import scippneutron
from scippneutron.file_loading.load_nexus import _load_nexus_json
from .nexus_helpers import (
    NexusBuilder,
    EventData,
    Detector,
    Log,
    Sample,
    Source,
    Transformation,
    TransformationType,
    Link,
    Monitor,
    Chopper,
    in_memory_hdf5_file_with_two_nxentry,
)

# representative sample of UTF-8 test strings from
# https://www.w3.org/2001/06/utf-8-test/UTF-8-demo.html
UTF8_TEST_STRINGS = (
    "∮ E⋅da = Q,  n → ∞, ∑ f(i) = ∏ g(i), ∀x∈ℝ: ⌈x⌉ = −⌊−x⌋, α ∧ ¬β = ¬(¬α ∨ β)",
    "2H₂ + O₂ ⇌ 2H₂O, R = 4.7 kΩ, ⌀ 200 mm",
    "Σὲ γνωρίζω ἀπὸ τὴν κόψη",
)


def _timestamp(date: str):
    return parse_date(date).timestamp()


def test_raises_exception_if_multiple_nxentry_in_file():
    with in_memory_hdf5_file_with_two_nxentry() as nexus_file:
        with pytest.raises(RuntimeError):
            scippneutron.load_nexus(nexus_file)


def test_no_exception_if_single_nxentry_found_below_root():
    with in_memory_hdf5_file_with_two_nxentry() as nexus_file:
        # There are 2 NXentry in the file, but root is used
        # to specify which to load data from
        assert scippneutron.load_nexus(nexus_file, root='/entry_1') is None


def load_from_nexus(builder: NexusBuilder, *args, **kwargs)\
        -> Union[sc.Dataset, sc.DataArray, None]:
    with builder.file() as nexus_file:
        return scippneutron.load_nexus(nexus_file, *args, **kwargs)


def load_from_json(builder: NexusBuilder, *args, **kwargs)\
        -> Union[sc.Dataset, sc.DataArray, None]:
    loaded_data, _ = _load_nexus_json(builder.json_string, *args, **kwargs)
    return loaded_data


@pytest.fixture(params=[load_from_nexus, load_from_json])
def load_function(request) -> Callable:
    """
    Each test with this fixture is executed with load_nexus_json
    loading JSON output from the NexusBuilder, and with load_nexus
    loading in-memory NeXus output from the NexusBuilder
    """
    return request.param


@pytest.mark.parametrize('path_type', (str, pathlib.Path))
def test_loads_from_file(path_type):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='Skipped load', category=UserWarning)
        da = scippneutron.load_nexus(path_type(scippneutron.data.bigfake()))
    assert da.sizes == {'detector_id': 90000, 'tof': 1}
    assert 'tof' in da.coords
    assert 'position' in da.coords


def test_no_exception_if_single_nxentry_in_file(load_function: Callable):
    builder = NexusBuilder()
    assert load_function(builder) is None


def test_loads_data_from_single_event_data_group(load_function: Callable):
    event_time_offsets = np.array([456, 743, 347, 345, 632])
    event_data = EventData(
        event_id=np.array([1, 2, 3, 1, 3]),
        event_time_offset=event_time_offsets,
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, 5]),
    )

    builder = NexusBuilder()
    builder.add_event_data(event_data)

    loaded_data = load_function(builder)

    assert loaded_data.bins.constituents['data'].unit == 'counts'

    # Expect time of flight to match the values in the
    # event_time_offset dataset
    # May be reordered due to binning (hence np.sort)
    assert sc.identical(
        sc.sort(loaded_data.bins.concat('detector_id').values[0].coords['tof'],
                key="event"),
        sc.sort(sc.array(dims=["event"], values=event_time_offsets, unit=sc.units.ns),
                key="event"))

    counts_on_detectors = loaded_data.bins.sum()
    # No detector_number dataset in file so expect detector_id to be
    # binned according to whatever detector_ids are present in event_id
    # dataset: 2 on det 1, 1 on det 2, 2 on det 3
    expected_counts = np.array([[2], [1], [2]])
    assert sc.identical(
        counts_on_detectors.data,
        sc.array(dims=['detector_id', 'tof'],
                 unit='counts',
                 dtype='float32',
                 values=expected_counts,
                 variances=expected_counts))
    expected_detector_ids = np.array([1, 2, 3])
    assert np.array_equal(loaded_data.coords['detector_id'].values,
                          expected_detector_ids)


@pytest.mark.parametrize("unit,multiplier",
                         (("ns", 1), ("us", 10**3), ("ms", 10**6), ("s", 10**9)))
def test_loads_pulse_times_from_single_event_with_different_units(
        load_function: Callable, unit: str, multiplier: float):

    offsets = np.array([12, 34, 56, 78])
    zeros = np.array([12., 34., 56., 78.], dtype="float64")
    event_data = EventData(
        event_id=np.array([1, 2, 3, 4]),
        event_time_offset=offsets,
        event_time_zero=zeros,
        event_index=np.array([0, 3, 3, 4]),
        event_time_zero_unit=unit,
    )

    builder = NexusBuilder()
    builder.add_detector(
        Detector(detector_numbers=np.array([1, 2, 3, 4]), event_data=event_data))

    loaded_data = load_function(builder)

    for event, pulse_time in enumerate([12, 12, 12, 56]):
        _time = np.array("1970-01-01").astype("datetime64[ns]") \
                + np.array(pulse_time).astype("timedelta64[ns]") * multiplier

        # Allow 1ns difference for rounding errors between different routes
        assert all(
            np.abs(loaded_data.values[event].coords['pulse_time'].values -
                   _time) <= np.array(1).astype("timedelta64[ns]"))


@pytest.mark.parametrize("time_zero_offset,time_zero,time_zero_unit,expected_time", (
    ("1980-01-01T00:00:00.0", 30, "s", "1980-01-01T00:00:30.0"),
    ("1990-01-01T00:00:00.0", 5000, "ms", "1990-01-01T00:00:05.0"),
    ("2000-01-01T00:00:00.0", 3 * 10**6, "us", "2000-01-01T00:00:03.0"),
    ("2010-01-01T00:00:00.0", 12, "hour", "2010-01-01T12:00:00.0"),
))
def test_loads_pulse_times_with_combinations_of_offset_and_units(
        load_function: Callable, time_zero_offset: str, time_zero: float,
        time_zero_unit: str, expected_time: str):

    offsets = np.array([0])
    zeros = np.array([time_zero], dtype="float64")
    event_data = EventData(
        event_id=np.array([0]),
        event_time_offset=offsets,
        event_time_zero_offset=time_zero_offset,
        event_time_zero=zeros,
        event_index=np.array([0]),
        event_time_zero_unit=time_zero_unit,
    )

    builder = NexusBuilder()
    builder.add_detector(Detector(detector_numbers=np.array([0]),
                                  event_data=event_data))

    loaded_data = load_function(builder)

    _time = np.array(expected_time).astype("datetime64[ns]")

    # Allow 1ns difference for rounding errors between different routes
    assert np.abs(loaded_data.values[0].coords['pulse_time'].values[0] -
                  _time) <= np.array(1).astype("timedelta64[ns]")


def test_does_not_load_events_if_index_not_ordered(load_function: Callable):
    event_data_1 = EventData(
        event_id=np.array([0, 1]),
        event_time_offset=np.array([0, 1]),
        event_time_zero=np.array([0, 1]),
        event_index=np.array([2, 0]),
    )

    builder = NexusBuilder()
    builder.add_detector(
        Detector(detector_numbers=np.array([0, 1]), event_data=event_data_1))

    with pytest.warns(UserWarning, match="Invalid index in NXevent_data at "):
        load_function(builder)


def test_loads_pulse_times_from_multiple_event_data_groups(load_function: Callable):
    offsets = np.array([0, 0, 0, 0])

    zeros_1 = np.array([12 * 10**9, 34 * 10**9, 56 * 10**9, 78 * 10**9])
    zeros_2 = np.array([87 * 10**9, 65 * 10**9, 43 * 10**9, 21 * 10**9])

    event_data_1 = EventData(
        event_id=np.array([0, 1, 2, 3]),
        event_time_offset=offsets,
        event_time_zero=zeros_1,
        event_index=np.array([0, 3, 3, 4]),
    )
    event_data_2 = EventData(
        event_id=np.array([4, 5, 6, 7]),
        event_time_offset=offsets,
        event_time_zero=zeros_2,
        event_index=np.array([0, 3, 3, 4]),
    )

    builder = NexusBuilder()
    builder.add_detector(
        Detector(detector_numbers=np.array([0, 1, 2, 3]), event_data=event_data_1))
    builder.add_detector(
        Detector(detector_numbers=np.array([4, 5, 6, 7]), event_data=event_data_2))

    loaded_data = load_function(builder)

    for event, pulse_time in enumerate([12, 12, 12, 56, 87, 87, 87, 43]):
        _time = np.array("1970-01-01").astype("datetime64[ns]") \
                + np.array(pulse_time).astype("timedelta64[s]")

        assert sc.identical(
            loaded_data.values[event].coords['pulse_time'],
            sc.array(dims=["event"],
                     values=[_time],
                     unit=sc.units.ns,
                     dtype=sc.DType.datetime64))


def test_loads_data_from_multiple_event_data_groups(load_function: Callable):
    pulse_times = np.array([
        1600766730000000000, 1600766731000000000, 1600766732000000000,
        1600766733000000000
    ])
    event_time_offsets_1 = np.array([456, 743, 347, 345, 632])
    event_data_1 = EventData(
        event_id=np.array([1, 2, 3, 1, 3]),
        event_time_offset=event_time_offsets_1,
        event_time_zero=pulse_times,
        event_index=np.array([0, 3, 3, 5]),
    )
    detector_1_ids = np.array([0, 1, 2, 3])
    event_time_offsets_2 = np.array([682, 237, 941, 162, 52])
    event_data_2 = EventData(
        event_id=np.array([4, 5, 6, 4, 6]),
        event_time_offset=event_time_offsets_2,
        event_time_zero=pulse_times,
        event_index=np.array([0, 3, 3, 5]),
    )
    detector_2_ids = np.array([4, 5, 6, 7])

    builder = NexusBuilder()
    builder.add_detector(Detector(detector_1_ids, event_data_1))
    builder.add_detector(Detector(detector_2_ids, event_data_2))

    loaded_data = load_function(builder)

    # Expect time of flight to match the values in the
    # event_time_offset datasets
    # May be reordered due to binning (hence np.sort)
    assert np.array_equal(
        np.sort(loaded_data.bins.concat('detector_id').values[0].coords['tof'].values),
        np.sort(np.concatenate((event_time_offsets_1, event_time_offsets_2))))

    counts_on_detectors = loaded_data.bins.sum()
    # There are detector_number datasets in the NXdetector for each
    # NXevent_data, these are used for detector_id binning
    expected_counts = np.array([[0], [2], [1], [2], [2], [1], [2], [0]])
    assert np.array_equal(counts_on_detectors.data.values, expected_counts)
    expected_detector_ids = np.concatenate((detector_1_ids, detector_2_ids))
    assert np.array_equal(loaded_data.coords['detector_id'].values,
                          expected_detector_ids)


def test_skips_event_data_group_with_non_integer_event_ids(load_function: Callable):
    event_time_offsets = np.array([456, 743, 347, 345, 632])
    event_data = EventData(
        event_id=np.array([1.1, 2.2, 3.3, 1.1, 3.1]),
        event_time_offset=event_time_offsets,
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, 5]),
    )

    builder = NexusBuilder()
    builder.add_event_data(event_data)

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert loaded_data is None, "Expected no data to be loaded as " \
                                "event data has non integer event ids"


def test_skips_event_data_group_with_non_integer_detector_numbers(
        load_function: Callable):
    event_time_offsets = np.array([456, 743, 347, 345, 632])
    event_data = EventData(
        event_id=np.array([1, 2, 3, 1, 3]),
        event_time_offset=event_time_offsets,
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, 5]),
    )
    detector_numbers = np.array([0.1, 1.2, 2.3, 3.4])

    builder = NexusBuilder()
    builder.add_detector(Detector(detector_numbers, event_data))

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert loaded_data is None, "Expected no data to be loaded as " \
                                "detector has non integer detector numbers"


def test_skips_data_with_event_id_and_detector_number_type_unequal(
        load_function: Callable):
    event_time_offsets = np.array([456, 743, 347, 345, 632])
    event_data = EventData(
        event_id=np.array([1, 2, 3, 1, 3], dtype=np.int64),
        event_time_offset=event_time_offsets,
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, 5]),
    )
    detector_numbers = np.array([0, 1, 2, 3], dtype=np.int32)

    builder = NexusBuilder()
    builder.add_detector(Detector(detector_numbers, event_data))

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert loaded_data is None, "Expected no data to be loaded as event " \
                                "ids and detector numbers are of " \
                                "different types"


def test_loads_data_from_single_log_with_no_units(load_function: Callable):
    values = np.array([1, 2, 3])
    times = np.array([4, 5, 6])
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, values, times))

    loaded_data = load_function(builder)

    # Expect a sc.Dataset with log names as keys
    assert np.array_equal(loaded_data[name].data.values.values, values)


def test_loads_data_from_single_log_with_units(load_function: Callable):
    values = np.array([1.1, 2.2, 3.3])
    times = np.array([4.4, 5.5, 6.6])
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, values, times, value_units="m", time_units="s"))

    loaded_data = load_function(builder)

    # Expect a sc.Dataset with log names as keys
    assert np.allclose(loaded_data[name].data.values.values, values)
    assert loaded_data[name].data.values.unit == sc.units.m


def test_loads_data_from_multiple_logs(load_function: Callable):
    builder = NexusBuilder()
    log_1 = Log("test_log", np.array([1.1, 2.2, 3.3]), np.array([4.4, 5.5, 6.6]))
    log_2 = Log("test_log_2", np.array([123, 253, 756]), np.array([246, 1235, 2369]))
    builder.add_log(log_1)
    builder.add_log(log_2)

    loaded_data = load_function(builder)

    # Expect a sc.Dataset with log names as keys
    assert np.allclose(loaded_data[log_1.name].data.values.values, log_1.value)
    assert np.array_equal(loaded_data[log_2.name].data.values.values, log_2.value)


def test_loads_logs_with_non_supported_int_types(load_function: Callable):
    builder = NexusBuilder()
    log_int8 = Log("test_log_int8",
                   np.array([1, 2, 3]).astype(np.int8), np.array([4.4, 5.5, 6.6]))
    log_int16 = Log("test_log_int16",
                    np.array([123, 253, 756]).astype(np.int16),
                    np.array([246, 1235, 2369]))
    log_uint8 = Log("test_log_uint8",
                    np.array([1, 2, 3]).astype(np.uint8), np.array([4.4, 5.5, 6.6]))
    log_uint16 = Log("test_log_uint16",
                     np.array([123, 253, 756]).astype(np.uint16),
                     np.array([246, 1235, 2369]))
    logs = (log_int8, log_int16, log_uint8, log_uint16)
    for log in logs:
        builder.add_log(log)

    loaded_data = load_function(builder)

    # Expect a sc.Dataset with log names as keys
    for log in logs:
        assert np.allclose(loaded_data[log.name].data.values.values, log.value)


def test_loads_multidimensional_log(load_function: Callable):
    multidim_values = np.array([[1, 2, 3], [1, 2, 3]])
    times = np.array([4, 5])
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, multidim_values, times, value_units=''))

    loaded_data = load_function(builder)

    log = loaded_data[name].data.value
    expected = sc.DataArray(sc.array(dims=['time', 'dim_1'], values=multidim_values))
    expected.coords['time'] = sc.epoch(unit='s') + sc.array(
        dims=['time'], values=times, unit='s')
    assert sc.identical(log, expected)


def test_skips_log_with_no_value_dataset(load_function: Callable):
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, None, np.array([4, 5, 6])))

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert loaded_data is None


def test_loads_log_with_empty_value_and_time_datasets(load_function: Callable):
    empty_values = np.array([]).astype(np.int32)
    empty_times = np.array([]).astype(np.int32)
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, empty_values, empty_times))

    loaded_data = load_function(builder)

    assert loaded_data[name].data.values.sizes == {'time': 0}


def test_warns_given_log_with_mismatched_value_and_time(load_function: Callable):
    values = np.array([1, 2, 3]).astype(np.int32)
    # Note that if times exceeds length by 1 it is loaded as bin edges. It is unclear
    # if this is considered valid Nexus.
    times = np.array([1, 2, 3, 4, 5]).astype(np.int32)
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, values, times))

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert 'time' not in loaded_data['test_log'].value.coords


def test_loads_data_from_non_timeseries_log(load_function: Callable):
    values = np.array([1.1, 2.2, 3.3])
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, values))

    loaded_data = load_function(builder)

    assert np.allclose(loaded_data[name].data.values.values, values)


def test_loads_data_from_multiple_logs_with_same_name(load_function: Callable):
    values_1 = np.array([1.1, 2.2, 3.3])
    values_2 = np.array([4, 5, 6])
    name = "test_log"

    # Add one log to NXentry and the other to an NXdetector,
    # both have the same group name
    builder = NexusBuilder()
    builder.add_log(Log(name, values_1))
    builder.add_detector(Detector(log=Log(name, values_2)))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message='Skipped loading',
                                category=UserWarning)
        loaded_data = load_function(builder)

    assert len(loaded_data) == 2
    for key, da in loaded_data.items():
        assert key.endswith('test_log')
        if 'detector_0' in key:
            assert np.allclose(da.value.values, values_2)
        else:
            assert np.allclose(da.value.values, values_1)


def test_load_instrument_name(load_function: Callable):
    name = "INSTR"
    builder = NexusBuilder()
    builder.add_instrument(name)

    loaded_data = load_function(builder)

    assert loaded_data['instrument_name'].values == name


def test_load_experiment_title(load_function: Callable):
    title = "my experiment"
    builder = NexusBuilder()
    builder.add_title(title)

    loaded_data = load_function(builder)

    assert loaded_data['experiment_title'].values == title


def test_loads_event_and_log_data_from_single_file(load_function: Callable):
    event_time_offsets = np.array([456, 743, 347, 345, 632])
    event_data = EventData(
        event_id=np.array([1, 2, 3, 1, 3]),
        event_time_offset=event_time_offsets,
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, 5]),
    )

    log_1 = Log("test_log", np.array([1.1, 2.2, 3.3]), np.array([4.4, 5.5, 6.6]))
    log_2 = Log("test_log_2", np.array([123, 253, 756]), np.array([246, 1235, 2369]))

    builder = NexusBuilder()
    builder.add_event_data(event_data)
    builder.add_log(log_1)
    builder.add_log(log_2)

    loaded_data = load_function(builder)

    # Expect time of flight to match the values in the
    # event_time_offset dataset
    # May be reordered due to binning (hence np.sort)
    assert np.allclose(
        np.sort(loaded_data.bins.concat('detector_id').values[0].coords['tof'].values),
        np.sort(event_time_offsets))

    counts_on_detectors = loaded_data.bins.sum()
    # No detector_number dataset in file so expect detector_id to be
    # binned from the min to the max detector_id recorded in event_id
    # dataset: 2 on det 1, 1 on det 2, 2 on det 3
    expected_counts = np.array([[2], [1], [2]])
    assert np.allclose(counts_on_detectors.data.values, expected_counts)
    expected_detector_ids = np.array([1, 2, 3])
    assert np.allclose(loaded_data.coords['detector_id'].values, expected_detector_ids)
    assert "position" not in loaded_data.meta.keys(
    ), "The NXdetectors had no pixel position datasets so we " \
       "should not find 'position' coord"

    # Logs should have been added to the DataArray as attributes
    assert np.allclose(loaded_data.attrs[log_1.name].values.values, log_1.value)
    assert np.allclose(loaded_data.attrs[log_2.name].values.values, log_2.value)


def test_loads_pixel_positions_with_event_data(load_function: Callable):
    pulse_times = np.array([
        1600766730000000000, 1600766731000000000, 1600766732000000000,
        1600766733000000000
    ])
    event_time_offsets_1 = np.array([456, 743, 347, 345, 632])
    event_data_1 = EventData(
        event_id=np.array([1, 2, 3, 1, 3]),
        event_time_offset=event_time_offsets_1,
        event_time_zero=pulse_times,
        event_index=np.array([0, 3, 3, 5]),
    )
    detector_1_ids = np.array([0, 1, 2, 3])
    x_pixel_offset_1 = np.array([0.1, 0.2, 0.1, 0.2])
    y_pixel_offset_1 = np.array([0.1, 0.1, 0.2, 0.2])
    z_pixel_offset_1 = np.array([0.1, 0.2, 0.3, 0.4])

    event_time_offsets_2 = np.array([682, 237, 941, 162, 52])
    event_data_2 = EventData(
        event_id=np.array([4, 5, 6, 4, 6]),
        event_time_offset=event_time_offsets_2,
        event_time_zero=pulse_times,
        event_index=np.array([0, 3, 3, 5]),
    )
    # Multidimensional is fine as long as the shape of
    # the ids and the pixel offsets match
    detector_2_ids = np.array([[4, 5], [6, 7]])
    x_pixel_offset_2 = np.array([[1.1, 1.2], [1.1, 1.2]])
    y_pixel_offset_2 = np.array([[0.1, 0.1], [0.2, 0.2]])

    builder = NexusBuilder()
    offsets_units = "mm"
    builder.add_detector(
        Detector(detector_1_ids,
                 event_data_1,
                 x_offsets=x_pixel_offset_1,
                 y_offsets=y_pixel_offset_1,
                 z_offsets=z_pixel_offset_1,
                 offsets_unit=offsets_units))
    builder.add_detector(
        Detector(detector_2_ids,
                 event_data_2,
                 x_offsets=x_pixel_offset_2,
                 y_offsets=y_pixel_offset_2,
                 offsets_unit=offsets_units))

    loaded_data = load_function(builder)

    # If z offsets are missing they should be zero
    z_pixel_offset_2 = np.array([[0., 0.], [0., 0.]])
    expected_pixel_positions = np.array([
        np.concatenate((x_pixel_offset_1, x_pixel_offset_2.flatten())),
        np.concatenate((y_pixel_offset_1, y_pixel_offset_2.flatten())),
        np.concatenate((z_pixel_offset_1, z_pixel_offset_2.flatten()))
    ]).T
    assert np.allclose(loaded_data.coords['position'].values, expected_pixel_positions)
    assert loaded_data.meta['position'].unit == 'mm'


@pytest.mark.skip(reason="It is unclear if NXdetector with neither data nor event_data "
                  "needs to be considered valid.")
def test_loads_pixel_positions_without_event_data(load_function: Callable):
    """
    This is important in the live-data feature as geometry and event data
    are streamed separately
    """
    detector_1_ids = np.array([0, 1, 2, 3])
    x_pixel_offset_1 = np.array([0.1, 0.2, 0.1, 0.2])
    y_pixel_offset_1 = np.array([0.1, 0.1, 0.2, 0.2])
    z_pixel_offset_1 = np.array([0.1, 0.2, 0.3, 0.4])

    detector_2_ids = np.array([[4, 5], [6, 7]])
    x_pixel_offset_2 = np.array([[1.1, 1.2], [1.1, 1.2]])
    y_pixel_offset_2 = np.array([[0.1, 0.1], [0.2, 0.2]])

    builder = NexusBuilder()
    offsets_units = "mm"
    builder.add_detector(
        Detector(detector_numbers=detector_1_ids,
                 x_offsets=x_pixel_offset_1,
                 y_offsets=y_pixel_offset_1,
                 z_offsets=z_pixel_offset_1,
                 offsets_unit=offsets_units))
    builder.add_detector(
        Detector(detector_numbers=detector_2_ids,
                 x_offsets=x_pixel_offset_2,
                 y_offsets=y_pixel_offset_2,
                 offsets_unit=offsets_units))

    loaded_data = load_function(builder)

    # If z offsets are missing they should be zero
    z_pixel_offset_2 = np.array([[0., 0.], [0., 0.]])
    expected_pixel_positions = np.array([
        np.concatenate((x_pixel_offset_1, x_pixel_offset_2.flatten())),
        np.concatenate((y_pixel_offset_1, y_pixel_offset_2.flatten())),
        np.concatenate((z_pixel_offset_1, z_pixel_offset_2.flatten()))
    ]).T / 1_000  # Divide by 1000 for mm to metres
    assert np.allclose(loaded_data.coords['position'].values, expected_pixel_positions)
    assert loaded_data.meta[
               'base_position'].unit == sc.units.m, \
        "Expected positions to be converted to metres"


def test_loads_pixel_positions_with_no_units(load_function: Callable):
    pulse_times = np.array([
        1600766730000000000, 1600766731000000000, 1600766732000000000,
        1600766733000000000
    ])
    event_time_offsets = np.array([456, 743, 347, 345, 632])
    event_data = EventData(
        event_id=np.array([1, 2, 3, 1, 3]),
        event_time_offset=event_time_offsets,
        event_time_zero=pulse_times,
        event_index=np.array([0, 3, 3, 5]),
    )
    detector_ids = np.array([0, 1, 2, 3])
    x_pixel_offset = np.array([0.1, 0.2, 0.1, 0.2])
    y_pixel_offset = np.array([0.1, 0.1, 0.2, 0.2])
    z_pixel_offset = np.array([0.1, 0.2, 0.3, 0.4])

    builder = NexusBuilder()
    builder.add_detector(
        Detector(detector_ids,
                 event_data,
                 x_offsets=x_pixel_offset,
                 y_offsets=y_pixel_offset,
                 z_offsets=z_pixel_offset,
                 offsets_unit=None))

    loaded_data = load_function(builder)

    assert loaded_data.coords['position'].unit is None


def test_sample_position_at_origin_if_not_explicit_in_file(load_function: Callable):
    # The sample position is the origin of the coordinate
    # system in NeXus files.
    # If there is an NXsample in the file, but it has no "distance" dataset
    # or "depends_on" pointing to NXtransformations then it should be
    # assumed to be at the origin.
    builder = NexusBuilder()
    builder.add_sample(Sample("sample"))
    loaded_data = load_function(builder)

    origin = np.array([0, 0, 0])
    assert np.allclose(loaded_data["sample_position"].values, origin)


def test_loads_multiple_samples(load_function: Callable):
    # More than one sample in the file is possible, but we cannot guess
    # which to record as _the_ "sample_position",
    # instead record them in the form "<GROUP_NAME>_position"
    builder = NexusBuilder()
    sample_1_name = "sample_1"
    sample_2_name = "sample_2"
    builder.add_sample(Sample(sample_1_name))

    distance = 0.762
    units = "m"
    builder.add_sample(Sample(sample_2_name, distance=distance, distance_units=units))
    loaded_data = load_function(builder)

    origin = np.array([0, 0, 0])
    assert np.allclose(
        loaded_data[f"{sample_1_name}_position"].values, origin
    ), "Sample did not have explicit location so expect position " \
       "to be recorded as the origin"
    expected_position = np.array([0, 0, distance])
    assert np.allclose(loaded_data[f"{sample_2_name}_position"].values,
                       expected_position)


def test_skips_loading_source_if_more_than_one_in_file(load_function: Callable):
    # More than one source is a serious error in the file, so load_nexus will not
    # define a source_position rather than guessing which is the "correct" one.
    builder = NexusBuilder()
    builder.add_source(Source("source_1"))
    builder.add_source(Source("source_2"))
    loaded_data = load_function(builder)
    assert 'source_1' in loaded_data
    assert 'source_2' in loaded_data
    assert 'source_position' not in loaded_data


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
def test_component_position_from_distance_dataset_missing_unit(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        load_function: Callable):
    builder = NexusBuilder()
    distance = 4.2
    builder.add_component(
        component_class(component_name, distance=distance, distance_units=None))
    loaded_data = load_function(builder)
    assert loaded_data[f'{component_name}_position'].unit is None


@pytest.mark.parametrize("component_class,component_name", [(Sample, "sample"),
                                                            (Source, "source")])
@pytest.mark.parametrize("transform_type,value,value_units,expected_position",
                         ((TransformationType.ROTATION, 0.27, "rad", [0, 0, 0]),
                          (TransformationType.TRANSLATION, 230, "cm", [0, 0, 2.3])))
def test_loads_component_position_from_single_transformation(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        transform_type: TransformationType, value: float, value_units: str,
        expected_position: List[float], load_function: Callable):
    builder = NexusBuilder()
    transformation = Transformation(transform_type,
                                    vector=np.array([0, 0, 1]),
                                    value=np.array(value),
                                    value_units=value_units)
    builder.add_component(component_class(component_name, depends_on=transformation))
    loaded_data = load_function(builder)
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    # Resulting position will always be in metres, whatever units are
    # used in the NeXus file
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


@pytest.mark.parametrize("component_class,component_name", [(Sample, "sample"),
                                                            (Source, "source")])
@pytest.mark.parametrize("transform_type,value,value_units,expected_position",
                         ((TransformationType.ROTATION, 180, "deg", [-1, -2, 3]),
                          (TransformationType.TRANSLATION, 230, "cm", [1, 2, 5.3])))
def test_loads_component_position_from_single_transformation_with_offset(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        transform_type: TransformationType, value: float, value_units: str,
        expected_position: List[float], load_function: Callable):
    builder = NexusBuilder()
    transformation = Transformation(transform_type,
                                    vector=np.array([0, 0, 1]),
                                    value=np.array(value),
                                    value_units=value_units,
                                    offset=[1, 2, 3],
                                    offset_unit='m')
    builder.add_component(component_class(component_name, depends_on=transformation))
    loaded_data = load_function(builder)
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    # Resulting position will always be in metres, whatever units are
    # used in the NeXus file
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


@pytest.mark.parametrize("component_class,component_name", [(Sample, "sample"),
                                                            (Source, "source")])
@pytest.mark.parametrize("transform_type,value,value_units,expected_position",
                         ((TransformationType.ROTATION, 180, "deg", [-1, -2, 3]),
                          (TransformationType.TRANSLATION, 230, "cm", [1, 2, 5.3])))
def test_raises_if_offset_but_not_offset_units_found(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        transform_type: TransformationType, value: float, value_units: str,
        expected_position: List[float], load_function: Callable):
    builder = NexusBuilder()
    transformation = Transformation(transform_type,
                                    vector=np.array([0, 0, 1]),
                                    value=np.array(value),
                                    value_units=value_units,
                                    offset=[1, 2, 3],
                                    offset_unit=None)
    builder.add_component(component_class(component_name, depends_on=transformation))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message='Skipped loading',
                                category=UserWarning)
        loaded_data = load_function(builder)
    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
@pytest.mark.parametrize("transform_type,value,value_units,expected_position",
                         ((TransformationType.ROTATION, 0.27, "rad", [0, 0, 0]),
                          (TransformationType.TRANSLATION, 230, "cm", [0, 0, 2.3])))
def test_loads_component_position_from_log_transformation(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        transform_type: TransformationType, value: float, value_units: str,
        expected_position: List[float], load_function: Callable):
    builder = NexusBuilder()
    # Provide "time" data, the builder will write the transformation as
    # an NXlog
    transformation = Transformation(transform_type,
                                    vector=np.array([0, 0, 1]),
                                    value=np.array(value),
                                    value_units=value_units)
    builder.add_component(component_class(component_name, depends_on=transformation))
    loaded_data = load_function(builder)

    # Should load as usual despite the transformation being an NXlog
    # as it only has a single value
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
@pytest.mark.parametrize(
    "transform_type,value,value_units,expected_position",
    ((TransformationType.ROTATION, [0.27, 0.73], "rad", [0, 0, 0]),
     (TransformationType.TRANSLATION, [230, 310], "cm", [0, 0, 2.3])))
def test_loads_component_position_with_multi_value_log_transformation(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        transform_type: TransformationType, value: List[float], value_units: str,
        expected_position: float, load_function: Callable):
    builder = NexusBuilder()
    # Provide "time" data, the builder will write the transformation as
    # an NXlog. This would be encountered in a file from an experiment
    # involving a scan of a motion axis.
    transformation = Transformation(transform_type,
                                    vector=np.array([0, 0, 1]),
                                    value=np.array(value),
                                    time=np.array([1.3, 6.4]),
                                    time_units="s",
                                    value_units=value_units)
    builder.add_component(component_class(component_name, depends_on=transformation))
    loaded_data = load_function(builder)
    assert sc.identical(
        loaded_data[component_name].value['depends_on'].coords["time"],
        sc.Variable(dims=["time"],
                    values=[1300000000, 6400000000],
                    unit="ns",
                    dtype=sc.DType.datetime64))


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
def test_loads_component_position_with_multiple_multi_valued_log_transformations(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        load_function: Callable):
    builder = NexusBuilder()
    # Provide "time" data, the builder will write the transformation as
    # an NXlog. This would be encountered in a file from an experiment
    # involving a scan of a motion axis.
    t1 = Transformation(TransformationType.TRANSLATION,
                        vector=np.array([0, 0, 1]),
                        value=np.array([1, 10]),
                        time=np.array([0, 1]),
                        time_units="s",
                        value_units="m")

    t2 = Transformation(TransformationType.TRANSLATION,
                        vector=np.array([0, 0, 1]),
                        value=np.array([5, 50]),
                        time=np.array([0, 1]),
                        time_units="s",
                        value_units="m",
                        depends_on=t1)

    builder.add_component(component_class(component_name, depends_on=t2))
    loaded_data = load_function(builder)
    comp = loaded_data[component_name].value
    assert sc.identical(
        comp['depends_on'].coords["time"],
        sc.Variable(dims=["time"], values=[0, 1], unit="s", dtype=sc.DType.datetime64))


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
def test_multi_valued_log_transformations_time_axis_interpolated_and_trimmed(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        load_function: Callable):
    builder = NexusBuilder()
    t1 = Transformation(TransformationType.TRANSLATION,
                        vector=np.array([0, 0, 1]),
                        value=np.array([1, 10]),
                        time=np.array([0, 1]),
                        time_units="s",
                        value_units="m")
    t2 = Transformation(TransformationType.TRANSLATION,
                        vector=np.array([0, 0, 1]),
                        value=np.array([5, 50]),
                        time=np.array([0.1, 1.1]),
                        time_units="s",
                        value_units="m",
                        depends_on=t1)
    builder.add_component(component_class(component_name, depends_on=t2))

    loaded_data = load_function(builder)

    # Note the start at 100 ms, since there is no known value of t2 at 0 ms (when
    # t1 starts)
    comp = loaded_data[component_name].value
    assert sc.identical(
        comp['depends_on'].coords["time"],
        sc.array(dims=["time"],
                 values=[100, 1000, 1100],
                 unit="ms",
                 dtype=sc.DType.datetime64).to(unit="ns"))


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
@pytest.mark.parametrize("transform_type,value_units",
                         ((TransformationType.ROTATION, "deg"),
                          (TransformationType.TRANSLATION, "cm")))
def test_skips_component_position_with_empty_value_log_transformation(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        transform_type: TransformationType, value_units: str, load_function: Callable):
    builder = NexusBuilder()
    empty_value = np.array([])
    transformation = Transformation(transform_type,
                                    vector=np.array([0, 0, 1]),
                                    value=empty_value,
                                    time=np.array([1.3, 6.4]),
                                    time_units="s",
                                    value_units=value_units)
    builder.add_component(component_class(component_name, depends_on=transformation))
    with pytest.warns(UserWarning):
        load_function(builder)


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
def test_load_component_position_prefers_transform_over_distance(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        load_function: Callable):
    # The "distance" dataset gives the position along the z axis.
    # If there is a "depends_on" pointing to transformations then we
    # prefer to use that instead as it is likely to be more accurate; it
    # can define position and orientation in 3D.
    builder = NexusBuilder()
    transformation = Transformation(TransformationType.TRANSLATION,
                                    vector=np.array([0, 0, 1]),
                                    value=np.array(2.3),
                                    value_units="m")
    builder.add_component(
        component_class(component_name,
                        depends_on=transformation,
                        distance=4.2,
                        distance_units="m"))
    loaded_data = load_function(builder)

    expected_position = np.array([0, 0, transformation.value])
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
@pytest.mark.parametrize("transform_type",
                         (TransformationType.ROTATION, TransformationType.TRANSLATION))
def test_skips_component_position_from_transformation_missing_unit(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        transform_type: TransformationType, load_function: Callable):
    builder = NexusBuilder()
    transformation = Transformation(transform_type,
                                    np.array([0, 0, -1]),
                                    value=np.array(2.3))
    builder.add_component(component_class(component_name, depends_on=transformation))
    with pytest.warns(UserWarning):
        load_function(builder)


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
def test_loads_component_position_from_multiple_transformations(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        load_function: Callable):
    builder = NexusBuilder()
    transformation_1 = Transformation(TransformationType.ROTATION,
                                      np.array([0, 1, 0]),
                                      value=np.array(90),
                                      value_units="deg")
    transformation_2 = Transformation(TransformationType.TRANSLATION,
                                      np.array([0, 0, 1]),
                                      value=np.array(2.3),
                                      value_units="m",
                                      depends_on=transformation_1)
    builder.add_component(component_class(component_name, depends_on=transformation_2))
    loaded_data = load_function(builder)

    expected_position = np.array([2.3, 0, 0])
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


def test_skips_source_position_if_not_given_in_file(load_function: Callable):
    builder = NexusBuilder()
    builder.add_source(Source("source"))
    loaded_data = load_function(builder)
    assert 'source_position' not in loaded_data


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
def test_loads_component_position_from_distance_dataset(
        component_class: Union[Type[Source], Type[Sample]], component_name: str,
        load_function: Callable):
    # If the NXsource or NXsample contains a "distance" dataset
    # this gives the position along the z axis. If there was a "depends_on"
    # pointing to transformations then we'd use that instead as it is
    # likely to be more accurate; it can define position and orientation in 3D.
    builder = NexusBuilder()
    distance = 4.2
    units = "m"
    builder.add_component(
        component_class(component_name, distance=distance, distance_units=units))
    loaded_data = load_function(builder)

    expected_position = np.array([0, 0, distance])
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit(units)


def test_loads_source_position_dependent_on_sample_position(load_function: Callable):
    builder = NexusBuilder()
    transformation_0 = Transformation(TransformationType.ROTATION,
                                      np.array([0, 1, 0]),
                                      value=np.array(90),
                                      value_units="deg")
    transformation_1 = Transformation(TransformationType.TRANSLATION,
                                      np.array([0, 0, 1]),
                                      value=np.array(2.3),
                                      value_units="m",
                                      depends_on=transformation_0)
    builder.add_sample(Sample("sample", depends_on=transformation_1))
    transformation_2 = Transformation(
        TransformationType.TRANSLATION,
        np.array([0, 0, 1]),
        value=np.array(1.0),
        value_units="m",
        depends_on="/entry/sample/transformations/transform_1")
    builder.add_source(Source("source", depends_on=transformation_2))
    loaded_data = load_function(builder)

    expected_position = np.array([3.3, 0, 0])
    assert np.allclose(loaded_data["source_position"].values, expected_position)
    assert loaded_data["source_position"].unit == sc.Unit("m")


def test_loads_pixel_positions_with_transformations(load_function: Callable):
    pulse_times = np.array([
        1600766730000000000, 1600766731000000000, 1600766732000000000,
        1600766733000000000
    ])
    event_time_offsets_1 = np.array([456, 743, 347, 345, 632])
    event_data_1 = EventData(
        event_id=np.array([1, 2, 3, 1, 3]),
        event_time_offset=event_time_offsets_1,
        event_time_zero=pulse_times,
        event_index=np.array([0, 3, 3, 5]),
    )
    detector_1_ids = np.array([0, 1, 2, 3])
    x_pixel_offset_1 = np.array([0.1, 0.2, 0.1, 0.2])
    y_pixel_offset_1 = np.array([0.1, 0.1, 0.2, 0.2])
    z_pixel_offset_1 = np.array([0.1, 0.2, 0.3, 0.4])

    distance = 57  # cm
    transformation = Transformation(TransformationType.TRANSLATION,
                                    vector=np.array([0, 0, 1]),
                                    value=np.array(distance),
                                    value_units="cm")

    builder = NexusBuilder()
    builder.add_detector(
        Detector(detector_1_ids,
                 event_data_1,
                 x_offsets=x_pixel_offset_1,
                 y_offsets=y_pixel_offset_1,
                 z_offsets=z_pixel_offset_1,
                 offsets_unit="m",
                 depends_on=transformation))

    loaded_data = load_function(builder)

    expected_pixel_positions = np.array(
        [x_pixel_offset_1, y_pixel_offset_1, z_pixel_offset_1]).T
    assert np.allclose(loaded_data.meta['base_position'].values,
                       expected_pixel_positions)

    expected_transform = sc.spatial.affine_transform(unit=sc.units.m,
                                                     value=[[1, 0, 0, 0], [0, 1, 0, 0],
                                                            [0, 0, 1, 0.57],
                                                            [0, 0, 0, 1]])

    assert np.allclose(loaded_data.meta['position_transformations'].value.values,
                       expected_transform.values)

    assert np.allclose(
        (loaded_data.meta["position_transformations"].value *
         loaded_data.meta["base_position"]["detector_id", 0]).values,
        [0.1, 0.1, 0.67],
    )

    assert np.allclose(loaded_data.coords["position"]["detector_id", 0].values,
                       [0.1, 0.1, 0.67])


def test_loads_pixel_positions_with_multiple_transformations(load_function: Callable):
    event_data_1 = EventData(
        event_id=np.array([0, 0, 0, 0, 0]),
        event_time_offset=(np.array([456, 743, 347, 345, 632])),
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, 5]),
    )
    event_data_2 = EventData(
        event_id=np.array([1, 1, 1, 1, 1]),
        event_time_offset=(np.array([456, 743, 347, 345, 632])),
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, 5]),
    )

    transformation1 = Transformation(TransformationType.TRANSLATION,
                                     vector=np.array([0, 0, 1]),
                                     value=np.array(12),
                                     value_units="cm")
    transformation2 = Transformation(TransformationType.TRANSLATION,
                                     vector=np.array([0, 0, 1]),
                                     value=np.array(34),
                                     value_units="cm")

    builder = NexusBuilder()
    builder.add_detector(
        Detector(np.array([0]),
                 event_data_1,
                 x_offsets=np.array([0.1]),
                 y_offsets=np.array([0.1]),
                 z_offsets=np.array([0.1]),
                 offsets_unit="m",
                 depends_on=transformation1))

    builder.add_detector(
        Detector(np.array([1]),
                 event_data_2,
                 x_offsets=np.array([0.6]),
                 y_offsets=np.array([0.6]),
                 z_offsets=np.array([0.6]),
                 offsets_unit="m",
                 depends_on=transformation2))

    loaded_data = load_function(builder)

    assert np.allclose(
        (loaded_data.meta["position_transformations"]["detector_id", 0].value *
         loaded_data.meta["base_position"]["detector_id", 0]).values,
        [0.1, 0.1, 0.1 + 0.12],
    )
    assert np.allclose(
        (loaded_data.meta["position_transformations"]["detector_id", 1].value *
         loaded_data.meta["base_position"]["detector_id", 1]).values,
        [0.6, 0.6, 0.6 + 0.34],
    )


def test_links_to_event_data_group_are_ignored(load_function: Callable):
    event_time_offsets = np.array([456, 743, 347, 345, 632])
    event_data = EventData(
        event_id=np.array([1, 2, 3, 1, 3]),
        event_time_offset=event_time_offsets,
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, 5]),
    )

    builder = NexusBuilder()
    builder.add_event_data(event_data)
    builder.add_hard_link(Link("/entry/hard_link_to_events", "/entry/events_0"))
    builder.add_soft_link(Link("/entry/soft_link_to_events", "/entry/events_0"))

    loaded_data = load_function(builder)

    # The output Variable must contain the events from the added event
    # dataset with no duplicate data due to the links

    # Expect time of flight to match the values in the
    # event_time_offset dataset
    # May be reordered due to binning (hence np.sort)
    assert np.array_equal(
        np.sort(loaded_data.bins.concat('detector_id').values[0].coords['tof'].values),
        np.sort(event_time_offsets))

    counts_on_detectors = loaded_data.bins.sum()
    # No detector_number dataset in file so expect detector_id to be
    # binned according to whatever detector_ids are present in event_id
    # dataset: 2 on det 1, 1 on det 2, 2 on det 3
    expected_counts = np.array([[2], [1], [2]])
    assert np.array_equal(counts_on_detectors.data.values, expected_counts)
    expected_detector_ids = np.array([1, 2, 3])
    assert np.array_equal(loaded_data.coords['detector_id'].values,
                          expected_detector_ids)


def test_links_in_transformation_paths_are_followed(load_function: Callable):
    builder = NexusBuilder()
    distance = 13.6
    builder.add_component(Source("source"))
    builder.add_dataset_at_path(
        "/entry/transform", np.array(distance), {
            "vector": np.array([0, 0, 1]),
            "units": "m",
            "transformation_type": "translation",
            "depends_on": "."
        })
    builder.add_dataset_at_path("/entry/source/depends_on", "/entry/transform_link", {})
    builder.add_soft_link(Link("/entry/transform_link", "/entry/transform"))
    loaded_data = load_function(builder)

    assert np.allclose(loaded_data["source_position"].values, [0, 0, distance])
    # Resulting position will always be in metres, whatever units are
    # used in the NeXus file
    assert loaded_data["source_position"].unit == sc.Unit("m")


def test_relative_links_in_transformation_paths_are_followed(load_function: Callable):
    builder = NexusBuilder()
    distance = 13.6
    builder.add_component(Source("source"))
    builder.add_dataset_at_path(
        "/entry/transform1", np.array(distance), {
            "vector": np.array([0, 0, 1]),
            "units": "m",
            "transformation_type": "translation",
            "depends_on": "."
        })
    builder.add_dataset_at_path(
        "/entry/transform2", np.array(distance), {
            "vector": np.array([0, 0, 1]),
            "units": "m",
            "transformation_type": "translation",
            "depends_on": "transform1"
        })
    builder.add_dataset_at_path("/entry/source/depends_on", "/entry/transform2", {})
    loaded_data = load_function(builder)

    assert np.allclose(loaded_data["source_position"].values, [0, 0, 2 * distance])
    # Resulting position will always be in metres, whatever units are
    # used in the NeXus file
    assert loaded_data["source_position"].unit == sc.Unit("m")


def test_linked_datasets_are_found(load_function: Callable):
    event_data = EventData(
        event_id=np.array([1, 2, 3, 1, 3]),
        event_time_offset=np.array([456, 743, 347, 345, 632]),
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, 5]),
    )

    builder = NexusBuilder()
    builder.add_event_data(event_data)
    replaced_ids = np.array([1, 1, 1, 2, 3])
    builder.add_dataset_at_path("/entry/ids", replaced_ids, {})
    replaced_tofs = np.array([273, 546, 573, 812, 932])
    builder.add_dataset_at_path("/entry/tofs", replaced_tofs, {})
    # Replace dataset in the NXevent_data with a link to the
    # replacement dataset
    builder.add_hard_link(Link("/entry/events_0/event_id", "/entry/ids"))
    builder.add_soft_link(Link("/entry/events_0/event_time_offset", "/entry/tofs"))

    loaded_data = load_function(builder)

    assert np.array_equal(
        np.sort(loaded_data.bins.concat('detector_id').values[0].coords['tof'].values),
        np.sort(replaced_tofs))

    counts_on_detectors = loaded_data.bins.sum()
    expected_counts = np.array([[3], [1], [1]])
    assert np.array_equal(counts_on_detectors.data.values, expected_counts)
    expected_detector_ids = np.array([1, 2, 3])
    assert np.array_equal(loaded_data.coords['detector_id'].values,
                          expected_detector_ids)


def test_loads_sample_ub_matrix(load_function: Callable):
    builder = NexusBuilder()
    builder.add_component(Sample("sample", ub_matrix=np.ones(shape=[3, 3])))
    loaded_data = load_function(builder)
    sample = loaded_data['sample'].value
    assert sc.identical(
        sample['ub_matrix'],
        sc.spatial.linear_transform(value=np.ones(shape=[3, 3]),
                                    unit=sc.units.angstrom**-1))
    assert "orientation_matrix" not in sample


def test_loads_sample_orientation_matrix(load_function: Callable):
    builder = NexusBuilder()
    builder.add_component(Sample("sample", orientation_matrix=np.ones(shape=[3, 3])))
    loaded_data = load_function(builder)
    sample = loaded_data['sample'].value
    assert sc.identical(
        sample["orientation_matrix"],
        sc.spatial.linear_transform(value=np.ones(shape=[3, 3]), unit=sc.units.one))
    assert "ub_matrix" not in sample


def test_loads_multiple_sample_ub_matrix(load_function: Callable):
    builder = NexusBuilder()
    builder.add_component(Sample("sample1", ub_matrix=np.ones(shape=[3, 3])))
    builder.add_component(Sample("sample2", ub_matrix=np.identity(3)))
    builder.add_component(Sample("sample3"))  # No ub specified
    loaded_data = load_function(builder)
    assert sc.identical(
        loaded_data["sample1"].value["ub_matrix"],
        sc.spatial.linear_transform(value=np.ones(shape=[3, 3]),
                                    unit=sc.units.angstrom**-1))
    assert sc.identical(
        loaded_data["sample2"].value["ub_matrix"],
        sc.spatial.linear_transform(value=np.identity(3), unit=sc.units.angstrom**-1))
    assert "ub_matrix" not in loaded_data['sample3'].value


def test_warning_but_no_error_for_unrecognised_log_unit(load_function: Callable):
    values = np.array([1.1, 2.2, 3.3])
    times = np.array([4.4, 5.5, 6.6])
    name = "test_log"
    builder = NexusBuilder()
    unknown_unit = "elephants"
    builder.add_log(Log(name, values, times, value_units=unknown_unit, time_units="s"))

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    # Expect a sc.Dataset with log names as keys
    assert name in loaded_data


def test_start_and_end_times_appear_in_dataset_if_set(load_function: Callable):
    builder = NexusBuilder()
    builder.add_run_start_time("2001-01-01T00:00:00")
    builder.add_run_end_time("2002-02-02T00:00:00")

    loaded_data = load_function(builder)

    assert sc.identical(loaded_data["start_time"],
                        sc.DataArray(sc.scalar("2001-01-01T00:00:00")))
    assert sc.identical(loaded_data["end_time"],
                        sc.DataArray(sc.scalar("2002-02-02T00:00:00")))


@pytest.mark.parametrize("log_start,scaling_factor",
                         (("2000-01-01T01:00:00", 1000), ("2000-01-01T00:00:00", 0.001),
                          ("2010-01-01T00:00:00", None)))
def test_load_log_times(log_start: str, scaling_factor: float, load_function: Callable):

    times = np.array([0., 10., 20., 30., 40., 50.], dtype="float64")

    builder = NexusBuilder()
    builder.add_log(
        Log(name="test_log",
            value=np.zeros(shape=(len(times), )),
            time=times,
            start_time=log_start,
            scaling_factor=scaling_factor))

    loaded_data = load_function(builder)

    times_ns = sc.to_unit(
        sc.array(dims=["time"],
                 values=times * (scaling_factor or 1.0),
                 unit=sc.units.s,
                 dtype=sc.DType.float64), sc.units.ns).astype(sc.DType.int64)

    expected = sc.scalar(value=np.datetime64(log_start),
                         unit=sc.units.ns,
                         dtype=sc.DType.datetime64) + times_ns

    actual = loaded_data["test_log"].values.coords['time']

    diffs = np.abs(actual.values - expected.values)

    # Allow 1ns difference for rounding errors between different routes
    assert all(diffs <= np.array(1).astype("timedelta64[ns]"))


def test_load_log_times_when_logs_do_not_have_start_time(load_function: Callable):
    # If an NXLog doesn't have a start time attribute then 1970-01-01 should be
    # assumed instead

    times = np.array([-10., 0., 10., 20., 30., 40., 50.], dtype="float64")

    builder = NexusBuilder()
    builder.add_log(
        Log(name="test_log",
            value=np.zeros(shape=(len(times), )),
            time=times,
            start_time=None))

    loaded_data = load_function(builder)

    times_ns = sc.to_unit(
        sc.array(dims=["time"], values=times, unit=sc.units.s, dtype=sc.DType.float64),
        sc.units.ns).astype(sc.DType.int64)

    expected = sc.scalar(value=np.datetime64("1970-01-01T00:00:00"),
                         unit=sc.units.ns,
                         dtype=sc.DType.datetime64) + times_ns

    actual = loaded_data["test_log"].values.coords['time']

    diffs = np.abs(actual.values - expected.values)

    # Allow 1ns difference for rounding errors between different routes
    assert all(diffs <= np.array(1).astype("timedelta64[ns]"))


@pytest.mark.parametrize("units", ("ps", "ns", "us", "ms", "s", "minute", "hour"))
def test_adjust_log_times_with_different_time_units(units, load_function: Callable):

    times = [1, 2, 3]

    builder = NexusBuilder()
    builder.add_log(
        Log(name="test_log",
            value=np.zeros(shape=(len(times), )),
            time=np.array(times, dtype="float64"),
            time_units=units,
            start_time="2000-01-01T00:00:00"), )

    loaded_data = load_function(builder)

    times_ns = sc.to_unit(
        sc.array(dims=["time"], values=times, unit=units, dtype=sc.DType.int64),
        sc.units.ns)

    expected = sc.scalar(value=np.datetime64("2000-01-01T00:00:00"),
                         unit=sc.units.ns,
                         dtype=sc.DType.datetime64) + times_ns

    actual = loaded_data["test_log"].values.coords['time']

    diffs = np.abs(actual.values - expected.values)

    # Allow 1ns difference for rounding errors between different routes
    assert all(diffs <= np.array(1).astype("timedelta64[ns]"))


def test_nexus_file_with_invalid_nxlog_time_units_loads_dataset_as_non_datetime(
        load_function: Callable):
    builder = NexusBuilder()
    builder.add_log(
        Log(
            name="test_log_1",
            value=np.zeros(shape=(1, )),
            time=np.array([1.0]),
            time_units="m",  # Time in metres, should fail.
            start_time="1970-01-01T00:00:00"))

    loaded_data = load_function(builder)
    assert loaded_data['test_log_1'].value.coords['time'].dtype == sc.DType('float64')
    assert loaded_data['test_log_1'].value.coords['time'].unit == 'm'


def test_nexus_file_with_invalid_log_start_date_uses_epoch(load_function: Callable):
    builder = NexusBuilder()
    builder.add_log(
        Log(name="test_log_1",
            value=np.zeros(shape=(1, )),
            time=np.array([1]),
            start_time="this_isnt_a_valid_log_start_time"))
    builder.add_log(
        Log(name="test_log_2",
            value=np.zeros(shape=(1, )),
            time=np.array([1]),
            start_time="1970-01-01T00:00:00"))

    loaded_data = load_function(builder)
    assert sc.identical(
        loaded_data['test_log_1'].value.coords['time'],
        sc.epoch(unit='s') +
        sc.array(dims=['time'], values=[1], unit='s', dtype='int64'))


def test_extended_ascii_in_ascii_encoded_dataset(load_function: Callable):
    if load_function == load_from_json:
        pytest.skip("JSON serialiser can only serialize strings, not bytes.")

    builder = NexusBuilder()
    # When writing, if we use bytes h5py will write as ascii encoding
    # 0xb0 = degrees symbol in latin-1 encoding.
    builder.add_title(b"run at rot=90" + bytes([0xb0]))

    with pytest.warns(UserWarning, match="contains characters in extended ascii range"):
        loaded_data = load_function(builder)

        assert sc.identical(loaded_data["experiment_title"],
                            sc.DataArray(data=sc.scalar("run at rot=90°")))


@pytest.mark.parametrize("test_string", UTF8_TEST_STRINGS)
def test_utf8_encoded_dataset(load_function: Callable, test_string):
    builder = NexusBuilder()
    # When writing, if we use str h5py will write as utf8 encoding
    builder.add_title(test_string)

    loaded_data = load_function(builder)

    assert sc.identical(loaded_data["experiment_title"],
                        sc.DataArray(data=sc.scalar(test_string)))


def test_extended_ascii_in_ascii_encoded_attribute(load_function: Callable):
    if load_function == load_from_json:
        pytest.skip("JSON serialiser can only serialize strings, not bytes.")

    builder = NexusBuilder()
    # When writing, if we use bytes h5py will write as ascii encoding
    # 0xb0 = degrees symbol in latin-1 encoding.
    builder.add_log(Log(name="testlog", value_units=bytes([0xb0]), value=np.array([0])))

    with pytest.warns(UserWarning, match="contains characters in extended ascii range"):
        loaded_data = load_function(builder)

        assert loaded_data["testlog"].data.values.unit == sc.units.deg


# Can't use UTF-8 test strings as above for this test as the units need to be valid.
# Just do a single test with degrees.
def test_utf8_encoded_attribute(load_function: Callable):
    builder = NexusBuilder()
    # When writing, if we use str h5py will write as utf8 encoding
    builder.add_log(Log(name="testlog", value_units="°", value=np.array([0])))

    loaded_data = load_function(builder)
    assert loaded_data["testlog"].data.values.unit == sc.units.deg


def test_load_nexus_adds_single_tof_bin(load_function: Callable):
    event_time_offsets = np.array([456, 743, 347, 345, 632], dtype="float64")
    event_data = EventData(
        event_id=np.array([1, 2, 3, 1, 3]),
        event_time_offset=event_time_offsets,
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, 5]),
    )

    builder = NexusBuilder()
    builder.add_event_data(event_data)

    loaded_data = load_function(builder)

    # Size 2 for each of the two bin edges around a single bin
    assert loaded_data.coords["tof"].shape == (2, )

    # Assert bin edges correspond to smallest and largest+1 time-of-flights
    # in data.
    assert sc.identical(loaded_data.coords["tof"]["tof", 0],
                        sc.scalar(value=np.min(event_time_offsets), unit=sc.units.ns))
    assert sc.identical(
        loaded_data.coords["tof"]["tof", 1],
        sc.scalar(value=np.nextafter(np.max(event_time_offsets), float("inf")),
                  unit=sc.units.ns))


def test_nexus_file_with_choppers(load_function: Callable):
    builder = NexusBuilder()
    builder.add_instrument("dummy")
    builder.add_chopper(
        Chopper("chopper_1",
                distance=10.0,
                rotation_speed=60.0,
                rotation_units="Hz",
                distance_units="m"))
    loaded_data = load_function(builder)
    chopper = loaded_data['chopper_1'].value
    assert sc.identical(chopper["rotation_speed"], 60.0 * sc.Unit("Hz"))
    assert sc.identical(chopper["distance"], 10.0 * sc.Unit("m"))


def test_nexus_file_with_two_choppers(load_function: Callable):
    builder = NexusBuilder()
    builder.add_instrument("dummy")
    builder.add_chopper(
        Chopper("chopper_1",
                distance=11.0 * 1000,
                rotation_speed=65.0 / 1000,
                rotation_units="MHz",
                distance_units="mm"))
    builder.add_chopper(
        Chopper("chopper_2",
                distance=10.0,
                rotation_speed=60.0,
                rotation_units="Hz",
                distance_units="m"))
    loaded_data = load_function(builder)
    chopper1 = loaded_data['chopper_1'].value
    chopper2 = loaded_data['chopper_2'].value
    assert sc.identical(chopper1["rotation_speed"], (65.0 / 1000) * sc.Unit("MHz"))
    assert sc.identical(chopper1["distance"], (11.0 * 1000) * sc.Unit("mm"))
    assert sc.identical(chopper2["rotation_speed"], 60.0 * sc.Unit("Hz"))
    assert sc.identical(chopper2["distance"], 10.0 * sc.Unit("m"))


def test_load_monitors(load_function: Callable):
    builder = NexusBuilder()

    # Monitor with data
    builder.add_monitor(
        Monitor(
            name="monitor1",
            data=np.ones(shape=(2, 4, 6), dtype="float64"),
            axes=[
                ("event_index", np.arange(3, 5, dtype="float64")),
                ("period_index", np.arange(3, 7, dtype="float64")),
                ("time_of_flight", np.arange(3, 9, dtype="float64")),
            ],
        ))

    # Monitor with only one "axis" and only one data item
    builder.add_monitor(
        Monitor("monitor2",
                data=np.array([1.]),
                axes=[("time_of_flight", np.array([1.]))]))

    assert sc.identical(
        load_function(builder)["monitor1"].data.values,
        sc.DataArray(data=sc.ones(
            dims=["event_index", "period_index", "time_of_flight"],
            shape=(2, 4, 6),
            dtype=sc.DType.float64),
                     coords={
                         "event_index":
                         sc.Variable(dims=["event_index"],
                                     values=np.arange(3, 5, dtype="float64")),
                         "period_index":
                         sc.Variable(dims=["period_index"],
                                     values=np.arange(3, 7, dtype="float64")),
                         "time_of_flight":
                         sc.Variable(dims=["time_of_flight"],
                                     values=np.arange(3, 9, dtype="float64")),
                     }))

    assert sc.identical(
        load_function(builder)["monitor2"].data.values,
        sc.DataArray(data=sc.ones(dims=["time_of_flight"], shape=(1, )),
                     coords={
                         "time_of_flight":
                         sc.Variable(dims=["time_of_flight"], values=np.array([1.])),
                     }))


def test_load_monitors_with_event_mode_data(load_function: Callable):
    builder = NexusBuilder()

    # Monitor with data
    builder.add_monitor(
        Monitor(name="monitor1",
                data=np.ones(shape=(2, 3, 4), dtype="float64"),
                axes=[
                    ("event_index", np.array([0, 5], dtype="int64")),
                    ("period_index", np.array([2, 3, 4], dtype="int64")),
                    ("time_of_flight", np.array([3, 4, 5, 6], dtype="float64")),
                ],
                events=EventData(
                    event_id=np.array([0, 0, 0, 0, 0], dtype="int64"),
                    event_time_offset=np.array([1, 2, 3, 4, 5], dtype="float64"),
                    event_time_zero=np.array([0, 1], dtype="float64"),
                    event_index=np.array([0, 5], dtype="int64"),
                )))

    # Monitor with only one "axis" and only one data item
    builder.add_monitor(
        Monitor("monitor2",
                data=np.array([1.], dtype="float64"),
                axes=[("time_of_flight", np.array([1.], dtype="float64"))],
                events=EventData(
                    event_id=np.array([1, 1, 1, 1, 1], dtype="int64"),
                    event_time_offset=np.array([6, 7, 8, 9, 10], dtype="float64"),
                    event_time_zero=np.array([0, 1], dtype="float64"),
                    event_index=np.array([0, 5], dtype="int64"),
                )))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='Skipped load', category=UserWarning)
        mon_1_events = load_function(builder)["monitor1"].data.values
        mon_2_events = load_function(builder)["monitor2"].data.values

    assert sc.identical(
        mon_1_events.values[0].coords["tof"],
        sc.array(dims=["event"],
                 values=[1, 2, 3, 4, 5],
                 unit=sc.units.ns,
                 dtype=sc.DType.float64))
    assert sc.identical(
        mon_2_events.values[0].coords["tof"],
        sc.array(dims=["event"],
                 values=[6, 7, 8, 9, 10],
                 unit=sc.units.ns,
                 dtype=sc.DType.float64))


def test_load_monitor_with_transformation(load_function: Callable):
    builder = NexusBuilder()

    transformation = Transformation(TransformationType.TRANSLATION,
                                    vector=np.array([0, 0, 1]),
                                    value=np.array(6.5),
                                    value_units='m',
                                    offset=[1, 2, 3],
                                    offset_unit='m')

    # Monitor with data and transformation
    builder.add_monitor(
        Monitor(name="monitor1",
                data=np.ones(shape=(2, 4, 6), dtype="float64"),
                axes=[
                    ("event_index", np.arange(3, 5, dtype="float64")),
                    ("period_index", np.arange(3, 7, dtype="float64")),
                    ("time_of_flight", np.arange(3, 9, dtype="float64")),
                ],
                depends_on=transformation))

    loaded = load_function(builder)["monitor1"].data.values

    assert sc.identical(loaded.coords["position"],
                        sc.vector(value=[1., 2., 9.5], unit="m"))


def test_load_nexus_file_containing_empty_arrays(load_function: Callable):
    event_data = EventData(
        event_id=np.array([], dtype='int32'),
        event_time_offset=np.array([]),
        event_time_zero=np.array([]),
        event_index=np.array([]),
    )

    builder = NexusBuilder()
    builder.add_event_data(event_data)
    builder.add_log(Log("test_log", np.array([0, 1, 2]), np.array([4, 5, 6])))

    # Empty datasets should not stop other data (e.g. metadata) from being loaded.
    loaded_data = load_function(builder)
    assert "test_log" in loaded_data.attrs
    assert all(loaded_data.attrs["test_log"].value.values == np.array([0, 1, 2]))
