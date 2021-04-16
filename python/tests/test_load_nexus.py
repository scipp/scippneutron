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
    in_memory_hdf5_file_with_two_nxentry,
)
import numpy as np
import pytest
import scippneutron
import scipp as sc
from typing import List, Type, Union, Callable


def test_raises_exception_if_multiple_nxentry_in_file():
    with in_memory_hdf5_file_with_two_nxentry() as nexus_file:
        with pytest.raises(RuntimeError):
            scippneutron.load_nexus(nexus_file)


def test_no_exception_if_single_nxentry_found_below_root():
    with in_memory_hdf5_file_with_two_nxentry() as nexus_file:
        # There are 2 NXentry in the file, but root is used
        # to specify which to load data from
        assert scippneutron.load_nexus(nexus_file, root='/entry_1') is None


def load_from_nexus(builder: NexusBuilder) -> sc.Variable:
    with builder.file() as nexus_file:
        return scippneutron.load_nexus(nexus_file)


def load_from_json(builder: NexusBuilder) -> sc.Variable:
    return scippneutron.load_nexus_json(builder.json_string)


@pytest.fixture(params=[load_from_nexus, load_from_json])
def load_function(request) -> Callable:
    """
    Each test with this fixture is executed with load_nexus_json
    loading JSON output from the NexusBuilder, and with load_nexus
    loading in-memory NeXus output from the NexusBuilder
    """
    return request.param


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

    # Expect time of flight to match the values in the
    # event_time_offset dataset
    # May be reordered due to binning (hence np.sort)
    assert np.array_equal(
        np.sort(
            loaded_data.bins.concatenate(
                'detector_id').values.coords['tof'].values),
        np.sort(event_time_offsets))

    counts_on_detectors = loaded_data.bins.sum()
    # No detector_number dataset in file so expect detector_id to be
    # binned according to whatever detector_ids are present in event_id
    # dataset: 2 on det 1, 1 on det 2, 2 on det 3
    expected_counts = np.array([2, 1, 2])
    assert np.array_equal(counts_on_detectors.data.values, expected_counts)
    expected_detector_ids = np.array([1, 2, 3])
    assert np.array_equal(loaded_data.coords['detector_id'].values,
                          expected_detector_ids)


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
        np.sort(
            loaded_data.bins.concatenate(
                'detector_id').values.coords['tof'].values),
        np.sort(np.concatenate((event_time_offsets_1, event_time_offsets_2))))

    counts_on_detectors = loaded_data.bins.sum()
    # There are detector_number datasets in the NXdetector for each
    # NXevent_data, these are used for detector_id binning
    expected_counts = np.array([0, 2, 1, 2, 2, 1, 2, 0])
    assert np.array_equal(counts_on_detectors.data.values, expected_counts)
    expected_detector_ids = np.concatenate((detector_1_ids, detector_2_ids))
    assert np.array_equal(loaded_data.coords['detector_id'].values,
                          expected_detector_ids)


def test_skips_event_data_group_with_non_integer_event_ids(
        load_function: Callable):
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
    assert np.array_equal(loaded_data[name].data.values.coords['time'].values,
                          times)


def test_loads_data_from_single_log_with_units(load_function: Callable):
    values = np.array([1.1, 2.2, 3.3])
    times = np.array([4.4, 5.5, 6.6])
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, values, times, value_units="m", time_units="s"))

    loaded_data = load_function(builder)

    # Expect a sc.Dataset with log names as keys
    assert np.allclose(loaded_data[name].data.values.values, values)
    assert np.allclose(loaded_data[name].data.values.coords['time'].values,
                       times)
    assert loaded_data[name].data.values.unit == sc.units.m
    assert loaded_data[name].data.values.coords['time'].unit == sc.units.s


def test_loads_data_from_multiple_logs(load_function: Callable):
    builder = NexusBuilder()
    log_1 = Log("test_log", np.array([1.1, 2.2, 3.3]),
                np.array([4.4, 5.5, 6.6]))
    log_2 = Log("test_log_2", np.array([123, 253, 756]),
                np.array([246, 1235, 2369]))
    builder.add_log(log_1)
    builder.add_log(log_2)

    loaded_data = load_function(builder)

    # Expect a sc.Dataset with log names as keys
    assert np.allclose(loaded_data[log_1.name].data.values.values, log_1.value)
    assert np.allclose(
        loaded_data[log_1.name].data.values.coords['time'].values, log_1.time)
    assert np.array_equal(loaded_data[log_2.name].data.values.values,
                          log_2.value)
    assert np.array_equal(
        loaded_data[log_2.name].data.values.coords['time'].values, log_2.time)


def test_loads_logs_with_non_supported_int_types(load_function: Callable):
    builder = NexusBuilder()
    log_int8 = Log("test_log_int8",
                   np.array([1, 2, 3]).astype(np.int8),
                   np.array([4.4, 5.5, 6.6]))
    log_int16 = Log("test_log_int16",
                    np.array([123, 253, 756]).astype(np.int16),
                    np.array([246, 1235, 2369]))
    log_uint8 = Log("test_log_uint8",
                    np.array([1, 2, 3]).astype(np.uint8),
                    np.array([4.4, 5.5, 6.6]))
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
        assert np.allclose(
            loaded_data[log.name].data.values.coords['time'].values, log.time)


def test_skips_multidimensional_log(load_function: Callable):
    # Loading NXlogs with more than 1 dimension is not yet implemented
    # We need to come up with a sensible approach to labelling the dimensions

    multidim_values = np.array([[1, 2, 3], [1, 2, 3]])
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, multidim_values, np.array([4, 5, 6])))

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert loaded_data is None


def test_skips_log_with_no_value_dataset(load_function: Callable):
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, None, np.array([4, 5, 6])))

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert loaded_data is None


def test_skips_log_with_empty_value_and_time_datasets(load_function: Callable):
    empty_values = np.array([]).astype(np.int32)
    empty_times = np.array([]).astype(np.int32)
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, empty_values, empty_times))

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert loaded_data is None


def test_skips_log_with_mismatched_value_and_time(load_function: Callable):
    values = np.array([1, 2, 3]).astype(np.int32)
    times = np.array([1, 2, 3, 4]).astype(np.int32)
    name = "test_log"
    builder = NexusBuilder()
    builder.add_log(Log(name, values, times))

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert loaded_data is None


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
    builder.add_detector(Detector(np.array([1, 2, 3]), log=Log(name,
                                                               values_2)))

    loaded_data = load_function(builder)

    # Expect two logs
    # The log group name for one of them should have been prefixed with
    # its the parent group name to avoid duplicate log names
    if np.allclose(loaded_data[name].data.values.values, values_1):
        # Then the other log should be
        assert np.allclose(
            loaded_data[f"detector_1_{name}"].data.values.values, values_2)
    elif np.allclose(loaded_data[name].data.values.values, values_2):
        # Then the other log should be
        assert np.allclose(loaded_data[f"entry_{name}"].data.values.values,
                           values_1)
    else:
        assert False


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

    log_1 = Log("test_log", np.array([1.1, 2.2, 3.3]),
                np.array([4.4, 5.5, 6.6]))
    log_2 = Log("test_log_2", np.array([123, 253, 756]),
                np.array([246, 1235, 2369]))

    builder = NexusBuilder()
    builder.add_event_data(event_data)
    builder.add_log(log_1)
    builder.add_log(log_2)

    loaded_data = load_function(builder)

    # Expect time of flight to match the values in the
    # event_time_offset dataset
    # May be reordered due to binning (hence np.sort)
    assert np.allclose(
        np.sort(
            loaded_data.bins.concatenate(
                'detector_id').values.coords['tof'].values),
        np.sort(event_time_offsets))

    counts_on_detectors = loaded_data.bins.sum()
    # No detector_number dataset in file so expect detector_id to be
    # binned from the min to the max detector_id recorded in event_id
    # dataset: 2 on det 1, 1 on det 2, 2 on det 3
    expected_counts = np.array([2, 1, 2])
    assert np.allclose(counts_on_detectors.data.values, expected_counts)
    expected_detector_ids = np.array([1, 2, 3])
    assert np.allclose(loaded_data.coords['detector_id'].values,
                       expected_detector_ids)
    assert "position" not in loaded_data.coords.keys(
    ), "The NXdetectors had no pixel position datasets so we " \
       "should not find 'position' coord"

    # Logs should have been added to the DataArray as attributes
    assert np.allclose(loaded_data.attrs[log_1.name].values.values,
                       log_1.value)
    assert np.allclose(
        loaded_data.attrs[log_1.name].values.coords['time'].values, log_1.time)
    assert np.allclose(loaded_data.attrs[log_2.name].values.values,
                       log_2.value)
    assert np.allclose(
        loaded_data.attrs[log_2.name].values.coords['time'].values, log_2.time)


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
    ]).T / 1_000  # Divide by 1000 for mm to metres
    assert np.allclose(loaded_data.coords['position'].values,
                       expected_pixel_positions)
    assert loaded_data.coords[
        'position'].unit == sc.units.m, "Expected positions " \
                                        "to be converted to metres"


def test_skips_loading_pixel_positions_with_non_matching_shape(
        load_function: Callable):
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
    # The size of the ids and the pixel offsets do not match
    detector_2_ids = np.array([[4, 5, 6, 7, 8]])
    x_pixel_offset_2 = np.array([1.1, 1.2, 1.1, 1.2])
    y_pixel_offset_2 = np.array([0.1, 0.1, 0.2, 0.2])

    builder = NexusBuilder()
    builder.add_detector(
        Detector(detector_1_ids,
                 event_data_1,
                 x_offsets=x_pixel_offset_1,
                 y_offsets=y_pixel_offset_1,
                 z_offsets=z_pixel_offset_1,
                 offsets_unit="m"))
    builder.add_detector(
        Detector(detector_2_ids,
                 event_data_2,
                 x_offsets=x_pixel_offset_2,
                 y_offsets=y_pixel_offset_2,
                 offsets_unit="m"))

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert "position" not in loaded_data.coords.keys(
    ), "One of the NXdetectors pixel positions arrays did not match the " \
       "size of its detector ids so we should not find 'position' coord"
    # Even though detector_1's offsets and ids are matches in size, we do not
    # load them as the "position" coord would not have positions for all
    # the detector ids (loading event data from all detectors is prioritised).


def test_skips_loading_pixel_positions_with_no_units(load_function: Callable):
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

    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert "position" not in loaded_data.coords.keys()


def test_sample_position_at_origin_if_not_explicit_in_file(
        load_function: Callable):
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
    builder.add_sample(
        Sample(sample_2_name, distance=distance, distance_units=units))
    loaded_data = load_function(builder)

    origin = np.array([0, 0, 0])
    assert np.allclose(
        loaded_data[f"{sample_1_name}_position"].values, origin
    ), "Sample did not have explicit location so expect position " \
       "to be recorded as the origin"
    expected_position = np.array([0, 0, distance])
    assert np.allclose(loaded_data[f"{sample_2_name}_position"].values,
                       expected_position)


def test_skips_loading_source_if_more_than_one_in_file(
        load_function: Callable):
    # More than one source is a serious error in the file, so
    # load_nexus will display a warning and skip loading any sample rather
    # than guessing which is the "correct" one.
    builder = NexusBuilder()
    builder.add_source(Source("source_1"))
    builder.add_source(Source("source_2"))
    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)
    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
def test_skips_component_position_from_distance_dataset_missing_unit(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str, load_function: Callable):
    builder = NexusBuilder()
    distance = 4.2
    builder.add_component(
        component_class(component_name, distance=distance,
                        distance_units=None))
    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)
    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         [(Sample, "sample"), (Source, "source")])
@pytest.mark.parametrize(
    "transform_type,value,value_units,expected_position",
    ((TransformationType.ROTATION, 0.27, "rad", [0, 0, 0]),
     (TransformationType.TRANSLATION, 230, "cm", [0, 0, 2.3])))
def test_loads_component_position_from_single_transformation(
        component_class: Union[Type[Source],
                               Type[Sample]], component_name: str,
        transform_type: TransformationType, value: float, value_units: str,
        expected_position: List[float], load_function: Callable):
    builder = NexusBuilder()
    transformation = Transformation(transform_type,
                                    vector=np.array([0, 0, -1]),
                                    value=np.array([value]),
                                    value_units=value_units)
    builder.add_component(
        component_class(component_name, depends_on=transformation))
    loaded_data = load_function(builder)

    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    # Resulting position will always be in metres, whatever units are
    # used in the NeXus file
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
@pytest.mark.parametrize(
    "transform_type,value,value_units,expected_position",
    ((TransformationType.ROTATION, 0.27, "rad", [0, 0, 0]),
     (TransformationType.TRANSLATION, 230, "cm", [0, 0, 2.3])))
def test_loads_component_position_from_log_transformation(
        component_class: Union[Type[Source],
                               Type[Sample]], component_name: str,
        transform_type: TransformationType, value: float, value_units: str,
        expected_position: List[float], load_function: Callable):
    builder = NexusBuilder()
    # Provide "time" data, the builder will write the transformation as
    # an NXlog
    transformation = Transformation(transform_type,
                                    vector=np.array([0, 0, -1]),
                                    value=np.array([value]),
                                    time=np.array([1.3]),
                                    time_units="s",
                                    value_units=value_units)
    builder.add_component(
        component_class(component_name, depends_on=transformation))
    loaded_data = load_function(builder)

    # Should load as usual despite the transformation being an NXlog
    # as it only has a single value
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
@pytest.mark.parametrize("transform_type,value,value_units",
                         ((TransformationType.ROTATION, [26, 73], "deg"),
                          (TransformationType.TRANSLATION, [230, 310], "cm")))
def test_skips_component_position_with_multi_value_log_transformation(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str, transform_type: TransformationType,
        value: List[float], value_units: str, load_function: Callable):
    builder = NexusBuilder()
    # Provide "time" data, the builder will write the transformation as
    # an NXlog. This would be encountered in a file from an experiment
    # involving a scan of a motion axis.
    transformation = Transformation(transform_type,
                                    vector=np.array([0, 0, -1]),
                                    value=np.array(value),
                                    time=np.array([1.3, 6.4]),
                                    time_units="s",
                                    value_units=value_units)
    builder.add_component(
        component_class(component_name, depends_on=transformation))
    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    # Loading component position from transformations recorded as
    # NXlogs with multiple values is not yet implemented
    # However the NXlog itself will be loaded
    # (loaded_data is not None)
    assert f"{component_name}_position" not in loaded_data.keys()


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
@pytest.mark.parametrize("transform_type,value_units",
                         ((TransformationType.ROTATION, "deg"),
                          (TransformationType.TRANSLATION, "cm")))
def test_skips_component_position_with_empty_value_log_transformation(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str, transform_type: TransformationType,
        value_units: str, load_function: Callable):
    builder = NexusBuilder()
    empty_value = np.array([])
    transformation = Transformation(transform_type,
                                    vector=np.array([0, 0, -1]),
                                    value=empty_value,
                                    time=np.array([1.3, 6.4]),
                                    time_units="s",
                                    value_units=value_units)
    builder.add_component(
        component_class(component_name, depends_on=transformation))
    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)

    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
def test_load_component_position_prefers_transform_over_distance(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str, load_function: Callable):
    # The "distance" dataset gives the position along the z axis.
    # If there is a "depends_on" pointing to transformations then we
    # prefer to use that instead as it is likely to be more accurate; it
    # can define position and orientation in 3D.
    builder = NexusBuilder()
    transformation = Transformation(TransformationType.TRANSLATION,
                                    np.array([0, 0, -1]),
                                    np.array([2.3]),
                                    value_units="m")
    builder.add_component(
        component_class(component_name,
                        depends_on=transformation,
                        distance=4.2,
                        distance_units="m"))
    loaded_data = load_function(builder)

    expected_position = np.array([0, 0, transformation.value[0]])
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
@pytest.mark.parametrize(
    "transform_type",
    (TransformationType.ROTATION, TransformationType.TRANSLATION))
def test_skips_component_position_from_transformation_missing_unit(
        component_class: Union[Type[Source],
                               Type[Sample]], component_name: str,
        transform_type: TransformationType, load_function: Callable):
    builder = NexusBuilder()
    transformation = Transformation(transform_type, np.array([0, 0, -1]),
                                    np.array([2.3]))
    builder.add_component(
        component_class(component_name, depends_on=transformation))
    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)
    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
@pytest.mark.parametrize("transform_type,value_units",
                         ((TransformationType.ROTATION, "deg"),
                          (TransformationType.TRANSLATION, "m")))
def test_skips_component_position_with_transformation_with_small_vector(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str, transform_type: TransformationType,
        value_units: str, load_function: Callable):
    # The vector defines the direction of the translation or axis
    # of the rotation so it is ill-defined if it is close to zero
    # in magnitude
    builder = NexusBuilder()
    zero_vector = np.array([0, 0, 0])
    transformation = Transformation(transform_type,
                                    zero_vector,
                                    np.array([2.3]),
                                    value_units=value_units)
    builder.add_component(
        component_class(component_name, depends_on=transformation))
    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)
    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
def test_loads_component_position_from_multiple_transformations(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str, load_function: Callable):
    builder = NexusBuilder()
    transformation_1 = Transformation(TransformationType.ROTATION,
                                      np.array([0, 1, 0]),
                                      np.array([90]),
                                      value_units="deg")
    transformation_2 = Transformation(TransformationType.TRANSLATION,
                                      np.array([0, 0, -1]),
                                      np.array([2.3]),
                                      value_units="m",
                                      depends_on=transformation_1)
    builder.add_component(
        component_class(component_name, depends_on=transformation_2))
    loaded_data = load_function(builder)

    # Transformations in NeXus are "passive transformations", so in this
    # test case the coordinate system is rotated 90 degrees anticlockwise
    # around the y axis and then shifted 2.3m in the z direction. In
    # the lab reference frame this corresponds to
    # setting the sample position to -2.3m in the x direction.
    expected_position = np.array([-transformation_2.value[0], 0, 0])
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


def test_skips_source_position_if_not_given_in_file(load_function: Callable):
    builder = NexusBuilder()
    builder.add_source(Source("source"))
    with pytest.warns(UserWarning):
        loaded_data = load_function(builder)
    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         ((Sample, "sample"), (Source, "source")))
def test_loads_component_position_from_distance_dataset(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str, load_function: Callable):
    # If the NXsource or NXsample contains a "distance" dataset
    # this gives the position along the z axis. If there was a "depends_on"
    # pointing to transformations then we'd use that instead as it is
    # likely to be more accurate; it can define position and orientation in 3D.
    builder = NexusBuilder()
    distance = 4.2
    units = "m"
    builder.add_component(
        component_class(component_name,
                        distance=distance,
                        distance_units=units))
    loaded_data = load_function(builder)

    expected_position = np.array([0, 0, distance])
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit(units)


def test_loads_source_position_dependent_on_sample_position(
        load_function: Callable):
    builder = NexusBuilder()
    transformation_0 = Transformation(TransformationType.ROTATION,
                                      np.array([0, 1, 0]),
                                      np.array([90]),
                                      value_units="deg")
    transformation_1 = Transformation(TransformationType.TRANSLATION,
                                      np.array([0, 0, -1]),
                                      np.array([2.3]),
                                      value_units="m",
                                      depends_on=transformation_0)
    builder.add_sample(Sample("sample", depends_on=transformation_1))
    transformation_2 = Transformation(
        TransformationType.TRANSLATION,
        np.array([0, 0, -1]),
        np.array([1.0]),
        value_units="m",
        depends_on="/entry/sample/transformations/transform_1")
    builder.add_source(Source("source", depends_on=transformation_2))
    loaded_data = load_function(builder)

    # Transformations in NeXus are "passive transformations", so in this
    # test case the coordinate system is rotated 90 degrees anticlockwise
    # around the y axis and then shifted 2.3m in the z direction, then
    # another 1.0m in the z direction. In
    # the lab reference frame this corresponds to
    # setting the sample position to -3.3m in the x direction.
    expected_position = np.array([-3.3, 0, 0])
    assert np.allclose(loaded_data["source_position"].values,
                       expected_position)
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
                                    vector=np.array([0, 0, -1]),
                                    value=np.array([distance]),
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

    expected_pixel_positions = np.array([
        x_pixel_offset_1, y_pixel_offset_1, z_pixel_offset_1 + distance / 100.
    ]).T
    assert np.allclose(loaded_data.coords['position'].values,
                       expected_pixel_positions)


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
    builder.add_hard_link(Link("/entry/hard_link_to_events",
                               "/entry/events_0"))
    builder.add_soft_link(Link("/entry/soft_link_to_events",
                               "/entry/events_0"))

    loaded_data = load_function(builder)

    # The output Variable must contain the events from the added event
    # dataset with no duplicate data due to the links

    # Expect time of flight to match the values in the
    # event_time_offset dataset
    # May be reordered due to binning (hence np.sort)
    assert np.array_equal(
        np.sort(
            loaded_data.bins.concatenate(
                'detector_id').values.coords['tof'].values),
        np.sort(event_time_offsets))

    counts_on_detectors = loaded_data.bins.sum()
    # No detector_number dataset in file so expect detector_id to be
    # binned according to whatever detector_ids are present in event_id
    # dataset: 2 on det 1, 1 on det 2, 2 on det 3
    expected_counts = np.array([2, 1, 2])
    assert np.array_equal(counts_on_detectors.data.values, expected_counts)
    expected_detector_ids = np.array([1, 2, 3])
    assert np.array_equal(loaded_data.coords['detector_id'].values,
                          expected_detector_ids)
