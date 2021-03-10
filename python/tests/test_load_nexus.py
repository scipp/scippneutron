from .nexus_helpers import (
    InMemoryNexusFileBuilder,
    EventData,
    Detector,
    Log,
    Sample,
    Source,
    Transformation,
    TransformationType,
    in_memory_hdf5_file_with_two_nxentry,
)
import numpy as np
import pytest
import scippneutron
import scipp as sc
from typing import List, Type, Union


def test_raises_exception_if_multiple_nxentry_in_file():
    with in_memory_hdf5_file_with_two_nxentry() as nexus_file:
        with pytest.raises(RuntimeError):
            scippneutron.load_nexus(nexus_file)


def test_no_exception_if_single_nxentry_in_file():
    builder = InMemoryNexusFileBuilder()
    with builder.file() as nexus_file:
        assert scippneutron.load_nexus(nexus_file) is None


def test_no_exception_if_single_nxentry_found_below_root():
    with in_memory_hdf5_file_with_two_nxentry() as nexus_file:
        # There are 2 NXentry in the file, but root is used
        # to specify which to load data from
        assert scippneutron.load_nexus(nexus_file, root='/entry_1') is None


def test_loads_data_from_single_event_data_group():
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

    builder = InMemoryNexusFileBuilder()
    builder.add_event_data(event_data)

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

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


def test_loads_data_from_multiple_event_data_groups():
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

    builder = InMemoryNexusFileBuilder()
    builder.add_detector(Detector(detector_1_ids, event_data_1))
    builder.add_detector(Detector(detector_2_ids, event_data_2))

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

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


def test_skips_event_data_group_with_non_integer_event_ids():
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

    builder = InMemoryNexusFileBuilder()
    builder.add_event_data(event_data)

    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)

    assert loaded_data is None, "Expected no data to be loaded as " \
                                "event data has non integer event ids"


def test_skips_event_data_group_with_non_integer_detector_numbers():
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

    builder = InMemoryNexusFileBuilder()
    builder.add_detector(Detector(detector_numbers, event_data))

    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)

    assert loaded_data is None, "Expected no data to be loaded as " \
                                "detector has non integer detector numbers"


def test_skips_data_with_event_id_and_detector_number_type_unequal():
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

    builder = InMemoryNexusFileBuilder()
    builder.add_detector(Detector(detector_numbers, event_data))

    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)

    assert loaded_data is None, "Expected no data to be loaded as event " \
                                "ids and detector numbers are of " \
                                "different types"


def test_loads_data_from_single_log_with_no_units():
    values = np.array([1, 2, 3])
    times = np.array([4, 5, 6])
    name = "test_log"
    builder = InMemoryNexusFileBuilder()
    builder.add_log(Log(name, values, times))

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    # Expect a sc.Dataset with log names as keys
    assert np.array_equal(loaded_data[name].data.values.values, values)
    assert np.array_equal(loaded_data[name].data.values.coords['time'].values,
                          times)


def test_loads_data_from_single_log_with_units():
    values = np.array([1.1, 2.2, 3.3])
    times = np.array([4.4, 5.5, 6.6])
    name = "test_log"
    builder = InMemoryNexusFileBuilder()
    builder.add_log(Log(name, values, times, value_units="m", time_units="s"))

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    # Expect a sc.Dataset with log names as keys
    assert np.allclose(loaded_data[name].data.values.values, values)
    assert np.allclose(loaded_data[name].data.values.coords['time'].values,
                       times)
    assert loaded_data[name].data.values.unit == sc.units.m
    assert loaded_data[name].data.values.coords['time'].unit == sc.units.s


def test_loads_data_from_multiple_logs():
    builder = InMemoryNexusFileBuilder()
    log_1 = Log("test_log", np.array([1.1, 2.2, 3.3]),
                np.array([4.4, 5.5, 6.6]))
    log_2 = Log("test_log_2", np.array([123, 253, 756]),
                np.array([246, 1235, 2369]))
    builder.add_log(log_1)
    builder.add_log(log_2)

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    # Expect a sc.Dataset with log names as keys
    assert np.allclose(loaded_data[log_1.name].data.values.values, log_1.value)
    assert np.allclose(
        loaded_data[log_1.name].data.values.coords['time'].values, log_1.time)
    assert np.array_equal(loaded_data[log_2.name].data.values.values,
                          log_2.value)
    assert np.array_equal(
        loaded_data[log_2.name].data.values.coords['time'].values, log_2.time)


def test_loads_logs_with_non_supported_int_types():
    builder = InMemoryNexusFileBuilder()
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

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    # Expect a sc.Dataset with log names as keys
    for log in logs:
        assert np.allclose(loaded_data[log.name].data.values.values, log.value)
        assert np.allclose(loaded_data[log.name].data.values.coords['time'],
                           log.time)


def test_skips_multidimensional_log():
    # Loading NXlogs with more than 1 dimension is not yet implemented
    # We need to come up with a sensible approach to labelling the dimensions

    multidim_values = np.array([[1, 2, 3], [1, 2, 3]])
    name = "test_log"
    builder = InMemoryNexusFileBuilder()
    builder.add_log(Log(name, multidim_values, np.array([4, 5, 6])))

    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)

    assert loaded_data is None


def test_skips_log_with_no_value_dataset():
    name = "test_log"
    builder = InMemoryNexusFileBuilder()
    builder.add_log(Log(name, None, np.array([4, 5, 6])))

    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)

    assert loaded_data is None


def test_skips_log_with_empty_value_and_time_datasets():
    empty_values = np.array([]).astype(np.int32)
    empty_times = np.array([]).astype(np.int32)
    name = "test_log"
    builder = InMemoryNexusFileBuilder()
    builder.add_log(Log(name, empty_values, empty_times))

    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)

    assert loaded_data is None


def test_skips_log_with_mismatched_value_and_time():
    values = np.array([1, 2, 3]).astype(np.int32)
    times = np.array([1, 2, 3, 4]).astype(np.int32)
    name = "test_log"
    builder = InMemoryNexusFileBuilder()
    builder.add_log(Log(name, values, times))

    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)

    assert loaded_data is None


def test_loads_data_from_non_timeseries_log():
    values = np.array([1.1, 2.2, 3.3])
    name = "test_log"
    builder = InMemoryNexusFileBuilder()
    builder.add_log(Log(name, values))

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    assert np.allclose(loaded_data[name].data.values.values, values)


def test_loads_data_from_multiple_logs_with_same_name():
    values_1 = np.array([1.1, 2.2, 3.3])
    values_2 = np.array([4, 5, 6])
    name = "test_log"

    # Add one log to NXentry and the other to an NXdetector,
    # both have the same group name
    builder = InMemoryNexusFileBuilder()
    builder.add_log(Log(name, values_1))
    builder.add_detector(Detector(np.array([1, 2, 3]), log=Log(name,
                                                               values_2)))

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

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


def test_load_instrument_name():
    name = "INSTR"
    builder = InMemoryNexusFileBuilder()
    builder.add_instrument(name)

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    assert loaded_data['instrument_name'].values == name


def test_load_experiment_title():
    title = "my experiment"
    builder = InMemoryNexusFileBuilder()
    builder.add_title(title)

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    assert loaded_data['experiment_title'].values == title


def test_loads_event_and_log_data_from_single_file():
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

    builder = InMemoryNexusFileBuilder()
    builder.add_event_data(event_data)
    builder.add_log(log_1)
    builder.add_log(log_2)

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

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


def test_loads_pixel_positions_with_event_data():
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

    builder = InMemoryNexusFileBuilder()
    builder.add_detector(
        Detector(detector_1_ids,
                 event_data_1,
                 x_offsets=x_pixel_offset_1,
                 y_offsets=y_pixel_offset_1,
                 z_offsets=z_pixel_offset_1))
    builder.add_detector(
        Detector(detector_2_ids,
                 event_data_2,
                 x_offsets=x_pixel_offset_2,
                 y_offsets=y_pixel_offset_2))

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    # If z offsets are missing they should be zero
    z_pixel_offset_2 = np.array([[0., 0.], [0., 0.]])
    expected_pixel_positions = np.array([
        np.concatenate((x_pixel_offset_1, x_pixel_offset_2.flatten())),
        np.concatenate((y_pixel_offset_1, y_pixel_offset_2.flatten())),
        np.concatenate((z_pixel_offset_1, z_pixel_offset_2.flatten()))
    ]).T
    assert np.allclose(loaded_data.coords['position'].values,
                       expected_pixel_positions)


def test_skips_loading_pixel_positions_with_non_matching_shape():
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

    builder = InMemoryNexusFileBuilder()
    builder.add_detector(
        Detector(detector_1_ids,
                 event_data_1,
                 x_offsets=x_pixel_offset_1,
                 y_offsets=y_pixel_offset_1,
                 z_offsets=z_pixel_offset_1))
    builder.add_detector(
        Detector(detector_2_ids,
                 event_data_2,
                 x_offsets=x_pixel_offset_2,
                 y_offsets=y_pixel_offset_2))

    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)

    assert "position" not in loaded_data.coords.keys(
    ), "One of the NXdetectors pixel positions arrays did not match the " \
       "size of its detector ids so we should not find 'position' coord"
    # Even though detector_1's offsets and ids are matches in size, we do not
    # load them as the "position" coord would not have positions for all
    # the detector ids (loading event data from all detectors is prioritised).


def test_sample_position_at_origin_if_not_explicit_in_file():
    # The sample position is the origin of the coordinate
    # system in NeXus files.
    # If there is an NXsample in the file, but it has no "distance" dataset
    # or "depends_on" pointing to NXtransformations then it should be
    # assumed to be at the origin.
    builder = InMemoryNexusFileBuilder()
    builder.add_sample(Sample("sample"))
    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    origin = np.array([0, 0, 0])
    assert np.allclose(loaded_data["sample_position"].values, origin)


@pytest.mark.parametrize("component_class,component_name",
                         [(Sample, "sample"), (Source, "source")])
def test_skips_loading_component_if_more_than_one_in_file(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str):
    # More than one source or sample is a serious error in the file, so
    # load_nexus will display a warning and skip loading any sample rather
    # than guessing which is the "correct" one.
    builder = InMemoryNexusFileBuilder()
    builder.add_component(component_class(f"{component_name}_1"))
    builder.add_component(component_class(f"{component_name}_2"))
    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)
    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         [(Sample, "sample"), (Source, "source")])
def test_skips_component_position_from_distance_dataset_missing_unit(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str):
    builder = InMemoryNexusFileBuilder()
    distance = 4.2
    builder.add_component(
        component_class(component_name, distance=distance,
                        distance_units=None))
    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)
    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         [(Sample, "sample"), (Source, "source")])
@pytest.mark.parametrize(
    "transform_type,value,value_units,expected_position",
    [(TransformationType.ROTATION, 0.27, "rad", [0, 0, 0]),
     (TransformationType.TRANSLATION, 230, "cm", [0, 0, 2.3])])
def test_loads_component_position_from_single_transformation(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str, transform_type: TransformationType, value: float,
        value_units: str, expected_position: List[float]):
    builder = InMemoryNexusFileBuilder()
    transformation = Transformation(transform_type,
                                    vector=np.array([0, 0, -1]),
                                    value=np.array([value]),
                                    value_units=value_units)
    builder.add_component(
        component_class(component_name, depends_on=transformation))
    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    # Resulting position will always be in metres, whatever units are
    # used in the NeXus file
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


@pytest.mark.parametrize("component_class,component_name",
                         [(Sample, "sample"), (Source, "source")])
@pytest.mark.parametrize(
    "transform_type,value,value_units,expected_position",
    [(TransformationType.ROTATION, 0.27, "rad", [0, 0, 0]),
     (TransformationType.TRANSLATION, 230, "cm", [0, 0, 2.3])])
def test_loads_component_position_from_log_transformation(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str, transform_type: TransformationType, value: float,
        value_units: str, expected_position: List[float]):
    builder = InMemoryNexusFileBuilder()
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
    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    # Should load as usual despite the transformation being an NXlog
    # as it only has a single value
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


@pytest.mark.parametrize("component_class,component_name",
                         [(Sample, "sample"), (Source, "source")])
@pytest.mark.parametrize("transform_type,value,value_units",
                         [(TransformationType.ROTATION, [26, 73], "deg"),
                          (TransformationType.TRANSLATION, [230, 310], "cm")])
def test_skips_component_position_with_multi_value_log_transformation(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str, transform_type: TransformationType,
        value: List[float], value_units: str):
    builder = InMemoryNexusFileBuilder()
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
    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)

    # Loading component position from transformations recorded as
    # NXlogs with multiple values is not yet implemented
    # However the NXlog itself will be loaded
    # (loaded_data is not None)
    assert f"{component_name}_position" not in loaded_data.keys()


@pytest.mark.parametrize("component_class,component_name",
                         [(Sample, "sample"), (Source, "source")])
def test_load_component_position_prefers_transform_over_distance(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str):
    # The "distance" dataset gives the position along the z axis.
    # If there is a "depends_on" pointing to transformations then we
    # prefer to use that instead as it is likely to be more accurate; it
    # can define position and orientation in 3D.
    builder = InMemoryNexusFileBuilder()
    transformation = Transformation(TransformationType.TRANSLATION,
                                    np.array([0, 0, -1]),
                                    np.array([2.3]),
                                    value_units="m")
    builder.add_component(
        component_class(component_name,
                        depends_on=transformation,
                        distance=4.2,
                        distance_units="m"))
    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    expected_position = np.array([0, 0, transformation.value[0]])
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


@pytest.mark.parametrize("component_class,component_name",
                         [(Sample, "sample"), (Source, "source")])
@pytest.mark.parametrize(
    "transform_type",
    [TransformationType.ROTATION, TransformationType.TRANSLATION])
def test_skips_component_position_from_transformation_missing_unit(
        component_class: Union[Type[Source], Type[Sample]],
        component_name: str, transform_type: TransformationType):
    builder = InMemoryNexusFileBuilder()
    transformation = Transformation(transform_type, np.array([0, 0, -1]),
                                    np.array([2.3]))
    builder.add_component(
        component_class(component_name, depends_on=transformation))
    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)
    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         [(Sample, "sample"), (Source, "source")])
@pytest.mark.parametrize("transform_type,value_units",
                         [(TransformationType.ROTATION, "deg"),
                          (TransformationType.TRANSLATION, "m")])
def test_skips_component_position_with_transformation_with_small_vector(
        component_class: Union[Type[Source],
                               Type[Sample]], component_name: str,
        transform_type: TransformationType, value_units: str):
    # The vector defines the direction of the translation or axis
    # of the rotation so it is ill-defined if it is close to zero
    # in magnitude
    builder = InMemoryNexusFileBuilder()
    zero_vector = np.array([0, 0, 0])
    transformation = Transformation(transform_type,
                                    zero_vector,
                                    np.array([2.3]),
                                    value_units=value_units)
    builder.add_component(
        component_class(component_name, depends_on=transformation))
    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)
    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         [(Sample, "sample"), (Source, "source")])
def test_loads_component_position_from_multiple_transformations(
    component_class: Union[Type[Source], Type[Sample]],
    component_name: str,
):
    builder = InMemoryNexusFileBuilder()
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
    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    # Transformations in NeXus are "passive transformations", so in this
    # test case the coordinate system is rotated 90 degrees anticlockwise
    # around the y axis and then shifted 2.3m in the z direction. In
    # the lab reference frame this corresponds to
    # setting the sample position to -2.3m in the x direction.
    expected_position = np.array([-transformation_2.value[0], 0, 0])
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit("m")


def test_skips_source_position_if_not_given_in_file():
    builder = InMemoryNexusFileBuilder()
    builder.add_source(Source("source"))
    with builder.file() as nexus_file:
        with pytest.warns(UserWarning):
            loaded_data = scippneutron.load_nexus(nexus_file)
    assert loaded_data is None


@pytest.mark.parametrize("component_class,component_name",
                         [(Sample, "sample"), (Source, "source")])
def test_loads_component_position_from_distance_dataset(
    component_class: Union[Type[Source], Type[Sample]],
    component_name: str,
):
    # If the NXsource or NXsample contains a "distance" dataset
    # this gives the position along the z axis. If there was a "depends_on"
    # pointing to transformations then we'd use that instead as it is
    # likely to be more accurate; it can define position and orientation in 3D.
    builder = InMemoryNexusFileBuilder()
    distance = 4.2
    units = "m"
    builder.add_component(
        component_class(component_name,
                        distance=distance,
                        distance_units=units))
    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    expected_position = np.array([0, 0, distance])
    assert np.allclose(loaded_data[f"{component_name}_position"].values,
                       expected_position)
    assert loaded_data[f"{component_name}_position"].unit == sc.Unit(units)


def test_loads_source_position_dependent_on_sample_position():
    builder = InMemoryNexusFileBuilder()
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
    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

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


def test_loads_pixel_positions_with_transformations():
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

    builder = InMemoryNexusFileBuilder()
    builder.add_detector(
        Detector(detector_1_ids,
                 event_data_1,
                 x_offsets=x_pixel_offset_1,
                 y_offsets=y_pixel_offset_1,
                 z_offsets=z_pixel_offset_1,
                 depends_on=transformation))

    with builder.file() as nexus_file:
        loaded_data = scippneutron.load_nexus(nexus_file)

    expected_pixel_positions = np.array([
        x_pixel_offset_1, y_pixel_offset_1, z_pixel_offset_1 + distance / 100.
    ]).T
    assert np.allclose(loaded_data.coords['position'].values,
                       expected_pixel_positions)
