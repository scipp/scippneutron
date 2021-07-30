# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from dataclasses import dataclass
import h5py
from typing import Optional, List, Any, Dict, Union
import numpy as np
from ._common import (BadSource, SkipSource, MissingDataset, Group)
import scipp as sc
from warnings import warn
from itertools import groupby
from ._transformations import get_full_transformation_matrix
from ._nexus import LoadFromNexus, GroupObject

_detector_dim_name = "detector_id"
_event_dim_name = "event"
_event_index_dim_name = "event_index"
_time_of_flight_dim_name = "tof"


class DetectorIdError(Exception):
    pass


def _all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def _check_for_missing_fields(group: Group, nexus: LoadFromNexus):
    if group.contains_stream:
        # Do not warn about missing datasets if the group contains
        # a stream, as this will provide the missing data
        raise SkipSource("Data source is missing datasets"
                         "but contains a stream source for the data")

    required_fields = (
        "event_time_zero",
        "event_index",
        "event_id",
        "event_time_offset",
    )
    for field in required_fields:
        found, msg = nexus.dataset_in_group(group.group, field)
        if not found:
            raise BadSource(msg)


def _convert_array_to_metres(array: np.ndarray, unit: str) -> np.ndarray:
    return sc.to_unit(
        sc.Variable(["temporary_variable"],
                    values=array,
                    unit=unit,
                    dtype=np.float64), "m").values


def _load_pixel_positions(detector_group: GroupObject, detector_ids_size: int,
                          file_root: h5py.File,
                          nexus: LoadFromNexus) -> Optional[sc.Variable]:
    offsets_unit = nexus.get_unit(
        nexus.get_dataset_from_group(detector_group, "x_pixel_offset"))
    if offsets_unit == sc.units.dimensionless:
        warn(f"Skipped loading pixel positions as no units found on "
             f"x_pixel_offset dataset in {nexus.get_name(detector_group)}")
        return None

    try:
        x_positions = nexus.load_dataset_from_group_as_numpy_array(
            detector_group, "x_pixel_offset").flatten()
        y_positions = nexus.load_dataset_from_group_as_numpy_array(
            detector_group, "y_pixel_offset").flatten()
    except MissingDataset:
        return None
    try:
        z_positions = nexus.load_dataset_from_group_as_numpy_array(
            detector_group, "z_pixel_offset").flatten()
    except MissingDataset:
        # According to the NeXus standard z offsets are allowed to be
        # missing, in which case use zeros
        z_positions = np.zeros_like(x_positions)

    if not _all_equal((x_positions.size, y_positions.size, z_positions.size,
                       detector_ids_size)):
        warn(f"Skipped loading pixel positions as pixel offset and id "
             f"dataset sizes do not match in {nexus.get_name(detector_group)}")
        return None

    x_positions = _convert_array_to_metres(x_positions, offsets_unit)
    y_positions = _convert_array_to_metres(y_positions, offsets_unit)
    z_positions = _convert_array_to_metres(z_positions, offsets_unit)

    array = np.array([x_positions, y_positions, z_positions]).T

    found_depends_on, _ = nexus.dataset_in_group(detector_group, "depends_on")
    if found_depends_on:
        # Add fourth element of 1 to each vertex, indicating these are
        # positions not direction vectors
        n_rows = array.shape[0]
        array = np.hstack((array, np.ones((n_rows, 1))))

        # Get and apply transformation matrix
        transformation = get_full_transformation_matrix(
            detector_group, file_root, nexus)
        for row_index in range(n_rows):
            array[row_index, :] = np.matmul(transformation,
                                            array[row_index, :])

        # Now the transformations are done we do not need the 4th
        # element in each position
        array = array[:, :3]

    return sc.vectors(dims=[_detector_dim_name],
                      values=array,
                      unit=sc.units.m)


@dataclass
class DetectorData:
    events: Optional[sc.DataArray] = None
    detector_ids: Optional[sc.Variable] = None
    pixel_positions: Optional[sc.Variable] = None
    pulse_times: Optional[sc.Variable] = None
    event_index: Optional[sc.Variable] = None


def _create_empty_events_data_array(
        tof_dtype: Any = np.int64,
        tof_unit: Union[str, sc.Unit] = "ns",
        detector_id_dtype: Any = np.int32) -> sc.DataArray:
    return sc.DataArray(data=sc.empty(dims=[_event_dim_name],
                                      shape=[0],
                                      variances=True,
                                      dtype=np.float32),
                        coords={
                            _time_of_flight_dim_name:
                            sc.empty(dims=[_event_dim_name],
                                     shape=[0],
                                     dtype=tof_dtype,
                                     unit=tof_unit),
                            _detector_dim_name:
                            sc.empty(dims=[_event_dim_name],
                                     shape=[0],
                                     dtype=detector_id_dtype)
                        })


def _load_pulse_times(group: Group, nexus: LoadFromNexus) -> sc.Variable:
    time_zero_group = "event_time_zero"
    _event_time_zeros_units = sc.Unit(
        nexus.get_string_attribute(group.group[time_zero_group], "units"))

    return sc.Variable(
        dims=[_event_index_dim_name],
        values=nexus.load_dataset_from_group_as_numpy_array(group.group,
                                                            time_zero_group),
        unit=_event_time_zeros_units,
        dtype=sc.dtype.float64)


def _load_event_index(group: Group, nexus: LoadFromNexus) -> sc.Variable:
    event_index = nexus.load_dataset_from_group_as_numpy_array(
        group.group, "event_index")
    return sc.Variable(
        dims=[_event_index_dim_name],
        values=event_index,
        unit=sc.units.dimensionless
    )


def _load_detector(group: Group, file_root: h5py.File,
                   nexus: LoadFromNexus) -> DetectorData:
    detector_number_ds_name = "detector_number"
    dataset_in_group, _ = nexus.dataset_in_group(group.group,
                                                 detector_number_ds_name)
    detector_ids = None
    if dataset_in_group:
        detector_ids = nexus.load_dataset_from_group_as_numpy_array(
            group.group, detector_number_ds_name).flatten()
        detector_id_type = detector_ids.dtype.type

        detector_ids = sc.Variable(dims=[_detector_dim_name],
                                   values=detector_ids,
                                   dtype=detector_id_type)

    pixel_positions = None
    pixel_positions_found, _ = nexus.dataset_in_group(group.group,
                                                      "x_pixel_offset")
    if pixel_positions_found and detector_ids is not None:
        pixel_positions = _load_pixel_positions(group.group,
                                                detector_ids.shape[0],
                                                file_root, nexus)

    return DetectorData(detector_ids=detector_ids,
                        pixel_positions=pixel_positions)


def _load_event_group(group: Group, file_root: h5py.File, nexus: LoadFromNexus,
                      detector_data: DetectorData,
                      quiet: bool) -> DetectorData:
    _check_for_missing_fields(group, nexus)

    # There is some variation in the last recorded event_index in files
    # from different institutions. We try to make sure here that it is what
    # would be the first index of the next pulse.
    # In other words, ensure that event_index includes the bin edge for
    # the last pulse.
    event_id = nexus.load_dataset(group.group, "event_id", [_event_dim_name])
    number_of_event_ids = event_id.sizes['event']
    event_index = nexus.load_dataset_from_group_as_numpy_array(
        group.group, "event_index")
    if event_index[-1] < number_of_event_ids:
        event_index = np.append(
            event_index,
            np.array([number_of_event_ids - 1]).astype(event_index.dtype),
        )
    else:
        event_index[-1] = number_of_event_ids

    number_of_events = event_index[-1]
    event_time_offset = nexus.load_dataset(group.group, "event_time_offset",
                                           [_event_dim_name])

    # Weights are not stored in NeXus, so use 1s
    weights = sc.ones(dims=[_event_dim_name],
                      shape=event_id.shape,
                      dtype=np.float32,
                      variances=True)

    if detector_data.detector_ids is None:
        # If detector ids were not found in an associated detector group
        # we will just have to bin according to whatever
        # ids we have a events for (pixels with no recorded events
        # will not have a bin)
        detector_data.detector_ids = sc.Variable(dims=[_detector_dim_name],
                                                 values=np.unique(
                                                     event_id.values))

    _check_event_ids_and_det_number_types_valid(
        detector_data.detector_ids.dtype, event_id.dtype)

    event_index = _load_event_index(group=group, nexus=nexus)
    pulse_times = _load_pulse_times(group=group, nexus=nexus)

    data_dict = {
        "data": weights,
        "coords": {
            _time_of_flight_dim_name: event_time_offset,
            _detector_dim_name: event_id,
        },
    }
    detector_data.events = sc.DataArray(**data_dict)

    detector_group = group.parent
    pixel_positions_found, _ = nexus.dataset_in_group(detector_group,
                                                      "x_pixel_offset")
    # Checking for positions here is needed because, in principle, the standard
    # allows not to always have them. ESS files should however always have
    # them.
    if pixel_positions_found:
        detector_data.pixel_positions = _load_pixel_positions(
            detector_group, detector_data.detector_ids.shape[0], file_root,
            nexus)

    detector_data.pulse_times = pulse_times
    detector_data.event_index = event_index

    if not quiet:
        print(f"Loaded event data from "
              f"{group.path} containing {number_of_events} events")

    return detector_data


def _check_event_ids_and_det_number_types_valid(detector_id_type: Any,
                                                event_id_type: Any):
    """
    These must be integers and must be the same type or we'll have
    problems trying to bin events by detector id. Check here so that
    we can give a useful warning to the user and skip loading the
    current event group.
    """
    def is_integer_type(type_to_check: Any) -> bool:
        return type_to_check == sc.dtype.int32 or \
               type_to_check == sc.dtype.int64

    if not is_integer_type(detector_id_type):
        raise DetectorIdError(
            "detector_numbers dataset in NXdetector is not an integer "
            "type")
    if not is_integer_type(event_id_type):
        raise BadSource("event_ids dataset is not an integer type")
    if detector_id_type != event_id_type:
        raise DetectorIdError(
            "event_ids and detector_numbers datasets in corresponding "
            "NXdetector were not of the same type")


def load_detector_data(event_data_groups: List[Group],
                       detector_groups: List[Group], file_root: h5py.File,
                       nexus: LoadFromNexus,
                       quiet: bool) -> Optional[sc.DataArray]:
    detector_data = _load_data_from_each_nx_detector(detector_groups,
                                                     file_root, nexus)

    event_data = _load_data_from_each_nx_event_data(detector_data,
                                                    event_data_groups,
                                                    file_root, nexus, quiet)

    if not event_data:
        # If there were no data to load we are done
        return

    def get_detector_id(data: DetectorData):
        # Assume different detector banks do not have
        # intersecting ranges of detector ids
        return data.detector_ids.values[0]

    event_data.sort(key=get_detector_id)

    _create_empty_event_data(event_data)

    pixel_positions_loaded = all(
        [data.pixel_positions is not None for data in event_data])
    detector_data = event_data.pop(0)

    # Events in the NeXus file are effectively binned by pulse
    # (because they are recorded chronologically)
    # but for reduction it is more useful to bin by detector id
    events = sc.bin(detector_data.events, groups=[detector_data.detector_ids])

    if pixel_positions_loaded:
        # TODO: the name 'position' should probably not be hard-coded but moved
        # to a variable that cah be changed in a single place.
        events.coords['position'] = detector_data.pixel_positions
    while event_data:
        detector_data = event_data.pop(0)

        # print(f"events: {detector_data.events}")
        # print(f"det ids: {detector_data.detector_ids}")
        # print(f"index: {detector_data.event_index}")

        unbinned_pulse_times = sc.DataArray(data=detector_data.pulse_times, coords={
            _event_index_dim_name: detector_data.event_index
        })

        binned_pulse_times = sc.bin(unbinned_pulse_times, edges=[detector_data.event_index])

        # print(f"1. binned_pulse_times: {binned_pulse_times}")

        pt = sc.Variable(value=sc.flatten(binned_pulse_times, to=_event_dim_name))

        print(f"1a. pt: {pt}")
        #
        # binned_pulse_times.events.coords[_event_index_dim_name] = sc.empty(
        #     shape=binned_pulse_times.events.shape,
        #     dtype=binned_pulse_times.events.dtype,
        #     unit=binned_pulse_times.events.unit,
        #     dims=binned_pulse_times.events.dims
        # )

        # print(f"2. binned_pulse_times: {binned_pulse_times}")
        # print(f"2a. {binned_pulse_times.bins.coords[_event_index_dim_name]}")
        #
        # tmp = binned_pulse_times.coords[_event_index_dim_name]
        # print(f"2b. tmp = {tmp}")
        # print(f"2c. coords = {binned_pulse_times.bins.coords[_event_index_dim_name]}")
        # print(f"2d. coords = {binned_pulse_times.bins.coords[_event_index_dim_name][_event_index_dim_name, :]}")
        #
        # binned_pulse_times.bins.coords[_event_index_dim_name][_event_index_dim_name, :] = tmp
        #
        # print(f"3. binned_pulse_times: {binned_pulse_times}")

        new_events = sc.bin(detector_data.events,
                            groups=[detector_data.detector_ids])

        if pixel_positions_loaded:
            new_events.coords['position'] = detector_data.pixel_positions

        events = sc.concatenate(events, new_events, dim=_detector_dim_name)
        events.coords['pulse_times'] = pt

        # print("events: {}".format(events))
        # print("event index: {}".format(detector_data.event_index))
        #
        # events_by_pulse_time = sc.bin(
        #     events,
        #     edges=[detector_data.event_index])
        # print("Events by pulse time: {}".format(events_by_pulse_time))
    return events


def _create_empty_event_data(event_data: List[DetectorData]):
    """
    If any NXdetector groups had pixel position data but no events
    then add an empty data array to make it easier to concatenate
    the data from different groups
    """
    empty_events = None
    detector_id_dtype = None
    for data in event_data:
        if data.events is not None:
            # If any event data were loaded then use an empty data
            # array with the same data types
            tof_dtype = data.events.coords[_time_of_flight_dim_name].dtype
            tof_unit = data.events.coords[_time_of_flight_dim_name].unit
            empty_events = _create_empty_events_data_array(
                tof_dtype, tof_unit,
                data.events.coords[_detector_dim_name].dtype)
            break
        elif data.detector_ids is not None:
            detector_id_dtype = data.detector_ids.dtype
    if empty_events is None:
        if detector_id_dtype is None:
            # Create empty data array with types/unit matching streamed event
            # data, this avoids need to convert to concatenate with event data
            # arriving from stream
            empty_events = _create_empty_events_data_array(
                np.int64, "ns", np.int32)
        else:
            # If detector_ids were loaded then match the type used for those
            empty_events = _create_empty_events_data_array(
                np.int64, "ns", detector_id_dtype)
    for data in event_data:
        if data.events is None:
            data.events = empty_events


def _load_data_from_each_nx_event_data(detector_data: Dict,
                                       event_data_groups: List[Group],
                                       file_root: h5py.File,
                                       nexus: LoadFromNexus,
                                       quiet: bool) -> List[DetectorData]:
    event_data = []
    for group in event_data_groups:
        parent_path = "/".join(group.path.split("/")[:-1])
        try:
            new_event_data = _load_event_group(
                group, file_root, nexus,
                detector_data.get(parent_path, DetectorData()), quiet)
            event_data.append(new_event_data)
            # Only pop from dictionary if we did not raise an
            # exception when loading events
            detector_data.pop(parent_path, DetectorData())
        except DetectorIdError as e:
            warn(f"Skipped loading detector ids for {group.path} due to:\n{e}")
            detector_data.pop(parent_path, DetectorData())
        except BadSource as e:
            warn(f"Skipped loading {group.path} due to:\n{e}")
        except SkipSource:
            pass  # skip without warning user

    for _, remaining_data in detector_data.items():
        if remaining_data.detector_ids is not None:
            event_data.append(remaining_data)

    return event_data


def _load_data_from_each_nx_detector(detector_groups: List[Group],
                                     file_root: h5py.File,
                                     nexus: LoadFromNexus) -> Dict:
    detector_data = {}
    for detector_group in detector_groups:
        detector_data[detector_group.path] = _load_detector(
            detector_group, file_root, nexus)
    return detector_data
