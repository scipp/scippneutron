# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from dataclasses import dataclass
import h5py
from typing import Optional, List
import numpy as np
from ._loading_common import (BadSource, MissingDataset, Group)
import scipp as sc
from datetime import datetime
from warnings import warn
from itertools import groupby
from ._loading_transformations import get_full_transformation_matrix
from ._loading_nexus import LoadFromNexus, GroupObject

_detector_dimension = "detector_id"
_event_dimension = "event"


def _all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def _check_for_missing_fields(group: GroupObject, nexus: LoadFromNexus) -> str:
    required_fields = (
        "event_time_zero",
        "event_index",
        "event_id",
        "event_time_offset",
    )
    for field in required_fields:
        found, msg = nexus.dataset_in_group(group, field)
        if not found:
            return msg
    return ""


def _iso8601_to_datetime(iso8601: str) -> Optional[datetime]:
    try:
        return datetime.strptime(
            iso8601.translate(str.maketrans('', '', ':-Z')),
            "%Y%m%dT%H%M%S.%f")
    except ValueError:
        # Did not understand the format of the input string
        return None


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

    return sc.Variable([_detector_dimension],
                       values=array,
                       dtype=sc.dtype.vector_3_float64,
                       unit=sc.units.m)


@dataclass
class DetectorData:
    events: sc.Variable
    detector_ids: sc.Variable
    pixel_positions: Optional[sc.Variable] = None


def _load_event_group(group: Group, file_root: h5py.File, nexus: LoadFromNexus,
                      quiet: bool) -> DetectorData:
    error_msg = _check_for_missing_fields(group.group, nexus)
    if error_msg:
        raise BadSource(error_msg)

    # There is some variation in the last recorded event_index in files
    # from different institutions. We try to make sure here that it is what
    # would be the first index of the next pulse.
    # In other words, ensure that event_index includes the bin edge for
    # the last pulse.
    event_id = nexus.load_dataset(group.group, "event_id", [_event_dimension])
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
                                           [_event_dimension])

    # Weights are not stored in NeXus, so use 1s
    weights = sc.ones(dims=[_event_dimension],
                      shape=event_id.shape,
                      dtype=np.float32)

    detector_number_ds_name = "detector_number"
    dataset_in_group, _ = nexus.dataset_in_group(group.parent,
                                                 detector_number_ds_name)
    if dataset_in_group:
        # Hopefully the detector ids are recorded in the file
        detector_ids = nexus.load_dataset_from_group_as_numpy_array(
            group.parent, detector_number_ds_name).flatten()
    else:
        # Otherwise we'll just have to bin according to whatever
        # ids we have a events for (pixels with no recorded events
        # will not have a bin)
        detector_ids = np.unique(event_id.values)

    detector_id_type = detector_ids.dtype.type
    event_id_type = nexus.get_dataset_numpy_dtype(group.group, "event_id")
    _check_event_ids_and_det_number_types_valid(detector_id_type,
                                                event_id_type)

    detector_ids = sc.Variable(dims=[_detector_dimension],
                               values=detector_ids,
                               dtype=detector_id_type)

    data_dict = {
        "data": weights,
        "coords": {
            "tof": event_time_offset,
            _detector_dimension: event_id
        }
    }
    data = sc.detail.move_to_data_array(**data_dict)

    detector_group = group.parent
    pixel_positions = None
    pixel_positions_found, _ = nexus.dataset_in_group(detector_group,
                                                      "x_pixel_offset")
    if pixel_positions_found:
        pixel_positions = _load_pixel_positions(detector_group,
                                                detector_ids.shape[0],
                                                file_root, nexus)

    if not quiet:
        print(f"Loaded event data from "
              f"{group.path} containing {number_of_events} events")

    return DetectorData(data, detector_ids, pixel_positions)


def _check_event_ids_and_det_number_types_valid(detector_id_type: np.dtype,
                                                event_id_type: np.dtype):
    """
    These must be integers and must be the same type or we'll have
    problems trying to bin events by detector id. Check here so that
    we can give a useful warning to the user and skip loading the
    current event group.
    """
    if not np.issubdtype(detector_id_type, np.integer):
        raise BadSource(
            "detector_numbers dataset in NXdetector is not an integer "
            "type")
    if not np.issubdtype(event_id_type, np.integer):
        raise BadSource("event_ids dataset is not an integer type")
    if detector_id_type != event_id_type:
        raise BadSource(
            "event_ids and detector_numbers datasets in corresponding "
            "NXdetector were not of the same type")


def load_detector_data(event_data_groups: List[Group], file_root: h5py.File,
                       nexus: LoadFromNexus,
                       quiet: bool) -> Optional[sc.DataArray]:
    event_data = []
    for group in event_data_groups:
        try:
            new_event_data = _load_event_group(group, file_root, nexus, quiet)
            event_data.append(new_event_data)
        except BadSource as e:
            warn(f"Skipped loading {group.path} due to:\n{e}")

    if not event_data:
        return

    def get_detector_id(detector_data: DetectorData):
        # Assume different detector banks do not have
        # intersecting ranges of detector ids
        return detector_data.detector_ids.values[0]

    event_data.sort(key=get_detector_id)

    pixel_positions_loaded = all(
        [data.pixel_positions is not None for data in event_data])
    detector_data = event_data.pop(0)
    # Events in the NeXus file are effectively binned by pulse
    # (because they are recorded chronologically)
    # but for reduction it is more useful to bin by detector id
    events = sc.bin(detector_data.events, groups=[detector_data.detector_ids])
    if pixel_positions_loaded:
        events.coords['position'] = detector_data.pixel_positions
    while event_data:
        detector_data = event_data.pop(0)
        new_events = sc.bin(detector_data.events,
                            groups=[detector_data.detector_ids])
        if pixel_positions_loaded:
            new_events.coords['position'] = detector_data.pixel_positions
        events = sc.concatenate(events, new_events, dim=_detector_dimension)
    return events
