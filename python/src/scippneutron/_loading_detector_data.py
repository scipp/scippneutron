# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from dataclasses import dataclass
import h5py
from typing import Optional, List
import numpy as np
from ._loading_common import BadSource, ensure_supported_int_type, load_dataset
import scipp as sc
from datetime import datetime
from warnings import warn
from itertools import groupby
from ._loading_transformations import get_full_transformation_matrix

_detector_dimension = "detector_id"
_event_dimension = "event"


def _all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def _check_for_missing_fields(group: h5py.Group) -> str:
    error_message = ""
    required_fields = (
        "event_time_zero",
        "event_index",
        "event_id",
        "event_time_offset",
    )
    for field in required_fields:
        if field not in group:
            error_message += f"Unable to load data from NXevent_data " \
                             f"at '{group.name}' due to missing '{field}'" \
                             f" field\n"
    return error_message


def _iso8601_to_datetime(iso8601: str) -> Optional[datetime]:
    try:
        return datetime.strptime(
            iso8601.translate(str.maketrans('', '', ':-Z')),
            "%Y%m%dT%H%M%S.%f")
    except ValueError:
        # Did not understand the format of the input string
        return None


def _load_pixel_positions(detector_group: h5py.Group, detector_ids_size: int,
                          file_root: h5py.File) -> Optional[sc.Variable]:
    try:
        x_positions = detector_group["x_pixel_offset"][...].flatten()
        y_positions = detector_group["y_pixel_offset"][...].flatten()
    except KeyError:
        return None
    try:
        z_positions = detector_group["z_pixel_offset"][...].flatten()
    except KeyError:
        z_positions = np.zeros_like(x_positions)

    if not _all_equal((x_positions.size, y_positions.size, z_positions.size,
                       detector_ids_size)):
        warn(f"Skipped loading pixel positions as pixel offset and id "
             f"dataset sizes do not match in {detector_group.name}")
        return None

    array = np.array([x_positions, y_positions, z_positions]).T

    if "depends_on" in detector_group:
        # Add fourth element of 1 to each vertex, indicating these are
        # positions not direction vectors
        n_rows = array.shape[0]
        array = np.hstack((array, np.ones((n_rows, 1))))

        # Get and apply transformation matrix
        transformation = get_full_transformation_matrix(
            detector_group, file_root)
        for row_index in range(array.shape[0]):
            array[row_index, :] = np.matmul(transformation,
                                            array[row_index, :])

        # Now the transformations are done we do not need the 4th
        # element in each position
        array = array[:, :3]

    return sc.Variable([_detector_dimension],
                       values=array,
                       dtype=sc.dtype.vector_3_float64)


@dataclass
class DetectorData:
    events: sc.Variable
    detector_ids: sc.Variable
    pixel_positions: Optional[sc.Variable] = None


def _load_event_group(group: h5py.Group, file_root: h5py.File,
                      quiet: bool) -> DetectorData:
    error_msg = _check_for_missing_fields(group)
    if error_msg:
        raise BadSource(error_msg)

    # There is some variation in the last recorded event_index in files
    # from different institutions. We try to make sure here that it is what
    # would be the first index of the next pulse.
    # In other words, ensure that event_index includes the bin edge for
    # the last pulse.
    event_id_ds = group["event_id"]
    event_index = group["event_index"][...].astype(
        ensure_supported_int_type(group["event_index"].dtype.type))
    if event_index[-1] < event_id_ds.len():
        event_index = np.append(
            event_index,
            np.array([event_id_ds.len() - 1]).astype(event_index.dtype),
        )
    else:
        event_index[-1] = event_id_ds.len()

    number_of_events = event_index[-1]
    event_time_offset = load_dataset(group["event_time_offset"],
                                     [_event_dimension])
    event_id = load_dataset(event_id_ds, [_event_dimension])

    # Weights are not stored in NeXus, so use 1s
    weights = sc.ones(dims=[_event_dimension],
                      shape=event_id.shape,
                      dtype=np.float32)

    detector_number_ds_name = "detector_number"
    if detector_number_ds_name in group.parent:
        # Hopefully the detector ids are recorded in the file
        detector_ids = group.parent[detector_number_ds_name][...].flatten()
    else:
        # Otherwise we'll just have to bin according to whatever
        # ids we have a events for (pixels with no recorded events
        # will not have a bin)
        detector_ids = np.unique(event_id.values)

    detector_id_type = ensure_supported_int_type(detector_ids.dtype.type)
    event_id_type = ensure_supported_int_type(event_id_ds.dtype.type)
    _check_event_ids_and_det_number_types_valid(detector_id_type,
                                                event_id_type, group.name)

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
    if "x_pixel_offset" in detector_group:
        pixel_positions = _load_pixel_positions(detector_group,
                                                detector_ids.shape[0],
                                                file_root)

    if not quiet:
        print(f"Loaded event data from {group.name} containing "
              f"{number_of_events} events")

    return DetectorData(data, detector_ids, pixel_positions)


def _check_event_ids_and_det_number_types_valid(detector_id_type: np.dtype,
                                                event_id_type: np.dtype,
                                                group_name: str):
    """
    These must be integers and must be the same type or we'll have
    problems trying to bin events by detector id. Check here so that
    we can give a useful warning to the user and skip loading the
    current event group.
    """
    if not np.issubdtype(detector_id_type, np.integer):
        raise BadSource(
            f"detector_numbers dataset in NXdetector is not an integer "
            f"type, skipping loading {group_name}")
    if not np.issubdtype(event_id_type, np.integer):
        raise BadSource(f"event_ids dataset is not an integer type, "
                        f"skipping loading {group_name}")
    if detector_id_type != event_id_type:
        raise BadSource(
            f"event_ids and detector_numbers datasets in corresponding "
            f"NXdetector were not of the same type, skipping "
            f"loading {group_name}")


def load_detector_data(event_data_groups: List[h5py.Group],
                       file_root: h5py.File,
                       quiet: bool) -> Optional[sc.DataArray]:
    event_data = []
    for group in event_data_groups:
        try:
            new_event_data = _load_event_group(group, file_root, quiet)
            event_data.append(new_event_data)
        except BadSource as e:
            warn(f"Skipped loading {group.name} due to:\n{e}")

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
