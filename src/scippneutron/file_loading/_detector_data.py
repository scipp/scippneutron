# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Union
import numpy as np
import scipp

from ._common import (BadSource, SkipSource, MissingDataset, MissingAttribute, Group)
from ._common import to_plain_index
import scipp as sc
from warnings import warn
from ._transformations import get_full_transformation_matrix
from ._nexus import LoadFromNexus
from .nxobject import NXobject

_bank_dimension = "bank"
_detector_dimension = "detector_id"
_event_dimension = "event"
_pulse_dimension = "pulse"
_pulse_time = "pulse_time"
_time_of_flight = "tof"


class DetectorIdError(Exception):
    pass


def _check_for_missing_fields(group: Group, nexus: LoadFromNexus):
    if nexus.contains_stream(group):
        # Do not warn about missing datasets if the group contains
        # a stream, as this will provide the missing data
        raise SkipSource("Data source is missing datasets"
                         "but contains a stream source for the data")

    required_fields = (
        "event_time_zero",
        "event_index",
        "event_time_offset",
    )
    for field in required_fields:
        found, msg = nexus.dataset_in_group(group, field)
        if not found:
            raise BadSource(msg)


def _convert_array_to_metres(array: np.ndarray, unit: str) -> np.ndarray:
    return sc.to_unit(
        sc.Variable(dims=["temporary_variable"],
                    values=array,
                    unit=unit,
                    dtype=np.float64), "m").values


def _load_pixel_positions(detector_group: Group, detector_ids_size: int,
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

    list_of_sizes = [
        x_positions.size, y_positions.size, z_positions.size, detector_ids_size
    ]
    if list_of_sizes.count(list_of_sizes[0]) != len(list_of_sizes):
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
        transformation = get_full_transformation_matrix(detector_group, nexus)
        for row_index in range(n_rows):
            array[row_index, :] = np.matmul(transformation, array[row_index, :])

        # Now the transformations are done we do not need the 4th
        # element in each position
        array = array[:, :3]

    return sc.vectors(dims=[_detector_dimension], values=array, unit=sc.units.m)


@dataclass
class DetectorData:
    event_data: Optional[sc.DataArray] = None
    detector_ids: Optional[sc.Variable] = None
    pixel_positions: Optional[sc.Variable] = None


def _create_empty_events_data_array(tof_dtype: Any = np.int64,
                                    tof_unit: Union[str, sc.Unit] = "ns",
                                    detector_id_dtype: Any = np.int32) -> sc.DataArray:
    data = sc.DataArray(data=sc.empty(dims=[_event_dimension],
                                      shape=[0],
                                      unit='counts',
                                      with_variances=True,
                                      dtype=np.float32),
                        coords={
                            _time_of_flight:
                            sc.empty(dims=[_event_dimension],
                                     shape=[0],
                                     dtype=tof_dtype,
                                     unit=tof_unit),
                            _detector_dimension:
                            sc.empty(dims=[_event_dimension],
                                     shape=[0],
                                     dtype=detector_id_dtype),
                        })
    indices = sc.array(dims=[_pulse_dimension], values=[], dtype='int64')
    return sc.DataArray(data=sc.bins(begin=indices,
                                     end=indices,
                                     dim=_event_dimension,
                                     data=data),
                        coords={
                            'pulse_time':
                            sc.zeros(dims=[_pulse_dimension],
                                     shape=[0],
                                     dtype='datetime64',
                                     unit='ns')
                        })


def _load_pulse_times(group: Group, nexus: LoadFromNexus, index=...) -> sc.Variable:
    time_zero_group = "event_time_zero"

    event_time_zero = nexus.load_dataset(group,
                                         time_zero_group,
                                         dimensions=[_pulse_dimension],
                                         index=index)

    try:
        pulse_times = sc.to_unit(event_time_zero, sc.units.ns, copy=False)
    except sc.UnitError:
        raise BadSource(f"Could not load pulse times: units attribute "
                        f"'{event_time_zero.unit}' in NXEvent at "
                        f"{group.name}/{time_zero_group} is not convertible"
                        f" to nanoseconds.")

    try:
        time_offset = nexus.get_string_attribute(
            nexus.get_dataset_from_group(group, time_zero_group), "offset")
    except MissingAttribute:
        time_offset = "1970-01-01T00:00:00Z"

    # Need to convert the values which were loaded as float64 into int64 to be able
    # to do datetime arithmetic. This needs to be done after conversion to ns to
    # avoid unnecessary loss of accuracy.
    pulse_times = pulse_times.astype(sc.DType.int64, copy=False)
    return pulse_times + sc.scalar(
        np.datetime64(time_offset), unit=sc.units.ns, dtype=sc.DType.datetime64)


def _load_detector(group: Group, nexus: LoadFromNexus) -> DetectorData:
    detector_number_ds_name = "detector_number"
    dataset_in_group, _ = nexus.dataset_in_group(group, detector_number_ds_name)
    detector_ids = None
    if dataset_in_group:
        detector_ids = nexus.load_dataset_from_group_as_numpy_array(
            group, detector_number_ds_name).flatten()
        detector_id_type = detector_ids.dtype.type

        detector_ids = sc.Variable(dims=[_detector_dimension],
                                   values=detector_ids,
                                   dtype=detector_id_type)

    pixel_positions = None
    pixel_positions_found, _ = nexus.dataset_in_group(group, "x_pixel_offset")
    if pixel_positions_found and detector_ids is not None:
        pixel_positions = _load_pixel_positions(group, detector_ids.shape[0], nexus)

    return DetectorData(detector_ids=detector_ids, pixel_positions=pixel_positions)


class NXevent_data(NXobject):
    @property
    def shape(self):
        return self._loader.get_shape(
            self._loader.get_dataset_from_group(self._group, "event_index"))

    @property
    def dims(self):
        return [_pulse_dimension]

    @property
    def unit(self):
        # Binned data, bins do not have a unit
        return None

    def _getitem(self, index):
        data = _load_event_group(self._group, self._loader, quiet=False, select=index)
        if 'detector_id' in data.bins.coords:
            data.bins.coords['event_id'] = data.bins.coords.pop('detector_id')
        data.bins.coords['event_time_offset'] = data.bins.coords.pop('tof')
        data.coords['event_time_zero'] = data.coords.pop('pulse_time')
        return data


def _load_event_group(group: Group, nexus: LoadFromNexus, quiet: bool,
                      select=tuple()) -> DetectorData:
    _check_for_missing_fields(group, nexus)
    index = to_plain_index([_pulse_dimension], select)

    def shape(name):
        return nexus.get_shape(nexus.get_dataset_from_group(group, name))

    max_index = shape("event_index")[0]
    single = False
    if index is Ellipsis or index == tuple():
        last_loaded = False
    else:
        if isinstance(index, int):
            single = True
            start, stop, _ = slice(index, None).indices(max_index)
            if start == stop:
                raise IndexError('Index {start} is out of range')
            index = slice(start, start + 1)
        start, stop, stride = index.indices(max_index)
        if stop + stride > max_index:
            last_loaded = False
        else:
            stop += stride
            last_loaded = True
        index = slice(start, stop, stride)

    event_index = nexus.load_dataset_from_group_as_numpy_array(
        group, "event_index", index)
    pulse_times = _load_pulse_times(group, nexus, index)

    num_event = shape("event_time_offset")[0]
    # Some files contain uint64 "max" indices, which turn into negatives during
    # conversion to int64. This is a hack to get arround this.
    event_index[event_index < 0] = num_event

    if len(event_index) > 0:
        event_select = slice(event_index[0],
                             event_index[-1] if last_loaded else num_event)
    else:
        event_select = slice(None)

    if nexus.dataset_in_group(group, "event_id")[0]:
        event_id = nexus.load_dataset(group,
                                      "event_id", [_event_dimension],
                                      index=event_select)
    else:
        event_id = None

    event_time_offset = nexus.load_dataset(group,
                                           "event_time_offset", [_event_dimension],
                                           index=event_select)

    # Weights are not stored in NeXus, so use 1s
    weights = sc.ones(dims=[_event_dimension],
                      shape=event_time_offset.shape,
                      unit='counts',
                      dtype=np.float32,
                      with_variances=True)

    events = sc.DataArray(data=weights, coords={_time_of_flight: event_time_offset})
    if event_id is not None:
        events.coords[_detector_dimension] = event_id

    if not last_loaded:
        event_index = np.append(event_index, num_event)
    else:
        # Not a bin-edge coord, all events in bin are associated with same (previous)
        # pulse time value
        pulse_times = pulse_times[:-1]

    event_index = sc.array(dims=[_pulse_dimension],
                           values=event_index,
                           dtype=sc.DType.int64)

    event_index -= event_index.min()

    # There is some variation in the last recorded event_index in files from different
    # institutions. We try to make sure here that it is what would be the first index of
    # the next pulse. In other words, ensure that event_index includes the bin edge for
    # the last pulse.
    if single:
        begins = event_index[_pulse_dimension, 0]
        ends = event_index[_pulse_dimension, 1]
        pulse_times = pulse_times[_pulse_dimension, 0]
    else:
        begins = event_index[_pulse_dimension, :-1]
        ends = event_index[_pulse_dimension, 1:]

    try:
        binned = sc.bins(data=events, dim=_event_dimension, begin=begins, end=ends)
    except sc.SliceError:
        raise BadSource(f"Event index in NXEvent at {group.name}/event_index was not"
                        f" ordered. The index must be ordered to load pulse times.")

    if not quiet:
        print(f"Loaded {len(event_time_offset)} events from "
              f"{nexus.get_name(group)} containing {num_event} events")

    return sc.DataArray(data=binned, coords={"pulse_time": pulse_times})


def _check_event_ids_and_det_number_types_valid(detector_id_type: Any,
                                                event_id_type: Any):
    """
    These must be integers and must be the same type or we'll have
    problems trying to bin events by detector id. Check here so that
    we can give a useful warning to the user and skip loading the
    current event group.
    """
    def is_integer_type(type_to_check: Any) -> bool:
        return type_to_check == sc.DType.int32 or \
               type_to_check == sc.DType.int64

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


def load_detector_data(event_data_groups: List[Group], detector_groups: List[Group],
                       nexus: LoadFromNexus, quiet: bool,
                       bin_by_pixel: bool) -> Optional[sc.DataArray]:
    detectors = _load_data_from_each_nx_detector(detector_groups, nexus)
    detectors = _load_data_from_each_nx_event_data(detectors, event_data_groups, nexus,
                                                   quiet)

    if not detectors:
        # If there were no data to load we are done
        return

    def get_detector_id(data: DetectorData):
        # Assume different detector banks do not have
        # intersecting ranges of detector ids
        if data.detector_ids is None:
            return 0
        return data.detector_ids.values[0]

    detectors.sort(key=get_detector_id)

    _create_empty_event_data(detectors)

    pixel_positions_loaded = all(
        [data.pixel_positions is not None for data in detectors])

    def _bin_events(data: DetectorData):
        if not bin_by_pixel:
            # If loading "raw" data, leave binned by pulse.
            return data.event_data
        if data.detector_ids is None:
            # If detector ids were not found in an associated detector group
            # we will just have to bin according to whatever
            # ids we have a events for (pixels with no recorded events
            # will not have a bin)
            event_id = data.event_data.bins.constituents['data'].coords[
                _detector_dimension]
            data.detector_ids = sc.array(dims=[_detector_dimension],
                                         values=np.unique(event_id.values))

        # Events in the NeXus file are effectively binned by pulse
        # (because they are recorded chronologically)
        # but for reduction it is more useful to bin by detector id
        # Broadcast pulse times to events
        data.event_data.bins.coords['pulse_time'] = sc.bins_like(
            data.event_data, fill_value=data.event_data.coords['pulse_time'])
        # TODO Look into using `erase=[_pulse_dimension]` instead of binning
        # underlying buffer. Must prove that performance can be unaffected.
        da = sc.bin(data.event_data.bins.constituents['data'],
                    groups=[data.detector_ids])
        # Add a single time-of-flight bin
        da = sc.DataArray(data=sc.broadcast(da.data,
                                            dims=da.dims + [_time_of_flight],
                                            shape=da.shape + [1]),
                          coords={_detector_dimension: data.detector_ids})
        if pixel_positions_loaded:
            # TODO: the name 'position' should probably not be hard-coded but moved
            # to a variable that cah be changed in a single place.
            da.coords['position'] = data.pixel_positions
        return da

    _dim = _detector_dimension if bin_by_pixel else _bank_dimension
    events = sc.concat([_bin_events(item) for item in detectors], _dim)

    if bin_by_pixel:
        _min_tof = events.bins.coords[_time_of_flight].min()
        _max_tof = events.bins.coords[_time_of_flight].max()
        # This can happen if there were no events in the file at all as sc.min will
        # return double_max and sc.max will return double_min
        if _min_tof.value >= _max_tof.value:
            _min_tof, _max_tof = _max_tof, _min_tof
        if np.issubdtype(type(_max_tof.value), np.integer):
            if _max_tof.value != np.iinfo(type(_max_tof.value)).max:
                _max_tof += sc.ones_like(_max_tof)
        else:
            if _max_tof.value != np.finfo(type(_max_tof.value)).max:
                _max_tof.value = np.nextafter(_max_tof.value, float("inf"))
        events.coords[_time_of_flight] = sc.concat([_min_tof, _max_tof],
                                                   _time_of_flight)

    return events


def _create_empty_event_data(detectors: List[DetectorData]):
    """
    If any NXdetector groups had pixel position data but no events
    then add an empty data array to make it easier to concatenate
    the data from different groups
    """
    empty_events = None
    detector_id_dtype = None
    for data in detectors:
        if data.event_data is not None:
            # If any event data were loaded then use an empty data
            # array with the same data types
            constituents = data.event_data.bins.constituents
            constituents['begin'] = sc.zeros_like(constituents['begin'])
            constituents['end'] = sc.zeros_like(constituents['end'])
            constituents['data'] = constituents['data'][_event_dimension, 0:0].copy()
            empty_events = data.event_data.copy(deep=False)
            empty_events.data = sc.bins(**constituents)
            break
        elif data.detector_ids is not None:
            detector_id_dtype = data.detector_ids.dtype
    if empty_events is None:
        if detector_id_dtype is None:
            # Create empty data array with types/unit matching streamed event
            # data, this avoids need to convert to concatenate with event data
            # arriving from stream
            empty_events = _create_empty_events_data_array(np.int64, "ns", np.int32)
        else:
            # If detector_ids were loaded then match the type used for those
            empty_events = _create_empty_events_data_array(np.int64, "ns",
                                                           detector_id_dtype)
    for data in detectors:
        if data.event_data is None:
            data.event_data = empty_events


def _load_data_from_each_nx_event_data(detector_data: Dict,
                                       event_data_groups: List[Group],
                                       nexus: LoadFromNexus,
                                       quiet: bool) -> List[DetectorData]:
    event_data = []
    for group in event_data_groups:
        parent_path = "/".join(group.name.split("/")[:-1])
        try:
            new_detector_data = detector_data.get(parent_path, DetectorData())
            new_event_data = _load_event_group(group, nexus, quiet)

            event_id_dtype = new_event_data.bins.constituents['data'].coords[
                _detector_dimension].dtype
            _check_event_ids_and_det_number_types_valid(
                event_id_dtype if new_detector_data.detector_ids is None else
                new_detector_data.detector_ids.dtype, event_id_dtype)
            new_detector_data.event_data = new_event_data

            event_data.append(new_detector_data)
            # Only pop from dictionary if we did not raise an
            # exception when loading events
            detector_data.pop(parent_path, DetectorData())
        except DetectorIdError as e:
            warn(f"Skipped loading detector ids for {group.name} due to:\n{e}")
            detector_data.pop(parent_path, DetectorData())
        except BadSource as e:
            warn(f"Skipped loading {group.name} due to:\n{e}")
        except SkipSource:
            pass  # skip without warning user

    for _, remaining_data in detector_data.items():
        if remaining_data.detector_ids is not None:
            event_data.append(remaining_data)

    return event_data


def _load_data_from_each_nx_detector(detector_groups: List[Group],
                                     nexus: LoadFromNexus) -> Dict:
    detector_data = {}
    for detector_group in detector_groups:
        detector_data[detector_group.name] = _load_detector(detector_group, nexus)
    return detector_data
