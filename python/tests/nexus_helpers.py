from dataclasses import dataclass
from typing import List, Union, Iterator, Optional
import h5py
import numpy as np
from enum import Enum
from contextlib import contextmanager

h5root = Union[h5py.File, h5py.Group]


def _create_nx_class(group_name: str, nx_class_name: str,
                     parent: h5root) -> h5py.Group:
    nx_class = parent.create_group(group_name)
    nx_class.attrs["NX_class"] = nx_class_name
    return nx_class


@contextmanager
def in_memory_hdf5_file_with_two_nxentry() -> Iterator[h5py.File]:
    nexus_file = h5py.File('in_memory_events.nxs',
                           mode='w',
                           driver="core",
                           backing_store=False)
    try:
        _create_nx_class("entry_1", "NXentry", nexus_file)
        _create_nx_class("entry_2", "NXentry", nexus_file)
        yield nexus_file
    finally:
        nexus_file.close()


@dataclass
class EventData:
    event_id: np.ndarray
    event_time_offset: np.ndarray
    event_time_zero: np.ndarray
    event_index: np.ndarray


@dataclass
class Log:
    name: str
    value: Optional[np.ndarray]
    time: Optional[np.ndarray] = None
    value_units: Optional[str] = None
    time_units: Optional[str] = None


@dataclass
class Detector:
    detector_numbers: np.ndarray
    event_data: Optional[EventData] = None
    log: Optional[Log] = None
    x_offsets: Optional[np.ndarray] = None
    y_offsets: Optional[np.ndarray] = None
    z_offsets: Optional[np.ndarray] = None


class TransformationType(Enum):
    TRANSLATION = "translation"
    ROTATION = "rotation"


@dataclass
class Transformation:
    transform_type: TransformationType
    vector: np.ndarray
    value: Optional[np.ndarray]
    time: Optional[np.ndarray] = None
    depends_on: Optional["Transformation"] = None
    offset: Optional[np.ndarray] = None
    value_units: Optional[str] = None
    time_units: Optional[str] = None


@dataclass
class Sample:
    name: str
    depends_on: Optional[Transformation] = None
    distance: Optional[float] = None


def _add_event_data_group_to_file(data: EventData, parent_group: h5py.Group,
                                  group_name: str):
    event_group = _create_nx_class(group_name, "NXevent_data", parent_group)
    event_group.create_dataset("event_id", data=data.event_id)
    event_time_offset_ds = event_group.create_dataset(
        "event_time_offset", data=data.event_time_offset)
    event_time_offset_ds.attrs["units"] = "ns"
    event_time_zero_ds = event_group.create_dataset("event_time_zero",
                                                    data=data.event_time_zero)
    event_time_zero_ds.attrs["units"] = "ns"
    event_group.create_dataset("event_index", data=data.event_index)


def _add_detector_group_to_file(detector: Detector, parent_group: h5py.Group,
                                group_name: str) -> h5py.Group:
    detector_group = _create_nx_class(group_name, "NXdetector", parent_group)
    detector_group.create_dataset("detector_number",
                                  data=detector.detector_numbers)
    if detector.x_offsets is not None:
        detector_group.create_dataset("x_pixel_offset",
                                      data=detector.x_offsets)
    if detector.y_offsets is not None:
        detector_group.create_dataset("y_pixel_offset",
                                      data=detector.y_offsets)
    if detector.z_offsets is not None:
        detector_group.create_dataset("z_pixel_offset",
                                      data=detector.z_offsets)
    return detector_group


def _add_log_group_to_file(log: Log, parent_group: h5py.Group) -> h5py.Group:
    log_group = _create_nx_class(log.name, "NXlog", parent_group)
    if log.value is not None:
        value_ds = log_group.create_dataset("value", data=log.value)
        if log.value_units is not None:
            value_ds.attrs.create("units", data=log.value_units)
    if log.time is not None:
        time_ds = log_group.create_dataset("time", data=log.time)
        if log.time_units is not None:
            time_ds.attrs.create("units", data=log.time_units)
    return log_group


def _add_transformations_to_file(transform: Transformation,
                                 parent_group: h5py.Group) -> str:
    transform_chain = []
    while transform.depends_on is not None:
        transform_chain.append(transform.depends_on)
        transform = transform.depends_on

    transforms_group = _create_nx_class("transformations", "NXtransformations",
                                        parent_group)
    transform_chain.reverse()
    depends_on_str = None
    for transform_number, transform in enumerate(transform_chain):
        if transform.time is not None:
            depends_on_str = _add_transformation_as_log(
                transform, transform_number, transforms_group, depends_on_str)
        else:
            depends_on_str = _add_transformation_as_dataset(
                transform, transform_number, transforms_group, depends_on_str)
    return depends_on_str


def _add_transformation_as_dataset(transform: Transformation,
                                   transform_number: int,
                                   transforms_group: h5py.Group,
                                   depends_on: Optional[str]) -> str:
    added_transform = transforms_group.create_dataset(
        f"transform_{transform_number}", data=transform.value)
    _add_transform_attributes(added_transform, depends_on, transform)
    if transform.value_units is not None:
        added_transform.attrs["units"] = transform.value_units
    return added_transform.name


def _add_transform_attributes(added_transform: Union[h5py.Group, h5py.Dataset],
                              depends_on: Optional[str],
                              transform: Transformation):
    added_transform.attrs["vector"] = transform.vector
    added_transform.attrs[
        "transformation_type"] = transform.transform_type.value
    if transform.offset is not None:
        added_transform.attrs["offset"] = transform.offset
    if depends_on is not None:
        added_transform.attrs["depends_on"] = depends_on
    else:
        added_transform.attrs["depends_on"] = "."  # means end of chain


def _add_transformation_as_log(transform: Transformation,
                               transform_number: int,
                               transforms_group: h5py.Group,
                               depends_on: Optional[str]) -> str:
    added_transform = _add_log_group_to_file(
        Log(f"transform_{transform_number}", transform.value, transform.time,
            transform.value_units, transform.time_units), transforms_group)
    _add_transform_attributes(added_transform, depends_on, transform)
    return added_transform.name


class InMemoryNexusFileBuilder:
    """
    Allows building an in-memory NeXus file for use in tests
    """
    def __init__(self):
        self._event_data: List[EventData] = []
        self._detectors: List[Detector] = []
        self._logs: List[Log] = []
        self._instrument_name = None
        self._title = None
        self._sample = None

    def add_detector(self, detector: Detector):
        self._detectors.append(detector)

    def add_event_data(self, event_data: EventData):
        self._event_data.append(event_data)

    def add_log(self, log: Log):
        self._logs.append(log)

    def add_instrument(self, name: str):
        self._instrument_name = name

    def add_title(self, title: str):
        self._title = title

    def add_sample(self, sample: Sample):
        self._sample = sample

    @contextmanager
    def file(self) -> Iterator[h5py.File]:
        # "core" driver means file is "in-memory" not on disk.
        # backing_store=False prevents file being written to
        # disk on flush() or close().
        nexus_file = h5py.File('in_memory_events.nxs',
                               mode='w',
                               driver="core",
                               backing_store=False)
        try:
            entry_group = _create_nx_class("entry", "NXentry", nexus_file)
            if self._title is not None:
                entry_group.create_dataset("title", data=self._title)
            self._write_event_data(entry_group)
            self._write_logs(entry_group)
            self._write_sample(entry_group)
            if self._instrument_name is None:
                parent_group = entry_group
            else:
                parent_group = self._write_instrument(entry_group)
            self._write_detectors(parent_group)
            yield nexus_file
        finally:
            nexus_file.close()

    def _write_sample(self, parent_group: h5py.Group):
        if self._sample is not None:
            sample_group = _create_nx_class(self._sample.name, "NXsample",
                                            parent_group)
            if self._sample.depends_on is not None:
                depends_on = _add_transformations_to_file(
                    self._sample.depends_on, sample_group)
                sample_group.create_dataset("depends_on", data=depends_on)
            if self._sample.distance is not None:
                sample_group.create_dataset("distance",
                                            data=self._sample.distance)

    def _write_instrument(self, parent_group: h5py.Group) -> h5py.Group:
        instrument_group = _create_nx_class("instrument", "NXinstrument",
                                            parent_group)
        instrument_group.create_dataset("name", data=self._instrument_name)
        return instrument_group

    def _write_detectors(self, parent_group: h5py.Group):
        for detector_index, detector in enumerate(self._detectors):
            detector_group = _add_detector_group_to_file(
                detector, parent_group, f"detector_{detector_index}")
            if detector.event_data is not None:
                _add_event_data_group_to_file(detector.event_data,
                                              detector_group, "events")
            if detector.log is not None:
                _add_log_group_to_file(detector.log, detector_group)

    def _write_event_data(self, parent_group: h5py.Group):
        for event_data_index, event_data in enumerate(self._event_data):
            _add_event_data_group_to_file(event_data, parent_group,
                                          f"events_{event_data_index}")

    def _write_logs(self, parent_group: h5py.Group):
        for log in self._logs:
            _add_log_group_to_file(log, parent_group)
