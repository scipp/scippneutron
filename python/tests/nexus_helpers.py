from dataclasses import dataclass
from typing import List, Union, Iterator, Optional, Dict
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


class TransformationType(Enum):
    TRANSLATION = "translation"
    ROTATION = "rotation"


@dataclass
class Transformation:
    transform_type: TransformationType
    vector: np.ndarray
    value: Optional[np.ndarray]
    time: Optional[np.ndarray] = None
    depends_on: Union["Transformation", str, None] = None
    offset: Optional[np.ndarray] = None
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
    offsets_unit: Optional[str] = None
    depends_on: Optional[Transformation] = None


@dataclass
class Sample:
    name: str
    depends_on: Optional[Transformation] = None
    distance: Optional[float] = None
    distance_units: Optional[str] = None


@dataclass
class Source:
    name: str
    depends_on: Optional[Transformation] = None
    distance: Optional[float] = None
    distance_units: Optional[str] = None


@dataclass
class Link:
    new_path: str
    target_path: str


class InMemoryNeXusWriter:
    @staticmethod
    def add_dataset(parent: h5py.Group, name: str,
                    data: Union[str, np.ndarray]) -> h5py.Dataset:
        return parent.create_dataset(name, data=data)

    @staticmethod
    def add_attribute(parent: Union[h5py.Group, h5py.Dataset], name: str,
                      value: Union[str, np.ndarray]):
        parent.attrs[name] = value

    @staticmethod
    def add_group(parent: h5py.Group, name: str) -> h5py.Group:
        return parent.create_group(name)

    @staticmethod
    def add_hard_link(file_root: h5py.File, new_path: str, target_path: str):
        file_root[new_path] = file_root[target_path]

    @staticmethod
    def add_soft_link(file_root: h5py.File, new_path: str, target_path: str):
        file_root[new_path] = h5py.SoftLink(target_path)


class JsonWriter:
    pass


class NexusBuilder:
    """
    Allows building an in-memory NeXus file for use in tests
    """
    def __init__(self):
        self._event_data: List[EventData] = []
        self._detectors: List[Detector] = []
        self._logs: List[Log] = []
        self._instrument_name: Optional[str] = None
        self._title: Optional[str] = None
        self._sample: List[Sample] = []
        self._source: List[Source] = []
        self._hard_links: List[Link] = []
        self._soft_links: List[Link] = []
        self._writer = None

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
        self._sample.append(sample)

    def add_source(self, source: Source):
        self._source.append(source)

    def add_hard_link(self, link: Link):
        self._hard_links.append(link)

    def add_soft_link(self, link: Link):
        self._soft_links.append(link)

    def add_component(self, component: Union[Sample, Source]):
        # This is a little ugly, but allows parametrisation
        # of tests which should work for sample and source
        if isinstance(component, Sample):
            self.add_sample(component)
        elif isinstance(component, Source):
            self.add_source(component)

    @contextmanager
    def file(self) -> Iterator[h5py.File]:
        # "core" driver means file is "in-memory" not on disk.
        # backing_store=False prevents file being written to
        # disk on flush() or close().
        nexus_file = h5py.File('in_memory_events.nxs',
                               mode='w',
                               driver="core",
                               backing_store=False)
        self._writer = InMemoryNeXusWriter()
        try:
            self._write_file(nexus_file)
            yield nexus_file
        finally:
            nexus_file.close()

    def _write_file(self, nexus_file: h5py.File):
        entry_group = self._create_nx_class("entry", "NXentry", nexus_file)
        if self._title is not None:
            self._writer.add_dataset(entry_group, "title", data=self._title)
        self._write_event_data(entry_group)
        self._write_logs(entry_group)
        self._write_sample(entry_group)
        self._write_source(entry_group)
        if self._instrument_name is None:
            parent_group = entry_group
        else:
            parent_group = self._write_instrument(entry_group)
        self._write_detectors(parent_group)
        self._write_links(nexus_file)

    def create_file_on_disk(self, filename: str):
        """
        Create a file on disk, do not use this in tests, it is intended to
        be used as a tool during test development. Output file can be
        explored using a tool such as HDFView.
        """
        nexus_file = h5py.File(filename, mode='w')
        self._writer = InMemoryNeXusWriter()
        try:
            self._write_file(nexus_file)
        finally:
            nexus_file.close()

    def _write_links(self, file_root: Union[h5py.Group, Dict]):
        for hard_link in self._hard_links:
            self._writer.add_hard_link(file_root, hard_link.new_path,
                                       hard_link.target_path)
        for soft_link in self._soft_links:
            self._writer.add_soft_link(file_root, soft_link.new_path,
                                       soft_link.target_path)

    def _write_sample(self, parent_group: Union[h5py.Group, Dict]):
        for sample in self._sample:
            sample_group = self._create_nx_class(sample.name, "NXsample",
                                                 parent_group)
            if sample.depends_on is not None:
                depends_on = self._add_transformations_to_file(
                    sample.depends_on, sample_group)
                self._writer.add_dataset(sample_group,
                                         "depends_on",
                                         data=depends_on)
            if sample.distance is not None:
                distance_ds = self._writer.add_dataset(sample_group,
                                                       "distance",
                                                       data=sample.distance)
                if sample.distance_units is not None:
                    self._writer.add_attribute(distance_ds, "units",
                                               sample.distance_units)

    def _write_source(self, parent_group: Union[h5py.Group, Dict]):
        for source in self._source:
            source_group = self._create_nx_class(source.name, "NXsource",
                                                 parent_group)
            if source.depends_on is not None:
                depends_on = self._add_transformations_to_file(
                    source.depends_on, source_group)
                self._writer.add_dataset(source_group,
                                         "depends_on",
                                         data=depends_on)
            if source.distance is not None:
                distance_ds = self._writer.add_dataset(source_group,
                                                       "distance",
                                                       data=source.distance)
                if source.distance_units is not None:
                    self._writer.add_attribute(distance_ds, "units",
                                               source.distance_units)

    def _write_instrument(
            self, parent_group: Union[h5py.Group,
                                      Dict]) -> Union[h5py.Group, Dict]:
        instrument_group = self._create_nx_class("instrument", "NXinstrument",
                                                 parent_group)
        self._writer.add_dataset(instrument_group, "name",
                                 self._instrument_name)
        return instrument_group

    def _write_detectors(self, parent_group: Union[h5py.Group, Dict]):
        for detector_index, detector in enumerate(self._detectors):
            detector_group = self._add_detector_group_to_file(
                detector, parent_group, f"detector_{detector_index}")
            if detector.event_data is not None:
                self._add_event_data_group_to_file(detector.event_data,
                                                   detector_group, "events")
            if detector.log is not None:
                self._add_log_group_to_file(detector.log, detector_group)
            if detector.depends_on is not None:
                depends_on = self._add_transformations_to_file(
                    detector.depends_on, detector_group)
                self._writer.add_dataset(detector_group,
                                         "depends_on",
                                         data=depends_on)

    def _write_event_data(self, parent_group: Union[h5py.Group, Dict]):
        for event_data_index, event_data in enumerate(self._event_data):
            self._add_event_data_group_to_file(event_data, parent_group,
                                               f"events_{event_data_index}")

    def _write_logs(self, parent_group: Union[h5py.Group, Dict]):
        for log in self._logs:
            self._add_log_group_to_file(log, parent_group)

    def _add_event_data_group_to_file(self, data: EventData,
                                      parent_group: h5py.Group,
                                      group_name: str):
        event_group = self._create_nx_class(group_name, "NXevent_data",
                                            parent_group)
        self._writer.add_dataset(event_group, "event_id", data=data.event_id)
        event_time_offset_ds = self._writer.add_dataset(
            event_group, "event_time_offset", data=data.event_time_offset)
        self._writer.add_attribute(event_time_offset_ds, "units", "ns")
        event_time_zero_ds = self._writer.add_dataset(
            event_group, "event_time_zero", data=data.event_time_zero)
        self._writer.add_attribute(event_time_zero_ds, "units", "ns")
        self._writer.add_dataset(event_group,
                                 "event_index",
                                 data=data.event_index)

    def _add_transformations_to_file(self, transform: Transformation,
                                     parent_group: h5py.Group) -> str:
        transform_chain = [transform]
        while transform.depends_on is not None and not isinstance(
                transform.depends_on, str):
            transform_chain.append(transform.depends_on)
            transform = transform.depends_on

        transforms_group = self._create_nx_class("transformations",
                                                 "NXtransformations",
                                                 parent_group)
        transform_chain.reverse()
        depends_on_str = transform.depends_on if isinstance(
            transform.depends_on, str) else None
        for transform_number, transform in enumerate(transform_chain):
            if transform.time is not None:
                depends_on_str = self._add_transformation_as_log(
                    transform, transform_number, transforms_group,
                    depends_on_str)
            else:
                depends_on_str = self._add_transformation_as_dataset(
                    transform, transform_number, transforms_group,
                    depends_on_str)
        return depends_on_str

    def _add_transformation_as_dataset(self, transform: Transformation,
                                       transform_number: int,
                                       transforms_group: h5py.Group,
                                       depends_on: Optional[str]) -> str:
        added_transform = self._writer.add_dataset(
            transforms_group,
            f"transform_{transform_number}",
            data=transform.value)
        self._add_transform_attributes(added_transform, depends_on, transform)
        if transform.value_units is not None:
            self._writer.add_attribute(added_transform, "units",
                                       transform.value_units)
        return added_transform.name

    def _add_log_group_to_file(self, log: Log,
                               parent_group: h5py.Group) -> h5py.Group:
        log_group = self._create_nx_class(log.name, "NXlog", parent_group)
        if log.value is not None:
            value_ds = self._writer.add_dataset(log_group, "value", log.value)
            if log.value_units is not None:
                self._writer.add_attribute(value_ds, "units", log.value_units)
        if log.time is not None:
            time_ds = self._writer.add_dataset(log_group,
                                               "time",
                                               data=log.time)
            if log.time_units is not None:
                self._writer.add_attribute(time_ds, "units", log.time_units)
        return log_group

    def _add_transformation_as_log(self, transform: Transformation,
                                   transform_number: int,
                                   transforms_group: h5py.Group,
                                   depends_on: Optional[str]) -> str:
        added_transform = self._add_log_group_to_file(
            Log(f"transform_{transform_number}", transform.value,
                transform.time, transform.value_units, transform.time_units),
            transforms_group)
        self._add_transform_attributes(added_transform, depends_on, transform)
        return added_transform.name

    def _add_detector_group_to_file(self, detector: Detector,
                                    parent_group: h5py.Group,
                                    group_name: str) -> h5py.Group:
        detector_group = self._create_nx_class(group_name, "NXdetector",
                                               parent_group)
        self._writer.add_dataset(detector_group, "detector_number",
                                 detector.detector_numbers)
        for dataset_name, array in (("x_pixel_offset", detector.x_offsets),
                                    ("y_pixel_offset", detector.y_offsets),
                                    ("z_pixel_offset", detector.z_offsets)):
            if array is not None:
                offsets_ds = self._writer.add_dataset(detector_group,
                                                      dataset_name, array)
                if detector.offsets_unit is not None:
                    self._writer.add_attribute(offsets_ds, "units",
                                               detector.offsets_unit)
        return detector_group

    def _add_transform_attributes(self, added_transform: Union[h5py.Group,
                                                               h5py.Dataset],
                                  depends_on: Optional[str],
                                  transform: Transformation):
        self._writer.add_attribute(added_transform, "vector", transform.vector)
        self._writer.add_attribute(added_transform, "transformation_type",
                                   transform.transform_type.value)
        if transform.offset is not None:
            self._writer.add_attribute(added_transform, "offset",
                                       transform.offset)
        if depends_on is not None:
            self._writer.add_attribute(added_transform, "depends_on",
                                       depends_on)
        else:
            self._writer.add_attribute(added_transform, "depends_on",
                                       ".")  # means end of chain

    def _create_nx_class(self, group_name: str, nx_class_name: str,
                         parent: h5root) -> h5py.Group:
        nx_class = self._writer.add_group(parent, group_name)
        self._writer.add_attribute(nx_class, "NX_class", nx_class_name)
        return nx_class
