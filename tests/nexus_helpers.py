# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

import scipp as sc
from dataclasses import dataclass
from typing import List, Union, Iterator, Optional, Dict, Any, Tuple
import h5py
import numpy as np
from enum import Enum
from contextlib import contextmanager
import json
from scippneutron.file_loading._json_nexus import (JSONGroup, make_json_attr,
                                                   make_json_dataset)

h5root = Union[h5py.File, h5py.Group]


def _create_nx_class(group_name: str, nx_class_name: str, parent: h5root) -> h5py.Group:
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
    event_id: Optional[np.ndarray]
    event_time_offset: Optional[np.ndarray]
    event_time_zero: Optional[np.ndarray]
    event_index: Optional[np.ndarray]
    event_time_zero_unit: Optional[Union[str, bytes]] = "ns"
    event_time_zero_offset: Optional[Union[str, bytes]] = "1970-01-01T00:00:00Z"
    event_time_offset_unit: Optional[Union[str, bytes]] = "ns"


@dataclass
class Log:
    name: str
    value: Optional[np.ndarray]
    time: Optional[np.ndarray] = None
    value_units: Optional[Union[str, bytes]] = None

    # From
    # https://manual.nexusformat.org/classes/base_classes/NXlog.html?highlight=nxlog
    # time units are non-optional if time series data is present, and the unit
    # must be a unit of time (i.e. convertible to seconds).
    time_units: Optional[Union[str, bytes]] = "s"

    start_time: Optional[Union[str, bytes]] = "1970-01-01T00:00:00Z"
    scaling_factor: Optional[float] = None


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
    offset_unit: Optional[str] = None
    value_units: Optional[Union[str, bytes]] = None
    time_units: Optional[Union[str, bytes]] = None


@dataclass
class Detector:
    detector_numbers: Optional[np.ndarray] = None
    event_data: Optional[EventData] = None
    log: Optional[Log] = None
    x_offsets: Optional[np.ndarray] = None
    y_offsets: Optional[np.ndarray] = None
    z_offsets: Optional[np.ndarray] = None
    offsets_unit: Optional[Union[str, bytes]] = None
    depends_on: Optional[Transformation] = None
    data: Optional[np.ndarray] = None


@dataclass
class Data:
    name: str
    data: sc.DataArray
    attrs: dict = None


@dataclass
class Sample:
    name: str
    depends_on: Optional[Transformation] = None
    distance: Optional[float] = None
    distance_units: Optional[Union[str, bytes]] = None
    ub_matrix: Optional[np.ndarray] = None
    orientation_matrix: Optional[np.ndarray] = None


@dataclass
class Source:
    name: str
    depends_on: Union[Transformation, None, str] = None
    distance: Optional[float] = None
    distance_units: Optional[Union[str, bytes]] = None


@dataclass
class Chopper:
    name: str
    distance: float
    rotation_speed: float
    distance_units: Optional[str] = None
    rotation_units: Optional[str] = None


@dataclass
class Link:
    new_path: str
    target_path: str


@dataclass
class DatasetAtPath:
    path: str
    data: np.ndarray
    attributes: Dict[str, Any]


@dataclass
class Stream:
    """
    Only present in the JSON NeXus file templates, not in HDF5 NeXus files.
    Records where to find data in Kafka that are streamed during an experiment.
    """
    # Where the builder should place the stream object
    path: str

    # The following members correspond to fields in stream object.
    # Some of them may not be of interest to Scipp but are to other
    # software which consume the json template, for example
    # the Filewriter (https://github.com/ess-dmsc/kafka-to-nexus)

    # Kafka topic (named data stream)
    topic: str = "motion_devices_topic"
    # Source name, allows filtering and multiplexing to different
    # writer_modules by the filewriter software
    source: str = "linear_axis"
    # Tells filewriter which plugin to use to deserialise
    # messages in this stream and how to write the data to file.
    # For example the "f142" writer module deserialises messages which
    # were serialised with the "f142" flatbuffer schema
    # (https://github.com/ess-dmsc/streaming-data-types/) and
    # writes resulting timeseries data to file as an NXlog
    # (https://manual.nexusformat.org/classes/base_classes/NXlog.html)
    writer_module: str = "f142"
    # Deserialised values are expected to be of this type
    type: str = "double"
    # Values have these units
    value_units: str = "m"


@dataclass
class Monitor:
    name: str
    data: np.ndarray
    axes: List[Tuple[str, np.ndarray]]
    events: Optional[EventData] = None
    depends_on: Optional[Transformation] = None


class InMemoryNeXusWriter:

    def add_dataset_at_path(self, file_root: h5py.File, path: str, data: np.ndarray,
                            attributes: Dict):
        path_split = path.split("/")
        dataset_name = path_split[-1]
        parent_path = "/".join(path_split[:-1])
        dataset = self.add_dataset(file_root[parent_path], dataset_name, data)
        for name, value in attributes.items():
            self.add_attribute(dataset, name, value)

    @staticmethod
    def add_dataset(parent: h5py.Group, name: str,
                    data: Union[str, bytes, np.ndarray]) -> h5py.Dataset:
        return parent.create_dataset(name, data=data)

    @staticmethod
    def add_attribute(parent: Union[h5py.Group, h5py.Dataset], name: str,
                      value: Union[str, bytes, np.ndarray]):
        parent.attrs[name] = value

    @staticmethod
    def add_group(parent: h5py.Group, name: str) -> h5py.Group:
        return parent.create_group(name)

    @staticmethod
    def add_hard_link(file_root: h5py.File, new_path: str, target_path: str):
        try:
            _ = file_root[new_path]
            del file_root[new_path]
        except KeyError:
            pass
        file_root[new_path] = file_root[target_path]

    @staticmethod
    def add_soft_link(file_root: h5py.File, new_path: str, target_path: str):
        try:
            _ = file_root[new_path]
            del file_root[new_path]
        except KeyError:
            pass
        file_root[new_path] = h5py.SoftLink(target_path)


def _get_child(obj, name):
    children = obj["children"]
    for child in children:
        if child.get('name') == name:
            return child
    return None


def _get_object_by_path(file_root, path):
    if not path.startswith('/'):
        return _get_child(file_root, path)
    path = path.split('/')[1:]  # Trim leading slash
    obj = file_root
    for name in path:
        obj = _get_child(obj, name)
    return obj


def _add_link_to_json(file_root: Dict, new_path: str, target_path: str):
    new_path_split = new_path.split("/")
    link_name = new_path_split[-1]
    parent_path = "/".join(new_path_split[:-1])
    parent_group = _get_object_by_path(file_root, parent_path)
    link = {"type": "link", "name": link_name, "target": target_path}
    existing_object = _get_object_by_path(parent_group, link_name)
    if existing_object is not None:
        parent_group["children"].remove(existing_object)
    parent_group["children"].append(link)


def _parent_and_name_from_path(file_root: Dict, path: str) -> Tuple[Dict, str]:
    path_split = path.split("/")
    name = path_split[-1]
    parent_path = '/'.join(path_split[:-1])
    parent_group = _get_object_by_path(file_root, parent_path)
    return parent_group, name


class JsonWriter:

    def add_dataset_at_path(self, file_root: Dict, path: str, data: np.ndarray,
                            attributes: Dict):
        parent_group, dataset_name = _parent_and_name_from_path(file_root, path)
        dataset = self.add_dataset(parent_group, dataset_name, data)
        for name, value in attributes.items():
            self.add_attribute(dataset, name, value)

    @staticmethod
    def add_dataset(parent: Dict, name: str, data: Union[str, bytes,
                                                         np.ndarray]) -> Dict:
        dataset = make_json_dataset(name, data)
        parent["children"].append(dataset)
        return dataset

    @staticmethod
    def add_attribute(parent: Dict, name: str, value: Union[str, bytes, list,
                                                            np.ndarray]):
        attr = make_json_attr(name, value)
        parent["attributes"].append(attr)

    @staticmethod
    def add_group(parent: Dict, name: str) -> Dict:
        new_group = {"type": "group", "name": name, "children": [], "attributes": []}
        parent["children"].append(new_group)
        return new_group

    @staticmethod
    def add_hard_link(file_root: Dict, new_path: str, target_path: str):
        _add_link_to_json(file_root, new_path, target_path)

    @staticmethod
    def add_soft_link(file_root: Dict, new_path: str, target_path: str):
        _add_link_to_json(file_root, new_path, target_path)

    def add_stream(self, file_root: Dict, stream: Stream):
        new_stream = {
            "type": "stream",
            "stream": {
                "topic": stream.topic,
                "source": stream.source,
                "writer_module": stream.writer_module,
                "type": stream.type,
                "value_units": stream.value_units
            }
        }
        if (group := _get_object_by_path(file_root, stream.path)) is None:
            parent, name = _parent_and_name_from_path(file_root, stream.path)
            group = self.add_group(parent, name)
        group["children"].append(new_stream)


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NexusBuilder:
    """
    Allows building an in-memory NeXus file for use in tests
    """

    def __init__(self):
        self._event_data: List[EventData] = []
        self._detectors: List[Detector] = []
        self._logs: List[Log] = []
        self._instrument_name: Optional[str] = None
        self._choppers: List[Chopper] = []
        self._title: Optional[str] = None
        self._start_time: Optional[str] = None
        self._end_time: Optional[str] = None
        self._sample: List[Sample] = []
        self._source: List[Source] = []
        self._hard_links: List[Link] = []
        self._soft_links: List[Link] = []
        self._writer = None
        self._datasets: List[DatasetAtPath] = []
        self._streams = []
        self._monitors = []
        self._datas = []

    def add_dataset_at_path(self, path: str, data: np.ndarray, attributes: Dict):
        self._datasets.append(DatasetAtPath(path, data, attributes))

    def _write_datasets(self, root: Union[Dict, h5py.File]):
        for dataset in self._datasets:
            self._writer.add_dataset_at_path(root, dataset.path, dataset.data,
                                             dataset.attributes)

    def add_stream(self, stream: Stream):
        self._streams.append(stream)

    def add_detector(self, detector: Detector):
        self._detectors.append(detector)

    def add_data(self, data: Data):
        self._datas.append(data)

    def add_event_data(self, event_data: EventData):
        self._event_data.append(event_data)

    def add_log(self, log: Log):
        self._logs.append(log)

    def add_instrument(self, name: str):
        self._instrument_name = name

    def add_chopper(self, chopper: Chopper):
        self._choppers.append(chopper)

    def add_title(self, title: str):
        self._title = title

    def add_run_start_time(self, start_time: str):
        self._start_time = start_time

    def add_run_end_time(self, end_time: str):
        self._end_time = end_time

    def add_sample(self, sample: Sample):
        self._sample.append(sample)

    def add_source(self, source: Source):
        self._source.append(source)

    def add_hard_link(self, link: Link):
        """
        If there is a group or dataset at the link path it will
        be replaced by the link
        """
        self._hard_links.append(link)

    def add_soft_link(self, link: Link):
        """
        If there is a group or dataset at the link path it will
        be replaced by the link
        """
        self._soft_links.append(link)

    def add_component(self, component: Union[Sample, Source]):
        # This is a little ugly, but allows parametrisation
        # of tests which should work for sample and source
        if isinstance(component, Sample):
            self.add_sample(component)
        elif isinstance(component, Source):
            self.add_source(component)

    def add_monitor(self, monitor: Monitor):
        self._monitors.append(monitor)

    @property
    def json_string(self):
        self._writer = JsonWriter()
        root = {"children": []}
        self._write_file(root)
        return json.dumps(root, indent=4, cls=NumpyEncoder)

    def create_json_file(self):
        """
        Create a file on disk, do not use this in tests, it is intended to
        be used as a tool during test development
        """
        self._writer = JsonWriter()
        root = {"children": []}
        self._write_file(root)
        with open("test_json.txt", "w") as json_file:
            return json.dump(root, json_file, indent=4, cls=NumpyEncoder)

    @contextmanager
    def json(self):
        try:
            yield JSONGroup(json.loads(self.json_string))
        finally:
            pass

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

    def _write_file(self, nexus_file: Union[h5py.File, Dict]):
        entry_group = self._create_nx_class("entry", "NXentry", nexus_file)
        if self._title is not None:
            self._writer.add_dataset(entry_group, "title", data=self._title)
        if self._start_time is not None:
            self._writer.add_dataset(entry_group, "start_time", data=self._start_time)
        if self._end_time is not None:
            self._writer.add_dataset(entry_group, "end_time", data=self._end_time)
        self._write_event_data(entry_group)
        self._write_logs(entry_group)
        self._write_sample(entry_group)
        self._write_source(entry_group)
        if self._instrument_name is None:
            parent_group = entry_group
            parent_path = "/entry"
        else:
            parent_group = self._write_instrument(entry_group)
            parent_path = "/entry/instrument"
        self._write_choppers(parent_group)
        self._write_detectors(parent_group, parent_path)
        self._write_datasets(nexus_file)
        self._write_streams(nexus_file)
        self._write_links(nexus_file)
        self._write_monitors(nexus_file)
        self._write_datas(parent_group)

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
            sample_group = self._create_nx_class(sample.name, "NXsample", parent_group)
            if sample.depends_on is not None:
                depends_on = self._add_transformations_to_file(
                    sample.depends_on, sample_group, f"/entry/{sample.name}")
                self._writer.add_dataset(sample_group, "depends_on", data=depends_on)
            if sample.distance is not None:
                distance_ds = self._writer.add_dataset(sample_group,
                                                       "distance",
                                                       data=sample.distance)
                if sample.distance_units is not None:
                    self._writer.add_attribute(distance_ds, "units",
                                               sample.distance_units)

            if sample.ub_matrix is not None:
                self._writer.add_dataset(sample_group,
                                         "ub_matrix",
                                         data=sample.ub_matrix)

            if sample.orientation_matrix is not None:
                self._writer.add_dataset(sample_group,
                                         "orientation_matrix",
                                         data=sample.orientation_matrix)

    def _write_source(self, parent_group: Union[h5py.Group, Dict]):
        for source in self._source:
            source_group = self._create_nx_class(source.name, "NXsource", parent_group)
            if source.depends_on is not None:
                if isinstance(source.depends_on, str):
                    depends_on = source.depends_on
                else:
                    depends_on = self._add_transformations_to_file(
                        source.depends_on, source_group, f"/entry/{source.name}")
                self._writer.add_dataset(source_group, "depends_on", data=depends_on)
            if source.distance is not None:
                distance_ds = self._writer.add_dataset(source_group,
                                                       "distance",
                                                       data=source.distance)
                if source.distance_units is not None:
                    self._writer.add_attribute(distance_ds, "units",
                                               source.distance_units)

    def _write_instrument(
            self, parent_group: Union[h5py.Group, Dict]) -> Union[h5py.Group, Dict]:
        instrument_group = self._create_nx_class("instrument", "NXinstrument",
                                                 parent_group)
        self._writer.add_dataset(instrument_group, "name", self._instrument_name)
        return instrument_group

    def _write_detectors(self, parent_group: Union[h5py.Group, Dict], parent_path: str):
        for detector_index, detector in enumerate(self._detectors):
            detector_name = f"detector_{detector_index}"
            detector_group = self._add_detector_group_to_file(
                detector, parent_group, detector_name)
            if detector.data is not None:
                da = detector.data
                ds = self._writer.add_dataset(detector_group, "data", data=da.values)
                self._writer.add_attribute(ds, "units", str(da.unit))
                axes = [dim if dim in da.coords else '.' for dim in da.dims]
                self._writer.add_attribute(detector_group, "axes", axes)
                for key, coord in da.coords.items():
                    ds = self._writer.add_dataset(detector_group,
                                                  key,
                                                  data=coord.values)
                    self._writer.add_attribute(ds, "units", str(coord.unit))

            if detector.event_data is not None:
                self._add_event_data_group_to_file(detector.event_data, detector_group,
                                                   "events")
            if detector.log is not None:
                self._add_log_group_to_file(detector.log, detector_group)
            if detector.depends_on is not None:
                depends_on = self._add_transformations_to_file(
                    detector.depends_on, detector_group,
                    f"{parent_path}/{detector_name}")
                self._writer.add_dataset(detector_group, "depends_on", data=depends_on)

    def _write_choppers(self, parent_group: Union[h5py.Group, Dict]):

        for chopper in self._choppers:
            chopper_group = self._create_nx_class(chopper.name, "NXdisk_chopper",
                                                  parent_group)
            distance_ds = self._writer.add_dataset(chopper_group,
                                                   "distance",
                                                   data=chopper.distance)
            rotation_ds = self._writer.add_dataset(chopper_group,
                                                   "rotation_speed",
                                                   data=chopper.rotation_speed)
            if chopper.distance_units is not None:
                self._writer.add_attribute(distance_ds, "units", chopper.distance_units)
            if chopper.rotation_units is not None:
                self._writer.add_attribute(rotation_ds, "units", chopper.rotation_units)

    def _write_event_data(self, parent_group: Union[h5py.Group, Dict]):
        for event_data_index, event_data in enumerate(self._event_data):
            self._add_event_data_group_to_file(event_data, parent_group,
                                               f"events_{event_data_index}")

    def _write_monitors(self, parent_group: Union[h5py.Group, Dict]):
        for monitor in self._monitors:
            self._add_monitor_group_to_file(monitor, parent_group)

    def _add_monitor_group_to_file(self, monitor: Monitor, parent_group: h5py.Group):
        monitor_group = self._create_nx_class(monitor.name, "NXmonitor", parent_group)
        data_group = self._writer.add_dataset(monitor_group, "data", monitor.data)
        self._writer.add_attribute(data_group, "units", '')
        self._writer.add_attribute(data_group, "axes",
                                   ",".join(name for name, _ in monitor.axes))

        if monitor.events:
            self._write_event_data_to_group(monitor_group, monitor.events)

        for axis_name, axis_data in monitor.axes:
            # We write event data (if exists) first - if we've already written event
            # data the event index will already have been created so we skip writing
            # it here.
            if not monitor.events or not axis_name == "event_index":
                ds = self._writer.add_dataset(monitor_group, axis_name, axis_data)
                self._writer.add_attribute(ds, "units", '')
        if monitor.depends_on is not None:
            if isinstance(monitor.depends_on, str):
                depends_on = monitor.depends_on
            else:
                depends_on = self._add_transformations_to_file(
                    monitor.depends_on, monitor_group, f"/{monitor.name}")
            self._writer.add_dataset(monitor_group, "depends_on", data=depends_on)

    def _write_datas(self, parent_group: Union[h5py.Group, Dict]):
        for data in self._datas:
            self._add_data_group_to_file(data, parent_group)

    def _add_data_group_to_file(self, data: Data, parent_group: h5py.Group):
        da = data.data
        group = self._create_nx_class(data.name, "NXdata", parent_group)
        self._writer.add_attribute(group, "axes", da.dims)
        self._writer.add_attribute(group, "signal", "signal1")
        signal = self._writer.add_dataset(group, "signal1", da.values)
        self._writer.add_attribute(signal, "units", str(da.unit))
        # Note: We are deliberately NOT adding AXISNAME_indices attributes for the
        # coords, since these were added late to the Nexus standard and therefore we
        # also need to support loading without the attributes. The attribute should be
        # set manually by the user if desired.
        for name, coord in da.coords.items():
            ds = self._writer.add_dataset(group, name, coord.values)
            self._writer.add_attribute(ds, "units", str(coord.unit))
        if data.attrs is not None:
            for k, v in data.attrs.items():
                self._writer.add_attribute(group, k, v)

    def _write_logs(self, parent_group: Union[h5py.Group, Dict]):
        for log in self._logs:
            self._add_log_group_to_file(log, parent_group)

    def _add_event_data_group_to_file(self, data: EventData, parent_group: h5py.Group,
                                      group_name: str):
        event_group = self._create_nx_class(group_name, "NXevent_data", parent_group)
        self._write_event_data_to_group(event_group, data)

    def _write_event_data_to_group(self, event_group: h5py.Group, data: EventData):
        if data.event_id is not None:
            self._writer.add_dataset(event_group, "event_id", data=data.event_id)
        if data.event_time_offset is not None:
            event_time_offset_ds = self._writer.add_dataset(event_group,
                                                            "event_time_offset",
                                                            data=data.event_time_offset)
            self._writer.add_attribute(event_time_offset_ds, "units",
                                       data.event_time_offset_unit)
        if data.event_time_zero is not None:
            event_time_zero_ds = self._writer.add_dataset(event_group,
                                                          "event_time_zero",
                                                          data=data.event_time_zero)
            self._writer.add_attribute(event_time_zero_ds, "units",
                                       data.event_time_zero_unit)
            self._writer.add_attribute(event_time_zero_ds, "offset",
                                       data.event_time_zero_offset)
        if data.event_index is not None:
            self._writer.add_dataset(event_group, "event_index", data=data.event_index)

    def _add_transformations_to_file(self, transform: Transformation,
                                     parent_group: h5py.Group, parent_path: str) -> str:
        transform_chain = [transform]
        while transform.depends_on is not None and not isinstance(
                transform.depends_on, str):
            transform_chain.append(transform.depends_on)
            transform = transform.depends_on

        transforms_group_name = "transformations"
        transforms_group = self._create_nx_class("transformations", "NXtransformations",
                                                 parent_group)
        transform_chain.reverse()
        depends_on_str = transform.depends_on if isinstance(transform.depends_on,
                                                            str) else None
        transform_group_path = f"{parent_path}/{transforms_group_name}"
        for transform_number, transform in enumerate(transform_chain):
            if transform.time is not None:
                depends_on_str = self._add_transformation_as_log(
                    transform, transform_number, transforms_group, transform_group_path,
                    depends_on_str)
            else:
                depends_on_str = self._add_transformation_as_dataset(
                    transform, transform_number, transforms_group, transform_group_path,
                    depends_on_str)
        return depends_on_str

    def _add_transformation_as_dataset(self, transform: Transformation,
                                       transform_number: int,
                                       transforms_group: h5py.Group, group_path: str,
                                       depends_on: Optional[str]) -> str:
        transform_name = f"transform_{transform_number}"
        added_transform = self._writer.add_dataset(transforms_group,
                                                   f"transform_{transform_number}",
                                                   data=transform.value)
        self._add_transform_attributes(added_transform, depends_on, transform)
        if transform.value_units is not None:
            self._writer.add_attribute(added_transform, "units", transform.value_units)
        return f"{group_path}/{transform_name}"

    def _add_log_group_to_file(self, log: Log, parent_group: h5py.Group) -> h5py.Group:
        log_group = self._create_nx_class(log.name, "NXlog", parent_group)
        if log.value is not None:
            value_ds = self._writer.add_dataset(log_group, "value", log.value)
            if log.value_units is not None:
                self._writer.add_attribute(value_ds, "units", log.value_units)
        if log.time is not None:
            time_ds = self._writer.add_dataset(log_group, "time", data=log.time)
            if log.time_units is not None:
                self._writer.add_attribute(time_ds, "units", log.time_units)
            if log.start_time is not None:
                self._writer.add_attribute(time_ds, "start", log.start_time)
            if log.scaling_factor is not None:
                self._writer.add_attribute(time_ds, "scaling_factor",
                                           log.scaling_factor)
        return log_group

    def _add_transformation_as_log(self, transform: Transformation,
                                   transform_number: int, transforms_group: h5py.Group,
                                   group_path: str, depends_on: Optional[str]) -> str:
        transform_name = f"transform_{transform_number}"
        added_transform = self._add_log_group_to_file(
            Log(transform_name, transform.value, transform.time, transform.value_units,
                transform.time_units), transforms_group)
        self._add_transform_attributes(added_transform, depends_on, transform)
        return f"{group_path}/{transform_name}"

    def _add_detector_group_to_file(self, detector: Detector, parent_group: h5py.Group,
                                    group_name: str) -> h5py.Group:
        detector_group = self._create_nx_class(group_name, "NXdetector", parent_group)
        if detector.detector_numbers is not None:
            self._writer.add_dataset(detector_group, "detector_number",
                                     detector.detector_numbers)
        for dataset_name, array in (("x_pixel_offset", detector.x_offsets),
                                    ("y_pixel_offset", detector.y_offsets),
                                    ("z_pixel_offset", detector.z_offsets)):
            if array is not None:
                offsets_ds = self._writer.add_dataset(detector_group, dataset_name,
                                                      array)
                if detector.offsets_unit is not None:
                    self._writer.add_attribute(offsets_ds, "units",
                                               detector.offsets_unit)
        return detector_group

    def _add_transform_attributes(self, added_transform: Union[h5py.Group,
                                                               h5py.Dataset],
                                  depends_on: Optional[str], transform: Transformation):
        self._writer.add_attribute(added_transform, "vector", transform.vector)
        self._writer.add_attribute(added_transform, "transformation_type",
                                   transform.transform_type.value)
        if transform.offset is not None:
            self._writer.add_attribute(added_transform, "offset", transform.offset)
        if transform.offset_unit is not None:
            self._writer.add_attribute(added_transform, "offset_units",
                                       transform.offset_unit)
        if depends_on is not None:
            self._writer.add_attribute(added_transform, "depends_on", depends_on)
        else:
            self._writer.add_attribute(added_transform, "depends_on",
                                       ".")  # means end of chain

    def _create_nx_class(self, group_name: str, nx_class_name: str,
                         parent: h5root) -> h5py.Group:
        nx_class = self._writer.add_group(parent, group_name)
        self._writer.add_attribute(nx_class, "NX_class", nx_class_name)
        return nx_class

    def _write_streams(self, root: Union[h5py.File, Dict]):
        if isinstance(self._writer, JsonWriter):
            for stream in self._streams:
                self._writer.add_stream(root, stream)
