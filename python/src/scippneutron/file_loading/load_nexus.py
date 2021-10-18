# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
import json

import scipp as sc

import scipp
import typing

from ._common import Group, MissingDataset
from ._detector_data import load_detector_data
from ._monitor_data import load_monitor_data
from ._log_data import load_logs
from ._hdf5_nexus import LoadFromHdf5
from ._json_nexus import LoadFromJson, get_streams_info, StreamInfo
from ._nexus import ScippData, NexusMeta
import h5py
from timeit import default_timer as timer
from typing import Union, List, Optional, Dict, Tuple, Set
from contextlib import contextmanager
from warnings import warn
import numpy as np
from ._positions import (load_position_of_unique_component,
                         load_positions_of_components)
from ._sample import load_ub_matrices_of_components

nx_event_data = "NXevent_data"
nx_log = "NXlog"
nx_entry = "NXentry"
nx_instrument = "NXinstrument"
nx_sample = "NXsample"
nx_source = "NXsource"
nx_detector = "NXdetector"
nx_disk_chopper = "NXdisk_chopper"
nx_monitor = "NXmonitor"


@contextmanager
def _open_if_path(file_in: Union[str, h5py.File]):
    """
    Open if file path is provided,
    otherwise yield the existing h5py.File object
    """
    if isinstance(file_in, str):
        with h5py.File(file_in, "r", libver='latest', swmr=True) as nexus_file:
            yield nexus_file
    else:
        yield file_in


def _load_instrument_name(instrument_groups: List[Group],
                          nexus_meta: NexusMeta) -> Dict:
    try:
        if len(instrument_groups) > 1:
            warn(f"More than one {nx_instrument} found in file, "
                 f"loading name from {instrument_groups[0].group.name} only")
        return {
            "instrument_name":
            sc.scalar(value=nexus_meta.nexus.load_scalar_string(
                instrument_groups[0].group, "name"))
        }
    except MissingDataset:
        return {}


def _load_chopper(chopper_groups: List[Group], nexus_meta: NexusMeta) -> Dict:
    choppers = {}
    for chopper_group in chopper_groups:
        chopper_name = chopper_group.path.split("/")[-1]
        rotation_speed = nexus_meta.nexus.load_dataset(group=chopper_group.group,
                                                       dataset_name="rotation_speed")
        distance = nexus_meta.nexus.load_dataset(group=chopper_group.group,
                                                 dataset_name="distance")
        choppers[chopper_name] = sc.DataArray(data=sc.scalar(value=chopper_name),
                                              attrs={
                                                  "rotation_speed": rotation_speed,
                                                  "distance": distance
                                              })

    return choppers


def _load_sample(sample_groups: List[Group], nexus_meta: NexusMeta):
    positions = load_positions_of_components(sample_groups,
                                             "sample",
                                             nx_sample,
                                             nexus_meta.nexus_file,
                                             nexus_meta.nexus,
                                             default_position=np.array([0, 0, 0]))

    ub_matrices = load_ub_matrices_of_components(sample_groups, "sample",
                                                 nexus_meta.nexus)

    return positions | ub_matrices


def _load_source(source_groups: List[Group], nexus_meta: NexusMeta):
    return load_position_of_unique_component(source_groups, "source", nx_source,
                                             nexus_meta.nexus_file, nexus_meta.nexus)


def _load_title(entry_group: List[Group], nexus_meta: NexusMeta) -> Dict:
    try:
        return {
            "experiment_title":
            sc.scalar(value=nexus_meta.nexus.load_scalar_string(
                entry_group[0].group, "title"))
        }
    except MissingDataset:
        return {}


def _load_start_and_end_time(entry_group: List[Group], nexus_meta: NexusMeta) -> Dict:
    times = {}
    for time in ["start_time", "end_time"]:
        try:
            times[time] = sc.scalar(
                value=nexus_meta.nexus.load_scalar_string(entry_group[0].group, time))
        except MissingDataset:
            pass
    return times


def load_nexus(data_file: Union[str, h5py.File],
               root: str = "/",
               quiet=True,
               bin_by_pixel: bool = True) -> Optional[ScippData]:
    """
    Load a NeXus file and return required information.

    :param data_file: path of NeXus file containing data to load
    :param root: path of group in file, only load data from the subtree of
      this group
    :param quiet: if False prints some details of what is being loaded
    :param bin_by_pixel: if True, bins the loaded detector data by pixel. If False, bins
      by pulse. Defaults to True.

    Usage example:
      data = scn.load_nexus('PG3_4844_event.nxs')
    """
    start_time = timer()

    with _open_if_path(data_file) as nexus_file:
        loaded_data = _load_data(NexusMeta(nexus_file=nexus_file,
                                           root=root,
                                           nexus=LoadFromHdf5()),
                                 quiet,
                                 bin_by_pixel=bin_by_pixel)

    if not quiet:
        print("Total time:", timer() - start_time)
    return loaded_data


def _load_data(nexus_meta: NexusMeta, quiet: bool, bin_by_pixel: bool) \
        -> Optional[ScippData]:
    """
    Main implementation for loading data is extracted to this function so that
    in-memory data can be used for unit tests.
    """
    if nexus_meta.root is not None:
        root_node = nexus_meta.nexus_file[nexus_meta.root]
    else:
        root_node = nexus_meta.nexus_file
    # Use visititems (in find_by_nx_class) to traverse the entire file tree,
    # looking for any NXClass that can be read.
    # groups is a dict with a key for each category (nx_log, nx_instrument...)
    groups = nexus_meta.nexus.find_by_nx_class(
        (nx_event_data, nx_log, nx_entry, nx_instrument, nx_sample, nx_source,
         nx_detector, nx_monitor, nx_disk_chopper), root_node)

    if len(groups[nx_entry]) > 1:
        # We can't sensibly load from multiple NXentry, for example each
        # could could contain a description of the same detector bank
        # and lead to problems with clashing detector ids etc
        raise RuntimeError(
            f"More than one {nx_entry} group in file, use 'root' argument "
            "to specify which to load data from, for example"
            f"{__name__}('my_file.nxs', '/entry_2')")

    loaded_data = load_detector_data(groups[nx_event_data], groups[nx_detector],
                                     nexus_meta, quiet, bin_by_pixel)
    # If no event data are found, make a Dataset and add the metadata as
    # Dataset entries. Otherwise, make a DataArray.
    if loaded_data is None:
        no_event_data = True
        loaded_data = sc.Dataset()
    else:
        no_event_data = False

    def add_metadata(metadata: Dict[str, sc.Variable]):
        for key, value in metadata.items():
            if isinstance(loaded_data, sc.DataArray):
                loaded_data.attrs[key] = value
            else:
                loaded_data[key] = value

    if groups[nx_entry]:
        add_metadata(_load_title(groups[nx_entry], nexus_meta))
        add_metadata(_load_start_and_end_time(groups[nx_entry], nexus_meta))
    if groups[nx_log]:
        add_metadata(load_logs(groups[nx_log], nexus_meta))
    if groups[nx_monitor]:
        add_metadata(load_monitor_data(groups[nx_monitor], nexus_meta))
    if groups[nx_sample]:
        add_metadata(_load_sample(groups[nx_sample], nexus_meta))
    if groups[nx_source]:
        add_metadata(_load_source(groups[nx_source], nexus_meta))
    if groups[nx_instrument]:
        add_metadata(_load_instrument_name(groups[nx_instrument], nexus_meta))
    if groups[nx_disk_chopper]:
        add_metadata(_load_chopper(groups[nx_disk_chopper], nexus_meta))

    # Return None if we have an empty dataset at this point
    if no_event_data and not loaded_data.keys():
        loaded_data = None
    return loaded_data


def _generate_wrapper(func: typing.Callable, nx_group_name: str, doc_summary: str) \
        -> typing.Callable:
    """
    Generates user-facing wrappers around each individual metadata-loading function.

    Args:
        func: the function to wrap. This function is expected to have the signature:
            func(nx_groups: List[Group], nexus_meta: NexusMeta) -> Dict[str, ScippData]
        nx_group_name: the class of nexus groups to pass to func
        doc_summary: a short summary of the purpose of this function (for docstring)
    """
    def _wrapper(data_file: Union[str, h5py.File],
                 root: str = "/",
                 quiet: bool = True) -> Dict[str, ScippData]:
        start_time = timer()

        with _open_if_path(data_file) as nexus_file:
            if root is not None:
                root_node = nexus_file[root]
            else:
                root_node = nexus_file

            nexus = LoadFromHdf5()
            nexus_meta = NexusMeta(nexus=nexus, root="/", nexus_file=nexus_file)
            nx_group = nexus.find_by_nx_class((nx_group_name, ), root_node)
            loaded_data = func(nx_group[nx_group_name], nexus_meta=nexus_meta)

        if not quiet:
            print("Total time:", timer() - start_time)
        return loaded_data

    _wrapper.__doc__ = f"""
        {doc_summary}.

        :param data_file: path of NeXus file containing data to load
        :param root: path of group in file, only load data from the subtree of
          this group
        :param quiet: if False prints some details of what is being loaded
        """

    return _wrapper


load_nexus_instrument_name = _generate_wrapper(
    _load_instrument_name,
    nx_instrument,
    doc_summary="Loads the instrument name from the provided nexus file")
load_nexus_disk_chopper = _generate_wrapper(
    _load_chopper,
    nx_disk_chopper,
    doc_summary="Loads disk chopper metadata from the provided nexus file")
load_nexus_monitors = _generate_wrapper(
    load_monitor_data,
    nx_monitor,
    doc_summary="Loads data from monitors in the provided nexus file")
load_nexus_sample = _generate_wrapper(
    _load_sample,
    nx_sample,
    doc_summary="Loads sample position metadata from the provided nexus file")
load_nexus_source = _generate_wrapper(
    _load_source,
    nx_source,
    doc_summary="Loads source position metadata from the provided nexus file")
load_nexus_logs = _generate_wrapper(
    load_logs, nx_log, doc_summary="Loads logs from the provided nexus file")
load_nexus_start_and_end_time = _generate_wrapper(
    _load_start_and_end_time,
    nx_entry,
    doc_summary="Loads the run start and end times from the provided nexus file")
load_nexus_title = _generate_wrapper(
    _load_title,
    nx_entry,
    doc_summary="Loads the run title from the provided nexus file")


def _load_nexus_json(
    json_template: str,
    get_start_info: bool = False,
    bin_by_pixel: bool = True,
) -> Tuple[Optional[ScippData], Optional[sc.Variable], Optional[Set[StreamInfo]]]:
    """
    Use this function for testing so that file io is not required
    """
    # We do not use cls to convert value lists to sc.Variable at this
    # point because we do not know what dimension names to use here
    loaded_json = json.loads(json_template)
    streams = None
    if get_start_info:
        streams = get_streams_info(loaded_json)
    return _load_data(NexusMeta(nexus_file=loaded_json,
                                root=None,
                                nexus=LoadFromJson(loaded_json)),
                      True,
                      bin_by_pixel=bin_by_pixel), streams


def load_nexus_json(json_filename: str,
                    bin_by_pixel: bool = True) -> Optional[ScippData]:
    with open(json_filename, 'r') as json_file:
        json_string = json_file.read()
    loaded_data, _ = _load_nexus_json(json_string, bin_by_pixel=bin_by_pixel)
    return loaded_data
