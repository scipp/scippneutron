# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
import json

import scipp as sc
from ._common import Group, MissingDataset
from ._detector_data import load_detector_data
from ._log_data import load_logs
from ._hdf5_nexus import LoadFromHdf5
from ._json_nexus import LoadFromJson, get_streams_info, StreamInfo
from ._nexus import LoadFromNexus, GroupObject, ScippData
import h5py
from timeit import default_timer as timer
from typing import Union, List, Optional, Dict, Tuple, Set
from contextlib import contextmanager
from warnings import warn
import numpy as np
from ._positions import (load_position_of_unique_component,
                         load_positions_of_components)

nx_event_data = "NXevent_data"
nx_log = "NXlog"
nx_entry = "NXentry"
nx_instrument = "NXinstrument"
nx_sample = "NXsample"
nx_source = "NXsource"
nx_detector = "NXdetector"


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


def _add_string_attr_to_loaded_data(group: GroupObject, dataset_name: str,
                                    attr_name: str, data: ScippData,
                                    nexus: LoadFromNexus):
    try:
        data = data.attrs
    except AttributeError:
        pass

    try:
        data[attr_name] = sc.Variable(
            value=nexus.load_scalar_string(group, dataset_name))
    except MissingDataset:
        pass


def _load_instrument_name(instrument_groups: List[Group], data: ScippData,
                          nexus: LoadFromNexus):
    if len(instrument_groups) > 1:
        warn(f"More than one {nx_instrument} found in file, "
             f"loading name from {instrument_groups[0].group.name} only")
    _add_string_attr_to_loaded_data(instrument_groups[0].group, "name",
                                    "instrument_name", data, nexus)


def _load_sample(sample_groups: List[Group], data: ScippData,
                 file_root: h5py.File, nexus: LoadFromNexus):
    load_positions_of_components(sample_groups,
                                 data,
                                 "sample",
                                 nx_sample,
                                 file_root,
                                 nexus,
                                 default_position=np.array([0, 0, 0]))


def _load_source(source_groups: List[Group], data: ScippData,
                 file_root: h5py.File, nexus: LoadFromNexus):
    load_position_of_unique_component(source_groups, data, "source", nx_source,
                                      file_root, nexus)


def _load_title(entry_group: Group, data: ScippData, nexus: LoadFromNexus):
    _add_string_attr_to_loaded_data(entry_group.group, "title",
                                    "experiment_title", data, nexus)


def _load_start_and_end_time(entry_group: Group, data: ScippData,
                             nexus: LoadFromNexus):
    _add_string_attr_to_loaded_data(entry_group.group, "start_time",
                                    "start_time", data, nexus)
    _add_string_attr_to_loaded_data(entry_group.group, "end_time", "end_time",
                                    data, nexus)


def load_nexus(data_file: Union[str, h5py.File],
               root: str = "/",
               quiet=True) -> Optional[ScippData]:
    """
    Load a NeXus file and return required information.

    :param data_file: path of NeXus file containing data to load
    :param root: path of group in file, only load data from the subtree of
      this group
    :param quiet: if False prints some details of what is being loaded

    Usage example:
      data = sc.neutron.load_nexus('PG3_4844_event.nxs')
    """
    total_time = timer()

    with _open_if_path(data_file) as nexus_file:
        loaded_data = _load_data(nexus_file, root, LoadFromHdf5(), quiet)

    if not quiet:
        print("Total time:", timer() - total_time)
    return loaded_data


def _load_data(nexus_file: Union[h5py.File, Dict], root: Optional[str],
               nexus: LoadFromNexus, quiet: bool) -> Optional[ScippData]:
    """
    Main implementation for loading data is extracted to this function so that
    in-memory data can be used for unit tests.
    """
    if root is not None:
        root_node = nexus_file[root]
    else:
        root_node = nexus_file
    # Use visititems (in find_by_nx_class) to traverse the entire file tree,
    # looking for any NXClass that can be read.
    # groups is a dict with a key for each category (nx_log, nx_instrument...)
    groups = nexus.find_by_nx_class(
        (nx_event_data, nx_log, nx_entry, nx_instrument, nx_sample, nx_source,
         nx_detector), root_node)
    if len(groups[nx_entry]) > 1:
        # We can't sensibly load from multiple NXentry, for example each
        # could could contain a description of the same detector bank
        # and lead to problems with clashing detector ids etc
        raise RuntimeError(
            f"More than one {nx_entry} group in file, use 'root' argument "
            "to specify which to load data from, for example"
            f"{__name__}('my_file.nxs', '/entry_2')")
    loaded_data = load_detector_data(groups[nx_event_data],
                                     groups[nx_detector], nexus_file, nexus,
                                     quiet)
    # If no event data are found, make a Dataset and add the metadata as
    # Dataset entries. Otherwise, make a DataArray.
    if loaded_data is None:
        no_event_data = True
        loaded_data = sc.Dataset({})
    else:
        no_event_data = False

    if groups[nx_entry]:
        _load_title(groups[nx_entry][0], loaded_data, nexus)
        _load_start_and_end_time(groups[nx_entry][0], loaded_data, nexus)

        try:
            run_start_time = nexus.load_scalar_string(
                groups[nx_entry][0].group, "start_time")
        except (AttributeError, MissingDataset):
            run_start_time = None
    else:
        run_start_time = None

    load_logs(loaded_data,
              groups[nx_log],
              nexus,
              run_start_time=run_start_time)

    if groups[nx_sample]:
        _load_sample(groups[nx_sample], loaded_data, nexus_file, nexus)
    if groups[nx_source]:
        _load_source(groups[nx_source], loaded_data, nexus_file, nexus)
    if groups[nx_instrument]:
        _load_instrument_name(groups[nx_instrument], loaded_data, nexus)
    # Return None if we have an empty dataset at this point
    if no_event_data and not loaded_data.keys():
        loaded_data = None
    return loaded_data


def _load_nexus_json(
    json_template: str,
    get_start_info: bool = False
) -> Tuple[Optional[ScippData], Optional[sc.Variable],
           Optional[Set[StreamInfo]]]:
    """
    Use this function for testing so that file io is not required
    """
    # We do not use cls to convert value lists to sc.Variable at this
    # point because we do not know what dimension names to use here
    loaded_json = json.loads(json_template)
    streams = None
    if get_start_info:
        streams = get_streams_info(loaded_json)
    return _load_data(loaded_json, None, LoadFromJson(loaded_json),
                      True), streams


def load_nexus_json(json_filename: str) -> Optional[ScippData]:
    with open(json_filename, 'r') as json_file:
        json_string = json_file.read()
    loaded_data, _ = _load_nexus_json(json_string)
    return loaded_data
