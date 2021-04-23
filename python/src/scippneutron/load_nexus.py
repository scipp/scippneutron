# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
import json

import scipp as sc
from ._loading_common import Group, MissingDataset
from ._loading_detector_data import load_detector_data
from ._loading_log_data import load_logs
from ._loading_hdf5_nexus import LoadFromHdf5
from ._loading_json_nexus import LoadFromJson
import h5py
from timeit import default_timer as timer
from typing import Union, List, Optional, Dict
from contextlib import contextmanager
from warnings import warn
import numpy as np
from ._loading_positions import (load_position_of_unique_component,
                                 load_positions_of_components)

nx_event_data = "NXevent_data"
nx_log = "NXlog"
nx_entry = "NXentry"
nx_instrument = "NXinstrument"
nx_sample = "NXsample"
nx_source = "NXsource"


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


def _add_string_attr_to_loaded_data(group: Union[h5py.Group,
                                                 Dict], dataset_name: str,
                                    attr_name: str, data: sc.Variable,
                                    nexus: Union[LoadFromHdf5, LoadFromJson]):
    try:
        data = data.attrs
    except AttributeError:
        pass

    try:
        data[attr_name] = sc.Variable(
            value=nexus.load_scalar_string(group, dataset_name))
    except MissingDataset:
        pass


def _load_instrument_name(instrument_groups: List[Group], data: sc.Variable,
                          nexus: Union[LoadFromHdf5, LoadFromJson]):
    if len(instrument_groups) > 1:
        warn(f"More than one {nx_instrument} found in file, "
             f"loading name from {instrument_groups[0].group.name} only")
    _add_string_attr_to_loaded_data(instrument_groups[0].group, "name",
                                    "instrument_name", data, nexus)


def _load_sample(sample_groups: List[Group], data: sc.Variable,
                 file_root: h5py.File, nexus: Union[LoadFromHdf5,
                                                    LoadFromJson]):
    load_positions_of_components(sample_groups,
                                 data,
                                 "sample",
                                 nx_sample,
                                 file_root,
                                 nexus,
                                 default_position=np.array([0, 0, 0]))


def _load_source(source_groups: List[Group], data: sc.Variable,
                 file_root: h5py.File, nexus: Union[LoadFromHdf5,
                                                    LoadFromJson]):
    load_position_of_unique_component(source_groups, data, "source", nx_source,
                                      file_root, nexus)


def _load_title(entry_group: Group, data: sc.Variable,
                nexus: Union[LoadFromHdf5, LoadFromJson]):
    _add_string_attr_to_loaded_data(entry_group.group, "title",
                                    "experiment_title", data, nexus)


def load_nexus(data_file: Union[str, h5py.File],
               root: str = "/",
               quiet=True) -> Optional[sc.Variable]:
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
               nexus: Union[LoadFromHdf5, LoadFromJson], quiet: bool):
    if root is not None:
        root_node = nexus_file[root]
    else:
        root_node = nexus_file
    groups = nexus.find_by_nx_class(
        (nx_event_data, nx_log, nx_entry, nx_instrument, nx_sample, nx_source),
        root_node)
    if len(groups[nx_entry]) > 1:
        # We can't sensibly load from multiple NXentry, for example each
        # could could contain a description of the same detector bank
        # and lead to problems with clashing detector ids etc
        raise RuntimeError(
            f"More than one {nx_entry} group in file, use 'root' argument "
            "to specify which to load data from, for example"
            f"{__name__}('my_file.nxs', '/entry_2')")
    loaded_data = load_detector_data(groups[nx_event_data], nexus_file, nexus,
                                     quiet)
    if loaded_data is None:
        no_event_data = True
        loaded_data = sc.Dataset({})
    else:
        no_event_data = False
    load_logs(loaded_data, groups[nx_log], nexus)
    if groups[nx_sample]:
        _load_sample(groups[nx_sample], loaded_data, nexus_file, nexus)
    if groups[nx_source]:
        _load_source(groups[nx_source], loaded_data, nexus_file, nexus)
    if groups[nx_instrument]:
        _load_instrument_name(groups[nx_instrument], loaded_data, nexus)
    if groups[nx_entry]:
        _load_title(groups[nx_entry][0], loaded_data, nexus)
    # Return None if we have an empty dataset at this point
    if no_event_data and not loaded_data.keys():
        loaded_data = None
    return loaded_data


def _load_nexus_json(json_template: str) -> Optional[sc.Variable]:
    """
    Use this function for testing so that file io is not required
    """
    # We do not use cls to convert value lists to sc.Variable at this
    # point because we do not know what dimension names to use here
    loaded_json = json.loads(json_template)
    return _load_data(loaded_json, None, LoadFromJson(loaded_json), True)


def load_nexus_json(json_filename: str) -> Optional[sc.Variable]:
    with open(json_filename, 'r') as json_file:
        json_string = json_file.read()
    return _load_nexus_json(json_string)
