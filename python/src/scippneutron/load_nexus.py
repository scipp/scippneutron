# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

import scipp as sc
from ._loading_common import find_by_nx_class, ensure_str
from ._loading_detector_data import load_detector_data
from ._loading_log_data import load_logs
import h5py
from timeit import default_timer as timer
from typing import Union
from contextlib import contextmanager


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


def _add_string_data_as_attr(group: h5py.Group, dataset_name: str,
                             attr_name: str, data: sc.Variable):
    try:
        data = data.attrs
    except AttributeError:
        pass

    try:
        data[attr_name] = sc.Variable(
            value=ensure_str(group[dataset_name][...].item()))
    except KeyError:
        pass


def _load_instrument_name(instrument_group: h5py.Group, data: sc.Variable):
    _add_string_data_as_attr(instrument_group, "name", "instrument_name", data)


def _load_title(entry_group: h5py.Group, data: sc.Variable):
    _add_string_data_as_attr(entry_group, "title", "experiment_title", data)


def load_nexus(data_file: Union[str, h5py.File], root: str = "/", quiet=True):
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
        nx_event_data = "NXevent_data"
        nx_log = "NXlog"
        nx_entry = "NXentry"
        nx_instrument = "NXinstrument"
        nx_sample = "NXsample"
        groups = find_by_nx_class(
            (nx_event_data, nx_log, nx_entry, nx_instrument, nx_sample),
            nexus_file[root])

        if len(groups[nx_entry]) > 1:
            # We can't sensibly load from multiple NXentry, for example each
            # could could contain a description of the same detector bank
            # and lead to problems with clashing detector ids etc
            raise RuntimeError(
                "More than one NXentry group in file, use 'root' argument "
                "to specify which to load data from, for example"
                f"{__name__}('my_file.nxs', '/entry_2')")

        loaded_data = load_detector_data(groups[nx_event_data], quiet)
        if loaded_data is None:
            no_event_data = True
            loaded_data = sc.Dataset({})
        else:
            no_event_data = False

        load_logs(loaded_data, groups[nx_log])

        if groups[nx_instrument]:
            _load_instrument_name(groups[nx_instrument][0], loaded_data)

        if groups[nx_entry]:
            _load_title(groups[nx_entry][0], loaded_data)

    # Return None if we have an empty dataset at this point
    if no_event_data and not loaded_data.keys():
        loaded_data = None

    if not quiet:
        print("Total time:", timer() - total_time)
    return loaded_data
