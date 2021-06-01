# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

import numpy as np
from typing import Tuple, List
import scipp as sc
from ._common import (BadSource, MissingDataset, Group)
from ._nexus import LoadFromNexus, GroupObject, ScippData
from warnings import warn


def load_logs(loaded_data: ScippData, log_groups: List[Group],
              nexus: LoadFromNexus):
    for group in log_groups:
        try:
            log_data_name, log_data = _load_log_data_from_group(
                group.group, nexus)
            _add_log_to_data(log_data_name, log_data, group.path, loaded_data)
        except BadSource as e:
            warn(f"Skipped loading {group.path} due to:\n{e}")


def _add_log_to_data(log_data_name: str, log_data: sc.Variable,
                     group_path: str, data: ScippData):
    try:
        data = data.attrs
    except AttributeError:
        pass

    group_path = group_path.split('/')
    path_position = -2
    name_changed = False
    unique_name_found = False
    while not unique_name_found:
        if log_data_name not in data.keys():
            data[log_data_name] = sc.detail.move(log_data)
            unique_name_found = True
        else:
            name_changed = True
            log_data_name = f"{group_path[path_position]}_{log_data_name}"
            path_position -= 1
    if name_changed:
        warn(f"Name of log group at {'/'.join(group_path)} is not unique: "
             f"{log_data_name} used as attribute name.")


def _load_log_data_from_group(group: GroupObject,
                              nexus: LoadFromNexus) -> Tuple[str, sc.Variable]:
    property_name = nexus.get_name(group)
    value_dataset_name = "value"
    time_dataset_name = "time"

    try:
        values = nexus.load_dataset_from_group_as_numpy_array(
            group, value_dataset_name)
    except MissingDataset:
        raise BadSource(f"NXlog '{property_name}' has no value dataset")

    if values.size == 0:
        raise BadSource(f"NXlog '{property_name}' has an empty value dataset")

    unit = nexus.get_unit(
        nexus.get_dataset_from_group(group, value_dataset_name))

    try:
        dimension_label = "time"
        is_time_series = True
        times = nexus.load_dataset(group, time_dataset_name, [dimension_label])
        if tuple(times.shape) != values.shape:
            raise BadSource(f"NXlog '{property_name}' has time and value "
                            f"datasets of different shapes")
    except MissingDataset:
        dimension_label = property_name
        is_time_series = False

    if np.ndim(values) > 1:
        raise BadSource(f"NXlog '{property_name}' has {value_dataset_name} "
                        f"dataset with more than 1 dimension, handling "
                        f"this is not yet implemented")

    if np.ndim(values) == 0:
        property_data = sc.Variable(value=values,
                                    unit=unit,
                                    dtype=nexus.get_dataset_numpy_dtype(
                                        group, value_dataset_name))
    else:
        property_data = sc.Variable(values=values,
                                    unit=unit,
                                    dims=[dimension_label],
                                    dtype=nexus.get_dataset_numpy_dtype(
                                        group, value_dataset_name))

    if is_time_series:
        # If property has timestamps, create a DataArray
        data_array = {
            "data": property_data,
            "coords": {
                dimension_label: times
            }
        }
        return property_name, sc.Variable(value=sc.detail.move_to_data_array(
            **data_array))
    elif not np.isscalar(values):
        # If property is multi-valued, create a wrapper single
        # value variable. This prevents interference with
        # global dimensions for the output Dataset.
        return property_name, sc.Variable(value=property_data)
    return property_name, property_data
