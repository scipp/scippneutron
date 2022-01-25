# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

import numpy as np
from typing import Tuple, List, Dict
import scipp as sc
from ._common import (BadSource, SkipSource, MissingDataset, Group, load_time_dataset,
                      to_plain_index)
from ._nexus import LoadFromNexus
from .nxobject import NXobject
from warnings import warn


def load_logs(log_groups: List[Group], nexus: LoadFromNexus) -> Dict:
    logs = {}
    for group in log_groups:
        try:
            log_data_name, log_data = _load_log_data_from_group(group, nexus)
            _add_log_to_data(log_data_name, log_data, group.name, logs)
        except BadSource as e:
            warn(f"Skipped loading {group.name} due to:\n{e}")
        except SkipSource:
            pass  # skip without warning user
    return logs


def _add_log_to_data(log_data_name: str, log_data: sc.Variable, group_path: str,
                     data: Dict):
    """
    Add an attribute with a unique name.
    If an attribute name already exists, we iteratively walk up the file tree
    and prepend the parent name to the attribute name, until a unique name is
    found.
    """
    group_path = group_path.split('/')
    path_position = -2
    name_changed = False
    unique_name_found = False
    while not unique_name_found:
        if log_data_name not in data.keys():
            data[log_data_name] = log_data
            unique_name_found = True
        else:
            name_changed = True
            log_data_name = f"{group_path[path_position]}_{log_data_name}"
            path_position -= 1
    if name_changed:
        warn(f"Name of log group at {'/'.join(group_path)} is not unique: "
             f"{log_data_name} used as attribute name.")


class NXlog(NXobject):
    @property
    def shape(self):
        pass

    @property
    def dims(self):
        pass

    @property
    def unit(self):
        pass

    def _getitem(self, index):
        name, var = _load_log_data_from_group(self._group, self._loader, select=index)
        da = var.value
        da.name = name
        return da


def _load_log_data_from_group(group: Group, nexus: LoadFromNexus, select=tuple())\
        -> Tuple[str, sc.Variable]:
    property_name = nexus.get_name(group)
    value_dataset_name = "value"
    time_dataset_name = "time"
    # TODO This is wrong if the log just has a single value. Can we check
    # the shape in advance?
    index = to_plain_index(["time"], select)

    try:
        values = nexus.load_dataset_from_group_as_numpy_array(group,
                                                              value_dataset_name,
                                                              index=index)
    except MissingDataset:
        if nexus.contains_stream(group):
            raise SkipSource("Log is missing value dataset but contains stream")
        raise BadSource(f"NXlog '{property_name}' has no value dataset")

    if values.size == 0:
        raise BadSource(f"NXlog '{property_name}' has an empty value dataset")

    unit = nexus.get_unit(nexus.get_dataset_from_group(group, value_dataset_name))
    try:
        unit = sc.Unit(unit)
    except sc.UnitError:
        warn(f"Unrecognized unit '{unit}' for value dataset "
             f"in NXlog '{group.name}'; setting unit as 'dimensionless'")
        unit = sc.units.dimensionless

    try:
        is_time_series = True
        dimension_label = "time"
        times = load_time_dataset(nexus,
                                  group,
                                  time_dataset_name,
                                  dim=dimension_label,
                                  index=index)

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
        property_data = sc.scalar(values,
                                  unit=unit,
                                  dtype=nexus.get_dataset_numpy_dtype(
                                      nexus.get_dataset_from_group(
                                          group, value_dataset_name)))
    else:
        property_data = sc.Variable(values=values,
                                    unit=unit,
                                    dims=[dimension_label],
                                    dtype=nexus.get_dataset_numpy_dtype(
                                        nexus.get_dataset_from_group(
                                            group, value_dataset_name)))

    if is_time_series:
        # If property has timestamps, create a DataArray
        data_array = {"data": property_data, "coords": {dimension_label: times}}
        return property_name, sc.scalar(sc.DataArray(**data_array))
    elif not np.isscalar(values):
        # If property is multi-valued, create a wrapper single
        # value variable. This prevents interference with
        # global dimensions for the output Dataset.
        return property_name, sc.scalar(property_data)
    return property_name, property_data
