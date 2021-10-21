# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

import numpy as np
from typing import Tuple, List, Union, Dict
import scipp as sc
from ._common import (BadSource, SkipSource, MissingDataset, MissingAttribute, Group)
from ._nexus import LoadFromNexus
from warnings import warn


def load_logs(log_groups: List[Group], nexus: LoadFromNexus) -> Dict:
    logs = {}
    for group in log_groups:
        try:
            log_data_name, log_data = _load_log_data_from_group(group, nexus)
            _add_log_to_data(log_data_name, log_data, group.path, logs)
        except BadSource as e:
            warn(f"Skipped loading {group.path} due to:\n{e}")
        except SkipSource:
            pass  # skip without warning user
    return logs


def _convert_nxlog_time_to_datetime64(
        raw_times: sc.Variable,
        group_path: str,
        log_start: str = None,
        scaling_factor: Union[float, np.float_] = None) -> sc.Variable:
    """
    The nexus standard allows an arbitrary scaling factor to be inserted
    between the numbers in the `time` series and the unit of time reported
    in the nexus attribute.

    The times are also relative to a given log start time, which might be
    different for each log. If this log start time is not available, the start of the
    unix epoch (1970-01-01T00:00:00Z) is used instead.

    See https://manual.nexusformat.org/classes/base_classes/NXlog.html

    Args:
        raw_times: The raw time data from a nexus file.
        group_path: The path within the nexus file to the log being read.
            Used to generate warnings if loading the log fails.
        log_start: Optional, the start time of the log in an ISO8601
            string. If not provided, defaults to the beginning of the
            unix epoch (1970-01-01T00:00:00Z).
        scaling_factor: Optional, the scaling factor between the provided
            time series data and the unit of the raw_times Variable. If
            not provided, defaults to 1 (a no-op scaling factor).
    """
    try:
        raw_times_ns = sc.to_unit(raw_times, sc.units.ns)
    except sc.UnitError:
        raise BadSource(f"The units of time in the NXlog entry at "
                        f"'{group_path}/time{{units}}' must be convertible to seconds, "
                        f"but this cannot be done for '{raw_times.unit}'. Skipping "
                        f"loading NXLog at '{group_path}'.")

    if log_start is not None:
        try:
            _start_ts = sc.scalar(value=np.datetime64(log_start),
                                  unit=sc.units.ns,
                                  dtype=sc.dtype.datetime64)
        except ValueError:
            raise BadSource(
                f"The date string '{log_start}' in the NXLog entry at "
                f"'{group_path}/time@start' failed to parse as an ISO8601 date. "
                f"Skipping loading NXLog at '{group_path}'")
    else:
        _start_ts = sc.scalar(value=np.datetime64("1970-01-01T00:00:00Z"),
                              unit=sc.units.ns,
                              dtype=sc.dtype.datetime64)

    _scale = sc.scalar(value=scaling_factor if scaling_factor is not None else 1.,
                       unit=sc.units.dimensionless,
                       dtype=sc.dtype.float64)

    return _start_ts + (raw_times_ns * _scale).astype(sc.dtype.int64)


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


def _load_log_data_from_group(group: Group,
                              nexus: LoadFromNexus) -> Tuple[str, sc.Variable]:
    property_name = nexus.get_name(group.group)
    value_dataset_name = "value"
    time_dataset_name = "time"

    try:
        values = nexus.load_dataset_from_group_as_numpy_array(
            group.group, value_dataset_name)
    except MissingDataset:
        if group.contains_stream:
            raise SkipSource("Log is missing value dataset but contains stream")
        raise BadSource(f"NXlog '{property_name}' has no value dataset")

    if values.size == 0:
        raise BadSource(f"NXlog '{property_name}' has an empty value dataset")

    unit = nexus.get_unit(nexus.get_dataset_from_group(group.group, value_dataset_name))
    try:
        unit = sc.Unit(unit)
    except sc.UnitError:
        warn(f"Unrecognized unit '{unit}' for value dataset "
             f"in NXlog '{group.path}'; setting unit as 'dimensionless'")
        unit = sc.units.dimensionless

    try:
        dimension_label = "time"
        is_time_series = True
        raw_times = nexus.load_dataset(group.group,
                                       time_dataset_name, [dimension_label],
                                       dtype=sc.dtype.float64)

        time_dataset = nexus.get_dataset_from_group(group.group, time_dataset_name)
        try:
            log_start_time = nexus.get_string_attribute(time_dataset, "start")
        except (MissingAttribute, TypeError):
            log_start_time = None

        try:
            scaling_factor = nexus.get_attribute(time_dataset, "scaling_factor")
        except (MissingAttribute, TypeError):
            scaling_factor = None

        times = _convert_nxlog_time_to_datetime64(raw_times=raw_times,
                                                  log_start=log_start_time,
                                                  scaling_factor=scaling_factor,
                                                  group_path=group.path)

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
                                      group.group, value_dataset_name))
    else:
        property_data = sc.Variable(values=values,
                                    unit=unit,
                                    dims=[dimension_label],
                                    dtype=nexus.get_dataset_numpy_dtype(
                                        group.group, value_dataset_name))

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
