# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from typing import Union, Optional, Any
import h5py
from scipp.spatial import affine_transform, linear_transform, \
    rotation, translation
import scipp as sc
import numpy as np
from typing import Dict


class BadSource(Exception):
    """
    Raise if something is wrong with data source which
    prevents it being used. Warn the user.
    """
    pass


class SkipSource(Exception):
    """
    Raise to abort using the data source, do not
    warn the user.
    """
    pass


class MissingDataset(Exception):
    pass


class MissingAttribute(Exception):
    pass


class JSONGroup(dict):
    def __init__(self, parent: dict, name: str, file: dict, group: dict):
        super().__init__(**group)
        self.parent = parent
        self.name = name
        self.file = file


Dataset = Union[h5py.Dataset, Dict]
Group = Union[h5py.Group, JSONGroup]


def _add_attr_to_loaded_data(attr_name: str,
                             data: sc.Variable,
                             value: np.ndarray,
                             unit: sc.Unit,
                             dtype: Optional[Any] = None):
    try:
        data = data.attrs
    except AttributeError:
        pass

    try:
        if dtype is not None:
            if dtype == sc.DType.vector3:
                data[attr_name] = sc.vector(value=value, unit=unit)
            elif dtype == sc.DType.affine_transform3:
                data[attr_name] = affine_transform(value=value, unit=unit)
            elif dtype == sc.DType.linear_transform3:
                data[attr_name] = linear_transform(value=value, unit=unit)
            elif dtype == sc.DType.rotation3:
                if unit != sc.units.one:
                    raise sc.UnitError(
                        f'Rotations must be dimensionless, got unit {unit}')
                data[attr_name] = rotation(value=value)
            elif dtype == sc.DType.translation3:
                data[attr_name] = translation(value=value, unit=unit)
            else:
                data[attr_name] = sc.scalar(value=value, dtype=dtype, unit=unit)
        else:
            data[attr_name] = sc.scalar(value=value, unit=unit)
    except KeyError:
        pass


def _convert_time_to_datetime64(
        raw_times: sc.Variable,
        group_path: str,
        start: str = None,
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
        start: Optional, the start time of the log in an ISO8601
            string. If not provided, defaults to the beginning of the
            unix epoch (1970-01-01T00:00:00Z).
        scaling_factor: Optional, the scaling factor between the provided
            time series data and the unit of the raw_times Variable. If
            not provided, defaults to 1 (a no-op scaling factor).
    """
    try:
        raw_times_ns = sc.to_unit(raw_times, sc.units.ns, copy=False)
    except sc.UnitError:
        raise BadSource(f"The units of time in the entry at "
                        f"'{group_path}/time{{units}}' must be convertible to seconds, "
                        f"but this cannot be done for '{raw_times.unit}'. Skipping "
                        f"loading group at '{group_path}'.")

    try:
        _start_ts = sc.scalar(value=np.datetime64(start or "1970-01-01T00:00:00Z"),
                              unit=sc.units.ns,
                              dtype=sc.DType.datetime64)
    except ValueError:
        raise BadSource(
            f"The date string '{start}' in the entry at "
            f"'{group_path}/time@start' failed to parse as an ISO8601 date. "
            f"Skipping loading group at '{group_path}'")

    _scale = sc.scalar(value=scaling_factor if scaling_factor is not None else 1,
                       unit=sc.units.dimensionless)

    return _start_ts + (raw_times_ns * _scale).astype(sc.DType.int64, copy=False)


def load_time_dataset(nexus, group, dataset_name, dim, group_name=None, index=...):
    raw_times = nexus.load_dataset(group, dataset_name, [dim], index=index)

    time_dataset = nexus.get_dataset_from_group(group, dataset_name)
    try:
        start = nexus.get_string_attribute(time_dataset, "start")
    except (MissingAttribute, TypeError):
        start = None

    try:
        scaling_factor = nexus.get_attribute(time_dataset, "scaling_factor")
    except (MissingAttribute, TypeError):
        scaling_factor = None

    return _convert_time_to_datetime64(raw_times=raw_times,
                                       start=start,
                                       scaling_factor=scaling_factor,
                                       group_path=group_name or group.name)


def to_plain_index(dims, select, ignore_missing=False):
    """
    Given a valid "scipp" index 'select', return an equivalent plain numpy-style index.
    """
    def check_1d():
        if len(dims) != 1:
            raise ValueError(f"Dataset has multiple dimensions {dims}, "
                             "specify the dimension to index.")

    if select is Ellipsis:
        return select
    if isinstance(select, tuple) and len(select) == 0:
        return select
    if isinstance(select, tuple) and isinstance(select[0], str):
        key, sel = select
        select = {key: sel}

    if isinstance(select, tuple):
        check_1d()
        return select
    elif isinstance(select, int) or isinstance(select, slice):
        check_1d()
        return select
    elif isinstance(select, dict):
        index = [slice(None)] * len(dims)
        for key, sel in select.items():
            if not ignore_missing and key not in dims:
                raise ValueError(
                    f"'{key}' used for indexing not found in dataset dims {dims}.")
            index[dims.index(key)] = sel
        return tuple(index)
    raise ValueError("Cannot process index {select}.")
