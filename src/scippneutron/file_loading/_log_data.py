# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

from typing import Tuple, List, Dict
import scipp as sc
from ._common import (BadSource, SkipSource, MissingDataset, Group,
                      _convert_time_to_datetime64)
from ._nexus import LoadFromNexus
from .nxobject import NXobject, ScippIndex
from .nxdata import NXdata
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
        return self._nxbase.shape

    @property
    def dims(self):
        return self._nxbase.dims

    @property
    def unit(self):
        return self._nxbase.unit

    @property
    def _nxbase(self) -> NXdata:
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXlog uses a "hard-coded" signal name 'value', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        axes = ['.'] * self['value'].ndim
        # The outermost axis in NXlog is hard-coded to 'time' (if present)
        if 'time' in self:
            axes[0] = 'time'
        return NXdata(self._group, self._loader, signal='value', axes=axes)

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        data = self._nxbase[select]
        # The 'time' field in NXlog contains extra properties 'start' and
        # 'scaling_factor' that are not handled by NXdata. These are used
        # to transform to a datetime-coord.
        if 'time' in self:
            data.coords['time'] = _convert_time_to_datetime64(
                raw_times=data.coords.pop('time'),
                start=self['time'].attrs.get('start'),
                scaling_factor=self['time'].attrs.get('scaling_factor'),
                group_path=self['time'].name)
        return data


def _load_log_data_from_group(group: Group, nexus: LoadFromNexus, select=tuple())\
        -> Tuple[str, sc.Variable]:
    property_name = nexus.get_name(group)
    try:
        return property_name, sc.scalar(NXlog(group, nexus)[select])
    except sc.DimensionError:
        raise BadSource(f"NXlog '{property_name}' has time and value "
                        f"datasets of different shapes")
    except (KeyError, MissingDataset):
        if nexus.contains_stream(group):
            raise SkipSource("Log is missing value dataset but contains stream")
        raise BadSource(f"NXlog '{property_name}' has no value dataset")
