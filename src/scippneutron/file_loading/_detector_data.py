# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from typing import List, Union
import numpy as np
import scipp as sc

from ._common import (BadSource, SkipSource, Group)
from ._common import to_plain_index, convert_time_to_datetime64
from ._nexus import LoadFromNexus
from .nxobject import NXobject, ScippIndex, NexusStructureError

_event_dimension = "event"
_pulse_dimension = "pulse"


def _check_for_missing_fields(group: Group, nexus: LoadFromNexus):
    if nexus.contains_stream(group):
        # Do not warn about missing datasets if the group contains
        # a stream, as this will provide the missing data
        raise SkipSource("Data source is missing datasets"
                         "but contains a stream source for the data")

    required_fields = (
        "event_time_zero",
        "event_index",
        "event_time_offset",
    )
    for field in required_fields:
        found, msg = nexus.dataset_in_group(group, field)
        if not found:
            raise BadSource(msg)


class NXevent_data(NXobject):
    @property
    def shape(self) -> List[int]:
        return self._loader.get_shape(
            self._loader.get_dataset_from_group(self._group, "event_index"))

    @property
    def dims(self) -> List[str]:
        return [_pulse_dimension]

    @property
    def unit(self) -> None:
        # Binned data, bins do not have a unit
        return None

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        return self._load_event_group(self._group, self._loader, select=select)

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        if name in ['event_time_zero', 'event_index']:
            return [_pulse_dimension]
        if name in ['event_time_offset', 'event_id']:
            return [_event_dimension]
        return None

    def _load_event_group(self, group: Group, nexus: LoadFromNexus,
                          select) -> sc.DataArray:
        _check_for_missing_fields(group, nexus)
        index = to_plain_index([_pulse_dimension], select)

        max_index = self["event_index"].shape[0]
        single = False
        if index is Ellipsis or index == tuple():
            last_loaded = False
        else:
            if isinstance(index, int):
                single = True
                start, stop, _ = slice(index, None).indices(max_index)
                if start == stop:
                    raise IndexError('Index {start} is out of range')
                index = slice(start, start + 1)
            start, stop, stride = index.indices(max_index)
            if stop + stride > max_index:
                last_loaded = False
            else:
                stop += stride
                last_loaded = True
            index = slice(start, stop, stride)

        event_index = self['event_index'][index].values
        event_time_zero = self['event_time_zero']
        event_time_zero = convert_time_to_datetime64(
            event_time_zero[index],
            start=event_time_zero.attrs.get('offset'),
            group_path=self.name)

        num_event = self["event_time_offset"].shape[0]
        # Some files contain uint64 "max" indices, which turn into negatives during
        # conversion to int64. This is a hack to get arround this.
        event_index[event_index < 0] = num_event

        if len(event_index) > 0:
            event_select = slice(event_index[0],
                                 event_index[-1] if last_loaded else num_event)
        else:
            event_select = slice(None)

        if nexus.dataset_in_group(group, "event_id")[0]:
            event_id = nexus.load_dataset(group,
                                          "event_id", [_event_dimension],
                                          index=event_select)
            if event_id.dtype not in [sc.DType.int32, sc.DType.int64]:
                raise NexusStructureError(
                    "NXevent_data contains event_id field with non-integer values")
        else:
            event_id = None

        event_time_offset = nexus.load_dataset(group,
                                               "event_time_offset", [_event_dimension],
                                               index=event_select)

        # Weights are not stored in NeXus, so use 1s
        weights = sc.ones(dims=[_event_dimension],
                          shape=event_time_offset.shape,
                          unit='counts',
                          dtype=np.float32,
                          with_variances=True)

        events = sc.DataArray(data=weights,
                              coords={'event_time_offset': event_time_offset})
        if event_id is not None:
            events.coords['event_id'] = event_id

        if not last_loaded:
            event_index = np.append(event_index, num_event)
        else:
            # Not a bin-edge coord, all events in bin are associated with same
            # (previous) pulse time value
            event_time_zero = event_time_zero[:-1]

        event_index = sc.array(dims=[_pulse_dimension],
                               values=event_index,
                               dtype=sc.DType.int64,
                               unit=None)

        event_index -= event_index.min()

        # There is some variation in the last recorded event_index in files from
        # different institutions. We try to make sure here that it is what would be the
        # first index of the next pulse. In other words, ensure that event_index
        # includes the bin edge for the last pulse.
        if single:
            begins = event_index[_pulse_dimension, 0]
            ends = event_index[_pulse_dimension, 1]
            event_time_zero = event_time_zero[_pulse_dimension, 0]
        else:
            begins = event_index[_pulse_dimension, :-1]
            ends = event_index[_pulse_dimension, 1:]

        try:
            binned = sc.bins(data=events, dim=_event_dimension, begin=begins, end=ends)
        except sc.SliceError:
            raise BadSource(
                f"Event index in NXEvent at {group.name}/event_index was not"
                f" ordered. The index must be ordered to load pulse times.")

        return sc.DataArray(data=binned, coords={'event_time_zero': event_time_zero})
