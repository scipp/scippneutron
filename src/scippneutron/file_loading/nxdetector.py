# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
from copy import copy
from typing import List, Union
import scipp as sc
from .nxobject import NX_class, NXobject, Field, ScippIndex, NexusStructureError
from .nxdata import NXdata
from .nxevent_data import NXevent_data
from ._common import to_child_select


class EventSelector:
    """A proxy object for creating an NXdetector based on a selection of events.
    """
    def __init__(self, detector):
        self._detector = detector

    def __getitem__(self, select: ScippIndex) -> NXdetector:
        """Return an NXdetector based on a selection (slice) of events."""
        det = copy(self._detector)
        det._event_select = select
        return det


class NXdetector(NXobject):
    """A detector or detector bank providing an array of values or events.

    If the detector stores event data then the 'detector_number' field (if present)
    is used to map event do detector pixels. Otherwise this returns event data in the
    same format as NXevent_data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_select = tuple()

    @property
    def shape(self) -> List[int]:
        if self._is_events:
            if (signal := self._signal) is not None:
                return signal.shape
            if self._detector_number is None:
                raise NexusStructureError(
                    "Cannot get shape of NXdetector since no 'detector_number' "
                    "field found but detector contains event data.")
            return self._detector_number.shape
        return self._nxbase.shape

    @property
    def dims(self) -> List[str]:
        if (signal := self._signal) is not None:
            return signal.dims
        if self._is_events:
            default = [f'dim_{i}' for i in range(self.ndim)]
            if len(default) == 1:
                default = ['detector_number']
            # The NeXus standard is lacking information on a number of details on
            # NXdetector, but according to personal communication with Tobias Richter
            # it is "intended" to partially "subclass" NXdata. That is, e.g., attributes
            # defined for NXdata such as 'axes' may be used.
            return self.attrs.get('axes', default)
        return self._nxbase.dims

    @property
    def ndim(self) -> int:
        if self._is_events:
            if 'detector_number' not in self:
                return 1
            det_field = self._get_child('detector_number')
            return det_field.ndim
        return self['data'].ndim

    @property
    def unit(self) -> Union[sc.Unit, None]:
        return self._nxbase.unit

    @property
    def _is_events(self) -> bool:
        # The standard is unclear on whether the 'data' field may be NXevent_data or
        # whether the fields of NXevent_data should be stored directly within this
        # NXdetector. Both cases are observed in the wild.
        event_entries = self.by_nx_class()[NX_class.NXevent_data]
        if len(event_entries) > 1:
            raise NexusStructureError("No unique NXevent_data entry in NXdetector. "
                                      f"Found {len(event_entries)}.")
        elif len(event_entries) == 1:
            if 'data' in self and not isinstance(self._get_child('data'), NXevent_data):
                raise NexusStructureError("NXdetector contains data and event data.")
            return True
        return 'event_time_offset' in self

    @property
    def _signal(self) -> Union[Field, None]:
        nxdata = self._nxdata
        name = nxdata._signal_name
        if name is not None and name in nxdata:
            return nxdata._signal

    @property
    def _nxdata(self) -> NXdata:
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXdetector uses a "hard-coded" signal name 'data', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        return NXdata(self._group,
                      self._loader,
                      signal='data' if 'data' in self else None,
                      skip=['detector_number'])

    @property
    def _nxbase(self) -> Union[NXdata, NXevent_data]:
        """Return class for loading underlying data."""
        if self._is_events:
            if 'event_time_offset' in self:
                return NXevent_data(self._group, self._loader)
            event_entries = self.by_nx_class()[NX_class.NXevent_data]
            return next(iter(event_entries.values()))
        return self._nxdata

    @property
    def events(self) -> Union[None, NXevent_data]:
        """Return the underlying NXevent_data group, None if not event data."""
        if self._is_events:
            return self._nxbase

    @property
    def select_events(self) -> EventSelector:
        """
        Return a proxy object for selecting a slice of the underlying NXevent_data
        group, while keeping wrapping the NXdetector.
        """
        if not self._is_events:
            raise NexusStructureError(
                "Cannot select events in NXdetector not containing NXevent_data.")
        return EventSelector(self)

    @property
    def _detector_number(self) -> Field:
        if 'detector_number' in self:
            return self['detector_number']
        return self.get('pixel_id', None)

    def detector_number(self, select) -> sc.Variable:
        """Read and return the 'detector_number' field, None if it does not exist."""
        field = self._detector_number
        if field is None:
            if self._is_events and select not in (Ellipsis,
                                                  tuple()) and select != slice(None):
                raise NexusStructureError(
                    "Cannot load slice of NXdetector since it contains event data "
                    "but no 'detector_number' field, i.e., the shape is unknown. "
                    "Use ellipsis or an empty tuple to load the full detector.")
            return None
        select = to_child_select(self.dims, field.dims, select)
        if field.dtype not in ['int32', 'int64', 'uint32', 'uint64']:
            raise NexusStructureError(
                "NXdetector contains detector_number field with non-integer values")
        return field[select]

    def pixel_offset(self, select) -> sc.Variable:
        """Read the [xyz]_pixel_offset fields and return a variable of pixel offset
        vectors, None if x_pixel_offset does not exist."""
        if 'x_pixel_offset' not in self:
            return None
        x = self['x_pixel_offset']
        select = to_child_select(self.dims, x.dims, select)
        x = x[select]
        offset = sc.zeros(dims=self.dims,
                          shape=self.shape,
                          unit=x.unit,
                          dtype=sc.DType.vector3)
        offset.fields.x = x.to(dtype='float64', copy=False)
        if (y := self.get('y_pixel_offset')) is not None:
            offset.fields.y = y[select].to(dtype='float64', unit=x.unit, copy=False)
        if (z := self.get('z_pixel_offset')) is not None:
            offset.fields.z = z[select].to(dtype='float64', unit=x.unit, copy=False)
        return offset.rename_dims(dict(zip(offset.dims, self.dims)))

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        if self._is_events:
            if name in [
                    'event_time_zero', 'event_index', 'event_time_offset', 'event_id'
            ]:
                # Event field is direct child of this class
                return self._nxbase._get_field_dims(name)
            elif self._signal is None:
                return self.dims
            else:
                self._nxdata._get_field_dims(name)
        return self._nxdata._get_field_dims(name)

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        # Note that ._detector_data._load_detector provides a different loading
        # facility for NXdetector but handles only loading of detector_number,
        # as needed for event data loading
        coords = {
            'detector_number': self.detector_number(select),
            'pixel_offset': self.pixel_offset(select)
        }
        if self._is_events:
            # If there is a 'detector_number' field it is used to bin events into
            # detector pixels. Note that due to the nature of NXevent_data, which stores
            # events from all pixels and random order, we always have to load the entire
            # bank. Slicing with the provided 'select' is done while binning.
            event_data = self._nxbase[self._event_select]
            if coords['detector_number'] is None:
                # Ideally we would prefer to use np.unique, but a quick experiment shows
                # that this can easily be 100x slower, so it is not an option. In
                # practice most files have contiguous event_id values within a bank
                # (NXevent_data).
                id_min = event_data.bins.coords['event_id'].min()
                id_max = event_data.bins.coords['event_id'].max()
                coords['detector_number'] = sc.arange(dim='detector_number',
                                                      unit=None,
                                                      start=id_min.value,
                                                      stop=id_max.value + 1,
                                                      dtype=id_min.dtype)
            event_id = coords['detector_number'].flatten(to='event_id')
            event_data.bins.coords['event_time_zero'] = sc.bins_like(
                event_data, fill_value=event_data.coords['event_time_zero'])
            # After loading raw NXevent_data it is guaranteed that the event table
            # is contiguous and that there is no masking. We can therefore use the
            # more efficient approach of binning from scratch instead of erasing the
            # 'pulse' binning defined by NXevent_data.
            event_data = sc.bin(event_data.bins.constituents['data'], groups=[event_id])
            del event_data.coords['event_id']  # same as detector_number
            da = event_data.fold(dim='event_id', sizes=coords['detector_number'].sizes)
        else:
            da = self._nxbase[select]
        for name, coord in coords.items():
            if coord is not None:
                da.coords[name] = coord
        return da
