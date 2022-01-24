from typing import List, Union
import scipp as sc
from .nxobject import NXobject, Field, ScippIndex
from .nxdata import NXdata
from ._detector_data import NXevent_data


class NXdetector(NXobject):
    """A detector or detector bank providing an array of values or events.

    If the detector stores event data then the 'detector_number' field (if present)
    is used to map event do detector pixels. Otherwise this returns event data in the
    same format as NXevent_data.
    """
    @property
    def shape(self) -> List[int]:
        if self._is_events and self._detector_number is not None:
            return self._detector_number.shape
        # If event data but no detector_number then this gives the underlying
        # shape of NXevent_data
        return self._nxbase.shape

    @property
    def dims(self) -> List[str]:
        if self._is_events and self._detector_number is not None:
            return self.attrs.get('axes', ['detector_number'])
        # If event data but no detector_number then this gives the underlying
        # dims of NXevent_data
        return self._nxbase.dims

    @property
    def unit(self) -> Union[sc.Unit, None]:
        return self._nxbase.unit

    @property
    def _is_events(self) -> bool:
        # The standard is unclear on whether the 'data' field may be NXevent_data or
        # whether the fields of NXevent_data should be stored directly within this
        # NXdetector. Both cases are observed in the wild.
        if 'data' in self:
            return isinstance(self['data'], NXevent_data)
        return 'event_time_offset' in self

    @property
    def _nxbase(self) -> NXdata:
        """Return class for loading underlying data."""
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXdetector uses a "hard-coded" signal name 'data', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        if self._is_events:
            if 'event_time_offset' in self:
                return NXevent_data(self._group, self._loader)
            return self['data']
        return NXdata(self._group, self._loader, signal='data')

    @property
    def _detector_number(self) -> Field:
        return self.get('detector_number', None)

    @property
    def detector_number(self) -> sc.Variable:
        """Read and return the 'detector_number' field, None if it does not exist."""
        if self._detector_number is None:
            return None
        return sc.array(dims=self.dims, values=self._detector_number[...])

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        # Note that ._detector_data._load_detector provides a different loading
        # facility for NXdetector but handles only loading of detector_number,
        # as needed for event data loading
        if self._is_events:
            if self.detector_number is None:
                # If there is no 'detector_data' then self.dims is the same as that of
                # the underlying event data. Slicing here thus corresponds to slicing
                # NXevent_data.
                return self._nxbase[select]
            # If there is a 'detector_number' field it is used to bin events into
            # detector pixels. Note that due to the nature of NXevent_data, which stores
            # events from all pixels and random order we, always have to load the entire
            # bank. Slicing with the provided 'select' is done while binning.
            event_data = self._nxbase[...]
            event_data.bins.coords['detector_number'] = event_data.bins.coords.pop(
                'event_id')
            return sc.bin(event_data,
                          groups=[self.detector_number[select]],
                          erase=['pulse'])
        return self._nxbase[select]
