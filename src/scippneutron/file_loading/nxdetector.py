from .nxobject import NXobject
from .nxdata import NXdata


class NXdetector(NXobject):
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
        """Return class for loading underlying data.

        Raises if no data found."""
        signal_name = self.attrs.get('signal', 'data')
        if signal_name in self:
            return NXdata(self._group, self._loader, signal='data')
        raise NotImplementedError(f"NXdetector {self.name} does not contain data.")

    def _getitem(self, select):
        # Note that ._detector_data._load_detector provides a different loading
        # facility for NXdetector, but handles only loading of detector_number,
        # as needed for event data loading
        return self._nxbase[select]
