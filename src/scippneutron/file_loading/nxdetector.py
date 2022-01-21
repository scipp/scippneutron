from .nxobject import NXobject
from .nxdata import NXdata


class NXdetector(NXobject):
    @property
    def shape(self):
        pass

    @property
    def dims(self):
        pass

    @property
    def unit(self):
        pass

    def _getitem(self, select):
        # Note that ._detector_data._load_detector provides a different loading
        # facility for NXdetector, but handles only loading of detector_number,
        # as needed for event data loading
        signal_name = self.attrs.get('signal', 'data')
        if signal_name in self:
            return NXdata(self._group, self._loader)[select]
        raise NotImplementedError(
            f"NXdetector {self.name} does not contain data. Loading not implemented.")
