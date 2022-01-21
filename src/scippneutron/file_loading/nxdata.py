import scipp as sc
from ._common import to_plain_index, Group
from .nxobject import NXobject
from ._nexus import LoadFromNexus
from ._hdf5_nexus import LoadFromHdf5


class NXdata(NXobject):
    def __init__(self,
                 group: Group,
                 loader: LoadFromNexus = LoadFromHdf5(),
                 signal=None):
        super().__init__(group, loader)
        self._signal_name_default = signal

    @property
    def shape(self):
        return self._signal.shape

    @property
    def dims(self):
        return self._signal.attrs['axes'].split(",")

    @property
    def unit(self):
        return self._signal.unit

    @property
    def _signal_name(self):
        name = self.attrs.get('signal', self._signal_name_default)
        if name is not None:
            return name
        # Legacy NXdata defines signal not as group attribute, but attr on dataset
        for name in self.keys():
            if self[name].attrs.get('signal') == 1:
                return name
        return None

    @property
    def _signal(self):
        return self[self._signal_name]

    def _getitem(self, select):
        dims = self.dims
        index = to_plain_index(dims, select)
        # TODO Handle errors
        signal = self._loader.load_dataset(self._group,
                                           self._signal_name,
                                           dimensions=dims,
                                           index=index)
        da = sc.DataArray(data=signal)
        for dim in dims:
            index = to_plain_index([dim], select, ignore_missing=True)
            da.coords[dim] = self._loader.load_dataset(self._group,
                                                       dim,
                                                       dimensions=[dim],
                                                       index=index)
        return da
