import scipp as sc
from ._common import to_plain_index
from .nxobject import NXobject


class NXdata(NXobject):
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
        return self.attrs.get('signal', 'data')

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
