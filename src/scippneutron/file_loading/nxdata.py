import scipp as sc
from ._common import to_plain_index
from .nxobject import NXobject


class NXdata(NXobject):
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
        signal_name = self.attrs.get('signal', 'data')
        dims = self[signal_name].attrs['axes'].split(",")
        index = to_plain_index(dims, select)
        signal = self._loader.load_dataset(self._group,
                                           signal_name,
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
