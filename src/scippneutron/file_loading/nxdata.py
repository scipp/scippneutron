from typing import List, Union
import scipp as sc
from ._common import to_plain_index, Dataset, Group
from .nxobject import NXobject, ScippIndex
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
    def shape(self) -> List[int]:
        return self._signal.shape

    @property
    def dims(self) -> List[str]:
        # Apparently it is not possible to define dim labels unless there are
        # corresponding coords. Special case of '.' entries means "no coord".
        if 'axes' in self.attrs:
            axes = self.attrs['axes']
            return [f'dim_{i}' if a == '.' else a for i, a in enumerate(axes)]
        # Legacy NXdata defines axes not as group attribute, but attr on dataset
        return self._signal.attrs['axes'].split(',')

    @property
    def unit(self) -> Union[sc.Unit, None]:
        return self._signal.unit

    @property
    def _signal_name(self) -> str:
        name = self.attrs.get('signal', self._signal_name_default)
        if name is not None:
            return name
        # Legacy NXdata defines signal not as group attribute, but attr on dataset
        for name in self.keys():
            if self[name].attrs.get('signal') == 1:
                return name
        return None

    @property
    def _errors_name(self) -> str:
        if self._signal_name_default is None:
            return f'{self._signal_name_default}_errors'
        else:
            return 'errors'

    @property
    def _signal(self) -> Dataset:
        return self[self._signal_name]

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        dims = self.dims
        index = to_plain_index(dims, select)
        signal = self._loader.load_dataset(self._group,
                                           self._signal_name,
                                           dimensions=dims,
                                           index=index)
        if self._errors_name in self:
            stddevs = self._loader.load_dataset(self._group,
                                                self._errors_name,
                                                dimensions=dims,
                                                index=index)
            signal.variances = sc.pow(stddevs, 2).values
        da = sc.DataArray(data=signal)
        if 'axes' in self.attrs:
            # Unlike self.dims we *drop* entries that are '.'
            coords = [a for a in self.attrs['axes'] if a != '.']
        else:
            coords = self._signal.attrs['axes'].split(',')
        for dim in coords:
            index = to_plain_index([dim], select, ignore_missing=True)
            da.coords[dim] = self._loader.load_dataset(self._group,
                                                       dim,
                                                       dimensions=[dim],
                                                       index=index)
        return da
