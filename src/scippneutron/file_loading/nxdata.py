# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import List, Union
import scipp as sc
import numpy as np
from ._common import to_plain_index, Dataset, Group
from .nxobject import Field, NXobject, ScippIndex, NexusStructureError
from ._nexus import LoadFromNexus
from ._hdf5_nexus import LoadFromHdf5


class NXdata(NXobject):
    def __init__(self,
                 group: Group,
                 loader: LoadFromNexus = LoadFromHdf5(),
                 signal=None,
                 axes=None):
        """
        :param signal: Default signal name used, if no `signal` attribute found in file.
        :param axes: Default axes used, if no `axes` attribute found in file.
        """
        super().__init__(group, loader)
        self._signal_name_default = signal
        self._axes_default = axes

    @property
    def shape(self) -> List[int]:
        return self._signal.shape

    @property
    def dims(self) -> List[str]:
        # Apparently it is not possible to define dim labels unless there are
        # corresponding coords. Special case of '.' entries means "no coord".
        if self.attrs.get('axes', self._axes_default) is not None:
            axes = self.attrs.get('axes', self._axes_default)
            return [f'dim_{i}' if a == '.' else a for i, a in enumerate(axes)]
        # Legacy NXdata defines axes not as group attribute, but attr on dataset
        if 'axes' in self._signal.attrs:
            return self._signal.attrs['axes'].split(',')
        return [f'dim_{i}' for i in range(len(self.shape))]

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
            return 'errors'
        else:
            return f'{self._signal_name_default}_errors'

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

        skip = [self._signal_name, self._errors_name]
        skip += list(self.attrs.get('auxiliary_signals', []))
        items = [k for k in self.keys() if isinstance(self[k], Field)]
        items = [k for k in items if k not in skip]

        for name in items:
            # Newly written files should always contain indices attributes, but the
            # standard recommends that readers should also make "best effort" guess
            # since legacy files do not set this attribute.
            indices = self.attrs.get(f'{name}_indices')
            if indices is None:
                if (axes := self.attrs.get('axes', self._axes_default)) is not None:
                    # Unlike self.dims we *drop* entries that are '.'
                    axes = [a for a in axes if a != '.']
                elif 'axes' in self._signal.attrs:
                    axes = self._signal.attrs['axes'].split(',')
                else:
                    axes = None
                if name in axes:
                    # If there are named axes then items of same name are "dimension
                    # coordinates", i.e., have a dim matching their name.
                    dims = [name]
                else:
                    # Guess based on shape. Here we assume that axis order is same as
                    # for data (but axes may be missing). Favors first dim of matching
                    # length. We do not check for potential bin-edge coord in this case.
                    shape = list(self[name].shape)
                    dims = []
                    for dim, l in da.sizes.items():
                        if len(shape) == 0:
                            break
                        if shape[0] == l:
                            dims += [dim]
                            shape.pop(0)
                    if len(shape) != 0:
                        raise NexusStructureError("Could not determine axis indices")
            else:
                dims = np.array(da.dims)[np.array(indices).flatten()]
            index = to_plain_index(dims, select, ignore_missing=True)
            da.coords[name] = self._loader.load_dataset(self._group,
                                                        name,
                                                        dimensions=dims,
                                                        index=index)

        return da
