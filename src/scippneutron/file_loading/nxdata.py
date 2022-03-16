# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import List, Union
from warnings import warn
import scipp as sc
import numpy as np
from ._common import to_child_select, Dataset, Group
from .nxobject import Field, NXobject, ScippIndex, NexusStructureError
from ._nexus import LoadFromNexus
from ._hdf5_nexus import LoadFromHdf5


class NXdata(NXobject):
    def __init__(self,
                 group: Group,
                 loader: LoadFromNexus = LoadFromHdf5(),
                 signal: str = None,
                 axes: List[str] = None,
                 skip: List[str] = None):
        """
        :param signal: Default signal name used, if no `signal` attribute found in file.
        :param axes: Default axes used, if no `axes` attribute found in file.
        """
        super().__init__(group, loader)
        self._signal_name_default = signal
        self._axes_default = axes
        self._skip = skip if skip is not None else []

    @property
    def shape(self) -> List[int]:
        return self._signal.shape

    @property
    def dims(self) -> List[str]:
        # Apparently it is not possible to define dim labels unless there are
        # corresponding coords. Special case of '.' entries means "no coord".
        if (axes := self.attrs.get('axes', self._axes_default)) is not None:
            return [f'dim_{i}' if a == '.' else a for i, a in enumerate(axes)]
        # Legacy NXdata defines axes not as group attribute, but attr on dataset
        if (axes := self._signal.attrs.get('axes')) is not None:
            return axes.split(',')
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
            # TODO What is the meaning of the attribute value?
            if 'signal' in self._get_child(name).attrs:
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
        return self._get_child(self._signal_name)

    def _get_axes(self):
        """Return labels of named axes."""
        if (axes := self.attrs.get('axes', self._axes_default)) is not None:
            # Unlike self.dims we *drop* entries that are '.'
            return [a for a in axes if a != '.']
        elif 'axes' in self._signal.attrs:
            return self._signal.attrs['axes'].split(',')
        return []

    def _guess_dims(self, name: str):
        """Guess dims of non-signal dataset based on shape.

        Does not check for potential bin-edge coord.
        """
        lut = {}
        for d, s in zip(self.dims, self.shape):
            if self.shape.count(s) == 1:
                lut[s] = d
        shape = self._get_child(name).shape
        if self.shape == shape:
            return self.dims
        try:
            dims = [lut[s] for s in shape]
        except KeyError:
            raise NexusStructureError(
                "Could not determine axis indices for {self[name].name}")
        return dims

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        # Newly written files should always contain indices attributes, but the
        # standard recommends that readers should also make "best effort" guess
        # since legacy files do not set this attribute.
        if (indices := self.attrs.get(f'{name}_indices')) is not None:
            return np.array(self.dims)[np.array(indices).flatten()]
        signals = [self._signal_name, self._errors_name]
        signals += list(self.attrs.get('auxiliary_signals', []))
        if name in signals:
            return self.dims
        if name in self._get_axes():
            # If there are named axes then items of same name are "dimension
            # coordinates", i.e., have a dim matching their name.
            return [name]
        try:
            return self._guess_dims(name)
        except NexusStructureError:
            return None

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        signal = self[self._signal_name][select]
        if self._errors_name in self:
            stddevs = self[self._errors_name][select]
            signal.variances = sc.pow(stddevs, 2).values
        da = sc.DataArray(data=signal)

        skip = self._skip
        skip += [self._signal_name, self._errors_name]
        skip += list(self.attrs.get('auxiliary_signals', []))

        for name, field in self.items():
            if (not isinstance(field, Field)) or (name in skip):
                continue
            try:
                sel = to_child_select(self.dims, field.dims, select)
                da.coords[name] = self[name][sel]
            except sc.DimensionError as e:
                warn(f"Skipped load of axis {name} due to: {e}")

        return da
