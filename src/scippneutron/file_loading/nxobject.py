# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from __future__ import annotations
import warnings
from enum import Enum, auto
import functools
from typing import List, Union, NoReturn, Any, Dict, Tuple
import scipp as sc

from ._nexus import LoadFromNexus
from ._hdf5_nexus import LoadFromHdf5, _cset_to_encoding, _ensure_str
from ._json_nexus import JSONAttributeManager
from ._common import Group, Dataset, ScippIndex
from ._common import to_plain_index

NXobjectIndex = Union[str, ScippIndex]


class NexusStructureError(Exception):
    """Invalid or unsupported class and field structure in Nexus.
    """
    pass


class NX_class(Enum):
    NXdata = auto()
    NXdetector = auto()
    NXdisk_chopper = auto()
    NXentry = auto()
    NXevent_data = auto()
    NXinstrument = auto()
    NXlog = auto()
    NXmonitor = auto()
    NXroot = auto()
    NXsample = auto()
    NXsource = auto()


class Attrs:
    """HDF5 attributes.
    """
    def __init__(self, node: Union[Dataset, Group]):
        self._node = node
        if isinstance(node, dict):
            self._attrs = JSONAttributeManager(node)
        else:
            self._attrs = node.attrs

    def __contains__(self, name: str) -> bool:
        return name in self._attrs

    def __getitem__(self, name: str) -> Any:
        attr = self._attrs[name]
        # Is this check for string attributes sufficient? Is there a better way?
        is_json = isinstance(self._attrs, JSONAttributeManager)
        if isinstance(attr, (str, bytes)) and not is_json:
            import h5py
            cset = h5py.h5a.open(self._node.id,
                                 name.encode("utf-8")).get_type().get_cset()
            return _ensure_str(attr, _cset_to_encoding(cset))
        return attr

    def get(self, name: str, default=None) -> Any:
        return self[name] if name in self else default


class Field:
    """NeXus field.

    In HDF5 fields are represented as dataset.
    """
    def __init__(self,
                 dataset: Dataset,
                 loader: LoadFromNexus = LoadFromHdf5(),
                 dims=None):
        self._dataset = dataset
        self._loader = loader
        if dims is not None:
            self._dims = dims
        elif (axes := self.attrs.get('axes')) is not None:
            self._dims = axes.split(',')
        else:
            self._dims = [f'dim_{i}' for i in range(self.ndim)]

    def __getitem__(self, select) -> sc.Variable:
        index = to_plain_index(self.dims, select)
        return self._loader.load_dataset_direct(self._dataset,
                                                dimensions=self.dims,
                                                unit=self.unit,
                                                index=index)

    def __repr__(self) -> str:
        return f'<Nexus field "{self._dataset.name}">'

    @property
    def attrs(self) -> Attrs:
        return Attrs(self._dataset)

    @property
    def dtype(self) -> str:
        return self._loader.get_dtype(self._dataset)

    @property
    def name(self) -> str:
        return self._loader.get_path(self._dataset)

    @property
    def file(self) -> NXroot:
        return NXroot(self._dataset.file, self._loader)

    @property
    def parent(self) -> NXobject:
        return _make(self._dataset.parent, self._loader)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> List[int]:
        return self._loader.get_shape(self._dataset)

    @property
    def dims(self) -> List[str]:
        return self._dims

    @property
    def unit(self) -> Union[sc.Unit, None]:
        if (unit := self.attrs.get('units')) is not None:
            try:
                return sc.Unit(unit)
            except sc.UnitError:
                warnings.warn(f"Unrecognized unit '{unit}' for value dataset "
                              f"in '{self.name}'; setting unit as 'dimensionless'")
                return sc.units.one
        return None


class NXobject:
    """Base class for all NeXus groups.
    """
    def __init__(self, group: Group, loader: LoadFromNexus = LoadFromHdf5()):
        self._group = group
        self._loader = loader

    def _get_child(
            self,
            name: NXobjectIndex,
            use_field_dims: bool = False) -> Union['__class__', Field, sc.DataArray]:
        """Get item, with flag to control whether fields dims should be inferred"""
        if name is None:
            raise KeyError("None is not a valid index")
        if isinstance(name, str):
            item = self._loader.get_child_from_group(self._group, name)
            if item is None:
                raise KeyError(f"Unable to open object (object '{name}' doesn't exist)")
            if self._loader.is_group(item):
                return _make(item, self._loader)
            else:
                dims = self._get_field_dims(name) if use_field_dims else None
                return Field(item, self._loader, dims=dims)
        da = self._getitem(name)
        if (t := self.depends_on) is not None:
            da.coords['depends_on'] = t if isinstance(t, sc.Variable) else sc.scalar(t)
        return da

    def __getitem__(self,
                    name: NXobjectIndex) -> Union['__class__', Field, sc.DataArray]:
        return self._get_child(name, use_field_dims=True)

    def _getitem(self, index: ScippIndex) -> NoReturn:
        raise NotImplementedError(f'Loading {self.nx_class} is not supported.')

    def _get_field_dims(self, name: str) -> Union[None, List[str]]:
        """Subclasses should reimplement this to provide dimension labels for fields."""
        return None

    def __contains__(self, name: str) -> bool:
        return self._loader.dataset_in_group(self._group, name)

    def get(self, name: str, default=None) -> Union['__class__', Field, sc.DataArray]:
        return self[name] if name in self else default

    @property
    def attrs(self) -> Attrs:
        return Attrs(self._group)

    @property
    def name(self) -> str:
        return self._loader.get_path(self._group)

    @property
    def file(self) -> NXroot:
        return NXroot(self._group.file, self._loader)

    @property
    def parent(self) -> NXobject:
        return _make(self._group.parent, self._loader)

    def _ipython_key_completions_(self) -> List[str]:
        return list(self.keys())

    def keys(self) -> List[str]:
        return self._loader.keys(self._group)

    def values(self) -> List[Union[Field, '__class__']]:
        return [self[name] for name in self.keys()]

    def items(self) -> List[Tuple[str, Union[Field, '__class__']]]:
        return list(zip(self.keys(), self.values()))

    @functools.lru_cache()
    def by_nx_class(self) -> Dict[NX_class, Dict[str, '__class__']]:
        classes = self._loader.find_by_nx_class(tuple(_nx_class_registry()),
                                                self._group)
        out = {}
        for nx_class, groups in classes.items():
            names = [self._loader.get_name(group) for group in groups]
            if len(names) != len(set(names)):  # fall back to full path if duplicate
                names = [group.name for group in groups]
            out[NX_class[nx_class]] = {
                n: _make(g, self._loader)
                for n, g in zip(names, groups)
            }
        return out

    @property
    def nx_class(self) -> NX_class:
        """The value of the NX_class attribute of the group.

        In case of the subclass NXroot this returns 'NXroot' even if the attribute
        is not actually set. This is support the majority of all legacy files, which
        do not have this attribute.
        """
        return NX_class[self.attrs['NX_class']]

    @property
    def depends_on(self) -> Union[sc.Variable, sc.DataArray, None]:
        if (depends_on := self.get('depends_on')) is not None:
            # Imported late to avoid cyclic import
            from .nxtransformations import get_full_transformation
            return get_full_transformation(depends_on)
        return None

    def __repr__(self) -> str:
        return f'<{type(self).__name__} "{self._group.name}">'


class NXroot(NXobject):
    @property
    def nx_class(self) -> NX_class:
        # As an oversight in the NeXus standard and the reference implementation,
        # the NX_class was never set to NXroot. This applies to essentially all
        # files in existence before 2016, and files written by other implementations
        # that were inspired by the reference implementation. We thus hardcode NXroot:
        return NX_class['NXroot']


class NXentry(NXobject):
    pass


class NXinstrument(NXobject):
    pass


def _make(group, loader) -> NXobject:
    if (nx_class := Attrs(group).get('NX_class')) is not None:
        return _nx_class_registry().get(nx_class, NXobject)(group, loader)
    return group  # Return underlying (h5py) group


@functools.lru_cache()
def _nx_class_registry():
    from .nxevent_data import NXevent_data
    from .nxdata import NXdata
    from .nxdetector import NXdetector
    from .nxdisk_chopper import NXdisk_chopper
    from .nxlog import NXlog
    from .nxmonitor import NXmonitor
    from .nxsample import NXsample
    from .nxsource import NXsource
    return {
        cls.__name__: cls
        for cls in [
            NXroot, NXentry, NXevent_data, NXlog, NXmonitor, NXdata, NXdetector,
            NXsample, NXsource, NXdisk_chopper, NXinstrument
        ]
    }
