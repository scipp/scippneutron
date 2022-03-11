# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
import scipp as sc
from enum import Enum, auto
import functools
from typing import List, Union, NoReturn, Any, Dict, Tuple

from ._nexus import LoadFromNexus
from ._hdf5_nexus import LoadFromHdf5
from ._common import Group, Dataset, MissingAttribute, ScippIndex
from ._common import to_plain_index

NXobjectIndex = Union[str, ScippIndex]


class NexusStructureError(Exception):
    """Invalid or unsupported class and field structure in Nexus.
    """
    pass


class NX_class(Enum):
    NXdata = auto()
    NXdetector = auto()
    NXentry = auto()
    NXevent_data = auto()
    NXlog = auto()
    NXmonitor = auto()
    NXroot = auto()
    NXtransformations = auto()


class Attrs:
    """HDF5 attributes.
    """
    def __init__(self,
                 node: Union[Dataset, Group],
                 loader: LoadFromNexus = LoadFromHdf5()):
        self._node = node
        self._loader = loader

    def __contains__(self, name: str) -> bool:
        try:
            _ = self[name]
            return True
        except MissingAttribute:
            return False

    def __getitem__(self, name: str) -> Any:
        attr = self._loader.get_attribute(self._node, name)
        # Is this check for string attributes sufficient? Is there a better way?
        if isinstance(attr, (str, bytes)):
            return self._loader.get_string_attribute(self._node, name)
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
        self._dims = [f'dim_{i}' for i in range(self.ndim)] if dims is None else dims

    def __getitem__(self, select) -> sc.Variable:
        index = to_plain_index(self.dims, select)
        return self._loader.load_dataset_direct(self._dataset,
                                                dimensions=self.dims,
                                                index=index)

    def __repr__(self) -> str:
        return f'<Nexus field "{self._dataset.name}">'

    @property
    def attrs(self) -> Attrs:
        return Attrs(self._dataset, self._loader)

    @property
    def dtype(self) -> str:
        return self._loader.get_dtype(self._dataset)

    @property
    def name(self) -> str:
        return self._loader.get_path(self._dataset)

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
        if 'units' in self.attrs:
            return sc.Unit(self._loader.get_unit(self._dataset))
        return None


class DependsOn:
    def __init__(self, field: Field):
        self._field = field

    def __getitem__(self, select) -> sc.Variable:
        index = to_plain_index([], select)
        if index != tuple():
            raise ValueError("Cannot select slice when loading 'depends_on'")
        from .nxtransformations import get_full_transformation_matrix
        return get_full_transformation_matrix(self._field._group, self._field._loader)


class NXobject:
    """Base class for all NeXus groups.
    """
    def __init__(self, group: Group, loader: LoadFromNexus = LoadFromHdf5()):
        self._group = group
        self._loader = loader

    def _make(self, group) -> '__class__':
        nx_class = self._loader.get_string_attribute(group, 'NX_class')
        return _nx_class_registry().get(nx_class, NXobject)(group, self._loader)

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
                return self._make(item)
            else:
                dims = self._get_field_dims(name) if use_field_dims else None
                return Field(item, self._loader, dims=dims)
        da = self._getitem(name)
        if (depends_on := self.depends_on) is not None:
            da.coords['depends_on'] = depends_on[()]
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
        return self._loader.dataset_in_group(self._group, name)[0]

    def get(self, name: str, default=None) -> Union['__class__', Field, sc.DataArray]:
        return self[name] if name in self else default

    @property
    def attrs(self) -> Attrs:
        return Attrs(self._group, self._loader)

    @property
    def name(self) -> str:
        return self._loader.get_path(self._group)

    def _ipython_key_completions_(self) -> List[str]:
        return list(self.keys())

    def keys(self) -> List[str]:
        return self._loader.keys(self._group)

    def values(self) -> List[Union[Field, '__class__']]:
        return [self[name] for name in self.keys()]

    def items(self) -> List[Tuple[str, Union[Field, '__class__']]]:
        return list(zip(self.keys(), self.values()))

    @functools.lru_cache()
    def by_nx_class(self) -> Dict[NX_class, List['__class__']]:
        classes = self._loader.find_by_nx_class(tuple(_nx_class_registry()),
                                                self._group)
        out = {}
        for nx_class, groups in classes.items():
            names = [self._loader.get_name(group) for group in groups]
            if len(names) != len(set(names)):  # fall back to full path if duplicate
                names = [group.name for group in groups]
            out[NX_class[nx_class]] = {n: self._make(g) for n, g in zip(names, groups)}
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
    def depends_on(self) -> DependsOn:
        if 'depends_on' in self:
            return DependsOn(self)
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


@functools.lru_cache()
def _nx_class_registry():
    from .nxmonitor import NXmonitor
    from ._detector_data import NXevent_data
    from .nxlog import NXlog
    from .nxdata import NXdata
    from .nxdetector import NXdetector
    from .nxtransformations import NXtransformations
    return {
        cls.__name__: cls
        for cls in [
            NXroot, NXentry, NXevent_data, NXlog, NXmonitor, NXdata, NXdetector,
            NXtransformations
        ]
    }
