# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
import warnings

from typing import Union, Any, List, Optional, Tuple, Dict

import h5py
import numpy as np
import scipp as sc
from ._common import Group, MissingDataset


def _cset_to_encoding(cset: int) -> str:
    """
    Converts a HDF5 cset into a python encoding. Allowed values for cset are
    h5py.h5t.CSET_ASCII and h5py.h5t.CSET_UTF8.

    Args:
        cset: The HDF character set to convert

    Returns:
        A string describing the encoding suitable for calls to str(encoding=...)
        Either "ascii" or "utf-8".
    """
    if cset == h5py.h5t.CSET_ASCII:
        return "ascii"
    elif cset == h5py.h5t.CSET_UTF8:
        return "utf-8"
    else:
        raise ValueError(f"Unknown character set in HDF5 data file. Expected data "
                         f"types are {h5py.h5t.CSET_ASCII=} or "
                         f"{h5py.h5t.CSET_UTF8=} but got '{cset}'. ")


def _get_attr_as_str(h5_object, attribute_name: str) -> str:
    return _ensure_str(h5_object.attrs[attribute_name],
                       LoadFromHdf5.get_attr_encoding(h5_object, attribute_name))


def _warn_latin1_decode(obj, decoded, error):
    warnings.warn(f"Encoding for bytes '{obj}' declared as ascii, "
                  f"but contains characters in extended ascii range. Assuming "
                  f"extended ASCII (latin-1), but this behavior is not "
                  f"specified by the HDF5 or nexus standards and may therefore "
                  f"be incorrect. Decoded string using latin-1 is '{decoded}'. "
                  f"Error was '{error}'.")


def _ensure_str(str_or_bytes: Union[str, bytes], encoding: str) -> str:
    """
    See https://docs.h5py.org/en/stable/strings.html for justification about some of
    the operations performed in this method. In particular, variable-length strings
    are returned as `str` from h5py, but need to be encoded using the surrogateescape
    error handler and then decoded using the encoding specified in the nexus file in
    order to get a correctly encoded string in all cases.

    Note that the nexus standard leaves unspecified the behavior of H5T_CSET_ASCII
    for characters >=128. Common extensions are the latin-1 ("extended ascii") character
    set which appear to be used in nexus files from some facilities. Attempt to load
    these strings with the latin-1 extended character set, but warn as this is
    technically unspecified behavior.
    """
    if isinstance(str_or_bytes, str):
        str_or_bytes = str_or_bytes.encode("utf-8", errors="surrogateescape")

    if encoding == "ascii":
        try:
            return str(str_or_bytes, encoding="ascii")
        except UnicodeDecodeError as e:
            decoded = str(str_or_bytes, encoding="latin-1")
            _warn_latin1_decode(str_or_bytes, decoded, str(e))
            return decoded
    else:
        return str(str_or_bytes, encoding)


_map_to_supported_type = {
    np.int8: np.int32,
    np.int16: np.int32,
    np.uint8: np.int32,
    np.uint16: np.int32,
    np.uint32: np.int32,
    np.uint64: np.int64,
}


def _ensure_supported_int_type(dataset_type: Any):
    return _map_to_supported_type.get(dataset_type, dataset_type)


class LoadFromHdf5:
    def keys(self, group: h5py.Group):
        return group.keys()

    def values(self, group: h5py.Group):
        return group.values()

    @staticmethod
    def find_by_nx_class(
            nx_class_names: Tuple[str, ...],
            root: Union[h5py.File, h5py.Group]) -> \
            Dict[str, List[Group]]:
        """
        Finds groups with requested NX_class in the subtree of root

        Returns a dictionary with NX_class name as the key and
        list of matching groups as the value
        """
        found_groups: Dict[str, List[Group]] = {
            class_name: []
            for class_name in nx_class_names
        }

        def _match_nx_class(_, h5_object):
            if LoadFromHdf5.is_group(h5_object):
                try:
                    nx_class = _get_attr_as_str(h5_object, "NX_class")
                    if nx_class in nx_class_names:
                        found_groups[nx_class].append(h5_object)
                except KeyError:
                    pass

        root.visititems(_match_nx_class)
        return found_groups

    @staticmethod
    def dataset_in_group(group: h5py.Group, dataset_name: str) -> bool:
        return dataset_name in group

    def load_dataset_direct(self,
                            dataset: h5py.Dataset,
                            dimensions: Optional[List[str]] = [],
                            dtype: Optional[Any] = None,
                            index=tuple()) -> sc.Variable:
        """
        Load an HDF5 dataset into a Scipp Variable (array or scalar)
        :param group: Group containing dataset to load
        :param dataset_name: Name of the dataset to load
        :param dimensions: Dimensions for the output Variable. Empty for reading scalars
        :param dtype: Cast to this dtype during load,
          otherwise retain dataset dtype
        """
        if dtype is None:
            dtype = _ensure_supported_int_type(dataset.dtype.type)
        if h5py.check_string_dtype(dataset.dtype):
            dtype = sc.DType.string

        shape = list(dataset.shape)
        if dimensions == [] and shape == [1]:
            # NeXus treats [] and [1] interchangeably, in general this is ill-defined,
            # but this is the best we can do.
            shape = []
        if index is Ellipsis:
            index = tuple()
        if isinstance(index, slice):
            index = (index, )
        for i, ind in enumerate(index):
            shape[i] = len(range(*ind.indices(shape[i])))

        variable = sc.empty(dims=dimensions,
                            shape=shape,
                            dtype=dtype,
                            unit=self.get_unit(dataset))
        if dtype == sc.DType.string:
            try:
                strings = dataset.asstr()[index]
            except UnicodeDecodeError as e:
                strings = dataset.asstr(encoding='latin-1')[index]
                _warn_latin1_decode(dataset, strings, str(e))
            variable.values = np.asarray(strings).flatten()
        elif variable.values.flags["C_CONTIGUOUS"] and variable.values.size > 0:
            dataset.read_direct(variable.values, source_sel=index)
        else:
            variable.values = dataset[index]
        return variable

    @staticmethod
    def get_name(group: Union[h5py.Group, h5py.Dataset]) -> str:
        """
        Just the name of this group, not the full path
        """
        return group.name.split("/")[-1]

    @staticmethod
    def get_path(group: Union[h5py.Group, h5py.Dataset]) -> str:
        """
        The full path
        """
        return group.name

    @staticmethod
    def get_dtype(dataset: h5py.Dataset) -> str:
        """
        The dtype of the dataset
        """
        return dataset.dtype

    @staticmethod
    def get_shape(dataset: h5py.Dataset) -> List:
        """
        The shape of the dataset
        """
        return dataset.shape

    @staticmethod
    def get_unit(node: Union[h5py.Dataset, h5py.Group]) -> str:
        try:
            units = node.attrs["units"]
        except (AttributeError, KeyError):
            return None
        units = _ensure_str(units, LoadFromHdf5.get_attr_encoding(node, "units"))
        try:
            sc.Unit(units)
        except sc.UnitError:
            warnings.warn(f"Unrecognized unit '{units}' for value dataset "
                          f"in '{node.name}'; setting unit as 'dimensionless'")
            return "dimensionless"
        return units

    @staticmethod
    def get_child_from_group(group: Dict,
                             child_name: str) -> Union[h5py.Dataset, h5py.Group, None]:
        try:
            return group[child_name]
        except KeyError:
            return None

    @staticmethod
    def get_attr_encoding(group: h5py.Group, dataset_name: str) -> str:
        cset = h5py.h5a.open(group.id,
                             dataset_name.encode("utf-8")).get_type().get_cset()
        return _cset_to_encoding(cset)

    @staticmethod
    def get_object_by_path(group: Union[h5py.Group, h5py.File],
                           path: str) -> h5py.Dataset:
        try:
            return group[path]
        except KeyError:
            raise MissingDataset

    @staticmethod
    def is_group(node: Any):
        # Note: Not using isinstance(node, h5py.Group) so we can support other
        # libraries that look like h5py but are not, in particular data
        # adapted from `tiled`.
        return hasattr(node, 'visititems')

    @staticmethod
    def contains_stream(_):
        # HDF5 groups never contain streams.
        return False
