# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from dataclasses import dataclass
from typing import Union, Dict, Optional, Any
import h5py
import scipp as sc
import numpy as np


class BadSource(Exception):
    """
    Raise if something is wrong with data source which
    prevents it being used. Warn the user.
    """
    pass


class SkipSource(Exception):
    """
    Raise to abort using the data source, do not
    warn the user.
    """
    pass


class MissingDataset(Exception):
    pass


class MissingAttribute(Exception):
    pass


@dataclass
class Group:
    """
    This class exists because h5py.Group has a "parent" property,
    but we also need to access the parent when parsing Dict
    loaded from json
    """
    group: Union[h5py.Group, Dict]
    parent: Union[h5py.Group, Dict]
    path: str
    contains_stream: bool = False


def _add_attr_to_loaded_data(attr_name: str,
                             data: sc.Variable,
                             value: np.ndarray,
                             unit: sc.Unit,
                             dtype: Optional[Any] = None):
    try:
        data = data.attrs
    except AttributeError:
        pass

    try:
        if dtype is not None:
            if dtype == sc.dtype.vector_3_float64:
                data[attr_name] = sc.vector(value=value, unit=unit)
            elif dtype == sc.dtype.matrix_3_float64:
                data[attr_name] = sc.matrix(value=value, unit=unit)
            else:
                data[attr_name] = sc.Variable(value=value, dtype=dtype, unit=unit)
        else:
            data[attr_name] = sc.Variable(value=value, unit=unit)
    except KeyError:
        pass
