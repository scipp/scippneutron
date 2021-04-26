# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from dataclasses import dataclass
from typing import Union, Dict
import h5py


class BadSource(Exception):
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
