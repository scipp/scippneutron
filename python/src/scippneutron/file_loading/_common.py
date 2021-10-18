# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones

from dataclasses import dataclass
from typing import Union, Dict, Optional
import h5py
from ._nexus import LoadFromNexus


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


@dataclass
class NexusMeta:
    """
    Data class to encapsulate access to an open nexus file.
    """
    nexus_file: Union[h5py.File, Dict]
    nexus: LoadFromNexus
    root: Optional[str] = "/"
