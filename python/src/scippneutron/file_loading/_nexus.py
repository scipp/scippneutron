from ._json_nexus import LoadFromJson
from ._hdf5_nexus import LoadFromHdf5
from typing import Union, Dict, Optional
from dataclasses import dataclass
import h5py
import scipp as sc

LoadFromNexus = Union[LoadFromJson, LoadFromHdf5]
GroupObject = Union[h5py.Group, Dict]
ScippData = Union[sc.Dataset, sc.DataArray, None]


@dataclass
class NexusMeta:
    """
    Data class to encapsulate access to an open nexus file.
    """
    nexus_file: Union[h5py.File, Dict]
    nexus: LoadFromNexus
    root: Optional[str] = "/"
