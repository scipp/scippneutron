from ._json_nexus import LoadFromJson
from ._hdf5_nexus import LoadFromHdf5
from typing import Union, Dict
import h5py
import scipp as sc

LoadFromNexus = Union[LoadFromJson, LoadFromHdf5]
GroupObject = Union[h5py.Group, Dict]
ScippData = Union[sc.Dataset, sc.DataArray, None]
