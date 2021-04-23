from ._loading_json_nexus import LoadFromJson
from ._loading_hdf5_nexus import LoadFromHdf5
from typing import Union, Dict
import h5py

LoadFromNexus = Union[LoadFromJson, LoadFromHdf5]
GroupObject = Union[h5py.Group, Dict]
