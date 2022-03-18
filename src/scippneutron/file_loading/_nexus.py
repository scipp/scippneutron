from ._json_nexus import LoadFromJson
from ._hdf5_nexus import LoadFromHdf5
from typing import Union
import scipp as sc

LoadFromNexus = Union[LoadFromJson, LoadFromHdf5]
ScippData = Union[sc.Dataset, sc.DataArray, None]
