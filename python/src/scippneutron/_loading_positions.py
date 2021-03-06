from _warnings import warn
from typing import List, Optional, Any
import h5py
import numpy as np
import scipp as sc
from ._loading_common import get_units


def load_position_of_unique_component(
        groups: List[h5py.Group],
        data: sc.Variable,
        name: str,
        nx_class: str,
        default_position: Optional[np.ndarray] = None):
    if len(groups) > 1:
        warn(f"More than one {nx_class} found in file, "
             f"skipping loading {name} position")
        return
    group = groups[0]
    if "distance" in group:
        position = np.array([0, 0, group["distance"][...]])
        unit_str = get_units(group["distance"])
        if not unit_str:
            warn(f"'distance' dataset in {nx_class} is missing "
                 f"units attribute, skipping loading {name} position")
            return
        units = sc.Unit(unit_str)
    elif default_position is None:
        warn(f"No position given for {name} in file")
        return
    else:
        position = np.array([0, 0, 0])
        units = sc.units.m
    _add_attr_to_loaded_data(f"{name}_position",
                             data,
                             position,
                             unit=units,
                             dtype=sc.dtype.vector_3_float64)


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
            data[attr_name] = sc.Variable(value=value, dtype=dtype, unit=unit)
        else:
            data[attr_name] = sc.Variable(value=value, unit=unit)
    except KeyError:
        pass
