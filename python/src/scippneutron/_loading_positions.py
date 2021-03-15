from _warnings import warn
from typing import List, Optional, Any, Tuple
import h5py
import numpy as np
import scipp as sc
from ._loading_common import get_units
from ._loading_transformations import (get_position_from_transformations,
                                       TransformationError)


class PositionError(Exception):
    pass


def load_position_of_unique_component(
        groups: List[h5py.Group],
        data: sc.Variable,
        name: str,
        nx_class: str,
        file_root: h5py.File,
        default_position: Optional[np.ndarray] = None):
    if len(groups) > 1:
        warn(f"More than one {nx_class} found in file, "
             f"skipping loading {name} position")
        return
    try:
        position, units = _get_position_of_component(groups[0], name, nx_class,
                                                     file_root,
                                                     default_position)
    except PositionError:
        return
    _add_attr_to_loaded_data(f"{name}_position",
                             data,
                             position,
                             unit=units,
                             dtype=sc.dtype.vector_3_float64)


def load_positions_of_components(
        groups: List[h5py.Group],
        data: sc.Variable,
        name: str,
        nx_class: str,
        file_root: h5py.File,
        default_position: Optional[np.ndarray] = None):
    for group in groups:
        try:
            position, units = _get_position_of_component(
                group, name, nx_class, file_root, default_position)
        except PositionError:
            continue
        if len(groups) == 1:
            _add_attr_to_loaded_data(f"{name}_position",
                                     data,
                                     position,
                                     unit=units,
                                     dtype=sc.dtype.vector_3_float64)
        else:
            _add_attr_to_loaded_data(f"{group.name.split('/')[-1]}_position",
                                     data,
                                     position,
                                     unit=units,
                                     dtype=sc.dtype.vector_3_float64)


def _get_position_of_component(
    group: h5py.Group,
    name: str,
    nx_class: str,
    file_root: h5py.File,
    default_position: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, sc.Unit]:
    if "depends_on" in group:
        try:
            position = get_position_from_transformations(group, file_root)
        except TransformationError as e:
            warn(f"Skipping loading {name} position due to error: {e}")
            raise PositionError
        units = sc.units.m
    elif "distance" in group:
        position = np.array([0, 0, group["distance"][...]])
        unit_str = get_units(group["distance"])
        if not unit_str:
            warn(f"'distance' dataset in {nx_class} is missing "
                 f"units attribute, skipping loading {name} position")
            raise PositionError
        units = sc.Unit(unit_str)
    elif default_position is None:
        warn(f"No position given for {name} in file")
        raise PositionError
    else:
        position = np.array([0, 0, 0])
        units = sc.units.m

    return position, units


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
