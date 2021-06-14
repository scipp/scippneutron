from _warnings import warn
from typing import List, Tuple, Union
import h5py
import numpy as np
import scipp as sc
from ._common import Group, _add_attr_to_loaded_data
from ._nexus import LoadFromNexus, GroupObject


def _get_matrix_of_component(group: GroupObject, nx_class: str,
                             nexus: LoadFromNexus,
                             name: str) -> Tuple[np.ndarray, sc.Unit]:
    matrix_found, _ = nexus.dataset_in_group(group, name)
    matrix_unit_found, _ = nexus.dataset_in_group(group, f"{name}_units")
    if matrix_found:
        matrix = nexus.load_dataset_from_group_as_numpy_array(group, name)
        units = nexus.get_unit(nexus.get_dataset_from_group(group, name))
        if units == sc.units.dimensionless:
            warn(f"'{name}' dataset in {nx_class} is missing units")
        return matrix, units
    else:
        return None, None


def _get_ub_of_component(group: GroupObject, nx_class: str,
                         nexus: LoadFromNexus) -> Tuple[np.ndarray, sc.Unit]:
    return _get_matrix_of_component(group, nx_class, nexus, "ub_matrix")


def _get_u_of_component(group: GroupObject, nx_class: str,
                        nexus: LoadFromNexus) -> Tuple[np.ndarray, sc.Unit]:
    return _get_matrix_of_component(group, nx_class, nexus,
                                    "orientation_matrix")


def load_ub_matrices_of_components(groups: List[Group],
                                   data: Union[sc.DataArray, sc.Dataset],
                                   name: str, nx_class: str,
                                   file_root: h5py.File, nexus: LoadFromNexus):
    properties = {
        "ub_matrix": _get_ub_of_component,
        "u_matrix": _get_u_of_component
    }
    for sc_property, extractor in properties.items():
        for group in groups:
            matrix, units = extractor(group.group, nx_class, nexus)
            if matrix is None:
                continue
            if len(groups) == 1:
                _add_attr_to_loaded_data(f"{name}_{sc_property}",
                                         data,
                                         matrix,
                                         unit=units,
                                         dtype=sc.dtype.matrix_3_float64)
            else:
                _add_attr_to_loaded_data(
                    f"{nexus.get_name(group.group)}_{sc_property}",
                    data,
                    matrix,
                    unit=units,
                    dtype=sc.dtype.matrix_3_float64)
