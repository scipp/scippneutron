from _warnings import warn
from typing import List, Optional, Tuple
import h5py
import numpy as np
import scipp as sc
from ._common import Group, _add_attr_to_loaded_data
from ._nexus import LoadFromNexus, GroupObject


def _get_ub_of_component(
        group: GroupObject,
        nx_class: str,
        nexus: LoadFromNexus) -> Tuple[np.ndarray, sc.Unit]:
    ub_matrix_found, _ = nexus.dataset_in_group(group, "ub_matrix")
    ub_matrix_unit_found, _ = nexus.dataset_in_group(group, "ub_matrix_units")
    if ub_matrix_found:
        ub_matrix = nexus.load_dataset_from_group_as_numpy_array(
            group, "ub_matrix")
        units = nexus.get_unit(nexus.get_dataset_from_group(
            group, "ub_matrix"))
        if units == sc.units.dimensionless:
            warn(f"'ub_matrix' dataset in {nx_class} is missing ")
        return ub_matrix, units
    else:
        return None, None


def load_ub_matrices_of_components(groups: List[Group],
                                   data: sc.Variable,
                                   name: str,
                                   nx_class: str,
                                   file_root: h5py.File,
                                   nexus: LoadFromNexus):
    for group in groups:
        ub_matrix, units = _get_ub_of_component(group.group, nx_class, nexus)
        if ub_matrix is None:
            return
        if len(groups) == 1:
            _add_attr_to_loaded_data(f"{name}_ub_matrix",
                                     data,
                                     ub_matrix,
                                     unit=units,
                                     dtype=sc.dtype.matrix_3_float64)
        else:
            _add_attr_to_loaded_data(
                f"{nexus.get_name(group.group)}_ub_matrix",
                data,
                ub_matrix,
                unit=units,
                dtype=sc.dtype.matrix_3_float64)
