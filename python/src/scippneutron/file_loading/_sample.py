from typing import List, Tuple
import numpy as np
import scipp as sc
from ._common import Group
from ._nexus import LoadFromNexus, GroupObject


def _get_matrix_of_component(group: GroupObject, nexus: LoadFromNexus,
                             name: str) -> Tuple[np.ndarray, sc.Unit]:
    matrix_found, _ = nexus.dataset_in_group(group, name)
    if matrix_found:
        matrix = nexus.load_dataset_from_group_as_numpy_array(group, name)
        return matrix
    else:
        return None


def _get_ub_of_component(group: GroupObject,
                         nexus: LoadFromNexus) -> Tuple[np.ndarray, sc.Unit]:
    return _get_matrix_of_component(group, nexus, "ub_matrix"), sc.units.angstrom**-1


def _get_u_of_component(group: GroupObject,
                        nexus: LoadFromNexus) -> Tuple[np.ndarray, sc.Unit]:
    return _get_matrix_of_component(group, nexus, "orientation_matrix"), sc.units.one


def load_ub_matrices_of_components(groups: List[Group], name: str,
                                   nexus: LoadFromNexus):
    ub_matrices = {}
    properties = {"ub_matrix": _get_ub_of_component, "u_matrix": _get_u_of_component}
    for sc_property, extractor in properties.items():
        for group in groups:
            matrix, units = extractor(group.group, nexus)
            if matrix is None:
                continue

            if len(groups) == 1:
                name = f"{name}_{sc_property}"
            else:
                name = f"{nexus.get_name(group.group)}_{sc_property}"

            ub_matrices[name] = sc.matrix(value=matrix, unit=units)
    return ub_matrices
