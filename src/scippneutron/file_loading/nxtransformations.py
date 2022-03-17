# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
import warnings

import numpy as np
from ._common import MissingDataset, MissingAttribute, Group
from typing import Union, List
import scipp as sc
import scipp.spatial
import scipp.interpolate
import h5py
from cmath import isclose
from ._nexus import LoadFromNexus, GroupObject
from ._json_nexus import contains_stream
from .nxlog import NXlog
from .nxobject import Field, NXobject, ScippIndex


class TransformationError(Exception):
    pass


class Transformation:
    def __init__(self, obj: Field):  # could be an NXlog
        self._obj = obj

    @property
    def attrs(self):
        return self._obj.attrs

    @property
    def name(self):
        return self._obj.name

    @property
    def depends_on(self):
        if (path := self.attrs.get('depends_on')) is not None:
            if path.startswith('/'):
                return Transformation(self._obj.file[path])
            elif path != '.':
                return Transformation(self._obj.parent[path])
        return None

    @property
    def offset(self):
        if (offset := self.attrs.get('offset')) is None:
            return None
        if (offset_units := self.attrs.get('offset_units')) is None:
            raise TransformationError(
                f"Found {offset=} but no corresponding 'offset_units' "
                f"attribute at {self.name}")
        return sc.spatial.translation(value=offset, unit=offset_units)

    @property
    def vector(self) -> sc.Variable:
        return sc.vector(value=self.attrs.get('vector'))

    def __getitem__(self, select: ScippIndex):
        transformation_type = self.attrs.get('transformation_type')
        t = self._obj[select] * self.vector
        v = t if isinstance(t, sc.Variable) else t.data
        if transformation_type == 'translation':
            v = v.to(unit='m', copy=False)
            v = sc.spatial.translations(dims=v.dims, values=v.values, unit=v.unit)
        elif transformation_type == 'rotation':
            v = sc.spatial.rotations_from_rotvecs(v)
        else:
            raise TransformationError(
                f"{transformation_type=} attribute at {self.name},"
                " expected 'translation' or 'rotation'.")
        if isinstance(t, sc.Variable):
            t = v
        else:
            t.data = v
        if (offset := self.offset) is None:
            return t
        offset = sc.vector(value=offset.values, unit=offset.unit).to(unit='m')
        offset = sc.spatial.translation(value=offset.value, unit=offset.unit)
        return t * offset


def _interpolate_transform(transform, xnew):
    # scipy can't interpolate with a single value
    if transform.sizes["time"] == 1:
        transform = sc.concat([transform, transform], dim="time")

    transform = sc.interpolate.interp1d(transform,
                                        "time",
                                        kind="previous",
                                        fill_value="extrapolate")(xnew=xnew)

    return transform


def get_full_transformation_matrix(group: Group, nexus: LoadFromNexus) -> sc.DataArray:
    """
    Get the 4x4 transformation matrix for a component, resulting
    from the full chain of transformations linked by "depends_on"
    attributes

    :param group: The HDF5 group of the component, containing depends_on
    :param nexus: wrap data access to hdf file or objects from json
    :return: 4x4 active transformation matrix as a data array
    """
    transformations = []
    try:
        depends_on = nexus.load_scalar_string(group, "depends_on")
    except MissingDataset:
        depends_on = '.'
    _get_transformations(depends_on, transformations, group, nexus.get_name(group),
                         nexus)

    total_transform = sc.spatial.affine_transform(value=np.identity(4), unit=sc.units.m)

    for transform in transformations:
        if isinstance(total_transform, sc.DataArray) and isinstance(
                transform, sc.DataArray):
            xnew = sc.datetimes(values=np.unique(
                sc.concat([
                    total_transform.coords["time"].to(unit=sc.units.ns, copy=False),
                    transform.coords["time"].to(unit=sc.units.ns, copy=False),
                ],
                          dim="time").values),
                                dims=["time"],
                                unit=sc.units.ns)
            total_transform = _interpolate_transform(
                transform, xnew) * _interpolate_transform(total_transform, xnew)
        else:
            total_transform = transform * total_transform
    if isinstance(total_transform, sc.DataArray):
        time_dependent = [t for t in transformations if isinstance(t, sc.DataArray)]
        times = [da.coords['time'][0] for da in time_dependent]
        latest_log_start = sc.reduce(times).max()
        return total_transform['time', latest_log_start:].copy()
    return total_transform


def _transformation_is_nx_log_stream(transform: Union[h5py.Dataset, GroupObject],
                                     nexus: LoadFromNexus):
    # Stream objects are only in the dict loaded from json
    if isinstance(transform, dict):
        # If transform is a group and contains a stream but not a value dataset
        # then assume it is a streamed NXlog transformation
        try:
            if nexus.is_group(transform):
                found_value_dataset, _ = nexus.dataset_in_group(transform, "value")
                if not found_value_dataset and contains_stream(transform):
                    return True
        except KeyError:
            pass
    return False


def _get_transformations(transform_path: str, transformations: List[np.ndarray],
                         group: Group, group_name: str, nexus: LoadFromNexus):
    """
    Get all transformations in the depends_on chain.

    :param transform_path: The first depends_on path string
    :param transformations: List of transformations to populate
    :param root: root of the file, depends_on paths assumed to be
      relative to this
    """
    if transform_path == '.':
        return

    g = NXobject(group, nexus)
    if transform_path.startswith('/'):
        t = g.file[transform_path]
    else:
        t = g[transform_path]
    if (not isinstance(t, (Field, NXobject))) or _transformation_is_nx_log_stream(
            t._group if isinstance(t, NXobject) else t._dataset, t._loader):
        warnings.warn("Streamed NXlog found in transformation "
                      "chain, getting its value from stream is "
                      "not yet implemented and instead it will be "
                      "treated as a 0-distance translation")
        transformations.append(
            sc.spatial.affine_transform(value=np.identity(4, dtype=float),
                                        unit=sc.units.m))
        return
    t = Transformation(t)
    while t is not None:
        transformations.append(t[()])
        t = t.depends_on
    return

    # TODO: this list of transformation should probably be cached in the future
    # to deal with changing beamline components (e.g. pixel positions) during a
    # live data stream (see https://github.com/scipp/scippneutron/issues/76).

    if transform_path != '.':
        try:
            transform = nexus.get_object_by_path(group.file, transform_path)
        except MissingDataset:
            raise TransformationError(
                f"Non-existent depends_on path '{transform_path}' found "
                f"in transformations chain for {group_name}")
        next_depends_on = _append_transformation(transform, transformations, group_name,
                                                 nexus)

        if not next_depends_on == "." and not next_depends_on.startswith("/"):
            # Path is relative - convert it to an absolute path relative to the parent
            # of the transform it was loaded from.
            parent = "/".join(nexus.get_path(transform).split("/")[:-1])
            next_depends_on = f"{parent}/{next_depends_on}"

        _get_transformations(next_depends_on, transformations, group, group_name, nexus)


def _append_transformation(transform: Union[h5py.Dataset, GroupObject],
                           transformations: List[np.ndarray], group_name: str,
                           nexus: LoadFromNexus) -> str:
    if _transformation_is_nx_log_stream(transform, nexus):
        warnings.warn("Streamed NXlog found in transformation "
                      "chain, getting its value from stream is "
                      "not yet implemented and instead it will be "
                      "treated as a 0-distance translation")
        transformations.append(
            sc.spatial.affine_transform(value=np.identity(4, dtype=float),
                                        unit=sc.units.m))
    else:
        try:
            vector = nexus.get_attribute_as_numpy_array(transform,
                                                        "vector").astype(float)
            vector = _normalize(vector, nexus.get_name(transform))
        except MissingAttribute:
            raise TransformationError(f"Missing 'vector' attribute in transformation "
                                      f"at {nexus.get_name(transform)}")

        try:
            offset = nexus.get_attribute_as_numpy_array(transform,
                                                        "offset").astype(float)
            try:
                offset_unit = nexus.get_string_attribute(transform, "offset_units")
            except MissingAttribute:
                raise TransformationError(
                    f"Found offset={offset} but no corresponding 'offset_units' "
                    f"attribute at {nexus.get_name(transform)}")
            offset = sc.vector(value=offset, unit=offset_unit)
        except MissingAttribute:
            offset = sc.vector(value=np.array([0., 0., 0.], dtype=float),
                               unit=sc.units.m)

        transform_type = nexus.get_string_attribute(transform, "transformation_type")
        if transform_type == 'translation':
            _append_translation(offset, transform, transformations, vector, group_name,
                                nexus)
        elif transform_type == 'rotation':
            _append_rotation(offset, transform, transformations, vector, group_name,
                             nexus)
        else:
            raise TransformationError(f"Unknown transformation type "
                                      f"'{transform_type}'"
                                      f" at {nexus.get_name(transform)}")
    try:
        depends_on = nexus.get_string_attribute(transform, "depends_on")
    except MissingAttribute:
        depends_on = "."
    return depends_on


def _normalize(vector: np.ndarray, transform_name: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if isclose(norm, 0.):
        raise TransformationError(
            f"Magnitude of 'vector' attribute in transformation at "
            f"{transform_name} is too close to zero")
    return vector / norm


def _append_translation(offset: sc.Variable, transform: GroupObject,
                        transformations: List[np.ndarray],
                        direction_unit_vector: np.ndarray, group_name: str,
                        nexus: LoadFromNexus):
    loaded_transform = _get_transformation_magnitude_and_unit(
        group_name, transform, nexus)

    loaded_transform_m = loaded_transform.to(dtype=sc.DType.float64,
                                             unit=sc.units.m,
                                             copy=False)

    vectors = sc.vector(value=direction_unit_vector) * loaded_transform_m + offset
    translations = sc.spatial.translations(dims=loaded_transform_m.dims,
                                           values=vectors.values,
                                           unit=sc.units.m)

    if isinstance(loaded_transform, sc.DataArray):
        t = sc.DataArray(data=translations,
                         coords={"time": loaded_transform.coords["time"]})
    else:
        t = translations

    transformations.append(t)


def _get_transformation_magnitude_and_unit(group_name: str,
                                           transform: Union[h5py.Dataset, GroupObject],
                                           nexus: LoadFromNexus) -> sc.DataArray:
    """
    Gets a scipp data array containing magnitudes and timestamps of a transformation.
    """
    if nexus.is_group(transform):
        try:
            log = NXlog(transform, nexus)[()]
        except sc.DimensionError:
            raise TransformationError(f"Mismatched time and value dataset lengths "
                                      f"for transformation at {group_name}")
        except MissingDataset:
            raise TransformationError(
                f"Encountered {nexus.get_name(transform)} in transformation "
                f"chain for {group_name} but it is a group without a value "
                "dataset; not a valid transformation")
    else:
        magnitude = nexus.load_dataset_as_numpy_array(transform).astype(float).item()
        unit = nexus.get_unit(transform)
        log = sc.scalar(value=magnitude, unit=unit, dtype=sc.DType.float64)

    if log.unit is None:
        raise TransformationError(f"Missing units for transformation at "
                                  f"{nexus.get_name(transform)}")
    return log


def _append_rotation(offset: sc.Variable, transform: GroupObject,
                     transformations: List[np.ndarray], rotation_axis: np.ndarray,
                     group_name: str, nexus: LoadFromNexus):
    angles = _get_transformation_magnitude_and_unit(group_name, transform, nexus)
    try:
        angles = angles.to(dtype=sc.DType.float64, unit=sc.units.rad, copy=False)
    except sc.UnitError:
        raise TransformationError(f"Unit for rotation transformation must be radians "
                                  f"or degrees, problem in {transform.name}")

    offset = sc.spatial.translation(value=offset.values, unit=offset.unit)

    rotvecs = sc.vector(value=rotation_axis) * angles.astype('float64', copy=False)
    rotations = sc.spatial.rotations_from_rotvecs(rotvecs) * offset

    if isinstance(angles, sc.DataArray):
        t = sc.DataArray(data=rotations, coords={"time": angles.coords["time"]})
    else:
        t = rotations

    transformations.append(t)
