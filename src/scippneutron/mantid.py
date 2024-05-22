# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock, Neil Vaytet

import os
import re
import uuid
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import scipp as sc
from scipp.core.util import VisibleDeprecationWarning

from ._utils import get_attrs


@contextmanager
def run_mantid_alg(alg, *args, **kwargs):
    try:
        from mantid import simpleapi as mantid
        from mantid.api import AnalysisDataService
    except ImportError:
        raise ImportError(
            "Mantid Python API was not found, please install Mantid framework "
            "as detailed in the installation instructions ("
            "https://scipp.github.io/scippneutron/getting-started/"
            "installation.html)"
        ) from None
    # Deal with multiple calls to this function, which may have conflicting
    # names in the global AnalysisDataService by using uuid.
    ws_name = f'scipp.run_mantid_alg.{uuid.uuid4()}'
    # Deal with non-standard ways to define the prefix of output workspaces
    if alg == 'Fit':
        kwargs['Output'] = ws_name
    elif alg == 'LoadDiffCal':
        kwargs['WorkspaceName'] = ws_name
    else:
        kwargs['OutputWorkspace'] = ws_name
    ws = getattr(mantid, alg)(*args, **kwargs)
    try:
        yield ws
    finally:
        for name in AnalysisDataService.Instance().getObjectNames():
            if name.startswith(ws_name):
                mantid.DeleteWorkspace(name)


def get_pos(pos):
    return (
        None
        if pos is None
        else sc.vector(value=[pos.X(), pos.Y(), pos.Z()], unit=sc.units.m)
    )


def make_run(ws):
    return sc.scalar(deepcopy(ws.run()))


additional_unit_mapping = {
    " ": sc.units.one,
    "none": sc.units.one,
}


def process_run_logs(ws):
    for property_name in ws.run().keys():
        units_string = ws.run()[property_name].units
        try:
            unit = additional_unit_mapping.get(units_string, sc.Unit(units_string))
        except RuntimeError:  # TODO catch UnitError once exposed from C++
            # Parsing unit string failed
            unit = None

        values = deepcopy(ws.run()[property_name].value)

        if units_string and unit is None:
            warnings.warn(
                f"Workspace run log '{property_name}' "
                f"has unrecognised units: '{units_string}'",
                stacklevel=4,
            )
        if unit is None:
            unit = sc.units.one

        try:
            times = deepcopy(ws.run()[property_name].times)
            is_time_series = True
            dimension_label = "time"
        except AttributeError:
            times = None
            is_time_series = False
            dimension_label = property_name

        if np.isscalar(values):
            property_data = sc.scalar(values, unit=unit)
        else:
            property_data = sc.array(values=values, unit=unit, dims=[dimension_label])

        if is_time_series:
            # If property has timestamps, create a DataArray
            data_array = sc.DataArray(
                data=property_data,
                coords={
                    dimension_label: sc.array(dims=[dimension_label], values=times)
                },
            )
            yield property_name, data_array
        else:
            yield property_name, property_data


def make_mantid_sample(ws):
    return sc.scalar(deepcopy(ws.sample()))


def make_sample_ub(ws):
    # B matrix transforms the h,k,l triplet into a Cartesian system
    # https://docs.mantidproject.org/nightly/concepts/Lattice.html
    return sc.spatial.linear_transform(
        value=ws.sample().getOrientedLattice().getUB(), unit=sc.units.angstrom**-1
    )


def make_sample_u(ws):
    # U matrix rotation for sample alignment
    # https://docs.mantidproject.org/nightly/concepts/Lattice.html
    return sc.spatial.linear_transform(value=ws.sample().getOrientedLattice().getU())


def make_component_info(ws):
    component_info = ws.componentInfo()

    if component_info.hasSource():
        sourcePos = component_info.sourcePosition()
    else:
        sourcePos = None

    if component_info.hasSample():
        samplePos = component_info.samplePosition()
    else:
        samplePos = None

    return get_pos(sourcePos), get_pos(samplePos)


def make_detector_info(ws, spectrum_dim):
    det_info = ws.detectorInfo()
    # det -> spec mapping
    nDet = det_info.size()
    spectrum = np.empty(shape=(nDet,), dtype=np.int32)
    has_spectrum = np.full((nDet,), False)
    spec_info = ws.spectrumInfo()
    for i, spec in enumerate(spec_info):
        spec_def = spec.spectrumDefinition
        for j in range(len(spec_def)):
            det, time = spec_def[j]
            if time != 0:
                raise RuntimeError(
                    "Conversion of Mantid Workspace with scanning instrument "
                    "not supported yet."
                )
            spectrum[det] = i
            has_spectrum[det] = True

    # Store only information about detectors with data (a spectrum). The rest
    # mostly just gets in the way and including it in the default converter
    # is probably not required.
    spectrum = sc.array(dims=['detector'], values=spectrum[has_spectrum], unit=None)
    detector = sc.array(
        dims=['detector'], values=det_info.detectorIDs()[has_spectrum], unit=None
    )

    # May want to include more information here, such as detector positions,
    # but for now this is not necessary.

    return sc.Dataset(coords={'detector': detector, spectrum_dim: spectrum})


def md_dimension(mantid_dim, index):
    # Look for q dimensions
    patterns = [f"^q.*{coord}$" for coord in ['x', 'y', 'z']]
    q_dims = ['Q_x', 'Q_y', 'Q_z']
    pattern_result = zip(patterns, q_dims, strict=True)
    if mantid_dim.getMDFrame().isQ():
        for pattern, result in pattern_result:
            if re.search(pattern, mantid_dim.name, re.IGNORECASE):
                return result

    # Look for common/known mantid dimensions
    patterns = ["DeltaE", "T"]
    dims = ['energy_transfer', 'temperature']
    pattern_result = zip(patterns, dims, strict=True)
    for pattern, result in pattern_result:
        if re.search(pattern, mantid_dim.name, re.IGNORECASE):
            return result

    # Look for common spatial dimensions
    patterns = [f"^{coord}$" for coord in ['x', 'y', 'z']]
    dims = ['x', 'y', 'z']
    pattern_result = zip(patterns, dims, strict=True)
    for pattern, result in pattern_result:
        if re.search(pattern, mantid_dim.name, re.IGNORECASE):
            return result

    raise ValueError(
        f"Cannot infer scipp dimension from input mantid dimension {mantid_dim.name()}"
    )


def md_unit(frame):
    known_md_units = {
        "Angstrom^-1": sc.units.dimensionless / sc.units.angstrom,
        "r.l.u": sc.units.dimensionless,
        "T": sc.units.K,
        "DeltaE": sc.units.meV,
    }
    if frame.getUnitLabel().ascii() in known_md_units:
        return known_md_units[frame.getUnitLabel().ascii()]
    else:
        return sc.units.dimensionless


def validate_and_get_unit(unit, allow_empty=False):
    if hasattr(unit, 'unitID'):
        if unit.unitID() == 'Label':
            unit = unit.caption()
        else:
            unit = unit.unitID()
    known_units = {
        "DeltaE": ['energy_transfer', sc.units.meV],
        "TOF": ['tof', sc.units.us],
        "Wavelength": ['wavelength', sc.units.angstrom],
        "Energy": ['energy', sc.units.meV],
        "dSpacing": ['dspacing', sc.units.angstrom],
        "MomentumTransfer": ['Q', sc.units.dimensionless / sc.units.angstrom],
        "QSquared": [
            'Q^2',
            sc.units.dimensionless / (sc.units.angstrom * sc.units.angstrom),
        ],
        "Spectrum": ['spectrum', None],
        "Empty": ['empty', sc.units.dimensionless],
        "Counts": ['counts', sc.units.counts],
    }

    if unit not in known_units.keys():
        return [str(unit), sc.units.dimensionless]
    else:
        return known_units[unit]


def _to_spherical(pos, output):
    output["r"] = sc.sqrt(sc.dot(pos, pos))
    output["t"] = sc.acos(pos.fields.z / output["r"].data)
    signed_phi = sc.atan2(y=pos.fields.y, x=pos.fields.x)
    abs_phi = sc.abs(signed_phi)
    output["p-delta"] = (
        np.pi * sc.units.rad
    ) - abs_phi  # angular delta (magnitude) from pole
    output['p-sign'] = signed_phi  # weighted sign of phi
    return output


def _rot_from_vectors(vec1, vec2):
    a = sc.vector(value=vec1.value / np.linalg.norm(vec1.value))
    b = sc.vector(value=vec2.value / np.linalg.norm(vec2.value))
    c = sc.vector(value=np.cross(a.value, b.value))
    angle = sc.acos(sc.dot(a, b)).value
    return sc.spatial.rotation(
        value=[*(c.value * np.sin(angle / 2)), np.cos(angle / 2)]
    )


def get_detector_pos(ws, spectrum_dim):
    nHist = ws.getNumberHistograms()
    pos = np.zeros([nHist, 3])

    spec_info = ws.spectrumInfo()
    for i in range(nHist):
        if spec_info.hasDetectors(i):
            p = spec_info.position(i)
            pos[i, 0] = p.X()
            pos[i, 1] = p.Y()
            pos[i, 2] = p.Z()
        else:
            pos[i, :] = [np.nan, np.nan, np.nan]
    return sc.vectors(dims=[spectrum_dim], values=pos, unit=sc.units.m)


def get_detector_properties(
    ws, source_pos, sample_pos, spectrum_dim, advanced_geometry=False
):
    if not advanced_geometry:
        return (get_detector_pos(ws, spectrum_dim), None, None)
    spec_info = ws.spectrumInfo()
    det_info = ws.detectorInfo()
    comp_info = ws.componentInfo()
    nspec = len(spec_info)
    det_rot_quaternions = np.zeros([nspec, 4])
    det_bbox = np.zeros([nspec, 3])

    if sample_pos is not None and source_pos is not None:
        total_detectors = spec_info.detectorCount()
        act_beam = sample_pos - source_pos
        rot = _rot_from_vectors(act_beam, sc.vector(value=[0, 0, 1]))
        inv_rot = _rot_from_vectors(sc.vector(value=[0, 0, 1]), act_beam)

        # Create empty to hold position info for all spectra detectors
        pos_d = sc.Dataset(
            {"x": sc.zeros(dims=["detector"], shape=[total_detectors], unit=sc.units.m)}
        )
        pos_d["y"] = sc.zeros_like(pos_d["x"])
        pos_d["z"] = sc.zeros_like(pos_d["x"])
        pos_d.coords[spectrum_dim] = sc.array(
            dims=["detector"], values=np.empty(total_detectors)
        )

        spectrum_values = pos_d.coords[spectrum_dim].values

        x_values = pos_d["x"].values
        y_values = pos_d["y"].values
        z_values = pos_d["z"].values

        idx = 0
        for i, spec in enumerate(spec_info):
            if spec.hasDetectors:
                definition = spec_info.getSpectrumDefinition(i)
                n_dets = len(definition)
                quats = []
                bboxes = []
                for j in range(n_dets):
                    det_idx = definition[j][0]
                    p = det_info.position(det_idx)
                    r = det_info.rotation(det_idx)
                    spectrum_values[idx] = i
                    x_values[idx] = p.X()
                    y_values[idx] = p.Y()
                    z_values[idx] = p.Z()
                    idx += 1
                    quats.append(np.array([r.imagI(), r.imagJ(), r.imagK(), r.real()]))
                    if comp_info.hasValidShape(det_idx):
                        s = comp_info.shape(det_idx)
                        bboxes.append(s.getBoundingBox().width())
                det_rot_quaternions[i] = np.mean(quats, axis=0)
                det_bbox[i, :] = np.sum(bboxes, axis=0)

        rot_pos = rot * sc.spatial.as_vectors(
            pos_d["x"].data, pos_d["y"].data, pos_d["z"].data
        )

        _to_spherical(rot_pos, pos_d)

        averaged = sc.groupby(
            pos_d,
            spectrum_dim,
            bins=sc.Variable(
                dims=[spectrum_dim], values=np.arange(-0.5, len(spec_info) + 0.5, 1.0)
            ),
        ).mean("detector")

        sign = averaged["p-sign"].data / sc.abs(averaged["p-sign"].data)
        averaged["p"] = sign * ((np.pi * sc.units.rad) - averaged["p-delta"].data)
        averaged["x"] = (
            averaged["r"].data * sc.sin(averaged["t"].data) * sc.cos(averaged["p"].data)
        )
        averaged["y"] = (
            averaged["r"].data * sc.sin(averaged["t"].data) * sc.sin(averaged["p"].data)
        )
        averaged["z"] = averaged["r"].data * sc.cos(averaged["t"].data)

        pos = sc.spatial.as_vectors(
            averaged["x"].data, averaged["y"].data, averaged["z"].data
        )

        return (
            inv_rot * pos,
            sc.spatial.rotations(dims=[spectrum_dim], values=det_rot_quaternions),
            sc.vectors(dims=[spectrum_dim], values=det_bbox, unit=sc.units.m),
        )
    else:
        pos = np.zeros([nspec, 3])

        for i, spec in enumerate(spec_info):
            if spec.hasDetectors:
                definition = spec_info.getSpectrumDefinition(i)
                n_dets = len(definition)
                vec3s = []
                quats = []
                bboxes = []
                for j in range(n_dets):
                    det_idx = definition[j][0]
                    p = det_info.position(det_idx)
                    r = det_info.rotation(det_idx)
                    vec3s.append([p.X(), p.Y(), p.Z()])
                    quats.append(np.array([r.imagI(), r.imagJ(), r.imagK(), r.real()]))
                    if comp_info.hasValidShape(det_idx):
                        s = comp_info.shape(det_idx)
                        bboxes.append(s.getBoundingBox().width())
                pos[i, :] = np.mean(vec3s, axis=0)
                det_rot_quaternions[i] = np.mean(quats, axis=0)
                det_bbox[i, :] = np.sum(bboxes, axis=0)
            else:
                pos[i, :] = [np.nan, np.nan, np.nan]
                det_rot_quaternions[i] = [np.nan, np.nan, np.nan, np.nan]
                det_bbox[i, :] = [np.nan, np.nan, np.nan]
        return (
            sc.vectors(dims=[spectrum_dim], values=pos, unit=sc.units.m),
            sc.spatial.rotations(dims=[spectrum_dim], values=det_rot_quaternions),
            sc.vectors(
                dims=[spectrum_dim],
                values=det_bbox,
                unit=sc.units.m,
            ),
        )


def _get_dtype_from_values(values, coerce_floats_to_ints):
    if coerce_floats_to_ints and np.all(np.mod(values, 1.0) == 0.0):
        dtype = sc.DType.int32
    elif hasattr(values, 'dtype'):
        dtype = values.dtype
    else:
        if len(values) > 0:
            dtype = type(values[0])
            if dtype is str:
                dtype = sc.DType.string
            elif dtype is int:
                dtype = sc.DType.int64
            elif dtype is float:
                dtype = sc.DType.float64
            else:
                raise RuntimeError(
                    "Cannot handle the dtype that this " "workspace has on Axis 1."
                )
        else:
            raise RuntimeError(
                "Axis 1 of this workspace has no values. " "Cannot determine dtype."
            )
    return dtype


def init_spec_axis(ws):
    axis = ws.getAxis(1)
    dim, unit = validate_and_get_unit(axis.getUnit())
    values = axis.extractValues()
    dtype = _get_dtype_from_values(values, dim == 'spectrum')
    return dim, sc.array(dims=[dim], values=values, unit=unit, dtype=dtype)


def set_bin_masks(bin_masks, dim, index, masked_bins):
    for masked_bin in masked_bins:
        bin_masks['spectrum', index][dim, masked_bin].value = True


def _as_dict_of_variables(d: dict[str, Any]) -> dict[str, sc.Variable]:
    return {
        key: val if isinstance(val, sc.Variable) else sc.scalar(val)
        for key, val in d.items()
    }


def _convert_MatrixWorkspace_info(ws, advanced_geometry=False, load_run_logs=True):
    from mantid.kernel import DeltaEModeType

    common_bins = ws.isCommonBins()
    dim, unit = validate_and_get_unit(ws.getAxis(0).getUnit())
    source_pos, sample_pos = make_component_info(ws)
    spec_dim, spec_coord = init_spec_axis(ws)
    pos, rot, shp = get_detector_properties(
        ws, source_pos, sample_pos, spec_dim, advanced_geometry=advanced_geometry
    )

    coords = {spec_dim: spec_coord}
    # possible x - coord
    if not ws.id() == 'MaskWorkspace':
        if common_bins:
            coords[dim] = sc.Variable(dims=[dim], values=ws.readX(0), unit=unit)
        else:
            coords[dim] = sc.Variable(
                dims=[spec_dim, dim], values=ws.extractX(), unit=unit
            )

    info = {
        "coords": coords,
        "masks": {},
        "attrs": {
            "sample": make_mantid_sample(ws),
            "instrument_name": ws.componentInfo().name(ws.componentInfo().root()),
        },
    }

    if load_run_logs:
        for run_log_name, run_log in process_run_logs(ws):
            info["attrs"][run_log_name] = run_log

    if advanced_geometry:
        info["coords"]["detector_info"] = make_detector_info(ws, spec_dim)

    if not np.all(np.isnan(pos.values)):
        info["coords"].update({"position": pos})

    if rot is not None and shp is not None and not np.all(np.isnan(pos.values)):
        info["attrs"].update({"rotation": rot, "shape": shp})

    if source_pos is not None:
        info["coords"]["source_position"] = source_pos

    if sample_pos is not None:
        info["coords"]["sample_position"] = sample_pos

    if ws.detectorInfo().hasMaskedDetectors():
        spectrum_info = ws.spectrumInfo()
        mask = np.array(
            [spectrum_info.isMasked(i) for i in range(ws.getNumberHistograms())]
        )
        info["masks"][spec_dim] = sc.Variable(dims=[spec_dim], values=mask)

    if ws.getEMode() == DeltaEModeType.Direct:
        info["coords"]["incident_energy"] = _extract_einitial(ws)
    elif ws.getEMode() == DeltaEModeType.Indirect:
        info["coords"]["final_energy"] = _extract_efinal(ws, spec_dim)

    if ws.sample().hasOrientedLattice():
        info["attrs"].update(
            {"sample_ub": make_sample_ub(ws), "sample_u": make_sample_u(ws)}
        )
    return info


def convert_monitors_ws(ws, converter, **ignored):
    spec_dim, spec_coord = init_spec_axis(ws)
    spec_info = ws.spectrumInfo()
    comp_info = ws.componentInfo()
    monitors = []
    spec_indices = (
        (ws.getIndexFromSpectrumNumber(int(i)), i) for i in spec_coord.values
    )
    for index, number in spec_indices:
        definition = spec_info.getSpectrumDefinition(index)
        if not definition.size() == 1:
            raise RuntimeError("Cannot deal with grouped monitor detectors")
        det_index = definition[0][0]  # Ignore time index
        # We only ExtractSpectra for compatibility with
        # existing convert_Workspace2D_to_dataarray. This could instead be
        # refactored if found to be slow
        with run_mantid_alg(
            'ExtractSpectra', InputWorkspace=ws, WorkspaceIndexList=[index]
        ) as monitor_ws:
            # Run logs are already loaded in the data workspace
            single_monitor = converter(monitor_ws, load_run_logs=False)
        # Storing sample_position as an aligned coord of monitors means that monitor
        # data cannot be combined with scattered data even after conversion
        # to wavelength, d-spacing, etc. because conversions of monitors do
        # not use the sample position.
        single_monitor['data'].coords.set_aligned('sample_position', False)
        # Remove redundant information that is duplicated from workspace
        # We get this extra information from the generic converter reuse
        single_monitor['data'].coords.pop('detector_info', None)
        del single_monitor['sample']
        name = comp_info.name(det_index)
        if not comp_info.uniqueName(name):
            name = f'{name}_{number}'
        monitors.append((name, single_monitor))
    return monitors


def convert_monitors_ws_arrays(ws, converter, **ignored):
    spec_dim, spec_coord = init_spec_axis(ws)
    spec_info = ws.spectrumInfo()
    comp_info = ws.componentInfo()
    monitors = []
    spec_indices = (
        (ws.getIndexFromSpectrumNumber(int(i)), i) for i in spec_coord.values
    )
    for index, number in spec_indices:
        definition = spec_info.getSpectrumDefinition(index)
        if not definition.size() == 1:
            raise RuntimeError("Cannot deal with grouped monitor detectors")
        det_index = definition[0][0]  # Ignore time index
        # We only ExtractSpectra for compatibility with
        # existing convert_Workspace2D_to_dataarray. This could instead be
        # refactored if found to be slow
        with run_mantid_alg(
            'ExtractSpectra', InputWorkspace=ws, WorkspaceIndexList=[index]
        ) as monitor_ws:
            # Run logs are already loaded in the data workspace
            single_monitor = converter(monitor_ws, load_run_logs=False)
        # Storing sample_position as a coord of monitors means that monitor
        # data cannot be combined with scattered data even after conversion
        # to wavelength, d-spacing, etc. because conversions of monitors do
        # not use the sample position.
        # But keep it as an attr in case a user needs it.
        get_attrs(single_monitor)['sample_position'] = single_monitor.coords.pop(
            'sample_position'
        )
        # Remove redundant information that is duplicated from workspace
        # We get this extra information from the generic converter reuse
        if 'detector_info' in single_monitor.coords:
            del single_monitor.coords['detector_info']
        del get_attrs(single_monitor)['sample']
        name = comp_info.name(det_index)
        if not comp_info.uniqueName(name):
            name = f'{name}_{number}'
        monitors.append((name, single_monitor))
    return monitors


def convert_Workspace2D_to_data_array(
    ws, load_run_logs=True, advanced_geometry=False, **ignored
):
    warnings.warn(
        'convert_Workspace2D_to_data_array is deprecated in favor of '
        'convert_Workspace2D_to_data_group.',
        VisibleDeprecationWarning,
        stacklevel=4,
    )

    dim, unit = validate_and_get_unit(ws.getAxis(0).getUnit())
    spec_dim, spec_coord = init_spec_axis(ws)

    coords_labs_data = _convert_MatrixWorkspace_info(
        ws, advanced_geometry=advanced_geometry, load_run_logs=load_run_logs
    )
    coords_labs_data["coords"] = _as_dict_of_variables(coords_labs_data["coords"])
    coords_labs_data["attrs"] = _as_dict_of_variables(coords_labs_data["attrs"])
    _, data_unit = validate_and_get_unit(ws.YUnit(), allow_empty=True)
    if ws.id() == 'MaskWorkspace':
        coords_labs_data["data"] = sc.Variable(
            dims=[spec_dim],
            unit=None,
            values=ws.extractY().flatten(),
            dtype=sc.DType.bool,
        )
    else:
        stddev2 = ws.extractE()
        np.multiply(stddev2, stddev2, out=stddev2)  # much faster than np.power
        coords_labs_data["data"] = sc.Variable(
            dims=[spec_dim, dim],
            unit=data_unit,
            values=ws.extractY(),
            variances=stddev2,
        )
    array = sc.DataArray(**coords_labs_data)

    if ws.hasAnyMaskedBins():
        bin_mask = sc.zeros(dims=array.dims, shape=array.shape, dtype=sc.DType.bool)
        for i in range(ws.getNumberHistograms()):
            # maskedBinsIndices throws instead of returning empty list
            if ws.hasMaskedBins(i):
                set_bin_masks(bin_mask, dim, i, ws.maskedBinsIndices(i))
        common_mask = sc.all(bin_mask, spec_dim)
        if sc.identical(common_mask, sc.any(bin_mask, spec_dim)):
            array.masks["bin"] = common_mask
        else:
            array.masks["bin"] = bin_mask

    # Avoid creating dimensions that are not required since this mostly an
    # artifact of inflexible data structures and gets in the way when working
    # with scipp.
    if len(spec_coord.values) == 1:
        if 'position' in array.coords:
            array.coords['position'] = array.coords['position'][spec_dim, 0]
        array = array[spec_dim, 0].copy()
    return array


def convert_Workspace2D_to_data_group(
    ws, load_run_logs=True, advanced_geometry=False, **ignored
) -> sc.DataGroup:
    dim, unit = validate_and_get_unit(ws.getAxis(0).getUnit())
    spec_dim, spec_coord = init_spec_axis(ws)

    coords_labs_data = _convert_MatrixWorkspace_info(
        ws, advanced_geometry=advanced_geometry, load_run_logs=load_run_logs
    )
    _, data_unit = validate_and_get_unit(ws.YUnit(), allow_empty=True)
    if ws.id() == 'MaskWorkspace':
        data = sc.array(
            dims=[spec_dim],
            unit=None,
            values=ws.extractY().flatten(),
            dtype=sc.DType.bool,
        )
    else:
        stddev2 = ws.extractE()
        np.multiply(stddev2, stddev2, out=stddev2)  # much faster than np.power
        data = sc.array(
            dims=[spec_dim, dim],
            unit=data_unit,
            values=ws.extractY(),
            variances=stddev2,
        )
    res = sc.DataGroup(
        {
            "data": sc.DataArray(
                data,
                coords=_as_dict_of_variables(coords_labs_data["coords"]),
                masks=coords_labs_data["masks"],
            ),
            **coords_labs_data["attrs"],
        }
    )

    if ws.hasAnyMaskedBins():
        bin_mask = sc.zeros(dims=data.dims, shape=data.shape, dtype=sc.DType.bool)
        for i in range(ws.getNumberHistograms()):
            # maskedBinsIndices throws instead of returning empty list
            if ws.hasMaskedBins(i):
                set_bin_masks(bin_mask, dim, i, ws.maskedBinsIndices(i))
        common_mask = sc.all(bin_mask, spec_dim)
        if sc.identical(common_mask, sc.any(bin_mask, spec_dim)):
            res["data"].masks["bin"] = common_mask
        else:
            res["data"].masks["bin"] = bin_mask

    # Avoid creating dimensions that are not required since this mostly an
    # artifact of inflexible data structures and gets in the way when working
    # with scipp.
    if len(spec_coord.values) == 1:
        if 'position' in res["data"].coords:
            res["data"].coords['position'] = res["data"].coords['position'][spec_dim, 0]
        res["data"] = res["data"][spec_dim, 0].copy()
    return res


def _contains_weighted_events(spectrum) -> bool:
    from mantid.api import EventType

    return spectrum.getEventType() in (EventType.WEIGHTED, EventType.WEIGHTED_NOTIME)


def convert_EventWorkspace_to_data_array(
    ws, load_pulse_times=True, advanced_geometry=False, load_run_logs=True, **ignored
):
    warnings.warn(
        'convert_EventWorkspace_to_data_array is deprecated in favor of '
        'convert_EventWorkspace_to_data_group.',
        VisibleDeprecationWarning,
        stacklevel=4,
    )

    dim, unit = validate_and_get_unit(ws.getAxis(0).getUnit())
    spec_dim, spec_coord = init_spec_axis(ws)
    nHist = ws.getNumberHistograms()
    _, data_unit = validate_and_get_unit(ws.YUnit(), allow_empty=True)

    n_event = ws.getNumberEvents()
    coord = sc.empty(dims=['event'], shape=[n_event], unit=unit, dtype=sc.DType.float64)
    weights = sc.ones(
        dims=['event'],
        shape=[n_event],
        unit=data_unit,
        dtype=sc.DType.float32,
        with_variances=True,
    )
    pulse_times = (
        sc.empty(
            dims=['event'], shape=[n_event], dtype=sc.DType.datetime64, unit=sc.units.ns
        )
        if load_pulse_times
        else None
    )

    begins = sc.zeros(
        dims=[spec_dim, dim], shape=[nHist, 1], dtype=sc.DType.int64, unit=None
    )
    ends = begins.copy()
    if n_event > 0:  # Skip expensive loop if there are no events
        current = 0
        for i in range(nHist):
            sp = ws.getSpectrum(i)
            size = sp.getNumberEvents()
            begins.values[i] = current
            ends.values[i] = current + size
            if size == 0:  # Skip expensive getters
                continue
            coord['event', current : current + size].values = sp.getTofs()
            if load_pulse_times:
                pulse_times[
                    'event', current : current + size
                ].values = sp.getPulseTimesAsNumpy()
            if _contains_weighted_events(sp):
                weights['event', current : current + size].values = sp.getWeights()
                weights[
                    'event', current : current + size
                ].variances = sp.getWeightErrors()
            current += size

    proto_events = {'data': weights, 'coords': {dim: coord}}
    if load_pulse_times:
        proto_events["coords"]["pulse_time"] = pulse_times
    events = sc.DataArray(**proto_events)

    coords_labs_data = _convert_MatrixWorkspace_info(
        ws, advanced_geometry=advanced_geometry, load_run_logs=load_run_logs
    )
    coords_labs_data["coords"] = _as_dict_of_variables(coords_labs_data["coords"])
    coords_labs_data["attrs"] = _as_dict_of_variables(coords_labs_data["attrs"])
    # For now we ignore potential finer bin edges to avoid creating too many
    # bins. Use just a single bin along dim and use extents given by workspace
    # edges.
    # TODO If there are events outside edges this might create bins with
    # events that are not within bin bounds. Consider using `bin` instead
    # of `bins`?
    edges = coords_labs_data['coords'][dim]
    # Using range slice of thickness 1 to avoid transposing 2-D coords
    coords_labs_data['coords'][dim] = sc.concat([edges[dim, :1], edges[dim, -1:]], dim)
    coords_labs_data["data"] = sc.bins(begin=begins, end=ends, dim='event', data=events)
    return sc.DataArray(**coords_labs_data)


def convert_EventWorkspace_to_data_group(
    ws, load_pulse_times=True, advanced_geometry=False, load_run_logs=True, **ignored
):
    dim, unit = validate_and_get_unit(ws.getAxis(0).getUnit())
    spec_dim, spec_coord = init_spec_axis(ws)
    nHist = ws.getNumberHistograms()
    _, data_unit = validate_and_get_unit(ws.YUnit(), allow_empty=True)

    n_event = ws.getNumberEvents()
    coord = sc.empty(dims=['event'], shape=[n_event], unit=unit, dtype=sc.DType.float64)
    weights = sc.ones(
        dims=['event'],
        shape=[n_event],
        unit=data_unit,
        dtype=sc.DType.float32,
        with_variances=True,
    )
    pulse_times = (
        sc.empty(
            dims=['event'], shape=[n_event], dtype=sc.DType.datetime64, unit=sc.units.ns
        )
        if load_pulse_times
        else None
    )

    begins = sc.zeros(
        dims=[spec_dim, dim], shape=[nHist, 1], dtype=sc.DType.int64, unit=None
    )
    ends = begins.copy()
    if n_event > 0:  # Skip expensive loop if there are no events
        current = 0
        for i in range(nHist):
            sp = ws.getSpectrum(i)
            size = sp.getNumberEvents()
            begins.values[i] = current
            ends.values[i] = current + size
            if size == 0:  # Skip expensive getters
                continue
            coord['event', current : current + size].values = sp.getTofs()
            if load_pulse_times:
                pulse_times[
                    'event', current : current + size
                ].values = sp.getPulseTimesAsNumpy()
            if _contains_weighted_events(sp):
                weights['event', current : current + size].values = sp.getWeights()
                weights[
                    'event', current : current + size
                ].variances = sp.getWeightErrors()
            current += size

    proto_events = {'data': weights, 'coords': {dim: coord}}
    if load_pulse_times:
        proto_events["coords"]["pulse_time"] = pulse_times
    events = sc.DataArray(**proto_events)

    coords_labs_data = _convert_MatrixWorkspace_info(
        ws, advanced_geometry=advanced_geometry, load_run_logs=load_run_logs
    )
    # For now we ignore potential finer bin edges to avoid creating too many
    # bins. Use just a single bin along dim and use extents given by workspace
    # edges.
    # TODO If there are events outside edges this might create bins with
    # events that are not within bin bounds. Consider using `bin` instead
    # of `bins`?
    edges = coords_labs_data['coords'][dim]
    # Using range slice of thickness 1 to avoid transposing 2-D coords
    coords_labs_data['coords'][dim] = sc.concat([edges[dim, :1], edges[dim, -1:]], dim)

    data = sc.bins(begin=begins, end=ends, dim='event', data=events)
    return sc.DataGroup(
        {
            "data": sc.DataArray(
                data, coords=_as_dict_of_variables(coords_labs_data["coords"])
            ),
            **coords_labs_data["attrs"],
        }
    )


def convert_MDHistoWorkspace_to_data_group(md_histo, **ignored) -> sc.DataGroup:
    ndims = md_histo.getNumDims()
    coords = {}
    dims_used = []
    for i in range(ndims):
        dim = md_histo.getDimension(i)
        frame = dim.getMDFrame()
        sc_dim = md_dimension(dim, i)
        coords[sc_dim] = sc.array(
            dims=[sc_dim],
            values=np.linspace(dim.getMinimum(), dim.getMaximum(), dim.getNBins()),
            unit=md_unit(frame),
        )
        dims_used.append(sc_dim)
    data = sc.array(
        dims=dims_used,
        values=md_histo.getSignalArray(),
        variances=md_histo.getErrorSquaredArray(),
        unit=sc.units.counts,
    )
    nevents = sc.array(dims=dims_used, values=md_histo.getNumEventsArray())
    return sc.DataGroup(
        {'data': sc.DataArray(coords=coords, data=data), 'nevents': nevents}
    )


def convert_MDHistoWorkspace_to_data_array(md_histo, **ignored):
    warnings.warn(
        'convert_MDHistoWorkspace_to_data_array is deprecated in favor of '
        'convert_MDHistoWorkspace_to_data_group.',
        VisibleDeprecationWarning,
        stacklevel=4,
    )

    ndims = md_histo.getNumDims()
    coords = {}
    dims_used = []
    for i in range(ndims):
        dim = md_histo.getDimension(i)
        frame = dim.getMDFrame()
        sc_dim = md_dimension(dim, i)
        coords[sc_dim] = sc.Variable(
            dims=[sc_dim],
            values=np.linspace(dim.getMinimum(), dim.getMaximum(), dim.getNBins()),
            unit=md_unit(frame),
        )
        dims_used.append(sc_dim)
    data = sc.Variable(
        dims=dims_used,
        values=md_histo.getSignalArray(),
        variances=md_histo.getErrorSquaredArray(),
        unit=sc.units.counts,
    )
    nevents = sc.Variable(dims=dims_used, values=md_histo.getNumEventsArray())
    return sc.DataArray(coords=coords, data=data, attrs={'nevents': nevents})


def convert_TableWorkspace_to_dataset(ws, error_connection=None, **ignored):
    """
    Converts from a Mantid TableWorkspace to a scipp dataset.

    It is possible to assign a column as the error for another column,
    in which case data from the two columns will be represented by a single scipp
    variable with variance. This is done using the error_connection Keyword
    argument. The error is transformed to variance in this converter.

    Parameters
    ----------
    ws:
        Mantid TableWorkspace to be converted into scipp dataset.
    error_connection:
        Dict with data column names as keys to names of their error column.
    """

    # Extract information from workspace
    n_columns = ws.columnCount()
    columnNames = ws.getColumnNames()  # list of names matching each column
    columnTypes = ws.columnTypes()  # list of types matching each column

    # Types available in TableWorkspace that can not be loaded into scipp
    blacklist_types = []
    # Types for which the transformation from error to variance will fail
    blacklist_variance_types = ["str"]

    result = {}
    for i in range(n_columns):
        if columnTypes[i] in blacklist_types:
            continue  # skips loading data of this type

        data_name = columnNames[i]
        if error_connection is None:
            result[data_name] = sc.Variable(dims=['row'], values=ws.column(i))
        elif data_name in error_connection:
            # This data has error available
            error_name = error_connection[data_name]
            error_index = columnNames.index(error_name)

            if columnTypes[error_index] in blacklist_variance_types:
                # Raise error to avoid numpy square error for strings
                raise RuntimeError(
                    "Variance can not have type string. \n"
                    + "Data:     "
                    + str(data_name)
                    + "\n"
                    + "Variance: "
                    + str(error_name)
                    + "\n"
                )

            variance = np.array(ws.column(error_name)) ** 2
            result[data_name] = sc.Variable(
                dims=['row'], values=np.array(ws.column(i)), variances=variance
            )
        elif data_name not in error_connection.values():
            # This data is not an error for another dataset, and has no error
            result[data_name] = sc.Variable(dims=['row'], values=ws.column(i))

    return sc.Dataset(result) if result else sc.Dataset({})


def convert_WorkspaceGroup_to_data_group(group_workspace, **kwargs):
    workspace_dict = sc.DataGroup()
    for i in range(group_workspace.getNumberOfEntries()):
        workspace = group_workspace.getItem(i)
        workspace_name = (
            workspace.name().replace(f'{group_workspace.name()}', '').strip('_')
        )
        workspace_dict[workspace_name] = from_mantid(workspace, **kwargs)

    return workspace_dict


def convert_WorkspaceGroup_to_dataarray_dict(group_workspace, **kwargs):
    warnings.warn(
        'convert_WorkspaceGroup_to_dataarray_dict is deprecated in favor of '
        'convert_WorkspaceGroup_to_data_group.',
        VisibleDeprecationWarning,
        stacklevel=4,
    )

    workspace_dict = {}
    for i in range(group_workspace.getNumberOfEntries()):
        workspace = group_workspace.getItem(i)
        workspace_name = (
            workspace.name().replace(f'{group_workspace.name()}', '').strip('_')
        )
        workspace_dict[workspace_name] = from_mantid(workspace, **kwargs)

    return workspace_dict


def from_mantid(workspace, **kwargs) -> sc.DataGroup:
    """Convert Mantid workspace to a scipp data group.

    Parameters
    ----------
    workspace:
        Mantid workspace to convert.
    kwargs:
        Forwarded to the workspace-specific converter.

    Returns
    -------
        A data group representing ``workspace``.
    """
    monitor_ws = None
    workspaces_to_delete = []
    w_id = workspace.id()
    if w_id == 'Workspace2D' or w_id == 'RebinnedOutput' or w_id == 'MaskWorkspace':
        n_monitor = 0
        spec_info = workspace.spectrumInfo()
        for i in range(len(spec_info)):
            if spec_info.hasDetectors(i) and spec_info.isMonitor(i):
                n_monitor += 1
        # If there are *only* monitors we do not move them to an attribute
        if n_monitor > 0 and n_monitor < len(spec_info):
            import mantid.simpleapi as mantid

            workspace, monitor_ws = mantid.ExtractMonitors(workspace)
            workspaces_to_delete.append(workspace)
            workspaces_to_delete.append(monitor_ws)
        scipp_obj = convert_Workspace2D_to_data_group(workspace, **kwargs)
    elif w_id == 'EventWorkspace':
        scipp_obj = convert_EventWorkspace_to_data_group(workspace, **kwargs)
    elif w_id == 'TableWorkspace':
        scipp_obj = convert_TableWorkspace_to_dataset(workspace, **kwargs)
    elif w_id == 'MDHistoWorkspace':
        scipp_obj = convert_MDHistoWorkspace_to_data_group(workspace, **kwargs)
    elif w_id == 'WorkspaceGroup':
        scipp_obj = convert_WorkspaceGroup_to_data_group(workspace, **kwargs)
    else:
        raise RuntimeError(f'Unsupported workspace type {w_id}')

    # TODO Is there ever a case where a Workspace2D has a separate monitor
    # workspace? This is not handled by ExtractMonitors above, I think.
    if monitor_ws is None:
        if hasattr(workspace, 'getMonitorWorkspace'):
            try:
                monitor_ws = workspace.getMonitorWorkspace()
            except RuntimeError:
                # Have to try/fail here. No inspect method on Mantid for this.
                pass

    if monitor_ws is not None:
        if monitor_ws.id() == 'MaskWorkspace' or monitor_ws.id() == 'Workspace2D':
            converter = convert_Workspace2D_to_data_group
        elif monitor_ws.id() == 'EventWorkspace':
            converter = convert_EventWorkspace_to_data_group
        scipp_obj["monitors"] = sc.DataGroup(
            convert_monitors_ws(monitor_ws, converter, **kwargs)
        )
    for ws in workspaces_to_delete:
        mantid.DeleteWorkspace(ws)

    return scipp_obj


def array_from_mantid(workspace, **kwargs) -> sc.DataArray | sc.Dataset:
    """Convert Mantid workspace to a scipp data array or dataset.
    :param workspace: Mantid workspace to convert.

    .. deprecated:: 23.07.0
       The old ``from_mantid`` was renamed to ``array_from_mantid`` and is deprecated
       in favor of the new ``from_mantid`` which return a data group.
    """
    warnings.warn(
        "scippneutron.array_from_mantid has been deprecated and is scheduled for "
        "removal in scippneutron v24.01.0. "
        "Use the new scippneutron.from_mantid instead.",
        VisibleDeprecationWarning,
        stacklevel=4,
    )

    scipp_obj = None  # This is either a Dataset or DataArray
    monitor_ws = None
    workspaces_to_delete = []
    w_id = workspace.id()
    if w_id == 'Workspace2D' or w_id == 'RebinnedOutput' or w_id == 'MaskWorkspace':
        n_monitor = 0
        spec_info = workspace.spectrumInfo()
        for i in range(len(spec_info)):
            if spec_info.hasDetectors(i) and spec_info.isMonitor(i):
                n_monitor += 1
        # If there are *only* monitors we do not move them to an attribute
        if n_monitor > 0 and n_monitor < len(spec_info):
            import mantid.simpleapi as mantid

            workspace, monitor_ws = mantid.ExtractMonitors(workspace)
            workspaces_to_delete.append(workspace)
            workspaces_to_delete.append(monitor_ws)
        scipp_obj = convert_Workspace2D_to_data_array(workspace, **kwargs)
    elif w_id == 'EventWorkspace':
        scipp_obj = convert_EventWorkspace_to_data_array(workspace, **kwargs)
    elif w_id == 'TableWorkspace':
        scipp_obj = convert_TableWorkspace_to_dataset(workspace, **kwargs)
    elif w_id == 'MDHistoWorkspace':
        scipp_obj = convert_MDHistoWorkspace_to_data_array(workspace, **kwargs)
    elif w_id == 'WorkspaceGroup':
        scipp_obj = convert_WorkspaceGroup_to_dataarray_dict(workspace, **kwargs)

    if scipp_obj is None:
        raise RuntimeError(f'Unsupported workspace type {w_id}')

    # TODO Is there ever a case where a Workspace2D has a separate monitor
    # workspace? This is not handled by ExtractMonitors above, I think.
    if monitor_ws is None:
        if hasattr(workspace, 'getMonitorWorkspace'):
            try:
                monitor_ws = workspace.getMonitorWorkspace()
            except RuntimeError:
                # Have to try/fail here. No inspect method on Mantid for this.
                pass

    if monitor_ws is not None:
        if monitor_ws.id() == 'MaskWorkspace' or monitor_ws.id() == 'Workspace2D':
            converter = convert_Workspace2D_to_data_array
        elif monitor_ws.id() == 'EventWorkspace':
            converter = convert_EventWorkspace_to_data_array

        monitors = convert_monitors_ws_arrays(monitor_ws, converter, **kwargs)
        for name, monitor in monitors:
            get_attrs(scipp_obj)[name] = sc.scalar(monitor)
    for ws in workspaces_to_delete:
        mantid.DeleteWorkspace(ws)

    return scipp_obj


def load_with_mantid(
    filename: str | Path = "",
    load_pulse_times=True,
    instrument_filename=None,
    error_connection=None,
    mantid_alg='Load',
    mantid_args=None,
    advanced_geometry=False,
) -> sc.DataGroup:
    """Load a file using Mantid.

    Wraps Mantid's loaders and converts the result to a scipp data group.

    See also the neutron-data tutorial.

    Note that this function requires mantid to be installed and available in
    the same Python environment as scipp.

    Parameters
    ----------
    filename:
        The name of the Nexus/HDF file to be loaded.
    load_pulse_times:
        Read the pulse times if True.
    instrument_filename:
        If specified, over-write the instrument definition
        in the final dataset with the geometry contained in the file.
    error_connection:
        Dict with data column names as keys to names of their error column.
        Only used when the loaded workspace is a TableWorkspace.
        See scippneutron.mantid.convert_TableWorkspace_to_dataset
    mantid_alg:
        Mantid algorithm to use for loading. Default is `Load`.
    mantid_args:
        Dict of keyword arguments to forward to Mantid.
    advanced_geometry:
        If True, load the full detector geometry including shapes and rotations.
        The positions of grouped detectors are spherically averaged.
        If False, load only the detector position, and return
        the cartesian average of the grouped detector positions.

    Returns
    -------
    :
        A Data group containing the neutron event/histogram data and the
        instrument geometry.

    Raises
    ------
    RuntimeError
        If the Mantid workspace type returned by the Mantid loader is not
        either EventWorkspace or Workspace2D.

    Examples
    --------
    >>> from scippneutron import load_with_mantid
    >>> load_with_mantid(filename='PG3_4844_event.nxs',
    ...                  load_pulse_times=False,
    ...                  mantid_args={
    ...                      'BankName': 'bank184',
    ...                      'LoadMonitors': True})  # doctest: +SKIP
    """

    if mantid_args is None:
        mantid_args = {}

    _check_file_path(str(filename), mantid_alg)

    with run_mantid_alg(mantid_alg, str(filename), **mantid_args) as loaded:
        # Determine what Load has provided us
        from mantid.api import Workspace

        if isinstance(loaded, Workspace):
            # A single workspace
            data_ws = loaded
        else:
            # Separate data and monitor workspaces
            data_ws = loaded.OutputWorkspace

        if instrument_filename is not None:
            import mantid.simpleapi as mantid

            mantid.LoadInstrument(
                data_ws, FileName=instrument_filename, RewriteSpectraMap=True
            )
        return from_mantid(
            data_ws,
            load_pulse_times=load_pulse_times,
            error_connection=error_connection,
            advanced_geometry=advanced_geometry,
        )


def load(
    filename: str | Path = "",
    load_pulse_times=True,
    instrument_filename=None,
    error_connection=None,
    mantid_alg='Load',
    mantid_args=None,
    advanced_geometry=False,
) -> sc.Dataset:
    """Load a file using Mantid.

    Wraps Mantid's loaders and converts the result to a scipp dataset.

    See also the neutron-data tutorial.

    Note that this function requires mantid to be installed and available in
    the same Python environment as scipp.

    Parameters
    ----------
    filename:
        The name of the Nexus/HDF file to be loaded.
    load_pulse_times:
        Read the pulse times if True.
    instrument_filename:
        If specified, over-write the instrument definition
        in the final dataset with the geometry contained in the file.
    error_connection:
        Dict with data column names as keys to names of their error column.
        Only used when the loaded workspace is a TableWorkspace.
        See scippneutron.mantid.convert_TableWorkspace_to_dataset
    mantid_alg:
        Mantid algorithm to use for loading. Default is `Load`.
    mantid_args:
        Dict of keyword arguments to forward to Mantid.
    advanced_geometry:
        If True, load the full detector geometry including shapes and rotations.
        The positions of grouped detectors are spherically averaged.
        If False, load only the detector position, and return
        the cartesian average of the grouped detector positions.

    Returns
    -------
    :
        A Dataset containing the neutron event/histogram data and the
        instrument geometry.

    Raises
    ------
    RuntimeError
        If the Mantid workspace type returned by the Mantid loader is not
        either EventWorkspace or Workspace2D.

    Examples
    --------
    >>> from scippneutron import load_with_mantid
    >>> d = sc.Dataset()
    >>> d["sample"] = load(filename='PG3_4844_event.nxs',
    ...                    load_pulse_times=False,
    ...                    mantid_args={'BankName': 'bank184',
    ...                                 'LoadMonitors': True})  # doctest: +SKIP

    .. deprecated:: 23.07.0
       Use load_with_mantid instead.
    """
    warnings.warn(
        "scippneutron.load has been deprecated and is scheduled for removal in "
        "scippneutron v24.01.0. Use the new scippneutron.load_with_mantid instead.",
        VisibleDeprecationWarning,
        stacklevel=2,
    )

    if mantid_args is None:
        mantid_args = {}

    _check_file_path(str(filename), mantid_alg)

    with run_mantid_alg(mantid_alg, str(filename), **mantid_args) as loaded:
        # Determine what Load has provided us
        from mantid.api import Workspace

        if isinstance(loaded, Workspace):
            # A single workspace
            data_ws = loaded
        else:
            # Separate data and monitor workspaces
            data_ws = loaded.OutputWorkspace

        if instrument_filename is not None:
            import mantid.simpleapi as mantid

            mantid.LoadInstrument(
                data_ws, FileName=instrument_filename, RewriteSpectraMap=True
            )
        return array_from_mantid(
            data_ws,
            load_pulse_times=load_pulse_times,
            error_connection=error_connection,
            advanced_geometry=advanced_geometry,
        )


def _is_mantid_loadable(filename):
    from mantid.api import FileFinder

    if FileFinder.getFullPath(filename):
        return True
    else:
        try:
            # findRuns throws rather than return empty so need try-catch
            FileFinder.findRuns(filename)
            return True
        except Exception:
            return False


def _check_file_path(filename, mantid_alg):
    from mantid.api import AlgorithmManager, FileProperty, FrameworkManager

    FrameworkManager.Instance()
    alg = AlgorithmManager.createUnmanaged(mantid_alg)
    filename_property = [
        prop for prop in alg.getProperties() if isinstance(prop, FileProperty)
    ]
    # Only with FileProperty can Mantid take fuzzy matches to filenames and run numbers
    # If the top level Load algorithm is requested (has no properties of its own)
    # we know that it's child algorithm uses FileProperty. If the child algorithm
    # is called directly we attempt to find FileProperty. Otherwise paths should be
    # absolute
    if filename_property or mantid_alg == 'Load':
        if not _is_mantid_loadable(filename):
            raise ValueError(
                f"Mantid cannot find {filename} and therefore will not load it."
            ) from None
    else:
        if not os.path.isfile(filename):
            raise ValueError(
                f"Cannot find file {filename} and therefore will not load it."
            )


def validate_dim_and_get_mantid_string(unit_dim):
    known_units = {
        'energy_transfer': "DeltaE",
        'tof': "TOF",
        'wavelength': "Wavelength",
        'energy': "Energy",
        'dspacing': "dSpacing",
        'Q': "MomentumTransfer",
        'Q^2': "QSquared",
    }

    user_k = str(unit_dim).casefold()
    known_units = {k.casefold(): v for k, v in known_units.items()}

    if user_k not in known_units:
        raise RuntimeError(
            "Axis unit not currently supported."
            f"Possible values are: {list(known_units.keys())}, "
            f"got '{unit_dim}'. "
        )
    else:
        return known_units[user_k]


def to_mantid(data, dim, instrument_file=None):
    """
    Convert data to a Mantid workspace.

    The Mantid layout expect the spectra to be the Outer-most dimension,
    i.e. y.shape[0]. If that is not the case you might have to transpose
    your data to fit that, otherwise it will not be aligned correctly in the
    Mantid workspace.

    :param data: Data to be converted.
    :param dim: Coord to use for Mantid's first axis (X).
    :param instrument_file: Instrument file that will be
                            loaded into the workspace
    :returns: Workspace containing converted data. The concrete workspace type
              may differ depending on the content of `data`.
    """
    if not isinstance(data, sc.DataArray):
        raise RuntimeError(
            "Currently only data arrays can be converted to a Mantid workspace"
        )
    try:
        import mantid.simpleapi as mantid
    except ImportError:
        raise ImportError(
            "Mantid Python API was not found, please install Mantid framework "
            "as detailed in the installation instructions (https://scipp."
            "github.io/getting-started/installation.html)"
        ) from None
    x = data.coords[dim].values
    y = data.values
    e = data.variances

    if len(y.shape) not in (1, 2):
        raise ValueError("Currently can only handle 2D data.")

    e = np.sqrt(e) if e is not None else np.sqrt(y)

    # Convert a single array (e.g. single spectra) into 2d format
    if len(y.shape) == 1:
        y = np.array([y])

    if len(e.shape) == 1:
        e = np.array([e])

    unitX = validate_dim_and_get_mantid_string(dim)

    nspec = y.shape[0]
    if len(x.shape) == 1:
        # SCIPP is using a  1:n spectra coord mapping, Mantid needs
        # a 1:1 mapping so expand this out
        x = np.broadcast_to(x, shape=(nspec, len(x)))

    nbins = x.shape[1]
    nitems = y.shape[1]

    ws = mantid.WorkspaceFactory.create(
        "Workspace2D", NVectors=nspec, XLength=nbins, YLength=nitems
    )
    if data.unit != sc.units.counts:
        ws.setDistribution(True)

    for i in range(nspec):
        ws.setX(i, x[i])
        ws.setY(i, y[i])
        ws.setE(i, e[i])

    # Set X-Axis unit
    ws.getAxis(0).setUnit(unitX)

    if instrument_file is not None:
        mantid.LoadInstrument(ws, FileName=instrument_file, RewriteSpectraMap=True)

    return ws


def _table_to_data_array(table, key, value, stddev):
    stddevs = table[stddev].values
    dim = 'parameter'
    coord = table[key].data.copy().rename_dims({'row': dim})
    return sc.DataArray(
        data=sc.Variable(
            dims=[dim], values=table[value].values, variances=stddevs * stddevs
        ),
        coords={dim: coord},
    )


def _fit_workspace(ws, mantid_args):
    """
    Performs a fit on the workspace.

    :param ws: The workspace on which the fit will be performed
    :returns: Dataset containing all of Fit's outputs
    """
    with run_mantid_alg(
        'Fit', InputWorkspace=ws, **mantid_args, CreateOutput=True
    ) as fit:
        # This is assuming that all parameters are dimensionless. If this is
        # not the case we should use a dataset with a scalar variable per
        # parameter instead. Or better, a dict of scalar variables?
        parameters = convert_TableWorkspace_to_dataset(fit.OutputParameters)
        parameters = _table_to_data_array(
            parameters, key='Name', value='Value', stddev='Error'
        )
        out = convert_Workspace2D_to_data_array(fit.OutputWorkspace).drop_coords(
            'empty'
        )
        data = {}
        data['data'] = out['empty', 0]
        data['calculated'] = out['empty', 1]
        data['diff'] = out['empty', 2]
        parameters.coords['status'] = sc.scalar(fit.OutputStatus)
        parameters.coords['chi^2/d.o.f.'] = sc.scalar(fit.OutputChi2overDoF)
        parameters.coords['function'] = sc.scalar(str(fit.Function))
        parameters.coords['cost_function'] = sc.scalar(fit.CostFunction)
        return parameters, sc.Dataset(data)


def fit(data, mantid_args):
    if len(data.dims) != 1 or 'WorkspaceIndex' in mantid_args:
        raise RuntimeError(
            "Only 1D fitting is supported. Use scipp slicing and do not"
            "provide a WorkspaceIndex."
        )
    dim = data.dims[0]
    ws = to_mantid(data, dim)
    mantid_args['workspace_index'] = 0
    return _fit_workspace(ws, mantid_args)


def _try_except(op, possible_except, failure, **kwargs):
    try:
        return op(**kwargs)
    except possible_except:
        return failure


def _get_instrument_efixed(workspace):
    inst = workspace.getInstrument()
    if inst.hasParameter('Efixed'):
        return inst.getNumberParameter('EFixed')[0]

    if inst.hasParameter('analyser'):
        analyser_name = inst.getStringParameter('analyser')[0]
        analyser_comp = inst.getComponentByName(analyser_name)

        if analyser_comp is not None and analyser_comp.hasParameter('Efixed'):
            return analyser_comp.getNumberParameter('EFixed')[0]

    return None


def _extract_einitial(ws):
    if ws.run().hasProperty("Ei"):
        ei = ws.run().getProperty("Ei").value
    elif ws.run().hasProperty('EnergyRequest'):
        ei = ws.run().getProperty('EnergyRequest').value[-1]
    else:
        ei = 0
    return sc.scalar(ei, unit=sc.Unit("meV"))


def _extract_efinal(ws, spec_dim):
    detInfo = ws.detectorInfo()
    specInfo = ws.spectrumInfo()
    ef = np.empty(shape=(specInfo.size(),), dtype=float)
    ef[:] = np.nan
    analyser_ef = _get_instrument_efixed(workspace=ws)
    ids = detInfo.detectorIDs()
    for spec_index in range(len(specInfo)):
        detector_ef = None
        if specInfo.hasDetectors(spec_index):
            # Just like mantid, we only take the first entry of the group.
            det_index = specInfo.getSpectrumDefinition(spec_index)[0][0]
            detector_ef = _try_except(
                op=ws.getEFixed,
                possible_except=RuntimeError,
                failure=None,
                detId=int(ids[det_index]),
            )
        detector_ef = detector_ef if detector_ef is not None else analyser_ef
        if not detector_ef:
            continue  # Cannot assign an Ef. May or may not be an error
            # - i.e. a diffraction detector, monitor etc.
        ef[spec_index] = detector_ef

    return sc.Variable(dims=[spec_dim], values=ef, unit=sc.Unit("meV"))
