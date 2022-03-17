# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
import json
import numpy as np
import scipp as sc

from ..nexus import NXroot, NX_class

from ._common import Group, MissingDataset, BadSource, SkipSource
from ._common import add_position_and_transforms_to_data
from ._hdf5_nexus import LoadFromHdf5
from ._json_nexus import LoadFromJson, get_streams_info, StreamInfo
from ._nexus import LoadFromNexus, ScippData
import h5py
from timeit import default_timer as timer
from typing import Union, List, Optional, Dict, Tuple, Set
from contextlib import contextmanager
from warnings import warn
from .nxtransformations import TransformationError
from .nxobject import NexusStructureError

nx_entry = "NXentry"
nx_instrument = "NXinstrument"


@contextmanager
def _open_if_path(file_in: Union[str, h5py.File]):
    """
    Open if file path is provided,
    otherwise yield the existing h5py.File object
    """
    if isinstance(file_in, str):
        with h5py.File(file_in, "r", libver='latest', swmr=True) as nexus_file:
            yield nexus_file
    else:
        yield file_in


def _load_instrument_name(instrument_groups: List[Group], nexus: LoadFromNexus) -> Dict:
    try:
        if len(instrument_groups) > 1:
            warn(f"More than one {nx_instrument} found in file, "
                 f"loading name from {instrument_groups[0].name} only")
        return {
            "instrument_name":
            sc.scalar(value=nexus.load_scalar_string(instrument_groups[0], "name"))
        }
    except MissingDataset:
        return {}


def _load_title(entry_group: Group, nexus: LoadFromNexus) -> Dict:
    try:
        return {
            "experiment_title":
            sc.scalar(value=nexus.load_scalar_string(entry_group, "title"))
        }
    except MissingDataset:
        return {}


def _load_start_and_end_time(entry_group: Group, nexus: LoadFromNexus) -> Dict:
    times = {}
    for time in ["start_time", "end_time"]:
        try:
            times[time] = sc.scalar(value=nexus.load_scalar_string(entry_group, time))
        except MissingDataset:
            pass
    return times


def load_nexus(data_file: Union[str, h5py.File],
               root: str = "/",
               quiet=True) -> Optional[ScippData]:
    """
    Load a NeXus file and return required information.

    :param data_file: path of NeXus file containing data to load
    :param root: path of group in file, only load data from the subtree of
      this group
    :param quiet: if False prints some details of what is being loaded

    Usage example:
      data = sc.neutron.load_nexus('PG3_4844_event.nxs')
    """
    start_time = timer()

    with _open_if_path(data_file) as nexus_file:
        loaded_data = _load_data(nexus_file, root, LoadFromHdf5(), quiet)

    if not quiet:
        print("Total time:", timer() - start_time)
    return loaded_data


def _origin(unit) -> sc.Variable:
    return sc.vector(value=[0, 0, 0], unit=unit)


def _depends_on_to_position(da) -> Union[None, sc.Variable]:
    if (transform := da.coords.get('depends_on')) is not None:
        if transform.dtype == sc.DType.DataArray:
            return None  # cannot compute position if time-dependent
        else:
            return transform * _origin(transform.unit)


def _monitor_to_canonical(monitor):
    if monitor.bins is not None:
        monitor.bins.coords['tof'] = monitor.bins.coords.pop('event_time_offset')
        monitor.bins.coords['detector_id'] = monitor.bins.coords.pop('event_id')
        monitor.bins.coords['pulse_time'] = sc.bins_like(
            monitor, fill_value=monitor.coords.pop('event_time_zero'))
        da = sc.DataArray(
            sc.broadcast(monitor.data.bins.concat('pulse'), dims=['tof'], shape=[1]))
    else:
        da = monitor.copy(deep=False)
    if (position := _depends_on_to_position(monitor)) is not None:
        da.coords['position'] = position
    return da


def _zip_pixel_offset(da: sc.DataArray) -> sc.DataArray:
    """Remove [xyz]_pixel_offset and replace them by single 'pixel_offset' coord.

    Returns unchanged data array if x_pixel_offset is not found.
    """
    if 'x_pixel_offset' not in da.coords:
        return da
    x = da.coords.pop('x_pixel_offset')
    offset = sc.zeros(dims=da.dims, shape=da.shape, unit=x.unit, dtype=sc.DType.vector3)
    offset.fields.x = x.to(dtype='float64', copy=False)
    if (y := da.coords.pop('y_pixel_offset', None)) is not None:
        offset.fields.y = y.to(dtype='float64', unit=x.unit, copy=False)
    if (z := da.coords.pop('z_pixel_offset', None)) is not None:
        offset.fields.z = z.to(dtype='float64', unit=x.unit, copy=False)
    da.coords['pixel_offset'] = offset.rename_dims(dict(zip(offset.dims, da.dims)))
    return da


def _load_data(nexus_file: Union[h5py.File, Dict], root: Optional[str],
               nexus: LoadFromNexus, quiet: bool) \
        -> Optional[ScippData]:
    """
    Main implementation for loading data is extracted to this function so that
    in-memory data can be used for unit tests.
    """
    if root is not None:
        root_node = nexus_file[root]
    else:
        root_node = nexus_file
    # Use visititems (in find_by_nx_class) to traverse the entire file tree,
    # looking for any NXClass that can be read.
    # groups is a dict with a key for each category (nx_entry, nx_instrument...)
    groups = nexus.find_by_nx_class((nx_entry, nx_instrument), root_node)

    if len(groups[nx_entry]) > 1:
        # We can't sensibly load from multiple NXentry, for example each
        # could could contain a description of the same detector bank
        # and lead to problems with clashing detector ids etc
        raise RuntimeError(
            f"More than one {nx_entry} group in file, use 'root' argument "
            "to specify which to load data from, for example"
            f"{__name__}('my_file.nxs', '/entry_2')")

    no_event_data = True
    loaded_data = sc.Dataset()

    # Note: Currently this wastefully walks the tree in the file a second time.
    root = NXroot(nexus_file, nexus)
    classes = root.by_nx_class()

    # In the following, we map the file structure onto a partially flattened in-memory
    # structure. This behavior is quite error prone and cumbersome and will probably
    # disappear in this form. We therefore keep this length code directly in this
    # function to provide an overview and facility future refactoring steps.
    detectors = classes.get(NX_class.NXdetector, {})
    loaded_detectors = []
    for name, group in detectors.items():
        try:
            det = group[()]
            det = _zip_pixel_offset(det)
            det = det.flatten(to='detector_id')
            det.bins.coords['tof'] = det.bins.coords.pop('event_time_offset')
            det.bins.coords['pulse_time'] = det.bins.coords.pop('event_time_zero')
            if 'detector_number' in det.coords:
                det.coords['detector_id'] = det.coords.pop('detector_number')
            else:
                det.coords['detector_id'] = det.coords.pop('pixel_id')
            if 'pixel_offset' in det.coords:
                add_position_and_transforms_to_data(
                    data=det,
                    transform_name="position_transformations",
                    position_name="position",
                    base_position_name="base_position",
                    positions=det.coords.pop('pixel_offset'),
                    transforms=det.coords.pop('depends_on', None))
            loaded_detectors.append(det)
        except (BadSource, SkipSource, NexusStructureError, KeyError, sc.DTypeError,
                ValueError) as e:
            if not nexus.contains_stream(group._group):
                warn(f"Skipped loading {group.name} due to:\n{e}")

    # If no event data are found, make a Dataset and add the metadata as
    # Dataset entries. Otherwise, make a DataArray.
    if len(loaded_detectors):
        no_event_data = False
        loaded_data = sc.concat(loaded_detectors, 'detector_id')
    elif len(detectors) == 0:
        # If there are no NXdetector groups, load NXevent_data directly
        loaded_events = []
        for name, group in classes.get(NX_class.NXevent_data, {}).items():
            try:
                events = group[()]
                events.coords['pulse_time'] = events.coords.pop('event_time_zero')
                events.bins.coords['tof'] = events.bins.coords.pop('event_time_offset')
                events.bins.coords['detector_id'] = events.bins.coords.pop('event_id')
                det_min = events.bins.coords['detector_id'].min().value
                det_max = events.bins.coords['detector_id'].max().value
                if len(events.bins.constituents['data']) != 0:
                    # See scipp/scipp#2490
                    det_id = sc.arange('detector_id', det_min, det_max + 1, unit=None)
                    events = sc.bin(events, groups=[det_id], erase=['pulse', 'bank'])
                loaded_events.append(events)
            except (BadSource, SkipSource, NexusStructureError) as e:
                if not nexus.contains_stream(group._group):
                    warn(f"Skipped loading {group.name} due to:\n{e}")
        if len(loaded_events):
            no_event_data = False
            loaded_data = sc.concat(loaded_events, 'detector_id')

    if not no_event_data:
        # Add single tof bin
        loaded_data = sc.DataArray(loaded_data.data.fold(dim='detector_id',
                                                         sizes={
                                                             'detector_id': -1,
                                                             'tof': 1
                                                         }),
                                   coords=dict(loaded_data.coords.items()),
                                   attrs=dict(loaded_data.attrs.items()))
        tof_min = loaded_data.bins.coords['tof'].min().to(dtype='float64')
        tof_max = loaded_data.bins.coords['tof'].max().to(dtype='float64')
        tof_max.value = np.nextafter(tof_max.value, float("inf"))
        loaded_data.coords['tof'] = sc.concat([tof_min, tof_max], 'tof')

    def add_metadata(metadata: Dict[str, sc.Variable]):
        for key, value in metadata.items():
            if isinstance(loaded_data, sc.DataArray):
                loaded_data.attrs[key] = value
            else:
                loaded_data[key] = value

    if groups[nx_entry]:
        add_metadata(_load_title(groups[nx_entry][0], nexus))
        add_metadata(_load_start_and_end_time(groups[nx_entry][0], nexus))

    def load_and_add_metadata(groups, process=lambda x: x):
        items = {}
        loaded_groups = []
        for name, group in groups.items():
            try:
                items[name] = sc.scalar(process(group[()]))
                loaded_groups.append(name)
            except (BadSource, SkipSource, TransformationError, sc.DimensionError,
                    KeyError) as e:
                if not nexus.contains_stream(group._group):
                    warn(f"Skipped loading {group.name} due to:\n{e}")
        add_metadata(items)
        return loaded_groups

    load_and_add_metadata(classes.get(NX_class.NXdisk_chopper, {}))
    load_and_add_metadata(classes.get(NX_class.NXlog, {}))
    load_and_add_metadata(classes.get(NX_class.NXmonitor, {}), _monitor_to_canonical)
    for name, tag in {'sample': NX_class.NXsample, 'source': NX_class.NXsource}.items():
        comps = classes.get(tag, {})
        comps = load_and_add_metadata(comps)
        attrs = loaded_data if isinstance(loaded_data,
                                          sc.Dataset) else loaded_data.attrs
        coords = loaded_data if isinstance(loaded_data,
                                           sc.Dataset) else loaded_data.coords
        for comp_name in comps:
            comp = attrs[comp_name].value
            if (position := _depends_on_to_position(comp)) is not None:
                coords[f'{comp_name}_position'] = position
            elif (distance := comp.get('distance')) is not None:
                coords[f'{comp_name}_position'] = sc.vector(
                    value=[0, 0, distance.value], unit=distance.unit)
            elif name == 'sample':
                coords[f'{comp_name}_position'] = _origin('m')

    if groups[nx_instrument]:
        add_metadata(_load_instrument_name(groups[nx_instrument], nexus))

    # Return None if we have an empty dataset at this point
    if no_event_data and not loaded_data.keys():
        loaded_data = None
    return loaded_data


def _load_nexus_json(
    json_template: str,
    get_start_info: bool = False,
) -> Tuple[Optional[ScippData], Optional[sc.Variable], Optional[Set[StreamInfo]]]:
    """
    Use this function for testing so that file io is not required
    """
    # We do not use cls to convert value lists to sc.Variable at this
    # point because we do not know what dimension names to use here
    loaded_json = json.loads(json_template)
    streams = None
    if get_start_info:
        streams = get_streams_info(loaded_json)
    return _load_data(loaded_json, None, LoadFromJson(loaded_json), True), streams


def load_nexus_json(json_filename: str) -> Optional[ScippData]:
    with open(json_filename, 'r') as json_file:
        json_string = json_file.read()
    loaded_data, _ = _load_nexus_json(json_string)
    return loaded_data
