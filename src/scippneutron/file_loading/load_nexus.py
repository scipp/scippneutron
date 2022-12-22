# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
import json
import numpy as np
from pathlib import Path
from scipp.binning import make_binned
import scipp as sc

from ._json_nexus import get_streams_info, StreamInfo, JSONGroup
from ._json_nexus import contains_stream
from ._nexus import ScippData
import h5py
from timeit import default_timer as timer
from typing import Union, Optional, Dict, Tuple, Set
from contextlib import contextmanager
from warnings import warn
from scippnexus.nxtransformations import TransformationError
from scippnexus.nxobject import NexusStructureError, NXobject
from scippnexus import NXroot


class BadSource(Exception):
    """
    Raise if something is wrong with data source which
    prevents it being used. Warn the user.
    """
    pass


class SkipSource(Exception):
    """
    Raise to abort using the data source, do not
    warn the user.
    """
    pass


def add_position_and_transforms_to_data(data: Union[sc.DataArray,
                                                    sc.Dataset], transform_name: str,
                                        position_name: str, base_position_name: str,
                                        transforms: sc.Variable,
                                        positions: sc.Variable):
    if isinstance(data, sc.DataArray):
        coords = data.coords
        attrs = data.attrs
    else:
        coords = data
        attrs = data

    if transforms is None:
        coords[position_name] = positions
        attrs[base_position_name] = positions
    elif isinstance(transforms, sc.Variable):
        # If transform is not time-dependent.
        coords[position_name] = transforms * positions
        attrs[base_position_name] = positions
        attrs[transform_name] = sc.scalar(value=transforms)
    else:
        coords[base_position_name] = positions
        coords[transform_name] = sc.scalar(value=transforms)


@contextmanager
def _open_if_path(file_in: Union[str, Path, h5py.File]):
    """
    Open if file path is provided,
    otherwise yield the existing h5py.File object
    """
    if isinstance(file_in, (str, Path)):
        with h5py.File(file_in, "r", libver='latest', swmr=True) as nexus_file:
            yield nexus_file
    else:
        yield file_in


def _load_instrument_name(instruments: Dict[str, NXobject]) -> Dict:
    instrument = next(iter(instruments.values()))
    if len(instruments) > 1:
        warn(f"More than one NXinstrument found in file, "
             f"loading name from {instrument.name} only")
    if (name := instrument.get("name")) is not None:
        return {"instrument_name": sc.scalar(name[()])}
    return {}


def _load_title(entry: NXobject) -> Dict:
    if (title := entry.get('title')) is not None:
        return {"experiment_title": sc.scalar(title[()])}
    return {}


def _load_start_and_end_time(entry: NXobject) -> Dict:
    times = {}
    for time in ["start_time", "end_time"]:
        if (dataset := entry.get(time)) is not None:
            times[time] = sc.scalar(dataset[()])
    return times


def load_nexus(data_file: Union[str, Path, h5py.File],
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
        loaded_data = _load_data(nexus_file, root, quiet)

    if not quiet:
        print("Total time:", timer() - start_time)
    return loaded_data


def _origin(unit) -> sc.Variable:
    return sc.vector(value=[0, 0, 0], unit=unit)


def _depends_on_to_position(obj) -> Union[None, sc.Variable]:
    if (transform := obj.get('depends_on')) is not None:
        if isinstance(transform, sc.DataArray) or transform.dtype == sc.DType.DataArray:
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
    if (position := _depends_on_to_position(monitor.coords)) is not None:
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


def _by_nx_class(group) -> Dict[str, Dict[str, 'NXobject']]:
    from scippnexus.nxobject import _nx_class_registry
    classes = {name: [] for name in _nx_class_registry()}

    def _match_nx_class(_, node):
        if not hasattr(node, 'shape'):
            if (nx_class := node.attrs.get('NX_class')) is not None:
                if not isinstance(nx_class, str):
                    nx_class = nx_class.decode('UTF-8')
                if nx_class in _nx_class_registry():
                    classes[nx_class].append(node)

    group._group.visititems(_match_nx_class)

    out = {}
    for nx_class, groups in classes.items():
        names = [group.name.split('/')[-1] for group in groups]
        if len(names) != len(set(names)):  # fall back to full path if duplicate
            names = [group.name for group in groups]
        out[nx_class] = {n: group._make(g) for n, g in zip(names, groups)}
    return out


def _load_data(nexus_file: Union[h5py.File, Dict], root: Optional[str],
               quiet: bool) -> Optional[ScippData]:
    """
    Main implementation for loading data is extracted to this function so that
    in-memory data can be used for unit tests.
    """
    root = NXroot(nexus_file if root is None else nexus_file[root])
    classes = _by_nx_class(root)

    if len(classes['NXentry']) > 1:
        # We can't sensibly load from multiple NXentry, for example each
        # could could contain a description of the same detector bank
        # and lead to problems with clashing detector ids etc
        raise RuntimeError(f"More than one NXentry group in file, use 'root' argument "
                           "to specify which to load data from, for example"
                           f"{__name__}('my_file.nxs', '/entry_2')")

    # In the following, we map the file structure onto a partially flattened in-memory
    # structure. This behavior is quite error prone and cumbersome and will probably
    # disappear in this form. We therefore keep this length code directly in this
    # function to provide an overview and facility future refactoring steps.
    detectors = classes.get('NXdetector', {})
    loaded_detectors = []
    for group in detectors.values():
        try:
            det = group[()]
            det = _zip_pixel_offset(det)
            det = det.flatten(to='detector_id')
            det.bins.coords['tof'] = det.bins.coords.pop('event_time_offset')
            det.bins.coords['pulse_time'] = det.bins.coords.pop('event_time_zero')
            if 'detector_number' in det.coords:
                det.coords['detector_id'] = det.coords.pop('detector_number')
            elif 'pixel_id' in det.coords:
                det.coords['detector_id'] = det.coords.pop('pixel_id')
            elif 'spectrum_index' in det.coords:
                det.coords['detector_id'] = det.coords.pop('spectrum_index')
            else:
                raise KeyError(
                    "Found neither of detector_number, pixel_id, or spectrum_index.")
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
                ValueError, IndexError) as e:
            if not contains_stream(group._group):
                warn(f"Skipped loading {group.name} due to:\n{e}")

    no_event_data = True
    loaded_data = sc.Dataset()

    # If no event data are found, make a Dataset and add the metadata as
    # Dataset entries. Otherwise, make a DataArray.
    if len(loaded_detectors):
        no_event_data = False
        loaded_data = sc.concat(loaded_detectors, 'detector_id')
    elif len(detectors) == 0:
        # If there are no NXdetector groups, load NXevent_data directly
        loaded_events = []
        for group in classes.get('NXevent_data', {}).values():
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
                    events = make_binned(events,
                                         groups=[det_id],
                                         erase=['pulse', 'bank'])
                loaded_events.append(events)
            except (BadSource, SkipSource, NexusStructureError, IndexError) as e:
                if not contains_stream(group._group):
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

    if (entries := classes['NXentry']):
        entry = next(iter(entries.values()))
        add_metadata(_load_title(entry))
        add_metadata(_load_start_and_end_time(entry))
    if (instruments := classes['NXinstrument']):
        add_metadata(_load_instrument_name(instruments))

    def load_and_add_metadata(groups, process=lambda x: x):
        items = {}
        loaded_groups = []
        for name, group in groups.items():
            try:
                items[name] = sc.scalar(process(group[()]))
                loaded_groups.append(name)
            except (BadSource, SkipSource, TransformationError, sc.DimensionError,
                    sc.UnitError, NexusStructureError, KeyError, ValueError) as e:
                if not contains_stream(group._group):
                    warn(f"Skipped loading {group.name} due to:\n{e}")
        add_metadata(items)
        return loaded_groups

    load_and_add_metadata(classes.get('NXdisk_chopper', {}))
    load_and_add_metadata(classes.get('NXlog', {}))
    load_and_add_metadata(classes.get('NXmonitor', {}), _monitor_to_canonical)
    for name, tag in {'sample': 'NXsample', 'source': 'NXsource'}.items():
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
                if not isinstance(distance, sc.Variable):
                    distance = sc.scalar(distance, unit=None)
                coords[f'{comp_name}_position'] = sc.vector(
                    value=[0, 0, distance.value], unit=distance.unit)
            elif name == 'sample':
                coords[f'{comp_name}_position'] = _origin('m')

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
    group = JSONGroup(loaded_json)
    return _load_data(group, None, True), streams


def load_nexus_json(json_filename: str) -> Optional[ScippData]:
    with open(json_filename, 'r') as json_file:
        json_string = json_file.read()
    loaded_data, _ = _load_nexus_json(json_string)
    return loaded_data
