# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Matthew Jones
from dataclasses import dataclass, fields
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
from scippnexus import NXroot, NX_class, File


@dataclass
class NeutronData:
    detectors: Optional[Dict[str, sc.DataArray]] = None
    monitors: Optional[Dict[str, sc.DataArray]] = None
    logs: Optional[Dict[str, sc.DataArray]] = None
    disk_choppers: Optional[Dict[str, sc.DataArray]] = None
    sources: Optional[Dict[str, sc.DataArray]] = None
    samples: Optional[Dict[str, sc.DataArray]] = None


@dataclass
class NeutronDataLoader:
    detectors: Optional[Dict[str, NXobject]] = None
    monitors: Optional[Dict[str, NXobject]] = None
    logs: Optional[Dict[str, NXobject]] = None
    disk_choppers: Optional[Dict[str, NXobject]] = None
    sources: Optional[Dict[str, NXobject]] = None
    samples: Optional[Dict[str, NXobject]] = None

    def _load_section(self, groups, preprocess, postprocess):
        items = {}
        for name, group in groups.items():
            if (pre := preprocess(group)) is not None:
                try:
                    loaded = pre[()]
                except (NexusStructureError, TransformationError, sc.DTypeError,
                        sc.DimensionError, sc.UnitError, KeyError, ValueError,
                        IndexError) as e:
                    if not contains_stream(group._group):
                        warn(f"Skipped loading {group.name} due to:\n{e}")
                    continue
                if (post := postprocess(loaded)) is not None:
                    items[name] = post
        return items if items else None

    def load(self, preprocess=None, postprocess=None) -> NeutronData:
        """
        Callables in ``preprocess`` or ``postprocess`` may return None. This will drop
        the entry, before or after loading, respectively.
        """
        preprocess = {} if preprocess is None else preprocess
        postprocess = {} if postprocess is None else postprocess
        sections = {}
        for field in fields(self.__class__):
            section = getattr(self, field.name)
            pre = preprocess.get(field.name, lambda x: x)
            post = postprocess.get(field.name, lambda x: x)
            if section is not None:
                sections[field.name] = self._load_section(section,
                                                          preprocess=pre,
                                                          postprocess=post)
        return NeutronData(**sections)


def _make_loader(group) -> NeutronDataLoader:
    classes = group.by_nx_class()

    if len(classes[NX_class.NXentry]) > 1:
        # We can't sensibly load from multiple NXentry, for example each
        # could could contain a description of the same detector bank
        # and lead to problems with clashing detector ids etc
        raise RuntimeError(f"More than one NXentry group in file, use 'root' argument "
                           "to specify which to load data from, for example"
                           f"{__name__}('my_file.nxs', '/entry_2')")

    loader = NeutronDataLoader()
    loader.detectors = classes.get(NX_class.NXdetector)
    loader.monitors = classes.get(NX_class.NXmonitor)
    loader.logs = classes.get(NX_class.NXlog)
    loader.disk_choppers = classes.get(NX_class.NXdisk_chopper)
    loader.sources = classes.get(NX_class.NXsource)
    loader.samples = classes.get(NX_class.NXsample)
    # TODO instrument name
    return loader


@contextmanager
def open_entry(group_or_path: Union[str, Path, h5py.Group], /):
    if isinstance(group_or_path, (str, Path)):
        with File(group_or_path) as group:
            yield _make_loader(group)
    else:
        yield _make_loader(group_or_path)


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
        return {"instrument_name": name[()].squeeze()}
    return {}


def _load_title(entry: NXobject) -> Dict:
    if (title := entry.get('title')) is not None:
        return {"experiment_title": title[()].squeeze()}
    return {}


def _load_start_and_end_time(entry: NXobject) -> Dict:
    times = {}
    for time in ["start_time", "end_time"]:
        if (dataset := entry.get(time)) is not None:
            times[time] = dataset[()].squeeze()
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


def _detector_to_canonical(detector):
    det = _zip_pixel_offset(detector)
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
        raise KeyError("Found neither of detector_number, pixel_id, or spectrum_index.")
    if 'pixel_offset' in det.coords:
        add_position_and_transforms_to_data(data=det,
                                            transform_name="position_transformations",
                                            position_name="position",
                                            base_position_name="base_position",
                                            positions=det.coords.pop('pixel_offset'),
                                            transforms=det.coords.pop(
                                                'depends_on', None))
    return det


def _load_data(nexus_file: Union[h5py.File, Dict], root: Optional[str],
               quiet: bool) -> Optional[ScippData]:
    """
    Main implementation for loading data is extracted to this function so that
    in-memory data can be used for unit tests.
    """
    root = NXroot(nexus_file if root is None else nexus_file[root])
    classes = root.by_nx_class()

    with open_entry(root) as loader:
        data = loader.load(
            postprocess={
                'detectors': _detector_to_canonical,
                'monitors': lambda x: sc.scalar(_monitor_to_canonical(x)),
                'logs': lambda x: sc.scalar(x),
                'disk_choppers': lambda x: sc.scalar(x),
                'sources': lambda x: sc.scalar(x),
                'samples': lambda x: sc.scalar(x),
            })

    # In the following, we map the file structure onto a partially flattened in-memory
    # structure. This behavior is quite error prone and cumbersome and will probably
    # disappear in this form. We therefore keep this length code directly in this
    # function to provide an overview and facility future refactoring steps.
    detectors = classes.get(NX_class.NXdetector, {})

    no_event_data = True
    loaded_data = sc.Dataset()

    # If no event data are found, make a Dataset and add the metadata as
    # Dataset entries. Otherwise, make a DataArray.
    if data.detectors:
        no_event_data = False
        loaded_data = sc.concat(list(data.detectors.values()), 'detector_id')
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
                    events = make_binned(events,
                                         groups=[det_id],
                                         erase=['pulse', 'bank'])
                loaded_events.append(events)
            except (NexusStructureError, IndexError) as e:
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

    def add_metadata(metadata: Optional[Dict[str, sc.Variable]]):
        if metadata is None:
            return
        for key, value in metadata.items():
            if isinstance(loaded_data, sc.DataArray):
                loaded_data.attrs[key] = value
            else:
                loaded_data[key] = value

    if (entries := classes[NX_class.NXentry]):
        entry = next(iter(entries.values()))
        add_metadata(_load_title(entry))
        add_metadata(_load_start_and_end_time(entry))
    if (instruments := classes[NX_class.NXinstrument]):
        add_metadata(_load_instrument_name(instruments))

    add_metadata(data.disk_choppers)
    add_metadata(data.logs)
    add_metadata(data.monitors)
    add_metadata(data.sources)
    add_metadata(data.samples)

    for name, comps in [('sample', data.samples), ('source', data.sources)]:
        if comps is None:
            continue
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
