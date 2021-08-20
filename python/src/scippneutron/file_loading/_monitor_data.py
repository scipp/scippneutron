import h5py
from typing import List
from ._common import (MissingAttribute, Group)
import scipp as sc
from ._nexus import LoadFromNexus


def _load_data_from_histogram_mode_monitor(group: Group, file_root: h5py.File,
                                           nexus: LoadFromNexus):
    try:
        dims = nexus.get_string_attribute(group["data"], "axes").split(",")
    except MissingAttribute:
        dims = ["time_of_flight"]

    events_exist, _ = nexus.dataset_in_group(group, "raw_event_data")

    if events_exist:
        # detector_data = _create_empty_event_data([])
        # _load_event_group(group, file_root, nexus, detector_data, True)
        return sc.DataArray(data=[], coords={})
    else:
        data = nexus.load_dataset(group, "data", dimensions=dims)
        coords = {dim: nexus.load_dataset(group, dim, dimensions=[dim]) for dim in dims}
        return sc.DataArray(data=data, coords=coords)


def load_monitor_data(monitor_groups: List[Group], loaded_data, file_root: h5py.File,
                      nexus: LoadFromNexus):
    for group in monitor_groups:
        data = _load_data_from_histogram_mode_monitor(group.group, file_root, nexus)
        loaded_data.attrs[group.path.split("/")[-1]] = sc.Variable(value=data)
