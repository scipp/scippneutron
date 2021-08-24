import h5py
from typing import List, Dict
from ._common import (MissingAttribute, Group)
import scipp as sc
from ._nexus import LoadFromNexus
from ._detector_data import load_detector_data


def _load_data_from_histogram_mode_monitor(group: Dict, file_root: h5py.File,
                                           nexus: LoadFromNexus):
    try:
        dims = nexus.get_string_attribute(
            nexus.get_child_from_group(group.group, "data"), "axes").split(",")
    except MissingAttribute:
        dims = ["time_of_flight"]

    data = nexus.load_dataset(group.group, "data", dimensions=dims)
    coords = {
        dim: nexus.load_dataset(group.group, dim, dimensions=[dim])
        for dim in dims
    }
    return sc.DataArray(data=data, coords=coords)


def load_monitor_data(monitor_groups: List[Group], loaded_data: sc.Dataset,
                      file_root: h5py.File, nexus: LoadFromNexus):
    event_groups = []
    for group in monitor_groups:
        monitor_name = group.path.split("/")[-1]
        data = _load_data_from_histogram_mode_monitor(group, file_root, nexus)
        loaded_data[monitor_name] = sc.Variable(value=data)

        # Look for event mode data structures in NXMonitor
        if nexus.dataset_in_group(group.group, "event_index")[0]:
            event_groups.append(group)

    # If any monitors contain event-mode data, load it
    if event_groups:
        monitor_events = load_detector_data(event_groups, [], file_root, nexus, True)
        loaded_data["monitor_events"] = sc.Variable(value=monitor_events)
