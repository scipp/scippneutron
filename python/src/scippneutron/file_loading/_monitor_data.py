import h5py
from typing import List, Dict
from ._common import Group
import scipp as sc
from ._nexus import LoadFromNexus
from ._detector_data import load_detector_data


def _load_data_from_histogram_mode_monitor(group: Group, nexus: LoadFromNexus):
    dims = nexus.get_string_attribute(nexus.get_child_from_group(group.group, "data"),
                                      "axes").split(",")

    data = nexus.load_dataset(group.group, "data", dimensions=dims)
    coords = {
        dim: nexus.load_dataset(group.group, dim, dimensions=[dim])
        for dim in dims
    }
    return sc.DataArray(data=data, coords=coords)


def load_monitor_data(monitor_groups: List[Group], file_root: h5py.File,
                      nexus: LoadFromNexus) -> Dict:
    monitor_data = {}
    for group in monitor_groups:
        monitor_name = group.path.split("/")[-1]
        data = _load_data_from_histogram_mode_monitor(group, nexus)
        monitor_data[monitor_name] = sc.Variable(value=data)

        # Look for event mode data structures in NXMonitor
        if nexus.dataset_in_group(group.group, "event_index")[0]:

            monitor_events = load_detector_data([group], [], file_root, nexus, True)
            monitor_data[f"{monitor_name}_events"] = sc.Variable(value=monitor_events)

    return monitor_data
