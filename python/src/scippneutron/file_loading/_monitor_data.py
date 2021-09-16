import h5py
from typing import List, Dict
import warnings
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

        # Look for event mode data structures in NXMonitor. Event-mode data takes
        # precedence over histogram-mode-data if available.
        if nexus.dataset_in_group(group.group, "event_id")[0]:
            events = load_detector_data([group], [], file_root, nexus, True)
            monitor_data[monitor_name] = sc.Variable(value=events)
            warnings.warn(f"Event data present in NXMonitor group {group.path}. "
                          f"Histogram-mode monitor data from this group will be "
                          f"ignored.")
        else:
            data = _load_data_from_histogram_mode_monitor(group, nexus)
            monitor_data[monitor_name] = sc.Variable(value=data)

    return monitor_data
