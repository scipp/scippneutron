from typing import List, Dict
import warnings
from ._common import Group, to_plain_index
import scipp as sc
from ._nexus import LoadFromNexus
from ._detector_data import load_detector_data
from .nxobject import NXobject


class NXmonitor(NXobject):
    @property
    def shape(self):
        pass

    @property
    def dims(self):
        pass

    @property
    def unit(self):
        pass

    def _getitem(self, index):
        return load_monitor(self._group, self._loader, select=index)


def _load_data_from_histogram_mode_monitor(group: Group,
                                           nexus: LoadFromNexus,
                                           select=tuple()):
    data_group = nexus.get_child_from_group(group, "data")
    if data_group is not None:
        dims = nexus.get_string_attribute(data_group, "axes").split(",")
        index = to_plain_index(dims, select)
        data = nexus.load_dataset(group, "data", dimensions=dims, index=index)
        coords = {
            dim: nexus.load_dataset(group,
                                    dim,
                                    dimensions=[dim],
                                    index=to_plain_index([dim],
                                                         select,
                                                         ignore_missing=True))
            for dim in dims
        }
        return sc.DataArray(data=data, coords=coords)
    else:
        return None


def load_monitor(group: Group, nexus: LoadFromNexus, select=tuple()) -> sc.DataArray:
    # Look for event mode data structures in NXMonitor. Event-mode data takes
    # precedence over histogram-mode-data if available.
    if nexus.dataset_in_group(group, "event_time_offset")[0]:
        events = load_detector_data([group], [], nexus, True, True)
        warnings.warn(f"Event data present in NXmonitor group {group.name}. "
                      f"Histogram-mode monitor data from this group will be "
                      f"ignored.")
        return events
    else:
        data = _load_data_from_histogram_mode_monitor(group, nexus, select=select)
        if data is None:
            raise ValueError(f"No monitor data found in {group.name}")
        return data


def load_monitor_data(monitor_groups: List[Group], nexus: LoadFromNexus) -> Dict:
    monitor_data = {}
    for group in monitor_groups:
        try:
            monitor = load_monitor(group, nexus)
            monitor_name = group.name.split("/")[-1]
            monitor_data[monitor_name] = sc.scalar(value=monitor)
        except ValueError:
            warnings.warn(f"No event-mode or histogram-mode monitor data found for "
                          f"NXMonitor group {group.name}. Skipping this group.")

    return monitor_data
