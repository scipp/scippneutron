from typing import List, Dict
import warnings
from ._common import Group
import scipp as sc
from ._nexus import LoadFromNexus
from ._detector_data import load_detector_data
from .nxobject import NXobject
from .nxdata import NXdata


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


def load_monitor(group: Group, nexus: LoadFromNexus, select=tuple()) -> sc.DataArray:
    # Look for event mode data structures in NXMonitor. Event-mode data takes
    # precedence over histogram-mode-data if available.
    if nexus.dataset_in_group(group, "event_time_offset")[0]:
        events = load_detector_data([group], [], nexus, True, True)
        warnings.warn(f"Event data present in NXmonitor group {group.name}. "
                      f"Histogram-mode monitor data from this group will be "
                      f"ignored.")
        return events
    try:
        return NXdata(group, nexus)[select]
    except KeyError:
        raise ValueError(f"No monitor data found in {group.name}")


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
