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
        # TODO branch to NXevent_data
        return NXdata(self._group, self._loader, signal='data').shape

    @property
    def dims(self):
        return NXdata(self._group, self._loader, signal='data').dims

    @property
    def unit(self):
        return NXdata(self._group, self._loader, signal='data').unit

    @property
    def _is_events(self) -> bool:
        return self._loader.dataset_in_group(self._group, "event_time_offset")[0]

    def _getitem(self, select):
        # Look for event mode data structures in NXMonitor. Event-mode data takes
        # precedence over histogram-mode-data if available.
        if self._is_events:
            if select != tuple():
                raise NotImplementedError(
                    "Loading slice of event-mode monitor not implemented yet.")
            events = load_detector_data([self._group], [], self._loader, True, True)
            warnings.warn(f"Event data present in NXmonitor group {self.name}. "
                          f"Histogram-mode monitor data from this group will be "
                          f"ignored.")
            return events
        return NXdata(self._group, self._loader, signal='data')[select]


def load_monitor_data(monitor_groups: List[Group], nexus: LoadFromNexus) -> Dict:
    monitor_data = {}
    for group in monitor_groups:
        try:
            monitor = NXmonitor(group, nexus)[()]
            monitor_name = group.name.split("/")[-1]
            monitor_data[monitor_name] = sc.scalar(value=monitor)
        except KeyError:
            warnings.warn(f"No event-mode or histogram-mode monitor data found for "
                          f"NXMonitor group {group.name}. Skipping this group.")

    return monitor_data
