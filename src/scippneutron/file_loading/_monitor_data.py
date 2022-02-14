# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from typing import List, Dict, Union
import warnings
from ._common import Group
import scipp as sc
from ._nexus import LoadFromNexus
from ._detector_data import load_detector_data, NXevent_data
from .nxobject import NXobject, ScippIndex
from .nxdata import NXdata


class NXmonitor(NXobject):
    @property
    def shape(self) -> List[int]:
        return self._nxbase.shape

    @property
    def dims(self) -> List[str]:
        return self._nxbase.dims

    @property
    def unit(self) -> Union[sc.Unit, None]:
        return self._nxbase.unit

    @property
    def _is_events(self) -> bool:
        return 'event_time_offset' in self

    @property
    def _nxbase(self) -> Union[NXdata, NXevent_data]:
        """Branch between event-mode and histogram-mode monitor."""
        if self._is_events:
            return NXevent_data(self._group, self._loader)
        # NXdata uses the 'signal' attribute to define the field name of the signal.
        # NXmonitor uses a "hard-coded" signal name 'data', without specifying the
        # attribute in the file, so we pass this explicitly to NXdata.
        return NXdata(self._group, self._loader, signal='data')

    def _getitem(self, select: ScippIndex) -> sc.DataArray:
        """
        Load monitor data. Event-mode data takes precedence over histogram-mode data.
        """
        nxbase = self._nxbase
        if isinstance(nxbase, NXevent_data):
            warnings.warn(f"Event data present in NXmonitor group {self.name}. "
                          f"Histogram-mode monitor data from this group will be "
                          f"ignored.")
        return nxbase[select]


def load_monitor_data(monitor_groups: List[Group], nexus: LoadFromNexus) -> Dict:
    """
    Load monitor data. Event-mode data takes precedence over histogram-mode data.
    """
    monitor_data = {}
    for group in monitor_groups:
        try:
            nxmonitor = NXmonitor(group, nexus)
            # Standard loading requires binning monitor into pulses and adding
            # detector IDs. This is currently encapsulated in load_detector_data,
            # so we cannot readily use NXmonitor and bin afterwards without duplication.
            if nxmonitor._is_events:
                monitor = load_detector_data([group], [], nexus, True, True)
                warnings.warn(f"Event data present in NXmonitor group {group.name}. "
                              f"Histogram-mode monitor data from this group will be "
                              f"ignored.")
            else:
                monitor = nxmonitor[()]
            monitor_name = group.name.split("/")[-1]
            monitor_data[monitor_name] = sc.scalar(value=monitor)
        except KeyError:
            warnings.warn(f"No event-mode or histogram-mode monitor data found for "
                          f"NXMonitor group {group.name}. Skipping this group.")

    return monitor_data
