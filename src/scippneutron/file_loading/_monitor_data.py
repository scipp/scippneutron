from typing import List, Dict
import warnings
from ._common import Group
import scipp as sc
from ._nexus import LoadFromNexus
from ._detector_data import load_detector_data


def index_from_select(dims, select, ignore_missing=False):
    def check_1d():
        if len(dims) != 1:
            raise ValueError(
                f"Dataset has multiple dimensions {dims}, specify the dimension to index."
            )

    if select is Ellipsis:
        return ...
    if isinstance(select, tuple) and isinstance(select[0], str):
        key, sel = select
        select = {key: sel}

    if isinstance(select, tuple):
        check_1d()
        return select
    elif isinstance(select, int) or isinstance(select, slice):
        check_1d()
        return select
    elif isinstance(select, dict):
        index = [slice(None)] * len(dims)
        for key, sel in select.items():
            if not ignore_missing and key not in dims:
                raise ValueError(
                    f"'{key}' used for indexing not found in dataset dims {dims}.")
            index[dims.index(key)] = sel
        return tuple(index)
    raise ValueError("Cannot process index {select}")


def _load_data_from_histogram_mode_monitor(group: Group,
                                           nexus: LoadFromNexus,
                                           select=...):
    data_group = nexus.get_child_from_group(group, "data")
    if data_group is not None:
        dims = nexus.get_string_attribute(data_group, "axes").split(",")
        index = index_from_select(dims, select)
        data = nexus.load_dataset(group, "data", dimensions=dims, index=index)
        coords = {
            dim: nexus.load_dataset(group,
                                    dim,
                                    dimensions=[dim],
                                    index=index_from_select([dim],
                                                            select,
                                                            ignore_missing=True))
            for dim in dims
        }
        return sc.DataArray(data=data, coords=coords)
    else:
        return None


def load_monitor(group: Group, nexus: LoadFromNexus, select=...) -> sc.DataArray:

    monitor_name = group.name.split("/")[-1]

    # Look for event mode data structures in NXMonitor. Event-mode data takes
    # precedence over histogram-mode-data if available.
    if nexus.dataset_in_group(group, "event_id")[0]:
        events = load_detector_data([group], [], nexus, True, True)
        warnings.warn(f"Event data present in NXMonitor group {group.name}. "
                      f"Histogram-mode monitor data from this group will be "
                      f"ignored.")
        return events
    else:
        data = _load_data_from_histogram_mode_monitor(group, nexus, select=select)
        if data is None:
            raise ValueError("No monitor data found in {group.name}")
        return data


def load_monitor_data(monitor_groups: List[Group], nexus: LoadFromNexus) -> Dict:
    monitor_data = {}
    for group in monitor_groups:
        try:
            monitor = load_monitor(group, nexus)
        except ValueError:
            warnings.warn(f"No event-mode or histogram-mode monitor data found for "
                          f"NXMonitor group {group.name}. Skipping this group.")
        monitor_data[monitor_name] = sc.scalar(value=monitor)

    return monitor_data
