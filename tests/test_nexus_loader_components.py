from .nexus_helpers import (
    NexusBuilder,
    EventData,
)
import numpy as np
import pytest
from typing import Callable, Tuple
from scippneutron.file_loading._detector_data import _load_event_group
from scippneutron.file_loading._nexus import LoadFromNexus
from scippneutron.file_loading._hdf5_nexus import LoadFromHdf5
from scippneutron.file_loading._json_nexus import LoadFromJson


def open_nexus(builder: NexusBuilder):
    return builder.file


def open_json(builder: NexusBuilder):
    return builder.json


@pytest.fixture(params=[(open_nexus, LoadFromHdf5()), (open_json, LoadFromJson(''))])
def nexus_group(request):
    """
    Each test with this fixture is executed with load_nexus_json
    loading JSON output from the NexusBuilder, and with load_nexus
    loading in-memory NeXus output from the NexusBuilder
    """
    return request.param


def test_load_nx_event_data_selection_yields_correct_pulses(
        nexus_group: Tuple[Callable, LoadFromNexus]):
    event_time_offsets = np.array([456, 743, 347, 345, 632, 23])
    event_data = EventData(
        event_id=np.array([1, 2, 3, 1, 3, 2]),
        event_time_offset=event_time_offsets,
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, 5]),
    )

    builder = NexusBuilder()
    builder.add_event_data(event_data)
    resource, loader = nexus_group

    with resource(builder)() as f:
        group = loader.get_child_from_group(loader.get_child_from_group(f, 'entry'),
                                            'events_0')

        class Load:
            def __getitem__(self, select=...):
                da = _load_event_group(group, loader, quiet=False, select=select)
                return da.bins.size().values

        assert np.array_equal(Load()[...], [3, 0, 2, 1])
        assert np.array_equal(Load()['pulse', 0], 3)
        assert np.array_equal(Load()['pulse', 1], 0)
        assert np.array_equal(Load()['pulse', 3], 1)
        assert np.array_equal(Load()['pulse', -1], 1)
        assert np.array_equal(Load()['pulse', -2], 2)
        assert np.array_equal(Load()['pulse', 0:0], [])
        assert np.array_equal(Load()['pulse', 1:1], [])
        assert np.array_equal(Load()['pulse', 1:-3], [])
        assert np.array_equal(Load()['pulse', 3:3], [])
        assert np.array_equal(Load()['pulse', -1:-1], [])
        assert np.array_equal(Load()['pulse', 0:1], [3])
        assert np.array_equal(Load()['pulse', 0:-3], [3])
        assert np.array_equal(Load()['pulse', -1:], [1])
        assert np.array_equal(Load()['pulse', -2:-1], [2])
        assert np.array_equal(Load()['pulse', -2:], [2, 1])
        assert np.array_equal(Load()['pulse', :-2], [3, 0])
