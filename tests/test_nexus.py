from .nexus_helpers import (
    NexusBuilder,
    EventData,
    Log,
    Monitor,
)
import numpy as np
import pytest
from typing import Callable, Tuple
import scipp as sc
from scippneutron.file_loading._nexus import LoadFromNexus
from scippneutron.file_loading._hdf5_nexus import LoadFromHdf5
from scippneutron.file_loading._json_nexus import LoadFromJson
from scippneutron import nexus


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


def builder_with_events_monitor_and_log():
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
    builder.add_event_data(event_data)
    builder.add_monitor(
        Monitor("monitor",
                data=np.array([1.]),
                axes=[("time_of_flight", np.array([1.]))]))
    builder.add_log(Log("log", np.array([1.1, 2.2, 3.3]), np.array([4.4, 5.5, 6.6])))
    return builder


def test_nxobject_root(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        root = nexus.NXroot(f, loader)
        assert root.nx_class == nexus.NX_class.NXroot
        assert set(root.keys()) == {'entry', 'monitor'}


def test_nxobject_entry(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        entry = nexus.NXroot(f, loader)['entry']
        assert entry.nx_class == nexus.NX_class.NXentry
        assert set(entry.keys()) == {'events_0', 'events_1', 'log'}


def test_nxobject_monitor(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        monitor = nexus.NXroot(f, loader)['monitor']
        assert monitor.nx_class == nexus.NX_class.NXmonitor
        assert sc.identical(
            monitor[...],
            sc.DataArray(sc.array(dims=['time_of_flight'], values=[1.0]),
                         coords={
                             'time_of_flight':
                             sc.array(dims=['time_of_flight'], values=[1.0])
                         }))


def test_nxobject_log(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        log = nexus.NXroot(f, loader)['entry']['log']
        assert log.nx_class == nexus.NX_class.NXlog
        assert sc.identical(
            log[...],
            sc.DataArray(
                sc.array(dims=['time'], values=[1.1, 2.2, 3.3]),
                coords={
                    'time':
                    sc.epoch(unit='ns') +
                    sc.array(dims=['time'], unit='s', values=[4.4, 5.5, 6.6]).to(
                        unit='ns', dtype='int64')
                }))


def test_nxobject_event_data(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        event_data = nexus.NXroot(f, loader)['entry']['events_0']
        assert set(event_data.keys()) == set(
            ['event_id', 'event_index', 'event_time_offset', 'event_time_zero'])
        assert event_data.nx_class == nexus.NX_class.NXevent_data


def test_nxobject_getting_item_that_does_not_exists_raises_KeyError(
        nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        root = nexus.NXroot(f, loader)
        with pytest.raises(KeyError):
            root['abcde']


def test_nxobject_name_property_is_full_path(nexus_group: Tuple[Callable,
                                                                LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        root = nexus.NXroot(f, loader)
        assert root.name == '/'
        assert root['monitor'].name == '/monitor'
        assert root['entry'].name == '/entry'
        assert root['entry']['log'].name == '/entry/log'
        assert root['entry']['events_0'].name == '/entry/events_0'


def test_nxobject_grandchild_can_be_accessed_using_path(
        nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        root = nexus.NXroot(f, loader)
        assert root['entry/log'].name == '/entry/log'
        assert root['/entry/log'].name == '/entry/log'


def test_nxobject_by_nx_class_of_root_contains_everything(
        nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        root = nexus.NXroot(f, loader)
        classes = root.by_nx_class()
        assert list(classes[nexus.NX_class.NXentry]) == ['entry']
        assert list(classes[nexus.NX_class.NXmonitor]) == ['monitor']
        assert list(classes[nexus.NX_class.NXlog]) == ['log']
        assert set(classes[nexus.NX_class.NXevent_data]) == {'events_0', 'events_1'}


def test_nxobject_by_nx_class_contains_only_children(nexus_group: Tuple[Callable,
                                                                        LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        root = nexus.NXroot(f, loader)
        classes = root['entry'].by_nx_class()
        assert list(classes[nexus.NX_class.NXentry]) == []
        assert list(classes[nexus.NX_class.NXmonitor]) == []
        assert list(classes[nexus.NX_class.NXlog]) == ['log']
        assert set(classes[nexus.NX_class.NXevent_data]) == set(
            ['events_0', 'events_1'])


def test_nxobject_dataset_items_are_returned_as_Field(
        nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        field = nexus.NXroot(f, loader)['entry/events_0/event_time_offset']
        assert isinstance(field, nexus.Field)


def test_field_properties(nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        field = nexus.NXroot(f, loader)['entry/events_0/event_time_offset']
        assert field.dtype == np.array(1).dtype
        assert field.name == '/entry/events_0/event_time_offset'
        assert field.shape == (6, )
        assert field.unit == sc.Unit('ns')


def test_field_unit_is_none_if_no_units_attribute(nexus_group: Tuple[Callable,
                                                                     LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        field = nexus.NXroot(f, loader)['entry/log']
        assert field.unit is None


def test_field_getitem_returns_numpy_array_with_correct_size_and_values(
        nexus_group: Tuple[Callable, LoadFromNexus]):
    resource, loader = nexus_group
    with resource(builder_with_events_monitor_and_log())() as f:
        field = nexus.NXroot(f, loader)['entry/events_0/event_time_offset']
        assert np.array_equal(field[...],
                              np.array([456, 743, 347, 345, 632, 23], dtype='int64'))
        assert np.array_equal(field[1:],
                              np.array([743, 347, 345, 632, 23], dtype='int64'))
        assert np.array_equal(field[:-1],
                              np.array([456, 743, 347, 345, 632], dtype='int64'))


def test_negative_event_index_converted_to_num_event(nexus_group: Tuple[Callable,
                                                                        LoadFromNexus]):
    event_time_offsets = np.array([456, 743, 347, 345, 632, 23])
    event_data = EventData(
        event_id=np.array([1, 2, 3, 1, 3, 2]),
        event_time_offset=event_time_offsets,
        event_time_zero=np.array([
            1600766730000000000, 1600766731000000000, 1600766732000000000,
            1600766733000000000
        ]),
        event_index=np.array([0, 3, 3, -1000]),
    )

    builder = NexusBuilder()
    builder.add_event_data(event_data)
    resource, loader = nexus_group
    with resource(builder)() as f:
        root = nexus.NXroot(f, loader)
        events = root['entry/events_0'][...]
        assert events.bins.size().values[2] == 3
        assert events.bins.size().values[3] == 0
